import argparse
import time
import pickle

import mmcv
import numpy as np
import onnx
import onnxruntime
import torch
from onnx import helper, shape_inference
from onnx.utils import polish_model

from mmdet.core import coco_eval, results2json
from mmdet.core.bbox.transforms import bbox2result
from mmdet.core.mask.transforms import mask2result
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def postprocess(result, img_meta, num_classes=81, rescale=True):
    keys = sorted(result)
    dets = result[keys[0]][0, 0, :result[keys[1]][0, 0, 0, 0], :]
    det_bboxes = np.concatenate([dets[:, 3:], dets[:, 2:3]], axis=1).copy()
    det_labels = dets[:, 1].copy() - 1
    det_masks = result.get('masks', None)

    if rescale:
        img_h, img_w = img_meta[0]['ori_shape'][:2]
        scale = img_meta[0]['scale_factor']
        det_bboxes[:, :4] /= scale
    else:
        img_h, img_w = img_meta[0]['img_shape'][:2]

    det_bboxes[:, 0:4:2] = np.clip(det_bboxes[:, 0:4:2], 0, img_w - 1)
    det_bboxes[:, 1:4:2] = np.clip(det_bboxes[:, 1:4:2], 0, img_h - 1)

    bbox_results = bbox2result(det_bboxes, det_labels, num_classes)
    if det_masks is not None:
        segm_results = mask2result(
            det_bboxes,
            det_labels,
            det_masks,
            num_classes,
            mask_thr_binary=0.5,
            rle=True,
            full_size=True,
            img_size=(img_h, img_w))
        return bbox_results, segm_results
    return bbox_results


def empty_result(num_classes=81, with_mask=False):
    bbox_results = [
        np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes - 1)
    ]
    if with_mask:
        segm_results = [[] for _ in range(num_classes - 1)]
        return bbox_results, segm_results
    return bbox_results


class ONNXModel(object):

    def __init__(self, model_file_path, cfg=None, classes=None, tensor_rt=True):
        self.device = onnxruntime.get_device()
        self.model = onnx.load(model_file_path)
        #self.model = polish_model(self.model)
        self.classes = classes
        self.pt_model = None
        if cfg is not None:
            self.pt_model = build_detector(
                cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            if classes is not None:
                self.pt_model.CLASSES = classes

        self.sess_options = onnxruntime.SessionOptions()
        #self.sess_options.enable_sequential_execution = True
        #self.sess_options.set_graph_optimization_level(2)
        self.sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_options.enable_profiling = False

        if tensor_rt:
            execution_provider = ['TensorrtExecutionProvider']
        else:
            execution_provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(
            self.model.SerializeToString(), self.sess_options,
            providers = execution_provider)
        #self.session.set_providers(['TensorrtExecutionProvider'])
        self.input_names = []
        self.output_names = []
        for input in self.session.get_inputs():
            self.input_names.append(input.name)
        for output in self.session.get_outputs():
            self.output_names.append(output.name)

    def show(self, data, result, dataset=None, score_thr=0.3):
        if self.pt_model is not None:
            self.pt_model.show_result(
                data, result, dataset=dataset, score_thr=score_thr)

    def add_output(self, output_ids):
        if not isinstance(output_ids, (tuple, list, set)):
            output_ids = [
                output_ids,
            ]

        inferred_model = shape_inference.infer_shapes(self.model)
        all_blobs_info = {
            value_info.name: value_info
            for value_info in inferred_model.graph.value_info
        }

        extra_outputs = []
        for output_id in output_ids:
            value_info = all_blobs_info.get(output_id, None)
            if value_info is None:
                print('WARNING! No blob with name {}'.format(output_id))
                extra_outputs.append(
                    helper.make_empty_tensor_value_info(output_id))
            else:
                extra_outputs.append(value_info)

        self.model.graph.output.extend(extra_outputs)
        self.output_names.extend(output_ids)
        self.session = onnxruntime.InferenceSession(
            self.model.SerializeToString(), self.sess_options)

    def __call__(self, inputs, *args, **kwargs):
        if isinstance(inputs, (list, tuple)):
            inputs = dict(zip(self.input_names, inputs))
        outputs = self.session.run(None, inputs, *args, **kwargs)
        outputs = dict(zip(self.output_names, outputs))
        #for k in outputs:
        #    print(str(k) + ': ' + str(outputs[k].shape))
        #print(outputs)
        return outputs


def main(args):
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, imgs_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    if args.model is None:
        print('No model file provided. Trying to load evaluation results.')
        results = mmcv.load(args.out)
    else:
        dataset = data_loader.dataset
        classes_num = len(dataset.CLASSES)

        onnxruntime.set_default_logger_severity(1)

        model = ONNXModel(args.model, cfg=cfg, classes=dataset.CLASSES, tensor_rt=True)
        run_opts = onnxruntime.RunOptions()
        #run_opts.log_severity_level = 10
        #run_opts.log_verbosity_level = 10

        results = []
        temp_data_files = []
        prog_bar = mmcv.ProgressBar(len(dataset))

        total_model = 0.0
        total_pp = 0.0
        counter = 0

        for i, data in enumerate(data_loader):
            with torch.no_grad():
                im_data = data['img'][0].cpu().numpy()
                try:
                    start = time.time()
                    result = model([im_data], run_options=run_opts)
                    time_model = time.time()
                    result = postprocess(
                        result,
                        data['img_meta'][0].data[0],
                        num_classes=classes_num,
                        rescale=not args.show)
                    end = time.time()
                    if i > 100:
                        total_model += time_model - start
                        total_pp += end - time_model
                        counter += 1
                except Exception:
                    result = empty_result(
                        num_classes=classes_num,
                        with_mask=model.pt_model.with_mask)
            results.append(result)

            if args.show:
                model.show(data, result, score_thr=args.score_thr)

            MAX_IMGS_PER_FILE = 1000
            if i % MAX_IMGS_PER_FILE == 0:
                file_name = 'temp_' + str(i // MAX_IMGS_PER_FILE) + '.pkl'
                temp_data_files.append(file_name)
                with open(file_name, 'wb') as f:
                    pickle.dump(results, f)
                results = []

            batch_size = data['img'][0].size(0)
            for _ in range(batch_size):
                prog_bar.update()

        print('Model: ' + str(total_model) + ' s')
        print('Post Proc.: ' + str(total_pp) + ' s')
        print('\nFPS: ' + str(counter / total_model) + ' over ' + str(counter) + ' images')

        results_tmp = results
        results = []
        for file_name in temp_data_files:
            with open(file_name, 'rb') as f:
                results.extend(pickle.load(f))
        results.extend(results_tmp)
        
        session_profile_path = model.session.end_profiling()
        if session_profile_path:
            print('Session profile saved to {}'.format(session_profile_path))
        print('Writing results to {}'.format(args.out))
        mmcv.dump(results, args.out)

    eval_types = args.eval
    if eval_types:
        print('Starting evaluate {}'.format(' and '.join(eval_types)))
        if eval_types == ['proposal_fast']:
            result_file = args.out
            coco_eval(result_file, eval_types, dataset.coco)
        else:
            if not isinstance(results[0], dict):
                result_files = results2json(dataset, results, args.out)
                coco_eval(result_files, eval_types, dataset.coco)
            else:
                for name in results[0]:
                    print('\nEvaluating {}'.format(name))
                    outputs_ = [out[name] for out in results]
                    result_file = args.out + '.{}'.format(name)
                    result_files = results2json(dataset, outputs_, result_file)
                    coco_eval(result_files, eval_types, dataset.coco)

    # Save predictions in the COCO json format
    if args.json_out:
        if not isinstance(results[0], dict):
            results2json(dataset, results, args.json_out)
        else:
            for name in results[0]:
                outputs_ = [out[name] for out in results]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)


def parse_args():
    parser = argparse.ArgumentParser(description='Test ONNX model')
    parser.add_argument('config', help='path to configuration file')
    parser.add_argument(
        '--model',
        type=str,
        help='path to onnx model file. If not set, try to load results'
        'from the file specified by `--out` key.')
    parser.add_argument(
        '--out', type=str, help='path to file with inference results')
    parser.add_argument(
        '--json_out',
        type=str,
        help='output result file name without extension')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument(
        '--show', action='store_true', help='visualize results')
    parser.add_argument(
        '--score_thr',
        type=float,
        default=0.3,
        help='show only detection with confidence larger than threshold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
