import argparse
import time
import pickle
from collections import defaultdict
from operator import itemgetter
from statistics import stdev

import mmcv
import numpy as np
import onnx
import torch
from onnx import helper, shape_inference
from onnx.utils import polish_model
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from mmdet.core import coco_eval, results2json
from mmdet.core.bbox.transforms import bbox2result
from mmdet.core.mask.transforms import mask2result
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import mmdet.trt.profiler

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

    if det_bboxes.shape[1] == 5:
        det_bboxes[:, 0:4:2] = np.clip(det_bboxes[:, 0:4:2], 0, img_w - 1)
        det_bboxes[:, 1:4:2] = np.clip(det_bboxes[:, 1:4:2], 0, img_h - 1)

    bbox_results = bbox2result(det_bboxes, det_labels, num_classes + 1)
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


class Profiler(mmdet.trt.profiler.Profiler):
    def __init__(self):
        super().__init__(self.report_layer_time)
        self.layer_times = defaultdict(int)
        self.num_executions = 0

    def report_layer_time(self, layer_name, ms):
        self.layer_times[layer_name] += ms

    def count_execution(self):
        self.num_executions += 1

    def write_results(self, file_name):
        layer_times = self.layer_times.items()
        layer_times = sorted(layer_times, key=itemgetter(1), reverse=True)
        with open(file_name, 'w') as f:
            for name, ms in layer_times:
                f.write(name + '\t' + str(ms / self.num_executions) + '\n')


class TRTModel(object):

    def __init__(self, model_file_path, cfg=None, classes=None, profile_path=None):
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        with open(model_file_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.execution_context = self.engine.create_execution_context()
        self.classes = classes
        self.pt_model = None
        if cfg is not None:
            self.pt_model = build_detector(
                cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            if classes is not None:
                self.pt_model.CLASSES = classes

        self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        self.h_output_1 = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
        self.h_output_2 = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(2)), dtype=np.int32)
        
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output_1 = cuda.mem_alloc(self.h_output_1.nbytes)
        self.d_output_2 = cuda.mem_alloc(self.h_output_2.nbytes)

        self.stream = cuda.Stream()

        self.profile_path = profile_path
        if profile_path:
            self.execution_context.profiler = Profiler()
            self.execute = lambda x: self.execution_context.execute_v2(
                    bindings=x)
        else:
            self.execute = lambda x: self.execution_context.execute_async_v2(
                    bindings=x,
                    stream_handle=self.stream.handle)

        self.input_names = []
        self.output_names = []
        self.output_shapes = []
        for binding in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(binding)
            if self.engine.binding_is_input(binding):
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_shapes.append(self.engine.get_binding_shape(binding))

    def show(self, data, result, dataset=None, score_thr=0.3):
        if self.pt_model is not None:
            self.pt_model.show_result(
                data, result, dataset=dataset, score_thr=score_thr)

    def __call__(self, inputs, *args, **kwargs):
        assert inputs[0].flatten().shape == self.h_input.shape
        cuda.memcpy_htod_async(self.d_input, inputs[0], self.stream)
        self.execute([int(self.d_input), int(self.d_output_1), int(self.d_output_2)])
        result = self.download_outputs()
        if self.profile_path:
            self.execution_context.profiler.count_execution()
        return result

    def upload_inputs(self, inputs):
        cuda.memcpy_htod_async(self.d_input, inputs[0], self.stream)
        self.stream.synchronize()

    def run(self):
        self.execution_context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output_1),
                                                          int(self.d_output_2)],
                                                stream_handle=self.stream.handle)
        self.stream.synchronize()

    def download_outputs(self):
        cuda.memcpy_dtoh_async(self.h_output_1, self.d_output_1, self.stream)
        cuda.memcpy_dtoh_async(self.h_output_2, self.d_output_2, self.stream)
        self.stream.synchronize()
        outputs = [self.h_output_1, self.h_output_2]
        outputs = list(map(lambda x: x[0].reshape(x[1]), zip(outputs, self.output_shapes)))
        outputs = dict(zip(self.output_names, outputs))
        return outputs

    def write_profile_results(self):
        if self.profile_path:
            self.execution_context.profiler.write_results(self.profile_path)


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

        model = TRTModel(args.model, cfg=cfg, classes=dataset.CLASSES, profile_path=args.profile)

        results = []
        temp_data_files = []
        prog_bar = mmcv.ProgressBar(len(dataset))

        total_model = 0.0
        min_model = 100000.0 # max value
        max_model = 0.0
        total_pp = 0.0
        counter = 0
        times = []

        if args.measure_energy_ip:
            from mmdet.utils.tp_link_power_measure import PowerMeasure
            print(args.measure_energy_ip)
            measure = PowerMeasure(args.measure_energy_ip)
            measure.start()
        if args.measure_energy_internal:
            from mmdet.utils.internal_power_measure import PowerMeasureInternal
            measure_internal = PowerMeasureInternal()
            measure_internal.start()
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                im_data = data['img'][0].cpu().numpy()
                try:
                    start = time.time()
                    #if i < 100:
                    #    result = model([im_data])
                    #else:
                    #    result = None
                    result = model([im_data])
                    time_model = time.time()
                    result = postprocess(
                        result,
                        data['img_meta'][0].data[0],
                        num_classes=classes_num,
                        rescale=not args.show)
                    end = time.time()
                    if i >= 100:
                        total_model += time_model - start
                        min_model = min(min_model, time_model - start)
                        max_model = max(max_model, time_model - start)
                        times.append(time_model - start)
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

        model.write_profile_results()

        if args.measure_energy_ip:
            print('Average power: ' + str(measure.stop()) + ' mW')
        if args.measure_energy_internal:
            print('Average internal power: ' + str(measure_internal.stop()) + 'mW')
        print('Model: ' + str(total_model) + ' s')
        print('Model min: ' + str(min_model * 1000) + ' ms')
        print('Model max: ' + str(max_model * 1000) + ' ms')
        print('Post Proc.: ' + str(total_pp) + ' s')
        print('\nFPS: ' + str(counter / total_model) + ' over ' + str(counter) + ' images')
        print('Standard deviation: ' + str(stdev([i * 1000000 for i in times])) + ' us')

        results_tmp = results
        results = []
        for file_name in temp_data_files:
            with open(file_name, 'rb') as f:
                results.extend(pickle.load(f))
        results.extend(results_tmp)
        
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
    parser.add_argument(
        '--profile',
        type=str,
        default=None,
        help='path to profiler output')
    parser.add_argument(
        '--measure_energy_ip',
        type=str,
        default=None,
        help='IP of TP-Link HS110 to measure energy')
    parser.add_argument(
        '--measure_energy_internal',
        action='store_true',
        help='measures the power with the internal power meters')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
