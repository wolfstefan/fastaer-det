# requires mmdetection

import os
import sys
import numpy as np

import mmdet
from mmdet.apis import init_detector, show_result_pyplot, show_result, draw_poly_detections
from mmdet.core.bbox.transforms import bbox2result
from mmdet.datasets.pipelines import Compose
import mmcv
import matplotlib.pyplot as plt

from tools.test_trt import TRTModel


def postprocess(result, img_meta, num_classes=81, rescale=True):
    keys = sorted(result)
    dets = result[keys[0]][0, 0, :result[keys[1]][0, 0, 0, 0], :]
    det_bboxes = np.concatenate([dets[:, 3:], dets[:, 2:3]], axis=1).copy()
    det_labels = dets[:, 1].copy() - 1
    det_masks = result.get('masks', None)

    if rescale:
        img_h, img_w = img_meta['ori_shape'][:2]
        scale = img_meta['scale_factor']
        det_bboxes[:, :4] /= scale
    else:
        img_h, img_w = img_meta['img_shape'][:2]

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

class Model:
    def __init__(self, cfg, model):
        self.cfg = mmcv.Config.fromfile(cfg)
        self.model = TRTModel(model)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, cfg, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    # forward the model
    result = model([np.ascontiguousarray(data['img'][0].cpu().numpy())])
    result = postprocess(result, data['img_meta'][0].data)
    return result


config_file = sys.argv[1]
model = TRTModel(sys.argv[2])
cfg = mmcv.Config.fromfile(config_file)
cfg.model.pretrained = None
cfg.data.test.test_mode = True

#model = Model(config_file, model)

img_path = sys.argv[3]
img = mmcv.imread(img_path)

result = inference_detector(model, cfg, img)

if result[0].shape[1] > 5:
    img = draw_poly_detections(img, result, [''] * 16, 1.0, threshold=0.3)
    mmcv.imwrite(img, 'images/' + os.path.basename(img_path))
    plt.imshow(img)
else:
    # save image
    show_result(img, result, [''] * 16, show=False, out_file='images/' + os.path.basename(img_path), score_thr=0.2)

    show_result_pyplot(img, result, [''] * 16, score_thr=0.2)
plt.show()

