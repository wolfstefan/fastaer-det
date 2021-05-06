import torch
import torch.onnx.symbolic_helper as sym_help
from torch.autograd import Function
from torch.onnx.symbolic_opset9 import reshape, unsqueeze, squeeze
from torch.onnx.symbolic_opset10 import _slice
#from torch.onnx.symbolic_opset11 import gather

from mmdet.core.utils.misc import topk
from mmdet.ops.nms import nms_wrapper


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (float): NMS operation config
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    combined_bboxes = GenericMulticlassNMS.apply(multi_bboxes, multi_scores,
                                                  score_thr, nms_cfg, int(multi_bboxes.shape[0]),
                                                  max_num, score_factors)
    bboxes = combined_bboxes[:, :5]
    labels = combined_bboxes[:, 5].view(-1)
    return bboxes, labels


class GenericMulticlassNMS(Function):

    @staticmethod
    def forward(ctx,
                multi_bboxes,
                multi_scores,
                score_thr,
                nms_cfg,
                num_priors,
                max_num=-1,
                score_factors=None):
        nms_op_cfg = nms_cfg.copy()
        nms_op_type = nms_op_cfg.pop('type', 'nms')
        nms_op = getattr(nms_wrapper, nms_op_type)

        num_classes = multi_scores.shape[1]
        bboxes, labels = [], []
        for i in range(1, num_classes):
            cls_inds = multi_scores[:, i] > score_thr
            if not cls_inds.any():
                continue
            # Get bboxes and scores of this class.
            if multi_bboxes.shape[1] == 4:
                _bboxes = multi_bboxes[cls_inds, :]
            else:
                _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
            _scores = multi_scores[cls_inds, i]
            if score_factors is not None:
                _scores *= score_factors[cls_inds]
            cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
            cls_dets, _ = nms_op(cls_dets, **nms_op_cfg)
            cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                               i - 1,
                                               dtype=torch.long)
            bboxes.append(cls_dets)
            labels.append(cls_labels)

        if bboxes:
            bboxes = torch.cat(bboxes)
            labels = torch.cat(labels)
            if bboxes.shape[0] > max_num:
                _, inds = bboxes[:, -1].topk(max_num, sorted=True)
                bboxes = bboxes[inds]
                labels = labels[inds]
            combined_bboxes = torch.cat(
                [bboxes, labels.to(bboxes.dtype).unsqueeze(-1)], dim=1)
        else:
            combined_bboxes = multi_bboxes.new_zeros((0, 6))

        return combined_bboxes

    @staticmethod
    def symbolic(g,
                 multi_bboxes,
                 multi_scores,
                 score_thr,
                 nms_cfg,
                 num_priors,
                 max_num=-1,
                 score_factors=None):
        
        nms_op_type = nms_cfg.get('type', 'nms')
        assert nms_op_type == 'nms'
        assert 'iou_thr' in nms_cfg
        iou_threshold = nms_cfg['iou_thr']
        assert 0 <= iou_threshold <= 1

        if score_factors is not None:
            score_factors = reshape(g, score_factors, [-1, 1])
            multi_scores = g.op("Mul", multi_scores, score_factors)
        
        multi_bboxes = reshape(g, multi_bboxes, [1, -1, 1, 1])
        multi_scores = reshape(g, multi_scores, [1, -1, 1, 1])
        
        result, count = g.op(
            'TRT_NonMaxSuppression', multi_bboxes, multi_scores,
            g.op('Constant', value_t=torch.LongTensor([max_num])),
            g.op('Constant', value_t=torch.FloatTensor([iou_threshold])),
            g.op('Constant', value_t=torch.FloatTensor([score_thr])),
            num_priors_i=num_priors,
            outputs=2)
            
        result = reshape(g, result, [1, -1, 7])
        count = reshape(g, count, [1])
        
        starts = g.op("Constant", value_t=torch.tensor([0]))
        axes = g.op("Constant", value_t=torch.tensor([1]))
        result = g.op('Slice', result, starts, count, axes)
        result = squeeze(g, result, 0)
            
        return result

