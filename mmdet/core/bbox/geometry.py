import torch
from .bbox import bbox_overlaps_cython
import numpy as np
from mmdet.core.bbox.transforms_rbbox import RotBox2Polys, poly2bbox, mask2poly, Tuplelist2Polylist


def bbox_overlaps_cy(boxes, query_boxes):
    box_device = boxes.device
    boxes_np = boxes.cpu().numpy().astype(np.float)
    query_boxes_np = query_boxes.cpu().numpy().astype(np.float)
    ious = bbox_overlaps_cython(boxes_np, query_boxes_np)
    return torch.from_numpy(ious).to(box_device)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


def rbbox_overlaps_cy_warp(rbboxes, query_boxes):
    # TODO: first calculate the hbb overlaps, for overlaps > 0, calculate the obb overlaps
    # import pdb
    # pdb.set_trace()
    box_device = query_boxes.device
    query_boxes_np = query_boxes.cpu().numpy().astype(np.float)

    # polys_np = RotBox2Polys(boxes_np)
    # TODO: change it to only use pos gt_masks
    # polys_np = mask2poly(gt_masks)
    # polys_np = np.array(Tuplelist2Polylist(polys_np)).astype(np.float)

    polys_np = RotBox2Polys(rbboxes).astype(np.float)
    query_polys_np = RotBox2Polys(query_boxes_np)

    h_bboxes_np = poly2bbox(polys_np)
    h_query_bboxes_np = poly2bbox(query_polys_np)

    # hious
    ious = bbox_overlaps_cython(h_bboxes_np, h_query_bboxes_np)
    import pdb
    # pdb.set_trace()
    inds = np.where(ious > 0)
    for index in range(len(inds[0])):
        box_index = inds[0][index]
        query_box_index = inds[1][index]

        box = polys_np[box_index]
        query_box = query_polys_np[query_box_index]

        # calculate obb iou
        # import pdb
        # pdb.set_trace()
        overlap = polyiou.iou_poly(polyiou.VectorDouble(box), polyiou.VectorDouble(query_box))
        ious[box_index][query_box_index] = overlap

    return torch.from_numpy(ious).to(box_device)
