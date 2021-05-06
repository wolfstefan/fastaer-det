import numpy as np

from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class ISaidDatasetOriented(CocoDataset):

    CLASSES = ('storage_tank', 'Large_Vehicle', 'Small_Vehicle', 'ship', 'Harbor',
               'baseball_diamond', 'Ground_Track_Field', 'Soccer_ball_field', 'Swimming_pool',
               'Roundabout', 'tennis_court', 'basketball_court', 'plane', 'Helicopter', 'Bridge')

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            if not np.any(self.coco.annToMask(ann)):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        assert len(gt_masks_ann) > 0

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small, without ground truths or without masks."""
        valid_inds = super(ISaidDatasetOriented, self)._filter_imgs(min_size)
        to_remove = set()
        for i in valid_inds:
            image_id = self.img_ids[i]
            valid_gt = False
            for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id])):
                if np.any(self.coco.annToMask(ann)):
                    valid_gt = True
                    break
            if not valid_gt:
                to_remove.add(i)
        return [i for i in valid_inds if i not in to_remove]

