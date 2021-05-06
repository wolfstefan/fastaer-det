import  torch

from ..registry import DETECTORS
from .retinanet import RetinaNet

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder


@DETECTORS.register_module
class RetinaMask(RetinaNet):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

        self.mask_roi_extractor = builder.build_roi_extractor(mask_roi_extractor)
        self.mask_head = self.mask_head = builder.build_head(mask_head)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        losses = dict()

        x = self.extract_feat(img)
        rpn_outs = self.bbox_head(x)

        rpn_loss_inputs = rpn_outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        rpn_losses = self.bbox_head.loss(
            *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(rpn_losses)

        #proposal_cfg = self.train_cfg.get('rpn_proposal',
        #                                  self.test_cfg.rpn)
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg)
        proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
        proposal_list = self.bbox_head.get_bboxes(*proposal_inputs)

        bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
        bbox_sampler = build_sampler(
            self.train_cfg.rcnn.sampler, context=self)
        num_imgs = img.size(0)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = bbox_assigner.assign(proposal_list[i][0],
                                                 gt_bboxes[i],
                                                 gt_bboxes_ignore[i],
                                                 gt_labels[i])
            sampling_result = bbox_sampler.sample(
                assign_result,
                proposal_list[i][0],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        pos_rois = bbox2roi(
            [res.pos_bboxes for res in sampling_results])
        mask_feats = self.mask_roi_extractor(
            x[:self.mask_roi_extractor.num_inputs], pos_rois)

        if mask_feats.shape[0] > 0:
            mask_pred = self.mask_head(mask_feats)
            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    @property
    def with_mask(self):
        return False # hasattr(self, 'mask_head') and self.mask_head is not None
