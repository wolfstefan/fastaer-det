import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from ..builder import build_loss
from .anchor_head import AnchorHead
from mmdet.core import AnchorGenerator, multi_apply

@HEADS.register_module
class CustomHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=[256],
                 stacked_convs=4,
                 anchor_scales=[[8, 16, 32]],
                 anchor_ratios=[[0.5, 1.0, 2.0]],
                 anchor_strides=[4],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=None):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds
        self.stacked_convs = stacked_convs

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.fp16_enabled = False

        self.anchor_generators = []
        self.num_anchors = []
        for anchor_base, scales, ratios in zip(self.anchor_base_sizes, anchor_scales, anchor_ratios):
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, scales, ratios))
            self.num_anchors.append(len(ratios) * len(scales))

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.retina_cls = nn.ModuleList()
        self.retina_reg = nn.ModuleList()
        for i in range(len(self.anchor_strides)):
            for j in range(self.stacked_convs):
                self.cls_convs.append(nn.ModuleList())
                self.reg_convs.append(nn.ModuleList())
                chn = self.in_channels if j == 0 else self.feat_channels[i]
                self.cls_convs[i].append(
                    ConvModule(
                        chn,
                        self.feat_channels[i],
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
                self.reg_convs[i].append(
                    ConvModule(
                        chn,
                        self.feat_channels[i],
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            self.retina_cls.append(nn.Conv2d(
                self.feat_channels[i],
                self.num_anchors[i] * self.cls_out_channels,
                3,
                padding=1))
            self.retina_reg.append(nn.Conv2d(
                self.feat_channels[i], self.num_anchors[i] * 4, 3, padding=1))

    def init_weights(self):
        for cls_convs, reg_convs, retina_cls, retina_reg in zip(self.cls_convs, self.reg_convs,
                                                                self.retina_cls, self.retina_reg):
            for m in cls_convs:
                normal_init(m.conv, std=0.01)
            for m in reg_convs:
                normal_init(m.conv, std=0.01)
            bias_cls = bias_init_with_prob(0.01)
            normal_init(retina_cls, std=0.01, bias=bias_cls)
            normal_init(retina_reg, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, zip(feats, range(len(feats))))

    def forward_single(self, parameters):
        x, i = parameters
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs[i]:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs[i]:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls[i](cls_feat)
        bbox_pred = self.retina_reg[i](reg_feat)
        return cls_score, bbox_pred
