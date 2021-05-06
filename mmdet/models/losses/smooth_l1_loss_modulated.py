import math

import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def smooth_l1_loss_modulated(pred, target, beta=1.0):
    assert beta > 0
    assert pred.shape[1] == 5
    assert target.shape[1] == 6

    diff1 = pred - target[:, :5]

    log_ratio = torch.log(target[:, 5])
    dx = diff1[:, 0]
    dy = diff1[:, 1]
    dw = pred[:, 2] - target[:, 3] + log_ratio
    dh = pred[:, 3] - target[:, 2] - log_ratio
    dangle = diff1[:, 4].abs() - math.pi / 2
    diff2 = torch.stack((dx, dy, dw, dh, dangle), -1)

    diff1 = diff1.abs()
    diff2 = diff2.abs()

    cond = (diff1.sum(1) < diff2.sum(1)).unsqueeze(-1)

    diff = torch.where(cond, diff1, diff2)

    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@LOSSES.register_module
class SmoothL1LossModulated(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1LossModulated, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.is_modulated_loss = True

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss_modulated(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
