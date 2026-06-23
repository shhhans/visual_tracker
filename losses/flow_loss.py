"""
Losses for FlowNetC and the keypoint head.

FlowLoss   : multi-scale EPE (endpoint error) + edge-aware smoothness
KeypointLoss: focal-style heatmap loss + L1 offset loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


# ---------------------------------------------------------------------------
# FlowNet losses
# ---------------------------------------------------------------------------

def epe_loss(pred_flow: torch.Tensor, gt_flow: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Endpoint Error — L2 distance between predicted and ground-truth flow vectors.
    mask: (B, 1, H, W) float, 1 where GT flow is valid.
    """
    diff = pred_flow - gt_flow
    epe  = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-6)  # (B,1,H,W)
    if mask is not None:
        epe = epe * mask
        return epe.sum() / mask.sum().clamp(min=1)
    return epe.mean()


def smoothness_loss(flow: torch.Tensor, image: torch.Tensor = None) -> torch.Tensor:
    """
    Edge-aware first-order smoothness.
    If image is given, penalise flow gradients less at strong image edges.
    """
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]   # (B, 2, H, W-1)
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]   # (B, 2, H-1, W)

    if image is not None:
        img_dx = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(1, keepdim=True)
        img_dy = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(1, keepdim=True)
        weight_x = torch.exp(-img_dx * 10)
        weight_y = torch.exp(-img_dy * 10)
        return (dx.abs() * weight_x).mean() + (dy.abs() * weight_y).mean()

    return dx.abs().mean() + dy.abs().mean()


class FlowLoss(nn.Module):
    """
    Multi-scale EPE + smoothness for FlowNetC.

    The network outputs flows at strides [/8, /4, /2] (coarse→fine).
    GT flow is downsampled to match each level.

    Loss weights follow the FlowNet paper: coarser levels count less.
    """

    def __init__(
        self,
        level_weights: List[float] = (0.32, 0.16, 0.08),
        smooth_weight: float = 0.1,
    ):
        super().__init__()
        # finest level last → weights[0] = /8, weights[1] = /4, weights[2] = /2
        self.level_weights = level_weights
        self.smooth_weight = smooth_weight

    def forward(
        self,
        pred_flows: List[torch.Tensor],   # coarse → fine
        gt_flow:    torch.Tensor,          # (B, 2, H, W) full resolution
        frame_t:    torch.Tensor = None,   # (B, 3, H, W) for edge-aware smoothness
        flow_mask:  torch.Tensor = None,   # (B, 1, H, W) valid-flow mask
    ) -> Dict[str, torch.Tensor]:

        total_epe = gt_flow.new_zeros(1)

        for pred, w in zip(pred_flows, self.level_weights):
            H_p, W_p = pred.shape[-2:]
            # Downsample GT to match prediction resolution
            gt_down = F.interpolate(gt_flow, size=(H_p, W_p), mode="bilinear", align_corners=False)
            gt_down[:, 0] *= W_p / gt_flow.shape[-1]
            gt_down[:, 1] *= H_p / gt_flow.shape[-2]

            mask_down = None
            if flow_mask is not None:
                mask_down = F.interpolate(flow_mask, size=(H_p, W_p), mode="nearest")

            total_epe = total_epe + w * epe_loss(pred, gt_down, mask_down)

        # Smoothness on the finest prediction
        smooth = smoothness_loss(pred_flows[-1], frame_t)
        total  = total_epe + self.smooth_weight * smooth

        return {"loss": total, "epe": total_epe, "smooth": smooth}


# ---------------------------------------------------------------------------
# Keypoint head losses
# ---------------------------------------------------------------------------

def keypoint_focal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Modified focal loss for heatmap regression (CornerNet variant).
    pred   : (B, K, H, W)  raw logits
    target : (B, K, H, W)  Gaussian heatmaps in [0, 1]
    """
    p  = pred.sigmoid()
    # Negative weighting: reduce penalty near GT peaks
    neg_weights = (1 - target) ** 4

    pos_loss = -(target * torch.log(p.clamp(1e-6)) * (1 - p) ** 2)
    neg_loss = -(neg_weights * torch.log((1 - p).clamp(1e-6)) * p ** 2)

    num_pos = (target == 1).sum().clamp(min=1).float()
    return (pos_loss + neg_loss * (1 - target)).sum() / num_pos


class KeypointLoss(nn.Module):
    def __init__(self, hm_weight: float = 1.0, off_weight: float = 0.5):
        super().__init__()
        self.hm_w  = hm_weight
        self.off_w = off_weight

    def forward(
        self,
        pred_hm:     torch.Tensor,   # (B, K, H, W) raw logits
        pred_offset: torch.Tensor,   # (B, K*2, H, W)
        gt_hm:       torch.Tensor,   # (B, K, H, W) Gaussian targets
        gt_offset:   torch.Tensor,   # (B, K*2, H, W)
        offset_mask: torch.Tensor,   # (B, K, H, W)
    ) -> Dict[str, torch.Tensor]:

        hm_loss = keypoint_focal_loss(pred_hm, gt_hm)

        # L1 offset loss only at keypoint locations
        # Expand mask to cover both dx and dy channels
        mask2 = offset_mask.repeat_interleave(2, dim=1)  # (B, K*2, H, W)
        num   = mask2.sum().clamp(min=1)
        off_loss = (F.l1_loss(pred_offset, gt_offset, reduction="none") * mask2).sum() / num

        total = self.hm_w * hm_loss + self.off_w * off_loss
        return {"loss": total, "hm_loss": hm_loss, "off_loss": off_loss}
