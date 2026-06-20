"""Bounding box utilities."""
import torch
import torchvision.ops as tv_ops
from typing import Optional


def distance2bbox(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy) + (l, t, r, b) distances to (x1, y1, x2, y2)."""
    x1 = points[:, 0] - distances[:, 0]
    y1 = points[:, 1] - distances[:, 1]
    x2 = points[:, 0] + distances[:, 2]
    y2 = points[:, 1] + distances[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox2distance(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) boxes + anchor points to (l, t, r, b)."""
    l = points[:, 0] - boxes[:, 0]
    t = points[:, 1] - boxes[:, 1]
    r = boxes[:, 2] - points[:, 0]
    b = boxes[:, 3] - points[:, 1]
    return torch.stack([l, t, r, b], dim=-1)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """NMS applied per class to avoid cross-class suppression."""
    return tv_ops.batched_nms(boxes.float(), scores.float(), labels, iou_threshold)


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    return tv_ops.box_iou(boxes_a, boxes_b)


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Generalised IoU loss (element-wise)."""
    return tv_ops.generalized_box_iou_loss(pred, target, reduction="none")
