"""FCOS training losses: Focal + GIoU + Binary-cross-entropy centerness."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from utils.box_ops import giou_loss, distance2bbox


def sigmoid_focal_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    p = torch.sigmoid(preds)
    ce = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class FCOSLoss(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        lcfg = cfg["loss"]
        self.cls_w  = lcfg["cls_loss_weight"]
        self.reg_w  = lcfg["reg_loss_weight"]
        self.ctr_w  = lcfg["centerness_loss_weight"]
        self.alpha  = lcfg["focal_alpha"]
        self.gamma  = lcfg["focal_gamma"]

    def forward(
        self,
        cls_preds:  List[torch.Tensor],
        reg_preds:  List[torch.Tensor],
        ctr_preds:  List[torch.Tensor],
        points:     List[torch.Tensor],
        targets:    List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        targets: list of dicts with keys
            boxes  (M, 4) x1y1x2y2
            labels (M,)
        """
        num_classes = cls_preds[0].shape[1]
        all_cls_t, all_reg_t, all_ctr_t, all_pos_mask = [], [], [], []

        for lvl, (pts, cls_p) in enumerate(zip(points, cls_preds)):
            h, w = cls_p.shape[2], cls_p.shape[3]
            n_pts = pts.shape[0]
            B = cls_p.shape[0]

            cls_t = torch.zeros(B, n_pts, num_classes, device=cls_p.device)
            reg_t = torch.zeros(B, n_pts, 4, device=cls_p.device)
            ctr_t = torch.zeros(B, n_pts, 1, device=cls_p.device)
            pos_m = torch.zeros(B, n_pts, dtype=torch.bool, device=cls_p.device)

            for b, tgt in enumerate(targets):
                if tgt["boxes"].numel() == 0:
                    continue
                boxes  = tgt["boxes"].to(cls_p.device)   # (M, 4)
                labels = tgt["labels"].to(cls_p.device)  # (M,)

                # Assign points inside GT boxes
                px, py = pts[:, 0], pts[:, 1]              # (N,)
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

                in_box = (
                    (px[:, None] >= x1[None]) & (px[:, None] <= x2[None]) &
                    (py[:, None] >= y1[None]) & (py[:, None] <= y2[None])
                )  # (N, M)

                for m in range(boxes.shape[0]):
                    pt_ids = in_box[:, m].nonzero(as_tuple=True)[0]
                    if pt_ids.numel() == 0:
                        continue
                    l = px[pt_ids] - x1[m]
                    t = py[pt_ids] - y1[m]
                    r = x2[m] - px[pt_ids]
                    bo = y2[m] - py[pt_ids]
                    reg_t[b, pt_ids] = torch.stack([l, t, r, bo], dim=-1)
                    # centerness
                    ctr_t[b, pt_ids, 0] = torch.sqrt(
                        (torch.minimum(l, r) / torch.maximum(l, r).clamp(1e-6)) *
                        (torch.minimum(t, bo) / torch.maximum(t, bo).clamp(1e-6))
                    )
                    cls_t[b, pt_ids, labels[m]] = 1.0
                    pos_m[b, pt_ids] = True

            all_cls_t.append(cls_t)
            all_reg_t.append(reg_t)
            all_ctr_t.append(ctr_t)
            all_pos_mask.append(pos_m)

        # Flatten levels
        def _flat_preds(preds, perm):
            return torch.cat([p.permute(*perm).reshape(p.shape[0], -1, p.shape[1]) for p in preds], dim=1)

        cls_pred_flat = _flat_preds(cls_preds, (0, 2, 3, 1))   # (B, N_total, C)
        ctr_pred_flat = _flat_preds(ctr_preds, (0, 2, 3, 1))   # (B, N_total, 1)
        reg_pred_flat = _flat_preds(reg_preds, (0, 2, 3, 1))   # (B, N_total, 4)

        cls_t_flat = torch.cat(all_cls_t, dim=1)
        reg_t_flat = torch.cat(all_reg_t, dim=1)
        ctr_t_flat = torch.cat(all_ctr_t, dim=1)
        pos_flat   = torch.cat(all_pos_mask, dim=1)

        num_pos = pos_flat.sum().clamp(min=1).float()

        # Classification loss (focal)
        cls_loss = sigmoid_focal_loss(cls_pred_flat, cls_t_flat, self.alpha, self.gamma) / num_pos

        # Regression + centerness only on positive points
        if pos_flat.any():
            pts_all = torch.cat(points, dim=0)  # (N_total, 2)

            def _decode(reg, pts):
                boxes = []
                for b in range(reg.shape[0]):
                    boxes.append(distance2bbox(pts, reg[b]))
                return torch.stack(boxes)

            reg_pred_pos = reg_pred_flat[pos_flat]
            reg_t_pos    = reg_t_flat[pos_flat]
            pts_pos      = pts_all[pos_flat.any(0)]

            # GIoU loss
            pred_boxes = distance2bbox(pts_pos.repeat(pos_flat.shape[0], 1)[:reg_pred_pos.shape[0]], reg_pred_pos)
            gt_boxes   = distance2bbox(pts_pos.repeat(pos_flat.shape[0], 1)[:reg_t_pos.shape[0]], reg_t_pos)
            reg_loss = giou_loss(pred_boxes, gt_boxes).sum() / num_pos

            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr_pred_flat[pos_flat], ctr_t_flat[pos_flat], reduction="sum"
            ) / num_pos
        else:
            reg_loss = reg_pred_flat.sum() * 0
            ctr_loss = ctr_pred_flat.sum() * 0

        total = self.cls_w * cls_loss + self.reg_w * reg_loss + self.ctr_w * ctr_loss
        return {"loss": total, "cls_loss": cls_loss, "reg_loss": reg_loss, "ctr_loss": ctr_loss}
