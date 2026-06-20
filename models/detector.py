"""Full detector: backbone + FPN neck + FCOS head."""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .backbone import ResNetBackbone
from .neck import FPN
from .head import FCOSHead
from utils.box_ops import distance2bbox, batched_nms


class Detector(nn.Module):
    """End-to-end single-stage object detector."""

    def __init__(self, cfg: Dict):
        super().__init__()
        bcfg = cfg["model"]["backbone"]
        ncfg = cfg["model"]["neck"]
        hcfg = cfg["model"]["head"]

        self.backbone = ResNetBackbone(
            depth=bcfg["depth"],
            pretrained=bcfg.get("pretrained", True),
            out_indices=bcfg.get("out_indices", [1, 2, 3, 4]),
        )
        self.neck = FPN(
            in_channels=ncfg["in_channels"],
            out_channels=ncfg["out_channels"],
            num_levels=ncfg["num_levels"],
        )
        self.head = FCOSHead(
            in_channels=ncfg["out_channels"],
            num_classes=hcfg["num_classes"],
            strides=hcfg["strides"],
        )
        self.strides = hcfg["strides"]
        self.num_classes = hcfg["num_classes"]

    def forward(self, images: torch.Tensor):
        feats = self.backbone(images)
        fpn_feats = self.neck(feats)
        cls_preds, reg_preds, ctr_preds = self.head(fpn_feats)
        return cls_preds, reg_preds, ctr_preds, fpn_feats

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        conf_thresh: float = 0.05,
        nms_thresh: float = 0.6,
        max_dets: int = 300,
    ) -> List[Dict]:
        """
        Returns a list of dicts (one per image):
            boxes  : (N, 4) x1y1x2y2 in pixel coords
            scores : (N,)
            labels : (N,) int
        """
        cls_preds, reg_preds, ctr_preds, fpn_feats = self(images)
        points = self.head.get_points(fpn_feats, images.device)

        batch_size = images.shape[0]
        results = []

        for b in range(batch_size):
            all_boxes, all_scores, all_labels = [], [], []

            for cls_p, reg_p, ctr_p, pts, stride in zip(
                cls_preds, reg_preds, ctr_preds, points, self.strides
            ):
                # cls_p: (B, C, H, W) → (HW, C)
                cls  = cls_p[b].permute(1, 2, 0).reshape(-1, self.num_classes)
                ctr  = ctr_p[b].permute(1, 2, 0).reshape(-1, 1).sigmoid()
                reg  = reg_p[b].permute(1, 2, 0).reshape(-1, 4)

                scores = cls.sigmoid() * ctr
                max_scores, labels = scores.max(dim=-1)

                keep = max_scores > conf_thresh
                if not keep.any():
                    continue

                boxes = distance2bbox(pts[keep], reg[keep])
                all_boxes.append(boxes)
                all_scores.append(max_scores[keep])
                all_labels.append(labels[keep])

            if not all_boxes:
                results.append({"boxes": torch.empty(0, 4), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)})
                continue

            boxes  = torch.cat(all_boxes)
            scores = torch.cat(all_scores)
            labels = torch.cat(all_labels)

            keep = batched_nms(boxes, scores, labels, nms_thresh)[:max_dets]
            results.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})

        return results
