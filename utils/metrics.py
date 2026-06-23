"""COCO-style mean Average Precision."""
import torch
import numpy as np
from typing import List, Dict, Optional


class MeanAveragePrecision:
    """
    Accumulates predictions and ground-truths, then computes mAP at IoU 0.5.
    Compatible with COCO evaluation conventions.
    """

    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self._preds: List[Dict] = []
        self._gts:   List[Dict] = []

    def update(self, preds: List[Dict], targets: List[Dict]):
        self._preds.extend(preds)
        self._gts.extend(targets)

    def compute(self) -> Dict[str, float]:
        aps = []
        for cls in range(self.num_classes):
            ap = self._ap_per_class(cls)
            if ap is not None:
                aps.append(ap)
        mAP = float(np.mean(aps)) if aps else 0.0
        return {"mAP": mAP, "num_classes_with_gt": len(aps)}

    def _ap_per_class(self, cls: int) -> Optional[float]:
        # Gather GT boxes for this class
        gt_by_img = {}
        for img_id, gt in enumerate(self._gts):
            mask = gt["labels"] == cls
            gt_by_img[img_id] = gt["boxes"][mask].numpy() if mask.any() else np.empty((0, 4))

        total_gt = sum(len(b) for b in gt_by_img.values())
        if total_gt == 0:
            return None

        # Gather predictions sorted by score
        all_scores, all_tp, all_fp = [], [], []
        for img_id, pred in enumerate(self._preds):
            mask = pred["labels"] == cls
            boxes  = pred["boxes"][mask].numpy()
            scores = pred["scores"][mask].numpy()

            gt_boxes = gt_by_img.get(img_id, np.empty((0, 4)))
            matched = np.zeros(len(gt_boxes), dtype=bool)

            order = np.argsort(-scores)
            for idx in order:
                all_scores.append(scores[idx])
                if len(gt_boxes) == 0:
                    all_tp.append(0); all_fp.append(1)
                    continue
                ious = self._iou(boxes[idx][None], gt_boxes)[0]
                best = ious.argmax()
                if ious[best] >= self.iou_threshold and not matched[best]:
                    matched[best] = True
                    all_tp.append(1); all_fp.append(0)
                else:
                    all_tp.append(0); all_fp.append(1)

        order = np.argsort(-np.array(all_scores))
        tp = np.cumsum(np.array(all_tp)[order])
        fp = np.cumsum(np.array(all_fp)[order])
        recall    = tp / total_gt
        precision = tp / (tp + fp + 1e-6)
        return float(self._voc_ap(recall, precision))

    @staticmethod
    def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[0.0], precision, [0.0]])
        for i in range(mpre.size - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        ix1 = np.maximum(ax1[:, None], bx1[None])
        iy1 = np.maximum(ay1[:, None], by1[None])
        ix2 = np.minimum(ax2[:, None], bx2[None])
        iy2 = np.minimum(ay2[:, None], by2[None])
        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a[:, None] + area_b[None] - inter + 1e-6)
