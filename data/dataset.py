"""COCO-compatible detection dataset with online augmentation."""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from typing import Dict, List, Optional, Tuple


class COCODetectionDataset(Dataset):
    """Loads images + annotations from a COCO-format JSON file."""

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        img_size: int = 640,
        augment: bool = False,
    ):
        self.coco     = COCO(ann_file)
        self.img_dir  = img_dir
        self.img_size = img_size
        self.augment  = augment

        # Filter images with at least one annotation
        self.img_ids = sorted([
            img_id for img_id in self.coco.imgs
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ])

        # Continuous category mapping
        cat_ids = sorted(self.coco.getCatIds())
        self.cat2label = {c: i for i, c in enumerate(cat_ids)}
        self.label2cat = {i: c for c, i in self.cat2label.items()}

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id  = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        path = os.path.join(self.img_dir, img_info["file_name"])

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)
        boxes, labels = self._parse_anns(anns, image.shape[:2])

        if self.augment:
            image, boxes = self._augment(image, boxes)

        image, scale = self._resize(image)
        boxes = boxes * scale if len(boxes) else boxes

        image  = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image  = self._normalize(image)
        target = {
            "boxes":  torch.from_numpy(boxes).float(),
            "labels": torch.from_numpy(labels).long(),
            "img_id": img_id,
        }
        return image, target

    def _parse_anns(self, anns: List[Dict], img_shape: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img_shape
        boxes, labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat2label[ann["category_id"]])
        if boxes:
            return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

    def _resize(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        # Pad to square
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=114)
        return image, scale

    def _augment(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Random horizontal flip
        if np.random.random() < 0.5:
            image = image[:, ::-1].copy()
            if len(boxes):
                w = image.shape[1]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        # Random brightness / contrast
        if np.random.random() < 0.5:
            alpha = np.random.uniform(0.7, 1.3)
            beta  = np.random.randint(-30, 30)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return image, boxes

    @staticmethod
    def _normalize(t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        return (t - mean) / std


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def build_dataloader(cfg: Dict, split: str = "train") -> DataLoader:
    is_train = split == "train"
    ann_key  = f"{split}_ann"
    img_key  = f"{split}_img"
    dataset  = COCODetectionDataset(
        ann_file=cfg["data"][ann_key],
        img_dir =cfg["data"][img_key],
        img_size=cfg["data"]["img_size"],
        augment =is_train and cfg["data"].get("augment", True),
    )
    return DataLoader(
        dataset,
        batch_size =cfg["train"]["batch_size"],
        shuffle    =is_train,
        num_workers=cfg["data"]["num_workers"],
        collate_fn =collate_fn,
        pin_memory =True,
    )
