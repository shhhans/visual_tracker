"""
BSDS500 DataLoader
==================
期望的目录结构（解压 BSR_bsds500.tgz 后）：

    data/BSR/BSDS500/data/
        images/
            train/   200 × .jpg
            val/     100 × .jpg
            test/    200 × .jpg
        groundTruth/
            train/   200 × .mat
            val/     100 × .mat
            test/    200 × .mat

每个 .mat 文件包含一个 cell array `groundTruth`，每个 cell 有字段：
    .Boundaries  : H×W uint8，某标注者的二值边缘图
    .Segmentation: H×W uint32，分割图（我们不用）

GT 处理：
  - 将所有标注者的 Boundaries 取均值 → 软标签 gt_soft ∈ [0,1]
  - 训练用软标签；评估时将其与预测值一起扫阈值
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import scipy.io


class BSDS500Dataset(Dataset):
    """
    Args:
        root      : 含有 images/ 和 groundTruth/ 的根目录
                    默认 'data/BSR/BSDS500/data'
        split     : 'train' | 'val' | 'test'
        size      : 将图像 resize 到 (size, size)；None = 不 resize（原始分辨率，需自行处理批量）
        augment   : 训练时随机翻转/旋转
        soft_sigma: 对软标签再施一次 Gaussian 模糊（0 = 不做）
    """

    def __init__(
        self,
        root: str = "data/BSR/BSDS500/data",
        split: str = "train",
        size: int = 320,
        augment: bool = True,
        soft_sigma: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        self.size    = size
        self.augment = augment and (split == "train")
        self.soft_sigma = soft_sigma

        img_dir = os.path.join(root, "images", split)
        gt_dir  = os.path.join(root, "groundTruth", split)

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"BSDS500 not found at '{root}'.\n"
                "Download BSR_bsds500.tgz from:\n"
                "  https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz\n"
                "and extract to data/BSR/"
            )

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.samples = []
        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            gp   = os.path.join(gt_dir, stem + ".mat")
            if os.path.exists(gp):
                self.samples.append((ip, gp))

        assert len(self.samples) > 0, f"No matched image/GT pairs in {img_dir}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        # ── Image ─────────────────────────────────────────────────────────────
        img = Image.open(img_path).convert("RGB")
        if self.size is not None:
            img = img.resize((self.size, self.size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0   # H×W×3, [0,1]

        # ── GT: average all annotators ────────────────────────────────────────
        gt_soft = self._load_gt(gt_path, img.size[1], img.size[0])  # H×W

        # Optional extra smoothing
        if self.soft_sigma > 0:
            import cv2
            gt_soft = cv2.GaussianBlur(gt_soft, (0, 0), self.soft_sigma)
            mx = gt_soft.max()
            if mx > 1e-6:
                gt_soft = np.clip(gt_soft / mx, 0.0, 1.0)

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.augment:
            if np.random.rand() > 0.5:          # horizontal flip
                img_np = img_np[:, ::-1, :].copy()
                gt_soft = gt_soft[:, ::-1].copy()
            if np.random.rand() > 0.5:          # vertical flip
                img_np = img_np[::-1, :, :].copy()
                gt_soft = gt_soft[::-1, :].copy()
            # 90° rotation
            k = np.random.randint(4)
            if k:
                img_np  = np.rot90(img_np,  k).copy()
                gt_soft = np.rot90(gt_soft, k).copy()

        # ── To tensors ────────────────────────────────────────────────────────
        # image : (3, H, W), normalised to [-1, 1] to match synthetic pipeline
        img_t = torch.from_numpy(img_np.transpose(2, 0, 1)) * 2.0 - 1.0
        gt_t  = torch.from_numpy(gt_soft).unsqueeze(0)   # (1, H, W)

        return {"image": img_t, "edge_mask": gt_t, "path": img_path}

    # ──────────────────────────────────────────────────────────────────────────

    def _load_gt(self, mat_path: str, H: int, W: int) -> np.ndarray:
        """Return averaged boundary map, resized to (H, W)."""
        mat = scipy.io.loadmat(mat_path, squeeze_me=True)
        gt_cell = mat["groundTruth"]

        # gt_cell may be a 0-d object array; normalise to list
        if gt_cell.ndim == 0:
            gt_cell = [gt_cell.item()]
        else:
            gt_cell = list(gt_cell.flat)

        boundaries = []
        for ann in gt_cell:
            b = ann["Boundaries"]
            # scipy loads structured arrays differently depending on version
            if hasattr(b, "item"):
                b = b.item()
            b = np.array(b, dtype=np.float32)
            # BSDS500 GT is stored at original resolution; resize if needed
            if b.shape != (H, W):
                from PIL import Image as _PIL
                b_pil = _PIL.fromarray((b * 255).astype(np.uint8))
                b_pil = b_pil.resize((W, H), _PIL.NEAREST)
                b = np.array(b_pil, dtype=np.float32) / 255.0
            boundaries.append(b)

        gt_soft = np.mean(boundaries, axis=0).astype(np.float32)
        return gt_soft


def make_bsds_loaders(
    root: str = "data/BSR/BSDS500/data",
    size: int = 320,
    batch_size: int = 4,
    num_workers: int = 0,
    soft_sigma: float = 0.0,
):
    """Convenience: return (train_loader, val_loader)."""
    from torch.utils.data import DataLoader

    train_ds = BSDS500Dataset(root, split="train", size=size,
                              augment=True, soft_sigma=soft_sigma)
    val_ds   = BSDS500Dataset(root, split="val",   size=size,
                              augment=False, soft_sigma=0.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
