"""
Path 1 训练脚本 — 单帧边缘检测
================================
网络：UNet(in_ch=3, out_ch=1)
输入：单帧 RGB (3, 128, 128)
输出：边缘热图 (1, 128, 128)  logits → sigmoid → 概率
损失：Focal-BCE（处理正负样本严重不平衡：边缘像素 << 背景像素）

训练完后位移通过 两帧热图互相关 计算（见末尾 demo）。
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, os.path.dirname(__file__))
from data.synthetic import SyntheticEdgeDataset
from models.unet import UNet


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def focal_bce(logits: torch.Tensor, targets: torch.Tensor,
              alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal BCE for imbalanced edge detection.
    alpha  : weight for positive (edge) pixels — set high because edges are rare
    gamma  : focusing parameter (higher → more focus on hard examples)
    """
    bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p    = torch.sigmoid(logits)
    p_t  = p * targets + (1 - p) * (1 - targets)
    a_t  = alpha * targets + (1 - alpha) * (1 - targets)
    loss = a_t * (1 - p_t) ** gamma * bce
    return loss.mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    tp    = (preds * targets).sum()
    fp    = (preds * (1 - targets)).sum()
    fn    = ((1 - preds) * targets).sum()
    prec  = tp / (tp + fp + 1e-6)
    rec   = tp / (tp + fn + 1e-6)
    f1    = 2 * prec * rec / (prec + rec + 1e-6)
    return {"precision": prec.item(), "recall": rec.item(), "f1": f1.item()}


# ---------------------------------------------------------------------------
# Cross-correlation displacement estimator (inference-time, no training needed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_displacement(heatmap_t: torch.Tensor,
                          heatmap_t1: torch.Tensor,
                          max_shift: int = 20) -> tuple:
    """
    Estimate rigid (dx, dy) displacement between two edge heatmaps
    via normalised cross-correlation.

    heatmap_t, heatmap_t1 : (H, W) tensors in [0, 1]
    Returns (dx, dy) in pixels.
    """
    H, W = heatmap_t.shape
    # FFT-based cross-correlation
    f1 = torch.fft.rfft2(heatmap_t)
    f2 = torch.fft.rfft2(heatmap_t1)
    cc = torch.fft.irfft2(f1 * f2.conj(), s=(H, W))
    cc = torch.fft.fftshift(cc)

    # Restrict search window
    cy, cx = H // 2, W // 2
    region = cc[cy - max_shift:cy + max_shift + 1,
                cx - max_shift:cx + max_shift + 1]
    peak   = region.argmax()
    py, px = divmod(peak.item(), region.shape[1])
    dy     = py - max_shift
    dx     = px - max_shift
    return float(dx), float(dy)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Dataset
    ds_train = SyntheticEdgeDataset(
        size=args.img_size,
        texture_strength=args.texture,
        with_pairs=False,
        length=args.steps * args.batch,
        seed=0,
    )
    ds_val = SyntheticEdgeDataset(
        size=args.img_size,
        texture_strength=args.texture,
        with_pairs=True,       # val includes pairs for displacement demo
        length=200,
        seed=999,
    )

    # Model
    model = UNet(in_ch=3, out_ch=1, base_ch=args.base_ch).to(device)
    print(f"Parameters: {model.param_count():,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1   = 0.0
    step      = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0         = time.time()
        epoch_loss = 0.0

        for b in range(args.steps):
            batch = ds_train.get_batch(args.batch, start_idx=step % len(ds_train))
            imgs   = batch["image"].to(device)      # (B, 3, H, W)
            edges  = batch["edge_mask"].to(device)  # (B, 1, H, W)

            logits = model(imgs)                    # (B, 1, H, W)
            loss   = focal_bce(logits, edges, alpha=args.focal_alpha, gamma=args.focal_gamma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step       += args.batch

        scheduler.step()
        avg_loss = epoch_loss / args.steps

        # Validation
        model.eval()
        val_metrics = []
        for vi in range(0, 40, args.batch):
            vb = ds_val.get_batch(args.batch, start_idx=vi)
            with torch.no_grad():
                vlogits = model(vb["image"].to(device))
            val_metrics.append(compute_metrics(vlogits, vb["edge_mask"].to(device)))

        f1   = np.mean([m["f1"]        for m in val_metrics])
        prec = np.mean([m["precision"] for m in val_metrics])
        rec  = np.mean([m["recall"]    for m in val_metrics])

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  f1={f1:.3f}  p={prec:.3f}  r={rec:.3f}  "
              f"({elapsed:.1f}s)")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "f1": f1, "args": vars(args)},
                       os.path.join(args.save_dir, "path1_best.pt"))

    # ---------------------------------------------------------------------------
    # Displacement demo (cross-correlation on heatmap pairs)
    # ---------------------------------------------------------------------------
    print("\n--- Displacement Estimation Demo (cross-correlation) ---")
    model.eval()
    errs = []
    for i in range(20):
        s = ds_val[i]
        if "image_t1" not in s:
            continue
        img_t  = torch.from_numpy(s["image"]).unsqueeze(0)
        img_t1 = torch.from_numpy(s["image_t1"]).unsqueeze(0)
        with torch.no_grad():
            hm_t  = torch.sigmoid(model(img_t))[0, 0]
            hm_t1 = torch.sigmoid(model(img_t1))[0, 0]

        dx_pred, dy_pred = estimate_displacement(hm_t, hm_t1)

        # GT displacement = mean flow at boundary pixels
        flow  = s["flow"]       # (2, H, W)
        emask = s["edge_mask"][0]
        if emask.sum() > 0:
            dx_gt = float(flow[0][emask > 0].mean())
            dy_gt = float(flow[1][emask > 0].mean())
        else:
            dx_gt, dy_gt = 0., 0.

        err = math.sqrt((dx_pred - dx_gt) ** 2 + (dy_pred - dy_gt) ** 2)
        errs.append(err)
        if i < 5:
            print(f"  [{i}] GT=({dx_gt:+.1f},{dy_gt:+.1f})  "
                  f"pred=({dx_pred:+.1f},{dy_pred:+.1f})  err={err:.2f}px")

    print(f"Mean displacement error: {np.mean(errs):.2f} px  (over {len(errs)} samples)")
    print(f"\nBest validation F1: {best_f1:.3f}")
    print(f"Checkpoint saved to {args.save_dir}/path1_best.pt")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

import math

def parse_args():
    p = argparse.ArgumentParser(description="Path 1: single-frame edge detection")
    p.add_argument("--img_size",    type=int,   default=128)
    p.add_argument("--texture",     type=float, default=0.4,
                   help="Texture strength [0=flat, 1=heavy]")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--steps",       type=int,   default=200,
                   help="Gradient steps per epoch")
    p.add_argument("--batch",       type=int,   default=8)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--base_ch",     type=int,   default=32,
                   help="U-Net base channels (16=tiny, 32=default)")
    p.add_argument("--focal_alpha", type=float, default=0.75)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--save_dir",    type=str,   default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
