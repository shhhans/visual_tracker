"""
Path 2 训练脚本 — 帧对输入，直接输出边界 + 位移向量
======================================================
网络：UNet(in_ch=6, out_ch=3)
输入：[frame_t | frame_t+1] 在通道维度拼接 (6, 128, 128)
输出：
  ch 0   : 边缘存在概率 logit  (sigmoid → mask)
  ch 1-2 : 该像素处的 (dx, dy) 位移 (仅在边界像素上有意义)

损失：
  L_edge : Focal-BCE  (边缘检测)
  L_flow : Masked-L1  (仅在 GT 边界处监督 flow)
  L_total = L_edge + λ * L_flow
"""

import os
import time
import argparse
import math
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
# Losses
# ---------------------------------------------------------------------------

def focal_bce(logits: torch.Tensor, targets: torch.Tensor,
              alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p    = torch.sigmoid(logits)
    p_t  = p * targets + (1 - p) * (1 - targets)
    a_t  = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


def masked_flow_loss(pred_flow: torch.Tensor,
                     gt_flow:   torch.Tensor,
                     mask:      torch.Tensor) -> torch.Tensor:
    """
    L1 loss on flow vectors, averaged only over GT boundary pixels.
    pred_flow : (B, 2, H, W)
    gt_flow   : (B, 2, H, W)   — zero everywhere except boundary pixels
    mask      : (B, 1, H, W)   — 1 at boundary pixels
    """
    diff   = (pred_flow - gt_flow).abs()    # (B, 2, H, W)
    masked = diff * mask                    # zero out non-boundary
    num    = mask.sum() * 2 + 1e-6         # ×2 for dx+dy channels
    return masked.sum() / num


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(pred_logit: torch.Tensor, pred_flow: torch.Tensor,
                    gt_mask: torch.Tensor, gt_flow: torch.Tensor,
                    threshold: float = 0.5):
    # Edge F1
    pred_mask = (torch.sigmoid(pred_logit) > threshold).float()
    tp  = (pred_mask * gt_mask).sum()
    fp  = (pred_mask * (1 - gt_mask)).sum()
    fn  = ((1 - pred_mask) * gt_mask).sum()
    p   = tp / (tp + fp + 1e-6)
    r   = tp / (tp + fn + 1e-6)
    f1  = 2 * p * r / (p + r + 1e-6)

    # Flow EPE at GT boundary pixels
    mask     = gt_mask                                    # (B, 1, H, W)
    err_xy   = (pred_flow - gt_flow).pow(2).sum(dim=1, keepdim=True).sqrt()
    epe_vals = err_xy[mask.bool()]
    epe      = epe_vals.mean() if epe_vals.numel() > 0 else torch.tensor(0.)

    return {"f1": f1.item(), "precision": p.item(),
            "recall": r.item(), "epe": epe.item()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cpu")
    print(f"Device: {device}")

    ds_train = SyntheticEdgeDataset(
        size=args.img_size,
        texture_strength=args.texture,
        with_pairs=True,
        max_shift=args.max_shift,
        max_rot=args.max_rot,
        length=args.steps * args.batch,
        seed=0,
    )
    ds_val = SyntheticEdgeDataset(
        size=args.img_size,
        texture_strength=args.texture,
        with_pairs=True,
        max_shift=args.max_shift,
        max_rot=args.max_rot,
        length=200,
        seed=999,
    )

    # 6-channel input (frame_t + frame_t+1), 3-channel output (edge + dx + dy)
    model = UNet(in_ch=6, out_ch=3, base_ch=args.base_ch).to(device)
    print(f"Parameters: {model.param_count():,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    os.makedirs(args.save_dir, exist_ok=True)
    best_f1 = 0.0
    step    = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0         = time.time()
        epoch_loss = 0.0
        epoch_el   = 0.0   # edge loss component
        epoch_fl   = 0.0   # flow loss component

        for b in range(args.steps):
            batch = ds_train.get_batch(args.batch, start_idx=step % len(ds_train))

            # Concatenate frame pair on channel dimension
            x      = torch.cat([batch["image"], batch["image_t1"]], dim=1).to(device)  # (B,6,H,W)
            gt_edge = batch["edge_mask"].to(device)   # (B, 1, H, W)
            gt_flow = batch["flow"].to(device)         # (B, 2, H, W)

            out        = model(x)                      # (B, 3, H, W)
            pred_logit = out[:, :1]                    # (B, 1, H, W)
            pred_flow  = out[:, 1:]                    # (B, 2, H, W)

            l_edge  = focal_bce(pred_logit, gt_edge,
                                alpha=args.focal_alpha, gamma=args.focal_gamma)
            l_flow  = masked_flow_loss(pred_flow, gt_flow, gt_edge)
            loss    = l_edge + args.flow_weight * l_flow

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_el   += l_edge.item()
            epoch_fl   += l_flow.item()
            step       += args.batch

        scheduler.step()

        # Validation
        model.eval()
        val_metrics = []
        for vi in range(0, 40, args.batch):
            vb = ds_val.get_batch(args.batch, start_idx=vi)
            vx = torch.cat([vb["image"], vb["image_t1"]], dim=1).to(device)
            with torch.no_grad():
                vout = model(vx)
            val_metrics.append(compute_metrics(
                vout[:, :1], vout[:, 1:],
                vb["edge_mask"].to(device), vb["flow"].to(device)
            ))

        f1  = np.mean([m["f1"]  for m in val_metrics])
        epe = np.mean([m["epe"] for m in val_metrics])
        p   = np.mean([m["precision"] for m in val_metrics])
        r   = np.mean([m["recall"]    for m in val_metrics])

        elapsed = time.time() - t0
        n = args.steps
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={epoch_loss/n:.4f}  "
              f"edge={epoch_el/n:.4f}  flow={epoch_fl/n:.4f}  "
              f"| val f1={f1:.3f} p={p:.3f} r={r:.3f} EPE={epe:.2f}px  "
              f"({elapsed:.1f}s)")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "f1": f1, "epe": epe, "args": vars(args)},
                       os.path.join(args.save_dir, "path2_best.pt"))

    print(f"\nBest val F1: {best_f1:.3f}")
    print(f"Checkpoint: {args.save_dir}/path2_best.pt")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Path 2: boundary flow estimation")
    p.add_argument("--img_size",    type=int,   default=128)
    p.add_argument("--texture",     type=float, default=0.4)
    p.add_argument("--max_shift",   type=float, default=8.0,
                   help="Max pixel shift between frame pair")
    p.add_argument("--max_rot",     type=float, default=10.0,
                   help="Max rotation between frame pair (degrees)")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--steps",       type=int,   default=200)
    p.add_argument("--batch",       type=int,   default=8)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--base_ch",     type=int,   default=32)
    p.add_argument("--focal_alpha", type=float, default=0.75)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--flow_weight", type=float, default=1.0,
                   help="Weight of flow loss relative to edge loss")
    p.add_argument("--save_dir",    type=str,   default="checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
