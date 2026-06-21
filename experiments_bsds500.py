"""
BSDS500 训练与评估
===================
用 BSDS500 真实图片训练 HED-style UNet（深度监督），
与顶尖模型结果进行对比。

数据集目录（下载并解压后）：
    data/BSR/BSDS500/data/images/{train,val,test}/
    data/BSR/BSDS500/data/groundTruth/{train,val,test}/

下载地址：
    https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz

解压方法：
    cd data && tar -xzf BSR_bsds500.tgz

SOTA 对比（BSDS500 test set，ODS F1）：
    Canny (1986)          : 0.611   无预训练
    gPb   (2011)          : 0.726   无预训练
    HED   (2015, VGG-16)  : 0.790   ImageNet 预训练
    RCF   (2017, VGG-16)  : 0.806   ImageNet 预训练
    Human upper bound     : ~0.803
    ─────────────────────────────────────
    本脚本（无预训练骨干） : 预估 0.70-0.74

评估指标：
    ODS F1：在整个 val set 上扫阈值，取最优阈值下的 F1
    GT 二值化阈值：0.3（BSDS500 软标签 > 0.3 视为边缘）
    注意：我们使用像素精确匹配，SOTA 用 1px 容忍距离（tolerant matching），
          因此我们的数字会偏低约 0.03-0.05，属正常差异。
"""

import os, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from data.bsds500 import BSDS500Dataset
from models.unet_hed import HEDUNet

# ─── 配置 ────────────────────────────────────────────────────────────────────

BSDS_ROOT = "data/BSR/BSDS500/data"

TRAIN_CFG = dict(
    img_size     = 320,     # resize 到 320×320（保留足够分辨率）
    batch        = 4,
    epochs       = 30,
    lr           = 1e-3,
    base_ch      = 32,
    focal_alpha  = 0.75,    # 正样本权重（边缘像素占少数）
    focal_gamma  = 2.0,
    side_weight  = 0.5,     # 深度监督侧分支 loss 权重
    soft_sigma   = 0.0,     # GT 额外平滑（0 = 不做）
    gt_threshold = 0.3,     # 软标签 > 0.3 视为边缘（ODS 评估）
)

ODS_THRESHOLDS = np.linspace(0.05, 0.95, 19).tolist()

# ─── Loss / Metrics ──────────────────────────────────────────────────────────

def focal_bce(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p   = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


def hed_loss(outputs, gt, alpha, gamma, side_w):
    """
    outputs : (logit_final, s1, s2, s3) — 训练模式下的四个输出
    gt      : (B, 1, H, W) 软标签
    """
    logit_final, s1, s2, s3 = outputs
    L_final = focal_bce(logit_final, gt, alpha, gamma)
    L_sides = (focal_bce(s1, gt, alpha, gamma) +
               focal_bce(s2, gt, alpha, gamma) +
               focal_bce(s3, gt, alpha, gamma)) / 3.0
    return L_final + side_w * L_sides, L_final.item(), L_sides.item()


@torch.no_grad()
def ods_f1_val(model, val_ds, gt_thr=0.3, n_max=100):
    """
    在 val 集上计算 ODS F1。
    GT 软标签 > gt_thr 视为边缘（二值化）。
    使用像素精确匹配（非容忍距离）。
    """
    stats = {t: np.zeros(3) for t in ODS_THRESHOLDS}
    n = min(len(val_ds), n_max)

    for i in range(n):
        item = val_ds[i]
        inp  = item["image"].unsqueeze(0)
        gt_s = item["edge_mask"].squeeze().numpy()   # H×W soft

        logit = model(inp)                           # eval mode → single output
        prob  = torch.sigmoid(logit).squeeze().numpy()

        gt_hard = (gt_s > gt_thr).astype(np.float32)

        for t in ODS_THRESHOLDS:
            pred = (prob > t).astype(np.float32)
            stats[t][0] += (pred * gt_hard).sum()           # TP
            stats[t][1] += (pred * (1 - gt_hard)).sum()     # FP
            stats[t][2] += ((1 - pred) * gt_hard).sum()     # FN

    best_f1, best_thr = 0.0, 0.5
    for t, (tp, fp, fn) in stats.items():
        p  = tp / (tp + fp + 1e-6)
        r  = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    return float(best_f1), float(best_thr)

# ─── 训练 ────────────────────────────────────────────────────────────────────

def run():
    device = torch.device("cpu")

    train_ds = BSDS500Dataset(
        root    = BSDS_ROOT,
        split   = "train",
        size    = TRAIN_CFG["img_size"],
        augment = True,
        soft_sigma = TRAIN_CFG["soft_sigma"],
    )
    val_ds = BSDS500Dataset(
        root    = BSDS_ROOT,
        split   = "val",
        size    = TRAIN_CFG["img_size"],
        augment = False,
    )

    print(f"  Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    model     = HEDUNet(in_ch=3, base_ch=TRAIN_CFG["base_ch"]).to(device)
    print(f"  Model params: {model.param_count():,}")

    optimizer = Adam(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=TRAIN_CFG["epochs"],
                                  eta_min=TRAIN_CFG["lr"] * 0.01)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CFG["batch"],
                              shuffle=True, drop_last=True)

    best_ods = 0.0
    history  = []

    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        model.train()
        t0   = time.time()
        ep_l = 0.0
        n_steps = 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            gts  = batch["edge_mask"].to(device)

            outputs = model(imgs)
            loss, _, _ = hed_loss(
                outputs, gts,
                TRAIN_CFG["focal_alpha"],
                TRAIN_CFG["focal_gamma"],
                TRAIN_CFG["side_weight"],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_l   += loss.item()
            n_steps += 1

        scheduler.step()

        # ── 验证 ──────────────────────────────────────────────────────────────
        model.eval()
        ods, best_thr = ods_f1_val(
            model, val_ds,
            gt_thr=TRAIN_CFG["gt_threshold"],
            n_max=100,
        )
        avg_loss = ep_l / max(n_steps, 1)
        history.append({"epoch": epoch, "loss": avg_loss,
                        "ods": ods, "thr": best_thr})

        print(f"  Epoch {epoch:3d}/{TRAIN_CFG['epochs']}  "
              f"loss={avg_loss:.4f}  ODS={ods:.3f}(thr={best_thr:.2f})  "
              f"({time.time()-t0:.1f}s)")

        if ods > best_ods:
            best_ods = ods
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "ods": ods, "best_thr": best_thr, "history": history,
            }, "checkpoints/exp_bsds500_best.pt")

    print(f"\n  ✓ Best ODS F1 = {best_ods:.3f}")
    return history, model, best_ods


# ─── 可视化 ──────────────────────────────────────────────────────────────────

def visualize(history, model):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    val_ds = BSDS500Dataset(
        root=BSDS_ROOT, split="val",
        size=TRAIN_CFG["img_size"], augment=False,
    )
    ck = torch.load("checkpoints/exp_bsds500_best.pt",
                    map_location="cpu", weights_only=False)
    best_thr = ck["best_thr"]

    # ── 训练曲线 ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [h["epoch"] for h in history]

    axes[0].plot(epochs, [h["ods"]  for h in history], color="darkorange", lw=2)
    axes[0].axhline(0.611, ls="--", color="gray",      lw=1, label="Canny 0.611")
    axes[0].axhline(0.726, ls="--", color="steelblue", lw=1, label="gPb  0.726")
    axes[0].axhline(0.790, ls="--", color="crimson",   lw=1, label="HED  0.790 (VGG)")
    axes[0].axhline(0.803, ls=":",  color="black",     lw=1, label="Human ~0.803")
    axes[0].set_title("ODS F1 on BSDS500 val (↑ better)", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0.4, 0.85)

    axes[1].plot(epochs, [h["loss"] for h in history], color="mediumseagreen", lw=2)
    axes[1].set_title("Training Loss (↓)", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=0.3)

    fig.suptitle("HED-UNet (no pretrained backbone) on BSDS500",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/bsds500/viz_bsds500_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved viz_bsds500_curves.png")

    # ── 样本网格 ──────────────────────────────────────────────────────────────
    n = 5
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 2.8),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})
    titles = ["Input (BSDS500)", "GT (soft avg, thr=0.3)", "Pred edge"]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=8, fontweight="bold", pad=3)

    model.eval()
    for i in range(n):
        item = val_ds[i]
        inp  = item["image"].unsqueeze(0)
        gt_s = item["edge_mask"].squeeze().numpy()

        with torch.no_grad():
            logit = model(inp)
        prob = torch.sigmoid(logit).squeeze().numpy()

        img_np = ((item["image"].numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
        gt_hard = (gt_s > TRAIN_CFG["gt_threshold"]).astype(np.uint8) * 255
        pred_b  = (prob > best_thr).astype(np.uint8) * 255

        # F1 for this sample
        gt_f = (gt_s > TRAIN_CFG["gt_threshold"]).astype(float)
        pr_f = (prob > best_thr).astype(float)
        tp = (pr_f * gt_f).sum(); fp = (pr_f * (1-gt_f)).sum(); fn = ((1-pr_f) * gt_f).sum()
        f1 = 2*tp / (2*tp + fp + fn + 1e-6)

        axes[i, 0].imshow(img_np); axes[i, 0].axis("off")
        axes[i, 1].imshow(gt_hard, cmap="gray"); axes[i, 1].axis("off")
        axes[i, 2].imshow(pred_b, cmap="gray"); axes[i, 2].axis("off")
        axes[i, 2].set_xlabel(f"F1={f1:.3f}", fontsize=7.5, labelpad=2)
        axes[i, 0].set_ylabel(f"#{i+1}", fontsize=7.5, va="center", labelpad=4)

    fig.suptitle(f"HED-UNet on BSDS500 val\n(thr={best_thr:.2f}, no pretrained backbone)",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.savefig("results/bsds500/viz_bsds500_samples.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved viz_bsds500_samples.png")

    # ── SOTA 对比摘要 ─────────────────────────────────────────────────────────
    best = max(history, key=lambda h: h["ods"])
    print("\n" + "=" * 62)
    print(f"{'模型':<30s}  {'ODS F1':>8s}  {'骨干':>12s}")
    print("-" * 62)
    rows = [
        ("Canny (1986)",           0.611, "无"),
        ("gPb (2011)",             0.726, "无"),
        (f"本脚本 HED-UNet (epoch {best['epoch']})", best["ods"], "无"),
        ("HED (2015)",             0.790, "VGG-16/ImageNet"),
        ("RCF (2017)",             0.806, "VGG-16/ImageNet"),
        ("Human upper bound",      0.803, "—"),
    ]
    for name, ods, backbone in sorted(rows, key=lambda r: r[1]):
        marker = " ◀" if "HED-UNet" in name else ""
        print(f"  {name:<28s}  {ods:>8.3f}  {backbone:>12s}{marker}")
    print("=" * 62)
    print("注：SOTA 数字使用 1px 容忍距离；本脚本为像素精确，偏低约 0.03-0.05")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results/bsds500", exist_ok=True)
    history, model, best_ods = run()
    visualize(history, model)


if __name__ == "__main__":
    main()
