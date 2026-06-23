"""
Antialias soft-label experiment
================================
比较两种 antialias 训练方式：

  hard : antialias=True，GT = INTER_AREA 下采样后的硬/随机软值（原始方式）
  soft : antialias=True，GT = 上述基础上再用 σ=1.5px Gaussian blur，
         归一化为一致的热力图

评估：ODS F1（最优阈值下的 F1，阈值在验证集上搜索）。
训练时用 hard label 的 0.5 阈值 F1 监控进度（快），保存时用 ODS F1。

输出：
  checkpoints/exp_aa_hard_best.pt
  checkpoints/exp_aa_soft_best.pt
  viz_aa_soft_compare.png
"""

import os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from data.synthetic import SyntheticEdgeDataset
from models.unet import UNet


# ─── config ──────────────────────────────────────────────────────────────────

CONFIGS = [
    {
        "name":   "aa_hard",
        "label":  "Antialias — hard label\n(INTER_AREA, fixed thr=0.5)",
        "color":  "steelblue",
        "soft_sigma": 0.0,
    },
    {
        "name":   "aa_soft",
        "label":  "Antialias — soft label\n(Gaussian σ=1.5, ODS thr)",
        "color":  "darkorange",
        "soft_sigma": 1.5,
    },
]

TRAIN_CFG = dict(
    img_size    = 128,
    texture     = 0.4,
    epochs      = 30,
    steps       = 50,
    batch       = 8,
    lr          = 3e-4,
    base_ch     = 16,
    focal_alpha = 0.75,
    focal_gamma = 2.0,
)

ODS_THRESHOLDS = np.linspace(0.1, 0.9, 17).tolist()   # 17 points: 0.1, 0.15, ..., 0.9


# ─── loss / metrics ──────────────────────────────────────────────────────────

def focal_bce(logits, targets, alpha=0.75, gamma=2.0):
    bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p    = torch.sigmoid(logits)
    p_t  = p * targets + (1 - p) * (1 - targets)
    a_t  = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


@torch.no_grad()
def f1_fixed(logits, targets_hard, thr=0.5):
    """F1 at a fixed threshold; targets_hard must be binary {0,1}."""
    pred = (torch.sigmoid(logits) > thr).float()
    tp = (pred * targets_hard).sum()
    fp = (pred * (1 - targets_hard)).sum()
    fn = ((1 - pred) * targets_hard).sum()
    p  = tp / (tp + fp + 1e-6)
    r  = tp / (tp + fn + 1e-6)
    return float(2 * p * r / (p + r + 1e-6))


@torch.no_grad()
def ods_f1(model, ds_val, n_samples=80):
    """
    Optimal-Dataset-Scale F1: sweep thresholds over all val samples,
    pick the threshold that gives the best aggregate F1.

    GT is binarised at 0.5 (geometric edge as ground truth),
    regardless of whether training used soft labels.
    """
    # Accumulate TP/FP/FN per threshold across all samples
    stats = {t: np.zeros(3) for t in ODS_THRESHOLDS}   # [tp, fp, fn]

    for i in range(n_samples):
        s   = ds_val[i]
        inp = torch.from_numpy(s["image"]).unsqueeze(0)
        logit = model(inp)
        prob  = torch.sigmoid(logit).squeeze().numpy()

        gt_hard = (s["edge_mask"][0] > 0.5).astype(np.float32)   # binary GT

        for t in ODS_THRESHOLDS:
            pred = (prob > t).astype(np.float32)
            stats[t][0] += (pred * gt_hard).sum()            # tp
            stats[t][1] += (pred * (1 - gt_hard)).sum()      # fp
            stats[t][2] += ((1 - pred) * gt_hard).sum()      # fn

    best_f1, best_thr = 0.0, 0.5
    for t, (tp, fp, fn) in stats.items():
        p  = tp / (tp + fp + 1e-6)
        r  = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    return float(best_f1), float(best_thr)


# ─── training ────────────────────────────────────────────────────────────────

def run_experiment(cfg, train_cfg):
    name  = cfg["name"]
    sigma = cfg["soft_sigma"]
    device = torch.device("cpu")

    ds_kw = dict(
        size             = train_cfg["img_size"],
        texture_strength = train_cfg["texture"],
        with_pairs       = False,
        antialias        = True,
        gaussian_noise   = 0.0,
        freq_mode        = "low",
        soft_label_sigma = sigma,
    )
    ds_train = SyntheticEdgeDataset(
        length = train_cfg["steps"] * train_cfg["batch"] * train_cfg["epochs"],
        seed   = 0,
        **ds_kw,
    )
    # Val dataset: always use same sigma (model trained on these labels)
    ds_val = SyntheticEdgeDataset(**{**ds_kw, "length": 200, "seed": 999})

    model     = UNet(in_ch=3, out_ch=1, base_ch=train_cfg["base_ch"])
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"],
                                  eta_min=train_cfg["lr"] * 0.01)

    best_ods = 0.0
    history  = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        t0  = time.time()
        ep_l = 0.0

        for step in range(train_cfg["steps"]):
            idx   = (epoch * train_cfg["steps"] + step) % len(ds_train)
            batch = ds_train.get_batch(train_cfg["batch"], start_idx=idx)
            imgs  = batch["image"].to(device)
            edges = batch["edge_mask"].to(device)   # soft if sigma>0

            logits = model(imgs)
            loss   = focal_bce(logits, edges,
                               alpha=train_cfg["focal_alpha"],
                               gamma=train_cfg["focal_gamma"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_l += loss.item()

        scheduler.step()

        # Quick monitoring: fixed-0.5 F1 on a small hard-GT val slice
        model.eval()
        quick_f1s = []
        for vi in range(0, 40, train_cfg["batch"]):
            vb = ds_val.get_batch(train_cfg["batch"], start_idx=vi)
            with torch.no_grad():
                vl = model(vb["image"].to(device))
            # Hard GT for monitoring
            gt_hard = (vb["edge_mask"] > 0.5).float().to(device)
            quick_f1s.append(f1_fixed(vl, gt_hard, thr=0.5))
        monitor_f1 = float(np.mean(quick_f1s))

        # ODS F1 (full val, sweep thresholds)
        ods, best_thr = ods_f1(model, ds_val, n_samples=80)

        history.append({
            "epoch": epoch,
            "loss":  ep_l / train_cfg["steps"],
            "f1_05": monitor_f1,
            "ods":   ods,
            "thr":   best_thr,
        })

        print(f"  [{name}] Epoch {epoch:3d}/{train_cfg['epochs']}  "
              f"loss={ep_l/train_cfg['steps']:.4f}  "
              f"f1@0.5={monitor_f1:.3f}  ODS={ods:.3f}(thr={best_thr:.2f})  "
              f"({time.time()-t0:.1f}s)")

        if ods > best_ods:
            best_ods = ods
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "ods": ods, "best_thr": best_thr,
                        "history": history, "cfg": cfg},
                       f"checkpoints/exp_{name}_best.pt")

    print(f"  [{name}] ✓ Best ODS F1 = {best_ods:.3f}\n")
    return history, model, best_ods


# ─── visualisation ───────────────────────────────────────────────────────────

def visualize(results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for cfg, history, model, best_ods in results:
        epochs  = [h["epoch"] for h in history]
        color   = cfg["color"]
        label   = cfg["label"].replace("\n", " | ")
        axes[0].plot(epochs, [h["ods"]   for h in history], color=color, label=label, lw=2)
        axes[1].plot(epochs, [h["f1_05"] for h in history], color=color, label=label, lw=2)
        axes[2].plot(epochs, [h["thr"]   for h in history], color=color, label=label, lw=2)

    axes[0].set_title("ODS F1 (optimal threshold, ↑)", fontweight="bold")
    axes[1].set_title("F1 @ fixed thr=0.5 (↑)", fontweight="bold")
    axes[2].set_title("Best threshold per epoch", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[2].set_ylim(0, 1)

    fig.suptitle("Antialias: hard label vs soft label (Gaussian σ=1.5)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig("viz_aa_soft_compare.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved viz_aa_soft_compare.png")

    # ── sample grid: 4 samples × [image | GT soft | pred hard | pred soft] ──
    common_ds_hard = SyntheticEdgeDataset(
        size=128, texture_strength=0.4, with_pairs=False, length=200, seed=77,
        antialias=True, gaussian_noise=0.0, freq_mode="low", soft_label_sigma=0.0,
    )
    common_ds_soft = SyntheticEdgeDataset(
        size=128, texture_strength=0.4, with_pairs=False, length=200, seed=77,
        antialias=True, gaussian_noise=0.0, freq_mode="low", soft_label_sigma=1.5,
    )

    n = 4
    fig, axes = plt.subplots(n, 5, figsize=(13, n * 2.8),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})

    titles = ["Input", "GT (hard)", "GT (soft σ=1.5)", "Pred hard-label", "Pred soft-label"]
    for j, t in enumerate(titles):
        color = "steelblue" if j == 3 else ("darkorange" if j == 4 else "black")
        axes[0, j].set_title(t, fontsize=8.5, fontweight="bold", color=color, pad=3)

    models_dict = {cfg["name"]: (model, cfg) for cfg, _, model, _ in results}

    def to_img(t):
        return ((np.array(t).transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)

    def overlay(img, mask, color=(255, 80, 80), thr=0.5):
        out = img.astype(float).copy()
        m   = np.array(mask).squeeze() > thr
        for c, v in enumerate(color):
            out[:, :, c] = np.where(m, 0.7 * v + 0.3 * out[:, :, c], out[:, :, c])
        return out.clip(0, 255).astype(np.uint8)

    for i in range(n):
        s_hard = common_ds_hard[i]
        s_soft = common_ds_soft[i]
        img    = to_img(torch.from_numpy(s_hard["image"]))

        gt_hard = s_hard["edge_mask"][0]
        gt_soft = s_soft["edge_mask"][0]

        axes[i, 0].imshow(img); axes[i, 0].axis("off")
        axes[i, 1].imshow((gt_hard * 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        axes[i, 1].axis("off")
        axes[i, 2].imshow((gt_soft * 255).astype(np.uint8), cmap="hot",  vmin=0, vmax=255)
        axes[i, 2].axis("off")

        inp = torch.from_numpy(s_hard["image"]).unsqueeze(0)

        for col_idx, mname in enumerate(["aa_hard", "aa_soft"]):
            model_m, cfg_m = models_dict[mname]
            model_m.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model_m(inp)).squeeze().numpy()

            # Load best threshold for this model
            ck   = torch.load(f"checkpoints/exp_{mname}_best.pt",
                              map_location="cpu", weights_only=False)
            thr  = ck.get("best_thr", 0.5)
            ods  = ck.get("ods", 0.0)

            ovl = overlay(img, prob, thr=thr)
            axes[i, 3 + col_idx].imshow(ovl); axes[i, 3 + col_idx].axis("off")

            pred_b = (prob > thr).astype(float)
            tp = (pred_b * (gt_hard > 0.5)).sum()
            fp = (pred_b * (gt_hard < 0.5)).sum()
            fn = ((1 - pred_b) * (gt_hard > 0.5)).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            axes[i, 3 + col_idx].set_xlabel(
                f"F1={f1:.3f}  thr={thr:.2f}", fontsize=7.5, labelpad=2,
                color=cfg_m["color"])

        axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=7.5, va="center", labelpad=4)

    fig.suptitle("Antialias soft-label ablation — sample comparison",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.savefig("viz_aa_soft_samples.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved viz_aa_soft_samples.png")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'实验':25s}  {'Best ODS F1':>12s}  {'Best Threshold':>14s}")
    print("-" * 60)
    for cfg, history, model, best_ods in results:
        ck  = torch.load(f"checkpoints/exp_{cfg['name']}_best.pt",
                         map_location="cpu", weights_only=False)
        thr = ck.get("best_thr", 0.5)
        print(f"{cfg['name']:25s}  {best_ods:>12.3f}  {thr:>14.2f}")
    print("=" * 60)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("checkpoints", exist_ok=True)
    results = []

    for cfg in CONFIGS:
        print("=" * 60)
        print(f"Experiment: {cfg['name']}  sigma={cfg['soft_sigma']}")
        print("=" * 60)
        history, model, best_ods = run_experiment(cfg, TRAIN_CFG)
        results.append((cfg, history, model, best_ods))

    visualize(results)


if __name__ == "__main__":
    main()
