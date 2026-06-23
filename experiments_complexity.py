"""
复杂度消融：baseline vs complex 数据集
========================================
baseline : 单方块、低频纹理、无噪声（原始设置）
complex  : 1-3 个多边形/圆形 + 运动模糊 + 局部光照 + 投影阴影
           + Gaussian 软标签 (σ=1.5) + antialias

两组都训练 Path1 (UNet in=3 out=1 base_ch=16)。
评估用 ODS F1（验证集上扫阈值，GT 二值化 >0.5）。

输出：
  checkpoints/exp_cplx_baseline_best.pt
  checkpoints/exp_cplx_complex_best.pt
  viz_complexity.png
"""

import os, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from data.synthetic import SyntheticEdgeDataset
from models.unet import UNet


# ─── 实验配置 ────────────────────────────────────────────────────────────────

CONFIGS = [
    {
        "name":  "baseline",
        "label": "Baseline\n(single quad, no complexity)",
        "color": "steelblue",
        "ds_kwargs": dict(
            antialias        = False,
            gaussian_noise   = 0.0,
            freq_mode        = "low",
            soft_label_sigma = 0.0,
            n_objects_range  = (1, 1),
            shape_types      = ["quad"],
            motion_blur      = False,
            local_lighting   = False,
            cast_shadows     = False,
        ),
    },
    {
        "name":  "complex",
        "label": "Complex\n(1-3 shapes, blur, lighting, shadows)",
        "color": "darkorange",
        "ds_kwargs": dict(
            antialias        = True,
            gaussian_noise   = 0.02,
            freq_mode        = "low",
            soft_label_sigma = 1.5,
            n_objects_range  = (1, 3),
            shape_types      = ["quad", "triangle", "pentagon", "hexagon", "circle"],
            motion_blur      = True,
            local_lighting   = True,
            cast_shadows     = True,
        ),
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

ODS_THRESHOLDS = np.linspace(0.1, 0.9, 17).tolist()


# ─── loss / metrics ──────────────────────────────────────────────────────────

def focal_bce(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p   = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


@torch.no_grad()
def ods_f1(model, ds_val, n_samples=80):
    """F1 at the threshold that maximises it over the val set (binary GT)."""
    stats = {t: np.zeros(3) for t in ODS_THRESHOLDS}
    for i in range(n_samples):
        s    = ds_val[i]
        inp  = torch.from_numpy(s["image"]).unsqueeze(0)
        prob = torch.sigmoid(model(inp)).squeeze().numpy()
        gt   = (s["edge_mask"][0] > 0.5).astype(np.float32)
        for t in ODS_THRESHOLDS:
            pred = (prob > t).astype(np.float32)
            stats[t][0] += (pred * gt).sum()
            stats[t][1] += (pred * (1 - gt)).sum()
            stats[t][2] += ((1 - pred) * gt).sum()

    best_f1, best_thr = 0.0, 0.5
    for t, (tp, fp, fn) in stats.items():
        p  = tp / (tp + fp + 1e-6)
        r  = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_f1), float(best_thr)


# ─── single experiment ───────────────────────────────────────────────────────

def run_experiment(cfg, train_cfg):
    name   = cfg["name"]
    device = torch.device("cpu")

    base_kw = dict(
        size             = train_cfg["img_size"],
        texture_strength = train_cfg["texture"],
        with_pairs       = False,
        seed             = 0,
        **cfg["ds_kwargs"],
    )
    ds_train = SyntheticEdgeDataset(
        length = train_cfg["steps"] * train_cfg["batch"] * train_cfg["epochs"],
        **base_kw,
    )
    ds_val = SyntheticEdgeDataset(**{**base_kw, "length": 200, "seed": 999})

    model     = UNet(in_ch=3, out_ch=1, base_ch=train_cfg["base_ch"])
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"],
                                  eta_min=train_cfg["lr"] * 0.01)
    best_ods = 0.0
    history  = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        t0   = time.time()
        ep_l = 0.0

        for step in range(train_cfg["steps"]):
            idx   = (epoch * train_cfg["steps"] + step) % len(ds_train)
            batch = ds_train.get_batch(train_cfg["batch"], start_idx=idx)
            imgs  = batch["image"].to(device)
            edges = batch["edge_mask"].to(device)

            logits = model(imgs)
            loss   = focal_bce(logits, edges,
                               alpha=train_cfg["focal_alpha"],
                               gamma=train_cfg["focal_gamma"])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_l += loss.item()

        scheduler.step()

        model.eval()
        ods, best_thr = ods_f1(model, ds_val, n_samples=80)
        avg_loss = ep_l / train_cfg["steps"]
        history.append({"epoch": epoch, "loss": avg_loss,
                        "ods": ods, "thr": best_thr})

        print(f"  [{name}] Epoch {epoch:3d}/{train_cfg['epochs']}  "
              f"loss={avg_loss:.4f}  ODS={ods:.3f}(thr={best_thr:.2f})  "
              f"({time.time()-t0:.1f}s)")

        if ods > best_ods:
            best_ods = ods
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "ods": ods, "best_thr": best_thr,
                        "history": history, "cfg": cfg},
                       f"checkpoints/exp_cplx_{name}_best.pt")

    print(f"  [{name}] ✓ Best ODS F1 = {best_ods:.3f}\n")
    return history, model, best_ods


# ─── visualisation ───────────────────────────────────────────────────────────

def visualize(results):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for cfg, history, model, best_ods in results:
        epochs = [h["epoch"] for h in history]
        color  = cfg["color"]
        label  = cfg["label"].replace("\n", " | ")
        axes[0].plot(epochs, [h["ods"]  for h in history], color=color, label=label, lw=2)
        axes[1].plot(epochs, [h["loss"] for h in history], color=color, label=label, lw=2)
        axes[2].plot(epochs, [h["thr"]  for h in history], color=color, label=label, lw=2)

    axes[0].set_title("ODS F1 (↑)", fontweight="bold")
    axes[1].set_title("Training Loss (↓)", fontweight="bold")
    axes[2].set_title("Best threshold per epoch", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[2].set_ylim(0, 1)

    fig.suptitle("Complexity ablation: baseline vs complex dataset",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig("viz_complexity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved viz_complexity.png")

    # Sample grid: 4 rows × [input | GT | pred-baseline | pred-complex]
    n = 4
    fig, axes = plt.subplots(n, 4, figsize=(11, n * 2.8),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})

    models_dict = {cfg["name"]: (model, cfg) for cfg, _, model, _ in results}

    # Use the complex val dataset for the visual comparison
    vis_ds = SyntheticEdgeDataset(
        size=128, texture_strength=0.4, with_pairs=False, length=200, seed=77,
        n_objects_range=(1, 3),
        shape_types=["quad", "triangle", "pentagon", "hexagon", "circle"],
        motion_blur=True, local_lighting=True, cast_shadows=True,
        antialias=True, soft_label_sigma=0.0,   # hard GT for display
    )

    col_titles = ["Input (complex)", "GT edge",
                  "Pred: Baseline model", "Pred: Complex model"]
    colors_t   = ["black", "black", "steelblue", "darkorange"]
    for j, (t, c) in enumerate(zip(col_titles, colors_t)):
        axes[0, j].set_title(t, fontsize=8, fontweight="bold", color=c, pad=3)

    def to_img(t):
        return ((np.array(t).transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)

    def overlay(img, prob, thr, color=(255, 80, 80)):
        out = img.astype(float).copy()
        m   = prob > thr
        for c, v in enumerate(color):
            out[:, :, c] = np.where(m, 0.7*v + 0.3*out[:, :, c], out[:, :, c])
        return out.clip(0, 255).astype(np.uint8)

    for i in range(n):
        s      = vis_ds[i]
        img    = to_img(torch.from_numpy(s["image"]))
        gt_hard = (s["edge_mask"][0] > 0.5)
        gt_rgb  = np.stack([gt_hard.astype(np.uint8) * 255] * 3, axis=-1)

        axes[i, 0].imshow(img);    axes[i, 0].axis("off")
        axes[i, 1].imshow(gt_rgb); axes[i, 1].axis("off")
        axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=7.5, va="center", labelpad=4)

        inp = torch.from_numpy(s["image"]).unsqueeze(0)
        for col_j, mname in enumerate(["baseline", "complex"]):
            m, cfg_m = models_dict[mname]
            m.eval()
            with torch.no_grad():
                prob = torch.sigmoid(m(inp)).squeeze().numpy()

            ck  = torch.load(f"checkpoints/exp_cplx_{mname}_best.pt",
                             map_location="cpu", weights_only=False)
            thr = ck.get("best_thr", 0.5)

            pred_b = (prob > thr).astype(float)
            tp = (pred_b * gt_hard).sum()
            fp = (pred_b * (~gt_hard)).sum()
            fn = ((1 - pred_b) * gt_hard).sum()
            f1 = 2*tp / (2*tp + fp + fn + 1e-6)

            ovl = overlay(img, prob, thr, color=(255, 80, 80))
            axes[i, 2 + col_j].imshow(ovl); axes[i, 2 + col_j].axis("off")
            axes[i, 2 + col_j].set_xlabel(
                f"F1={f1:.3f}  thr={thr:.2f}", fontsize=7.5, labelpad=2,
                color=cfg_m["color"])

    fig.suptitle("Predictions on complex val scenes\n"
                 "(red = predicted edge at optimal threshold)",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.savefig("viz_complexity_samples.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved viz_complexity_samples.png")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'Config':20s}  {'Best ODS F1':>12s}  {'Best Threshold':>14s}")
    print("-" * 60)
    for cfg, history, model, best_ods in results:
        ck  = torch.load(f"checkpoints/exp_cplx_{cfg['name']}_best.pt",
                         map_location="cpu", weights_only=False)
        thr = ck.get("best_thr", 0.5)
        print(f"{cfg['name']:20s}  {best_ods:>12.3f}  {thr:>14.2f}")
    print("=" * 60)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("checkpoints", exist_ok=True)
    results = []
    for cfg in CONFIGS:
        print("=" * 60)
        print(f"Experiment: {cfg['name']}")
        print("=" * 60)
        history, model, best_ods = run_experiment(cfg, TRAIN_CFG)
        results.append((cfg, history, model, best_ods))
    visualize(results)


if __name__ == "__main__":
    main()
