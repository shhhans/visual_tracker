"""
Path1 消融实验：四组数据集配置对比
=====================================
  baseline    : 当前设置（低频纹理，无噪声，无抗锯齿）
  antialias   : 2× 超采样后下采样，消除多边形锯齿
  gaussian    : 高斯噪声 σ=0.05（像素归一化后）
  high_freq   : 高频纹理（短波长正弦波）

运行：python experiments_path1.py
输出：
  checkpoints/exp_baseline_best.pt
  checkpoints/exp_antialias_best.pt
  checkpoints/exp_gaussian_best.pt
  checkpoints/exp_highfreq_best.pt
  viz_ablation.png
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

EXPERIMENTS = [
    {
        "name":    "baseline",
        "label":   "Baseline\n(low-freq, no noise, no AA)",
        "color":   "steelblue",
        "ds_kwargs": dict(antialias=False, gaussian_noise=0.0, freq_mode="low"),
    },
    {
        "name":    "antialias",
        "label":   "Anti-aliasing\n(2× supersample)",
        "color":   "darkorange",
        "ds_kwargs": dict(antialias=True, gaussian_noise=0.0, freq_mode="low"),
    },
    {
        "name":    "gaussian",
        "label":   "Gaussian noise\n(σ=0.05)",
        "color":   "mediumseagreen",
        "ds_kwargs": dict(antialias=False, gaussian_noise=0.05, freq_mode="low"),
    },
    {
        "name":    "highfreq",
        "label":   "High-freq texture\n(λ ∈ [W/16, W/4])",
        "color":   "orchid",
        "ds_kwargs": dict(antialias=False, gaussian_noise=0.0, freq_mode="high"),
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


# ─── loss / metrics ─────────────────────────────────────────────────────────

def focal_bce(logits, targets, alpha=0.75, gamma=2.0):
    bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p    = torch.sigmoid(logits)
    p_t  = p * targets + (1-p)*(1-targets)
    a_t  = alpha * targets + (1-alpha)*(1-targets)
    return (a_t * (1-p_t)**gamma * bce).mean()

@torch.no_grad()
def f1_score(logits, targets, thr=0.5):
    pred = (torch.sigmoid(logits) > thr).float()
    tp   = (pred * targets).sum()
    fp   = (pred * (1-targets)).sum()
    fn   = ((1-pred) * targets).sum()
    p    = tp / (tp+fp+1e-6); r = tp / (tp+fn+1e-6)
    return float(2*p*r/(p+r+1e-6)), float(p), float(r)


# ─── single experiment ──────────────────────────────────────────────────────

def run_experiment(cfg, train_cfg):
    name   = cfg["name"]
    device = torch.device("cpu")

    ds_kw = dict(
        size             = train_cfg["img_size"],
        texture_strength = train_cfg["texture"],
        with_pairs       = False,
        seed             = 0,
        **cfg["ds_kwargs"],
    )
    ds_train = SyntheticEdgeDataset(length=train_cfg["steps"]*train_cfg["batch"]*train_cfg["epochs"], **ds_kw)
    ds_val   = SyntheticEdgeDataset(**{**ds_kw, "length": 200, "seed": 999})

    model     = UNet(in_ch=3, out_ch=1, base_ch=train_cfg["base_ch"])
    optimizer = Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"],
                                  eta_min=train_cfg["lr"]*0.01)

    best_f1  = 0.0
    history  = []            # list of {epoch, loss, f1, p, r}

    for epoch in range(1, train_cfg["epochs"]+1):
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

        # validation
        model.eval()
        vf1s, vps, vrs = [], [], []
        for vi in range(0, 40, train_cfg["batch"]):
            vb = ds_val.get_batch(train_cfg["batch"], start_idx=vi)
            with torch.no_grad():
                vl = model(vb["image"].to(device))
            f1, p, r = f1_score(vl, vb["edge_mask"].to(device))
            vf1s.append(f1); vps.append(p); vrs.append(r)

        f1 = float(np.mean(vf1s))
        p  = float(np.mean(vps))
        r  = float(np.mean(vrs))
        history.append({"epoch": epoch, "loss": ep_l/train_cfg["steps"], "f1": f1, "p": p, "r": r})

        print(f"  [{name}] Epoch {epoch:3d}/{train_cfg['epochs']}  "
              f"loss={ep_l/train_cfg['steps']:.4f}  "
              f"f1={f1:.3f}  p={p:.3f}  r={r:.3f}  ({time.time()-t0:.1f}s)")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"epoch": epoch, "model": model.state_dict(), "f1": f1,
                        "history": history, "cfg": cfg},
                       f"checkpoints/exp_{name}_best.pt")

    print(f"  [{name}] ✓ Best F1 = {best_f1:.3f}\n")
    return history, model, best_f1


# ─── visualisation ──────────────────────────────────────────────────────────

def visualize_ablation(results):
    """
    results: list of (cfg, history, model, best_f1)
    Generates viz_ablation.png  (training curves + per-sample grid)
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ── Figure 1: training curves ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for cfg, history, model, best_f1 in results:
        epochs = [h["epoch"] for h in history]
        color  = cfg["color"]
        label  = cfg["label"].replace("\n", " | ")
        axes[0].plot(epochs, [h["f1"]   for h in history], color=color, label=label, lw=2)
        axes[1].plot(epochs, [h["loss"] for h in history], color=color, label=label, lw=2)

    axes[0].set_title("Validation F1 (↑)", fontsize=10, fontweight="bold")
    axes[1].set_title("Training Loss (↓)", fontsize=10, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    fig.suptitle("Path1 消融实验 — 训练曲线对比", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig("viz_ablation_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved viz_ablation_curves.png")

    # ── Figure 2: sample grid ────────────────────────────────────────────────
    # rows = 4 experiments, cols = 5 samples each showing [image | gt | pred | overlay]
    n_samples = 4
    n_cols    = 1 + 3 * len(results)    # sample-index col + 3 cols per exp

    # Build a shared validation dataset for fair visual comparison
    # Use baseline settings for the images (same visual input across all)
    common_ds = SyntheticEdgeDataset(
        size=128, texture_strength=0.4, with_pairs=False, length=200, seed=77,
        antialias=False, gaussian_noise=0.0, freq_mode="low",
    )

    def to_img(t):
        return ((np.array(t).transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8)

    def overlay_edge(img, mask, color=(255,80,80), alpha=0.7):
        out = img.astype(float).copy(); m = np.array(mask).squeeze() > 0.5
        for c, v in enumerate(color):
            out[:,:,c] = np.where(m, alpha*v+(1-alpha)*out[:,:,c], out[:,:,c])
        return out.clip(0,255).astype(np.uint8)

    fig, axes = plt.subplots(n_samples, 2 + len(results),
                             figsize=((2+len(results))*2.5, n_samples*2.5),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})

    # Column headers
    axes[0, 0].set_title("Input", fontsize=8, fontweight="bold", pad=3)
    axes[0, 1].set_title("GT edge", fontsize=8, fontweight="bold", pad=3)
    for j, (cfg, _, _, _) in enumerate(results):
        short = cfg["label"].split("\n")[0]
        axes[0, 2+j].set_title(short, fontsize=8, fontweight="bold",
                                color=cfg["color"], pad=3)

    for i in range(n_samples):
        s      = common_ds[i]
        img_t  = to_img(torch.from_numpy(s["image"]))
        gt_np  = s["edge_mask"][0]

        # GT edge as coloured heatmap (grayscale)
        gt_vis = (gt_np * 255).astype(np.uint8)
        gt_rgb = np.stack([gt_vis]*3, axis=-1)

        axes[i, 0].imshow(img_t);  axes[i, 0].axis("off")
        axes[i, 1].imshow(gt_rgb); axes[i, 1].axis("off")

        for j, (cfg, history, model, best_f1) in enumerate(results):
            # Run this experiment's model on the SAME input image,
            # but also render the same sample with this exp's dataset config
            # to show how the input looks different
            exp_ds = SyntheticEdgeDataset(
                size=128, texture_strength=0.4, with_pairs=False, length=200, seed=77,
                **cfg["ds_kwargs"]
            )
            s_exp  = exp_ds[i]
            img_exp = to_img(torch.from_numpy(s_exp["image"]))

            model.eval()
            inp = torch.from_numpy(s_exp["image"]).unsqueeze(0)
            with torch.no_grad():
                logit = model(inp)
            prob = torch.sigmoid(logit).squeeze().numpy()

            # compute F1 for this sample
            pred_b = (prob > 0.5).astype(float)
            gt_b   = s_exp["edge_mask"][0]
            tp = (pred_b*gt_b).sum(); fp = (pred_b*(1-gt_b)).sum(); fn = ((1-pred_b)*gt_b).sum()
            f1 = float(2*tp / (2*tp+fp+fn+1e-6))

            ovl = overlay_edge(img_exp, prob > 0.5, color=(255,80,80))
            axes[i, 2+j].imshow(ovl); axes[i, 2+j].axis("off")
            axes[i, 2+j].set_xlabel(f"F1={f1:.3f}", fontsize=7.5, labelpad=2,
                                     color=cfg["color"])

        # Row label
        axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=7.5, rotation=90,
                               labelpad=4, va="center")

    fig.suptitle("Path1 消融实验 — 各配置样本可视化\n(红色叠加 = 预测边缘，阈值 0.5)",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.savefig("viz_ablation_samples.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved viz_ablation_samples.png")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'实验':20s}  {'Best F1':>8s}  {'最终 F1':>8s}  {'最终 Loss':>10s}")
    print("-"*60)
    for cfg, history, model, best_f1 in results:
        last = history[-1]
        print(f"{cfg['name']:20s}  {best_f1:8.3f}  {last['f1']:8.3f}  {last['loss']:10.4f}")
    print("="*60)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("checkpoints", exist_ok=True)
    results = []

    for cfg in EXPERIMENTS:
        print("=" * 60)
        print(f"Experiment: {cfg['name']}  —  {cfg['label'].replace(chr(10),' | ')}")
        print("=" * 60)
        history, model, best_f1 = run_experiment(cfg, TRAIN_CFG)
        results.append((cfg, history, model, best_f1))

    visualize_ablation(results)


if __name__ == "__main__":
    main()
