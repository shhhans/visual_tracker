"""
串联网络（Arch A）在复杂数据上的训练
=======================================
Path1  : 已在复杂数据上训练好（exp_cplx_complex_best.pt），权重冻结
FlowNet: 从头训练，输入 frame_t(3) + frame_t+1(3) + hm_t(1) = 7ch

数据集：复杂配置（1-3 个多形状物体、运动模糊、局部光照、投影阴影）
         with_pairs=True → 返回 frame_t, frame_t+1, gt_flow

Flow Loss：边缘软标签加权 L1
  L = mean( edge_weight * ||pred_flow - gt_flow||_1 )
  edge_weight = edge_mask（软标签，热力图形式，中心=1，衰减到0）
  → 网络只在边界区域受到强烈监督，背景处梯度弱

评估：
  - EPE (endpoint error) at boundary pixels (edge_mask > 0.3)
  - Path1 hm_t 质量（F1 与 ODS threshold）

输出：
  checkpoints/exp_serial_complex_best.pt
  viz_serial_complex.png
  viz_serial_complex_samples.png
"""

import os, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from data.synthetic import SyntheticEdgeDataset
from models.arch_compare import ArchA

# ─── config ──────────────────────────────────────────────────────────────────

PATH1_CKPT = "checkpoints/exp_cplx_complex_best.pt"

DS_KWARGS = dict(
    size             = 128,
    texture_strength = 0.4,
    with_pairs       = True,
    n_objects_range  = (1, 3),
    shape_types      = ["quad", "triangle", "pentagon", "hexagon", "circle"],
    motion_blur      = True,
    local_lighting   = True,
    cast_shadows     = True,
    antialias        = True,
    soft_label_sigma = 1.5,
)

TRAIN_CFG = dict(
    epochs      = 40,
    steps       = 60,
    batch       = 6,
    lr          = 3e-4,
    base_ch     = 16,
    flow_weight = 1.0,
)

ODS_THR = np.linspace(0.1, 0.9, 17).tolist()


# ─── losses / metrics ────────────────────────────────────────────────────────

def weighted_flow_loss(pred_flow, gt_flow, edge_weight):
    """
    Soft-masked L1 flow loss.
    edge_weight : (B, 1, H, W) float in [0,1] — soft edge heatmap
    Only boundary pixels (high weight) contribute strongly.
    """
    err = (pred_flow - gt_flow).abs().sum(dim=1, keepdim=True)   # (B,1,H,W)
    return (edge_weight * err).mean()


@torch.no_grad()
def eval_epe(pred_flow, gt_flow, edge_mask, thr=0.3):
    """Mean EPE at pixels where edge_mask > thr."""
    mask = edge_mask.squeeze(1) > thr           # (B, H, W)
    if not mask.any():
        return float("nan")
    err  = (pred_flow - gt_flow).pow(2).sum(1).sqrt()  # (B, H, W)
    return float(err[mask].mean())


@torch.no_grad()
def eval_path1_ods(hm_t, gt_edge_hard, n_thr=17):
    """ODS F1 of Path1's heatmap against hard binary GT."""
    stats = {t: np.zeros(3) for t in ODS_THR}
    B = hm_t.shape[0]
    for b in range(B):
        prob = hm_t[b, 0].numpy()
        gt   = (gt_edge_hard[b, 0] > 0.5).numpy().astype(float)
        for t in ODS_THR:
            pred = (prob > t).astype(float)
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


# ─── training ────────────────────────────────────────────────────────────────

def run():
    device = torch.device("cpu")

    ds_train = SyntheticEdgeDataset(
        length = TRAIN_CFG["steps"] * TRAIN_CFG["batch"] * TRAIN_CFG["epochs"],
        seed   = 0,
        **DS_KWARGS,
    )
    ds_val = SyntheticEdgeDataset(**{**DS_KWARGS, "length": 200, "seed": 999})

    model     = ArchA(PATH1_CKPT, base_ch=TRAIN_CFG["base_ch"]).to(device)
    optimizer = Adam(model.trainable_params(), lr=TRAIN_CFG["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CFG["epochs"],
                                  eta_min=TRAIN_CFG["lr"] * 0.01)

    best_epe = float("inf")
    history  = []

    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        model.flow_net.train()
        t0   = time.time()
        ep_l = 0.0

        for step in range(TRAIN_CFG["steps"]):
            idx   = (epoch * TRAIN_CFG["steps"] + step) % len(ds_train)
            batch = ds_train.get_batch(TRAIN_CFG["batch"], start_idx=idx)

            frame_t  = batch["image"].to(device)
            frame_t1 = batch["image_t1"].to(device)
            gt_flow  = batch["flow"].to(device)
            edge_w   = batch["edge_mask"].to(device)   # soft heatmap

            hm_t, pred_flow = model(frame_t, frame_t1)

            loss = weighted_flow_loss(pred_flow, gt_flow, edge_w)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_l += loss.item()

        scheduler.step()

        # ── validation ───────────────────────────────────────────────────────
        model.flow_net.eval()
        val_epes, val_hm_f1s = [], []

        for vi in range(0, 48, TRAIN_CFG["batch"]):
            vb = ds_val.get_batch(TRAIN_CFG["batch"], start_idx=vi)
            ft  = vb["image"].to(device)
            ft1 = vb["image_t1"].to(device)
            gf  = vb["flow"].to(device)
            em  = vb["edge_mask"].to(device)

            with torch.no_grad():
                hm_t, pf = model(ft, ft1)

            epe = eval_epe(pf, gf, em, thr=0.3)
            if not math.isnan(epe):
                val_epes.append(epe)

            # Path1 hm quality (use hard GT = soft_label_sigma=0 version)
            gt_hard = (em > 0.5).float()
            hm_f1, _ = eval_path1_ods(hm_t.cpu(), gt_hard.cpu())
            val_hm_f1s.append(hm_f1)

        mean_epe   = float(np.mean(val_epes))   if val_epes   else float("nan")
        mean_hm_f1 = float(np.mean(val_hm_f1s)) if val_hm_f1s else 0.0
        avg_loss   = ep_l / TRAIN_CFG["steps"]

        history.append({"epoch": epoch, "loss": avg_loss,
                        "epe": mean_epe, "hm_f1": mean_hm_f1})

        print(f"  Epoch {epoch:3d}/{TRAIN_CFG['epochs']}  "
              f"loss={avg_loss:.4f}  EPE={mean_epe:.3f}px  "
              f"hm_F1={mean_hm_f1:.3f}  ({time.time()-t0:.1f}s)")

        if mean_epe < best_epe:
            best_epe = mean_epe
            torch.save({
                "epoch": epoch, "flow_net": model.flow_net.state_dict(),
                "epe": mean_epe, "hm_f1": mean_hm_f1, "history": history,
                "path1_ckpt": PATH1_CKPT,
            }, "checkpoints/exp_serial_complex_best.pt")

    print(f"\n  ✓ Best EPE = {best_epe:.3f} px")
    return history, model, best_epe


# ─── visualisation ───────────────────────────────────────────────────────────

def visualize(history, model):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    epochs = [h["epoch"] for h in history]

    # ── Training curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(epochs, [h["epe"]    for h in history], color="darkorange", lw=2)
    axes[1].plot(epochs, [h["hm_f1"]  for h in history], color="steelblue",  lw=2)
    axes[2].plot(epochs, [h["loss"]   for h in history], color="mediumseagreen", lw=2)

    axes[0].set_title("Flow EPE at edges (↓ better)", fontweight="bold")
    axes[1].set_title("Path1 heatmap ODS-F1 (frozen, for ref)", fontweight="bold")
    axes[2].set_title("Training Loss (↓)", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)

    fig.suptitle("Serial (Arch A) on complex dataset\n"
                 "Path1 frozen (exp_cplx_complex), FlowNet trained from scratch",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig("viz_serial_complex.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved viz_serial_complex.png")

    # ── Sample grid ───────────────────────────────────────────────────────────
    vis_ds = SyntheticEdgeDataset(**{**DS_KWARGS,
                                     "length": 200, "seed": 77,
                                     "soft_label_sigma": 0.0})   # hard GT for display

    n = 5
    fig, axes = plt.subplots(n, 6, figsize=(15, n * 2.8),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})

    titles = ["Frame t", "Frame t+1", "Path1 hm_t",
              "GT edge+flow", "Pred flow", "Flow EPE (px)"]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=8, fontweight="bold", pad=3)

    def to_img(arr):
        return ((arr.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)

    def overlay_flow(img, flow, mask, stride=10, scale=2.5, color="deepskyblue"):
        ax_tmp = None
        return img, flow, mask, stride, scale, color  # placeholder, drawn via quiver

    model.flow_net.eval()

    for i in range(n):
        s       = vis_ds[i]
        ft      = torch.from_numpy(s["image"]).unsqueeze(0)
        ft1     = torch.from_numpy(s["image_t1"]).unsqueeze(0)
        gt_flow = torch.from_numpy(s["flow"])           # (2,H,W)
        gt_edge = torch.from_numpy(s["edge_mask"])      # (1,H,W)

        with torch.no_grad():
            hm_t, pred_flow = model(ft, ft1)

        hm_np   = hm_t[0, 0].numpy()
        pf_np   = pred_flow[0].numpy()
        gf_np   = gt_flow.numpy()
        ge_np   = gt_edge[0].numpy()
        img_t   = to_img(s["image"])
        img_t1  = to_img(s["image_t1"])

        # EPE map
        epe_map = np.sqrt(((pf_np - gf_np) ** 2).sum(0))
        mask_b  = ge_np > 0.5
        epe_masked = np.where(mask_b, epe_map, np.nan)
        mean_epe = float(epe_map[mask_b].mean()) if mask_b.any() else 0.0

        # Col 0: frame_t
        axes[i, 0].imshow(img_t); axes[i, 0].axis("off")

        # Col 1: frame_t+1
        axes[i, 1].imshow(img_t1); axes[i, 1].axis("off")

        # Col 2: Path1 heatmap
        axes[i, 2].imshow(hm_np, cmap="hot", vmin=0, vmax=1)
        axes[i, 2].axis("off")

        # Col 3: GT edge + GT flow arrows
        gt_ovl = img_t.copy().astype(float)
        gt_ovl[ge_np > 0.5] = [80, 200, 80]
        axes[i, 3].imshow(gt_ovl.clip(0, 255).astype(np.uint8))
        axes[i, 3].axis("off")
        H, W = ge_np.shape
        stride = 10
        ys  = np.arange(stride // 2, H, stride)
        xs  = np.arange(stride // 2, W, stride)
        XX, YY = np.meshgrid(xs, ys)
        m_q = mask_b[YY, XX]
        if m_q.any():
            axes[i, 3].quiver(XX[m_q], YY[m_q],
                              gf_np[0, YY[m_q], XX[m_q]],
                              -gf_np[1, YY[m_q], XX[m_q]],
                              angles="xy", scale_units="xy", scale=1/2.5,
                              width=0.004, color="lime", alpha=0.85)

        # Col 4: predicted flow arrows
        axes[i, 4].imshow(img_t)
        axes[i, 4].axis("off")
        if m_q.any():
            axes[i, 4].quiver(XX[m_q], YY[m_q],
                              pf_np[0, YY[m_q], XX[m_q]],
                              -pf_np[1, YY[m_q], XX[m_q]],
                              angles="xy", scale_units="xy", scale=1/2.5,
                              width=0.004, color="deepskyblue", alpha=0.85)

        # Col 5: EPE heatmap
        im = axes[i, 5].imshow(epe_masked, cmap="plasma", vmin=0, vmax=8)
        axes[i, 5].axis("off")
        axes[i, 5].set_xlabel(f"EPE={mean_epe:.2f}px", fontsize=7.5, labelpad=2)

        axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=7.5, va="center", labelpad=4)

    # Colorbar for EPE
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap="plasma", norm=Normalize(0, 8))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="EPE (px)")

    fig.suptitle("Serial (Arch A) on complex data\n"
                 "green arrows=GT flow  |  blue arrows=pred flow  |  col3=hm_t heatmap",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.savefig("viz_serial_complex_samples.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved viz_serial_complex_samples.png")

    # Summary
    best = min(history, key=lambda h: h["epe"])
    print("\n" + "=" * 55)
    print(f"Best EPE      : {best['epe']:.3f} px  (epoch {best['epoch']})")
    print(f"Path1 hm ODS  : {best['hm_f1']:.3f}  (frozen reference)")
    print(f"Final EPE     : {history[-1]['epe']:.3f} px")
    print("=" * 55)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("checkpoints", exist_ok=True)
    history, model, best_epe = run()
    visualize(history, model)


if __name__ == "__main__":
    main()
