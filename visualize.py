"""
可视化 Path1 / Path2 的 GT 与预测对比。
运行：python visualize.py
输出：viz_path1.png  viz_path2.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from data.synthetic import SyntheticEdgeDataset
from models.unet import UNet


# ── helpers ────────────────────────────────────────────────────────────────

def to_img(t):
    """(3,H,W) float [-1,1] → (H,W,3) uint8"""
    return ((t.numpy().transpose(1,2,0) + 1) * 127.5).clip(0,255).astype(np.uint8)

def to_heatmap(t, cmap="hot"):
    """(1,H,W) or (H,W) float → (H,W,3) uint8 coloured heatmap"""
    arr = t.squeeze().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    rgba = plt.get_cmap(cmap)(arr)
    return (rgba[:,:,:3] * 255).astype(np.uint8)

def overlay_edge(img_rgb, mask, color=(255,80,80), alpha=0.7):
    """Overlay binary edge mask in colour on top of image."""
    out = img_rgb.astype(float).copy()
    m   = mask.squeeze().numpy() > 0.5
    for c, v in enumerate(color):
        out[:,:,c] = np.where(m, alpha*v + (1-alpha)*out[:,:,c], out[:,:,c])
    return out.clip(0,255).astype(np.uint8)

def draw_flow_arrows(ax, flow, mask, stride=8, scale=3.0, color="deepskyblue"):
    """Draw (dx,dy) arrows on a matplotlib axis, sampled every `stride` pixels."""
    H, W  = mask.shape
    ys    = np.arange(stride//2, H, stride)
    xs    = np.arange(stride//2, W, stride)
    XX, YY = np.meshgrid(xs, ys)
    m  = mask[YY, XX] > 0.5
    if not m.any():
        return
    u  = flow[0, YY, XX]
    v  = flow[1, YY, XX]
    ax.quiver(XX[m], YY[m], u[m], -v[m],          # flip v for image coords
              angles="xy", scale_units="xy",
              scale=1/scale, width=0.004,
              color=color, alpha=0.85)


# ── Path 1 ─────────────────────────────────────────────────────────────────

def visualize_path1(n_samples=6, ckpt="checkpoints/path1_best.pt", out="viz_path1.png"):
    model = UNet(in_ch=3, out_ch=1, base_ch=16)
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_data["model"])
    model.eval()

    ds = SyntheticEdgeDataset(size=128, texture_strength=0.4,
                               with_pairs=False, length=200, seed=77)

    # columns: image | GT edge | predicted heatmap | overlay
    ncols = 4
    fig, axes = plt.subplots(n_samples, ncols,
                             figsize=(ncols*2.8, n_samples*2.8),
                             gridspec_kw={"wspace":0.05, "hspace":0.12})

    col_titles = ["Input image", "GT edge mask", "Predicted heatmap", "Prediction overlay"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=9, fontweight="bold", pad=4)

    for i in range(n_samples):
        s       = ds[i]
        img_t   = torch.from_numpy(s["image"]).unsqueeze(0)
        gt_edge = torch.from_numpy(s["edge_mask"])   # (1,H,W)

        with torch.no_grad():
            logit = model(img_t)                     # (1,1,H,W)
        pred_prob = torch.sigmoid(logit[0])          # (1,H,W)

        img_rgb   = to_img(torch.from_numpy(s["image"]))
        gt_vis    = to_heatmap(gt_edge, cmap="gray")
        pred_vis  = to_heatmap(pred_prob, cmap="hot")
        ovl_vis   = overlay_edge(img_rgb, pred_prob > 0.5)

        for j, vis in enumerate([img_rgb, gt_vis, pred_vis, ovl_vis]):
            ax = axes[i, j]
            ax.imshow(vis)
            ax.axis("off")
            # F1 annotation on overlay column
            if j == 3:
                pred_b = (pred_prob.squeeze().numpy() > 0.5).astype(float)
                gt_b   = gt_edge.squeeze().numpy()
                tp = (pred_b * gt_b).sum()
                fp = (pred_b * (1-gt_b)).sum()
                fn = ((1-pred_b) * gt_b).sum()
                f1 = 2*tp / (2*tp + fp + fn + 1e-6)
                ax.set_xlabel(f"F1={f1:.3f}", fontsize=7.5, labelpad=2)

    fig.suptitle("Path 1 — Single-frame Edge Detection\n"
                 "(red overlay = predicted edge, threshold 0.5)",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Path 2 ─────────────────────────────────────────────────────────────────

def visualize_path2(n_samples=5, ckpt="checkpoints/path2_best.pt", out="viz_path2.png"):
    model = UNet(in_ch=6, out_ch=3, base_ch=16)
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_data["model"])
    model.eval()

    ds = SyntheticEdgeDataset(size=128, texture_strength=0.4,
                               with_pairs=True, max_shift=8.0, max_rot=10.0,
                               length=200, seed=77)

    # columns: frame_t | frame_t+1 | GT edge+flow | Pred edge | Pred flow | flow error
    ncols = 6
    fig, axes = plt.subplots(n_samples, ncols,
                             figsize=(ncols*2.8, n_samples*2.8),
                             gridspec_kw={"wspace":0.05, "hspace":0.15})

    col_titles = ["Frame t", "Frame t+1",
                  "GT edge + flow", "Pred edge mask",
                  "Pred flow", "Flow error (px)"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=8.5, fontweight="bold", pad=4)

    for i in range(n_samples):
        s      = ds[i]
        x      = torch.cat([torch.from_numpy(s["image"]),
                             torch.from_numpy(s["image_t1"])], dim=0).unsqueeze(0)
        gt_edge = torch.from_numpy(s["edge_mask"])   # (1,H,W)
        gt_flow = torch.from_numpy(s["flow"])         # (2,H,W)

        with torch.no_grad():
            pred = model(x)[0]                        # (3,H,W)
        pred_logit = pred[:1]                         # (1,H,W)
        pred_flow  = pred[1:]                         # (2,H,W)
        pred_prob  = torch.sigmoid(pred_logit)

        img_t   = to_img(torch.from_numpy(s["image"]))
        img_t1  = to_img(torch.from_numpy(s["image_t1"]))

        # GT edge + flow arrows
        ax_gt = axes[i, 2]
        gt_vis = overlay_edge(img_t, gt_edge, color=(80,200,80), alpha=0.75)
        ax_gt.imshow(gt_vis); ax_gt.axis("off")
        draw_flow_arrows(ax_gt, gt_flow.numpy(), gt_edge.squeeze().numpy(),
                         stride=10, scale=2.5, color="lime")

        # Predicted edge mask
        ax_pe = axes[i, 3]
        ovl   = overlay_edge(img_t, pred_prob > 0.5, color=(255,80,80), alpha=0.75)
        ax_pe.imshow(ovl); ax_pe.axis("off")
        pred_b = (pred_prob.squeeze().numpy() > 0.5).astype(float)
        gt_b   = gt_edge.squeeze().numpy()
        tp = (pred_b * gt_b).sum()
        fp = (pred_b * (1-gt_b)).sum()
        fn = ((1-pred_b) * gt_b).sum()
        f1 = 2*tp / (2*tp + fp + fn + 1e-6)
        ax_pe.set_xlabel(f"F1={f1:.3f}", fontsize=7.5, labelpad=2)

        # Predicted flow
        ax_pf = axes[i, 4]
        ax_pf.imshow(img_t); ax_pf.axis("off")
        draw_flow_arrows(ax_pf, pred_flow.numpy(), gt_edge.squeeze().numpy(),
                         stride=10, scale=2.5, color="deepskyblue")

        # Per-pixel EPE (endpoint error) at boundary pixels
        ax_err = axes[i, 5]
        epe  = (pred_flow - gt_flow).pow(2).sum(0).sqrt().numpy()  # (H,W)
        mask = gt_edge.squeeze().numpy() > 0.5
        epe_masked = np.where(mask, epe, np.nan)
        im   = ax_err.imshow(epe_masked, cmap="plasma", vmin=0, vmax=8)
        ax_err.axis("off")
        mean_epe = float(epe[mask].mean()) if mask.any() else 0
        ax_err.set_xlabel(f"EPE={mean_epe:.2f}px", fontsize=7.5, labelpad=2)

        # Simple frame visuals
        axes[i, 0].imshow(img_t);  axes[i, 0].axis("off")
        axes[i, 1].imshow(img_t1); axes[i, 1].axis("off")

    # Shared colourbar for EPE column
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    sm = ScalarMappable(cmap="plasma", norm=Normalize(0, 8))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="EPE (px)")

    fig.suptitle("Path 2 — Frame-pair Boundary Flow Estimation\n"
                 "green arrows=GT flow  |  blue arrows=predicted flow  |  red=predicted edge",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    visualize_path1()
    visualize_path2()
    print("Done.")
