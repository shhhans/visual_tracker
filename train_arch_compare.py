"""
同时训练 Arch A（串联）和 Arch B（双头），最后生成对比图。

运行：
    python train_arch_compare.py

依赖：checkpoints/path1_best.pt  （必须先跑 train_path1.py）
输出：
    checkpoints/archA_best.pt
    checkpoints/archB_best.pt
    viz_compare.png
"""

import os, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))
from data.synthetic import SyntheticEdgeDataset
from models.arch_compare import ArchA, DualHeadUNet


# ─── losses ──────────────────────────────────────────────────────────────────

def focal_bce(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p   = torch.sigmoid(logits)
    p_t = p * targets + (1-p) * (1-targets)
    a_t = alpha * targets + (1-alpha) * (1-targets)
    return (a_t * (1-p_t)**gamma * bce).mean()

def masked_l1(pred_flow, gt_flow, mask):
    diff   = (pred_flow - gt_flow).abs()
    masked = diff * mask
    return masked.sum() / (mask.sum()*2 + 1e-6)


# ─── metrics ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def metrics(pred_logit, pred_flow, gt_mask, gt_flow, thr=0.5):
    pm  = (torch.sigmoid(pred_logit) > thr).float()
    tp  = (pm * gt_mask).sum(); fp = (pm*(1-gt_mask)).sum(); fn = ((1-pm)*gt_mask).sum()
    p   = tp/(tp+fp+1e-6); r = tp/(tp+fn+1e-6)
    f1  = 2*p*r/(p+r+1e-6)
    epe = (pred_flow - gt_flow).pow(2).sum(1,keepdim=True).sqrt()
    epe_vals = epe[gt_mask.bool()]
    epe_mean = float(epe_vals.mean()) if epe_vals.numel() > 0 else 0.
    return {"f1": f1.item(), "p": p.item(), "r": r.item(), "epe": epe_mean}


# ─── training loop (generic) ─────────────────────────────────────────────────

def run_training(name, model, get_params, forward_fn,
                 ds_train, ds_val, args):
    """
    Generic training loop.

    forward_fn(model, batch, device) → (edge_logit, flow, gt_edge, gt_flow)
    get_params(model)                → iterable of parameters to optimise
    """
    device    = torch.device("cpu")
    optimizer = Adam(get_params(model), lr=args["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args["epochs"], eta_min=args["lr"]*0.01)

    best_f1   = 0.
    log_rows  = []

    for epoch in range(1, args["epochs"]+1):
        model.train()
        t0 = time.time()
        el = fl = 0.

        for step in range(args["steps"]):
            idx   = (epoch*args["steps"] + step) % len(ds_train)
            batch = ds_train.get_batch(args["batch"], start_idx=idx)
            edge_logit, flow, gt_edge, gt_flow = forward_fn(model, batch, device)

            le = focal_bce(edge_logit, gt_edge)
            lf = masked_l1(flow, gt_flow, gt_edge)
            loss = le + args["flow_w"] * lf

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(get_params(model), 5.)
            optimizer.step()
            el += le.item(); fl += lf.item()

        scheduler.step()

        # validation
        model.eval()
        vm = []
        for vi in range(0, 40, args["batch"]):
            vb = ds_val.get_batch(args["batch"], start_idx=vi)
            with torch.no_grad():
                ve, vf, vge, vgf = forward_fn(model, vb, device)
            vm.append(metrics(ve, vf, vge, vgf))

        f1  = np.mean([m["f1"]  for m in vm])
        epe = np.mean([m["epe"] for m in vm])
        n   = args["steps"]
        print(f"[{name}] Epoch {epoch:3d}/{args['epochs']}  "
              f"edge={el/n:.4f}  flow={fl/n:.4f}  "
              f"| f1={f1:.3f}  EPE={epe:.2f}px  ({time.time()-t0:.1f}s)")

        log_rows.append({"epoch": epoch, "f1": f1, "epe": epe})

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "f1": f1, "epe": epe},
                       f"checkpoints/{name}_best.pt")

    print(f"[{name}] Best F1={best_f1:.3f}\n")
    return log_rows


# ─── architecture-specific forward functions ─────────────────────────────────

def forward_archA(model, batch, device):
    ft  = batch["image"].to(device)
    ft1 = batch["image_t1"].to(device)
    ge  = batch["edge_mask"].to(device)
    gf  = batch["flow"].to(device)
    hm_t, flow = model(ft, ft1)
    # For Arch A: edge output = Path1's frozen heatmap (logit form)
    # We don't supervise it here; only supervise flow
    # Return a dummy edge logit = hm_t converted back to logit space
    edge_logit = torch.logit(hm_t.clamp(1e-4, 1-1e-4))
    return edge_logit, flow, ge, gf

def forward_archB(model, batch, device):
    x   = torch.cat([batch["image"], batch["image_t1"]], dim=1).to(device)
    ge  = batch["edge_mask"].to(device)
    gf  = batch["flow"].to(device)
    edge_logit, flow = model(x)
    return edge_logit, flow, ge, gf


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("checkpoints", exist_ok=True)

    args = {
        "epochs":  30,
        "steps":   50,
        "batch":   8,
        "lr":      3e-4,
        "flow_w":  0.1,      # ← 关键：降低 flow 权重，缓解梯度淹没
        "base_ch": 16,
        "texture": 0.4,
    }

    ds_train = SyntheticEdgeDataset(
        size=128, texture_strength=args["texture"],
        with_pairs=True, max_shift=8., max_rot=10.,
        length=args["steps"]*args["batch"]*args["epochs"], seed=0)

    ds_val = SyntheticEdgeDataset(
        size=128, texture_strength=args["texture"],
        with_pairs=True, max_shift=8., max_rot=10.,
        length=200, seed=999)

    # ── Arch A ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Training Arch A  (Path1 frozen → FlowNet)")
    print("=" * 60)
    arch_a = ArchA(path1_ckpt="checkpoints/path1_best.pt",
                   base_ch=args["base_ch"])
    print(f"  FlowNet params: {arch_a.flow_net.param_count():,}  "
          f"(Path1 frozen: {sum(p.numel() for p in arch_a.path1.parameters()):,})")

    logs_a = run_training(
        name       = "archA",
        model      = arch_a,
        get_params = lambda m: m.trainable_params(),
        forward_fn = forward_archA,
        ds_train   = ds_train,
        ds_val     = ds_val,
        args       = args,
    )

    # ── Arch B ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Training Arch B  (Dual-head U-Net, flow_w=0.1)")
    print("=" * 60)
    arch_b = DualHeadUNet(in_ch=6, base_ch=args["base_ch"])
    print(f"  Total params: {arch_b.param_count():,}")

    logs_b = run_training(
        name       = "archB",
        model      = arch_b,
        get_params = lambda m: m.parameters(),
        forward_fn = forward_archB,
        ds_train   = ds_train,
        ds_val     = ds_val,
        args       = args,
    )

    # ── 生成对比图 ────────────────────────────────────────────────────────────
    visualize_compare(arch_a, arch_b, ds_val, logs_a, logs_b, args)


# ─── visualisation ───────────────────────────────────────────────────────────

def visualize_compare(arch_a, arch_b, ds_val, logs_a, logs_b, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from models.unet import UNet

    def to_img(t):
        return ((t.numpy().transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8)

    def overlay(img, mask, color=(255,80,80), alpha=0.7):
        out = img.astype(float).copy()
        m   = mask > 0.5
        for c, v in enumerate(color):
            out[:,:,c] = np.where(m, alpha*v+(1-alpha)*out[:,:,c], out[:,:,c])
        return out.clip(0,255).astype(np.uint8)

    def arrows(ax, flow, mask, stride=10, scale=2.5, color="deepskyblue"):
        H,W = mask.shape
        ys  = np.arange(stride//2,H,stride); xs = np.arange(stride//2,W,stride)
        XX,YY = np.meshgrid(xs,ys); m = mask[YY,XX]>0.5
        if not m.any(): return
        ax.quiver(XX[m],YY[m],flow[0,YY,XX][m],-flow[1,YY,XX][m],
                  angles="xy",scale_units="xy",scale=1/scale,
                  width=0.005,color=color,alpha=0.9)

    # ── also load path2 for baseline comparison ───────────────────────────────
    path2 = UNet(in_ch=6, out_ch=3, base_ch=args["base_ch"])
    ck2   = torch.load("checkpoints/path2_best.pt", map_location="cpu", weights_only=False)
    path2.load_state_dict(ck2["model"]); path2.eval()

    arch_a.eval(); arch_b.eval()

    n_samples = 5
    models_cfg = [
        ("Path2\n(single head, baseline)", "path2",  "tomato"),
        ("Arch A\n(serial cascade)",        "arch_a", "dodgerblue"),
        ("Arch B\n(dual head, flow_w=0.1)", "arch_b", "mediumseagreen"),
    ]

    # ── Figure 1: per-sample visual comparison ────────────────────────────────
    # rows = samples,  cols = [frame_t | gt_edge+flow | path2 | archA | archB]
    ncols = 5
    fig, axes = plt.subplots(n_samples, ncols,
                             figsize=(ncols*2.8, n_samples*2.8),
                             gridspec_kw={"wspace":0.04,"hspace":0.12})

    for j, t in enumerate(["Frame t / t+1",
                            "GT edge + flow",
                            "Path2 (baseline)",
                            "Arch A (serial)",
                            "Arch B (dual-head)"]):
        axes[0,j].set_title(t, fontsize=8.5, fontweight="bold", pad=4)

    for i in range(n_samples):
        s      = ds_val[i]
        ft     = torch.from_numpy(s["image"]).unsqueeze(0)
        ft1    = torch.from_numpy(s["image_t1"]).unsqueeze(0)
        ge     = torch.from_numpy(s["edge_mask"])    # (1,H,W)
        gf     = torch.from_numpy(s["flow"])         # (2,H,W)
        img_t  = to_img(torch.from_numpy(s["image"]))

        # Frame pair (stacked)
        img_t1  = to_img(torch.from_numpy(s["image_t1"]))
        pair    = np.concatenate([img_t, img_t1], axis=0)     # tall stack
        axes[i,0].imshow(img_t); axes[i,0].axis("off")

        # GT
        ax = axes[i,1]
        ax.imshow(overlay(img_t, ge.squeeze().numpy(), color=(80,200,80)))
        ax.axis("off")
        arrows(ax, gf.numpy(), ge.squeeze().numpy(), color="lime")

        # Path2
        with torch.no_grad():
            x2  = torch.cat([ft, ft1], dim=1)
            o2  = path2(x2)[0]
        p2_logit = o2[:1]; p2_flow = o2[1:]
        p2_prob  = torch.sigmoid(p2_logit).squeeze().numpy()
        ax = axes[i,2]
        ax.imshow(overlay(img_t, p2_prob))
        ax.axis("off")
        arrows(ax, p2_flow.squeeze(0).numpy(), ge.squeeze().numpy(), color="deepskyblue")
        mask = ge.squeeze().numpy()>0.5
        epe2 = float((p2_flow.squeeze(0)-gf).pow(2).sum(0).sqrt().numpy()[mask].mean()) if mask.any() else 0
        ax.set_xlabel(f"EPE={epe2:.2f}px", fontsize=7.5, labelpad=2)

        # Arch A
        with torch.no_grad():
            hm_t, fa = arch_a(ft, ft1)
        pa_prob = hm_t.squeeze().numpy()
        ax = axes[i,3]
        ax.imshow(overlay(img_t, pa_prob))
        ax.axis("off")
        arrows(ax, fa.squeeze(0).numpy(), ge.squeeze().numpy(), color="deepskyblue")
        epeA = float((fa.squeeze(0)-gf).pow(2).sum(0).sqrt().numpy()[mask].mean()) if mask.any() else 0
        ax.set_xlabel(f"EPE={epeA:.2f}px", fontsize=7.5, labelpad=2)

        # Arch B
        with torch.no_grad():
            xb  = torch.cat([ft, ft1], dim=1)
            eb, fb = arch_b(xb)
        pb_prob = torch.sigmoid(eb).squeeze().numpy()
        ax = axes[i,4]
        ax.imshow(overlay(img_t, pb_prob))
        ax.axis("off")
        arrows(ax, fb.squeeze(0).numpy(), ge.squeeze().numpy(), color="deepskyblue")
        epeB = float((fb.squeeze(0)-gf).pow(2).sum(0).sqrt().numpy()[mask].mean()) if mask.any() else 0
        ax.set_xlabel(f"EPE={epeB:.2f}px", fontsize=7.5, labelpad=2)

    fig.suptitle("架构对比：红色=预测边缘叠加 | 蓝色箭头=预测 flow | 绿色箭头=GT flow",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.savefig("viz_compare.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: training curves ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for logs, label, color in [
        (logs_a, "Arch A (serial)",        "dodgerblue"),
        (logs_b, "Arch B (dual-head)",      "mediumseagreen"),
    ]:
        epochs = [r["epoch"] for r in logs]
        ax1.plot(epochs, [r["f1"]  for r in logs], color=color, label=label, lw=2)
        ax2.plot(epochs, [r["epe"] for r in logs], color=color, label=label, lw=2)

    # baseline Path2 final values (horizontal dashed lines)
    ck2_data = torch.load("checkpoints/path2_best.pt", map_location="cpu", weights_only=False)
    ax1.axhline(ck2_data.get("f1",  0.197), color="tomato", ls="--", lw=1.5, label="Path2 baseline")
    ax2.axhline(ck2_data.get("epe", 2.98),  color="tomato", ls="--", lw=1.5, label="Path2 baseline")

    ax1.set_title("Edge F1 (↑)"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_title("Flow EPE px (↓)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("训练曲线对比", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig("viz_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("Saved viz_compare.png  viz_curves.png")


if __name__ == "__main__":
    main()
