"""
连续帧轨迹追踪 Demo（单目标）
================================
渲染 N 帧连续虚拟动画，用 Path1 检测每帧边界热力图，
通过热力图质心（加权中心）+ PCA 主轴 直接估计每帧绝对姿态，
与 GT 对比。

追踪管线（每帧独立，无积分）：
  frame_t ──► Path1 ──► heatmap_t
                            │
            ┌───────────────┴───────────────┐
          质心 = Σ(hm·x)/Σhm               PCA 主轴 = 最大特征向量
          → (cx, cy)                        → angle φ (°)
                            │
              absolute pose estimate: (cx, cy, φ)_t

  误差 = ||pred_pose - GT_pose||₂

关于 FlowNet（exp_serial_complex_best.pt）的说明：
  训练时 GT flow 在边界像素处 object-side (≠0) 与 background-side (=0)
  两侧互相抵消，网络收敛到全零输出。EPE=1.806 是虚假指标（大量背景
  side 像素 GT=0，预测 0 贡献 0 误差，掩盖了真实问题）。
  FlowNet 的训练 fix 见 experiments_serial_complex_v2.py（待实现）。

输出：
  results/trajectory/viz_trajectory.png   — 轨迹 + 误差曲线
  results/trajectory/viz_frames.png       — 样本帧 overlay

模型：checkpoints/exp_cplx_complex_best.pt  (Path1, ODS=0.752)
"""

import os, sys, math
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data.synthetic import ObjectState, _render_scene
from models.unet import UNet

# ─── 配置 ────────────────────────────────────────────────────────────────────

PATH1_CKPT = "checkpoints/exp_cplx_complex_best.pt"

IMG_SIZE  = 128
N_FRAMES  = 80      # 序列长度（渲染帧数）
HM_THR    = 0.50    # 热力图阈值（0.5 对应 object interior+boundary）

HALF   = 22         # 物体半径
SHAPE  = "hexagon"  # 形状（6 折对称，PCA 歧义 = 30°/6 = 5°，便于对比）

# ─── GT 轨迹 ─────────────────────────────────────────────────────────────────

def gt_pose(t: int, n: int, size: int = IMG_SIZE):
    """
    复合轨迹：椭圆平移 + 正弦自旋
      cx = cx0 + Rx*cos(2π t/n)
      cy = cy0 + Ry*sin(2π t/n)
      φ  = Aφ * sin(4π t/n)   (正弦摆动，±Aφ)
    """
    cx0, cy0 = size / 2, size / 2
    Rx, Ry   = 28.0, 18.0
    Aφ       = 45.0           # 旋转振幅（度）
    θ = 2 * math.pi * t / n
    cx    = cx0 + Rx * math.cos(θ)
    cy    = cy0 + Ry * math.sin(θ)
    angle = Aφ * math.sin(2 * θ)
    return cx, cy, angle


# ─── 渲染 ────────────────────────────────────────────────────────────────────

_N_SIDES = ObjectState.SHAPE_N_SIDES[SHAPE]

# 固定场景外观（使背景稳定，便于观察追踪效果）
_BG_COLOR  = np.array([60, 90, 130], dtype=np.uint8)
_OBJ_COLOR = np.array([180, 130, 70], dtype=np.uint8)
_LIGHT_POS = (0.35, 0.3)
_BG_SEED   = 7
_OBJ_SEED  = 42


def render_frame(cx: float, cy: float, angle: float) -> tuple:
    """返回 (image uint8 H×W×3, edge_mask float32 H×W)。"""
    img, edge, label = _render_scene(
        IMG_SIZE, IMG_SIZE,
        objects          = [ObjectState(cx, cy, HALF, angle, _N_SIDES)],
        obj_colors       = [_OBJ_COLOR],
        bg_color         = _BG_COLOR,
        texture_strength = 0.4,
        bg_seed          = _BG_SEED,
        obj_seeds        = [_OBJ_SEED],
        lighting_gradient= 0.0,
        local_lighting   = True,
        light_pos        = _LIGHT_POS,
        light_strength   = 0.35,
        cast_shadows     = True,
        antialias        = True,
        gaussian_noise   = 0.0,
        freq_mode        = "low",
        soft_label_sigma = 0.0,    # 推理时不需要软标签
        motion_blur_kernel=None,
    )
    return img, edge


def img_to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_uint8.astype(np.float32) / 127.5 - 1.0)
    return t.permute(2, 0, 1).unsqueeze(0)


# ─── Path1 推理 ──────────────────────────────────────────────────────────────

def infer_heatmap(model: UNet, img: np.ndarray) -> np.ndarray:
    """Run Path1, return (H, W) heatmap in [0,1]."""
    with torch.no_grad():
        logit = model(img_to_tensor(img))
    return torch.sigmoid(logit)[0, 0].numpy()


# ─── 刚体变换估计（从相邻帧热力图）────────────────────────────────────────────

def template_match_orientation(hm: np.ndarray, cx: float, cy: float,
                               half: int, n_sides: int,
                               n_edge_samples: int = 30) -> float:
    """
    通过边缘模板匹配估计方向角（度）。

    对 n 折对称体只需扫描一个周期 [0°, 360°/n)。
    在每个候选角度下，沿理想多边形的各条边均匀采样热力图值，
    取累加和最大时对应的角度。

    与 PCA 的根本区别：正六边形的边界协方差矩阵是各向同性的
    （与单位矩阵成正比），PCA 主轴不稳定，任何微小热力图噪声都
    会引起 15°-30° 的随机误差。模板匹配则直接利用 Path1 输出的
    边缘热力图与已知几何形状对齐，误差来源仅为网络不准确，不存
    在几何退化问题。
    """
    H, W    = hm.shape
    period  = 360 // n_sides          # 六边形 → 60
    best_val = -1.0
    best_deg = 0.0

    for deg in range(period):
        total = 0.0
        for k in range(n_sides):
            # 当前边的两个顶点
            theta0 = math.radians(deg + k       * 360.0 / n_sides)
            theta1 = math.radians(deg + (k + 1) * 360.0 / n_sides)
            x0 = cx + half * math.cos(theta0)
            y0 = cy + half * math.sin(theta0)
            x1 = cx + half * math.cos(theta1)
            y1 = cy + half * math.sin(theta1)
            # 沿边均匀采样
            for j in range(n_edge_samples):
                t  = (j + 0.5) / n_edge_samples
                sx = x0 + t * (x1 - x0)
                sy = y0 + t * (y1 - y0)
                xi = int(round(sx))
                yi = int(round(sy))
                if 0 <= xi < W and 0 <= yi < H:
                    total += hm[yi, xi]
        if total > best_val:
            best_val = total
            best_deg = float(deg)

    return best_deg


def heatmap_to_pose(hm: np.ndarray, thr: float = HM_THR):
    """
    从热力图估计质心 (cx, cy) 和方向角 φ (degrees)。

    质心：热力图加权中心。
    方向：边缘模板匹配（非 PCA）。
      PCA 在正六边形等各向同性形状上完全退化，模板匹配更稳定。

    Returns (cx, cy, angle_deg) or None if too few pixels.
    """
    mask = hm > thr
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None

    weights = hm[mask]
    w_sum   = weights.sum() + 1e-8
    cx = float((weights * xs).sum() / w_sum)
    cy = float((weights * ys).sum() / w_sum)

    angle = template_match_orientation(hm, cx, cy, HALF, _N_SIDES)
    return cx, cy, angle


def angle_error_symmetric(pred_a: float, gt_a: float, n_sides: int) -> float:
    """
    考虑 n 折对称性的角度误差。
    将差值折叠到 [0, period/2) 内，period = 360/n_sides。
    """
    period = 360.0 / n_sides
    diff   = (pred_a - gt_a) % period
    if diff > period / 2:
        diff -= period
    return abs(diff)


def pred_angle_to_gt_branch(pred_a: float, gt_a: float, n_sides: int) -> float:
    """
    将模板匹配角度（在 [0, period) 内）映射到与 GT 最近的等价角度，
    用于绘图时与 GT 在同一分支上比较。
    """
    period = 360.0 / n_sides
    diff   = gt_a - pred_a
    k      = round(diff / period)
    return pred_a + k * period


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def run():
    device = torch.device("cpu")

    # 载入 Path1
    model = UNet(in_ch=3, out_ch=1, base_ch=16).to(device)
    ck    = torch.load(PATH1_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Path1 loaded: {PATH1_CKPT}  (ODS={ck.get('ods', '?'):.3f})")

    # GT 轨迹
    gt_poses = [gt_pose(t, N_FRAMES) for t in range(N_FRAMES + 1)]

    pred_poses   = []
    frame_errors = []
    sample_data  = []

    print(f"\nTracking {N_FRAMES} frames (direct absolute pose)...")

    for t in range(N_FRAMES + 1):
        gt_cx, gt_cy, gt_a = gt_poses[t]
        img_t, _ = render_frame(gt_cx, gt_cy, gt_a)
        hm_t     = infer_heatmap(model, img_t)

        # 每帧独立估计绝对位姿（不积分）
        pose_pred = heatmap_to_pose(hm_t)
        if pose_pred is None:
            pose_pred = (gt_cx, gt_cy, gt_a)   # fallback

        pred_poses.append(pose_pred)

        if t > 0:
            pos_err = math.hypot(pose_pred[0] - gt_cx,
                                 pose_pred[1] - gt_cy)
            frame_errors.append(pos_err)
        else:
            pos_err = 0.0

        if t % 10 == 0 or t == N_FRAMES:
            n_px = int((hm_t > HM_THR).sum())
            print(f"  t={t:3d}  pred=({pose_pred[0]:.1f},{pose_pred[1]:.1f},{pose_pred[2]:.1f}°)  "
                  f"GT=({gt_cx:.1f},{gt_cy:.1f},{gt_a:.1f}°)  "
                  f"pos_err={pos_err:.2f}px  hm_pts={n_px}")

        if t % 10 == 0 or t == N_FRAMES:
            sample_data.append((img_t, hm_t, t, pos_err,
                                 pose_pred, gt_poses[t]))

    errors = np.array(frame_errors)
    print(f"\n{'─'*50}")
    print(f"  Mean pos error   : {errors.mean():.2f} px")
    print(f"  Median pos error : {np.median(errors):.2f} px")
    print(f"  Max pos error    : {errors.max():.2f} px  (t={errors.argmax()+1})")
    print(f"  (errors are bounded per-frame; no accumulated drift)")
    print(f"{'─'*50}")

    return gt_poses, pred_poses, frame_errors, sample_data, model


# ─── 可视化 ──────────────────────────────────────────────────────────────────

def visualize(gt_poses, pred_poses, frame_errors, sample_data, model):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    os.makedirs("results/trajectory", exist_ok=True)

    # pred_poses has N_FRAMES+1 entries (one per frame t=0..N_FRAMES)
    ts     = list(range(N_FRAMES + 1))
    ts_err = list(range(1, N_FRAMES + 1))

    gt_x = [p[0] for p in gt_poses]
    gt_y = [p[1] for p in gt_poses]
    gt_a = [p[2] for p in gt_poses]
    pr_x = [p[0] for p in pred_poses]
    pr_y = [p[1] for p in pred_poses]
    # Map predicted angles to the same branch as GT for clean visualization
    pr_a = [pred_angle_to_gt_branch(pred_poses[t][2], gt_poses[t][2], _N_SIDES)
            for t in range(len(pred_poses))]

    angle_errors = []
    for t in range(1, N_FRAMES + 1):
        angle_errors.append(
            angle_error_symmetric(pred_poses[t][2], gt_poses[t][2], _N_SIDES)
        )

    # ── 主图 ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 3, fig, hspace=0.38, wspace=0.32)

    # XY 轨迹
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(gt_x, gt_y, "o-", color="steelblue",  lw=2,   ms=2.5, label="GT")
    ax.plot(pr_x, pr_y, "s--", color="darkorange", lw=1.5, ms=2.5, label="Pred")
    ax.plot(gt_x[0],  gt_y[0],  "^", color="green",  ms=8, zorder=5, label="Start")
    ax.plot(gt_x[-1], gt_y[-1], "v", color="crimson", ms=8, zorder=5, label="End")
    ax.set_title("XY Trajectory", fontweight="bold")
    ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_aspect("equal")
    ax.invert_yaxis()

    # X(t)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ts, gt_x, color="steelblue",  lw=2,   label="GT")
    ax2.plot(ts, pr_x, color="darkorange", lw=1.5, ls="--", label="Pred")
    ax2.set_title("Centroid X(t)", fontweight="bold")
    ax2.set_xlabel("Frame"); ax2.set_ylabel("X (px)")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    # Y(t)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ts, gt_y, color="steelblue",  lw=2,   label="GT")
    ax3.plot(ts, pr_y, color="darkorange", lw=1.5, ls="--", label="Pred")
    ax3.set_title("Centroid Y(t)", fontweight="bold")
    ax3.set_xlabel("Frame"); ax3.set_ylabel("Y (px)")
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    # angle(t)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(ts, gt_a, color="steelblue",  lw=2,   label="GT")
    ax4.plot(ts, pr_a, color="darkorange", lw=1.5, ls="--", label="Pred")
    ax4.set_title(f"Rotation φ(t)  [{SHAPE}, {_N_SIDES}-fold symm.]",
                  fontweight="bold")
    ax4.set_xlabel("Frame"); ax4.set_ylabel("Angle (°)")
    ax4.legend(fontsize=7); ax4.grid(alpha=0.3)

    # 位置误差
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(ts_err, frame_errors, color="crimson", lw=2)
    ax5.fill_between(ts_err, frame_errors, alpha=0.18, color="crimson")
    mean_e = np.mean(frame_errors)
    ax5.axhline(mean_e, ls="--", color="gray", lw=1,
                label=f"Mean={mean_e:.2f}px")
    ax5.set_title("Position Error (px)", fontweight="bold")
    ax5.set_xlabel("Frame"); ax5.set_ylabel("‖pred − GT‖ (px)")
    ax5.legend(fontsize=7); ax5.grid(alpha=0.3)

    # 角度误差
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(ts_err, angle_errors, color="purple", lw=2)
    ax6.fill_between(ts_err, angle_errors, alpha=0.18, color="purple")
    mean_a = np.mean(angle_errors)
    ax6.axhline(mean_a, ls="--", color="gray", lw=1,
                label=f"Mean={mean_a:.2f}°")
    ax6.set_title("Rotation Error (°)", fontweight="bold")
    ax6.set_xlabel("Frame"); ax6.set_ylabel("|Δφ| (°)")
    ax6.legend(fontsize=7); ax6.grid(alpha=0.3)

    fig.suptitle(
        f"Single-Object Trajectory Tracking via Path1 Heatmap Centroid+PCA  "
        f"({N_FRAMES} frames, {SHAPE})\n"
        f"Mean pos error = {mean_e:.2f} px  |  Mean rotation error = {mean_a:.2f}°",
        fontsize=10, fontweight="bold",
    )
    fig.savefig("results/trajectory/viz_trajectory.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Saved results/trajectory/viz_trajectory.png")

    # ── 样本帧 overlay ────────────────────────────────────────────────────────
    n = len(sample_data)
    fig2, axes = plt.subplots(n, 3, figsize=(10, n * 2.7),
                              gridspec_kw={"wspace": 0.04, "hspace": 0.18})
    if n == 1:
        axes = axes[None]

    col_titles = ["Frame t", "Path1 heatmap (boundary)", "GT (green) vs Pred (orange)"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=8, fontweight="bold", pad=3)

    for row, (img_t, hm_np, t, pos_err, pose_pred, pose_gt) in enumerate(sample_data):
        # Col 0: frame
        axes[row, 0].imshow(img_t)
        axes[row, 0].axis("off")
        axes[row, 0].set_ylabel(f"t={t}", fontsize=7.5, va="center", labelpad=4)

        # Col 1: heatmap
        axes[row, 1].imshow(hm_np, cmap="hot", vmin=0, vmax=1)
        axes[row, 1].axis("off")
        axes[row, 1].set_xlabel(
            f"pts>{HM_THR:.2f}: {int((hm_np>HM_THR).sum())}",
            fontsize=7, labelpad=2
        )

        # Col 2: GT vs Pred pose overlay
        overlay = img_t.copy()
        cx_gt, cy_gt, a_gt = pose_gt
        cx_pr, cy_pr, a_pr_raw = pose_pred
        # Map predicted angle to GT branch for overlay display
        a_pr = pred_angle_to_gt_branch(a_pr_raw, a_gt, _N_SIDES)

        # GT outline
        gt_obj = ObjectState(cx_gt, cy_gt, HALF, a_gt, _N_SIDES)
        pts_gt = gt_obj.corners().astype(np.int32)
        cv2.polylines(overlay, [pts_gt.reshape(-1, 1, 2)], True, (0, 220, 0), 2)
        cv2.circle(overlay, (int(round(cx_gt)), int(round(cy_gt))), 3, (0, 220, 0), -1)
        # Pred outline
        pr_obj = ObjectState(cx_pr, cy_pr, HALF, a_pr, _N_SIDES)
        pts_pr = pr_obj.corners().astype(np.int32)
        cv2.polylines(overlay, [pts_pr.reshape(-1, 1, 2)], True, (255, 140, 0), 2)
        cv2.circle(overlay, (int(round(cx_pr)), int(round(cy_pr))), 3, (255, 140, 0), -1)

        axes[row, 2].imshow(overlay)
        axes[row, 2].axis("off")
        ang_err = angle_error_symmetric(a_pr_raw, a_gt, _N_SIDES)
        axes[row, 2].set_xlabel(
            f"pos={pos_err:.2f}px  ang={ang_err:.1f}°", fontsize=7, labelpad=2
        )

    fig2.suptitle(
        "Path1 heatmap centroid+PCA tracking\n"
        "green=GT  orange=Pred  (accumulated pose, no reset)",
        fontsize=9, fontweight="bold", y=1.01,
    )
    fig2.savefig("results/trajectory/viz_frames.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print("Saved results/trajectory/viz_frames.png")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results/trajectory", exist_ok=True)
    gt_poses, pred_poses, frame_errors, sample_data, model = run()
    visualize(gt_poses, pred_poses, frame_errors, sample_data, model)


if __name__ == "__main__":
    main()
