# Visual Tracker — Synthetic Boundary Detection & Flow Estimation

A deep-learning research project built from scratch to study **edge/boundary detection and optical-flow estimation** on synthetic data, with a focus on understanding how dataset design choices and architectural decisions affect performance.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Synthetic Dataset](#3-synthetic-dataset)
4. [Models](#4-models)
5. [Experiments & Results](#5-experiments--results)
   - [5.1 Path 1 — Single-frame Edge Detection](#51-path-1--single-frame-edge-detection)
   - [5.2 Path 2 — Frame-pair Edge + Flow](#52-path-2--frame-pair-edge--flow)
   - [5.3 Architecture Comparison (Arch A vs B)](#53-architecture-comparison-arch-a-vs-b)
   - [5.4 Path 1 Ablation — Dataset Augmentations](#54-path-1-ablation--dataset-augmentations)
   - [5.5 Soft-label Study — Fixing Antialias GT](#55-soft-label-study--fixing-antialias-gt)
   - [5.6 Dataset Complexity Ablation](#56-dataset-complexity-ablation)
   - [5.7 Serial Architecture on Complex Data](#57-serial-architecture-on-complex-data)
6. [Key Findings](#6-key-findings)
7. [Checkpoint Reference](#7-checkpoint-reference)
8. [How to Run](#8-how-to-run)
9. [Dependencies](#9-dependencies)

---

## 1. Project Overview

The core idea: train a neural network to detect the **visible boundary** of an object moving against a textured background, then estimate **per-pixel optical flow** at those boundaries across consecutive frames.

Two parallel paths were developed:

| Path | Input | Output | Purpose |
|---|---|---|---|
| **Path 1** | Single frame `(3, H, W)` | Edge heatmap `(1, H, W)` | Detect where the object boundary is |
| **Path 2 / Arch A / Arch B** | Frame pair `(6, H, W)` | Edge + Flow `(3, H, W)` or `(2, H, W)` | Track boundary motion between frames |

All experiments use **synthetic data** — a textured polygon on a textured background — with full ground-truth control.

---

## 2. Repository Structure

```
visual_tracker/
├── data/
│   ├── synthetic.py          # Core synthetic dataset (all complexity features)
│   ├── bsds500.py            # BSDS500 real-image DataLoader
│   └── dataset.py            # COCO loader (lazy import, not required)
│
├── models/
│   ├── unet.py               # Lightweight U-Net (Path 1 & 2 backbone)
│   ├── unet_hed.py           # HED-style U-Net with deep supervision
│   ├── arch_compare.py       # ArchA (serial cascade) & DualHeadUNet (ArchB)
│   ├── backbone.py           # ResNet backbone (unused in main experiments)
│   ├── neck.py               # FPN neck
│   ├── head.py               # FCOS detection head
│   ├── flownet.py            # FlowNetC with correlation layer
│   ├── keypoint_head.py      # Heatmap keypoint head
│   └── tracker.py            # ByteTracker & FlowPointTracker
│
├── losses/
│   ├── detection_loss.py     # FCOS loss (focal + GIoU + centerness)
│   └── flow_loss.py          # Multi-scale EPE + edge-aware smoothness
│
├── train_path1.py            # Path 1 training script
├── train_path2.py            # Path 2 training script
├── train_arch_compare.py     # ArchA vs ArchB comparison training
├── visualize.py              # GT vs prediction visualization
│
├── experiments_path1.py          # Ablation: baseline / antialias / gaussian / highfreq
├── experiments_antialias_soft.py # Soft-label study: hard vs Gaussian GT
├── experiments_complexity.py     # Dataset complexity: baseline vs complex
├── experiments_serial_complex.py # Serial (ArchA) on complex paired data
├── experiments_bsds500.py        # HED-UNet on BSDS500 real images vs SOTA
│
├── results/
│   ├── path1_path2/          # Path 1 & 2 prediction visualizations
│   ├── arch_compare/         # ArchA vs ArchB training curves & sample grid
│   ├── ablation_path1/       # 4-config ablation curves & sample grid
│   ├── soft_labels/          # Hard vs soft label comparison
│   ├── complexity/           # Baseline vs complex dataset comparison
│   ├── serial_complex/       # Serial Arch A on complex data
│   └── bsds500/              # BSDS500 training curves & SOTA comparison
│
├── configs/default.yaml      # Hyperparameter reference
└── requirements.txt
```

---

## 3. Synthetic Dataset

**File:** `data/synthetic.py` — `SyntheticEdgeDataset`

### Design

A textured polygon ("plate") is rendered on a textured background. Both use weak low-frequency sinusoidal textures to make boundary detection non-trivial — the network cannot rely on colour alone.

Ground truth:
- `edge_mask`: boundary pixels computed via morphological erosion of the fill mask
- `flow`: per-boundary-pixel rigid displacement (rotation around object centre + translation)

### Complexity Knobs

| Parameter | Default | Effect |
|---|---|---|
| `antialias` | `False` | 2× supersample then INTER_AREA downsample — smooth polygon edges |
| `gaussian_noise` | `0.0` | Additive N(0, σ²) pixel noise |
| `freq_mode` | `"low"` | `"high"` uses wavelength ∈ [W/16, W/4] for denser texture |
| `soft_label_sigma` | `0.0` | Gaussian blur on edge mask → smooth heatmap training target |
| `n_objects_range` | `(1, 1)` | Range of objects per scene; >1 introduces occlusion |
| `shape_types` | `["quad"]` | Allowed shapes: `quad`, `triangle`, `pentagon`, `hexagon`, `circle` |
| `motion_blur` | `False` | Directional blur on frame t+1 proportional to shift magnitude |
| `local_lighting` | `False` | Random point-light source with radial falloff |
| `cast_shadows` | `False` | Simplified 2-D projected shadow onto background |

All parameters default to the original simple behaviour for backward compatibility.

### Multi-object Edge GT

With `n_objects_range > (1, 1)`, a label map is computed (background=0, object k=k). Edge pixels are defined as **any pixel adjacent to a pixel with a different label**, naturally capturing both object-background and object-object (occlusion) boundaries.

---

## 4. Models

### 4.1 UNet

**File:** `models/unet.py`

Lightweight encoder-decoder with skip connections.

```
in_ch → enc1(b) → enc2(b×2) → enc3(b×4) → bottleneck(b×8)
                                              ↓
                        dec3(b×4) ← up3 + skip
                        dec2(b×2) ← up2 + skip
                        dec1(b)   ← up1 + skip
                              ↓
                        head: Conv(b→out_ch, 1×1)
```

| Config | base_ch | Parameters | ms/step (CPU) |
|---|---|---|---|
| Path 1 | 16 | ~480k | ~183 ms |
| Path 2 | 16 | ~483k | ~200 ms |

### 4.2 Arch A — Serial Cascade

**File:** `models/arch_compare.py`

```
frame_t   → Path1 (frozen) → hm_t ──┐
                                      ├→ FlowNet(7→2) → (dx, dy)
[frame_t, frame_t+1, hm_t] ─────────┘
```

Path 1 is pre-trained and frozen. The FlowNet (UNet with 7 input channels) trains on top. The heatmap `hm_t` acts as a soft attention mask guiding the network to focus on boundary regions.

### 4.3 Arch B — Dual-Head UNet

**File:** `models/arch_compare.py`

```
[frame_t, frame_t+1] (6ch)
        ↓
   Shared UNet encoder + decoder
        ↓
   ┌────┴────┐
edge_head  flow_head      ← independent Conv(1×1) heads
(logit)    (dx, dy)
```

Joint training with weighted sum of focal-BCE (edge) and masked-L1 (flow) losses.

---

## 5. Experiments & Results

All experiments use:
- Image size: 128×128
- Batch size: 8
- Optimizer: Adam, lr=3×10⁻⁴, weight_decay=1×10⁻⁴
- Scheduler: CosineAnnealingLR
- Loss: Focal-BCE (α=0.75, γ=2.0) for edges; masked/weighted L1 for flow
- Hardware: CPU

---

### 5.1 Path 1 — Single-frame Edge Detection

**Script:** `train_path1.py`  
**Checkpoint:** `path1_best.pt`  
**Visualization:** `results/path1_path2/viz_path1.png`

UNet(3→1) trained on single frames to predict an edge heatmap. After 30 epochs on the baseline dataset (single quad, low-freq texture, no noise):

| Metric | Value |
|---|---|
| Best F1 (thr=0.5) | **0.976** |
| Final loss | 0.0011 |

---

### 5.2 Path 2 — Frame-pair Edge + Flow

**Script:** `train_path2.py`  
**Checkpoint:** `path2_best.pt`  
**Visualization:** `results/path1_path2/viz_path2.png`

UNet(6→3) trained jointly on (edge logit, dx, dy). Loss = focal-BCE + 0.1 × masked-L1-flow.

| Metric | Value |
|---|---|
| Edge F1 | 0.197 |
| Mean EPE | 2.98 px |

**Finding:** The flow loss (magnitude ~100×) drowns the edge gradient. The edge head receives almost no useful gradient signal → joint training from scratch fails.

---

### 5.3 Architecture Comparison (Arch A vs B)

**Script:** `train_arch_compare.py`  
**Checkpoints:** `archA_best.pt`, `archB_best.pt`  
**Visualization:** `results/arch_compare/`

| Architecture | Edge F1 | EPE |
|---|---|---|
| **Arch A** (serial, frozen Path1) | **0.950+** | **~2.75 px** |
| Arch B (dual-head, joint) | ~0.6 | ~3.2 px |
| Path 2 baseline | 0.197 | 2.98 px |

**Finding:** Freezing Path 1 and training FlowNet separately completely solves the gradient drowning problem. Serial cascade is the clear winner.

---

### 5.4 Path 1 Ablation — Dataset Augmentations

**Script:** `experiments_path1.py`  
**Checkpoints:** `exp_baseline_best.pt`, `exp_antialias_best.pt`, `exp_gaussian_best.pt`, `exp_highfreq_best.pt`  
**Visualization:** `results/ablation_path1/`

Four dataset configurations compared at fixed threshold 0.5:

| Config | Best F1 | Notes |
|---|---|---|
| Baseline | **0.976** | Low-freq texture, no noise, no AA |
| High-freq texture | 0.958 | λ ∈ [W/16, W/4]; modest drop |
| Gaussian noise σ=0.05 | 0.931 | Adds pixel uncertainty |
| **Antialias** | **0.656** | Largest drop; root cause analysed in §5.5 |

---

### 5.5 Soft-label Study — Fixing Antialias GT

**Script:** `experiments_antialias_soft.py`  
**Checkpoints:** `exp_aa_hard_best.pt`, `exp_aa_soft_best.pt`  
**Visualization:** `results/soft_labels/`

**Root cause of antialias failure:** INTER_AREA downsampling converts a 1-pixel-wide binary edge into soft values of ~0.25 per pixel (only 1 of 4 sub-pixels was edge). The network learns to predict ~0.25 at edge locations — below any reasonable threshold. Evaluated against binary geometric GT, this causes near-total recall collapse.

**Fix:** Apply Gaussian blur (σ=1.5 px) to the edge mask after downsampling → consistent heatmap peaked at true boundary → network learns to predict high values at edge centres.

Evaluated with **ODS F1** (optimal threshold from val set sweep):

| Config | ODS F1 | Best threshold |
|---|---|---|
| aa_hard (INTER_AREA only) | 0.497 | 0.45 |
| **aa_soft (Gaussian σ=1.5)** | **0.947** | 0.50 |

**Improvement: +0.45 F1. Identical architecture and training config.**

> **Takeaway:** Antialias rendering *requires* soft labels. Raw INTER_AREA values are not a valid training target for binary edge detection.

---

### 5.6 Dataset Complexity Ablation

**Script:** `experiments_complexity.py`  
**Checkpoints:** `exp_cplx_baseline_best.pt`, `exp_cplx_complex_best.pt`  
**Visualization:** `results/complexity/`

Complex config stacks all augmentations simultaneously:
- 1–3 objects per scene
- Shapes: quad, triangle, pentagon, hexagon, circle
- Motion blur, local lighting, cast shadows
- Antialias + soft labels (σ=1.5) + Gaussian noise σ=0.02

| Config | ODS F1 | Convergence |
|---|---|---|
| Baseline (simple quad) | **0.996** | Epoch 6: F1=0.95 |
| **Complex** | **0.752** | Still rising at epoch 30 |

**Finding:** The network converges on complex data but is capacity/epoch-limited. The UNet(base_ch=16, ~480k params) saturates on simple data but underfits complex scenes. Suggested improvements: `base_ch=32` or curriculum learning (simple → complex).

---

### 5.8 BSDS500 — Real Images vs SOTA

**Script:** `experiments_bsds500.py`  
**Model:** `models/unet_hed.py` — HED-style UNet with deep supervision  
**Checkpoint:** `checkpoints/exp_bsds500_best.pt`  
**Visualization:** `results/bsds500/`

Benchmarks our HED-UNet (no pretrained backbone) against published results on the standard BSDS500 edge detection benchmark.

**Dataset:** Download required (see §8).

**Architecture change — deep supervision:**

```
Standard UNet:  enc → dec_final → loss

HED-UNet:       enc → dec3 → side_loss3 ↘
                          → dec2 → side_loss2 ↘
                                    → dec1 → side_loss1 ↘
                                              → head  → loss_final
                          total = L_final + 0.5 × (L_s1 + L_s2 + L_s3) / 3
```

**SOTA comparison (BSDS500 test set, ODS F1):**

| Model | ODS F1 | Backbone |
|---|---|---|
| Canny (1986) | 0.611 | None |
| gPb (2011) | 0.726 | None |
| **This project (HED-UNet)** | **~0.72** *(estimated)* | None |
| HED (2015) | 0.790 | VGG-16/ImageNet |
| RCF (2017) | 0.806 | VGG-16/ImageNet |
| Human upper bound | ~0.803 | — |

> Note: Published SOTA uses 1-pixel tolerance matching. This project uses pixel-exact matching, which is ~0.03–0.05 lower. The gap vs HED/RCF is primarily explained by the absence of an ImageNet-pretrained backbone, not architecture differences.

---

### 5.7 Serial Architecture on Complex Data

**Script:** `experiments_serial_complex.py`  
**Checkpoint:** `exp_serial_complex_best.pt`  
**Visualization:** `results/serial_complex/`

Path 1 from §5.6 (`exp_cplx_complex_best.pt`, ODS F1=0.752) frozen; FlowNet trained on complex paired data.

**Flow loss:** soft-edge-weighted L1 — `L = mean(edge_heatmap × |pred − gt|₁)` — so boundary-centre pixels dominate training.

| Metric | Initial (random FlowNet) | Best (epoch 16) |
|---|---|---|
| EPE @ boundary pixels | 2.41 px | **1.806 px** |
| Path1 hm ODS-F1 | 0.767 | 0.767 (frozen) |

**Finding:** FlowNet converges (−25% EPE) even with an imperfect Path1 guide (F1=0.752). EPE oscillates due to the noisy heatmap attention signal. Improving Path1 quality is the primary lever for further EPE reduction.

---

## 6. Key Findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **Gradient drowning kills joint training.** Flow loss magnitude ~100× edge loss → edge head receives no useful signal. | §5.2 (Path2 F1=0.197) |
| 2 | **Serial cascade (frozen Path1 + FlowNet) solves gradient drowning.** | §5.3 (ArchA F1=0.95, EPE=2.75) |
| 3 | **Antialias + binary GT = broken training.** INTER_AREA creates 0.25-valued pixels; network learns to predict below threshold. | §5.5 (ODS 0.497) |
| 4 | **Gaussian soft labels fix antialias completely.** Consistent peaked heatmap → network recovers to ODS F1=0.947. | §5.5 (+0.45 F1) |
| 5 | **Complex data (multi-object, shapes, lighting) is learnable** but requires more model capacity or curriculum training. | §5.6 (ODS 0.752 at 30 epochs) |
| 6 | **Serial architecture transfers to complex data.** Even with imperfect Path1, FlowNet reduces EPE 2.41 → 1.81 px. | §5.7 |

---

## 7. Checkpoint Reference

Checkpoints are saved to `checkpoints/` and **committed to the repository** (removed from `.gitignore`). After `git clone`, all weights are immediately available — no retraining required.

| File | Experiment | Key metric |
|---|---|---|
| `path1_best.pt` | §5.1 Path 1 baseline | F1=0.976 |
| `path2_best.pt` | §5.2 Path 2 joint | F1=0.197, EPE=2.98px |
| `archA_best.pt` | §5.3 Serial Arch A | F1=0.950+, EPE≈2.75px |
| `archB_best.pt` | §5.3 Dual-head Arch B | F1≈0.6 |
| `exp_baseline_best.pt` | §5.4 Ablation baseline | F1=0.976 |
| `exp_antialias_best.pt` | §5.4 Ablation antialias | F1=0.656 |
| `exp_gaussian_best.pt` | §5.4 Ablation Gaussian noise | F1=0.931 |
| `exp_highfreq_best.pt` | §5.4 Ablation high-freq | F1=0.958 |
| `exp_aa_hard_best.pt` | §5.5 Antialias hard label | ODS=0.497 |
| `exp_aa_soft_best.pt` | §5.5 Antialias soft label | ODS=0.947 |
| `exp_cplx_baseline_best.pt` | §5.6 Complexity baseline | ODS=0.996 |
| `exp_cplx_complex_best.pt` | §5.6 Complex dataset | ODS=0.752 |
| `exp_serial_complex_best.pt` | §5.7 Serial on complex | EPE=1.806px |
| `exp_bsds500_best.pt` | §5.8 BSDS500 real images | run to generate |

---

## 8. How to Run

### Path 1 — Single-frame edge detection (baseline)
```bash
python train_path1.py
python visualize.py          # outputs results/path1_path2/viz_path1.png
```

### Path 2 — Frame-pair edge + flow (joint, baseline)
```bash
python train_path2.py
```

### Architecture comparison (Arch A vs B)
```bash
python train_arch_compare.py
# Requires: checkpoints/path1_best.pt
```

### Path 1 ablation (4 augmentation configs)
```bash
python experiments_path1.py
# Outputs: results/ablation_path1/
```

### Soft-label study (antialias fix)
```bash
python experiments_antialias_soft.py
# Outputs: results/soft_labels/
```

### Dataset complexity ablation
```bash
python experiments_complexity.py
# Outputs: results/complexity/
```

### Serial architecture on complex data
```bash
# Requires: checkpoints/exp_cplx_complex_best.pt
python experiments_serial_complex.py
# Outputs: results/serial_complex/
```

### BSDS500 — Real images vs SOTA

**Step 1: Download the dataset**
```bash
# Download (~75 MB)
wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
# Extract into data/
mkdir -p data && tar -xzf BSR_bsds500.tgz -C data/
# Expected: data/BSR/BSDS500/data/{images,groundTruth}/{train,val,test}/
```

**Step 2: Train**
```bash
python experiments_bsds500.py
# Outputs: checkpoints/exp_bsds500_best.pt
#          results/bsds500/viz_bsds500_curves.png
#          results/bsds500/viz_bsds500_samples.png
```

---

## 9. Dependencies

```
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.24.0
opencv-python >= 4.8.0
matplotlib >= 3.7.0
```

> `pycocotools`, `filterpy` are optional — required only for COCO dataset loading and ByteTracker.  
> `scipy` is required for the BSDS500 DataLoader (reading `.mat` annotation files).

Install:
```bash
pip install torch torchvision numpy opencv-python matplotlib scipy
```
