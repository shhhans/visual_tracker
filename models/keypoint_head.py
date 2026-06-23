"""
Heatmap-based keypoint detection head.

Design follows CenterNet / CornerNet conventions:
  - Predicts a Gaussian heatmap per keypoint type.
  - Sub-pixel localization via differentiable soft-argmax.
  - Optional local offset map for fine-grained accuracy.

Output per image:
  coords  : (K, 2)   (x, y) in input-image pixel space
  scores  : (K,)     peak confidence in [0, 1]
  heatmaps: (K, H/4, W/4)  raw heatmaps (for loss computation)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class KeypointHead(nn.Module):
    """
    Lightweight head attached after the FPN or the backbone.

    Args:
        in_channels : number of channels from the feature map (e.g. 256 from FPN)
        num_keypoints: K — number of distinct keypoint types to detect
        heatmap_size : spatial resolution of the output heatmaps relative to input
                       (default 4 → heatmap is H/4 × W/4)
    """

    def __init__(self, in_channels: int, num_keypoints: int, heatmap_size: int = 4):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size  = heatmap_size

        # 3-layer refinement tower
        self.tower = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Heatmap head: one channel per keypoint type
        self.heatmap_head = nn.Conv2d(128, num_keypoints, 1)

        # Local offset head: 2 channels (dx, dy) per keypoint type
        # — offsets refine the quantisation error of heatmap argmax
        self.offset_head = nn.Conv2d(128, num_keypoints * 2, 1)

        self._init_weights()

    def _init_weights(self):
        # Bias initialisation: make initial heatmaps near zero (like focal loss prior)
        prior = 0.01
        nn.init.constant_(self.heatmap_head.bias, -math.log((1 - prior) / prior))
        nn.init.normal_(self.heatmap_head.weight, std=0.01)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.normal_(self.offset_head.weight, std=0.01)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: (B, C, H, W) feature map (typically FPN P2 or the /4 level)

        Returns:
            heatmaps : (B, K, H, W)  raw logits
            offsets  : (B, K*2, H, W)  (dx, dy) offsets per cell per keypoint type
        """
        x = self.tower(feat)
        return self.heatmap_head(x), self.offset_head(x)

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        heatmaps: torch.Tensor,
        offsets:  torch.Tensor,
        stride:   int,
        top_k:    int = 100,
        threshold: float = 0.3,
    ) -> List[dict]:
        """
        Decode heatmaps + offsets into pixel-space keypoint coordinates.

        Args:
            heatmaps  : (B, K, H, W)  sigmoid-activated
            offsets   : (B, K*2, H, W)
            stride    : downsampling factor (feature map → input image)
            top_k     : max keypoints to keep per image per type
            threshold : minimum heatmap confidence

        Returns:
            List[dict] with keys:
              coords : (N, 2) float — (x, y) in input-image pixels
              scores : (N,)   float
              types  : (N,)   int   — keypoint type index
        """
        B, K, H, W = heatmaps.shape
        hm = heatmaps.sigmoid()

        # 3×3 max-pool to suppress non-maxima (nms on heatmaps)
        hm_max = F.max_pool2d(hm, 3, stride=1, padding=1)
        keep   = (hm == hm_max).float()
        hm     = hm * keep

        results = []
        for b in range(B):
            all_coords, all_scores, all_types = [], [], []

            for k in range(K):
                scores_k = hm[b, k]  # (H, W)

                flat_scores, flat_idx = scores_k.flatten().topk(min(top_k, H * W))
                valid = flat_scores >= threshold
                if not valid.any():
                    continue

                flat_scores = flat_scores[valid]
                flat_idx    = flat_idx[valid]

                iy = flat_idx // W
                ix = flat_idx  % W

                # Offset refinement
                dx = offsets[b, k * 2,     iy, ix]
                dy = offsets[b, k * 2 + 1, iy, ix]

                # Convert to full-resolution coords
                x = (ix.float() + 0.5 + dx) * stride
                y = (iy.float() + 0.5 + dy) * stride

                all_coords.append(torch.stack([x, y], dim=-1))
                all_scores.append(flat_scores)
                all_types.append(torch.full_like(flat_scores, k, dtype=torch.long))

            if all_coords:
                results.append({
                    "coords": torch.cat(all_coords),
                    "scores": torch.cat(all_scores),
                    "types":  torch.cat(all_types),
                })
            else:
                results.append({
                    "coords": heatmaps.new_empty(0, 2),
                    "scores": heatmaps.new_empty(0),
                    "types":  torch.empty(0, dtype=torch.long, device=heatmaps.device),
                })

        return results


# ------------------------------------------------------------------
# Target generation for training
# ------------------------------------------------------------------

def gaussian_radius(box_size: Tuple[float, float], min_overlap: float = 0.7) -> float:
    """Compute Gaussian kernel radius for a given box size (CenterNet formula)."""
    h, w  = box_size
    a1    = 1
    b1    = h + w
    c1    = w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1   = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1    = (b1 - sq1) / (2 * a1)
    a2    = 4
    b2    = 2 * (h + w)
    c2    = (1 - min_overlap) * w * h
    sq2   = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2    = (b2 - sq2) / (2 * a2)
    a3    = 4 * min_overlap
    b3    = -2 * min_overlap * (h + w)
    c3    = (min_overlap - 1) * w * h
    sq3   = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3    = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def draw_gaussian(heatmap: torch.Tensor, cy: int, cx: int, radius: int):
    """Render a 2D Gaussian peak onto the heatmap tensor in-place."""
    diameter = 2 * radius + 1
    H, W = heatmap.shape

    ys = torch.arange(-radius, radius + 1, device=heatmap.device, dtype=torch.float32)
    xs = torch.arange(-radius, radius + 1, device=heatmap.device, dtype=torch.float32)
    gaussian = torch.exp(-(xs[None] ** 2 + ys[:, None] ** 2) / (2 * (radius / 3) ** 2 + 1e-6))

    y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
    gy0, gy1 = radius - (cy - y0), radius + (y1 - cy)
    gx0, gx1 = radius - (cx - x0), radius + (x1 - cx)

    if y1 > y0 and x1 > x0:
        heatmap[y0:y1, x0:x1] = torch.maximum(heatmap[y0:y1, x0:x1], gaussian[gy0:gy1, gx0:gx1])


def build_heatmap_targets(
    keypoint_list: List[dict],
    num_keypoints: int,
    heatmap_h: int,
    heatmap_w: int,
    stride: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build ground-truth heatmaps and offset maps for a batch.

    keypoint_list: one dict per image with keys
        coords (N, 2)  pixel-space (x, y)
        types  (N,)    keypoint type index

    Returns:
        hm_target    : (B, K, Hm, Wm)  Gaussian heatmaps in [0,1]
        offset_target: (B, K*2, Hm, Wm) sub-pixel offset ground-truth
        offset_mask  : (B, K, Hm, Wm)  1 where a keypoint exists
    """
    B = len(keypoint_list)
    hm_target     = torch.zeros(B, num_keypoints, heatmap_h, heatmap_w, device=device)
    offset_target = torch.zeros(B, num_keypoints * 2, heatmap_h, heatmap_w, device=device)
    offset_mask   = torch.zeros(B, num_keypoints, heatmap_h, heatmap_w, device=device)

    for b, kp in enumerate(keypoint_list):
        if kp["coords"].numel() == 0:
            continue
        coords = kp["coords"]
        types  = kp["types"]

        for n in range(coords.shape[0]):
            x, y = coords[n, 0].item(), coords[n, 1].item()
            k    = types[n].item()

            cx = int(x / stride)
            cy = int(y / stride)
            if not (0 <= cx < heatmap_w and 0 <= cy < heatmap_h):
                continue

            r = max(1, int(gaussian_radius((8, 8))))
            draw_gaussian(hm_target[b, k], cy, cx, r)

            # Sub-pixel offset from grid centre
            offset_target[b, k * 2,     cy, cx] = x / stride - cx - 0.5
            offset_target[b, k * 2 + 1, cy, cx] = y / stride - cy - 0.5
            offset_mask[b, k, cy, cx] = 1.0

    return hm_target, offset_target, offset_mask
