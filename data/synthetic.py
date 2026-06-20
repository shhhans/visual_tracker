"""
Synthetic dataset: a textured square plate on a textured background.

Ground-truth outputs
--------------------
edge_mask : (H, W) float32  — 1 at boundary pixels, 0 elsewhere
flow      : (H, W, 2) float32 — (dx, dy) displacement at boundary pixels
              (only meaningful for frame-pair samples, zero otherwise)

Texture design
--------------
Both background and plate use "weak texture":
  sum of low-frequency sinusoids with small amplitude.
Texture strength is controlled by `texture_strength` ∈ [0, 1].
At 0.0 the plate is flat-coloured; at 1.0 the texture nearly masks edges.

This mimics real-world scenarios where object and background have similar
local appearance, stressing the network to rely on geometric boundaries.
"""

import math
import random
import numpy as np
import cv2
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Low-level texture primitives
# ---------------------------------------------------------------------------

def _low_freq_noise(h: int, w: int, num_waves: int = 6, seed: int = 0) -> np.ndarray:
    """
    Sum of random low-frequency sine waves → smooth "natural" texture.
    Returns float32 in [0, 1].
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w), dtype=np.float32)
    ys  = np.arange(h, dtype=np.float32)
    xs  = np.arange(w, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)

    for _ in range(num_waves):
        # wavelength between 1/4 and full image size
        lx  = rng.uniform(w / 4, w)
        ly  = rng.uniform(h / 4, h)
        phi = rng.uniform(0, 2 * math.pi)
        amp = rng.uniform(0.3, 1.0)
        img += amp * np.sin(2 * math.pi * XX / lx + 2 * math.pi * YY / ly + phi)

    # Normalise to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img


def _make_texture_layer(h: int, w: int, base_color: np.ndarray,
                         texture_strength: float, seed: int) -> np.ndarray:
    """
    RGB texture layer = base_color blended with low-freq noise pattern.
    base_color: (3,) uint8
    Returns (H, W, 3) uint8
    """
    noise = _low_freq_noise(h, w, seed=seed)                       # [0,1]
    noise_rgb = np.stack([noise] * 3, axis=-1)                     # (H,W,3)
    base = base_color.astype(np.float32)[None, None, :] / 255.0   # (1,1,3)

    # Modulate base colour with noise
    layer = base * (1.0 - texture_strength * 0.5 + texture_strength * noise_rgb)
    return np.clip(layer * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Square plate rendering
# ---------------------------------------------------------------------------

class PlateState:
    """Holds the geometric state of the plate (position, size, angle)."""

    def __init__(self, cx: float, cy: float, half: float, angle_deg: float):
        self.cx    = cx          # centre x in pixels
        self.cy    = cy          # centre y in pixels
        self.half  = half        # half-side length in pixels
        self.angle = angle_deg   # rotation angle in degrees

    def corners(self) -> np.ndarray:
        """Return the 4 corners as (4, 2) float32 array (x, y)."""
        rad = math.radians(self.angle)
        cos, sin = math.cos(rad), math.sin(rad)
        offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
        rot = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        corners = (offsets * self.half) @ rot.T
        corners += np.array([self.cx, self.cy])
        return corners.astype(np.float32)

    def apply_transform(self, dx: float = 0, dy: float = 0,
                        dangle: float = 0, dscale: float = 1.0) -> 'PlateState':
        return PlateState(
            cx        = self.cx + dx,
            cy        = self.cy + dy,
            half      = self.half * dscale,
            angle_deg = self.angle + dangle,
        )


def _render_frame(h: int, w: int,
                  state: PlateState,
                  bg_color: np.ndarray,
                  plate_color: np.ndarray,
                  texture_strength: float,
                  bg_seed: int,
                  plate_seed: int,
                  lighting_gradient: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render one frame.  Returns:
        image     : (H, W, 3) uint8  RGB
        edge_mask : (H, W) float32   1 at 1-px boundary, 0 elsewhere
    """
    # Background
    image = _make_texture_layer(h, w, bg_color, texture_strength, seed=bg_seed)

    # Plate fill mask (anti-aliased polygon)
    corners = state.corners()
    fill_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(fill_mask, [corners.reshape(-1, 1, 2).astype(np.int32)], 255)

    # Plate texture
    plate_tex = _make_texture_layer(h, w, plate_color, texture_strength, seed=plate_seed)

    # Composite plate onto background
    alpha = (fill_mask[..., None].astype(np.float32) / 255.0)
    image = (plate_tex * alpha + image * (1 - alpha)).astype(np.uint8)

    # Optional lighting gradient across whole image
    if lighting_gradient > 0:
        ramp = np.linspace(1 - lighting_gradient, 1 + lighting_gradient, w,
                           dtype=np.float32)[None, :, None]
        image = np.clip(image.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

    # Edge mask: boundary = fill_mask XOR eroded fill_mask
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded    = cv2.erode(fill_mask, kernel, iterations=1)
    edge_mask = ((fill_mask - eroded) > 0).astype(np.float32)

    return image, edge_mask


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SyntheticEdgeDataset:
    """
    Generates (image, edge_mask) single-frame samples  — for Path 1.
    Call with_pairs=True to also get frame-pair + flow  — for Path 2.

    Parameters
    ----------
    size          : image size (square)
    texture_strength : 0 = flat colour, 1 = heavy texture (edges harder to see)
    with_pairs    : if True, each sample returns two frames + gt flow at boundary
    max_shift     : max pixel shift between frame pair (x, y)
    max_rot       : max rotation between frame pair (degrees)
    """

    def __init__(
        self,
        size:              int   = 128,
        texture_strength:  float = 0.4,
        with_pairs:        bool  = False,
        max_shift:         float = 8.0,
        max_rot:           float = 10.0,
        length:            int   = 2000,
        seed:              int   = 42,
    ):
        self.size             = size
        self.texture_strength = texture_strength
        self.with_pairs       = with_pairs
        self.max_shift        = max_shift
        self.max_rot          = max_rot
        self.length           = length
        self.rng              = random.Random(seed)
        self.np_rng           = np.random.RandomState(seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        rng = self.rng
        sz  = self.size

        # Random colours: ensure plate and bg are visually distinct enough
        bg_color    = np.array([rng.randint(30, 200)] * 3, dtype=np.uint8)
        plate_color = np.array([rng.randint(30, 200)] * 3, dtype=np.uint8)
        # Keep a minimum luminance difference
        while abs(int(bg_color[0]) - int(plate_color[0])) < 30:
            plate_color = np.array([rng.randint(30, 200)] * 3, dtype=np.uint8)

        # Random plate geometry
        half    = rng.uniform(sz * 0.15, sz * 0.35)
        margin  = half + 2
        cx      = rng.uniform(margin, sz - margin)
        cy      = rng.uniform(margin, sz - margin)
        angle   = rng.uniform(0, 45)

        state_t = PlateState(cx, cy, half, angle)

        # Texture seeds (vary per sample)
        bg_seed    = idx * 7 + 1
        plate_seed = idx * 7 + 3

        img_t, edge_t = _render_frame(
            sz, sz, state_t, bg_color, plate_color,
            self.texture_strength, bg_seed, plate_seed,
            lighting_gradient=rng.uniform(0, 0.15),
        )

        sample = {
            "image":     _to_float(img_t),     # (3, H, W) float32 in [-1, 1]
            "edge_mask": edge_t[None],          # (1, H, W) float32
        }

        if self.with_pairs:
            # Random rigid transform for frame t+1
            dx     = rng.uniform(-self.max_shift, self.max_shift)
            dy     = rng.uniform(-self.max_shift, self.max_shift)
            dangle = rng.uniform(-self.max_rot,   self.max_rot)

            state_t1 = state_t.apply_transform(dx=dx, dy=dy, dangle=dangle)
            img_t1, edge_t1 = _render_frame(
                sz, sz, state_t1, bg_color, plate_color,
                self.texture_strength, bg_seed + 1, plate_seed + 1,
                lighting_gradient=rng.uniform(0, 0.15),
            )

            # Ground-truth flow at boundary pixels of frame t:
            # each boundary pixel p moves to p' = R(p - c) + c + (dx, dy)
            flow = _compute_boundary_flow(edge_t, state_t, dx, dy, dangle)

            sample["image_t1"]  = _to_float(img_t1)   # (3, H, W)
            sample["edge_t1"]   = edge_t1[None]        # (1, H, W)
            sample["flow"]      = flow                  # (2, H, W) dx/dy

        return sample

    # Convenience: collate into batch tensors
    def get_batch(self, batch_size: int, start_idx: int = 0) -> Dict[str, 'torch.Tensor']:
        import torch
        samples = [self[i + start_idx] for i in range(batch_size)]
        return {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in samples[0]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(img: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 → (3,H,W) float32 in [-1, 1]"""
    return (img.astype(np.float32).transpose(2, 0, 1) / 127.5) - 1.0


def _compute_boundary_flow(edge_mask: np.ndarray,
                            state: PlateState,
                            dx: float, dy: float,
                            dangle: float) -> np.ndarray:
    """
    Analytically compute (dx, dy) displacement for each boundary pixel.

    The transform is: rotate by dangle around plate centre, then translate.
      p' = R(dangle) @ (p - c) + c + [dx, dy]
    So flow at p = p' - p
    """
    h, w     = edge_mask.shape
    rad      = math.radians(dangle)
    cos, sin = math.cos(rad), math.sin(rad)
    cx, cy_  = state.cx, state.cy

    ys, xs = np.where(edge_mask > 0)          # boundary pixel coords
    px = xs.astype(np.float32) - cx
    py = ys.astype(np.float32) - cy_

    rx =  cos * px - sin * py + cx + dx - xs
    ry =  sin * px + cos * py + cy_ + dy - ys

    flow = np.zeros((2, h, w), dtype=np.float32)
    flow[0, ys, xs] = rx   # dx
    flow[1, ys, xs] = ry   # dy
    return flow


# ---------------------------------------------------------------------------
# Quick visual sanity check (run directly: python -m data.synthetic)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    ds = SyntheticEdgeDataset(size=256, texture_strength=0.4, with_pairs=True)
    os.makedirs("debug_samples", exist_ok=True)

    for i in range(4):
        s = ds[i]
        img = ((s["image"].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        edge = (s["edge_mask"][0] * 255).astype(np.uint8)
        edge_bgr = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        # Overlay edge in red on image
        overlay = img.copy()
        overlay[s["edge_mask"][0] > 0] = [255, 0, 0]

        if "image_t1" in s:
            img_t1 = ((s["image_t1"].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            row = np.concatenate([img, overlay, img_t1], axis=1)
        else:
            row = np.concatenate([img, overlay], axis=1)

        cv2.imwrite(f"debug_samples/sample_{i:02d}.png",
                    cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        print(f"Sample {i}: image {s['image'].shape}  edge nonzero={s['edge_mask'].sum():.0f}")

    print("Saved to debug_samples/")
