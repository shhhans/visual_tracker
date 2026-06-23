"""
Synthetic dataset: textured objects on textured background.

Ground-truth outputs
--------------------
edge_mask : (1, H, W) float32 — 1 at visible boundary pixels, 0 elsewhere
flow      : (2, H, W) float32 — (dx, dy) displacement at boundary pixels
              (only for frame-pair samples)

Complexity knobs
----------------
shape_types      : list of "quad" | "triangle" | "pentagon" | "hexagon" | "circle"
n_objects_range  : (min, max) number of objects per frame (supports occlusion)
motion_blur      : directional blur on frame_t+1 (proportional to shift magnitude)
local_lighting   : random point-light with radial falloff
cast_shadows     : simplified 2-D projected shadow onto background
antialias        : 2× supersample then INTER_AREA downsample
gaussian_noise   : additive N(0, σ²) pixel noise
freq_mode        : "low" | "high" texture spatial frequency
soft_label_sigma : Gaussian blur on edge mask → smooth heatmap target
"""

import math
import random
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Low-level texture primitives
# ---------------------------------------------------------------------------

def _low_freq_noise(h: int, w: int, num_waves: int = 6, seed: int = 0,
                    freq_mode: str = "low") -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w), dtype=np.float32)
    ys  = np.arange(h, dtype=np.float32)
    xs  = np.arange(w, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    if freq_mode == "high":
        wl_lo, wl_hi = w / 16, w / 4
    else:
        wl_lo, wl_hi = w / 4,  w
    for _ in range(num_waves):
        lx  = rng.uniform(wl_lo, wl_hi)
        ly  = rng.uniform(wl_lo * (h / w), wl_hi * (h / w))
        phi = rng.uniform(0, 2 * math.pi)
        amp = rng.uniform(0.3, 1.0)
        img += amp * np.sin(2 * math.pi * XX / lx + 2 * math.pi * YY / ly + phi)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img


def _make_texture_layer(h: int, w: int, base_color: np.ndarray,
                         texture_strength: float, seed: int,
                         freq_mode: str = "low") -> np.ndarray:
    noise     = _low_freq_noise(h, w, seed=seed, freq_mode=freq_mode)
    noise_rgb = np.stack([noise] * 3, axis=-1)
    base      = base_color.astype(np.float32)[None, None, :] / 255.0
    layer = base * (1.0 - texture_strength * 0.5 + texture_strength * noise_rgb)
    return np.clip(layer * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Object geometry
# ---------------------------------------------------------------------------

class ObjectState:
    """
    Geometric state of one object: center, half-size, rotation, shape.

    n_sides : 0 = circle, 3 = triangle, 4 = quad, 5 = pentagon, 6 = hexagon, …
    """

    SHAPE_N_SIDES: Dict[str, int] = {
        "quad": 4, "triangle": 3, "pentagon": 5, "hexagon": 6, "circle": 0,
    }

    def __init__(self, cx: float, cy: float, half: float,
                 angle_deg: float, n_sides: int = 4):
        self.cx      = cx
        self.cy      = cy
        self.half    = half
        self.angle   = angle_deg
        self.n_sides = n_sides

    def get_fill_mask(self, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.n_sides == 0:
            cv2.circle(mask, (int(round(self.cx)), int(round(self.cy))),
                       int(round(self.half)), 255, -1)
        else:
            pts = self._poly_corners()
            cv2.fillPoly(mask, [pts.reshape(-1, 1, 2).astype(np.int32)], 255)
        return mask

    def _poly_corners(self) -> np.ndarray:
        n   = self.n_sides
        rad = math.radians(self.angle)
        angles = [2 * math.pi * k / n + rad for k in range(n)]
        return np.array([[self.cx + self.half * math.cos(a),
                          self.cy + self.half * math.sin(a)]
                         for a in angles], dtype=np.float32)

    def corners(self) -> np.ndarray:
        """Legacy compatibility: return corner points as (n, 2) float32."""
        if self.n_sides == 0:
            angles = [2 * math.pi * k / 16 for k in range(16)]
            return np.array([[self.cx + self.half * math.cos(a),
                              self.cy + self.half * math.sin(a)]
                             for a in angles], dtype=np.float32)
        return self._poly_corners()

    def apply_transform(self, dx: float = 0, dy: float = 0,
                        dangle: float = 0, dscale: float = 1.0) -> "ObjectState":
        return ObjectState(self.cx + dx, self.cy + dy,
                           self.half * dscale, self.angle + dangle,
                           self.n_sides)


# Legacy alias
PlateState = ObjectState


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _make_motion_blur_kernel(dx: float, dy: float,
                              min_len: int = 2, max_len: int = 15
                              ) -> Optional[np.ndarray]:
    """Directional motion-blur kernel aligned with displacement (dx, dy)."""
    length = int(np.clip(math.hypot(dx, dy), min_len, max_len))
    if length < 2:
        return None
    sz     = length * 2 + 1
    kernel = np.zeros((sz, sz), dtype=np.float32)
    cx     = length
    cv2.line(kernel, (0, cx), (sz - 1, cx), 1.0, 1)
    angle  = math.degrees(math.atan2(dy, dx))
    M      = cv2.getRotationMatrix2D((float(cx), float(cx)), -angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (sz, sz))
    total  = kernel.sum()
    return kernel / total if total > 0 else None


def _apply_local_lighting(image: np.ndarray,
                           light_pos: Tuple[float, float],
                           strength: float = 0.4) -> np.ndarray:
    """
    Radial point-light: bright near light_pos, darker at edges.
    light_pos is normalized (x, y) in [0, 1].
    """
    h, w  = image.shape[:2]
    lx    = light_pos[0] * w
    ly    = light_pos[1] * h
    ys    = np.arange(h, dtype=np.float32)[:, None]
    xs    = np.arange(w, dtype=np.float32)[None, :]
    dist  = np.sqrt((xs - lx) ** 2 + (ys - ly) ** 2)
    diag  = math.sqrt(h ** 2 + w ** 2) * 0.5 + 1e-6
    fall  = np.clip((1.0 + strength) - strength * dist / diag,
                    1.0 - strength, 1.0 + strength)[:, :, None]
    return np.clip(image.astype(np.float32) * fall, 0, 255).astype(np.uint8)


def _cast_shadow(image: np.ndarray,
                 fill_masks: List[np.ndarray],
                 light_pos: Tuple[float, float],
                 combined_fg: np.ndarray,
                 strength: float = 0.45,
                 blur_sigma: float = 3.0) -> np.ndarray:
    """
    Simplified 2-D shadow projection.
    For each object, shift its mask in the direction away from the light and
    darken background pixels that fall under the projected shadow.
    """
    h, w  = image.shape[:2]
    lx    = light_pos[0] * w
    ly    = light_pos[1] * h
    image = image.astype(np.float32)

    for mask in fill_masks:
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        ocx = float(xs.mean()); ocy = float(ys.mean())
        dir_x = ocx - lx;  dir_y = ocy - ly
        dist  = math.hypot(dir_x, dir_y) + 1e-6
        dir_x /= dist;  dir_y /= dist

        shadow_len = int(math.sqrt(float((mask > 0).sum())) * 0.3)
        sdx = int(dir_x * shadow_len)
        sdy = int(dir_y * shadow_len)

        shadow = np.zeros((h, w), dtype=np.float32)
        dst_y0 = max(0, sdy);     dst_y1 = min(h, h + sdy)
        dst_x0 = max(0, sdx);     dst_x1 = min(w, w + sdx)
        src_y0 = max(0, -sdy);    src_y1 = min(h, h - sdy)
        src_x0 = max(0, -sdx);    src_x1 = min(w, w - sdx)
        shadow[dst_y0:dst_y1, dst_x0:dst_x1] = \
            (mask[src_y0:src_y1, src_x0:src_x1] > 0).astype(np.float32)

        if blur_sigma > 0:
            shadow = cv2.GaussianBlur(shadow, (0, 0), blur_sigma)

        bg = combined_fg == 0
        for c in range(3):
            image[:, :, c][bg] *= (1.0 - strength * shadow[bg])

    return np.clip(image, 0, 255).astype(np.uint8)


def _edge_from_label_map(label_map: np.ndarray) -> np.ndarray:
    """
    Boundary pixels: any pixel adjacent (4-connected) to a pixel with
    a different label (background=0, object i → label i+1).
    Captures object-background and object-object boundaries.
    """
    edge = np.zeros(label_map.shape, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        edge |= label_map != np.roll(np.roll(label_map, dy, axis=0), dx, axis=1)
    return edge.astype(np.float32)


# ---------------------------------------------------------------------------
# Scene rendering
# ---------------------------------------------------------------------------

def _render_scene(
    h: int, w: int,
    objects:          List[ObjectState],
    obj_colors:       List[np.ndarray],
    bg_color:         np.ndarray,
    texture_strength: float,
    bg_seed:          int,
    obj_seeds:        List[int],
    lighting_gradient: float = 0.0,
    local_lighting:   bool  = False,
    light_pos:        Optional[Tuple[float, float]] = None,
    light_strength:   float = 0.4,
    cast_shadows:     bool  = False,
    antialias:        bool  = False,
    gaussian_noise:   float = 0.0,
    freq_mode:        str   = "low",
    soft_label_sigma: float = 0.0,
    motion_blur_kernel: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render a scene with multiple objects (painter's algorithm, back-to-front).

    Returns
    -------
    image     : (H, W, 3) uint8
    edge_mask : (H, W) float32  — visible boundaries (soft when antialias/sigma>0)
    label_map : (H, W) int32    — 0=background, k=object k (1-indexed, front wins)
    """
    ss = 2 if antialias else 1
    rh, rw = h * ss, w * ss

    objs_hi = ([ObjectState(o.cx * ss, o.cy * ss, o.half * ss, o.angle, o.n_sides)
                for o in objects]
               if ss > 1 else objects)

    # Background
    image = _make_texture_layer(rh, rw, bg_color, texture_strength,
                                seed=bg_seed, freq_mode=freq_mode)

    label_map_hi = np.zeros((rh, rw), dtype=np.int32)
    fill_masks_hi: List[np.ndarray] = []

    # Render objects back-to-front (later = closer)
    for i, (obj, color, seed) in enumerate(zip(objs_hi, obj_colors, obj_seeds)):
        fill = obj.get_fill_mask(rh, rw)
        fill_masks_hi.append(fill)
        label_map_hi[fill > 0] = i + 1
        tex   = _make_texture_layer(rh, rw, color, texture_strength,
                                    seed=seed, freq_mode=freq_mode)
        alpha = fill[..., None].astype(np.float32) / 255.0
        image = (tex * alpha + image * (1 - alpha)).astype(np.uint8)

    # Gradient lighting
    if lighting_gradient > 0:
        ramp  = np.linspace(1 - lighting_gradient, 1 + lighting_gradient, rw,
                            dtype=np.float32)[None, :, None]
        image = np.clip(image.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

    # Cast shadows and local lighting applied at render resolution
    if (cast_shadows or local_lighting) and light_pos is not None:
        combined_fg = np.zeros((rh, rw), dtype=np.uint8)
        for m in fill_masks_hi:
            combined_fg |= m
        if cast_shadows:
            image = _cast_shadow(image, fill_masks_hi, light_pos,
                                 combined_fg, strength=0.4, blur_sigma=3.0 * ss)
        if local_lighting:
            image = _apply_local_lighting(image, light_pos, strength=light_strength)

    # Edge mask at render resolution
    edge_hi = _edge_from_label_map(label_map_hi)

    # Downsample
    if ss > 1:
        image     = cv2.resize(image,      (w, h), interpolation=cv2.INTER_AREA)
        edge_mask = cv2.resize(edge_hi,    (w, h), interpolation=cv2.INTER_AREA)
        label_map = cv2.resize(label_map_hi, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        edge_mask = edge_hi
        label_map = label_map_hi

    # Gaussian pixel noise
    if gaussian_noise > 0:
        noise = np.random.randn(*image.shape).astype(np.float32) * gaussian_noise * 255
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Motion blur
    if motion_blur_kernel is not None:
        image = cv2.filter2D(image.astype(np.float32), -1, motion_blur_kernel)
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Soft label
    if soft_label_sigma > 0:
        edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), soft_label_sigma)
        mx = edge_mask.max()
        if mx > 1e-6:
            edge_mask = np.clip(edge_mask / mx, 0.0, 1.0)

    return image, edge_mask, label_map


def _render_frame(h: int, w: int,
                  state: ObjectState,
                  bg_color: np.ndarray,
                  plate_color: np.ndarray,
                  texture_strength: float,
                  bg_seed: int,
                  plate_seed: int,
                  lighting_gradient: float = 0.0,
                  antialias: bool = False,
                  gaussian_noise: float = 0.0,
                  freq_mode: str = "low",
                  soft_label_sigma: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy wrapper: single-object scene → (image, edge_mask)."""
    img, edge, _ = _render_scene(
        h, w,
        objects          = [state],
        obj_colors       = [plate_color],
        bg_color         = bg_color,
        texture_strength = texture_strength,
        bg_seed          = bg_seed,
        obj_seeds        = [plate_seed],
        lighting_gradient = lighting_gradient,
        antialias        = antialias,
        gaussian_noise   = gaussian_noise,
        freq_mode        = freq_mode,
        soft_label_sigma = soft_label_sigma,
    )
    return img, edge


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SyntheticEdgeDataset:
    """
    Generates (image, edge_mask) single-frame samples  — for Path 1.
    Set with_pairs=True for frame-pair + flow samples  — for Path 2.

    Parameters
    ----------
    size             : image size (square)
    texture_strength : 0 = flat colour, 1 = heavy texture
    with_pairs       : include frame_t+1 and ground-truth flow
    max_shift        : max translation between frames (pixels)
    max_rot          : max rotation between frames (degrees)
    antialias        : 2× supersample then INTER_AREA downsample
    gaussian_noise   : std of additive pixel noise in [0, 1]
    freq_mode        : "low" | "high" texture frequency
    soft_label_sigma : Gaussian blur sigma for soft edge labels (0 = binary)
    n_objects_range  : (min, max) number of objects per scene
    shape_types      : list of allowed shapes — "quad" | "triangle" | "pentagon"
                       | "hexagon" | "circle"
    motion_blur      : apply directional blur to frame_t+1 (and camera-shake
                       blur to single frames)
    local_lighting   : random point-light with radial falloff
    cast_shadows     : simplified projected shadow on background
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
        # ── augmentation ────────────────────────────────────────────────────
        antialias:         bool  = False,
        gaussian_noise:    float = 0.0,
        freq_mode:         str   = "low",
        soft_label_sigma:  float = 0.0,
        # ── scene complexity ────────────────────────────────────────────────
        n_objects_range:   Tuple[int, int] = (1, 1),
        shape_types:       Union[List[str], Tuple[str, ...]] = ("quad",),
        motion_blur:       bool  = False,
        local_lighting:    bool  = False,
        cast_shadows:      bool  = False,
    ):
        self.size             = size
        self.texture_strength = texture_strength
        self.with_pairs       = with_pairs
        self.max_shift        = max_shift
        self.max_rot          = max_rot
        self.length           = length
        self.rng              = random.Random(seed)
        self.np_rng           = np.random.RandomState(seed)
        self.antialias        = antialias
        self.gaussian_noise   = gaussian_noise
        self.freq_mode        = freq_mode
        self.soft_label_sigma = soft_label_sigma
        self.n_objects_range  = n_objects_range
        self.shape_types      = list(shape_types)
        self.motion_blur      = motion_blur
        self.local_lighting   = local_lighting
        self.cast_shadows     = cast_shadows

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        rng = self.rng
        sz  = self.size
        n_min, n_max = self.n_objects_range
        n_obj = rng.randint(n_min, n_max)

        # Background colour
        bg_color = np.array([rng.randint(30, 200)] * 3, dtype=np.uint8)

        # Generate objects
        objects:    List[ObjectState] = []
        obj_colors: List[np.ndarray] = []
        obj_seeds:  List[int] = []

        half_max = sz * (0.35 if n_obj == 1 else 0.28)

        for i in range(n_obj):
            shape   = rng.choice(self.shape_types)
            n_sides = ObjectState.SHAPE_N_SIDES.get(shape, 4)

            # Distinct colour from background and existing objects
            for _ in range(25):
                color = np.array([rng.randint(30, 200)] * 3, dtype=np.uint8)
                diffs = ([abs(int(bg_color[0]) - int(color[0]))] +
                         [abs(int(c[0]) - int(color[0])) for c in obj_colors])
                if all(d >= 25 for d in diffs):
                    break

            half   = rng.uniform(sz * 0.12, half_max)
            margin = half + 2
            cx     = rng.uniform(margin, sz - margin)
            cy     = rng.uniform(margin, sz - margin)
            angle  = rng.uniform(0, 360)

            objects.append(ObjectState(cx, cy, half, angle, n_sides))
            obj_colors.append(color)
            obj_seeds.append(idx * 7 + 3 + i * 2)

        bg_seed   = idx * 7 + 1
        light_pos = ((rng.random(), rng.random())
                     if (self.local_lighting or self.cast_shadows) else None)

        # Single-frame camera-shake blur
        mb_kernel_t = None
        if self.motion_blur and not self.with_pairs:
            a = rng.uniform(0, 2 * math.pi)
            length = rng.uniform(2, 5)
            mb_kernel_t = _make_motion_blur_kernel(
                length * math.cos(a), length * math.sin(a), min_len=2, max_len=6)

        img_t, edge_t, label_map_t = _render_scene(
            sz, sz, objects, obj_colors, bg_color,
            self.texture_strength, bg_seed, obj_seeds,
            lighting_gradient  = rng.uniform(0, 0.15),
            local_lighting     = self.local_lighting,
            light_pos          = light_pos,
            light_strength     = rng.uniform(0.2, 0.5),
            cast_shadows       = self.cast_shadows,
            antialias          = self.antialias,
            gaussian_noise     = self.gaussian_noise,
            freq_mode          = self.freq_mode,
            soft_label_sigma   = self.soft_label_sigma,
            motion_blur_kernel = mb_kernel_t,
        )

        sample: Dict[str, np.ndarray] = {
            "image":     _to_float(img_t),
            "edge_mask": edge_t[None],
        }

        if self.with_pairs:
            dx     = rng.uniform(-self.max_shift, self.max_shift)
            dy     = rng.uniform(-self.max_shift, self.max_shift)
            dangle = rng.uniform(-self.max_rot,   self.max_rot)

            objects_t1  = [o.apply_transform(dx=dx, dy=dy, dangle=dangle) for o in objects]
            obj_seeds_t1 = [s + 1 for s in obj_seeds]

            mb_kernel_t1 = (_make_motion_blur_kernel(dx, dy, min_len=2, max_len=12)
                            if self.motion_blur else None)

            img_t1, edge_t1, _ = _render_scene(
                sz, sz, objects_t1, obj_colors, bg_color,
                self.texture_strength, bg_seed + 1, obj_seeds_t1,
                lighting_gradient  = rng.uniform(0, 0.15),
                local_lighting     = self.local_lighting,
                light_pos          = light_pos,
                light_strength     = rng.uniform(0.2, 0.5),
                cast_shadows       = self.cast_shadows,
                antialias          = self.antialias,
                gaussian_noise     = self.gaussian_noise,
                freq_mode          = self.freq_mode,
                soft_label_sigma   = self.soft_label_sigma,
                motion_blur_kernel = mb_kernel_t1,
            )

            flow = _compute_multi_object_flow(
                edge_t, label_map_t, objects, dx, dy, dangle)

            sample["image_t1"] = _to_float(img_t1)
            sample["edge_t1"]  = edge_t1[None]
            sample["flow"]     = flow

        return sample

    def get_batch(self, batch_size: int,
                  start_idx: int = 0) -> Dict[str, "torch.Tensor"]:
        import torch
        samples = [self[i + start_idx] for i in range(batch_size)]
        return {k: torch.from_numpy(np.stack([s[k] for s in samples]))
                for k in samples[0]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 → (3, H, W) float32 in [-1, 1]."""
    return (img.astype(np.float32).transpose(2, 0, 1) / 127.5) - 1.0


def _compute_boundary_flow(edge_mask: np.ndarray,
                            state: ObjectState,
                            dx: float, dy: float,
                            dangle: float) -> np.ndarray:
    """
    Legacy: single-object analytical flow.
    p' = R(dangle) @ (p - centre) + centre + (dx, dy)
    """
    h, w     = edge_mask.shape
    rad      = math.radians(dangle)
    cos, sin = math.cos(rad), math.sin(rad)
    cx, cy_  = state.cx, state.cy

    ys, xs = np.where(edge_mask > 0)
    px = xs.astype(np.float32) - cx
    py = ys.astype(np.float32) - cy_

    rx =  cos * px - sin * py + cx + dx - xs
    ry =  sin * px + cos * py + cy_ + dy - ys

    flow = np.zeros((2, h, w), dtype=np.float32)
    flow[0, ys, xs] = rx
    flow[1, ys, xs] = ry
    return flow


def _compute_multi_object_flow(edge_mask: np.ndarray,
                                label_map: np.ndarray,
                                objects: List[ObjectState],
                                dx: float, dy: float,
                                dangle: float) -> np.ndarray:
    """
    Multi-object analytical flow.
    Each object rotates around its own centre then translates by (dx, dy).
    """
    h, w     = edge_mask.shape
    rad      = math.radians(dangle)
    cos_r    = math.cos(rad)
    sin_r    = math.sin(rad)
    flow     = np.zeros((2, h, w), dtype=np.float32)

    ys, xs = np.where(edge_mask > 0.5)
    if len(ys) == 0:
        return flow

    labels = label_map[ys, xs]

    for obj_i, obj in enumerate(objects):
        sel = labels == (obj_i + 1)
        if not sel.any():
            continue
        px = xs[sel].astype(np.float32) - obj.cx
        py = ys[sel].astype(np.float32) - obj.cy
        flow[0, ys[sel], xs[sel]] = cos_r*px - sin_r*py + obj.cx + dx - xs[sel]
        flow[1, ys[sel], xs[sel]] = sin_r*px + cos_r*py + obj.cy + dy - ys[sel]

    return flow


# ---------------------------------------------------------------------------
# Quick visual sanity check  (python -m data.synthetic)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs("debug_samples", exist_ok=True)

    ds = SyntheticEdgeDataset(
        size=256, texture_strength=0.4, with_pairs=True,
        n_objects_range=(1, 3),
        shape_types=["quad", "triangle", "pentagon", "hexagon", "circle"],
        motion_blur=True,
        local_lighting=True,
        cast_shadows=True,
        soft_label_sigma=1.5,
    )

    for i in range(6):
        s   = ds[i]
        img = ((s["image"].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        edge_hard = (s["edge_mask"][0] > 0.5).astype(np.uint8) * 255
        edge_soft = (s["edge_mask"][0] * 255).astype(np.uint8)
        overlay   = img.copy()
        overlay[s["edge_mask"][0] > 0.3] = [255, 0, 0]

        if "image_t1" in s:
            img_t1 = ((s["image_t1"].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
            row = np.concatenate([img, overlay, img_t1], axis=1)
        else:
            row = np.concatenate([img, overlay], axis=1)

        cv2.imwrite(f"debug_samples/sample_{i:02d}.png",
                    cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        print(f"Sample {i}: image {s['image'].shape}  "
              f"edge nonzero={s['edge_mask'].sum():.0f}")

    print("Saved to debug_samples/")
