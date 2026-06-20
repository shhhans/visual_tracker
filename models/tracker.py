"""
Two trackers:

ByteTracker      — original bbox-IoU + Kalman tracker (kept for reference).
FlowPointTracker — feature-point tracker driven by optical flow (FlowNetC).

FlowPointTracker pipeline per frame pair (t → t+1):
  1. Detect keypoints in frame_t  → positions {p_i}
  2. FlowNet(frame_t, frame_t+1)  → dense flow field F (dx, dy per pixel)
  3. Propagate: p_i^{t+1} = p_i^t + F(p_i^t)   (bilinear sampling of F)
  4. Detect keypoints in frame_t+1 → new positions {q_j}
  5. Match {propagated p_i} ↔ {q_j} by Euclidean distance (Hungarian)
  6. Update confirmed tracks; init new tracks from unmatched q_j
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from typing import List, Dict, Optional, Tuple


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute IoU between N boxes a and M boxes b. Returns (N, M) matrix."""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])

    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


class Track:
    _id_counter = 0

    def __init__(self, box: np.ndarray, score: float, label: int):
        Track._id_counter += 1
        self.id = Track._id_counter
        self.label = label
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.is_confirmed = False

        self.kf = self._init_kalman(box)

    @staticmethod
    def _init_kalman(box: np.ndarray) -> KalmanFilter:
        """State: [cx, cy, s, r, dcx, dcy, ds] where s=area, r=aspect ratio."""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.eye(7)
        for i in range(4):
            kf.F[i, i + 3] = 1.0

        kf.H = np.zeros((4, 7))
        kf.H[:4, :4] = np.eye(4)

        kf.R *= 10.0
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01

        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        s  = (box[2] - box[0]) * (box[3] - box[1])
        r  = (box[2] - box[0]) / (box[3] - box[1] + 1e-6)
        kf.x[:4] = np.array([[cx], [cy], [s], [r]])
        return kf

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, box: np.ndarray, score: float):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        s  = (box[2] - box[0]) * (box[3] - box[1])
        r  = (box[2] - box[0]) / (box[3] - box[1] + 1e-6)
        self.kf.update(np.array([[cx], [cy], [s], [r]]))
        self.hits += 1
        self.time_since_update = 0

    @property
    def box(self) -> np.ndarray:
        s = self.kf.x
        cx, cy, area, r = s[0, 0], s[1, 0], s[2, 0], s[3, 0]
        w = np.sqrt(max(area * r, 1.0))
        h = max(area / (w + 1e-6), 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class ByteTracker:
    """
    Simplified ByteTrack:
    1. High-conf detections → match with existing tracks via IoU.
    2. Low-conf detections  → match with unmatched tracks.
    3. Unmatched tracks age out after max_age frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, min_hits: int = 3):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        Track._id_counter = 0

    def update(self, detections: Dict) -> List[Dict]:
        """
        detections: dict with keys boxes (N,4), scores (N,), labels (N,)
        Returns list of active track dicts with keys: id, box, label, score.
        """
        boxes  = detections["boxes"].cpu().numpy()   if hasattr(detections["boxes"],  "cpu") else detections["boxes"]
        scores = detections["scores"].cpu().numpy()  if hasattr(detections["scores"], "cpu") else detections["scores"]
        labels = detections["labels"].cpu().numpy()  if hasattr(detections["labels"], "cpu") else detections["labels"]

        # Kalman predict step
        for t in self.tracks:
            t.predict()

        high_mask = scores >= 0.5
        dets_high = (boxes[high_mask], scores[high_mask], labels[high_mask])
        dets_low  = (boxes[~high_mask], scores[~high_mask], labels[~high_mask])

        unmatched_tracks = self._match_and_update(dets_high, list(range(len(self.tracks))))
        _ = self._match_and_update(dets_low, unmatched_tracks)

        # Initialise new tracks from unmatched high-conf detections
        matched_det_ids = set()
        # (already handled inside _match_and_update via return value)
        for i, (box, score, label) in enumerate(zip(*dets_high)):
            # simple heuristic: if this det wasn't matched, start new track
            if not self._was_det_matched(box):
                self.tracks.append(Track(box, score, int(label)))

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Confirm tracks with enough hits
        for t in self.tracks:
            if t.hits >= self.min_hits:
                t.is_confirmed = True

        return [
            {"id": t.id, "box": t.box, "label": t.label, "score": t.hits / (t.age + 1)}
            for t in self.tracks if t.is_confirmed and t.time_since_update == 0
        ]

    def _match_and_update(self, dets, track_indices) -> List[int]:
        """Match detections to tracks. Returns unmatched track indices."""
        boxes, scores, labels = dets
        if len(boxes) == 0 or len(track_indices) == 0:
            return track_indices

        track_boxes = np.stack([self.tracks[i].box for i in track_indices])
        cost = 1.0 - _iou(track_boxes, boxes)
        row_ind, col_ind = linear_sum_assignment(cost)

        unmatched = list(track_indices)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 1.0 - self.iou_threshold:
                self.tracks[track_indices[r]].update(boxes[c], scores[c])
                unmatched.remove(track_indices[r])
        return unmatched

    def _was_det_matched(self, box: np.ndarray) -> bool:
        for t in self.tracks:
            if t.time_since_update == 0:
                iou = _iou(box[None], t.box[None])
                if iou > self.iou_threshold:
                    return True
        return False


# ---------------------------------------------------------------------------
# FlowPointTracker
# ---------------------------------------------------------------------------

class PointTrack:
    """Single tracked feature point."""
    _id_counter = 0

    def __init__(self, xy: np.ndarray, score: float, kp_type: int):
        PointTrack._id_counter += 1
        self.id    = PointTrack._id_counter
        self.xy    = xy.copy()          # (2,) float
        self.type  = kp_type
        self.hits  = 1
        self.age   = 0
        self.miss  = 0
        self.confirmed = False

    def update(self, xy: np.ndarray):
        self.xy   = xy.copy()
        self.hits += 1
        self.miss  = 0

    def mark_missed(self):
        self.miss += 1
        self.age  += 1


def _sample_flow_at_points(flow: torch.Tensor, points: np.ndarray) -> np.ndarray:
    """
    Bilinearly sample a flow field at (x, y) pixel positions.

    Args:
        flow   : (1, 2, H, W) flow tensor on some device
        points : (N, 2) array of (x, y) coordinates in pixel space

    Returns:
        sampled : (N, 2) array of (dx, dy) values
    """
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    _, _, H, W = flow.shape
    pts = torch.from_numpy(points).float().to(flow.device)

    # Normalise to [-1, 1] grid for grid_sample
    norm_x = (pts[:, 0] / (W - 1)) * 2 - 1
    norm_y = (pts[:, 1] / (H - 1)) * 2 - 1
    grid   = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,N,2)

    sampled = F.grid_sample(flow, grid, mode="bilinear", align_corners=True)  # (1,2,1,N)
    return sampled[0, :, 0, :].T.cpu().numpy()  # (N, 2)


def _distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix between point sets a (N,2) and b (M,2)."""
    diff = a[:, None, :] - b[None, :, :]   # (N, M, 2)
    return np.sqrt((diff ** 2).sum(-1))     # (N, M)


class FlowPointTracker:
    """
    Feature-point tracker that uses optical flow (FlowNetC) to propagate
    tracked keypoints across frames instead of IoU-based box matching.

    Usage (inference loop):
        tracker = FlowPointTracker(flownet, dist_threshold=8.0)
        for frame_t, frame_t1 in frame_pairs:
            kp_t1 = detector.predict_keypoints(frame_t1)
            tracks = tracker.update(frame_t, frame_t1, kp_t1)
    """

    def __init__(
        self,
        flownet,                      # FlowNetC instance (already on device)
        dist_threshold: float = 12.0, # max pixel distance to match a propagated point
        max_miss:       int   = 5,    # frames without match before dropping track
        min_hits:       int   = 2,    # hits before confirming track
    ):
        self.flownet        = flownet
        self.dist_threshold = dist_threshold
        self.max_miss       = max_miss
        self.min_hits       = min_hits
        self.tracks: List[PointTrack] = []
        PointTrack._id_counter = 0

    def update(
        self,
        frame_t:  torch.Tensor,   # (1, 3, H, W) normalised, on device
        frame_t1: torch.Tensor,   # (1, 3, H, W) normalised, on device
        detections_t1: dict,      # output of KeypointHead.decode for frame_t+1
                                  # {"coords": (N,2), "scores": (N,), "types": (N,)}
    ) -> List[dict]:
        """
        Args:
            frame_t        : current frame (tensor, already on model device)
            frame_t1       : next frame
            detections_t1  : keypoints detected in frame_t+1

        Returns:
            List of active track dicts:
                id     : int
                xy     : (2,) ndarray  (x, y) pixel coords in frame_t+1
                type   : int  keypoint type
                hits   : int
        """
        self.flownet.eval()
        with torch.no_grad():
            flow = self.flownet.infer(frame_t, frame_t1)  # (1, 2, H, W)

        det_xy    = detections_t1["coords"].cpu().numpy()   # (M, 2)
        det_types = detections_t1["types"].cpu().numpy()    # (M,)

        # --- Step 1: propagate existing tracks via flow ---
        if self.tracks:
            prev_xy  = np.stack([t.xy for t in self.tracks])    # (N, 2)
            deltas   = _sample_flow_at_points(flow, prev_xy)     # (N, 2)
            prop_xy  = prev_xy + deltas                           # (N, 2)
        else:
            prop_xy = np.empty((0, 2))

        # --- Step 2: match propagated points to new detections ---
        matched_track_ids = set()
        matched_det_ids   = set()

        if len(self.tracks) > 0 and len(det_xy) > 0:
            dist = _distance_matrix(prop_xy, det_xy)            # (N, M)
            row_ind, col_ind = linear_sum_assignment(dist)

            for r, c in zip(row_ind, col_ind):
                if dist[r, c] <= self.dist_threshold:
                    self.tracks[r].update(det_xy[c])
                    matched_track_ids.add(r)
                    matched_det_ids.add(c)

        # --- Step 3: mark unmatched tracks as missed ---
        for i, t in enumerate(self.tracks):
            if i not in matched_track_ids:
                t.mark_missed()

        # --- Step 4: initialise new tracks from unmatched detections ---
        for j in range(len(det_xy)):
            if j not in matched_det_ids:
                self.tracks.append(PointTrack(det_xy[j], 1.0, int(det_types[j])))

        # --- Step 5: confirm and prune ---
        for t in self.tracks:
            if t.hits >= self.min_hits:
                t.confirmed = True

        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]

        return [
            {"id": t.id, "xy": t.xy, "type": t.type, "hits": t.hits}
            for t in self.tracks if t.confirmed
        ]
