"""Draw detection boxes and tracking trails on images."""
import cv2
import numpy as np
from typing import Dict, List, Optional


_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10),  (146, 204, 23), (61, 219, 134),
    (26, 147, 52),  (0, 212, 187),  (44, 153, 168), (0, 194, 255),
    (52, 69, 147),  (100, 115, 255),(0, 24, 236),   (132, 56, 255),
    (82, 0, 133),   (203, 56, 255), (255, 149, 200),(255, 55, 199),
]


def _color(idx: int):
    return _PALETTE[idx % len(_PALETTE)]


def draw_detections(
    image: np.ndarray,
    detections: Dict,
    class_names: Optional[List[str]] = None,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    img = image.copy()
    boxes  = detections.get("boxes", [])
    scores = detections.get("scores", [])
    labels = detections.get("labels", [])

    for box, score, label in zip(boxes, scores, labels):
        if score < conf_threshold:
            continue
        label = int(label)
        x1, y1, x2, y2 = map(int, box)
        color = _color(label)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        name = class_names[label] if class_names else str(label)
        text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def draw_tracks(image: np.ndarray, tracks: List[Dict], class_names: Optional[List[str]] = None) -> np.ndarray:
    img = image.copy()
    for track in tracks:
        tid   = track["id"]
        box   = list(map(int, track["box"]))
        label = track.get("label", 0)
        x1, y1, x2, y2 = box
        color = _color(tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        name = class_names[int(label)] if class_names else str(label)
        text = f"#{tid} {name}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img
