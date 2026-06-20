"""Run inference on a video or image directory with optional multi-object tracking."""
import argparse
import os
import cv2
import torch
import yaml
import numpy as np
from pathlib import Path

from models import Detector, ByteTracker
from utils import draw_detections, draw_tracks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",    required=True, help="Config file path")
    p.add_argument("--ckpt",   required=True, help="Checkpoint path")
    p.add_argument("--source", required=True, help="Video file or image directory")
    p.add_argument("--out",    default="output.mp4", help="Output video path")
    p.add_argument("--track",  action="store_true",  help="Enable ByteTracker")
    p.add_argument("--conf",   default=0.3, type=float)
    p.add_argument("--iou",    default=0.6, type=float)
    p.add_argument("--device", default="")
    return p.parse_args()


def preprocess(frame: np.ndarray, img_size: int, device) -> torch.Tensor:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w  = image.shape[:2]
    scale = img_size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    image  = cv2.resize(image, (nw, nh))
    pad    = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    pad[:nh, :nw] = image
    t = torch.from_numpy(pad.transpose(2, 0, 1)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    t = (t - mean) / std
    return t.unsqueeze(0).to(device), scale


def scale_boxes(boxes: torch.Tensor, scale: float) -> torch.Tensor:
    return boxes / scale


def main():
    args = parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}")

    model = Detector(cfg).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tracker = ByteTracker(**cfg["tracker"]) if args.track else None
    img_size = cfg["data"]["img_size"]

    src = args.source
    if os.path.isdir(src):
        frames = sorted(Path(src).glob("*.jpg")) + sorted(Path(src).glob("*.png"))
        cap = None
    else:
        cap = cv2.VideoCapture(src)
        frames = None

    fps  = cap.get(cv2.CAP_PROP_FPS) if cap else 30
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  if cap else 640
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap else 480
    out  = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def _iter_frames():
        if frames:
            for p in frames:
                yield cv2.imread(str(p))
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame

    for frame in _iter_frames():
        tensor, scale = preprocess(frame, img_size, device)
        dets = model.predict(tensor, conf_thresh=args.conf, nms_thresh=args.iou)[0]
        dets["boxes"] = scale_boxes(dets["boxes"], scale)

        if tracker:
            tracks = tracker.update(dets)
            vis = draw_tracks(frame, tracks)
        else:
            vis = draw_detections(frame, dets, conf_threshold=args.conf)

        out.write(vis)

    out.release()
    if cap:
        cap.release()
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
