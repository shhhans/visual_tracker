"""Evaluate a trained detector on the validation set."""
import argparse
import yaml
import torch
from tqdm import tqdm

from models import Detector
from data import build_dataloader
from utils import MeanAveragePrecision


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",   required=True,       help="Config file path")
    p.add_argument("--ckpt",  required=True,       help="Checkpoint path")
    p.add_argument("--device", default="",         help="cuda | mps | cpu")
    p.add_argument("--conf",   default=0.05, type=float)
    p.add_argument("--iou",    default=0.6,  type=float)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    model = Detector(cfg).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_loader = build_dataloader(cfg, split="val")
    num_classes = cfg["model"]["head"]["num_classes"]
    metric = MeanAveragePrecision(num_classes)

    for images, targets in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        preds  = model.predict(images, conf_thresh=args.conf, nms_thresh=args.iou)
        metric.update(preds, targets)

    results = metric.compute()
    print(f"\nmAP@0.5: {results['mAP']:.4f}  (over {results['num_classes_with_gt']} classes)")


if __name__ == "__main__":
    main()
