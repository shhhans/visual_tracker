"""Training entry point."""
import os
import sys
import argparse
import yaml
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

from models import Detector
from losses import FCOSLoss
from data import build_dataloader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",    default="configs/default.yaml", help="Config file path")
    p.add_argument("--resume", default="",  help="Checkpoint to resume from")
    p.add_argument("--device", default="",  help="cuda device or 'cpu'")
    return p.parse_args()


def train(cfg, device, resume=""):
    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    train_loader = build_dataloader(cfg, split="train")
    model        = Detector(cfg).to(device)
    criterion    = FCOSLoss(cfg).to(device)

    optimizer = SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
        nesterov=True,
    )

    epochs    = cfg["train"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["train"]["lr"] * 0.01)

    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    log_interval = cfg["train"]["log_interval"]
    grad_clip    = cfg["train"]["grad_clip"]

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

        for step, (images, targets) in enumerate(bar):
            images = images.to(device)

            cls_preds, reg_preds, ctr_preds, fpn_feats = model(images)
            points = model.head.get_points(fpn_feats, device)

            loss_dict = criterion(cls_preds, reg_preds, ctr_preds, points, targets)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            if (step + 1) % log_interval == 0:
                avg = total_loss / (step + 1)
                bar.set_postfix(loss=f"{avg:.4f}", cls=f"{loss_dict['cls_loss'].item():.3f}",
                                reg=f"{loss_dict['reg_loss'].item():.3f}")

        scheduler.step()

        # Save checkpoint
        ckpt_path = os.path.join(cfg["train"]["save_dir"], f"epoch_{epoch+1:03d}.pt")
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg":       cfg,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


def main():
    args = parse_args()
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train(cfg, device, resume=args.resume)


if __name__ == "__main__":
    main()
