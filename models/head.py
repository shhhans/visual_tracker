"""FCOS-style anchor-free detection head."""
import math
import torch
import torch.nn as nn
from typing import List, Tuple


class Scale(nn.Module):
    """Learnable scalar multiplier for each FPN level."""
    def __init__(self, init=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    """
    Anchor-free FCOS head.
    Predicts: class logits, centerness, and (l, t, r, b) distances per pixel.
    """

    def __init__(self, in_channels: int, num_classes: int, strides: List[int], num_convs: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        def _tower(in_c):
            layers = []
            for _ in range(num_convs):
                layers += [nn.Conv2d(in_c, in_c, 3, padding=1), nn.GroupNorm(32, in_c), nn.ReLU(inplace=True)]
                in_c = in_channels
            return nn.Sequential(*layers)

        self.cls_tower = _tower(in_channels)
        self.reg_tower = _tower(in_channels)

        self.cls_logits  = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred   = nn.Conv2d(in_channels, 4,           3, padding=1)
        self.centerness  = nn.Conv2d(in_channels, 1,           3, padding=1)
        self.scales      = nn.ModuleList(Scale(1.0) for _ in strides)

        self._init_weights()

    def _init_weights(self):
        prior = 0.01
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))

    def forward(self, features: List[torch.Tensor]) -> Tuple[List, List, List]:
        cls_preds, reg_preds, ctr_preds = [], [], []
        for feat, scale in zip(features, self.scales):
            cls_feat = self.cls_tower(feat)
            reg_feat = self.reg_tower(feat)

            cls_preds.append(self.cls_logits(cls_feat))
            ctr_preds.append(self.centerness(cls_feat))
            reg_preds.append(scale(self.bbox_pred(reg_feat)).exp())

        return cls_preds, reg_preds, ctr_preds

    @torch.no_grad()
    def get_points(self, features: List[torch.Tensor], device) -> List[torch.Tensor]:
        """Generate (x, y) grid points for each FPN level."""
        points = []
        for feat, stride in zip(features, self.strides):
            h, w = feat.shape[-2:]
            ys = torch.arange(0, h, device=device, dtype=torch.float32) * stride + stride // 2
            xs = torch.arange(0, w, device=device, dtype=torch.float32) * stride + stride // 2
            ys, xs = torch.meshgrid(ys, xs, indexing="ij")
            points.append(torch.stack([xs.flatten(), ys.flatten()], dim=-1))
        return points
