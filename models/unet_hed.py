"""
HED-style UNet with Deep Supervision
=====================================
在标准 UNet 的每个 decoder 层加 side-output head，
训练时联合监督所有中间层输出（参考 HED, Xie & Tu 2015）。

Side outputs:
  side3  ← dec3 (bottleneck 上方，最粗粒度)
  side2  ← dec2
  side1  ← dec1 (最细粒度，接近最终输出)
  final  ← head (标准最终输出)

训练 loss:
  L = w_f * L_final + w_s * (L_side1 + L_side2 + L_side3)

默认 w_f=1.0, w_s=0.5，所有 loss 均为 focal BCE。

推理：只返回 final（或可选择融合）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def _side_head(in_c: int) -> nn.Conv2d:
    """1×1 conv → logit for side supervision."""
    return nn.Conv2d(in_c, 1, 1)


class HEDUNet(nn.Module):
    """
    Args:
        in_ch   : 输入通道数（RGB = 3）
        base_ch : 宽度乘子（16 = tiny, 32 = default）
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        b = base_ch

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = _block(in_ch, b)
        self.enc2 = _block(b,     b * 2)
        self.enc3 = _block(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = _block(b * 4, b * 8)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.up3  = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = _block(b * 8, b * 4)

        self.up2  = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = _block(b * 4, b * 2)

        self.up1  = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = _block(b * 2, b)

        # ── Final head ───────────────────────────────────────────────────────
        self.head = nn.Conv2d(b, 1, 1)

        # ── Side-output heads (deep supervision) ─────────────────────────────
        self.side3 = _side_head(b * 4)
        self.side2 = _side_head(b * 2)
        self.side1 = _side_head(b)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            training mode : (logit_final, logit_side1, logit_side2, logit_side3)
                            所有输出分辨率与输入相同（side outputs 已 upsample）
            eval mode     : logit_final  (shape: B×1×H×W)
        """
        H, W = x.shape[2], x.shape[3]

        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        bn = self.bottleneck(self.pool(e3))

        # Decode
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        logit_final = self.head(d1)

        if not self.training:
            return logit_final

        # Side outputs — upsample to input resolution
        s3 = F.interpolate(self.side3(d3), size=(H, W),
                           mode="bilinear", align_corners=False)
        s2 = F.interpolate(self.side2(d2), size=(H, W),
                           mode="bilinear", align_corners=False)
        s1 = self.side1(d1)   # already at input resolution

        return logit_final, s1, s2, s3

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
