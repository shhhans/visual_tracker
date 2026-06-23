"""
Lightweight U-Net shared by Path 1 and Path 2.

Default config (base_ch=32) has ~500k parameters — fast on CPU.

        in_ch
          │
   ┌──── enc1 (32) ────────────────────────────────────┐
   │      │ pool                                        │ skip
   │     enc2 (64) ───────────────────────────────┐    │
   │      │ pool                                   │ skip
   │     enc3 (128) ──────────────────────────┐   │    │
   │      │ pool                               │ skip   │
   │     bottleneck (256)                      │   │    │
   │      │ up                                 │   │    │
   │     dec3 (128) ←─ skip enc3 ─────────────┘   │    │
   │      │ up                                     │    │
   │     dec2 (64)  ←─ skip enc2 ─────────────────┘    │
   │      │ up                                          │
   └─►   dec1 (32)  ←─ skip enc1 ────────────────────--┘
          │
         head (out_ch)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def _block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    Args:
        in_ch   : input channels (3 for single frame, 6 for frame pair)
        out_ch  : output channels (1 = edge mask, 3 = edge + flow dx + flow dy)
        base_ch : width multiplier (16 = tiny, 32 = default, 64 = larger)
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base_ch: int = 32):
        super().__init__()
        b = base_ch

        # Encoder
        self.enc1 = _block(in_ch, b)
        self.enc2 = _block(b,     b * 2)
        self.enc3 = _block(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _block(b * 4, b * 8)

        # Decoder
        self.up3  = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = _block(b * 8, b * 4)   # after concat with skip

        self.up2  = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = _block(b * 4, b * 2)

        self.up1  = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = _block(b * 2, b)

        # Output head — raw logits (no activation here, applied in loss/decode)
        self.head = nn.Conv2d(b, out_ch, 1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b  = self.bottleneck(self.pool(e3))

        # Decode with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
