"""
两种改进架构：

ArchA  (Serial / Cascade)
─────────────────────────
  frame_t   ──► Path1(frozen) ──► hm_t ──┐
                                           ├─► FlowNet(7ch→2ch) ──► flow
  frame_t+1 ──► Path1(frozen) ──► hm_t1  │
  [frame_t, frame_t+1, hm_t] ────────────┘

  Path1 已训练好，冻结；只训练 FlowNet。
  hm_t 作为 soft attention mask 引导网络聚焦边缘区域。

ArchB  (Dual-Head U-Net)
─────────────────────────
  [frame_t, frame_t+1]  (6ch)
           │
     U-Net encoder + decoder   (共享权重)
           │
     ┌─────┴─────┐
  edge_head    flow_head
  Conv(b,1,1)  Conv(b,2,1)   ← 独立参数，梯度互不干扰
  边缘 logit   (dx, dy)
"""

import torch
import torch.nn as nn
from .unet import UNet


# ─────────────────────────────────────────────────────────────────────────────
# Arch A
# ─────────────────────────────────────────────────────────────────────────────

class ArchA(nn.Module):
    """
    Serial (cascade) architecture.

    path1   : 已训练好的 UNet(in=3, out=1)，权重冻结
    flow_net: 可训练的 UNet(in=7, out=2)
              输入通道：frame_t(3) + frame_t+1(3) + hm_t(1) = 7
              输出：(dx, dy) per pixel
    """

    def __init__(self, path1_ckpt: str, base_ch: int = 16):
        super().__init__()

        # ── Stage 1: frozen Path1 edge detector ──────────────────────────────
        self.path1 = UNet(in_ch=3, out_ch=1, base_ch=base_ch)
        ck = torch.load(path1_ckpt, map_location="cpu", weights_only=False)
        self.path1.load_state_dict(ck["model"])
        for p in self.path1.parameters():
            p.requires_grad_(False)
        self.path1.eval()

        # ── Stage 2: trainable flow network ──────────────────────────────────
        # 7 input channels: frame_t(3) + frame_t1(3) + hm_t(1)
        self.flow_net = UNet(in_ch=7, out_ch=2, base_ch=base_ch)

    def forward(self, frame_t: torch.Tensor, frame_t1: torch.Tensor):
        """
        Returns:
            hm_t   : (B, 1, H, W) sigmoid edge probability from Path1
            flow   : (B, 2, H, W) predicted (dx, dy)
        """
        with torch.no_grad():
            hm_t = torch.sigmoid(self.path1(frame_t))   # (B,1,H,W) frozen

        x    = torch.cat([frame_t, frame_t1, hm_t], dim=1)   # (B,7,H,W)
        flow = self.flow_net(x)                                # (B,2,H,W)
        return hm_t, flow

    def trainable_params(self):
        return self.flow_net.parameters()


# ─────────────────────────────────────────────────────────────────────────────
# Arch B
# ─────────────────────────────────────────────────────────────────────────────

def _conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


class DualHeadUNet(nn.Module):
    """
    Dual-head U-Net for joint edge detection + flow estimation.

    Shared encoder + decoder; two independent 1×1 conv heads at the output.
    This isolates the gradient paths:
      ∂L_edge / ∂(shared weights)  and  ∂L_flow / ∂(shared weights)
    are summed in the backbone, but each head's own parameters receive
    only its own gradient signal.
    """

    def __init__(self, in_ch: int = 6, base_ch: int = 16):
        super().__init__()
        b = base_ch

        # Encoder
        self.enc1   = _conv_block(in_ch, b)
        self.enc2   = _conv_block(b,     b*2)
        self.enc3   = _conv_block(b*2,   b*4)
        self.pool   = nn.MaxPool2d(2)
        self.bottle = _conv_block(b*4,   b*8)

        # Decoder
        self.up3  = nn.ConvTranspose2d(b*8, b*4, 2, stride=2)
        self.dec3 = _conv_block(b*8, b*4)
        self.up2  = nn.ConvTranspose2d(b*4, b*2, 2, stride=2)
        self.dec2 = _conv_block(b*4, b*2)
        self.up1  = nn.ConvTranspose2d(b*2, b,   2, stride=2)
        self.dec1 = _conv_block(b*2, b)

        # ── Two independent heads ──────────────────────────────────────────
        self.edge_head = nn.Conv2d(b, 1, 1)   # → edge logit
        self.flow_head = nn.Conv2d(b, 2, 1)   # → (dx, dy)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x : (B, 6, H, W)  [frame_t | frame_t+1]
        Returns:
            edge_logit : (B, 1, H, W)
            flow       : (B, 2, H, W)
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottle(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.edge_head(d1), self.flow_head(d1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
