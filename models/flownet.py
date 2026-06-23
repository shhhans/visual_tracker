"""
FlowNetC — Correlation variant of FlowNet (Dosovitskiy et al., 2015).

Architecture:
  Frame_t   ──► ContractingPath ──► feat_t  ──┐
  Frame_t+1 ──► ContractingPath ──► feat_t1 ──┤
                                               ▼
                                    CorrelationLayer (cost volume)
                                               │
                                    RefineStream (concat feat_t)
                                               │
                                    ExpandingPath (skip connections)
                                               │
                              multi-scale flow predictions
                              [flow/2, flow/4, flow/8, flow/16, flow/32]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def conv(in_c, out_c, k=3, s=1, p=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=bias),
        nn.LeakyReLU(0.1, inplace=True),
    )


def deconv(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True),
    )


def predict_flow(in_c):
    return nn.Conv2d(in_c, 2, 3, padding=1, bias=True)


# ---------------------------------------------------------------------------
# Correlation layer — pure PyTorch, no CUDA extension required
# ---------------------------------------------------------------------------

class CorrelationLayer(nn.Module):
    """
    Computes a local cost volume between two feature maps.

    For each position x in feat_t, correlates with all positions
    x+d in feat_{t+1} where d ∈ [-max_d, max_d]².

    Output channels = (2 * max_displacement // stride + 1)²
    """

    def __init__(self, max_displacement: int = 4, stride: int = 1):
        super().__init__()
        self.max_d  = max_displacement
        self.stride = stride

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat1.shape
        d  = self.max_d
        s  = self.stride

        # Normalise to unit vectors (stabilises training)
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)

        # Pad feat2 so we can shift it by ±d
        feat2_pad = F.pad(feat2, [d, d, d, d])

        K       = 2 * d // s + 1
        out     = feat1.new_zeros(B, K * K, H, W)
        idx     = 0
        for dy in range(-d, d + 1, s):
            for dx in range(-d, d + 1, s):
                f2_shift = feat2_pad[:, :, d + dy: d + dy + H, d + dx: d + dx + W]
                # dot product per spatial location, summed over channels
                out[:, idx] = (feat1 * f2_shift).sum(dim=1)
                idx += 1

        return out / C  # scale by channel count


# ---------------------------------------------------------------------------
# Shared encoder (ContractingPath)
# ---------------------------------------------------------------------------

class ContractingPath(nn.Module):
    """
    6-level encoder that processes a single RGB frame.
    Mirrors the FlowNet contracting path (conv1 → conv6).
    """

    def __init__(self):
        super().__init__()
        self.conv1  = conv(3,   64,  7, s=2, p=3)   # /2
        self.conv2  = conv(64,  128, 5, s=2, p=2)   # /4
        self.conv3  = conv(128, 256, 5, s=2, p=2)   # /8
        self.conv3_1 = conv(256, 256)
        self.conv4  = conv(256, 512, s=2)            # /16
        self.conv4_1 = conv(512, 512)
        self.conv5  = conv(512, 512, s=2)            # /32
        self.conv5_1 = conv(512, 512)
        self.conv6  = conv(512, 1024, s=2)           # /64
        self.conv6_1 = conv(1024, 1024)

    def forward(self, x: torch.Tensor):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3_1(self.conv3(c2))
        c4 = self.conv4_1(self.conv4(c3))
        c5 = self.conv5_1(self.conv5(c4))
        c6 = self.conv6_1(self.conv6(c5))
        return c2, c3, c4, c5, c6   # skip connection features


# ---------------------------------------------------------------------------
# Refinement stream after correlation
# ---------------------------------------------------------------------------

class RefineStream(nn.Module):
    """
    After correlation, fuse cost volume with feat_t features at /8 scale.
    """

    def __init__(self, corr_channels: int, feat_channels: int = 256):
        super().__init__()
        in_c = corr_channels + feat_channels
        self.net = nn.Sequential(
            conv(in_c,       256),
            conv(256,        256),
        )

    def forward(self, corr: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([corr, feat], dim=1))


# ---------------------------------------------------------------------------
# Expanding path (decoder) with multi-scale flow supervision
# ---------------------------------------------------------------------------

class ExpandingPath(nn.Module):
    """
    Top-down decoder.  At each level emits a flow prediction.
    Skip connections from the encoder are concatenated before each upsampling.

    Level channels (after skip concat):
      /8  : 256 (refine out)
      /4  : 256 + 128 skip + 2 (upsamp-flow) = 386
      /2  : 128 + 64 skip  + 2 = 194
      /1  : 64  + 2 = 66
    """

    def __init__(self, refine_channels: int = 256):
        super().__init__()
        # /8 → /4
        self.dc1     = deconv(refine_channels, 128)
        self.flow8   = predict_flow(refine_channels)
        self.up_flow8 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1, bias=False)

        # /4 → /2
        self.dc2     = deconv(128 + 128 + 2, 64)
        self.flow4   = predict_flow(128 + 128 + 2)
        self.up_flow4 = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1, bias=False)

        # /2 → /1
        self.dc3     = deconv(64 + 64 + 2, 32)
        self.flow2   = predict_flow(64 + 64 + 2)

    def forward(
        self,
        x: torch.Tensor,          # (B, 256, H/8, W/8) from RefineStream
        skip4: torch.Tensor,       # (B, 128, H/4, W/4)  c2
        skip2: torch.Tensor,       # (B, 64,  H/2, W/2)  c1
    ) -> List[torch.Tensor]:
        flows = []

        # /8 scale
        flow8 = self.flow8(x)
        flows.append(flow8)
        up8 = self.up_flow8(flow8)
        x   = self.dc1(x)

        # /4 scale
        x    = torch.cat([x, skip4, up8], dim=1)
        flow4 = self.flow4(x)
        flows.append(flow4)
        up4  = self.up_flow4(flow4)
        x    = self.dc2(x)

        # /2 scale
        x    = torch.cat([x, skip2, up4], dim=1)
        flow2 = self.flow2(x)
        flows.append(flow2)

        return flows   # [flow/8, flow/4, flow/2]  finest last


# ---------------------------------------------------------------------------
# FlowNetC — public API
# ---------------------------------------------------------------------------

class FlowNetC(nn.Module):
    """
    FlowNetC: Correlation-based optical flow estimation network.

    Input : two consecutive RGB frames, each (B, 3, H, W)
    Output: list of flow tensors at decreasing strides [/8, /4, /2]
            each flow tensor shape (B, 2, H_i, W_i) giving (dx, dy) in pixels

    At inference, call .infer(frame_t, frame_t1) to get the full-resolution
    flow upsampled bilinearly to (H, W).
    """

    def __init__(self, max_displacement: int = 4):
        super().__init__()
        self.encoder = ContractingPath()

        K = 2 * max_displacement + 1
        corr_channels = K * K

        self.corr    = CorrelationLayer(max_displacement=max_displacement)
        self.refine  = RefineStream(corr_channels, feat_channels=256)
        self.decoder = ExpandingPath(refine_channels=256)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        frame_t:  torch.Tensor,
        frame_t1: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            frame_t  : (B, 3, H, W) frame at time t
            frame_t1 : (B, 3, H, W) frame at time t+1

        Returns:
            flows: list of (B, 2, H_i, W_i) at strides [/8, /4, /2]
                   flows[i][:, 0] = dx,  flows[i][:, 1] = dy  (pixels at full res)
        """
        # Shared encoder — note: we only use c2, c3 skips from frame_t
        c2_t,  c3_t,  c4_t,  c5_t,  c6_t  = self.encoder(frame_t)
        c2_t1, c3_t1, c4_t1, c5_t1, c6_t1 = self.encoder(frame_t1)

        # Correlation at /8 scale (c3)
        corr = self.corr(c3_t, c3_t1)

        # Refine with feat_t at same scale
        refined = self.refine(corr, c3_t)

        # Decode with skip connections from frame_t encoder
        flows = self.decoder(refined, skip4=c2_t, skip2=c2_t)

        # Scale flow values: the network predicts flow in feature-map pixels,
        # multiply by stride to convert to input-image pixels
        strides = [8, 4, 2]
        flows = [f * s for f, s in zip(flows, strides)]

        return flows   # coarse → fine

    @torch.no_grad()
    def infer(
        self,
        frame_t:  torch.Tensor,
        frame_t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns full-resolution flow (B, 2, H, W) by bilinear upsampling.
        """
        H, W  = frame_t.shape[-2:]
        flows = self(frame_t, frame_t1)
        flow_fine = flows[-1]   # /2 resolution, finest
        flow_full = F.interpolate(flow_fine, size=(H, W), mode="bilinear", align_corners=False)
        flow_full[:, 0] *= (W / flow_fine.shape[-1])
        flow_full[:, 1] *= (H / flow_fine.shape[-2])
        return flow_full
