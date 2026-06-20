"""Feature Pyramid Network (FPN) neck for multi-scale detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class FPN(nn.Module):
    """
    Classic top-down FPN (Lin et al., 2017).
    Optionally adds extra P6/P7 levels via stride-2 convolutions on P5.
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256, num_levels: int = 5):
        super().__init__()
        assert num_levels >= len(in_channels)

        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        )
        self.output_convs = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels
        )

        # Extra levels P6, P7 via downsampling
        self.extra_convs = nn.ModuleList()
        for i in range(num_levels - len(in_channels)):
            in_c = in_channels[-1] if i == 0 else out_channels
            self.extra_convs.append(nn.Conv2d(in_c, out_channels, 3, stride=2, padding=1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Build lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        # Extra levels
        src = features[-1]
        for conv in self.extra_convs:
            src = conv(src)
            outs.append(src)

        return outs
