"""ResNet backbone with configurable depth. Outputs multi-scale feature maps."""
import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import List


class ResNetBackbone(nn.Module):
    """Extracts C2-C5 feature maps from a pretrained ResNet."""

    _configs = {
        18:  (tv_models.resnet18,  [64, 128, 256, 512]),
        34:  (tv_models.resnet34,  [64, 128, 256, 512]),
        50:  (tv_models.resnet50,  [256, 512, 1024, 2048]),
        101: (tv_models.resnet101, [256, 512, 1024, 2048]),
    }

    def __init__(self, depth: int = 50, pretrained: bool = True, out_indices: List[int] = (1, 2, 3, 4)):
        super().__init__()
        if depth not in self._configs:
            raise ValueError(f"Unsupported depth {depth}. Choose from {list(self._configs)}")

        factory, self.out_channels = self._configs[depth]
        self.out_channels = [self.out_channels[i - 1] for i in out_indices]
        self.out_indices = out_indices

        weights = "DEFAULT" if pretrained else None
        base = factory(weights=weights)

        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1  # stride 4  → C2
        self.layer2 = base.layer2  # stride 8  → C3
        self.layer3 = base.layer3  # stride 16 → C4
        self.layer4 = base.layer4  # stride 32 → C5

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        x = self.layer0(x)
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=1):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs
