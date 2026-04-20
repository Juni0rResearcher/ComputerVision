from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class FashionResNet18(nn.Module):
    """ResNet18 adapted for Fashion-MNIST (1x32x32, 10 classes)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        try:
            self.backbone = resnet18(weights=None)
        except TypeError:
            # Compatibility with older torchvision versions.
            self.backbone = resnet18(pretrained=False)

        # CIFAR/Fashion-MNIST style stem.
        self.backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate features (before final FC)."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.get_features(x)
        return self.backbone.fc(features)


def build_model(num_classes: int = 10) -> FashionResNet18:
    return FashionResNet18(num_classes=num_classes)
