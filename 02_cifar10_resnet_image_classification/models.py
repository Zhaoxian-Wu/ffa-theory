"""CIFAR-style ResNets, split into a stem and four residual stages."""
from typing import List
import torch
from torch import nn
from torchvision.models import resnet18

STAGE_DEPTHS = {
    "resnet18": [2, 2, 2, 2],
    "resnet24": [3, 3, 3, 2],
    "resnet56": [7, 7, 7, 6],
    "resnet108": [13, 13, 13, 14],
}
STAGE_CHANNELS = [64, 64, 128, 256, 512]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


def _make_stage(in_planes: int, planes: int, n_blocks: int, stride: int) -> tuple[nn.Sequential, int]:
    layers = [BasicBlock(in_planes, planes, stride)]
    for _ in range(1, n_blocks):
        layers.append(BasicBlock(planes, planes, 1))
    return nn.Sequential(*layers), planes


def build_blocks(arch: str, in_channels: int = 3) -> List[nn.Module]:
    """Return [stem, stage1, stage2, stage3, stage4] for a CIFAR-style ResNet variant."""
    if arch not in STAGE_DEPTHS:
        raise ValueError(f"Unknown arch {arch!r}; expected one of {sorted(STAGE_DEPTHS)}")
    # Retain the original ResNet18 construction and initialization order.
    if arch == "resnet18":
        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        return [nn.Sequential(net.conv1, net.bn1, net.relu),
                net.layer1, net.layer2, net.layer3, net.layer4]
    stage_depths = STAGE_DEPTHS[arch]
    stem = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )
    in_planes = 64
    stage1, in_planes = _make_stage(in_planes, 64, stage_depths[0], stride=1)
    stage2, in_planes = _make_stage(in_planes, 128, stage_depths[1], stride=2)
    stage3, in_planes = _make_stage(in_planes, 256, stage_depths[2], stride=2)
    stage4, in_planes = _make_stage(in_planes, 512, stage_depths[3], stride=2)
    return [stem, stage1, stage2, stage3, stage4]


class LabelEmbedding(nn.Module):
    """Injects label as a learned extra spatial channel.

    Produces (B, 4, H, W) from (B, 3, H, W) input and (B,) labels.
    Used by: Vanilla FFA, SymBa, Layer Collab, Trifecta, SCFF.
    """

    def __init__(self, num_classes: int = 10, spatial: int = 32):
        super().__init__()
        self.emb = nn.Embedding(num_classes, spatial * spatial)
        self.spatial = spatial

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        e = self.emb(y).view(y.shape[0], 1, self.spatial, self.spatial)
        return torch.cat([x, e], dim=1)


class ResNetClassifier(nn.Module):
    def __init__(self, arch: str, num_classes: int):
        super().__init__()
        self.blocks = nn.Sequential(*build_blocks(arch, in_channels=3))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(STAGE_CHANNELS[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))
