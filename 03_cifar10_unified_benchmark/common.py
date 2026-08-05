"""
CIFAR10-CNN Benchmark: Shared fair-comparison utilities.

Only the *common fairness contract* lives here:
  - CNN3Backbone / CNN6Backbone: two shared CNN backbones for the benchmark
  - make_conv_blocks(arch): block-spec factory used by every algorithm that
    builds its own per-layer stack (Belilovsky, SFF, Nokland, ...)
  - BLOCK_CHANNELS(arch): channel list convention
  - ConvBlock: a minimal Conv+ReLU(+pool) block matching the backbone layout
  - make_backbone(arch, out_dim): uniform interface used by BP/Vanilla-FFA
  - get_cifar10_loaders: unified data pipeline (no augmentation)
  - evaluate_classifier: standard argmax top-1 test accuracy
  - TrainResult dataclass + JSON schema (now includes `arch` field)
  - @register decorator + ALGO_REGISTRY

Algorithm-specific components (label overlay, goodness, FFA loss, FFA evaluation)
are defined inside each algos/*.py file — NOT here — because those are design
choices of each algorithm, not fairness constants.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


# ================================================================
# Fair-comparison constants (must not vary across algorithms)
# ================================================================

FAIR_BATCH_SIZE = 128
FAIR_NUM_CLASSES = 10
FAIR_LR = 1e-3          # default Adam learning rate
FAIR_FEAT_DIM = 256     # backbone output dimension
FAIR_DATA_DIR = "./data"
FAIR_NORMALIZE = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
FAIR_ARCHES = ("cnn3", "cnn6", "cnn9")  # supported architectures
FAIR_DEFAULT_ARCH = "cnn3"              # default if cfg lacks "arch"


# ================================================================
# Shared 3-layer CNN backbone (replicates experiments/cnn_vit_ffa_vs_bp.py:60-73)
# ================================================================

class CNN3Backbone(nn.Module):
    """The single shared 3-layer CNN every algorithm must use.

    Architecture (exact replication of cnn_vit_ffa_vs_bp.py:60-73):
      conv(3->32) -> relu -> maxpool(2)
      conv(32->64) -> relu -> maxpool(2)
      conv(64->128) -> relu -> adaptive_avgpool(4)
      flatten -> linear(2048 -> out_dim)

    Provides two interfaces:
      forward(x) -> Tensor (B, out_dim)   standard feature output
      forward_with_intermediates(x) -> List[Tensor]  post-activation at each of 3 conv blocks
    """

    def __init__(self, out_dim: int = FAIR_FEAT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(128 * 4 * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1, h2, h3 = self._conv_activations(x)
        z = self.avgpool(h3).reshape(h3.shape[0], -1)
        return self.fc(z)

    def forward_with_intermediates(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return post-activation tensors at each of the 3 conv blocks + final FC output."""
        h1, h2, h3 = self._conv_activations(x)
        z = self.avgpool(h3).reshape(h3.shape[0], -1)
        out = self.fc(z)
        return [h1, h2, h3, out]

    def _conv_activations(self, x: torch.Tensor):
        h1 = self.pool(torch.relu(self.conv1(x)))
        h2 = self.pool(torch.relu(self.conv2(h1)))
        h3 = torch.relu(self.conv3(h2))
        return h1, h2, h3


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ================================================================
# Shared 6-layer CNN backbone (depth extension for GIM/Layer-Collab tests)
# ================================================================

class CNN6Backbone(nn.Module):
    """Shared 6-layer CNN backbone. Preserves three spatial scales (32, 16, 8)
    so that GIM's patch-InfoNCE objective has a non-degenerate hierarchy.

    Channel flow:   3 -> 32 -> 32 -> 64 -> 64 -> 128 -> 128
    Spatial flow:   32x32 -> 32x32 -> 16x16 -> 16x16 -> 8x8 -> 8x8 -> 8x8 (avgpool->4x4)
    Pools:          none, max, none, max, none, none, avg(4)
    Feature dim:    128*4*4 = 2048 -> Linear -> out_dim=256 (matches CNN3)

    forward_with_intermediates(x) -> [h1, h2, h3, h4, h5, h6, out]
    """

    def __init__(self, out_dim: int = FAIR_FEAT_DIM):
        super().__init__()
        # conv layers following block_specs_cnn6 (see make_conv_blocks)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(128 * 4 * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h6 = self._conv_activations(x)[-1]
        z = self.avgpool(h6).reshape(h6.shape[0], -1)
        return self.fc(z)

    def forward_with_intermediates(self, x: torch.Tensor) -> List[torch.Tensor]:
        hs = self._conv_activations(x)
        z = self.avgpool(hs[-1]).reshape(hs[-1].shape[0], -1)
        out = self.fc(z)
        return list(hs) + [out]

    def _conv_activations(self, x: torch.Tensor):
        h1 = torch.relu(self.conv1(x))            # (B, 32, 32, 32)
        h2 = self.pool(torch.relu(self.conv2(h1)))  # (B, 32, 16, 16)
        h3 = torch.relu(self.conv3(h2))           # (B, 64, 16, 16)
        h4 = self.pool(torch.relu(self.conv4(h3)))  # (B, 64, 8, 8)
        h5 = torch.relu(self.conv5(h4))           # (B, 128, 8, 8)
        h6 = torch.relu(self.conv6(h5))           # (B, 128, 8, 8)
        return [h1, h2, h3, h4, h5, h6]


# ================================================================
# Block-spec factory (one source of truth for both backbones)
# ================================================================

def make_conv_blocks(arch: str) -> List[dict]:
    """Return a list of per-block specs for the given arch.

    Each spec is a dict with keys {in_ch, out_ch, kernel, padding, pool} where
    pool is one of:
      - "max":  2x2 MaxPool after the ReLU (halves spatial)
      - "avg4": AdaptiveAvgPool2d(4) after the ReLU (forces 4x4 output)
      - "none": no pool

    The exact ordering matches the corresponding Backbone class, so any
    algorithm that builds its own block stack from this spec list will be
    strictly consistent with BP/Vanilla-FFA under the same arch.
    """
    # Convention: per-block pools shape spatial DOWN between blocks; the final
    # global AdaptiveAvgPool(4) that precedes the FC head is NOT a per-block
    # operator (it belongs to the classifier head). Algorithms that need a
    # final GAP/AvgPool apply it themselves. This matches Phase-I CNN3Backbone
    # (which runs AvgPool(4) only inside fc() / forward_with_intermediates-out)
    # and Phase-II CNN6Backbone (which does the same).
    if arch == "cnn3":
        return [
            {"in_ch": 3,  "out_ch": 32,  "kernel": 3, "padding": 1, "pool": "max"},    # (32,16,16)
            {"in_ch": 32, "out_ch": 64,  "kernel": 3, "padding": 1, "pool": "max"},    # (64,8,8)
            {"in_ch": 64, "out_ch": 128, "kernel": 3, "padding": 1, "pool": "none"},   # (128,8,8)
        ]
    elif arch == "cnn6":
        return [
            {"in_ch": 3,   "out_ch": 32,  "kernel": 3, "padding": 1, "pool": "none"},  # (32,32,32)
            {"in_ch": 32,  "out_ch": 32,  "kernel": 3, "padding": 1, "pool": "max"},   # (32,16,16)
            {"in_ch": 32,  "out_ch": 64,  "kernel": 3, "padding": 1, "pool": "none"},  # (64,16,16)
            {"in_ch": 64,  "out_ch": 64,  "kernel": 3, "padding": 1, "pool": "max"},   # (64,8,8)
            {"in_ch": 64,  "out_ch": 128, "kernel": 3, "padding": 1, "pool": "none"},  # (128,8,8)
            {"in_ch": 128, "out_ch": 128, "kernel": 3, "padding": 1, "pool": "none"},  # (128,8,8)
        ]
    elif arch == "cnn9":
        # 9 conv blocks; 4 spatial scales preserved (32/16/8/4). Channels scale
        # with depth (32 -> 64 -> 128 -> 256) so capacity grows sub-linearly
        # with depth, keeping the final FC input (256*4*4 = 4096) only 2x that
        # of cnn3/cnn6. VGG-style: two or three convs per spatial scale.
        return [
            {"in_ch": 3,   "out_ch": 32,  "kernel": 3, "padding": 1, "pool": "none"},  # (32,32,32)
            {"in_ch": 32,  "out_ch": 32,  "kernel": 3, "padding": 1, "pool": "max"},   # (32,16,16)
            {"in_ch": 32,  "out_ch": 64,  "kernel": 3, "padding": 1, "pool": "none"},  # (64,16,16)
            {"in_ch": 64,  "out_ch": 64,  "kernel": 3, "padding": 1, "pool": "max"},   # (64,8,8)
            {"in_ch": 64,  "out_ch": 128, "kernel": 3, "padding": 1, "pool": "none"},  # (128,8,8)
            {"in_ch": 128, "out_ch": 128, "kernel": 3, "padding": 1, "pool": "max"},   # (128,4,4)
            {"in_ch": 128, "out_ch": 256, "kernel": 3, "padding": 1, "pool": "none"},  # (256,4,4)
            {"in_ch": 256, "out_ch": 256, "kernel": 3, "padding": 1, "pool": "none"},  # (256,4,4)
            {"in_ch": 256, "out_ch": 256, "kernel": 3, "padding": 1, "pool": "none"},  # (256,4,4)
        ]
    else:
        raise ValueError(f"Unknown arch '{arch}'. Supported: {FAIR_ARCHES}")


def block_channels(arch: str) -> List[int]:
    """Shortcut: list of out_ch per block (used by algos that only need channels)."""
    return [b["out_ch"] for b in make_conv_blocks(arch)]


def num_blocks(arch: str) -> int:
    return len(make_conv_blocks(arch))


def final_feat_channels(arch: str) -> int:
    """Last block's channel count (used for FC input and GAP head dim)."""
    return block_channels(arch)[-1]


def head_avgpool_size(arch: str) -> int:
    """Target spatial size of the classifier-head AdaptiveAvgPool2d.

    Both CNN3 and CNN6 use AvgPool(4) before the FC. Algorithms that need to
    build an equivalent classifier head (BP, Local-CE aux heads, DTP classifier,
    PC classifier, DF prototype layer, etc.) should use this helper rather
    than hard-coding 4.
    """
    return 4  # identical across cnn3 / cnn6 by design


def final_fc_in_dim(arch: str) -> int:
    """Input dimension of the final FC layer after AdaptiveAvgPool(head_avgpool_size)."""
    s = head_avgpool_size(arch)
    return final_feat_channels(arch) * s * s


class ChannelLayerNorm2d(nn.Module):
    """Apply LayerNorm over channels independently at every spatial location."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channels_last = inputs.permute(0, 2, 3, 1).contiguous()
        normalized = self.norm(channels_last)
        return normalized.permute(0, 3, 1, 2).contiguous()


class ConvBlock(nn.Module):
    """A single conv block matching the backbone layout: Conv [-> norm] -> ReLU -> pool.

    Used by algorithms that construct their own per-block stack (Belilovsky,
    SFF, Nokland, DLL, PC, DF, CC, DTP, etc.). Consistent with CNN3Backbone
    and CNN6Backbone so that any algo built from make_conv_blocks(arch) has
    the exact same spatial/channel layout as BP/Vanilla-FFA.

    Phase II.1 addition: `use_bn` optionally inserts a BatchNorm2d between
    the Conv and the ReLU to provide standard regularisation. Default is
    `False` for backward compatibility with the Phase II main-experiment
    JSONs (which were trained with use_bn=False). Algorithms that already
    carry their own normalisation (e.g., SFF's channel-only LN) should
    pass use_bn=False.
    """

    def __init__(self, spec: dict, use_bn: bool = False, use_channel_ln: bool = False):
        super().__init__()
        if use_bn and use_channel_ln:
            raise ValueError("ConvBlock cannot use BatchNorm and channel LayerNorm together.")
        self.conv = nn.Conv2d(spec["in_ch"], spec["out_ch"],
                              kernel_size=spec["kernel"], padding=spec["padding"])
        if use_bn:
            self.norm: nn.Module = nn.BatchNorm2d(spec["out_ch"])
        elif use_channel_ln:
            self.norm = ChannelLayerNorm2d(spec["out_ch"])
        else:
            self.norm = nn.Identity()
        self.pool_kind = spec["pool"]
        if self.pool_kind == "max":
            self.pool: Optional[nn.Module] = nn.MaxPool2d(2)
        elif self.pool_kind == "avg4":
            self.pool = nn.AdaptiveAvgPool2d(4)
        elif self.pool_kind == "none":
            self.pool = None
        else:
            raise ValueError(f"Unknown pool '{spec['pool']}'")
        self.out_ch = spec["out_ch"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.norm(self.conv(x)))
        if self.pool is not None:
            h = self.pool(h)
        return h


class CNN9Backbone(nn.Module):
    """Shared 9-layer CNN backbone. Four spatial scales (32/16/8/4) for
    algorithms (GIM, Layer Collaboration) that benefit from a richer patch /
    block hierarchy than CNN6 can provide.

    Channel flow:  3 -> 32 -> 32 -> 64 -> 64 -> 128 -> 128 -> 256 -> 256 -> 256
    Spatial flow:  32 -> 32 -> 16 -> 16 -> 8 -> 8 -> 4 -> 4 -> 4 -> 4 (avgpool)
    Pools:         none, max, none, max, none, max, none, none, none, avg(4)
    Feature dim:   256 * 4 * 4 = 4096 -> Linear -> out_dim=256

    forward_with_intermediates(x) -> [h1..h9, out]
    """

    def __init__(self, out_dim: int = FAIR_FEAT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(256 * 4 * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h9 = self._conv_activations(x)[-1]
        z = self.avgpool(h9).reshape(h9.shape[0], -1)
        return self.fc(z)

    def forward_with_intermediates(self, x: torch.Tensor) -> List[torch.Tensor]:
        hs = self._conv_activations(x)
        z = self.avgpool(hs[-1]).reshape(hs[-1].shape[0], -1)
        out = self.fc(z)
        return list(hs) + [out]

    def _conv_activations(self, x: torch.Tensor):
        h1 = torch.relu(self.conv1(x))               # (B, 32, 32, 32)
        h2 = self.pool(torch.relu(self.conv2(h1)))    # (B, 32, 16, 16)
        h3 = torch.relu(self.conv3(h2))              # (B, 64, 16, 16)
        h4 = self.pool(torch.relu(self.conv4(h3)))    # (B, 64, 8, 8)
        h5 = torch.relu(self.conv5(h4))              # (B, 128, 8, 8)
        h6 = self.pool(torch.relu(self.conv6(h5)))    # (B, 128, 4, 4)
        h7 = torch.relu(self.conv7(h6))              # (B, 256, 4, 4)
        h8 = torch.relu(self.conv8(h7))              # (B, 256, 4, 4)
        h9 = torch.relu(self.conv9(h8))              # (B, 256, 4, 4)
        return [h1, h2, h3, h4, h5, h6, h7, h8, h9]


def make_backbone(arch: str, out_dim: int = FAIR_FEAT_DIM) -> nn.Module:
    """Factory returning the correct Backbone instance."""
    if arch == "cnn3":
        return CNN3Backbone(out_dim=out_dim)
    elif arch == "cnn6":
        return CNN6Backbone(out_dim=out_dim)
    elif arch == "cnn9":
        return CNN9Backbone(out_dim=out_dim)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Supported: {FAIR_ARCHES}")


# ================================================================
# Data loading
# ================================================================

def get_cifar10_loaders(
    batch_size: int = FAIR_BATCH_SIZE,
    augment: bool = False,
    data_dir: str = FAIR_DATA_DIR,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Unified CIFAR-10 loaders. Returns image tensors (B, 3, 32, 32), not flattened.

    `augment` must stay False for the fair benchmark.
    """
    assert augment is False, "Fair benchmark forbids augmentation"
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(*FAIR_NORMALIZE),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


# ================================================================
# Evaluation utilities (standard argmax classifier — NOT FFA-specific)
# ================================================================

@torch.no_grad()
def evaluate_classifier(model_fn: Callable[[torch.Tensor], torch.Tensor],
                        loader: DataLoader,
                        device: torch.device) -> float:
    """Standard top-1 accuracy. `model_fn(x) -> logits (B, n_classes)`.

    Use this ONLY for algorithms that output logits directly (BP, Local-CE, DTP, etc).
    FFA-family algorithms must define their own label-overlay argmax evaluation
    inside their algos/*.py file.
    """
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model_fn(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    return correct / max(total, 1)


# ================================================================
# Result schema + registry
# ================================================================

@dataclass
class TrainResult:
    algo: str
    status: str                         # "ok", "na", "failed"
    arch: Optional[str] = None          # "cnn3" | "cnn6" (Phase II extension)
    test_acc_final: Optional[float] = None
    test_acc_best: Optional[float] = None
    test_acc_curve: List[float] = field(default_factory=list)
    train_loss_curve: List[float] = field(default_factory=list)
    n_params: Optional[int] = None
    elapsed_s: Optional[float] = None
    na_reason: Optional[str] = None
    hyperparams: Dict = field(default_factory=dict)
    gpu: Optional[str] = None
    timestamp: Optional[str] = None
    tag: Optional[str] = None
    error_msg: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


ALGO_REGISTRY: Dict[str, Callable[[dict], TrainResult]] = {}


def register(name: str) -> Callable:
    """Decorator to register an algorithm's train() function."""
    def _inner(fn: Callable[[dict], TrainResult]) -> Callable[[dict], TrainResult]:
        if name in ALGO_REGISTRY:
            raise ValueError(f"Algorithm '{name}' already registered")
        ALGO_REGISTRY[name] = fn
        return fn
    return _inner


def save_result(result: TrainResult, out_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{result.algo}.json"
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return str(path)


# ================================================================
# Seed control
# ================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ================================================================
# Recipe helpers (Phase II.1: out-of-contract regularisation recipe)
# ================================================================

# Defaults for the Phase-II.1 "fair-regularised recipe":
#   - (a) LR decay: multiply the learning rate by `RECIPE_LR_DECAY_FACTOR`
#         after `RECIPE_LR_DECAY_EPOCH` to curb late-epoch memorisation.
#   - (b) Normalisation: insert BN after each conv in ConvBlock. Algorithms
#         that already carry LayerNorm or BatchNorm internally keep their
#         native normalisation and do not stack an extra BN.
#
# Each algorithm's train(cfg) reads these keys from cfg:
#     cfg["lr_decay_epoch"]   -- epoch index (0-based) at which to decay
#     cfg["lr_decay_factor"]  -- new_lr = lr * factor (factor < 1)
#     cfg["use_bn"]           -- whether to pass use_bn=True to ConvBlock
# and should record the final values in TrainResult.hyperparams.

RECIPE_LR_DECAY_EPOCH = 100
RECIPE_LR_DECAY_FACTOR = 0.1


def apply_lr_decay(optimizers, epoch: int, lr_init: float,
                   at_epoch: int = RECIPE_LR_DECAY_EPOCH,
                   factor: float = RECIPE_LR_DECAY_FACTOR,
                   verbose: bool = False) -> bool:
    """Apply LR decay at the start of `at_epoch` (0-based).

    Args:
        optimizers: a single `torch.optim.Optimizer` or an iterable thereof.
        epoch: current 0-based epoch index.
        lr_init: initial learning rate.
        at_epoch: epoch at which to trigger the decay.
        factor: multiplicative factor; new lr = lr_init * factor.
        verbose: if True, print a one-liner at the decay step.

    Returns:
        True iff a decay was applied this call.
    """
    if at_epoch is None or factor is None or epoch != at_epoch:
        return False
    if hasattr(optimizers, "param_groups"):
        optimizers = [optimizers]
    new_lr = lr_init * factor
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = new_lr
    if verbose:
        print(f"  [recipe] lr_decay at epoch {epoch}: lr -> {new_lr:g}",
              flush=True)
    return True


# ================================================================
# NA-placeholder helper
# ================================================================

def na_result(algo: str, reason: str, arch: Optional[str] = None) -> TrainResult:
    return TrainResult(
        algo=algo, status="na", arch=arch, na_reason=reason,
        timestamp=now_iso(),
    )
