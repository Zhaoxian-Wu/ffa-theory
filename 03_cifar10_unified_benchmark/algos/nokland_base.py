"""
Nokland & Eidnes (ICML 2019) "Training Neural Networks with Local Error Signals".

Shared utilities for the three Nokland variants (L_pred / L_sim / L_pred+sim).
This module intentionally does NOT register an algorithm -- run_benchmark.py
skips `nokland_base` during auto-import.

Architecture
------------
NoklandCNN is built from common.make_conv_blocks(arch), giving a per-arch
stack of ConvBlock(spec) that is structurally identical (channels, pools,
spatial flow) to CNN3Backbone / CNN6Backbone. Supported archs: cnn3, cnn6.

Each block's post-activation tensor feeds an auxiliary head; activations are
detached between blocks so gradients never cross layer boundaries.

Heads
-----
LPredHead:  conv(C, C, k=3, pad=1, bias=False) -> relu -> GAP(1) -> linear(C, 10)
            trained with CE against the true class label.
LSimHead:   placeholder for the L_sim variant (implemented in nokland_lsim.py).

Backward compatibility
----------------------
`BLOCK_CHANNELS` is exported and equals `block_channels("cnn3")` so Phase I
callers (auglocal, nokland_lpred) that imported the constant keep working.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ConvBlock,
    FAIR_DEFAULT_ARCH,
    FAIR_NUM_CLASSES,
    block_channels,
    make_conv_blocks,
)


# Back-compat: exported constant used by auglocal / nokland_lpred (Phase I code paths).
BLOCK_CHANNELS: List[int] = block_channels(FAIR_DEFAULT_ARCH)


class NoklandCNN(nn.Module):
    """Arch-parameterised CNN with inter-block gradient cut (for Nokland-family).

    `forward_block(ell, x_detached)` runs only block ell, assuming x_detached is
    the (already-detached) output of block ell-1 (or the raw input for ell=0).

    `forward_all_detached(x)` returns the list of post-activation tensors with
    .detach() applied at each inter-block boundary so each element is attached
    ONLY to its own block's parameters.
    """

    def __init__(self, arch: str = FAIR_DEFAULT_ARCH,
                 channels: Optional[List[int]] = None,
                 use_bn: bool = False):
        super().__init__()
        self.arch = arch
        self.use_bn = use_bn
        specs = make_conv_blocks(arch)
        # Optional channel override (rare; kept for tests). Overrides out_ch only.
        if channels is not None:
            assert len(channels) == len(specs), \
                f"channels override must match {len(specs)} blocks"
            new_specs = []
            prev_out = 3
            for s, out_ch in zip(specs, channels):
                new_specs.append({**s, "in_ch": prev_out, "out_ch": out_ch})
                prev_out = out_ch
            specs = new_specs
        self.blocks = nn.ModuleList([ConvBlock(s, use_bn=use_bn) for s in specs])
        self.channels = [s["out_ch"] for s in specs]
        self.n_blocks = len(self.blocks)

    def forward_block(self, ell: int, x_detached: torch.Tensor) -> torch.Tensor:
        """Run a single block. Caller must have already detached its input."""
        return self.blocks[ell](x_detached)

    def forward_all_detached(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return activations at each block, detaching between blocks."""
        h = x
        outs = []
        for block in self.blocks:
            h = block(h.detach())
            outs.append(h)
        return outs

    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = x
        outs = []
        for block in self.blocks:
            h = block(h)
            outs.append(h)
        return outs


# Back-compat alias so Phase I code keeps working.
NoklandCNN3 = NoklandCNN


class LPredHead(nn.Module):
    """L_pred auxiliary head.

    conv(C, C, k=3, pad=1, bias=False) -> relu -> AdaptiveAvgPool(1) -> flatten -> Linear(C, 10)
    Trained with cross-entropy against the true class label.
    """

    def __init__(self, channels: int, num_classes: int = FAIR_NUM_CLASSES):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.linear = nn.Linear(channels, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.conv(h))
        z = F.adaptive_avg_pool2d(z, 1).reshape(z.shape[0], -1)  # (B, C)
        return self.linear(z)


class LSimHead(nn.Module):
    """Placeholder for the L_sim similarity head.

    The actual L_sim loss is computed in nokland_lsim.py; this class exists so
    the base module can also be imported by the combined L_pred+sim variant.
    Structure follows Nokland 2019: conv -> GAP -> (optional) linear projection.
    """

    def __init__(self, channels: int, proj_dim: Optional[int] = None):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.proj = nn.Linear(channels, proj_dim) if proj_dim else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.conv(h))
        z = F.adaptive_avg_pool2d(z, 1).reshape(z.shape[0], -1)
        z = self.proj(z)
        # L2-normalize for cosine similarity (matching Nokland 2019 Section 3.1).
        return F.normalize(z, dim=1)
