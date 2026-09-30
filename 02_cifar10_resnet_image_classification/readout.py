"""Readout heads and feature utilities for detached classifier experiments."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import STAGE_CHANNELS


READOUT_MODES = ("last", "all")


def pool_features(features: List[torch.Tensor], mode: str = "last") -> torch.Tensor:
    if mode not in READOUT_MODES:
        raise ValueError(f"Unknown readout mode {mode!r}; expected one of {READOUT_MODES}")
    chosen = [features[-1]] if mode == "last" else features
    return torch.cat([F.adaptive_avg_pool2d(h, 1).flatten(1) for h in chosen], dim=1)


def feature_dim(mode: str = "last") -> int:
    if mode == "last":
        return STAGE_CHANNELS[-1]
    if mode == "all":
        return sum(STAGE_CHANNELS)
    raise ValueError(f"Unknown readout mode {mode!r}; expected one of {READOUT_MODES}")


class LinearReadout(nn.Module):
    def __init__(self, mode: str, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.mode = mode
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(feature_dim(mode), num_classes)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return self.fc(self.dropout(pool_features(features, self.mode)))


def train_readout_epoch(trainer, readout: nn.Module, optimizer, loader, device, criterion) -> dict:
    trainer.modules_for_mode(False)
    readout.train()
    loss_sum = 0.0
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            features = [h.detach() for h in trainer.extract_features(x)]
        logits = readout(features)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.shape[0]
        correct += (logits.detach().argmax(1) == y).sum().item()
        total += y.shape[0]
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1)}


@torch.no_grad()
def evaluate_readout(trainer, readout: nn.Module, loader, device, criterion, max_batches=None) -> dict:
    trainer.modules_for_mode(False)
    readout.eval()
    loss_sum = 0.0
    correct = total = 0
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        features = [h.detach() for h in trainer.extract_features(x)]
        logits = readout(features)
        loss_sum += criterion(logits, y).item() * y.shape[0]
        correct += (logits.argmax(1) == y).sum().item()
        total += y.shape[0]
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1), "n": total}
