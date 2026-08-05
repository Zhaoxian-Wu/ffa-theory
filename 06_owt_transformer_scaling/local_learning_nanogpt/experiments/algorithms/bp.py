"""Backpropagation baseline algorithm."""

from __future__ import annotations

from typing import Any

import torch

from local_learning_nanogpt.experiments.specs import AlgorithmSpec

from .common import prepare_standard_batch


SPEC = AlgorithmSpec(
    name="bp",
    model_mode="bp",
    description="Backpropagation baseline",
    supports_probe=False,
    supports_optimizers=("adam", "muon", "muon_paper"),
)


def prepare_batch(train_data, block_size: int, batch_size: int, device: str, model: torch.nn.Module):
    return prepare_standard_batch(train_data, block_size, batch_size, device)


def train_step(model: torch.nn.Module, optimizer_bundle: Any, batch: dict[str, Any]) -> float:
    _, loss = model.forward_bp(batch["x"], batch["y"])
    optimizer_bundle.zero_grad_bp()
    loss.backward()
    optimizer_bundle.step_bp(model)
    return loss.item()
