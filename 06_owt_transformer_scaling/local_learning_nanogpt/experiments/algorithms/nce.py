"""FFA-NCE algorithm."""

from __future__ import annotations

from typing import Any

import torch

from local_learning_nanogpt.experiments.specs import AlgorithmSpec

from .common import prepare_standard_batch


SPEC = AlgorithmSpec(
    name="nce",
    model_mode="ffa_nce",
    description="FFA with NCE goodness",
    supports_probe=True,
    supports_optimizers=("adam",),
)


def prepare_batch(train_data, block_size: int, batch_size: int, device: str, model: torch.nn.Module):
    return prepare_standard_batch(train_data, block_size, batch_size, device)


def train_step(model: torch.nn.Module, optimizer_bundle: Any, batch: dict[str, Any]) -> float:
    layer_losses = model.forward_ffa_nce(batch["x"], batch["y"])
    optimizer_bundle.zero_grad_embeddings()
    train_loss = 0.0
    for idx, layer_loss in enumerate(layer_losses):
        optimizer_bundle.zero_grad_unit(idx)
        layer_loss.backward()
        optimizer_bundle.step_unit(idx)
        train_loss += layer_loss.item()
    optimizer_bundle.step_embeddings()
    return train_loss / len(layer_losses)
