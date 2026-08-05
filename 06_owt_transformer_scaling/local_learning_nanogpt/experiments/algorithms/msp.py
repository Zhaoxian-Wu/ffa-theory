"""FFA-MSP algorithm."""

from __future__ import annotations

from typing import Any

import torch

from local_learning_nanogpt.experiments.specs import AlgorithmSpec

from .common import prepare_msp_batch


SPEC = AlgorithmSpec(
    name="msp",
    model_mode="ffa_msp",
    description="FFA with multi-scale predictive goodness",
    supports_probe=True,
    needs_future_tokens=True,
    supports_optimizers=("adam",),
)


def prepare_batch(train_data, block_size: int, batch_size: int, device: str, model: torch.nn.Module):
    return prepare_msp_batch(train_data, block_size, batch_size, device, model)


def train_step(model: torch.nn.Module, optimizer_bundle: Any, batch: dict[str, Any]) -> float:
    layer_losses = model.forward_ffa_msp(batch["x_full"])
    optimizer_bundle.zero_grad_embeddings()
    train_loss = 0.0
    for idx, layer_loss in enumerate(layer_losses):
        optimizer_bundle.zero_grad_unit(idx)
        layer_loss.backward()
        optimizer_bundle.step_unit(idx)
        train_loss += layer_loss.item()
    optimizer_bundle.step_embeddings()
    return train_loss / len(layer_losses)
