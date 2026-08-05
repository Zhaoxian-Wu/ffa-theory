"""FFA-LCE algorithm with untied local CE heads."""

from __future__ import annotations

from typing import Any

import torch

from local_learning_nanogpt.experiments.specs import AlgorithmSpec

from .common import prepare_standard_batch


SPEC = AlgorithmSpec(
    name="lce",
    model_mode="ffa_lce_untied",
    description="FFA with untied local cross-entropy heads",
    supports_probe=False,
    supports_optimizers=("adam", "muon", "muon_paper"),
)


def prepare_batch(train_data, block_size: int, batch_size: int, device: str, model: torch.nn.Module):
    return prepare_standard_batch(train_data, block_size, batch_size, device)


def train_step(model: torch.nn.Module, optimizer_bundle: Any, batch: dict[str, Any]) -> float:
    layer_losses = model.forward_ffa_lce_untied(batch["x"], batch["y"])
    optimizer_bundle.zero_grad_embeddings()
    train_loss = 0.0
    for idx, layer_loss in enumerate(layer_losses):
        optimizer_bundle.zero_grad_unit(idx)
        layer_loss.backward()
        optimizer_bundle.step_unit(idx)
        train_loss += layer_loss.item()
    optimizer_bundle.step_embeddings()
    return train_loss / len(layer_losses)
