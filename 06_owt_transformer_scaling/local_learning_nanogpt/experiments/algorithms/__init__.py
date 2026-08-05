"""Per-algorithm experiment modules and public algorithm registry."""

from __future__ import annotations

from typing import Any

import torch

from local_learning_nanogpt.experiments.specs import AlgorithmSpec

from . import bp, lce, lrva_lce, msp, nce, pd_lc, plce
from .common import compute_eranks, evaluate_perplexity, run_probe


ALGORITHM_MODULES = {
    module.SPEC.name: module
    for module in (bp, nce, lce, pd_lc, plce, lrva_lce, msp)
}
ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {
    name: module.SPEC for name, module in ALGORITHM_MODULES.items()
}


def list_algorithms() -> list[str]:
    return list(ALGORITHM_REGISTRY)


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    if name not in ALGORITHM_REGISTRY:
        raise KeyError(f"Unknown algorithm: {name}")
    return ALGORITHM_REGISTRY[name]


def select_algorithms(names: list[str]) -> list[AlgorithmSpec]:
    return [get_algorithm_spec(name) for name in names]


def prepare_batch(
    algorithm: AlgorithmSpec,
    train_data,
    block_size: int,
    batch_size: int,
    device: str,
    model: torch.nn.Module,
) -> dict[str, Any]:
    return ALGORITHM_MODULES[algorithm.name].prepare_batch(
        train_data,
        block_size,
        batch_size,
        device,
        model,
    )


def train_step(
    algorithm: AlgorithmSpec,
    model: torch.nn.Module,
    optimizer_bundle: Any,
    batch: dict[str, Any],
) -> float:
    return ALGORITHM_MODULES[algorithm.name].train_step(model, optimizer_bundle, batch)


__all__ = [
    "ALGORITHM_REGISTRY",
    "compute_eranks",
    "evaluate_perplexity",
    "get_algorithm_spec",
    "list_algorithms",
    "prepare_batch",
    "run_probe",
    "select_algorithms",
    "train_step",
]
