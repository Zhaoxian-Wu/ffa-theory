"""Shared helpers for algorithm-specific experiment modules."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from local_learning_nanogpt.core.ffa_gpt import get_batch, get_batch_msp
from local_learning_nanogpt.experiments.specs import AlgorithmSpec


def prepare_standard_batch(
    train_data: np.memmap,
    block_size: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    x, y = get_batch(train_data, block_size, batch_size, device)
    return {"x": x, "y": y}


def prepare_msp_batch(
    train_data: np.memmap,
    block_size: int,
    batch_size: int,
    device: str,
    model: torch.nn.Module,
) -> dict[str, Any]:
    x_full = get_batch_msp(train_data, block_size, batch_size, model.K_max, device)
    return {
        "x": x_full[:, :block_size],
        "y": x_full[:, 1 : block_size + 1],
        "x_full": x_full,
    }


def evaluate_perplexity(
    model: torch.nn.Module,
    val_data: np.memmap,
    block_size: int,
    batch_size: int,
    device: str,
    num_batches: int,
) -> float:
    losses = []
    for _ in range(num_batches):
        vx, vy = get_batch(val_data, block_size, batch_size, device)
        losses.append(model.evaluate_perplexity(vx, vy))
    return math.exp(min(float(np.mean(losses)), 20.0))


def compute_eranks(
    model: torch.nn.Module,
    val_data: np.memmap,
    block_size: int,
    batch_size: int,
    device: str,
) -> list[float]:
    vx, _ = get_batch(val_data, block_size, batch_size, device)
    return model.compute_gram_erank(vx)


def run_probe(
    algorithm: AlgorithmSpec,
    model: torch.nn.Module,
    train_data: np.memmap,
    val_data: np.memmap,
    *,
    block_size: int,
    batch_size: int,
    device: str,
    probe_iters: int,
    eval_batches: int,
    start_step: int,
) -> tuple[float, list[dict[str, Any]]]:
    if probe_iters <= 0 or not algorithm.supports_probe:
        return float("inf"), []

    for block in model.blocks:
        for parameter in block.parameters():
            parameter.requires_grad = False
    for parameter in model.wpe.parameters():
        parameter.requires_grad = False

    probe_optimizer = torch.optim.AdamW(model.lm_head.parameters(), lr=1e-3, weight_decay=0.01)
    probe_entries: list[dict[str, Any]] = []
    best_probe_ppl = float("inf")

    for probe_step in range(probe_iters):
        model.train()
        px, py = get_batch(train_data, block_size, batch_size, device)
        _, probe_loss = model.forward_bp(px, py)
        probe_optimizer.zero_grad()
        probe_loss.backward()
        probe_optimizer.step()

        if probe_step % 200 == 0:
            model.eval()
            probe_ppl = evaluate_perplexity(
                model,
                val_data,
                block_size,
                batch_size,
                device,
                num_batches=eval_batches,
            )
            best_probe_ppl = min(best_probe_ppl, probe_ppl)
            probe_entries.append(
                {
                    "step": start_step + probe_step,
                    "ppl": float(probe_ppl),
                    "phase": "probe",
                }
            )
            print(f"    probe {probe_step:4d} | ppl {probe_ppl:.1f}")

    for block in model.blocks:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.wpe.parameters():
        parameter.requires_grad = True

    return best_probe_ppl, probe_entries
