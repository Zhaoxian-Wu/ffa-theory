"""Structured specs for the parameterized experiment framework."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    n_layer: int
    n_embd: int
    n_head: int
    approx_params: int | None = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    default_lr: float
    default_min_lr: float
    default_batch_size: int = 32
    default_block_size: int = 256
    default_n_neg: int = 128


@dataclass
class DatasetBundle:
    name: str
    data_dir: str
    train_data: np.memmap
    val_data: np.memmap
    vocab_size: int


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    description: str


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    model_mode: str
    description: str
    supports_probe: bool = False
    needs_future_tokens: bool = False
    supports_optimizers: tuple[str, ...] = ("adam",)


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    description: str


@dataclass(frozen=True)
class ScalingPreset:
    name: str
    dataset: str
    profile: str
    algorithms: tuple[str, ...]
    optimizers: tuple[str, ...]
    scales: tuple[str, ...]
    output_filename: str
    default_max_iters: int | None = None
    default_probe_iters: int | None = None

