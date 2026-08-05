"""Dataset registry and loaders."""

from __future__ import annotations

import os
import pickle

import numpy as np

from local_learning_nanogpt.experiments.specs import DatasetBundle, DatasetSpec
from local_learning_nanogpt.paths import resolve_owt_data_dir, resolve_shakespeare_data_dir


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "shakespeare": DatasetSpec(
        name="shakespeare",
        description="Shakespeare char-level baseline dataset",
        default_lr=1e-3,
        default_min_lr=1e-4,
        default_n_neg=64,
    ),
    "owt_small": DatasetSpec(
        name="owt_small",
        description="OpenWebText-small (50M tokens)",
        default_lr=6e-4,
        default_min_lr=6e-5,
        default_n_neg=128,
    ),
    "owt_full": DatasetSpec(
        name="owt_full",
        description="Full OpenWebText",
        default_lr=6e-4,
        default_min_lr=6e-5,
        default_n_neg=128,
    ),
}


def list_datasets() -> list[str]:
    return list(DATASET_REGISTRY)


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name]


def _resolve_data_dir(name: str) -> str:
    if name == "shakespeare":
        return resolve_shakespeare_data_dir()
    if name in {"owt_small", "owt_full"}:
        return resolve_owt_data_dir(name)
    raise KeyError(f"Unknown dataset: {name}")


def load_dataset_bundle(name: str) -> DatasetBundle:
    data_dir = _resolve_data_dir(name)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Dataset {name} is missing train.bin, val.bin, or meta.pkl under {data_dir}."
        )

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    return DatasetBundle(
        name=name,
        data_dir=data_dir,
        train_data=train_data,
        val_data=val_data,
        vocab_size=meta["vocab_size"],
    )

