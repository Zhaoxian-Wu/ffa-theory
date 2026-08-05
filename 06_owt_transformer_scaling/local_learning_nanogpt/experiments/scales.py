"""Shared scale registry."""

from __future__ import annotations

from local_learning_nanogpt.experiments.specs import ScaleSpec


SCALE_REGISTRY: dict[str, ScaleSpec] = {
    "tiny": ScaleSpec("tiny", n_layer=2, n_embd=128, n_head=4, approx_params=7_000_000),
    "small": ScaleSpec("small", n_layer=4, n_embd=256, n_head=4, approx_params=16_000_000),
    "medium": ScaleSpec("medium", n_layer=6, n_embd=384, n_head=6, approx_params=30_000_000),
    "large": ScaleSpec("large", n_layer=8, n_embd=512, n_head=8, approx_params=51_000_000),
    "xlarge": ScaleSpec("xlarge", n_layer=12, n_embd=768, n_head=12, approx_params=124_000_000),
}


def get_scale(name: str) -> ScaleSpec:
    if name not in SCALE_REGISTRY:
        raise KeyError(f"Unknown scale: {name}")
    return SCALE_REGISTRY[name]


def list_scales() -> list[str]:
    return list(SCALE_REGISTRY)


def select_scales(names: list[str]) -> list[ScaleSpec]:
    return [get_scale(name) for name in names]

