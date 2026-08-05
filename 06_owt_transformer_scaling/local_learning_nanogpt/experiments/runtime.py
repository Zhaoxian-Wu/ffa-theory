"""Runtime helpers shared by the unified experiment CLI."""

from __future__ import annotations

import torch


def _format_sm_capability(capability: tuple[int, int]) -> str:
    major, minor = capability
    return f"sm_{major}{minor}"


def resolve_torch_device(device_preference: str, gpu: int) -> str:
    """Resolve the torch device string while handling unsupported CUDA builds."""

    if device_preference == "cpu":
        return "cpu"

    if not torch.cuda.is_available():
        if device_preference == "cuda":
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return "cpu"

    capability = torch.cuda.get_device_capability(gpu)
    arch = _format_sm_capability(capability)
    compiled_arches = set(torch.cuda.get_arch_list())
    arch_supported = not compiled_arches or arch in compiled_arches

    if not arch_supported:
        message = (
            f"CUDA device cuda:{gpu} reports {arch}, but this PyTorch build only supports "
            f"{sorted(compiled_arches)}."
        )
        if device_preference == "cuda":
            raise RuntimeError(message)
        print(f"Warning: {message} Falling back to CPU.")
        return "cpu"

    return f"cuda:{gpu}"
