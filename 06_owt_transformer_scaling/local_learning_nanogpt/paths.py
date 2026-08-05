"""Path helpers for the Local_Learning_Nanogpt workspace."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_ROOT = PROJECT_ROOT / "data"
LOCAL_RESULTS_ROOT = PROJECT_ROOT / "results"
LEGACY_RESULTS_ROOT = PROJECT_ROOT.parent.parent / "results"
DESKTOP_ROOT = PROJECT_ROOT.parent.parent.parent


def _first_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def project_data_dir(dataset_name: str) -> str:
    return os.fspath(LOCAL_DATA_ROOT / dataset_name)


def resolve_owt_data_dir(dataset_name: str) -> str:
    candidates = [
        LOCAL_DATA_ROOT / dataset_name,
        PROJECT_ROOT / f"data_{dataset_name}",
    ]
    resolved = _first_existing(candidates)
    return os.fspath(resolved or candidates[0])


def resolve_shakespeare_data_dir(explicit_path: str | None = None) -> str:
    if explicit_path:
        return os.fspath(Path(explicit_path).expanduser())

    env_path = os.environ.get("LOCAL_LEARNING_NANOGPT_SHAKESPEARE_DIR")
    if env_path:
        return os.fspath(Path(env_path).expanduser())

    candidates = [
        LOCAL_DATA_ROOT / "shakespeare_char",
        DESKTOP_ROOT / "nanoGPT" / "data" / "shakespeare_char",
        PROJECT_ROOT.parent.parent / "nanoGPT" / "data" / "shakespeare_char",
    ]
    resolved = _first_existing(candidates)
    return os.fspath(resolved or candidates[0])


def resolve_results_dir() -> str:
    env_dir = os.environ.get("LOCAL_LEARNING_NANOGPT_RESULTS_DIR")
    if env_dir:
        return os.fspath(Path(env_dir).expanduser())

    candidates = [LOCAL_RESULTS_ROOT, LEGACY_RESULTS_ROOT]
    resolved = _first_existing(candidates)
    return os.fspath(resolved or LOCAL_RESULTS_ROOT)


def prepare_results_path(filename: str) -> str:
    results_root = Path(resolve_results_dir())
    results_root.mkdir(parents=True, exist_ok=True)
    return os.fspath(results_root / filename)


def resolve_result_path(filename: str) -> str:
    env_dir = os.environ.get("LOCAL_LEARNING_NANOGPT_RESULTS_DIR")
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser() / filename)
    candidates.extend([
        LOCAL_RESULTS_ROOT / filename,
        LEGACY_RESULTS_ROOT / filename,
    ])
    resolved = _first_existing(candidates)
    return os.fspath(resolved or candidates[0])

