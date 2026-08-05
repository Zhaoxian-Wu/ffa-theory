"""Train BP and strict-local FFA while jointly profiling last-block Sigma and Gamma ranks."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve()
ROOT = HERE.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent))

from algos.strict_local_ffa import StrictLocalFFAModel  # noqa: E402
from bp_tricks import BPModel  # noqa: E402
from common import get_cifar10_loaders, set_seed  # noqa: E402
from train_bp_ffa_gamma_erank_trajectory import (  # noqa: E402
    DEFAULT_PROBE_EPOCHS,
    _empty_cache,
    balanced_test_loader,
    bp_block_outputs,
    profile_bp_last_gamma,
    profile_ffa_last_gamma,
    train_bp_epoch,
    train_ffa_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="+", default=["cnn3", "cnn6", "cnn9"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--probe-epochs", nargs="+", type=int,
                        default=list(DEFAULT_PROBE_EPOCHS))
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results" / "bp_ffa_sigma_gamma_matched_trajectory")
    return parser.parse_args()


def representation_erank(chunks: Sequence[torch.Tensor], device: torch.device) -> float:
    features = torch.cat(list(chunks), dim=0).to(device, non_blocking=True)
    feature_dim = features.shape[1]
    norms = features.norm(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = features * (feature_dim ** 0.5) / norms
    sigma = normalized @ normalized.T / feature_dim
    trace = torch.diagonal(sigma).sum()
    denominator = sigma.square().sum()
    value = 1.0 if denominator <= 1e-12 else float((trace.square() / denominator).item())
    del features, normalized, sigma
    _empty_cache()
    return value


@torch.no_grad()
def profile_bp_last_sigma(model: BPModel, loader: DataLoader,
                          device: torch.device) -> float:
    model.eval()
    chunks: List[torch.Tensor] = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        outputs, _ = bp_block_outputs(model, x)
        chunks.append(outputs[-1].reshape(outputs[-1].shape[0], -1).cpu())
    return representation_erank(chunks, device)


@torch.no_grad()
def profile_ffa_last_sigma(model: StrictLocalFFAModel, loader: DataLoader,
                           device: torch.device) -> float:
    model.eval()
    chunks: List[torch.Tensor] = []
    for x, _ in loader:
        h = x.to(device, non_blocking=True)
        for block in model.blocks:
            h = block(h)
        chunks.append(h.reshape(h.shape[0], -1).cpu())
    return representation_erank(chunks, device)


def profile_and_append(records: List[Dict[str, float]], method: str, model,
                       loader: DataLoader, epoch: int, device: torch.device,
                       seed: int, persist: Callable[[], None]) -> None:
    _empty_cache()
    if method == "bp":
        gamma = profile_bp_last_gamma(model, loader, device)
        sigma = profile_bp_last_sigma(model, loader, device)
        n_signals = len(loader.dataset)
    else:
        gamma = profile_ffa_last_gamma(model, loader, device, seed + epoch)
        sigma = profile_ffa_last_sigma(model, loader, device)
        n_signals = 2 * len(loader.dataset)
    record = {
        "epoch": epoch,
        "last_gamma_erank_pr": gamma,
        "last_sigma_erank_pr": sigma,
        "n_representation_samples": len(loader.dataset),
        "n_gamma_signals": n_signals,
    }
    if not all(torch.isfinite(torch.tensor(value)) for value in record.values()):
        raise RuntimeError(f"Non-finite probe record: {record}")
    records.append(record)
    persist()
    print(
        f"  [{method}] probe epoch={epoch:3d} "
        f"last_sigma_erank={sigma:.6f} last_gamma_erank={gamma:.6f}",
        flush=True,
    )


def train_method(method: str, arch: str, train_loader: DataLoader,
                 probe_loader: DataLoader, args: argparse.Namespace,
                 device: torch.device, records: List[Dict[str, float]],
                 persist: Callable[[], None]) -> None:
    set_seed(args.seed)
    probe_epochs = set(args.probe_epochs)
    if method == "bp":
        model = BPModel(arch=arch, use_bn=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        model = StrictLocalFFAModel(arch=arch, lr=args.lr, hidden_dim=256).to(device)
        optimizer = None
    profile_seed = args.seed + 100000 * (len(arch) + 1)
    if 0 in probe_epochs:
        profile_and_append(records, method, model, probe_loader, 0, device,
                           profile_seed, persist)
    for epoch in range(args.epochs):
        if epoch == 100:
            optimizers: Iterable[optim.Optimizer] = [optimizer] if optimizer else model.optimizers
            for current_optimizer in optimizers:
                for group in current_optimizer.param_groups:
                    group["lr"] = args.lr * 0.1
        if method == "bp":
            train_bp_epoch(model, train_loader, optimizer, device)
        else:
            train_ffa_epoch(model, train_loader, device)
        completed_epoch = epoch + 1
        if completed_epoch in probe_epochs:
            profile_and_append(records, method, model, probe_loader, completed_epoch,
                               device, profile_seed, persist)
    del model
    _empty_cache()


def plot_sigma_trajectory(payload: Dict, path: Path) -> None:
    figure, axes = plt.subplots(1, len(payload["archs"]), figsize=(7, 2.5), sharey=False)
    if len(payload["archs"]) == 1:
        axes = [axes]
    for index, (axis, arch) in enumerate(zip(axes, payload["archs"])):
        for method, color, label in (("bp", "#1f77b4", "BP"), ("ffa", "#d62728", "FFA")):
            records = payload["results"][arch][method]
            axis.plot([row["epoch"] for row in records],
                      [row["last_sigma_erank_pr"] for row in records],
                      marker="o", linewidth=2, color=color, label=label)
        axis.set_title(arch.upper())
        axis.set_xlabel("Epoch")
        if index == 0:
            axis.set_ylabel(r"$\mathrm{erank}(\Sigma)$", fontsize=12)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, loc="center right")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if any(epoch < 0 or epoch > args.epochs for epoch in args.probe_epochs):
        raise ValueError("Probe epochs must lie in [0, epochs].")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_loader, _ = get_cifar10_loaders(batch_size=args.batch_size, augment=False)
    probe_loader = balanced_test_loader(args.samples_per_class, args.probe_batch_size)
    payload: Dict = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "archs": args.archs,
        "epochs": args.epochs,
        "probe_epochs": sorted(set(args.probe_epochs)),
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "device": str(device),
        "sigma": "Sigma = Z Z^T / d with each row of Z normalized to sqrt(d)",
        "gamma": "Gamma^(ell) = mean_i delta_i delta_i^T",
        "erank": "(tr K)^2 / tr(K^2)",
        "sigma_input": "raw CIFAR-10 images at the post-block representation",
        "gamma_signal": "BP task CE; FFA local positive/negative goodness loss",
        "results": {},
    }

    def persist() -> None:
        (args.out_dir / "trajectory.partial.json").write_text(json.dumps(payload, indent=2))

    for arch in args.archs:
        print(f"=== Matched Sigma/Gamma trajectory: {arch} on {device} ===", flush=True)
        payload["results"][arch] = {"bp": [], "ffa": []}
        for method in ("bp", "ffa"):
            train_method(method, arch, train_loader, probe_loader, args, device,
                         payload["results"][arch][method], persist)
        persist()
    plot_sigma_trajectory(payload, args.out_dir / "bp_ffa_sigma_erank_trajectory")
    (args.out_dir / "trajectory.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved matched trajectory to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
