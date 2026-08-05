"""Train BP and local FFA while profiling the last-block Gamma effective rank."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve()
ROOT = HERE.parent
sys.path.insert(0, str(HERE.parent))

from algos.strict_local_ffa import StrictLocalFFAModel, local_ffa_loss  # noqa: E402
from algos.vanilla_ffa import _apply_overlay_image, _random_wrong_labels  # noqa: E402
from bp_tricks import BPModel  # noqa: E402
from common import FAIR_NUM_CLASSES, get_cifar10_loaders, set_seed  # noqa: E402
from measure_bp_strict_local_ffa_gamma import (  # noqa: E402
    bp_block_outputs,
    bp_logits_from_block,
)


DEFAULT_PROBE_EPOCHS = (0, 1, 5, 10, 25, 50, 100, 150, 200)


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
                        default=ROOT / "results" / "bp_ffa_gamma_erank_trajectory")
    return parser.parse_args()


def balanced_test_loader(samples_per_class: int, batch_size: int) -> DataLoader:
    _, test_loader = get_cifar10_loaders(batch_size=batch_size, augment=False)
    counts = [0] * FAIR_NUM_CLASSES
    chosen: List[int] = []
    for index, target in enumerate(test_loader.dataset.targets):
        if counts[target] < samples_per_class:
            chosen.append(index)
            counts[target] += 1
        if all(count == samples_per_class for count in counts):
            break
    if any(count != samples_per_class for count in counts):
        raise RuntimeError(f"Could not construct a balanced test subset: {counts}")
    return DataLoader(Subset(test_loader.dataset, chosen), batch_size=batch_size,
                      shuffle=False, num_workers=2, pin_memory=True)


def erank_from_signals(chunks: Sequence[torch.Tensor], device: torch.device) -> float:
    signals = torch.cat(list(chunks), dim=0).to(device, non_blocking=True)
    sample_gram = signals @ signals.T
    trace = torch.diagonal(sample_gram).sum()
    denominator = sample_gram.square().sum()
    value = 1.0 if denominator <= 1e-12 else float((trace.square() / denominator).item())
    del signals, sample_gram
    torch.cuda.empty_cache()
    return value


@torch.no_grad()
def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def profile_bp_last_gamma(model: BPModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    chunks: List[torch.Tensor] = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        outputs, boundaries = bp_block_outputs(model, x)
        h_last, boundary_last = outputs[-1], boundaries[-1]
        logits = bp_logits_from_block(model, h_last, boundary_last)
        loss = F.cross_entropy(logits, y, reduction="sum")
        delta = torch.autograd.grad(loss, h_last)[0]
        chunks.append(delta.reshape(delta.shape[0], -1).detach().cpu())
    return erank_from_signals(chunks, device)


def profile_ffa_last_gamma(model: StrictLocalFFAModel, loader: DataLoader,
                           device: torch.device, profile_seed: int) -> float:
    model.eval()
    chunks: List[torch.Tensor] = []
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(profile_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(profile_seed)
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            h_pos = _apply_overlay_image(x, y)
            h_neg = _apply_overlay_image(x, _random_wrong_labels(y))
            for block_index, (block, head) in enumerate(zip(model.blocks, model.heads)):
                h_pos_out = block(h_pos.detach())
                h_neg_out = block(h_neg.detach())
                if block_index + 1 == len(model.blocks):
                    g_pos = head.goodness(head(h_pos_out))
                    g_neg = head.goodness(head(h_neg_out))
                    theta = ((g_pos.mean() + g_neg.mean()) / 2).detach()
                    pos_loss = F.softplus(-(g_pos - theta)).sum()
                    neg_loss = F.softplus(g_neg - theta).sum()
                    delta_pos = torch.autograd.grad(pos_loss, h_pos_out, retain_graph=True)[0]
                    delta_neg = torch.autograd.grad(neg_loss, h_neg_out)[0]
                    chunks.append(torch.cat([
                        delta_pos.reshape(delta_pos.shape[0], -1),
                        delta_neg.reshape(delta_neg.shape[0], -1),
                    ], dim=0).detach().cpu())
                    break
                h_pos, h_neg = h_pos_out.detach(), h_neg_out.detach()
    return erank_from_signals(chunks, device)


def train_bp_epoch(model: BPModel, loader: DataLoader, optimizer: optim.Optimizer,
                   device: torch.device) -> float:
    model.train()
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach())
    return loss_sum / max(len(loader), 1)


def train_ffa_epoch(model: StrictLocalFFAModel, loader: DataLoader,
                    device: torch.device) -> float:
    model.train()
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        h_pos = _apply_overlay_image(x, y)
        h_neg = _apply_overlay_image(x, _random_wrong_labels(y))
        batch_loss = 0.0
        for block, head, optimizer in zip(model.blocks, model.heads, model.optimizers):
            h_pos_out = block(h_pos.detach())
            h_neg_out = block(h_neg.detach())
            loss = local_ffa_loss(head.goodness(head(h_pos_out)),
                                  head.goodness(head(h_neg_out)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(block.parameters()) + list(head.parameters()), 1.0,
            )
            optimizer.step()
            batch_loss += float(loss.detach())
            h_pos, h_neg = h_pos_out.detach(), h_neg_out.detach()
        loss_sum += batch_loss / len(model.blocks)
    return loss_sum / max(len(loader), 1)


def profile_and_append(records: List[Dict[str, float]], method: str, model, loader,
                       epoch: int, device: torch.device, seed: int) -> None:
    _empty_cache()
    if method == "bp":
        value = profile_bp_last_gamma(model, loader, device)
        n_signals = len(loader.dataset)
    else:
        value = profile_ffa_last_gamma(model, loader, device, seed + epoch)
        n_signals = 2 * len(loader.dataset)
    records.append({"epoch": epoch, "last_gamma_erank_pr": value,
                    "n_signals": n_signals})
    print(f"  [{method}] probe epoch={epoch:3d} last_gamma_erank={value:.6f}", flush=True)


def train_method(method: str, arch: str, train_loader: DataLoader, probe_loader: DataLoader,
                 args: argparse.Namespace, device: torch.device) -> List[Dict[str, float]]:
    set_seed(args.seed)
    probe_epochs = set(args.probe_epochs)
    records: List[Dict[str, float]] = []
    if method == "bp":
        model = BPModel(arch=arch, use_bn=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        model = StrictLocalFFAModel(arch=arch, lr=args.lr, hidden_dim=256).to(device)
        optimizer = None
    if 0 in probe_epochs:
        profile_and_append(records, method, model, probe_loader, 0, device,
                           args.seed + 100000 * (len(arch) + 1))
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
            profile_and_append(records, method, model, probe_loader, completed_epoch, device,
                               args.seed + 100000 * (len(arch) + 1))
    del model
    _empty_cache()
    return records


def plot_trajectory(payload: Dict, path: Path) -> None:
    figure, axes = plt.subplots(1, len(payload["archs"]), figsize=(7, 2.5), sharey=False)
    if len(payload["archs"]) == 1:
        axes = [axes]
    for index, (axis, arch) in enumerate(zip(axes, payload["archs"])):
        for method, color, label in (("bp", "#1f77b4", "BP"), ("ffa", "#d62728", "FFA")):
            records = payload["results"][arch][method]
            axis.plot([row["epoch"] for row in records],
                      [row["last_gamma_erank_pr"] for row in records],
                      marker="o", linewidth=2, color=color, label=label)
        axis.set_title(arch.upper())
        axis.set_xlabel("Epoch")
        if index == 0:
            axis.set_ylabel(r"$\mathrm{erank}(\Gamma)$", fontsize=12)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, loc="upper right")
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
        "gamma": "Gamma^(ell) = mean_i delta_i delta_i^T",
        "erank": "(tr Gamma)^2 / tr(Gamma^2)",
        "results": {},
    }
    for arch in args.archs:
        print(f"=== Gamma trajectory: {arch} on {device} ===", flush=True)
        payload["results"][arch] = {
            "bp": train_method("bp", arch, train_loader, probe_loader, args, device),
            "ffa": train_method("ffa", arch, train_loader, probe_loader, args, device),
        }
        (args.out_dir / "trajectory.json").write_text(json.dumps(payload, indent=2))
    plot_trajectory(payload, args.out_dir / "bp_ffa_gamma_erank_trajectory")
    (args.out_dir / "trajectory.json").write_text(json.dumps(payload, indent=2))
    print(f"Saved trajectory to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
