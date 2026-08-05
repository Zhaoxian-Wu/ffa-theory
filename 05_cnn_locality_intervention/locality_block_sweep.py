"""CIFAR-10 CNN12 locality sweep with grouped local cross-entropy.

The twelve convolutional layers are partitioned into equal consecutive groups.
Gradients propagate within a group and are detached at group boundaries. Each
group ends in a benchmark-shaped supervised CE head. With one twelve-layer
group, the training path is exactly ordinary end-to-end backpropagation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve()
ROOT = HERE.parent
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    ConvBlock,
    FAIR_FEAT_DIM,
    FAIR_LR,
    FAIR_NUM_CLASSES,
    count_params,
    get_cifar10_loaders,
    set_seed,
)


NUM_CONV_LAYERS = 12
VALID_LAYERS_PER_BLOCK = (1, 2, 3, 4, 6, 12)
DEFAULT_PROBE_EPOCHS = (0, 1, 5, 10, 25, 50, 100, 150, 200)


def cnn12_specs() -> List[Dict[str, Any]]:
    """Extend the benchmark CNN9 backbone by three 256-channel tail layers."""
    return [
        {"in_ch": 3, "out_ch": 32, "kernel": 3, "padding": 1, "pool": "none"},
        {"in_ch": 32, "out_ch": 32, "kernel": 3, "padding": 1, "pool": "max"},
        {"in_ch": 32, "out_ch": 64, "kernel": 3, "padding": 1, "pool": "none"},
        {"in_ch": 64, "out_ch": 64, "kernel": 3, "padding": 1, "pool": "max"},
        {"in_ch": 64, "out_ch": 128, "kernel": 3, "padding": 1, "pool": "none"},
        {"in_ch": 128, "out_ch": 128, "kernel": 3, "padding": 1, "pool": "max"},
        {"in_ch": 128, "out_ch": 256, "kernel": 3, "padding": 1, "pool": "none"},
        {"in_ch": 256, "out_ch": 256, "kernel": 3, "padding": 1, "pool": "none"},
        {"in_ch": 256, "out_ch": 256, "kernel": 3, "padding": 1, "pool": "none"},
        {"in_ch": 256, "out_ch": 256, "kernel": 3, "padding": 1, "pool": "none"},
        {"in_ch": 256, "out_ch": 256, "kernel": 3, "padding": 1, "pool": "none"},
        {"in_ch": 256, "out_ch": 256, "kernel": 3, "padding": 1, "pool": "none"},
    ]


def group_ranges(layers_per_block: int) -> List[Tuple[int, int]]:
    """Return half-open layer ranges covering all twelve layers exactly once."""
    if layers_per_block not in VALID_LAYERS_PER_BLOCK:
        raise ValueError(
            f"layers_per_block must be one of {VALID_LAYERS_PER_BLOCK}, "
            f"got {layers_per_block}"
        )
    return [
        (start, start + layers_per_block)
        for start in range(0, NUM_CONV_LAYERS, layers_per_block)
    ]


class BenchmarkCEHead(nn.Module):
    """Classifier head matching the benchmark BP projection and classifier."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.projection = nn.Linear(channels * 4 * 4, FAIR_FEAT_DIM)
        self.classifier = nn.Linear(FAIR_FEAT_DIM, FAIR_NUM_CLASSES)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.avgpool(h).reshape(h.shape[0], -1)
        return self.classifier(self.projection(z))


class GroupedLocalCE(nn.Module):
    """CNN12 with gradient isolation only at configured group boundaries."""

    def __init__(self, layers_per_block: int, use_bn: bool = True) -> None:
        super().__init__()
        specs = cnn12_specs()
        self.layers_per_block = layers_per_block
        self.ranges = group_ranges(layers_per_block)
        self.layers = nn.ModuleList([ConvBlock(spec, use_bn=use_bn) for spec in specs])
        self.heads = nn.ModuleList(
            [BenchmarkCEHead(specs[end - 1]["out_ch"]) for _, end in self.ranges]
        )

    @property
    def num_groups(self) -> int:
        return len(self.ranges)

    def forward_final(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.heads[-1](h)

    @torch.no_grad()
    def forward_final_eval(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_final(x)

    def group_parameters(self, group_index: int) -> Iterable[nn.Parameter]:
        start, end = self.ranges[group_index]
        for layer in self.layers[start:end]:
            yield from layer.parameters()
        yield from self.heads[group_index].parameters()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers-per-block", type=int, required=True,
                        choices=list(VALID_LAYERS_PER_BLOCK))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=100)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--probe-epochs", nargs="+", type=int,
                        default=list(DEFAULT_PROBE_EPOCHS))
    parser.add_argument("--lr", type=float, default=FAIR_LR)
    parser.add_argument("--lr-decay-epoch", type=int, default=100)
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results" / "CIFAR10-CNN_locality_block_sweep")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Smoke-test only; omit for formal runs.")
    parser.add_argument("--max-test-batches", type=int, default=None,
                        help="Smoke-test only; omit for formal runs.")
    parser.add_argument("--max-probe-batches", type=int, default=None,
                        help="Smoke-test only; omit for formal runs.")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def balanced_test_loader(
    test_loader: DataLoader,
    samples_per_class: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    counts = [0] * FAIR_NUM_CLASSES
    selected: List[int] = []
    for index, target in enumerate(test_loader.dataset.targets):
        if counts[target] < samples_per_class:
            selected.append(index)
            counts[target] += 1
        if all(count == samples_per_class for count in counts):
            break
    if any(count != samples_per_class for count in counts):
        raise RuntimeError(f"Could not build a balanced probe subset: {counts}")
    return DataLoader(
        Subset(test_loader.dataset, selected),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def limited_batches(loader: DataLoader, maximum: int | None):
    for batch_index, batch in enumerate(loader):
        if maximum is not None and batch_index >= maximum:
            break
        yield batch


def train_epoch(
    model: GroupedLocalCE,
    loader: DataLoader,
    optimizers: Sequence[optim.Optimizer],
    device: torch.device,
    max_batches: int | None,
) -> Dict[str, Any]:
    model.train()
    group_sums = [0.0] * model.num_groups
    num_batches = 0
    for x, y in limited_batches(loader, max_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        h = x
        for group_index, ((start, end), head, optimizer) in enumerate(
            zip(model.ranges, model.heads, optimizers)
        ):
            h = h.detach()
            for layer in model.layers[start:end]:
                h = layer(h)
            loss = F.cross_entropy(head(h), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            group_sums[group_index] += float(loss.detach())
        num_batches += 1
    if num_batches == 0:
        raise RuntimeError("Training loader produced zero batches.")
    group_losses = [value / num_batches for value in group_sums]
    return {
        "mean_group_ce": sum(group_losses) / len(group_losses),
        "terminal_group_ce": group_losses[-1],
        "group_ce": group_losses,
        "num_batches": num_batches,
    }


@torch.no_grad()
def evaluate(
    model: GroupedLocalCE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    num_batches = 0
    for x, y in limited_batches(loader, max_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model.forward_final_eval(x)
        loss_sum += float(F.cross_entropy(logits, y))
        correct += int((logits.argmax(dim=1) == y).sum())
        total += y.shape[0]
        num_batches += 1
    if total == 0:
        raise RuntimeError("Evaluation loader produced zero examples.")
    return {
        "test_accuracy": correct / total,
        "test_terminal_ce": loss_sum / num_batches,
        "num_examples": total,
    }


def erank_from_chunks(chunks: Sequence[torch.Tensor], device: torch.device) -> float:
    signals = torch.cat(list(chunks), dim=0).to(device, non_blocking=True)
    sample_gram = signals @ signals.T
    trace = torch.diagonal(sample_gram).sum()
    denominator = sample_gram.square().sum()
    value = 1.0 if float(denominator) <= 1e-12 else float(trace.square() / denominator)
    del signals, sample_gram
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return value


def profile_all_layer_gamma(
    model: GroupedLocalCE,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> Dict[str, Any]:
    """Measure each layer's local-loss derivative participation-ratio eRank."""
    model.eval()
    chunks: List[List[torch.Tensor]] = [[] for _ in range(NUM_CONV_LAYERS)]
    num_examples = 0
    for x, y in limited_batches(loader, max_batches):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        h = x
        for (start, end), head in zip(model.ranges, model.heads):
            h = h.detach()
            activations: List[torch.Tensor] = []
            for layer in model.layers[start:end]:
                h = layer(h)
                activations.append(h)
            loss = F.cross_entropy(head(h), y, reduction="sum")
            gradients = torch.autograd.grad(loss, activations)
            for offset, gradient in enumerate(gradients):
                chunks[start + offset].append(
                    gradient.reshape(gradient.shape[0], -1).detach().cpu()
                )
        num_examples += y.shape[0]
    if num_examples == 0:
        raise RuntimeError("Probe loader produced zero examples.")
    eranks = [erank_from_chunks(layer_chunks, device) for layer_chunks in chunks]
    return {
        "layer_gamma_erank_pr": eranks,
        "last_layer_gamma_erank_pr": eranks[-1],
        "num_signals": num_examples,
    }


def optimizer_state_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: optimizer_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [optimizer_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(optimizer_state_to_cpu(item) for item in value)
    return value


def save_checkpoint(
    path: Path,
    model: GroupedLocalCE,
    optimizers: Sequence[optim.Optimizer],
    payload: Dict[str, Any],
) -> None:
    checkpoint = {
        "format_version": 1,
        "epoch": payload["completed_epochs"],
        "config": payload["config"],
        "model_state": {
            key: tensor.detach().cpu() for key, tensor in model.state_dict().items()
        },
        "optimizer_states": [
            optimizer_state_to_cpu(optimizer.state_dict()) for optimizer in optimizers
        ],
        "curves": payload["curves"],
        "gamma_probes": payload["gamma_probes"],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def build_payload(args: argparse.Namespace, model: GroupedLocalCE) -> Dict[str, Any]:
    backbone_params = count_params(model.layers)
    head_params = count_params(model.heads)
    return {
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "completed_at": None,
        "completed_epochs": 0,
        "config": {
            "architecture": "cnn12_vgg_extension",
            "conv_specs": cnn12_specs(),
            "num_conv_layers": NUM_CONV_LAYERS,
            "layers_per_block": args.layers_per_block,
            "num_blocks": model.num_groups,
            "group_ranges_zero_based_half_open": model.ranges,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "probe_batch_size": args.probe_batch_size,
            "samples_per_class": args.samples_per_class,
            "probe_epochs": sorted(set(args.probe_epochs)),
            "optimizer": "Adam(per-group)",
            "lr": args.lr,
            "lr_decay_epoch": args.lr_decay_epoch,
            "lr_decay_factor": args.lr_decay_factor,
            "use_bn": True,
            "augmentation": False,
            "weight_decay": 0.0,
            "dropout": False,
            "head": "AdaptiveAvgPool(4)-Linear(.,256)-Linear(256,10)",
            "gradient_rule": "propagate within groups; detach between groups",
            "device": str(args.device),
            "max_train_batches": args.max_train_batches,
            "max_test_batches": args.max_test_batches,
            "max_probe_batches": args.max_probe_batches,
        },
        "parameter_counts": {
            "backbone": backbone_params,
            "auxiliary_heads": head_params,
            "total": backbone_params + head_params,
        },
        "curves": {
            "epoch": [],
            "train_mean_group_ce": [],
            "train_terminal_group_ce": [],
            "train_group_ce": [],
            "test_accuracy": [],
            "test_terminal_ce": [],
        },
        "gamma_probes": [],
        "summary": {},
    }


def main() -> None:
    args = parse_args()
    probe_epochs = sorted(set(args.probe_epochs))
    if any(epoch < 0 or epoch > args.epochs for epoch in probe_epochs):
        raise ValueError("Probe epochs must lie in [0, epochs].")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    run_dir = (
        args.out_root
        / f"lpb_{args.layers_per_block}"
        / f"seed_{args.seed}"
    )
    result_path = run_dir / "run.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if args.skip_completed and result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "completed":
            print(f"[skip] completed run exists: {run_dir}", flush=True)
            return
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        augment=False,
        num_workers=args.num_workers,
    )
    probe_loader = balanced_test_loader(
        test_loader,
        samples_per_class=args.samples_per_class,
        batch_size=args.probe_batch_size,
        num_workers=args.num_workers,
    )
    model = GroupedLocalCE(args.layers_per_block, use_bn=True).to(device)
    optimizers = [
        optim.Adam(model.group_parameters(index), lr=args.lr)
        for index in range(model.num_groups)
    ]
    payload = build_payload(args, model)
    atomic_json_dump(payload, result_path)

    print(
        f"[start] lpb={args.layers_per_block} blocks={model.num_groups} "
        f"seed={args.seed} device={device} params={payload['parameter_counts']}",
        flush=True,
    )
    started = time.time()
    if 0 in probe_epochs:
        probe = profile_all_layer_gamma(
            model, probe_loader, device, args.max_probe_batches
        )
        payload["gamma_probes"].append({"epoch": 0, **probe})
        atomic_json_dump(payload, result_path)
        print(
            f"  [probe] epoch=0 eranks="
            f"{[round(value, 4) for value in probe['layer_gamma_erank_pr']]}",
            flush=True,
        )

    for epoch_index in range(args.epochs):
        if epoch_index == args.lr_decay_epoch:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * args.lr_decay_factor
            print(
                f"  [recipe] epoch={epoch_index + 1} "
                f"lr={args.lr * args.lr_decay_factor:g}",
                flush=True,
            )
        train_metrics = train_epoch(
            model, train_loader, optimizers, device, args.max_train_batches
        )
        test_metrics = evaluate(
            model, test_loader, device, args.max_test_batches
        )
        completed_epoch = epoch_index + 1
        curves = payload["curves"]
        curves["epoch"].append(completed_epoch)
        curves["train_mean_group_ce"].append(train_metrics["mean_group_ce"])
        curves["train_terminal_group_ce"].append(train_metrics["terminal_group_ce"])
        curves["train_group_ce"].append(train_metrics["group_ce"])
        curves["test_accuracy"].append(test_metrics["test_accuracy"])
        curves["test_terminal_ce"].append(test_metrics["test_terminal_ce"])
        payload["completed_epochs"] = completed_epoch

        if completed_epoch in probe_epochs:
            probe = profile_all_layer_gamma(
                model, probe_loader, device, args.max_probe_batches
            )
            payload["gamma_probes"].append({"epoch": completed_epoch, **probe})
            print(
                f"  [probe] epoch={completed_epoch} eranks="
                f"{[round(value, 4) for value in probe['layer_gamma_erank_pr']]}",
                flush=True,
            )
        atomic_json_dump(payload, result_path)
        if completed_epoch == 1 or completed_epoch % 10 == 0:
            print(
                f"  [train] epoch={completed_epoch}/{args.epochs} "
                f"mean_ce={train_metrics['mean_group_ce']:.6f} "
                f"terminal_ce={train_metrics['terminal_group_ce']:.6f} "
                f"test_acc={test_metrics['test_accuracy']:.4f}",
                flush=True,
            )

    accuracies = payload["curves"]["test_accuracy"]
    payload["status"] = "completed"
    payload["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["summary"] = {
        "test_accuracy_final": accuracies[-1],
        "test_accuracy_best": max(accuracies),
        "test_accuracy_best_epoch": int(
            payload["curves"]["epoch"][accuracies.index(max(accuracies))]
        ),
        "train_mean_group_ce_final": payload["curves"]["train_mean_group_ce"][-1],
        "train_terminal_group_ce_final":
            payload["curves"]["train_terminal_group_ce"][-1],
        "test_terminal_ce_final": payload["curves"]["test_terminal_ce"][-1],
        "last_layer_gamma_erank_pr_final":
            payload["gamma_probes"][-1]["last_layer_gamma_erank_pr"],
        "elapsed_seconds": time.time() - started,
    }
    atomic_json_dump(payload, result_path)
    save_checkpoint(checkpoint_path, model, optimizers, payload)
    checkpoint_size = checkpoint_path.stat().st_size
    if not math.isfinite(payload["summary"]["test_accuracy_final"]):
        raise RuntimeError("Final accuracy is non-finite.")
    print(
        f"[done] result={result_path} checkpoint={checkpoint_path} "
        f"checkpoint_bytes={checkpoint_size} summary={payload['summary']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

