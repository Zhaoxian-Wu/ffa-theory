"""Paired spectral-normalization ablation for the CIFAR-10 ResNet benchmark.

The experiment compares BP and layer-local cross-entropy (LCE) with and
without spectral normalization on every backbone convolution. Classification
is measured by a detached readout, while representation collapse is measured
from normalized sample-Gram matrices at the stem and four stage endpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
STAGE_CHANNELS = (64, 128, 256, 512)
DEPTH_TO_BLOCKS = {
    18: (2, 2, 2, 2),
    24: (3, 3, 3, 2),
    56: (7, 7, 7, 6),
    108: (13, 13, 13, 14),
}
PAPER_REFERENCE = {
    "bp": {18: 0.9515, 24: 0.9542, 56: 0.9568},
    "lce": {18: 0.8906, 24: 0.8903, 56: 0.8959},
}


@dataclass(frozen=True)
class RunConfig:
    method: str
    depth: int
    normalization: str
    seed: int
    epochs: int
    batch_size: int
    workers: int
    diagnostic_samples: int
    diagnostic_interval: int
    spectral_power_iterations: int
    amp: bool

    @property
    def name(self) -> str:
        return (
            f"{self.method}_resnet{self.depth}_{self.normalization}"
            f"_seed{self.seed}"
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=False)


class CifarResNetBackbone(nn.Module):
    """CIFAR-style four-stage ResNet shared by BP and LCE."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        if depth not in DEPTH_TO_BLOCKS:
            raise ValueError(f"Unsupported depth: {depth}")
        self.depth = depth
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
        )
        in_channels = 64
        stages: list[nn.ModuleList] = []
        for stage_index, (out_channels, block_count) in enumerate(
            zip(STAGE_CHANNELS, DEPTH_TO_BLOCKS[depth])
        ):
            blocks: list[BasicBlock] = []
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(BasicBlock(in_channels, out_channels, stride))
                in_channels = out_channels
            stages.append(nn.ModuleList(blocks))
        self.stages = nn.ModuleList(stages)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def iter_blocks(self) -> Iterable[tuple[int, int, BasicBlock]]:
        for stage_index, stage in enumerate(self.stages):
            for block_index, block in enumerate(stage):
                yield stage_index, block_index, block

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        stem_feature = self.stem(x)
        h = stem_feature
        stage_features: list[torch.Tensor] = []
        for stage in self.stages:
            for block in stage:
                h = block(h)
            stage_features.append(h)
        return stem_feature, stage_features


class LinearHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        return self.fc(pooled)


class DetachedReadout(nn.Module):
    """Linear classifier on concatenated pooled stem/stage features."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(64 + sum(STAGE_CHANNELS), num_classes)

    def forward(
        self, stem_feature: torch.Tensor, stage_features: list[torch.Tensor]
    ) -> torch.Tensor:
        features = [stem_feature, *stage_features]
        pooled = [
            F.adaptive_avg_pool2d(feature.detach(), 1).flatten(1)
            for feature in features
        ]
        return self.fc(torch.cat(pooled, dim=1))


class BPModel(nn.Module):
    def __init__(self, depth: int, use_spectral_norm: bool, power_iters: int) -> None:
        super().__init__()
        self.backbone = CifarResNetBackbone(depth)
        if use_spectral_norm:
            apply_spectral_norm(self.backbone, power_iters)
        self.classifier = LinearHead(512)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        stem_feature, stage_features = self.backbone.forward_features(x)
        logits = self.classifier(stage_features[-1])
        return logits, stem_feature, stage_features


class LCEModel(nn.Module):
    """One detached full-softmax CE objective per residual block."""

    def __init__(self, depth: int, use_spectral_norm: bool, power_iters: int) -> None:
        super().__init__()
        self.backbone = CifarResNetBackbone(depth)
        if use_spectral_norm:
            apply_spectral_norm(self.backbone, power_iters)
        head_channels: list[int] = []
        for stage_index, _, _ in self.backbone.iter_blocks():
            head_channels.append(STAGE_CHANNELS[stage_index])
        self.local_heads = nn.ModuleList(
            [LinearHead(channels) for channels in head_channels]
        )

    def forward_local(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        stem_feature = self.backbone.stem(x)
        h = stem_feature
        logits: list[torch.Tensor] = []
        stage_features: list[torch.Tensor] = []
        flat_index = 0
        for stage in self.backbone.stages:
            for block in stage:
                if flat_index > 0:
                    h = h.detach()
                h = block(h)
                logits.append(self.local_heads[flat_index](h))
                flat_index += 1
            stage_features.append(h)
        return logits, stem_feature, stage_features

    def forward_eval(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        stem_feature, stage_features = self.backbone.forward_features(x)
        logits = self.local_heads[-1](stage_features[-1])
        return logits, stem_feature, stage_features


def apply_spectral_norm(module: nn.Module, power_iters: int) -> list[str]:
    """Parametrize every backbone convolution, including projection shortcuts."""
    wrapped: list[str] = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d):
            spectral_norm(
                child,
                name="weight",
                n_power_iterations=power_iters,
                eps=1e-12,
            )
            wrapped.append(name)
    return wrapped


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_loaders(
    data_root: Path,
    batch_size: int,
    workers: int,
    seed: int,
    diagnostic_samples: int,
    download: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_set = datasets.CIFAR10(
        root=data_root, train=True, transform=train_transform, download=download
    )
    test_set = datasets.CIFAR10(
        root=data_root, train=False, transform=test_transform, download=download
    )
    diagnostic_count = min(diagnostic_samples, len(test_set))
    diagnostic_set = Subset(test_set, list(range(diagnostic_count)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "worker_init_fn": worker_seed,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, drop_last=False, **common
    )
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common)
    diagnostic_loader = DataLoader(
        diagnostic_set, shuffle=False, drop_last=False, **common
    )
    return train_loader, test_loader, diagnostic_loader


def build_model_and_optimizers(
    config: RunConfig, device: torch.device
) -> tuple[
    nn.Module,
    DetachedReadout,
    list[torch.optim.Optimizer],
    list[torch.optim.lr_scheduler.LRScheduler],
    torch.optim.Optimizer,
]:
    use_sn = config.normalization == "spectral_norm"
    if config.method == "bp":
        model: nn.Module = BPModel(
            config.depth, use_sn, config.spectral_power_iterations
        )
        optimizers = [
            torch.optim.SGD(
                model.parameters(),
                lr=0.1,
                momentum=0.9,
                nesterov=True,
                weight_decay=5e-4,
            )
        ]
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] = [
            torch.optim.lr_scheduler.MultiStepLR(
                optimizers[0], milestones=[100, 150], gamma=0.1
            )
        ]
    elif config.method == "lce":
        model = LCEModel(config.depth, use_sn, config.spectral_power_iterations)
        lce_model = model
        assert isinstance(lce_model, LCEModel)
        optimizers = []
        schedulers = []
        for flat_index, (_, _, block) in enumerate(
            lce_model.backbone.iter_blocks()
        ):
            parameters: list[nn.Parameter] = list(block.parameters())
            parameters.extend(lce_model.local_heads[flat_index].parameters())
            if flat_index == 0:
                parameters.extend(lce_model.backbone.stem.parameters())
            optimizer = torch.optim.Adam(parameters, lr=1e-3)
            optimizers.append(optimizer)
            schedulers.append(
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=[100], gamma=0.1
                )
            )
    else:
        raise ValueError(f"Unsupported method: {config.method}")
    model.to(device)
    readout = DetachedReadout().to(device)
    readout_optimizer = torch.optim.AdamW(
        readout.parameters(), lr=1e-3, weight_decay=1e-4
    )
    return model, readout, optimizers, schedulers, readout_optimizer


def train_epoch(
    model: nn.Module,
    readout: DetachedReadout,
    optimizers: list[torch.optim.Optimizer],
    readout_optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    method: str,
    amp: bool,
) -> dict[str, float]:
    model.train()
    readout.train()
    total_native_loss = 0.0
    total_readout_loss = 0.0
    total_examples = 0
    native_correct = 0
    readout_correct = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        readout_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            if method == "bp":
                assert isinstance(model, BPModel)
                native_logits, stem_feature, stage_features = model(images)
                native_loss = F.cross_entropy(native_logits, labels)
            else:
                assert isinstance(model, LCEModel)
                local_logits, stem_feature, stage_features = model.forward_local(images)
                local_losses = [
                    F.cross_entropy(logits, labels) for logits in local_logits
                ]
                native_loss = torch.stack(local_losses).sum()
                native_logits = local_logits[-1]
            readout_logits = readout(stem_feature, stage_features)
            readout_loss = F.cross_entropy(readout_logits, labels)
            joint_loss = native_loss + readout_loss
        scaler.scale(joint_loss).backward()
        for optimizer in optimizers:
            scaler.step(optimizer)
        scaler.step(readout_optimizer)
        scaler.update()
        count = labels.shape[0]
        total_examples += count
        native_loss_for_log = (
            native_loss
            if method == "bp"
            else native_loss / len(optimizers)
        )
        total_native_loss += float(native_loss_for_log.detach()) * count
        total_readout_loss += float(readout_loss.detach()) * count
        native_correct += int((native_logits.argmax(1) == labels).sum())
        readout_correct += int((readout_logits.argmax(1) == labels).sum())
    return {
        "native_loss": total_native_loss / total_examples,
        "readout_loss": total_readout_loss / total_examples,
        "native_acc": native_correct / total_examples,
        "readout_acc": readout_correct / total_examples,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    readout: DetachedReadout,
    loader: DataLoader,
    device: torch.device,
    method: str,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    readout.eval()
    total_examples = 0
    native_correct = 0
    readout_correct = 0
    native_loss_sum = 0.0
    readout_loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            if method == "bp":
                assert isinstance(model, BPModel)
                native_logits, stem_feature, stage_features = model(images)
            else:
                assert isinstance(model, LCEModel)
                native_logits, stem_feature, stage_features = model.forward_eval(images)
            readout_logits = readout(stem_feature, stage_features)
            native_loss = F.cross_entropy(native_logits, labels)
            readout_loss = F.cross_entropy(readout_logits, labels)
        count = labels.shape[0]
        total_examples += count
        native_loss_sum += float(native_loss) * count
        readout_loss_sum += float(readout_loss) * count
        native_correct += int((native_logits.argmax(1) == labels).sum())
        readout_correct += int((readout_logits.argmax(1) == labels).sum())
    return {
        "native_loss": native_loss_sum / total_examples,
        "readout_loss": readout_loss_sum / total_examples,
        "native_acc": native_correct / total_examples,
        "readout_acc": readout_correct / total_examples,
    }


def gram_statistics(features: torch.Tensor) -> dict[str, float]:
    normalized = F.normalize(features.float(), p=2, dim=1, eps=1e-12)
    gram = normalized @ normalized.T
    sample_count = gram.shape[0]
    frob_sq = float(torch.sum(gram.square()))
    distance = float(torch.linalg.vector_norm(gram - 1.0)) / sample_count
    effective_rank = (sample_count**2) / max(frob_sq, 1e-12)
    offdiag_sum = float(gram.sum() - torch.diagonal(gram).sum())
    offdiag_count = sample_count * max(sample_count - 1, 1)
    return {
        "kernel_distance_to_rank_one": distance,
        "kernel_effective_rank_pr": effective_rank,
        "mean_offdiag_cosine": offdiag_sum / offdiag_count,
    }


@torch.inference_mode()
def representation_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    method: str,
    amp: bool,
) -> dict[str, dict[str, float]]:
    model.eval()
    buckets: list[list[torch.Tensor]] = [[] for _ in range(5)]
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            if method == "bp":
                assert isinstance(model, BPModel)
                _, stem_feature, stage_features = model(images)
            else:
                assert isinstance(model, LCEModel)
                _, stem_feature, stage_features = model.forward_eval(images)
        all_features = [stem_feature, *stage_features]
        for index, feature in enumerate(all_features):
            pooled = F.adaptive_avg_pool2d(feature, 1).flatten(1)
            buckets[index].append(pooled.float().cpu())
    names = ["stem", "stage1", "stage2", "stage3", "stage4"]
    return {
        name: gram_statistics(torch.cat(bucket, dim=0))
        for name, bucket in zip(names, buckets)
    }


def estimated_conv_spectral_norms(backbone: nn.Module) -> dict[str, float]:
    """Estimate flattened-kernel top singular values by power iteration."""
    values: list[float] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    with torch.inference_mode():
        for module in backbone.modules():
            if not isinstance(module, nn.Conv2d):
                continue
            matrix = module.weight.detach().float().flatten(1).cpu()
            vector = torch.randn(matrix.shape[1], generator=generator)
            vector = F.normalize(vector, dim=0)
            for _ in range(20):
                left = F.normalize(matrix @ vector, dim=0)
                vector = F.normalize(matrix.T @ left, dim=0)
            values.append(float(torch.dot(left, matrix @ vector)))
    values_array = np.asarray(values, dtype=np.float64)
    return {
        "count": float(len(values)),
        "min": float(values_array.min()),
        "median": float(np.median(values_array)),
        "max": float(values_array.max()),
        "mean": float(values_array.mean()),
    }


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    readout: DetachedReadout,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    readout_optimizer: torch.optim.Optimizer,
    history: list[dict[str, Any]],
    best: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "readout": readout.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "schedulers": [scheduler.state_dict() for scheduler in schedulers],
            "readout_optimizer": readout_optimizer.state_dict(),
            "history": history,
            "best": best,
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    readout: DetachedReadout,
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    readout_optimizer: torch.optim.Optimizer,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    readout.load_state_dict(checkpoint["readout"])
    for optimizer, state in zip(optimizers, checkpoint["optimizers"]):
        optimizer.load_state_dict(state)
    for scheduler, state in zip(schedulers, checkpoint["schedulers"]):
        scheduler.load_state_dict(state)
    readout_optimizer.load_state_dict(checkpoint["readout_optimizer"])
    return (
        int(checkpoint["epoch"]) + 1,
        list(checkpoint["history"]),
        dict(checkpoint["best"]),
    )


def run_config(
    config: RunConfig,
    data_root: Path,
    output_root: Path,
    device: torch.device,
    download: bool,
    resume: bool,
    checkpoint_interval: int,
) -> dict[str, Any]:
    run_dir = output_root / config.name
    result_path = run_dir / "result.json"
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("status") == "completed":
            print(f"[skip] {config.name}: completed result exists", flush=True)
            return result

    print(f"[start] {config.name}", flush=True)
    set_seed(config.seed)
    train_loader, test_loader, diagnostic_loader = build_loaders(
        data_root,
        config.batch_size,
        config.workers,
        config.seed,
        config.diagnostic_samples,
        download,
    )
    model, readout, optimizers, schedulers, readout_optimizer = (
        build_model_and_optimizers(config, device)
    )
    checkpoint_path = run_dir / "checkpoint.pt"
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {
        "epoch": -1,
        "readout_acc": -math.inf,
        "native_acc": -math.inf,
    }
    start_epoch = 0
    if resume and checkpoint_path.exists():
        start_epoch, history, best = load_checkpoint(
            checkpoint_path,
            model,
            readout,
            optimizers,
            schedulers,
            readout_optimizer,
        )
        model.to(device)
        readout.to(device)
        print(f"[resume] {config.name} from epoch {start_epoch}", flush=True)

    initial_diagnostics = (
        history[0].get("diagnostics")
        if history and history[0].get("epoch") == -1
        else representation_diagnostics(
            model, diagnostic_loader, device, config.method, config.amp
        )
    )
    if not history:
        history.append({"epoch": -1, "diagnostics": initial_diagnostics})
    start_time = time.time()
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        train_metrics = train_epoch(
            model,
            readout,
            optimizers,
            readout_optimizer,
            train_loader,
            device,
            config.method,
            config.amp,
        )
        test_metrics = evaluate(
            model, readout, test_loader, device, config.method, config.amp
        )
        for scheduler in schedulers:
            scheduler.step()
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train": train_metrics,
            "test": test_metrics,
            "seconds": time.time() - epoch_start,
            "learning_rates": [
                float(optimizer.param_groups[0]["lr"]) for optimizer in optimizers
            ],
        }
        should_diagnose = (
            epoch == config.epochs - 1
            or (epoch + 1) % config.diagnostic_interval == 0
        )
        if should_diagnose:
            epoch_record["diagnostics"] = representation_diagnostics(
                model, diagnostic_loader, device, config.method, config.amp
            )
        history.append(epoch_record)
        if test_metrics["readout_acc"] > best["readout_acc"]:
            best = {
                "epoch": epoch,
                "readout_acc": test_metrics["readout_acc"],
                "native_acc": test_metrics["native_acc"],
            }
        print(
            f"[{config.name}] epoch={epoch + 1:03d}/{config.epochs} "
            f"train_readout={train_metrics['readout_acc']:.4f} "
            f"test_readout={test_metrics['readout_acc']:.4f} "
            f"test_native={test_metrics['native_acc']:.4f} "
            f"best={best['readout_acc']:.4f}@{best['epoch'] + 1} "
            f"sec={epoch_record['seconds']:.1f}",
            flush=True,
        )
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                readout,
                optimizers,
                schedulers,
                readout_optimizer,
                history,
                best,
            )

    final_diagnostics = history[-1]["diagnostics"]
    backbone = model.backbone
    conv_norms = estimated_conv_spectral_norms(backbone)
    reference = PAPER_REFERENCE.get(config.method, {}).get(config.depth)
    result = {
        "status": "completed",
        "config": asdict(config),
        "best": best,
        "final": history[-1]["test"],
        "initial_diagnostics": initial_diagnostics,
        "final_diagnostics": final_diagnostics,
        "conv_flattened_kernel_spectral_norms": conv_norms,
        "paper_reference_readout_acc": reference,
        "best_minus_paper_reference_pp": (
            100.0 * (best["readout_acc"] - reference)
            if reference is not None
            else None
        ),
        "history": history,
        "wall_seconds_this_invocation": time.time() - start_time,
        "torch_version": torch.__version__,
        "device": str(device),
    }
    atomic_json_dump(result, result_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(
        f"[done] {config.name}: best_readout={best['readout_acc']:.4f}",
        flush=True,
    )
    return result


def build_summary(output_root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for result in results:
        config = result["config"]
        key = (
            config["method"],
            int(config["depth"]),
            config["normalization"],
            int(config["seed"]),
        )
        by_key[key] = result
        rows.append(
            {
                "method": config["method"],
                "depth": config["depth"],
                "normalization": config["normalization"],
                "seed": config["seed"],
                "best_epoch": result["best"]["epoch"],
                "best_readout_acc": result["best"]["readout_acc"],
                "final_readout_acc": result["final"]["readout_acc"],
                "final_native_acc": result["final"]["native_acc"],
                "paper_reference_readout_acc": result[
                    "paper_reference_readout_acc"
                ],
                "best_minus_paper_reference_pp": result[
                    "best_minus_paper_reference_pp"
                ],
                "stage4_kernel_distance_to_rank_one": result[
                    "final_diagnostics"
                ]["stage4"]["kernel_distance_to_rank_one"],
                "stage4_kernel_effective_rank_pr": result["final_diagnostics"][
                    "stage4"
                ]["kernel_effective_rank_pr"],
                "stage4_mean_offdiag_cosine": result["final_diagnostics"][
                    "stage4"
                ]["mean_offdiag_cosine"],
            }
        )
    comparisons: list[dict[str, Any]] = []
    groups = sorted({(row["method"], row["depth"], row["seed"]) for row in rows})
    for method, depth, seed in groups:
        baseline = by_key.get((method, depth, "baseline", seed))
        normalized = by_key.get((method, depth, "spectral_norm", seed))
        if baseline is None or normalized is None:
            continue
        base_diag = baseline["final_diagnostics"]["stage4"]
        norm_diag = normalized["final_diagnostics"]["stage4"]
        comparisons.append(
            {
                "method": method,
                "depth": depth,
                "seed": seed,
                "baseline_best_readout_acc": baseline["best"]["readout_acc"],
                "spectral_norm_best_readout_acc": normalized["best"][
                    "readout_acc"
                ],
                "spectral_norm_minus_baseline_acc_pp": 100.0
                * (
                    normalized["best"]["readout_acc"]
                    - baseline["best"]["readout_acc"]
                ),
                "baseline_stage4_kernel_distance_to_rank_one": base_diag[
                    "kernel_distance_to_rank_one"
                ],
                "spectral_norm_stage4_kernel_distance_to_rank_one": norm_diag[
                    "kernel_distance_to_rank_one"
                ],
                "spectral_norm_minus_baseline_kernel_distance": norm_diag[
                    "kernel_distance_to_rank_one"
                ]
                - base_diag["kernel_distance_to_rank_one"],
                "baseline_stage4_kernel_effective_rank_pr": base_diag[
                    "kernel_effective_rank_pr"
                ],
                "spectral_norm_stage4_kernel_effective_rank_pr": norm_diag[
                    "kernel_effective_rank_pr"
                ],
                "spectral_norm_minus_baseline_kernel_effective_rank_pr": norm_diag[
                    "kernel_effective_rank_pr"
                ]
                - base_diag["kernel_effective_rank_pr"],
            }
        )
    summary = {"runs": rows, "paired_comparisons": comparisons}
    atomic_json_dump(summary, output_root / "summary.json")
    if rows:
        with (output_root / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if comparisons:
        with (output_root / "paired_comparisons.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(comparisons[0].keys())
            )
            writer.writeheader()
            writer.writerows(comparisons)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=["bp", "lce"], default=["bp", "lce"])
    parser.add_argument(
        "--depths", nargs="+", type=int, choices=sorted(DEPTH_TO_BLOCKS), default=[18, 24, 56]
    )
    parser.add_argument(
        "--normalizations",
        nargs="+",
        choices=["baseline", "spectral_norm"],
        default=["baseline", "spectral_norm"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--diagnostic-samples", type=int, default=1024)
    parser.add_argument("--diagnostic-interval", type=int, default=20)
    parser.add_argument("--spectral-power-iterations", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/cifar10_resnet_spectral_norm"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    configs = [
        RunConfig(
            method=method,
            depth=depth,
            normalization=normalization,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            diagnostic_samples=args.diagnostic_samples,
            diagnostic_interval=args.diagnostic_interval,
            spectral_power_iterations=args.spectral_power_iterations,
            amp=not args.no_amp and device.type == "cuda",
        )
        for depth in args.depths
        for method in args.methods
        for normalization in args.normalizations
        for seed in args.seeds
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        {
            "configs": [asdict(config) for config in configs],
            "data_root": str(args.data_root),
            "output_root": str(args.output_root),
            "device": str(device),
        },
        args.output_root / "manifest.json",
    )
    results = [
        run_config(
            config,
            args.data_root,
            args.output_root,
            device,
            args.download,
            not args.no_resume,
            args.checkpoint_interval,
        )
        for config in configs
    ]
    summary = build_summary(args.output_root, results)
    print(json.dumps(summary["paired_comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
