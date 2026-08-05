"""SVD-controlled update-rank intervention for BP, FA, FFA, and Local-CE.

The data, backbone family, outer epoch budget, batch size, and no-augmentation
policy match ``tab:cifar10_cnn_bench``.  BP and FA use momentum SGD, following
Figure 3 of Boeshertz et al. (2026).  Strict-local FFA retains the benchmark's
native Adam optimizer, and the same SVD intervention is applied to Adam's
actual parameter update; forcing FFA onto SGD is kept as an optional ablation.

Primary comparisons:
  * BP: baseline versus top-r truncation;
  * FA: baseline versus full-rank spectral flattening;
  * strict-local FFA: baseline versus the same full-rank intervention.

The ``flat_rank*`` conditions provide a cleaner rank-only comparison against
``full_rank``: both interventions use equal nonzero singular values and
preserve the raw Frobenius norm, while changing only their count.  The
``interp*`` conditions instead contract the squared singular-value spectrum
toward its mean.  They preserve the raw Frobenius norm, recover the unmodified
update at interpolation zero, and recover the norm-matched polar direction at
interpolation one.

For convolutional weights, the update tensor is matricized as
``(out_channels, in_channels * kernel_height * kernel_width)``.  Biases and
normalization parameters are never transformed.  The primary ``conv`` scope
keeps classifier and local-goodness heads unchanged, so all three algorithms
receive the same intervention on the shared convolutional trunk.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from common import (  # noqa: E402
    FAIR_NUM_CLASSES,
    count_params,
    final_fc_in_dim,
    get_cifar10_loaders,
    make_conv_blocks,
    set_seed,
)
from algos.strict_local_ffa import (  # noqa: E402
    StrictLocalFFAModel,
    local_ffa_loss,
)
from algos.nokland_base import LPredHead, NoklandCNN  # noqa: E402
from algos.vanilla_ffa import (  # noqa: E402
    _apply_overlay_image,
    _random_wrong_labels,
)


CONDITION_SPECS = {
    "baseline": {"mode": "none", "rank": None},
    "rank1": {"mode": "truncate", "rank": 1},
    "rank2": {"mode": "truncate", "rank": 2},
    "rank5": {"mode": "truncate", "rank": 5},
    "flat_rank1": {"mode": "flat_truncate", "rank": 1},
    "flat_rank2": {"mode": "flat_truncate", "rank": 2},
    "flat_rank5": {"mode": "flat_truncate", "rank": 5},
    "flat20": {"mode": "flat_fraction", "rank": None, "fraction": 0.2},
    "flat40": {"mode": "flat_fraction", "rank": None, "fraction": 0.4},
    "flat60": {"mode": "flat_fraction", "rank": None, "fraction": 0.6},
    "flat80": {"mode": "flat_fraction", "rank": None, "fraction": 0.8},
    "flat100": {"mode": "flat_fraction", "rank": None, "fraction": 1.0},
    "interp0": {"mode": "spectral_interpolation", "rank": None, "fraction": 0.0},
    "interp20": {"mode": "spectral_interpolation", "rank": None, "fraction": 0.2},
    "interp40": {"mode": "spectral_interpolation", "rank": None, "fraction": 0.4},
    "interp60": {"mode": "spectral_interpolation", "rank": None, "fraction": 0.6},
    "interp80": {"mode": "spectral_interpolation", "rank": None, "fraction": 0.8},
    "interp100": {"mode": "spectral_interpolation", "rank": None, "fraction": 1.0},
    "full_rank": {"mode": "orthogonal", "rank": None},
}


def _matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim < 2:
        raise ValueError("Rank control requires a matrix-valued parameter.")
    return tensor.reshape(tensor.shape[0], -1)


def _effective_rank(singular_values: torch.Tensor) -> float:
    values = singular_values.detach().float()
    total = values.sum()
    if values.numel() == 0 or total <= 0:
        return 0.0
    probabilities = values / total
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def _participation_rank(singular_values: torch.Tensor) -> float:
    values = singular_values.detach().float()
    denominator = values.square().sum()
    if values.numel() == 0 or denominator <= 0:
        return 0.0
    return float(values.sum().square() / denominator)


def _gram_participation_rank(singular_values: torch.Tensor) -> float:
    values_squared = singular_values.detach().float().square()
    denominator = values_squared.square().sum()
    if values_squared.numel() == 0 or denominator <= 0:
        return 0.0
    return float(values_squared.sum().square() / denominator)


def transform_update(
    update: torch.Tensor,
    mode: str,
    rank: Optional[int],
    preserve_frobenius: bool,
    fraction: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Transform one momentum matrix and return exact spectral diagnostics."""
    matrix = _matrix_view(update)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    raw_norm = torch.linalg.vector_norm(s)

    if mode == "none":
        transformed_s = s
    elif mode == "truncate":
        if rank is None or rank < 1:
            raise ValueError("A positive rank is required for truncation.")
        keep = min(rank, s.numel())
        transformed_s = s.clone()
        transformed_s[keep:] = 0
    elif mode == "flat_truncate":
        if rank is None or rank < 1:
            raise ValueError("A positive rank is required for flat truncation.")
        keep = min(rank, s.numel())
        transformed_s = torch.zeros_like(s)
        transformed_s[:keep] = 1
    elif mode == "flat_fraction":
        if fraction is None or not 0 < fraction <= 1:
            raise ValueError("A fraction in (0, 1] is required.")
        keep = min(s.numel(), max(1, math.ceil(fraction * s.numel())))
        transformed_s = torch.zeros_like(s)
        transformed_s[:keep] = 1
    elif mode == "spectral_interpolation":
        if fraction is None or not 0 <= fraction <= 1:
            raise ValueError("An interpolation fraction in [0, 1] is required.")
        squared_s = s.square()
        mean_squared_s = squared_s.mean()
        transformed_s = torch.sqrt(
            ((1.0 - fraction) * squared_s + fraction * mean_squared_s)
            .clamp_min(0)
        )
    elif mode == "orthogonal":
        transformed_s = torch.ones_like(s)
    else:
        raise ValueError(f"Unknown rank-control mode: {mode}")

    transformed_norm = torch.linalg.vector_norm(transformed_s)
    if preserve_frobenius and transformed_norm > 0:
        transformed_s = transformed_s * (raw_norm / transformed_norm)

    transformed_matrix = (u * transformed_s.unsqueeze(0)) @ vh
    transformed = transformed_matrix.reshape_as(update)
    tolerance = (
        max(matrix.shape)
        * torch.finfo(s.dtype).eps
        * transformed_s.max()
    )
    numerical_rank = int((transformed_s > tolerance).sum())
    diagnostics = {
        "raw_effective_rank": _effective_rank(s),
        "raw_participation_rank": _participation_rank(s),
        "raw_gram_participation_rank": _gram_participation_rank(s),
        "applied_effective_rank": _effective_rank(transformed_s),
        "applied_participation_rank": _participation_rank(transformed_s),
        "applied_gram_participation_rank": _gram_participation_rank(transformed_s),
        "applied_numerical_rank": numerical_rank,
        "maximum_rank": int(s.numel()),
        "applied_rank_fraction": numerical_rank / max(int(s.numel()), 1),
        "raw_frobenius_norm": float(raw_norm),
        "applied_frobenius_norm": float(torch.linalg.vector_norm(transformed_s)),
    }
    return transformed, diagnostics


class RankControlledSGD(Optimizer):
    """Momentum SGD with an SVD transform on selected matrix parameters."""

    def __init__(
        self,
        named_parameters: Iterable[Tuple[str, nn.Parameter]],
        lr: float,
        momentum: float,
        mode: str,
        rank: Optional[int],
        fraction: Optional[float] = None,
        scope: str = "conv",
        preserve_frobenius: bool = True,
        metric_interval: int = 100,
    ) -> None:
        named_parameters = list(named_parameters)
        if not 0 <= momentum < 1:
            raise ValueError("momentum must lie in [0, 1).")
        if scope not in {"conv", "all_matrices"}:
            raise ValueError(f"Unknown intervention scope: {scope}")
        super().__init__([parameter for _, parameter in named_parameters], {"lr": lr})
        self.names = {id(parameter): name for name, parameter in named_parameters}
        self.momentum = momentum
        self.mode = mode
        self.rank = rank
        self.fraction = fraction
        self.scope = scope
        self.preserve_frobenius = preserve_frobenius
        self.metric_interval = max(int(metric_interval), 1)
        self.step_index = 0
        self.metric_sums: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.metric_counts: Dict[str, int] = defaultdict(int)
        self.last_metrics: Dict[str, Dict[str, float]] = {}

    def _is_selected(self, name: str, parameter: nn.Parameter) -> bool:
        if parameter.ndim < 2 or not name.endswith("weight"):
            return False
        return self.scope == "all_matrices" or (
            parameter.ndim == 4 and name.startswith("blocks.")
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        should_measure_baseline = self.step_index % self.metric_interval == 0
        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.mul_(self.momentum).add_(
                    gradient, alpha=1.0 - self.momentum
                )

                name = self.names[id(parameter)]
                selected = self._is_selected(name, parameter)
                diagnostics = None
                if selected and self.mode != "none":
                    applied_update, diagnostics = transform_update(
                        momentum_buffer,
                        mode=self.mode,
                        rank=self.rank,
                        preserve_frobenius=self.preserve_frobenius,
                        fraction=self.fraction,
                    )
                elif selected and should_measure_baseline:
                    _, diagnostics = transform_update(
                        momentum_buffer,
                        mode="none",
                        rank=None,
                        preserve_frobenius=self.preserve_frobenius,
                    )
                    applied_update = momentum_buffer
                else:
                    applied_update = momentum_buffer

                parameter.add_(applied_update, alpha=-lr)
                if diagnostics is not None:
                    self.last_metrics[name] = diagnostics
                    self.metric_counts[name] += 1
                    for key, value in diagnostics.items():
                        self.metric_sums[name][key] += float(value)
        self.step_index += 1
        return loss

    def spectral_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for name, sums in self.metric_sums.items():
            count = max(self.metric_counts[name], 1)
            summary[name] = {
                f"mean_{key}": value / count for key, value in sums.items()
            }
            summary[name].update(
                {f"last_{key}": value for key, value in self.last_metrics[name].items()}
            )
            summary[name]["measurements"] = self.metric_counts[name]
        return summary


class FAConv2dFunction(torch.autograd.Function):
    """Convolution with a fixed random kernel for input-gradient propagation."""

    @staticmethod
    def forward(
        ctx,
        inputs,
        weight,
        bias,
        feedback_weight,
        stride,
        padding,
        dilation,
        groups,
    ):
        ctx.save_for_backward(inputs, weight, feedback_weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.has_bias = bias is not None
        return F.conv2d(inputs, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, feedback_weight = ctx.saved_tensors
        grad_inputs = torch.nn.grad.conv2d_input(
            inputs.shape,
            feedback_weight,
            grad_output,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
        )
        grad_weight = torch.nn.grad.conv2d_weight(
            inputs,
            weight.shape,
            grad_output,
            ctx.stride,
            ctx.padding,
            ctx.dilation,
            ctx.groups,
        )
        grad_bias = (
            grad_output.sum(dim=(0, 2, 3)) if ctx.has_bias else None
        )
        return grad_inputs, grad_weight, grad_bias, None, None, None, None, None


class FAConv2d(nn.Module):
    """A Conv2d layer whose forward and feedback kernels are independent."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: int) -> None:
        super().__init__()
        kernel_size = _pair(kernel_size)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        feedback_weight = torch.empty_like(self.weight)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(feedback_weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.register_buffer("feedback_weight", feedback_weight)
        self.padding = _pair(padding)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return FAConv2dFunction.apply(
            inputs,
            self.weight,
            self.bias,
            self.feedback_weight,
            (1, 1),
            self.padding,
            (1, 1),
            1,
        )


class FALinearFunction(torch.autograd.Function):
    """Linear map with an independent fixed matrix for input gradients."""

    @staticmethod
    def forward(ctx, inputs, weight, bias, feedback_weight):
        ctx.save_for_backward(inputs, feedback_weight)
        ctx.has_bias = bias is not None
        return F.linear(inputs, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, feedback_weight = ctx.saved_tensors
        grad_inputs = grad_output @ feedback_weight
        grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).T @ (
            inputs.reshape(-1, inputs.shape[-1])
        )
        grad_bias = (
            grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
            if ctx.has_bias
            else None
        )
        return grad_inputs, grad_weight, grad_bias, None


class FALinear(nn.Module):
    """A Linear layer whose forward and feedback matrices are independent."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        feedback_weight = torch.empty_like(self.weight)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(feedback_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias, -bound, bound)
        self.register_buffer("feedback_weight", feedback_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return FALinearFunction.apply(
            inputs, self.weight, self.bias, self.feedback_weight
        )


class GlobalClassifier(nn.Module):
    """Shared CNN classifier trained with BP or layerwise feedback alignment."""

    def __init__(self, arch: str, algorithm: str) -> None:
        super().__init__()
        if algorithm not in {"bp", "fa"}:
            raise ValueError(f"Unsupported global algorithm: {algorithm}")
        blocks = []
        for spec in make_conv_blocks(arch):
            convolution = (
                nn.Conv2d(
                    spec["in_ch"],
                    spec["out_ch"],
                    spec["kernel"],
                    padding=spec["padding"],
                )
                if algorithm == "bp"
                else FAConv2d(
                    spec["in_ch"],
                    spec["out_ch"],
                    spec["kernel"],
                    padding=spec["padding"],
                )
            )
            pool = nn.MaxPool2d(2) if spec["pool"] == "max" else nn.Identity()
            blocks.append(nn.Sequential(convolution, nn.ReLU(), pool))
        self.blocks = nn.ModuleList(blocks)
        projection_type = nn.Linear if algorithm == "bp" else FALinear
        self.projection = projection_type(final_fc_in_dim(arch), 256)
        self.classifier = nn.Linear(256, FAIR_NUM_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden)
        hidden = F.adaptive_avg_pool2d(hidden, 4).flatten(1)
        return self.classifier(self.projection(hidden))


def _limited_batches(loader, maximum: Optional[int]) -> Iterator:
    for batch_index, batch in enumerate(loader):
        if maximum is not None and batch_index >= maximum:
            break
        yield batch


@torch.no_grad()
def evaluate_global(model, loader, device, maximum_batches: Optional[int]) -> float:
    model.eval()
    correct = total = 0
    for inputs, targets in _limited_batches(loader, maximum_batches):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        predictions = model(inputs).argmax(dim=1)
        correct += int((predictions == targets).sum())
        total += targets.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_ffa(model, loader, device, maximum_batches: Optional[int]) -> float:
    model.eval()
    correct = total = 0
    for inputs, targets in _limited_batches(loader, maximum_batches):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.shape[0]
        best_score = torch.full((batch_size,), -float("inf"), device=device)
        best_class = torch.zeros(batch_size, dtype=torch.long, device=device)
        for candidate in range(FAIR_NUM_CLASSES):
            candidate_labels = torch.full(
                (batch_size,), candidate, dtype=torch.long, device=device
            )
            hidden = _apply_overlay_image(inputs, candidate_labels)
            score = torch.zeros(batch_size, device=device)
            for block, head in zip(model.blocks, model.heads):
                hidden = block(hidden)
                score.add_(head.goodness(head(hidden)))
            selected = score > best_score
            best_score[selected] = score[selected]
            best_class[selected] = candidate
        correct += int((best_class == targets).sum())
        total += targets.numel()
    return correct / max(total, 1)


def _balanced_probe_loader(
    dataset,
    samples_per_class: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    targets = list(dataset.targets)
    counts = [0] * FAIR_NUM_CLASSES
    chosen = []
    for index, target in enumerate(targets):
        if counts[target] < samples_per_class:
            chosen.append(index)
            counts[target] += 1
        if all(count == samples_per_class for count in counts):
            break
    if any(count != samples_per_class for count in counts):
        raise RuntimeError(f"Could not construct a balanced probe subset: {counts}")
    return DataLoader(
        Subset(dataset, chosen),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def _gamma_erank_rows(
    chunks: Sequence[Sequence[torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, float]]:
    rows = []
    for layer_index, layer_chunks in enumerate(chunks, start=1):
        signals = torch.cat(list(layer_chunks), dim=0).to(
            device, non_blocking=True
        )
        sample_gram = signals @ signals.T
        trace = torch.diagonal(sample_gram).sum()
        denominator = sample_gram.square().sum()
        erank = (
            1.0
            if float(denominator) <= 1e-12
            else float((trace.square() / denominator).item())
        )
        rows.append({
            "layer": layer_index,
            "feature_dim": int(signals.shape[1]),
            "num_signals": int(signals.shape[0]),
            "gamma_erank_pr": erank,
        })
        del signals, sample_gram
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _summarize_gamma_probe(
    rows: Sequence[Dict[str, float]],
    base_examples: int,
    signals_per_example: int,
    delta_definition: str,
) -> Dict:
    eranks = [row["gamma_erank_pr"] for row in rows]
    return {
        "gamma_definition": "Gamma_l = N^{-1} sum_i delta_l(i) delta_l(i)^T",
        "erank_definition": "(tr Gamma_l)^2 / tr(Gamma_l^2)",
        "delta_definition": delta_definition,
        "rows": list(rows),
        "layer_gamma_erank_pr": eranks,
        "mean_layer_gamma_erank_pr": sum(eranks) / len(eranks),
        "last_layer_gamma_erank_pr": eranks[-1],
        "base_examples": base_examples,
        "signals_per_base_example": signals_per_example,
    }


def profile_global_gamma(
    model: GlobalClassifier,
    loader: DataLoader,
    device: torch.device,
    algorithm: str,
) -> Dict:
    model.eval()
    chunks: List[List[torch.Tensor]] = [[] for _ in model.blocks]
    base_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        hidden = inputs
        block_outputs = []
        for block in model.blocks:
            hidden = block(hidden)
            block_outputs.append(hidden)
        pooled = F.adaptive_avg_pool2d(hidden, 4).flatten(1)
        logits = model.classifier(model.projection(pooled))
        loss = F.cross_entropy(logits, targets, reduction="sum")
        gradients = torch.autograd.grad(loss, block_outputs)
        for layer_chunks, gradient in zip(chunks, gradients):
            layer_chunks.append(
                gradient.reshape(gradient.shape[0], -1).detach().cpu()
            )
        base_examples += targets.numel()
    rows = _gamma_erank_rows(chunks, device)
    delta_definition = (
        "task-CE derivative transported through fixed random feedback weights"
        if algorithm == "fa"
        else "task-CE derivative transported through transposed forward weights"
    )
    return _summarize_gamma_probe(
        rows, base_examples, 1, delta_definition
    )


def profile_ffa_gamma(
    model: StrictLocalFFAModel,
    loader: DataLoader,
    device: torch.device,
    profile_seed: int,
) -> Dict:
    model.eval()
    chunks: List[List[torch.Tensor]] = [[] for _ in model.blocks]
    base_examples = 0
    devices = (
        [device.index]
        if device.type == "cuda" and device.index is not None
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(profile_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(profile_seed)
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            positive = _apply_overlay_image(inputs, targets)
            negative = _apply_overlay_image(
                inputs, _random_wrong_labels(targets)
            )
            for layer_chunks, block, head in zip(
                chunks, model.blocks, model.heads
            ):
                positive_output = block(positive.detach())
                negative_output = block(negative.detach())
                positive_goodness = head.goodness(head(positive_output))
                negative_goodness = head.goodness(head(negative_output))
                threshold = (
                    (positive_goodness.mean() + negative_goodness.mean()) / 2
                ).detach()
                positive_loss = F.softplus(
                    -(positive_goodness - threshold)
                ).sum()
                negative_loss = F.softplus(
                    negative_goodness - threshold
                ).sum()
                positive_delta = torch.autograd.grad(
                    positive_loss, positive_output, retain_graph=True
                )[0]
                negative_delta = torch.autograd.grad(
                    negative_loss, negative_output
                )[0]
                layer_chunks.append(torch.cat([
                    positive_delta.reshape(positive_delta.shape[0], -1),
                    negative_delta.reshape(negative_delta.shape[0], -1),
                ], dim=0).detach().cpu())
                positive = positive_output.detach()
                negative = negative_output.detach()
            base_examples += targets.numel()
    rows = _gamma_erank_rows(chunks, device)
    return _summarize_gamma_probe(
        rows,
        base_examples,
        2,
        "current-block positive/negative local-goodness-loss derivative",
    )


class RankControlledAdam:
    """Adam followed by an SVD transform of its actual parameter update."""

    def __init__(
        self,
        named_parameters,
        lr,
        mode,
        rank,
        fraction,
        scope,
        preserve_frobenius,
        metric_interval,
        beta2=0.999,
        eps=1e-8,
    ):
        named_parameters = list(named_parameters)
        self.optimizer = torch.optim.Adam(
            [parameter for _, parameter in named_parameters],
            lr=lr,
            betas=(0.9, beta2),
            eps=eps,
        )
        self.param_groups = self.optimizer.param_groups
        self.names = {id(parameter): name for name, parameter in named_parameters}
        self.parameters = [parameter for _, parameter in named_parameters]
        self.mode = mode
        self.rank = rank
        self.fraction = fraction
        self.scope = scope
        self.preserve_frobenius = preserve_frobenius
        self.metric_interval = max(int(metric_interval), 1)
        self.step_index = 0
        self.metric_sums = defaultdict(lambda: defaultdict(float))
        self.metric_counts = defaultdict(int)
        self.last_metrics = {}

    def _is_selected(self, name, parameter):
        if parameter.ndim < 2 or not name.endswith("weight"):
            return False
        return self.scope == "all_matrices" or (
            parameter.ndim == 4 and name.startswith("blocks.")
        )

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        should_measure = self.step_index % self.metric_interval == 0
        snapshots = {}
        for parameter in self.parameters:
            name = self.names[id(parameter)]
            if self._is_selected(name, parameter) and (
                self.mode != "none" or should_measure
            ):
                snapshots[parameter] = parameter.detach().clone()
        self.optimizer.step()
        for parameter, before in snapshots.items():
            name = self.names[id(parameter)]
            raw_update = before - parameter
            if self.mode != "none":
                applied_update, diagnostics = transform_update(
                    raw_update,
                    mode=self.mode,
                    rank=self.rank,
                    preserve_frobenius=self.preserve_frobenius,
                    fraction=self.fraction,
                )
                parameter.copy_(before - applied_update)
            else:
                _, diagnostics = transform_update(
                    raw_update, mode="none", rank=None,
                    preserve_frobenius=self.preserve_frobenius,
                )
            self.last_metrics[name] = diagnostics
            self.metric_counts[name] += 1
            for key, value in diagnostics.items():
                self.metric_sums[name][key] += float(value)
        self.step_index += 1

    def spectral_summary(self):
        summary = {}
        for name, sums in self.metric_sums.items():
            count = max(self.metric_counts[name], 1)
            summary[name] = {
                f"mean_{key}": value / count for key, value in sums.items()
            }
            summary[name].update(
                {f"last_{key}": value for key, value in self.last_metrics[name].items()}
            )
            summary[name]["measurements"] = self.metric_counts[name]
        return summary

def _algorithm_lr(args, algorithm):
    override = getattr(args, f"{algorithm}_lr", None)
    return args.lr if override is None else override



def _make_optimizer(named_parameters, args, condition, algorithm):
    specification = CONDITION_SPECS[condition]
    uses_adam = (
        algorithm == "nokland_lpred"
        or (algorithm == "ffa" and args.ffa_optimizer == "adam")
    )
    if uses_adam:
        return RankControlledAdam(
            named_parameters,
            lr=_algorithm_lr(args, algorithm),
            mode=specification["mode"],
            rank=specification["rank"],
            fraction=specification.get("fraction"),
            scope=args.scope,
            preserve_frobenius=args.preserve_frobenius,
            metric_interval=args.metric_interval,
            beta2=args.adam_beta2,
            eps=args.adam_eps,
        )
    return RankControlledSGD(
        named_parameters,
        lr=_algorithm_lr(args, algorithm),
        momentum=args.momentum,
        mode=specification["mode"],
        rank=specification["rank"],
        fraction=specification.get("fraction"),
        scope=args.scope,
        preserve_frobenius=args.preserve_frobenius,
        metric_interval=args.metric_interval,
    )


def _set_learning_rate(optimizers: Sequence[Optimizer], learning_rate: float) -> None:
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate


def _to_cpu_tree(value):
    """Clone a nested checkpoint state onto CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


def _optimizer_checkpoint_state(optimizer) -> Dict:
    """Return resumable optimizer state and the rank-control step counter."""
    base_optimizer = (
        optimizer.optimizer
        if isinstance(optimizer, RankControlledAdam)
        else optimizer
    )
    return {
        "state_dict": _to_cpu_tree(base_optimizer.state_dict()),
        "step_index": optimizer.step_index,
        "mode": optimizer.mode,
        "rank": optimizer.rank,
        "fraction": optimizer.fraction,
    }


def train_global(args, algorithm: str, condition: str, train_loader, test_loader):
    device = torch.device(args.device)
    model = GlobalClassifier(args.arch, algorithm).to(device)
    optimizer = _make_optimizer(model.named_parameters(), args, condition, algorithm)
    best_accuracy = 0.0
    best_epoch = 0
    best_model_state = None
    accuracy_curve, loss_curve = [], []
    started = time.time()
    for epoch in range(args.epochs):
        if epoch == args.lr_decay_epoch:
            _set_learning_rate([optimizer], _algorithm_lr(args, algorithm) * args.lr_decay_factor)
        model.train()
        loss_sum = 0.0
        batches = 0
        for inputs, targets in _limited_batches(train_loader, args.max_train_batches):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = F.cross_entropy(model(inputs), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
        average_loss = loss_sum / max(batches, 1)
        accuracy = evaluate_global(
            model, test_loader, device, args.max_test_batches
        )
        loss_curve.append(average_loss)
        accuracy_curve.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            best_model_state = _to_cpu_tree(model.state_dict())
        print(
            f"[{algorithm}/{condition}/{args.arch}] epoch={epoch + 1}/{args.epochs} "
            f"loss={average_loss:.4f} acc={accuracy:.4f} best={best_accuracy:.4f}",
            flush=True,
        )
    return {
        "model": model,
        "optimizers": [optimizer],
        "elapsed_s": time.time() - started,
        "test_acc_final": accuracy_curve[-1],
        "test_acc_best": best_accuracy,
        "best_epoch": best_epoch,
        "best_model_state": best_model_state,
        "test_acc_curve": accuracy_curve,
        "train_loss_curve": loss_curve,
    }


def train_ffa(args, condition: str, train_loader, test_loader):
    device = torch.device(args.device)
    model = StrictLocalFFAModel(
        arch=args.arch,
        lr=_algorithm_lr(args, "ffa"),
        hidden_dim=args.ffa_head_dim,
        trunk_norm="none",
    ).to(device)
    optimizers = []
    for block_index, (block, head) in enumerate(zip(model.blocks, model.heads)):
        named_parameters = [
            (f"blocks.{block_index}.{name}", parameter)
            for name, parameter in block.named_parameters()
        ] + [
            (f"heads.{block_index}.{name}", parameter)
            for name, parameter in head.named_parameters()
        ]
        optimizers.append(
            _make_optimizer(named_parameters, args, condition, "ffa")
        )

    best_accuracy = 0.0
    best_epoch = 0
    best_model_state = None
    accuracy_curve, loss_curve = [], []
    started = time.time()
    for epoch in range(args.epochs):
        if epoch == args.lr_decay_epoch:
            _set_learning_rate(optimizers, _algorithm_lr(args, "ffa") * args.lr_decay_factor)
        model.train()
        loss_sum = 0.0
        batches = 0
        for inputs, targets in _limited_batches(train_loader, args.max_train_batches):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            positive = _apply_overlay_image(inputs, targets)
            negative = _apply_overlay_image(
                inputs, _random_wrong_labels(targets)
            )
            local_loss_values = []
            for block, head, optimizer in zip(
                model.blocks, model.heads, optimizers
            ):
                positive_output = block(positive.detach())
                negative_output = block(negative.detach())
                loss = local_ffa_loss(
                    head.goodness(head(positive_output)),
                    head.goodness(head(negative_output)),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(block.parameters()) + list(head.parameters()),
                    args.clip_grad_norm,
                )
                optimizer.step()
                local_loss_values.append(float(loss.detach()))
                positive = positive_output.detach()
                negative = negative_output.detach()
            loss_sum += sum(local_loss_values) / len(local_loss_values)
            batches += 1
        average_loss = loss_sum / max(batches, 1)
        accuracy = evaluate_ffa(
            model, test_loader, device, args.max_test_batches
        )
        loss_curve.append(average_loss)
        accuracy_curve.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            best_model_state = _to_cpu_tree(model.state_dict())
        print(
            f"[ffa/{condition}/{args.arch}] epoch={epoch + 1}/{args.epochs} "
            f"loss={average_loss:.4f} acc={accuracy:.4f} best={best_accuracy:.4f}",
            flush=True,
        )
    return {
        "model": model,
        "optimizers": optimizers,
        "elapsed_s": time.time() - started,
        "test_acc_final": accuracy_curve[-1],
        "test_acc_best": best_accuracy,
        "best_epoch": best_epoch,
        "best_model_state": best_model_state,
        "test_acc_curve": accuracy_curve,
        "train_loss_curve": loss_curve,
    }


@torch.no_grad()
def evaluate_nokland_lpred(
    model, heads, loader, device, maximum_batches: Optional[int]
) -> float:
    model.eval()
    heads.eval()
    correct = total = 0
    for inputs, targets in _limited_batches(loader, maximum_batches):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        activations = model.forward_eval(inputs)
        predictions = heads[-1](activations[-1]).argmax(dim=1)
        correct += int((predictions == targets).sum())
        total += targets.numel()
    return correct / max(total, 1)


def train_nokland_lpred(args, condition: str, train_loader, test_loader):
    """Train strict-local Nokland L_pred under its native benchmark recipe."""
    device = torch.device(args.device)
    model = NoklandCNN(
        arch=args.arch, use_bn=args.nokland_lpred_use_bn
    ).to(device)
    model.lpred_heads = nn.ModuleList(
        [LPredHead(channels, FAIR_NUM_CLASSES) for channels in model.channels]
    ).to(device)
    heads = model.lpred_heads
    optimizers = []
    for block_index, (block, head) in enumerate(zip(model.blocks, heads)):
        named_parameters = [
            (f"blocks.{block_index}.{name}", parameter)
            for name, parameter in block.named_parameters()
        ] + [
            (f"heads.{block_index}.{name}", parameter)
            for name, parameter in head.named_parameters()
        ]
        optimizers.append(
            _make_optimizer(
                named_parameters, args, condition, "nokland_lpred"
            )
        )

    best_accuracy = 0.0
    best_epoch = 0
    best_model_state = None
    accuracy_curve, loss_curve = [], []
    started = time.time()
    for epoch in range(args.epochs):
        if epoch == args.lr_decay_epoch:
            _set_learning_rate(
                optimizers,
                _algorithm_lr(args, "nokland_lpred") * args.lr_decay_factor,
            )
        model.train()
        heads.train()
        loss_sum = 0.0
        batches = 0
        for inputs, targets in _limited_batches(
            train_loader, args.max_train_batches
        ):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            hidden = inputs
            local_loss_values = []
            for block, head, optimizer in zip(
                model.blocks, heads, optimizers
            ):
                hidden = block(hidden.detach())
                loss = F.cross_entropy(head(hidden), targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                local_loss_values.append(float(loss.detach()))
            loss_sum += sum(local_loss_values) / len(local_loss_values)
            batches += 1
        average_loss = loss_sum / max(batches, 1)
        accuracy = evaluate_nokland_lpred(
            model, heads, test_loader, device, args.max_test_batches
        )
        loss_curve.append(average_loss)
        accuracy_curve.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            best_model_state = _to_cpu_tree(model.state_dict())
        print(
            f"[nokland_lpred/{condition}/{args.arch}] "
            f"epoch={epoch + 1}/{args.epochs} loss={average_loss:.4f} "
            f"acc={accuracy:.4f} best={best_accuracy:.4f}",
            flush=True,
        )
    return {
        "model": model,
        "optimizers": optimizers,
        "elapsed_s": time.time() - started,
        "test_acc_final": accuracy_curve[-1],
        "test_acc_best": best_accuracy,
        "best_epoch": best_epoch,
        "best_model_state": best_model_state,
        "test_acc_curve": accuracy_curve,
        "train_loss_curve": loss_curve,
    }


def _spectral_summaries(optimizers: Sequence) -> Dict:
    summaries = {}
    for optimizer_index, optimizer in enumerate(optimizers):
        for name, values in optimizer.spectral_summary().items():
            key = name if len(optimizers) == 1 else f"optimizer{optimizer_index}.{name}"
            summaries[key] = values
    return summaries


def run_one(args, algorithm: str, condition: str) -> Path:
    set_seed(args.seed)
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        augment=False,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )
    if algorithm == "ffa":
        outcome = train_ffa(args, condition, train_loader, test_loader)
    elif algorithm == "nokland_lpred":
        outcome = train_nokland_lpred(
            args, condition, train_loader, test_loader
        )
    else:
        outcome = train_global(
            args, algorithm, condition, train_loader, test_loader
        )
    terminal_gamma_probe = None
    if args.gamma_probe and algorithm in {"bp", "fa", "ffa"}:
        probe_loader = _balanced_probe_loader(
            test_loader.dataset,
            samples_per_class=args.gamma_samples_per_class,
            batch_size=args.gamma_batch_size,
            num_workers=args.num_workers,
        )
        probe_started = time.time()
        if algorithm == "ffa":
            terminal_gamma_probe = profile_ffa_gamma(
                outcome["model"],
                probe_loader,
                torch.device(args.device),
                profile_seed=args.seed + 1_000_003,
            )
        else:
            terminal_gamma_probe = profile_global_gamma(
                outcome["model"],
                probe_loader,
                torch.device(args.device),
                algorithm,
            )
        terminal_gamma_probe["probe_elapsed_s"] = time.time() - probe_started
        print(
            f"[{algorithm}/{condition}/{args.arch}] terminal Gamma eRanks="
            f"{terminal_gamma_probe['layer_gamma_erank_pr']}",
            flush=True,
        )
    uses_adam = (
        algorithm == "nokland_lpred"
        or (algorithm == "ffa" and args.ffa_optimizer == "adam")
    )
    result = {
        "status": "ok",
        "algorithm": algorithm,
        "condition": condition,
        "arch": args.arch,
        "seed": args.seed,
        "test_acc_final": outcome["test_acc_final"],
        "test_acc_best": outcome["test_acc_best"],
        "best_epoch": outcome["best_epoch"],
        "test_acc_curve": outcome["test_acc_curve"],
        "train_loss_curve": outcome["train_loss_curve"],
        "elapsed_s": outcome["elapsed_s"],
        "n_params": count_params(outcome["model"]),
        "spectral_summary": _spectral_summaries(outcome["optimizers"]),
        "terminal_gamma_probe": terminal_gamma_probe,
        "config": {
            "dataset": "CIFAR-10",
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": "RankControlledAdam" if uses_adam else "RankControlledSGD",
            "lr": _algorithm_lr(args, algorithm),
            "momentum": None if uses_adam else args.momentum,
            "momentum_convention": (
                None if uses_adam
                else "ema=(momentum*old)+(1-momentum)*gradient"
            ),
            "adam_beta2": args.adam_beta2 if uses_adam else None,
            "adam_eps": args.adam_eps if uses_adam else None,
            "lr_decay_epoch": args.lr_decay_epoch,
            "lr_decay_factor": args.lr_decay_factor,
            "rank_mode": CONDITION_SPECS[condition]["mode"],
            "target_rank": CONDITION_SPECS[condition]["rank"],
            "target_fraction": CONDITION_SPECS[condition].get("fraction"),
            "scope": args.scope,
            "preserve_frobenius": args.preserve_frobenius,
            "conv_matricization": "out_channels x (in_channels*kh*kw)",
            "clip_grad_norm": (
                None if algorithm == "nokland_lpred" else args.clip_grad_norm
            ),
            "augmentation": False,
            "normalization": (
                "BatchNorm in every backbone block"
                if algorithm == "nokland_lpred" and args.nokland_lpred_use_bn
                else "none in convolutional trunk"
            ),
            "locality": (
                "strict per-block loss with detached inter-block activations"
                if algorithm in {"ffa", "nokland_lpred"}
                else None
            ),
            "local_objective": (
                "cross-entropy at every block; deepest head for evaluation"
                if algorithm == "nokland_lpred"
                else None
            ),
            "max_train_batches": args.max_train_batches,
            "max_test_batches": args.max_test_batches,
            "gamma_probe": args.gamma_probe,
            "gamma_samples_per_class": args.gamma_samples_per_class,
            "gamma_batch_size": args.gamma_batch_size,
        },
    }
    output_directory = (
        Path(args.output_root)
        / args.arch
        / f"seed{args.seed}"
        / algorithm
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / f"{condition}.pt"
    if args.save_checkpoint:
        checkpoint = {
            "schema_version": 1,
            "algorithm": algorithm,
            "condition": condition,
            "arch": args.arch,
            "seed": args.seed,
            "completed_epochs": args.epochs,
            "best_epoch": outcome["best_epoch"],
            "test_acc_final": outcome["test_acc_final"],
            "test_acc_best": outcome["test_acc_best"],
            "test_acc_curve": outcome["test_acc_curve"],
            "train_loss_curve": outcome["train_loss_curve"],
            "config": result["config"],
            "model_state_dict": _to_cpu_tree(outcome["model"].state_dict()),
            "best_model_state_dict": outcome["best_model_state"],
            "optimizer_states": [
                _optimizer_checkpoint_state(optimizer)
                for optimizer in outcome["optimizers"]
            ],
            "terminal_gamma_probe": terminal_gamma_probe,
        }
        temporary_path = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(checkpoint_path)
        result["checkpoint_path"] = str(checkpoint_path)
        print(f"Saved {checkpoint_path}", flush=True)
    output_path = output_directory / f"{condition}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output_path}", flush=True)
    return output_path


def self_test(device_name: str) -> None:
    device = torch.device(device_name)
    torch.manual_seed(7)
    raw = torch.randn(8, 6, device=device)
    raw_norm = torch.linalg.vector_norm(raw)
    truncated, truncated_metrics = transform_update(raw, "truncate", 2, True)
    assert torch.linalg.matrix_rank(truncated).item() <= 2
    assert torch.allclose(torch.linalg.vector_norm(truncated), raw_norm, rtol=1e-4)
    flat_truncated, flat_truncated_metrics = transform_update(
        raw, "flat_truncate", 2, True
    )
    flat_truncated_s = torch.linalg.svdvals(flat_truncated)
    assert torch.linalg.matrix_rank(flat_truncated).item() <= 2
    assert torch.allclose(
        flat_truncated_s[0], flat_truncated_s[1], rtol=1e-4
    )
    assert torch.allclose(
        torch.linalg.vector_norm(flat_truncated), raw_norm, rtol=1e-4
    )
    fraction_update, fraction_metrics = transform_update(
        raw, "flat_fraction", None, True, fraction=0.4
    )
    fraction_s = torch.linalg.svdvals(fraction_update)
    assert fraction_metrics["applied_numerical_rank"] == 3
    assert fraction_metrics["applied_effective_rank"] == 3.0
    assert torch.allclose(fraction_s[0], fraction_s[2], rtol=1e-4)
    assert torch.allclose(
        torch.linalg.vector_norm(fraction_update), raw_norm, rtol=1e-4
    )
    interpolation_metrics = []
    interpolation_one = None
    for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        interpolated, metrics = transform_update(
            raw, "spectral_interpolation", None, True, fraction=fraction
        )
        assert torch.allclose(
            torch.linalg.vector_norm(interpolated), raw_norm, rtol=1e-4
        )
        interpolation_metrics.append(metrics)
        if fraction == 0.0:
            assert torch.allclose(interpolated, raw, rtol=1e-4, atol=1e-5)
        if fraction == 1.0:
            interpolation_one = interpolated
            interpolated_s = torch.linalg.svdvals(interpolated)
            assert torch.allclose(
                interpolated_s, interpolated_s.mean().expand_as(interpolated_s),
                rtol=1e-4, atol=1e-5,
            )
    interpolation_ranks = [
        metrics["applied_gram_participation_rank"]
        for metrics in interpolation_metrics
    ]
    assert all(
        current <= following + 1e-5
        for current, following in zip(interpolation_ranks, interpolation_ranks[1:])
    )

    orthogonal, orthogonal_metrics = transform_update(raw, "orthogonal", None, True)
    assert torch.linalg.matrix_rank(orthogonal).item() == 6
    assert torch.allclose(torch.linalg.vector_norm(orthogonal), raw_norm, rtol=1e-4)
    assert interpolation_one is not None
    assert torch.allclose(orthogonal, interpolation_one, rtol=1e-4, atol=1e-5)
    assert truncated_metrics["applied_numerical_rank"] == 2
    assert flat_truncated_metrics["applied_numerical_rank"] == 2
    assert orthogonal_metrics["applied_numerical_rank"] == 6

    inputs = torch.randn(4, 5, device=device, requires_grad=True)
    layer = FALinear(5, 3).to(device)
    upstream = torch.randn(4, 3, device=device)
    outputs = layer(inputs)
    outputs.backward(upstream)
    assert torch.allclose(inputs.grad, upstream @ layer.feedback_weight)
    expected_weight_gradient = upstream.T @ inputs.detach()
    assert torch.allclose(layer.weight.grad, expected_weight_gradient)

    conv_inputs = torch.randn(2, 3, 7, 7, device=device, requires_grad=True)
    conv = FAConv2d(3, 4, 3, padding=1).to(device)
    conv_outputs = conv(conv_inputs)
    conv_upstream = torch.randn_like(conv_outputs)
    conv_outputs.backward(conv_upstream)
    expected_input_gradient = torch.nn.grad.conv2d_input(
        conv_inputs.shape,
        conv.feedback_weight,
        conv_upstream,
        (1, 1),
        (1, 1),
        (1, 1),
        1,
    )
    assert torch.allclose(conv_inputs.grad, expected_input_gradient)
    print("rank_control_intervention self-test: PASS", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", default="bp,fa,ffa")
    parser.add_argument("--conditions", default="baseline,rank2,rank5,full_rank")
    parser.add_argument("--arch", choices=("cnn3", "cnn6", "cnn9"), default="cnn3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--bp-lr", type=float)
    parser.add_argument("--fa-lr", type=float)
    parser.add_argument("--ffa-lr", type=float, default=1e-3)
    parser.add_argument("--nokland-lpred-lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--ffa-optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--lr-decay-epoch", type=int, default=100)
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--scope", choices=("conv", "all_matrices"), default="conv")
    parser.add_argument(
        "--preserve-frobenius",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--metric-interval", type=int, default=100)
    parser.add_argument("--ffa-head-dim", type=int, default=256)
    parser.add_argument(
        "--nokland-lpred-use-bn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-root", default="results/CIFAR10-rank-control")
    parser.add_argument(
        "--save-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    parser.add_argument(
        "--gamma-probe",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--gamma-samples-per-class", type=int, default=100)
    parser.add_argument("--gamma-batch-size", type=int, default=100)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args.device)
        return
    algorithms = [item.strip() for item in args.algorithms.split(",") if item.strip()]
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    unknown_algorithms = set(algorithms) - {
        "bp", "fa", "ffa", "nokland_lpred"
    }
    unknown_conditions = set(conditions) - set(CONDITION_SPECS)
    if unknown_algorithms:
        raise ValueError(f"Unknown algorithms: {sorted(unknown_algorithms)}")
    if unknown_conditions:
        raise ValueError(f"Unknown conditions: {sorted(unknown_conditions)}")
    for algorithm in algorithms:
        for condition in conditions:
            run_one(args, algorithm, condition)


if __name__ == "__main__":
    main()
