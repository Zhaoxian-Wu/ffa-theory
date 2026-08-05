"""Estimate Jacobian concentration for an initialized CIFAR ResNet-18 tail.

The target is the final equal-width BasicBlock of a standard CIFAR ResNet-18
in evaluation mode.  Its 512 x 4 x 4 state is too large for dense Jacobians,
so JVP/VJP power iteration estimates the operator norms of J_x - mean_x J_x
and of the mean Jacobian.  The tau and s-plus estimates are lower estimates;
their ratio is an uncalibrated power-iteration proxy, not a certified bound.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import grad, jvp, vmap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from cifar10_resnet_spectral_norm import CifarResNetBackbone, LinearHead


ROOT = Path(__file__).resolve().parents[1]
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--power-iters", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "cifar10")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "init_resnet18_tau_j.json")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def normalize_rows(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min(1e-12)


def effective_rank_from_rows(rows: torch.Tensor) -> float:
    sample_count = rows.shape[0]
    trace = rows.square().sum() / sample_count
    sample_gram = rows @ rows.T / sample_count
    trace_square = sample_gram.square().sum()
    return float((trace.square() / trace_square).item()) if trace_square > 0 else 0.0


@torch.no_grad()
def collect_last_block_inputs(
    backbone: CifarResNetBackbone,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = backbone.stem(inputs)
    for stage_index, stage in enumerate(backbone.stages):
        for block_index, block in enumerate(stage):
            if stage_index == len(backbone.stages) - 1 and block_index == len(stage) - 1:
                return h, block(h)
            h = block(h)
    raise RuntimeError("The final BasicBlock was not found.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for this initialized ResNet measurement.")
    device = torch.device(args.device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    dataset = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=transform)
    if len(dataset) < args.samples:
        raise RuntimeError(f"CIFAR-10 has only {len(dataset)} samples, fewer than requested {args.samples}.")
    loader = DataLoader(Subset(dataset, list(range(args.samples))), batch_size=args.batch_size, shuffle=False)
    images, labels = next(iter(DataLoader(Subset(dataset, list(range(args.samples))), batch_size=args.samples)))
    images, labels = images.to(device), labels.to(device)

    backbone = CifarResNetBackbone(18).to(device).eval()
    head = LinearHead(512).to(device).eval()
    h_before, h_final = collect_last_block_inputs(backbone, images)
    block = backbone.stages[-1][-1]
    state_shape = tuple(h_before.shape[1:])
    state_dimension = int(h_before[0].numel())
    inputs_flat = h_before.flatten(1).detach()

    with torch.no_grad():
        logits = head(h_final)
        probabilities = torch.softmax(logits, dim=1)
        probabilities[torch.arange(args.samples, device=device), labels] -= 1.0
        pooled_deltas = probabilities @ head.fc.weight
        spatial_size = h_final.shape[-2] * h_final.shape[-1]
        output_deltas = (pooled_deltas[:, :, None, None] / spatial_size).expand_as(h_final).flatten(1).contiguous()
        gamma_erank = effective_rank_from_rows(output_deltas)
        init_accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())

    def tail_single(vector: torch.Tensor) -> torch.Tensor:
        return block(vector.reshape(1, *state_shape)).reshape(-1)

    def single_jvp(vector: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        return jvp(tail_single, (vector,), (direction,))[1]

    def single_vjp(vector: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
        return grad(lambda state: torch.dot(tail_single(state), cotangent))(vector)

    direct_jvp = vmap(single_jvp)
    direct_vjp = vmap(single_vjp)

    def mean_jvp(directions: torch.Tensor) -> torch.Tensor:
        return vmap(lambda direction: vmap(lambda vector: single_jvp(vector, direction))(inputs_flat))(directions).mean(dim=1)

    def mean_vjp(cotangents: torch.Tensor) -> torch.Tensor:
        return vmap(lambda cotangent: vmap(lambda vector: single_vjp(vector, cotangent))(inputs_flat))(cotangents).mean(dim=1)

    def centered_jvp(directions: torch.Tensor) -> torch.Tensor:
        return direct_jvp(inputs_flat, directions) - mean_jvp(directions)

    def centered_vjp(cotangents: torch.Tensor) -> torch.Tensor:
        return direct_vjp(inputs_flat, cotangents) - mean_vjp(cotangents)

    started = time.time()
    print(
        f"device={device} architecture=standard_cifar_resnet18 initialization=true samples={args.samples} "
        f"power_iters={args.power_iters} state_dimension={state_dimension}",
        flush=True,
    )
    directions = normalize_rows(torch.randn_like(inputs_flat))
    for iteration in range(args.power_iters):
        images_out = centered_jvp(directions)
        cotangents = normalize_rows(images_out)
        directions = normalize_rows(centered_vjp(cotangents))
        print(f"centered_power_iteration={iteration + 1}/{args.power_iters}", flush=True)
    centered_images = centered_jvp(directions)
    centered_norms = torch.linalg.vector_norm(centered_images, dim=1)

    mean_direction = normalize_rows(torch.randn(1, state_dimension, device=device))
    for iteration in range(args.power_iters):
        mean_image = mean_jvp(mean_direction)
        mean_cotangent = normalize_rows(mean_image)
        mean_direction = normalize_rows(mean_vjp(mean_cotangent))
        print(f"mean_power_iteration={iteration + 1}/{args.power_iters}", flush=True)
    s_plus_estimate = float(torch.linalg.vector_norm(mean_jvp(mean_direction)).item())
    tau_sup_estimate = float(centered_norms.max().item())
    tau_l2_estimate = float(torch.sqrt(torch.mean(centered_norms.square())).item())
    scale = math.sqrt(gamma_erank) / (s_plus_estimate * s_plus_estimate)
    summary = {
        "tail": "final same-width BasicBlock (stage 4, block 2) to pre-pooling representation",
        "state_shape": list(state_shape),
        "state_dimension": state_dimension,
        "tau_J_sup_power_lower_estimate": tau_sup_estimate,
        "tau_J_L2_power_lower_estimate": tau_l2_estimate,
        "s_plus_star_power_lower_estimate": s_plus_estimate,
        "s_minus_star": None,
        "s_minus_note": "Not estimated: a lower singular-value estimate needs a separate inverse/Lanczos routine.",
        "Gamma_L_effective_rank": gamma_erank,
        "rho_sup_power_iteration_proxy": tau_sup_estimate * scale,
        "rho_L2_power_iteration_proxy": tau_l2_estimate * scale,
        "per_sample_centered_opnorm_power_lower_estimates": [float(value) for value in centered_norms.cpu()],
        "mean_centered_opnorm_power_lower_estimate": float(centered_norms.mean().item()),
    }
    payload = {
        "experiment": "initialized_standard_cifar_resnet18_last_block_jacobian_concentration",
        "scope": (
            "Initialization diagnostic only. The standard ResNet has BatchNorm, no 1/L branch scaling, "
            "and stage transitions; this reports only its final same-width BasicBlock in evaluation mode."
        ),
        "device": str(device),
        "cuda_visible_devices": str(torch.cuda.current_device()),
        "seed": args.seed,
        "dataset": "CIFAR-10 test prefix",
        "num_measurement_samples": args.samples,
        "initial_accuracy_on_subset": init_accuracy,
        "reference_definition": "Empirical mean Jacobian action over the reported CIFAR-10 subset.",
        "norm_estimation": (
            "Six-step JVP/VJP power iteration; tau and s_plus values are lower estimates, "
            "so the reported rho values are lower estimates rather than certified bounds."
        ),
        "summary": summary,
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved={args.out}", flush=True)


if __name__ == "__main__":
    main()
