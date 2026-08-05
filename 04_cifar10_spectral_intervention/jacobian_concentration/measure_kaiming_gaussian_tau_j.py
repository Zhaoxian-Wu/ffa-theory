"""Measure tail-Jacobian concentration for Kaiming weights and Gaussian inputs.

The network exactly uses the paper's scaled residual block

    h_next = h + ReLU(LayerNorm(W h)) / L,

with affine-free LayerNorm, no bias, and independent Kaiming-normal weights.
All width-by-width Jacobians are formed explicitly on a finite Gaussian sample.
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
from torch.func import jacrev, vmap


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "kaiming_gaussian_tau_j.json",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def residual_block(
    h: torch.Tensor,
    weight: torch.Tensor,
    depth: int,
) -> torch.Tensor:
    preactivation = F.linear(h, weight)
    normalized = F.layer_norm(
        preactivation,
        normalized_shape=(weight.shape[0],),
        weight=None,
        bias=None,
        eps=1e-5,
    )
    return h + F.relu(normalized) / depth


def summarize_tail(
    tail_jacobians: torch.Tensor,
    ell: int,
    depth: int,
) -> dict:
    reference = tail_jacobians.mean(dim=0)
    deviations = tail_jacobians - reference
    deviation_norms = torch.linalg.matrix_norm(deviations, ord=2)
    singular_values = torch.linalg.svdvals(reference)
    tau_sup = float(deviation_norms.max().item())
    tau_l2 = float(torch.sqrt(torch.mean(deviation_norms.square())).item())
    s_plus = float(singular_values.max().item())
    s_minus = float(singular_values.min().item())
    delta_sup = 2.0 * s_plus * tau_sup + tau_sup * tau_sup
    delta_l2 = 2.0 * s_plus * tau_l2 + tau_l2 * tau_l2
    return {
        "ell": ell,
        "num_blocks_in_tail": depth - ell,
        "tau_J_sup": tau_sup,
        "tau_J_L2": tau_l2,
        "tau_J_mean": float(deviation_norms.mean().item()),
        "s_plus_star": s_plus,
        "s_minus_star": s_minus,
        "reference_condition_number": s_plus / s_minus,
        "tau_sup_over_s_plus": tau_sup / s_plus,
        "tau_L2_over_s_plus": tau_l2 / s_plus,
        "Delta_J_sup": delta_sup,
        "Delta_J_L2": delta_l2,
        "Delta_sup_over_s_minus_squared": delta_sup / (s_minus * s_minus),
        "Delta_L2_over_s_minus_squared": delta_l2 / (s_minus * s_minus),
    }


def run_configuration(
    width: int,
    depth: int,
    samples: int,
    seed: int,
    device: torch.device,
) -> dict:
    set_seed(seed)
    inputs = torch.randn(samples, width, device=device)
    weights = []
    for _ in range(depth):
        weight = torch.empty(width, width, device=device)
        torch.nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="relu")
        weights.append(weight)

    states = [inputs]
    block_jacobians = []
    for weight in weights:
        current = states[-1]

        def single_block(h: torch.Tensor) -> torch.Tensor:
            return residual_block(h, weight, depth)

        block_jacobians.append(vmap(jacrev(single_block))(current))
        states.append(residual_block(current, weight, depth))

    identity = torch.eye(width, device=device).expand(samples, width, width)
    tail_jacobians = identity.clone()
    layer_summaries = []
    for ell in range(depth - 1, -1, -1):
        tail_jacobians = tail_jacobians @ block_jacobians[ell]
        layer_summaries.append(summarize_tail(tail_jacobians, ell, depth))
    layer_summaries.reverse()
    full_tail = layer_summaries[0]
    max_tau_sup_layer = max(layer_summaries, key=lambda item: item["tau_J_sup"])
    max_relative_layer = max(
        layer_summaries,
        key=lambda item: item["tau_sup_over_s_plus"],
    )
    max_gap_layer = max(
        layer_summaries,
        key=lambda item: item["Delta_sup_over_s_minus_squared"],
    )
    return {
        "seed": seed,
        "depth": depth,
        "residual_scale": 1.0 / depth,
        "full_tail": full_tail,
        "max_over_nonempty_tails": {
            "tau_J_sup": max_tau_sup_layer["tau_J_sup"],
            "tau_J_sup_ell": max_tau_sup_layer["ell"],
            "tau_sup_over_s_plus": max_relative_layer["tau_sup_over_s_plus"],
            "tau_sup_over_s_plus_ell": max_relative_layer["ell"],
            "Delta_sup_over_s_minus_squared": max_gap_layer[
                "Delta_sup_over_s_minus_squared"
            ],
            "Delta_sup_over_s_minus_squared_ell": max_gap_layer["ell"],
        },
        "layers": layer_summaries,
    }


def aggregate(configurations: list[dict], depths: list[int]) -> list[dict]:
    summaries = []
    for depth in depths:
        selected = [item for item in configurations if item["depth"] == depth]

        def values(path: tuple[str, ...]) -> torch.Tensor:
            extracted = []
            for item in selected:
                value = item
                for key in path:
                    value = value[key]
                extracted.append(value)
            return torch.tensor(extracted, dtype=torch.float64)

        row = {"depth": depth, "num_seeds": len(selected)}
        for name, path in {
            "full_tau_J_sup": ("full_tail", "tau_J_sup"),
            "full_tau_J_L2": ("full_tail", "tau_J_L2"),
            "full_s_plus_star": ("full_tail", "s_plus_star"),
            "full_s_minus_star": ("full_tail", "s_minus_star"),
            "full_tau_sup_over_s_plus": ("full_tail", "tau_sup_over_s_plus"),
            "full_Delta_sup_over_s_minus_squared": (
                "full_tail",
                "Delta_sup_over_s_minus_squared",
            ),
            "max_tail_tau_J_sup": ("max_over_nonempty_tails", "tau_J_sup"),
            "max_tail_tau_sup_over_s_plus": (
                "max_over_nonempty_tails",
                "tau_sup_over_s_plus",
            ),
        }.items():
            metric = values(path)
            row[name] = {
                "mean": float(metric.mean().item()),
                "std": float(metric.std(unbiased=False).item()),
                "min": float(metric.min().item()),
                "max": float(metric.max().item()),
            }
        summaries.append(row)
    return summaries


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)
    started = time.time()
    print(
        f"device={device} width={args.width} samples={args.samples} "
        f"depths={args.depths} seeds={args.seeds}",
        flush=True,
    )
    configurations = []
    for depth in args.depths:
        for seed in args.seeds:
            print(f"start depth={depth} seed={seed}", flush=True)
            result = run_configuration(
                width=args.width,
                depth=depth,
                samples=args.samples,
                seed=seed,
                device=device,
            )
            configurations.append(result)
            full = result["full_tail"]
            print(
                f"done depth={depth} seed={seed} "
                f"tau_sup={full['tau_J_sup']:.6f} "
                f"tau_L2={full['tau_J_L2']:.6f} "
                f"s_plus={full['s_plus_star']:.6f} "
                f"s_minus={full['s_minus_star']:.6f} "
                f"gap_ratio={full['Delta_sup_over_s_minus_squared']:.6f}",
                flush=True,
            )

    payload = {
        "experiment": "kaiming_gaussian_scaled_resnet_tail_jacobian_concentration",
        "scope": (
            "Finite-sample initialization diagnostic for the exact scaled-LN "
            "residual architecture. It is not a population supremum or a "
            "training-time claim."
        ),
        "architecture": (
            "h_next = h + ReLU(LayerNorm(W h)) / L; affine-free LayerNorm; "
            "no bias"
        ),
        "initialization": "independent Kaiming normal, fan_in mode, ReLU gain",
        "input_distribution": "iid standard normal N(0, I_d)",
        "dtype": str(torch.get_default_dtype()),
        "device": str(device),
        "width": args.width,
        "samples": args.samples,
        "depths": args.depths,
        "seeds": args.seeds,
        "configurations": configurations,
        "aggregate": aggregate(configurations, args.depths),
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["aggregate"], indent=2), flush=True)
    print(f"elapsed_seconds={payload['elapsed_seconds']:.3f}", flush=True)
    print(f"saved={args.out}", flush=True)


if __name__ == "__main__":
    main()
