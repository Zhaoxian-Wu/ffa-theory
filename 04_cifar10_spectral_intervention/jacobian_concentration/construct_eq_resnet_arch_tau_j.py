"""Measure Jacobian concentration in an exact scaled-ResNet construction.

The residual block is exactly h <- h + ReLU(LayerNorm(W h)) / L, matching
eq:resnet_arch with an affine-free LayerNorm.  The construction constrains
W 1 = 0 and places the class signal in the all-ones skip direction.  Small
orthogonal data noise then makes the Jacobians nearly, but not exactly,
common across examples.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev, vmap
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


class ScaledResNetBlock(nn.Module):
    """One affine-free LayerNorm residual block from eq:resnet_arch."""

    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.raw_weight = nn.Parameter(torch.randn(width, width) / math.sqrt(width))
        self.layer_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-5)
        self.depth = depth
        direction = torch.ones(width) / math.sqrt(width)
        self.register_buffer("orthogonal_projector", torch.eye(width) - torch.outer(direction, direction))

    def effective_weight(self) -> torch.Tensor:
        return self.raw_weight @ self.orthogonal_projector

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        branch = F.relu(self.layer_norm(F.linear(h, self.effective_weight())))
        return h + branch / self.depth


class ExactScaledResNetClassifier(nn.Module):
    """Equal-width classifier whose residual stream matches eq:resnet_arch."""

    def __init__(self, width: int, depth: int, classes: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ScaledResNetBlock(width, depth) for _ in range(depth)])
        self.classifier = nn.Linear(width, classes)

    def representations(self, x: torch.Tensor) -> list[torch.Tensor]:
        representations = [x]
        for block in self.blocks:
            representations.append(block(representations[-1]))
        return representations

    def tail(self, ell: int, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks[ell:]:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.representations(x)[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--orthogonal-noise", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--block-lr", type=float, default=1e-3)
    parser.add_argument("--head-lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "eq_resnet_arch_tau_j.json")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_base_and_direction(width: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    direction = torch.ones(width) / math.sqrt(width)
    base = torch.randn(width, generator=generator)
    base = base - torch.dot(base, direction) * direction
    return base / torch.linalg.vector_norm(base) * math.sqrt(width), direction


def make_dataset(
    samples: int,
    classes: int,
    base: torch.Tensor,
    direction: torch.Tensor,
    orthogonal_noise: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(samples) % classes
    class_positions = torch.linspace(-4.5, 4.5, classes)
    noise = torch.randn(samples, base.numel(), generator=generator)
    noise = noise - (noise @ direction).unsqueeze(1) * direction
    inputs = base.unsqueeze(0) + class_positions[labels].unsqueeze(1) * direction + orthogonal_noise * noise
    return inputs, labels


def effective_rank(gram: torch.Tensor) -> float:
    trace = torch.trace(gram)
    return float((trace.square() / torch.sum(gram.square())).item())


def summarize_tail(model: ExactScaledResNetClassifier, ell: int, inputs: torch.Tensor, erank: float) -> dict:
    jacobians = vmap(jacrev(lambda h: model.tail(ell, h)))(inputs)
    reference = jacobians.mean(dim=0)
    deviations = jacobians - reference
    norms = torch.linalg.matrix_norm(deviations, ord=2)
    singular_values = torch.linalg.svdvals(reference)
    tau_sup = float(norms.max().item())
    tau_l2 = float(torch.sqrt(torch.mean(norms.square())).item())
    s_plus = float(singular_values.max().item())
    s_minus = float(singular_values.min().item())
    scale = math.sqrt(erank) / (s_plus * s_plus)
    return {
        "ell": ell,
        "num_residual_blocks_in_tail": len(model.blocks) - ell,
        "tau_J_sup": tau_sup,
        "tau_J_L2": tau_l2,
        "s_plus_star": s_plus,
        "s_minus_star": s_minus,
        "Gamma_L_effective_rank": erank,
        "rho_sup": tau_sup * scale,
        "rho_L2": tau_l2 * scale,
        "deviation_opnorm_mean": float(norms.mean().item()),
        "deviation_opnorm_cv": float((norms.std(unbiased=False) / norms.mean()).item()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")
    base, direction = make_base_and_direction(args.width, args.seed)
    train_x, train_y = make_dataset(
        args.train_samples, args.classes, base, direction, args.orthogonal_noise, args.seed + 1
    )
    test_x, test_y = make_dataset(
        args.test_samples, args.classes, base, direction, args.orthogonal_noise, args.seed + 2
    )
    model = ExactScaledResNetClassifier(args.width, args.depth, args.classes).to(device)
    block_parameters = [parameter for block in model.blocks for parameter in block.parameters()]
    optimizer = torch.optim.Adam(
        [
            {"params": block_parameters, "lr": args.block_lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ],
        weight_decay=1e-4,
    )
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    print(
        f"device={device} architecture=exact_eq_resnet_arch width={args.width} depth={args.depth} "
        f"orthogonal_noise={args.orthogonal_noise}",
        flush=True,
    )
    for epoch in range(args.epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                loss = F.cross_entropy(model(train_x), train_y).item()
                accuracy = (model(train_x).argmax(dim=1) == train_y).float().mean().item()
            print(f"epoch={epoch + 1}/{args.epochs} loss={loss:.6f} train_accuracy={accuracy:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        test_accuracy = float((logits.argmax(dim=1) == test_y).float().mean().item())
        probabilities = torch.softmax(logits, dim=1)
        probabilities[torch.arange(args.test_samples), test_y] -= 1.0
        deltas = probabilities @ model.classifier.weight
        gamma = deltas.T @ deltas / args.test_samples
        erank = effective_rank(gamma)
        representations = model.representations(test_x)
        invariance_errors = [
            float(torch.linalg.vector_norm(block.effective_weight() @ direction).item()) for block in model.blocks
        ]
    layers = [summarize_tail(model, ell, representations[ell], erank) for ell in range(args.depth)]
    payload = {
        "experiment": "exact_eq_resnet_arch_symmetry_protected_jacobian_concentration",
        "scope": (
            "Controlled synthetic construction. The residual architecture is exactly eq:resnet_arch, "
            "but W 1 = 0 and the data have a common orthogonal component plus small orthogonal noise. "
            "This is a sufficient mechanism, not a claim about generic ResNet training."
        ),
        "device": str(device),
        "seed": args.seed,
        "architecture": "h_next = h + ReLU(LayerNorm(W h)) / L; affine-free LayerNorm; no bias",
        "width": args.width,
        "depth": args.depth,
        "residual_scale": 1.0 / args.depth,
        "symmetry": {
            "weight_constraint": "W @ (1/sqrt(d)) * 1 = 0 for every block",
            "class_signal": "the all-ones skip direction",
            "orthogonal_noise": args.orthogonal_noise,
            "max_weight_constraint_error": max(invariance_errors),
        },
        "training": {
            "samples": args.train_samples,
            "epochs": args.epochs,
            "block_lr": args.block_lr,
            "head_lr": args.head_lr,
            "test_accuracy": test_accuracy,
        },
        "Gamma_L": {"effective_rank": erank, "shape": [args.width, args.width]},
        "layers": layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"saved={args.out}", flush=True)


if __name__ == "__main__":
    main()
