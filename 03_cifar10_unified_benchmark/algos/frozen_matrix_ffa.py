"""Strict-local FFA with a fixed identity matrix before each goodness."""
from __future__ import annotations

import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    ConvBlock, FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, TrainResult,
    apply_lr_decay, count_params, get_cifar10_loaders, make_conv_blocks,
    now_iso, register, set_seed,
)
from algos.strict_local_ffa import LocalFFAHead, local_ffa_loss
from algos.vanilla_ffa import _apply_overlay_image, _random_wrong_labels


class FixedGoodnessMatrix(nn.Module):
    """A fixed random map retained explicitly in the goodness path."""

    def __init__(self, dimension: int):
        super().__init__()
        self.linear = nn.Linear(dimension, dimension, bias=False)
        self.linear.weight.requires_grad_(False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class FrozenMatrixFFAModel(nn.Module):
    """Independent local FFA units with one fixed U before each goodness."""

    def __init__(self, arch: str, lr: float, hidden_dim: int = 256):
        super().__init__()
        self.specs = make_conv_blocks(arch)
        self.blocks = nn.ModuleList([ConvBlock(spec) for spec in self.specs])
        self.heads = nn.ModuleList([
            LocalFFAHead(spec["out_ch"], hidden_dim) for spec in self.specs
        ])
        self.goodness_matrices = nn.ModuleList([
            FixedGoodnessMatrix(hidden_dim) for _ in self.specs
        ])
        self.optimizers = [
            optim.Adam(list(block.parameters()) + list(head.parameters()), lr=lr)
            for block, head in zip(self.blocks, self.heads)
        ]
        self.hidden_dim = hidden_dim


def _goodness(head: LocalFFAHead, matrix: FixedGoodnessMatrix, h: torch.Tensor) -> torch.Tensor:
    """Compute goodness(U z) with a fixed identity U."""
    return head.goodness(matrix(head(h)))


@torch.no_grad()
def evaluate(model: FrozenMatrixFFAModel, loader, device: torch.device) -> float:
    """Use the benchmark's ten-way sum of local U-goodness scores."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = x.shape[0]
        best_score = torch.full((batch_size,), -float("inf"), device=device)
        best_class = torch.zeros(batch_size, dtype=torch.long, device=device)
        for candidate in range(FAIR_NUM_CLASSES):
            labels = torch.full((batch_size,), candidate, dtype=torch.long, device=device)
            h = _apply_overlay_image(x, labels)
            score = torch.zeros(batch_size, device=device)
            for block, head, matrix in zip(
                model.blocks, model.heads, model.goodness_matrices,
            ):
                h = block(h)
                score.add_(_goodness(head, matrix, h))
            choose = score > best_score
            best_score[choose] = score[choose]
            best_class[choose] = candidate
        correct += (best_class == y).sum().item()
        total += batch_size
    return correct / max(total, 1)


@register("frozen_matrix_ffa")
def train(cfg: dict) -> TrainResult:
    """Train the matched fixed-U control for Matrix-FFA."""
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = int(cfg.get("epochs", 200))
    lr = float(cfg.get("lr", FAIR_LR))
    batch_size = int(cfg.get("batch_size", 128))
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    hidden_dim = int(cfg.get("local_head_dim", 256))
    lr_decay_epoch = int(cfg.get("lr_decay_epoch", 100))
    lr_decay_factor = float(cfg.get("lr_decay_factor", 0.1))

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, augment=False)
    model = FrozenMatrixFFAModel(arch=arch, lr=lr, hidden_dim=hidden_dim).to(device)
    acc_curve: List[float] = []
    loss_curve: List[float] = []
    best_acc = 0.0
    start = time.time()

    for epoch in range(epochs):
        apply_lr_decay(model.optimizers, epoch, lr, at_epoch=lr_decay_epoch,
                       factor=lr_decay_factor, verbose=(epoch == lr_decay_epoch))
        model.train()
        loss_sum = 0.0
        batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            h_pos = _apply_overlay_image(x, y)
            h_neg = _apply_overlay_image(x, _random_wrong_labels(y))
            local_losses: List[float] = []
            for block, head, matrix, optimizer in zip(
                model.blocks, model.heads, model.goodness_matrices, model.optimizers,
            ):
                h_pos_out = block(h_pos.detach())
                h_neg_out = block(h_neg.detach())
                loss = local_ffa_loss(
                    _goodness(head, matrix, h_pos_out),
                    _goodness(head, matrix, h_neg_out),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(block.parameters()) + list(head.parameters()), 1.0,
                )
                optimizer.step()
                local_losses.append(float(loss.detach()))
                h_pos, h_neg = h_pos_out.detach(), h_neg_out.detach()
            loss_sum += sum(local_losses) / max(len(local_losses), 1)
            batches += 1

        average_loss = loss_sum / max(batches, 1)
        loss_curve.append(average_loss)
        accuracy = evaluate(model, test_loader, device)
        acc_curve.append(accuracy)
        best_acc = max(best_acc, accuracy)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  [FrozenMatrixFFA/{arch}] epoch {epoch + 1}/{epochs} "
                f"loss={average_loss:.4f} test_acc={accuracy:.4f} best={best_acc:.4f}",
                flush=True,
            )

    return TrainResult(
        algo="frozen_matrix_ffa", status="ok", arch=arch,
        test_acc_final=acc_curve[-1], test_acc_best=best_acc,
        test_acc_curve=acc_curve, train_loss_curve=loss_curve,
        n_params=count_params(model), elapsed_s=time.time() - start,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "Adam(per-block)", "arch": arch,
            "n_blocks": len(model.blocks), "local_head_dim": hidden_dim,
            "goodness": "mean_sq(U @ local-head output)",
            "goodness_feature_map": "fixed random square matrix before goodness",
            "matrix_shape": [hidden_dim, hidden_dim],
            "matrix_initialization": "default nn.Linear random",
            "matrix_trainable": False,
            "loss": "-log sigmoid(g_pos-theta) - log sigmoid(theta-g_neg)",
            "theta": "dynamic-midpoint", "overlay": "label-overlay",
            "negative": "random-wrong-label",
            "locality": "strict; each block input detached; one optimizer per block",
            "eval": "10-way argmax over sum of all block goodnesses(Uz)",
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": False,
            "bn_status": "native_ln_kept (per-block FFA head)",
            "seed": cfg.get("seed", 0),
        },
        gpu=str(device), timestamp=now_iso(),
    )
