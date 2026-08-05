"""Strict-local FFA with an identity-initialized matrix before each goodness."""
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


class MatrixFFAModel(nn.Module):
    """Independent block/head units with one learned U per local goodness."""

    def __init__(self, arch: str, lr: float, hidden_dim: int = 256):
        super().__init__()
        self.specs = make_conv_blocks(arch)
        self.blocks = nn.ModuleList([ConvBlock(spec) for spec in self.specs])
        self.heads = nn.ModuleList([
            LocalFFAHead(spec["out_ch"], hidden_dim) for spec in self.specs
        ])
        self.goodness_matrices = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in self.specs
        ])
        self.optimizers = [
            optim.Adam(
                list(block.parameters()) + list(head.parameters()) + list(matrix.parameters()),
                lr=lr,
            )
            for block, head, matrix in zip(self.blocks, self.heads, self.goodness_matrices)
        ]
        self.hidden_dim = hidden_dim


def _goodness(head: LocalFFAHead, matrix: nn.Linear, h: torch.Tensor) -> torch.Tensor:
    """Compute goodness(U z), where z is this block's detached-local feature."""
    return head.goodness(matrix(head(h)))


@torch.no_grad()
def evaluate(model: MatrixFFAModel, loader, device: torch.device) -> float:
    """Use the standard label-overlay ten-way sum of all local goodness scores."""
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


@register("matrix_ffa")
def train(cfg: dict) -> TrainResult:
    """Train strict-local FFA with goodness(U z) at every CNN block."""
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
    model = MatrixFFAModel(arch=arch, lr=lr, hidden_dim=hidden_dim).to(device)
    acc_curve: List[float] = []
    loss_curve: List[float] = []
    best_acc = 0.0
    start = time.time()

    for epoch in range(epochs):
        apply_lr_decay(
            model.optimizers, epoch, lr, at_epoch=lr_decay_epoch,
            factor=lr_decay_factor, verbose=(epoch == lr_decay_epoch),
        )
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
                    list(block.parameters()) + list(head.parameters()) + list(matrix.parameters()),
                    1.0,
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
                f"  [MatrixFFA/{arch}] epoch {epoch + 1}/{epochs} "
                f"loss={average_loss:.4f} test_acc={accuracy:.4f} best={best_acc:.4f}",
                flush=True,
            )

    return TrainResult(
        algo="matrix_ffa", status="ok", arch=arch,
        test_acc_final=acc_curve[-1], test_acc_best=best_acc,
        test_acc_curve=acc_curve, train_loss_curve=loss_curve,
        n_params=count_params(model), elapsed_s=time.time() - start,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "Adam(per-block)", "arch": arch,
            "n_blocks": len(model.blocks), "local_head_dim": hidden_dim,
            "goodness": "mean_sq(U @ local-head output)",
            "goodness_feature_map": "learned identity-initialized square matrix before goodness",
            "loss": "-log sigmoid(g_pos-theta) - log sigmoid(theta-g_neg)",
            "theta": "dynamic-midpoint", "overlay": "label-overlay",
            "negative": "random-wrong-label",
            "locality": "strict; each block input detached; one optimizer per block",
            "eval": "10-way argmax over sum of all block goodnesses(Uz)",
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": False,
            "bn_status": "native_ln_kept (per-block FFA head)",
            "matrix_shape": [hidden_dim, hidden_dim],
            "matrix_initialization": "identity",
            "seed": cfg.get("seed", 0),
        },
        gpu=str(device), timestamp=now_iso(),
    )
