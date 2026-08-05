"""Strict layer-local Vanilla FFA for the CIFAR-10 CNN benchmark.

This module is the locality-correct replacement for the legacy end-to-end
``vanilla_ffa`` benchmark entry.  Every convolutional block owns a separate
goodness head and Adam optimizer.  The activation passed to the next block is
detached, so no loss differentiates through another block.
"""
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
from algos.vanilla_ffa import _apply_overlay_image, _random_wrong_labels


class LocalFFAHead(nn.Module):
    """GAP plus two normalized MLP layers used only by one local block.

    When ``use_goodness_matrix`` is enabled, an identity-initialized linear map
    is applied to the local feature immediately before its goodness is
    computed. The map is owned by the same local unit as the block and head.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 256,
                 use_goodness_matrix: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.goodness_matrix = nn.Linear(hidden_dim, hidden_dim, bias=False) \
            if use_goodness_matrix else nn.Identity()
        if use_goodness_matrix:
            nn.init.eye_(self.goodness_matrix.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(h, 1).reshape(h.shape[0], -1)
        z = torch.relu(self.ln1(self.fc1(pooled)))
        return torch.relu(self.ln2(self.fc2(z)))

    def feature_for_goodness(self, z: torch.Tensor) -> torch.Tensor:
        """Apply the optional local feature map before measuring goodness."""
        return self.goodness_matrix(z)

    @staticmethod
    def goodness(z: torch.Tensor) -> torch.Tensor:
        return z.square().mean(dim=1)


class StrictLocalFFAModel(nn.Module):
    """A CNN block stack with independent local goodness heads."""

    def __init__(self, arch: str, lr: float, hidden_dim: int = 256,
                 trunk_norm: str = "none"):
        super().__init__()
        if trunk_norm not in {"none", "channel_ln"}:
            raise ValueError(f"Unsupported trunk norm: {trunk_norm}")
        self.specs = make_conv_blocks(arch)
        self.blocks = nn.ModuleList([
            ConvBlock(spec, use_channel_ln=(trunk_norm == "channel_ln"))
            for spec in self.specs
        ])
        self.heads = nn.ModuleList([
            LocalFFAHead(spec["out_ch"], hidden_dim) for spec in self.specs
        ])
        self.optimizers = [
            optim.Adam(list(block.parameters()) + list(head.parameters()), lr=lr)
            for block, head in zip(self.blocks, self.heads)
        ]
        self.hidden_dim = hidden_dim

        self.trunk_norm = trunk_norm

def local_ffa_loss(g_pos: torch.Tensor, g_neg: torch.Tensor) -> torch.Tensor:
    """Hinton-style positive/negative goodness loss with a detached midpoint."""
    theta = ((g_pos.mean() + g_neg.mean()) / 2).detach()
    return F.softplus(-(g_pos - theta)).mean() + F.softplus(g_neg - theta).mean()


@torch.no_grad()
def evaluate(model: StrictLocalFFAModel, loader, device: torch.device) -> float:
    """Predict by summing every block's goodness for each overlaid label."""
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
            for block, head in zip(model.blocks, model.heads):
                h = block(h)
                score.add_(head.goodness(head(h)))
            choose = score > best_score
            best_score[choose] = score[choose]
            best_class[choose] = candidate
        correct += (best_class == y).sum().item()
        total += batch_size
    return correct / max(total, 1)


@register("strict_local_ffa")
def train(cfg: dict) -> TrainResult:
    trunk_norm = str(cfg.get("trunk_norm", "none"))
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = int(cfg.get("epochs", 200))
    lr = float(cfg.get("lr", FAIR_LR))
    batch_size = int(cfg.get("batch_size", 128))
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    hidden_dim = int(cfg.get("local_head_dim", 256))
    lr_decay_epoch = int(cfg.get("lr_decay_epoch", 100))
    lr_decay_factor = float(cfg.get("lr_decay_factor", 0.1))

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, augment=False,
    )
    model = StrictLocalFFAModel(arch=arch, lr=lr, hidden_dim=hidden_dim,
                                trunk_norm=trunk_norm).to(device)
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
            for block, head, optimizer in zip(model.blocks, model.heads, model.optimizers):
                # This detach is the strict locality boundary between CNN blocks.
                h_pos_out = block(h_pos.detach())
                h_neg_out = block(h_neg.detach())
                loss = local_ffa_loss(
                    head.goodness(head(h_pos_out)),
                    head.goodness(head(h_neg_out)),
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
                f"  [StrictLocalFFA/{arch}] epoch {epoch + 1}/{epochs} "
                f"loss={average_loss:.4f} test_acc={accuracy:.4f} best={best_acc:.4f}",
                flush=True,
            )

    return TrainResult(
        algo="strict_local_ffa", status="ok", arch=arch,
        test_acc_final=acc_curve[-1], test_acc_best=best_acc,
        test_acc_curve=acc_curve, train_loss_curve=loss_curve,
        n_params=count_params(model), elapsed_s=time.time() - start,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "Adam(per-block)", "arch": arch,
            "n_blocks": len(model.blocks), "local_head_dim": hidden_dim,
            "goodness": "mean_sq(local-head output)",
            "loss": "-log sigmoid(g_pos-theta) - log sigmoid(theta-g_neg)",
            "theta": "dynamic-midpoint", "overlay": "label-overlay",
            "negative": "random-wrong-label",
            "locality": "strict; each block input detached; one optimizer per block",
            "eval": "10-way argmax over sum of all block goodnesses",
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": False,
            "bn_status": "native_ln_kept (per-block FFA head)",
            "trunk_norm": trunk_norm,
            "seed": cfg.get("seed", 0),
        },
        gpu=str(device), timestamp=now_iso(),
    )
