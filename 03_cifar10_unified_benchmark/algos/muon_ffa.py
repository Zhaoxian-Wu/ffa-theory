"""Strictly local Vanilla FFA trained with a Muon--AdamW optimizer pair."""
from __future__ import annotations

import time
from typing import List

import torch
import torch.nn as nn

from algos.strict_local_ffa import LocalFFAHead, evaluate, local_ffa_loss
from algos.vanilla_ffa import _apply_overlay_image, _random_wrong_labels
from common import (
    ConvBlock, FAIR_DEFAULT_ARCH, FAIR_LR, TrainResult, apply_lr_decay,
    count_params, get_cifar10_loaders, make_conv_blocks, now_iso, register,
    set_seed,
)
from muon_optimizer import LocalMuonOptimizer


class MuonFFAModel(nn.Module):
    """Strict-local FFA stack whose each local unit owns a Muon optimizer pair."""

    def __init__(
        self, arch: str, adam_lr: float, muon_lr: float, hidden_dim: int,
        momentum: float, ns_steps: int,
    ) -> None:
        super().__init__()
        specs = make_conv_blocks(arch)
        self.blocks = nn.ModuleList([ConvBlock(spec) for spec in specs])
        self.heads = nn.ModuleList([
            LocalFFAHead(spec["out_ch"], hidden_dim) for spec in specs
        ])
        self.optimizers = [
            LocalMuonOptimizer(
                list(block.parameters()) + list(head.parameters()),
                muon_lr=muon_lr,
                adam_lr=adam_lr,
                momentum=momentum,
                ns_steps=ns_steps,
            )
            for block, head in zip(self.blocks, self.heads)
        ]


@register("muon_ffa")
def train(cfg: dict) -> TrainResult:
    """Train the table's strict-local FFA protocol with local Muon updates."""
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = int(cfg.get("epochs", 200))
    adam_lr = float(cfg.get("lr", FAIR_LR))
    muon_lr = float(cfg.get("muon_lr", adam_lr))
    batch_size = int(cfg.get("batch_size", 128))
    arch = str(cfg.get("arch", FAIR_DEFAULT_ARCH))
    hidden_dim = int(cfg.get("local_head_dim", 256))
    momentum = float(cfg.get("muon_momentum", 0.95))
    ns_steps = int(cfg.get("muon_ns_steps", 5))
    lr_decay_epoch = int(cfg.get("lr_decay_epoch", 100))
    lr_decay_factor = float(cfg.get("lr_decay_factor", 0.1))

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, augment=False,
    )
    model = MuonFFAModel(
        arch=arch, adam_lr=adam_lr, muon_lr=muon_lr, hidden_dim=hidden_dim,
        momentum=momentum, ns_steps=ns_steps,
    ).to(device)
    acc_curve: List[float] = []
    loss_curve: List[float] = []
    best_acc = 0.0
    start = time.time()

    for epoch in range(epochs):
        apply_lr_decay(
            model.optimizers, epoch, muon_lr, at_epoch=lr_decay_epoch,
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
                f"  [MuonFFA/{arch}] epoch {epoch + 1}/{epochs} "
                f"loss={average_loss:.4f} test_acc={accuracy:.4f} best={best_acc:.4f}",
                flush=True,
            )

    return TrainResult(
        algo="muon_ffa", status="ok", arch=arch,
        test_acc_final=acc_curve[-1], test_acc_best=best_acc,
        test_acc_curve=acc_curve, train_loss_curve=loss_curve,
        n_params=count_params(model), elapsed_s=time.time() - start,
        hyperparams={
            "lr": adam_lr,
            "muon_lr": muon_lr,
            "muon_momentum": momentum,
            "muon_ns_steps": ns_steps,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Muon(matrix/convolution) + AdamW(bias/per-block)",
            "arch": arch,
            "n_blocks": len(model.blocks),
            "local_head_dim": hidden_dim,
            "goodness": "mean_sq(local-head output)",
            "loss": "-log sigmoid(g_pos-theta) - log sigmoid(theta-g_neg)",
            "theta": "dynamic-midpoint",
            "overlay": "label-overlay",
            "negative": "random-wrong-label",
            "locality": "strict; each block input detached; one optimizer pair per block",
            "conv_muon_matrix_view": "(out_channels, -1)",
            "eval": "10-way argmax over sum of all block goodnesses",
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": False,
            "bn_status": "native_ln_kept (per-block FFA head)",
            "seed": int(cfg.get("seed", 0)),
        },
        gpu=str(device), timestamp=now_iso(),
    )
