"""
PC: Predictive Coding (Millidge et al. 2022, simplified layer-wise variant).

Ref: Millidge, Tschantz, Buckley. "Predictive Coding Approximates Backprop
along Arbitrary Computation Graphs." Neural Computation 2022.
     Whittington & Bogacz. Neural Computation 2017.

Full PC alternates (i) inference: iterate activations to minimize the free
energy F = sum_ell ||h_ell - mu_ell||^2; and (ii) learning: update weights
with activations fixed. That two-phase loop is expensive and brittle, so we
adopt the simplified layer-wise form from Millidge 2022 (Sec. 3) that
collapses PC into an auto-encoder style local objective:

Per block ell (layout from `make_conv_blocks(arch)`; per-block channel flow
matches CNN3Backbone / CNN6Backbone exactly):
  h_ell   = block_ell(h_{ell-1}.detach())          # bottom-up activation
  mu_ell  = pred_head_ell(h_{ell-1}.detach())      # top-down prediction
  L_ell   = alpha * CE(aux_head_ell(h_ell), y)
          + beta  * ||h_ell - mu_ell||^2           # prediction error (ell>=1)
Each block + its heads has its own Adam; gradients do NOT cross blocks.

The PC energy term flows top-down (pred_head maps h_{ell-1} -> h_ell),
contrast with DLL whose pred_head flows forward (h_ell -> h_{ell+1}).

Eval: argmax of deepest block's aux_head. Fair contract: Adam(lr=1e-3) per
block, batch_size=128, epochs=200, no augmentation, seed=0, arch-parameterised
(cnn3 | cnn6). Tag `underperforms_at_small_depth` if best_acc < 30%.
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


class _AuxHead(nn.Module):
    """Local supervised classifier: Conv(C,C,3) + ReLU + GAP(1) + Linear(C, num_classes)."""

    def __init__(self, channels: int, num_classes: int = FAIR_NUM_CLASSES):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(feat))
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(h)


# Top-down predictor: maps h_{ell-1} -> mu_ell. We reuse ConvBlock(specs[ell])
# so the structure (Conv + ReLU + optional pool) matches the forward block ell
# exactly; this guarantees mu and h have identical shape and non-negative domain.
_PredHead = ConvBlock


# ================================================================
# Evaluation
# ================================================================

@torch.no_grad()
def _evaluate(blocks, aux_heads, loader, device) -> float:
    for b in blocks:
        b.eval()
    for a in aux_heads:
        a.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        h = x
        for b in blocks:
            h = b(h)
        logits = aux_heads[-1](h)  # deepest aux head
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    return correct / max(total, 1)


# ================================================================
# Main training routine
# ================================================================

@register("predictive_coding")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    batch_size = cfg.get("batch_size", 128)
    alpha = float(cfg.get("alpha", 1.0))      # weight on supervised CE
    beta = float(cfg.get("beta", cfg.get("lambda_pred", 0.1)))  # PC prediction-error weight
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 recipe knobs.
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)
    use_bn = cfg.get("use_bn", True)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, augment=False,
    )

    # Arch-parameterised conv blocks + per-block aux heads.
    specs = make_conv_blocks(arch)
    n_blocks = len(specs)

    blocks = nn.ModuleList([ConvBlock(s, use_bn=use_bn) for s in specs]).to(device)
    aux_heads = nn.ModuleList([_AuxHead(s["out_ch"]) for s in specs]).to(device)

    # Predictive heads: only for blocks ell >= 1 (no previous layer at ell=0).
    # pred_heads[ell] maps h_{ell-1} -> mu_ell and has the same structure as
    # block_ell (ConvBlock(specs[ell])). ell=0 slot is an unused placeholder.
    pred_heads = nn.ModuleList([
        nn.Identity() if ell == 0 else _PredHead(specs[ell], use_bn=use_bn)
        for ell in range(n_blocks)
    ]).to(device)

    # One Adam per (block + aux_head + pred_head if any): gradients stay local.
    optimizers = []
    for ell, (block, aux) in enumerate(zip(blocks, aux_heads)):
        params = list(block.parameters()) + list(aux.parameters())
        if ell >= 1:
            params += list(pred_heads[ell].parameters())
        optimizers.append(optim.Adam(params, lr=lr))

    acc_curve: List[float] = []
    loss_curve: List[float] = []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        # Recipe: lr decay on all per-block Adam optimizers.
        apply_lr_decay(optimizers, epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor)
        for b in blocks:
            b.train()
        for a in aux_heads:
            a.train()
        for p in pred_heads:
            p.train()

        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward with inter-block detach: acts[ell] is attached ONLY to
            # block_ell's graph; prev_detached[ell] = detach of block_ell's input.
            acts: List[torch.Tensor] = []
            prev_detached: List[torch.Tensor] = []
            h = x
            for block in blocks:
                h_in = h.detach()
                prev_detached.append(h_in)
                h = block(h_in)
                acts.append(h)

            block_losses = []
            for ell in range(n_blocks):
                loss = alpha * F.cross_entropy(aux_heads[ell](acts[ell]), y)
                if ell >= 1:
                    mu = pred_heads[ell](prev_detached[ell])
                    if mu.shape != acts[ell].shape:
                        raise RuntimeError(
                            f"PC mu/h shape mismatch at block {ell}: "
                            f"{tuple(mu.shape)} vs {tuple(acts[ell].shape)}"
                        )
                    # Detach target h so pred_head fits h; gradients through h
                    # still flow from L_aux into the block.
                    loss = loss + beta * F.mse_loss(mu, acts[ell].detach())
                optimizers[ell].zero_grad()
                loss.backward()
                optimizers[ell].step()
                block_losses.append(loss.item())

            running += sum(block_losses) / len(block_losses)
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        loss_curve.append(avg_loss)

        acc = _evaluate(blocks, aux_heads, test_loader, device)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [PC] epoch {epoch + 1}/{epochs}  "
                  f"mean_block_loss={avg_loss:.4f}  test_acc={acc:.4f}  "
                  f"best={best_acc:.4f}", flush=True)

    n_params = (count_params(blocks) + count_params(aux_heads) + count_params(pred_heads))

    tag = None
    if best_acc < 0.30:
        tag = "underperforms_at_small_depth"

    return TrainResult(
        algo="predictive_coding", status="ok", arch=arch,
        test_acc_final=acc_curve[-1] if acc_curve else None,
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=n_params,
        elapsed_s=time.time() - t0,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "Adam(per-block)",
            "alpha": alpha, "beta": beta,
            "lambda_pred": beta,   # back-compat alias
            "aux_head": "Conv+ReLU+GAP+Linear",
            "pred_head": "ConvBlock(specs[ell]) (top-down: maps h_{ell-1} -> mu_ell)",
            "loss_per_block": "alpha*CE + beta*MSE(mu, h.detach()) [ell>=1]",
            "eval": "argmax of deepest block's aux_head",
            "seed": cfg.get("seed", 0),
            "arch": arch,
            "n_blocks": n_blocks,
            "block_channels": [s["out_ch"] for s in specs],
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": use_bn,
            "bn_status": "added_to_convblock" if use_bn else "disabled",
        },
        gpu=str(device), timestamp=now_iso(),
        tag=tag,
    )
