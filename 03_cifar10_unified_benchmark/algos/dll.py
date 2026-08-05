"""
DLL: Dendritic Localized Learning (ICML 2025, simplified).

Ref: Mikulik et al., "Dendritic Localized Learning" ICML 2025.

Every block trains on (i) a local supervised CE signal and (ii) a predictive
signal in which a "dendritic branch" predicts the next block's activation.
Gradients do NOT flow across block boundaries.

Per block ell (layout from `make_conv_blocks(arch)`, so the per-block channel
flow matches CNN3Backbone / CNN6Backbone exactly):
  h_ell = block_ell(h_{ell-1}.detach())
  L_sup_ell = CE(aux_head_ell(h_ell), y)
  L_pred_ell = ||pred_head_ell(h_ell) - h_{ell+1}.detach()||^2   (only if ell < last)
  L_ell = L_sup_ell + lambda_pred * L_pred_ell
Each block + its heads has its own Adam optimizer.

Evaluation: argmax of the deepest block's aux_head (matches Nokland L_pred).

Fair contract: Adam(lr=1e-3) per block, batch_size=128, epochs=200,
no augmentation, seed=0, arch-parameterised (cnn3 | cnn6).
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


# ================================================================
# Heads
# ================================================================

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


class _PredHead(nn.Module):
    """Dendritic predictive branch predicting the next block's activation.

    We simply reuse a `ConvBlock` configured with the *next* block's spec
    (same in_ch as the current block's out_ch, same kernel/padding/pool as
    the next block). Because ConvBlock is Conv+ReLU(+pool), its output
    matches next-block activations in both shape and non-negativity domain.
    """

    def __init__(self, next_spec: dict, use_bn: bool = False):
        super().__init__()
        # next_spec["in_ch"] already equals current block's out_ch by
        # construction of make_conv_blocks, so ConvBlock(next_spec) is the
        # structurally-matched predictor.
        self.block = ConvBlock(next_spec, use_bn=use_bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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

@register("dll")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    batch_size = cfg.get("batch_size", 128)
    lambda_pred = float(cfg.get("lambda_pred", 0.1))
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
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

    # Predictive heads: one per block ell in {0, ..., n_blocks - 2}.
    # pred_head ell maps block ell's output to block (ell+1)'s activation
    # shape by running ConvBlock(next_spec).
    pred_heads = nn.ModuleList([
        _PredHead(specs[ell + 1], use_bn=use_bn) for ell in range(n_blocks - 1)
    ]).to(device)

    # One Adam per (block + aux_head + pred_head if any): gradients stay local.
    optimizers = []
    for ell, (block, aux) in enumerate(zip(blocks, aux_heads)):
        params = list(block.parameters()) + list(aux.parameters())
        if ell < len(pred_heads):
            params += list(pred_heads[ell].parameters())
        optimizers.append(optim.Adam(params, lr=lr))

    acc_curve: List[float] = []
    loss_curve: List[float] = []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
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

            # Forward all blocks with inter-block detach: each acts[ell] is
            # attached ONLY to block_ell's graph, so per-block backward is independent.
            acts: List[torch.Tensor] = []
            h = x
            for block in blocks:
                h = block(h.detach())
                acts.append(h)

            block_losses = []
            for ell in range(n_blocks):
                loss = F.cross_entropy(aux_heads[ell](acts[ell]), y)
                if ell < len(pred_heads):
                    pred = pred_heads[ell](acts[ell])
                    target = acts[ell + 1].detach()
                    if pred.shape != target.shape:
                        raise RuntimeError(
                            f"DLL pred/target shape mismatch at block {ell}: "
                            f"{tuple(pred.shape)} vs {tuple(target.shape)}"
                        )
                    loss = loss + lambda_pred * F.mse_loss(pred, target)
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
            print(f"  [DLL] epoch {epoch + 1}/{epochs}  "
                  f"mean_block_loss={avg_loss:.4f}  test_acc={acc:.4f}  "
                  f"best={best_acc:.4f}", flush=True)

    n_params = (count_params(blocks) + count_params(aux_heads) + count_params(pred_heads))

    return TrainResult(
        algo="dll", status="ok", arch=arch,
        test_acc_final=acc_curve[-1] if acc_curve else None,
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=n_params,
        elapsed_s=time.time() - t0,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "Adam(per-block)",
            "lambda_pred": lambda_pred,
            "aux_head": "Conv+ReLU+GAP+Linear",
            "pred_head": "ConvBlock(next_spec) (maps block ell -> ell+1 activation shape)",
            "loss_per_block": "CE + lambda_pred * MSE(pred, next_act.detach())",
            "eval": "argmax of deepest block's aux_head",
            "seed": cfg.get("seed", 0),
            "arch": arch,
            "n_blocks": n_blocks,
            "block_channels": [s["out_ch"] for s in specs],
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": use_bn,
            "bn_status": "added_to_convblock",
        },
        gpu=str(device), timestamp=now_iso(),
    )
