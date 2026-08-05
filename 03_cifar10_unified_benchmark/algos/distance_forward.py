"""
Distance-Forward (DF / DF-Orthogonal, Wu 2024, arXiv:2408.14925).

Per-layer cosine-distance classifier replacing FFA's squared-norm goodness:
each block emits an L2-normalised embedding h_l and owns K=10 learnable
L2-normalised prototypes p_{l,k}; logits = tau * cos(h_l, p_{l,k}) fed into
CE loss (DF-Orthogonal simplification: CE on normalised features decorrelates
prototypes automatically, no explicit orthogonality penalty).

Locality: the shared backbone is split block-by-block via
`make_conv_blocks(arch)`; activations are detached between blocks and each
block owns its own head + Adam optimiser. All blocks train simultaneously
each epoch (not greedy sequential like Belilovsky). Evaluation uses the last
block's cosine argmax.

Fair contract: Adam(lr=1e-3) per block, batch_size=128, epochs=200, no
augmentation, seed=0, arch-parameterised (cnn3 | cnn6).
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


class DFHead(nn.Module):
    """DF-Orthogonal head: GAP -> Linear(C, embed_dim) -> L2-normalise,
    logits = tau * cosine(h, p_k) against K learnable unit prototypes."""

    def __init__(self, channels: int, embed_dim: int = 64,
                 num_classes: int = FAIR_NUM_CLASSES, tau: float = 10.0):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels, embed_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.1)
        self.tau = tau

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = F.normalize(self.proj(self.gap(feat).flatten(1)), dim=1, eps=1e-8)
        p = F.normalize(self.prototypes, dim=1, eps=1e-8)
        return self.tau * (h @ p.t())


# ================================================================
# Main training routine
# ================================================================

@register("distance_forward")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    batch_size = cfg.get("batch_size", 128)
    tau = cfg.get("tau", 10.0)
    embed_dim = cfg.get("embed_dim", 64)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)
    use_bn = cfg.get("use_bn", True)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, augment=False,
    )

    # Arch-parameterised conv blocks matching CNN3/CNN6 backbone layout.
    specs = make_conv_blocks(arch)
    n_blocks = len(specs)

    blocks: List[nn.Module] = []
    heads: List[DFHead] = []
    optimizers: List[optim.Optimizer] = []
    for s in specs:
        blk = ConvBlock(s, use_bn=use_bn).to(device)
        head = DFHead(s["out_ch"], embed_dim=embed_dim, tau=tau).to(device)
        blocks.append(blk)
        heads.append(head)
        # One Adam optimiser per block: only the block's own params get
        # gradients from that block's CE loss (upstream blocks are detached).
        optimizers.append(optim.Adam(
            list(blk.parameters()) + list(head.parameters()), lr=lr,
        ))

    criterion = nn.CrossEntropyLoss()

    def _forward_last(x: torch.Tensor) -> torch.Tensor:
        """Inference path: all blocks in eval, classify from last block's head."""
        h = x
        for b in blocks:
            h = b(h)
        return heads[-1](h)

    @torch.no_grad()
    def _eval() -> float:
        for b in blocks:
            b.eval()
        for h in heads:
            h.eval()
        correct, total = 0, 0
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = _forward_last(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
        return correct / max(total, 1)

    acc_curve: List[float] = []
    loss_curve: List[float] = []
    t0 = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        apply_lr_decay(optimizers, epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor)
        for b in blocks:
            b.train()
        for h in heads:
            h.train()

        running = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Each block sees a detached copy of the previous block's output,
            # so gradients from block k's CE loss stay inside block k.
            h = x
            per_block_loss_sum = 0.0
            for blk, head, opt in zip(blocks, heads, optimizers):
                h = blk(h)
                logits = head(h)
                loss = criterion(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                per_block_loss_sum += loss.item()
                h = h.detach()   # cut gradient flow to the next block

            running += per_block_loss_sum / n_blocks
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        loss_curve.append(avg_loss)

        acc = _eval()
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(
                f"  [DF] epoch {epoch + 1}/{epochs}  "
                f"loss={avg_loss:.4f}  test_acc={acc:.4f}  best={best_acc:.4f}",
                flush=True,
            )

    total_params = sum(count_params(m) for m in blocks) + sum(count_params(m) for m in heads)

    return TrainResult(
        algo="distance_forward", status="ok", arch=arch,
        test_acc_final=acc_curve[-1] if acc_curve else None,
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=total_params,
        elapsed_s=time.time() - t0,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "Adam-per-block", "seed": cfg.get("seed", 0),
            "tau": tau, "embed_dim": embed_dim,
            "n_blocks": n_blocks,
            "goodness": "cosine-to-prototype",
            "locality": "detach-between-blocks",
            "arch": arch,
            "block_channels": [s["out_ch"] for s in specs],
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": use_bn,
            "bn_status": "added_to_convblock",
        },
        gpu=str(device), timestamp=now_iso(),
    )
