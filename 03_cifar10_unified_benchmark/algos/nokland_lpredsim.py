"""
Nokland L_predsim: per-layer combined local loss beta*L_pred + (1-beta)*L_sim.

Reference: Nokland & Eidnes, "Training Neural Networks with Local Error Signals",
ICML 2019 (papers/nokland2019_local_error.pdf, eq. (7) in Section 3.4). Our beta
convention follows the task spec: beta weights L_pred so beta=0.99 => L_pred
dominates. Each of the 3 conv blocks is trained ONLY by its own local loss;
activations are detached between blocks. Evaluation argmaxes the last block's
L_pred head. Self-contained — no dependency on nokland_base.py.

Fair-comparison contract: Adam(lr=1e-3), batch_size=128, no augmentation, seed=0.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    ConvBlock, FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, TrainResult,
    apply_lr_decay, count_params, get_cifar10_loaders, make_conv_blocks,
    now_iso, register, set_seed,
)


class LPredHead(nn.Module):
    """Local L_pred classifier: Conv(C,C,3) + ReLU + GAP(1) + Linear(C, n_classes)."""

    def __init__(self, channels: int, num_classes: int = FAIR_NUM_CLASSES):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.conv(h))
        z = self.gap(z).flatten(1)
        return self.fc(z)


# ================================================================
# L_sim: similarity-matching loss on globally-average-pooled features.
# ================================================================

def _adjusted_cosine_similarity(m: torch.Tensor) -> torch.Tensor:
    """Row-wise mean-centered cosine similarity matrix (eq. (5) in the paper).

    m : (B, D) feature matrix
    returns S : (B, B)  where S[i,j] = <m_i - mean(m_i), m_j - mean(m_j)> /
                                        (||m_i - mean(m_i)|| * ||m_j - mean(m_j)||)
    """
    m_c = m - m.mean(dim=1, keepdim=True)
    m_n = F.normalize(m_c, p=2, dim=1, eps=1e-8)
    return m_n @ m_n.t()


def _sim_loss(h: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """L_sim = || S(GAP(H)) - S(Y) ||_F^2 / B^2 (normalized Frobenius)."""
    # Global average pool feature maps to a (B, C) vector, then similarity.
    feats = F.adaptive_avg_pool2d(h, 1).flatten(1)
    s_h = _adjusted_cosine_similarity(feats)
    s_y = _adjusted_cosine_similarity(y_onehot.float())
    b = h.shape[0]
    return ((s_h - s_y) ** 2).sum() / (b * b)


# ================================================================
# Training loop
# ================================================================

@register("nokland_lpredsim")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    batch_size = cfg.get("batch_size", 128)
    beta = cfg.get("beta", 0.99)  # weight of L_pred (task spec convention)
    num_classes = FAIR_NUM_CLASSES
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 recipe: LR decay + BN inside ConvBlock (applied equally to all algos).
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)
    use_bn = cfg.get("use_bn", True)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, augment=False,
    )

    # Arch-parameterised conv blocks + one L_pred head per block.
    specs = make_conv_blocks(arch)
    blocks: List[nn.Module] = []
    heads: List[LPredHead] = []
    optimizers: List[optim.Optimizer] = []
    for spec in specs:
        block = ConvBlock(spec, use_bn=use_bn).to(device)
        head = LPredHead(spec["out_ch"], num_classes).to(device)
        blocks.append(block)
        heads.append(head)
        optimizers.append(optim.Adam(
            list(block.parameters()) + list(head.parameters()), lr=lr,
        ))

    ce = nn.CrossEntropyLoss()
    acc_curve: List[float] = []
    loss_curve: List[float] = []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        apply_lr_decay(optimizers, epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor,
                       verbose=(epoch == lr_decay_epoch))
        for b in blocks: b.train()
        for h in heads: h.train()

        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_onehot = F.one_hot(y, num_classes=num_classes).float()

            h_in = x
            batch_loss = 0.0
            for i in range(len(blocks)):
                # Forward through block i (input is detached so no gradient flows
                # back into previously-trained blocks).
                h_out = blocks[i](h_in.detach())

                logits = heads[i](h_out)
                l_pred = ce(logits, y)
                l_sim = _sim_loss(h_out, y_onehot)
                loss_i = beta * l_pred + (1.0 - beta) * l_sim

                optimizers[i].zero_grad()
                loss_i.backward()
                optimizers[i].step()

                batch_loss += float(loss_i.detach())
                # Feed (detached) activation to the next block.
                h_in = h_out.detach()

            running += batch_loss / len(blocks)
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        loss_curve.append(avg_loss)

        # Evaluate with the *last* block's L_pred head.
        for b in blocks: b.eval()
        for h in heads: h.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                h_in = x
                for i in range(len(blocks)):
                    h_in = blocks[i](h_in)
                logits = heads[-1](h_in)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.shape[0]
        acc = correct / max(total, 1)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [Nokland L_predsim beta={beta}] epoch {epoch + 1}/{epochs}  "
                  f"loss={avg_loss:.4f}  test_acc={acc:.4f}  best={best_acc:.4f}",
                  flush=True)

    n_params = sum(count_params(m) for m in blocks) + sum(count_params(m) for m in heads)

    return TrainResult(
        algo="nokland_lpredsim", status="ok", arch=arch,
        test_acc_final=acc_curve[-1],
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=n_params,
        elapsed_s=time.time() - t0,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": batch_size,
                     "optimizer": "Adam-per-block", "beta": beta,
                     "loss": "beta*L_pred + (1-beta)*L_sim",
                     "arch": arch, "n_blocks": len(blocks),
                     "block_channels": [s["out_ch"] for s in specs],
                     "eval_head": "last-block L_pred",
                     "seed": cfg.get("seed", 0),
                     "lr_decay_epoch": lr_decay_epoch,
                     "lr_decay_factor": lr_decay_factor,
                     "use_bn": use_bn,
                     "bn_status": "added_to_convblock"},
        gpu=str(device), timestamp=now_iso(),
    )
