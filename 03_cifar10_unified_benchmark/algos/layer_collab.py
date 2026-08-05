"""Layer Collaboration FFA (Lorberbom et al., NeurIPS 2024) on CNN backbones.

Reference: Lorberbom, Gat, Adi, Schwing, and Hazan, "Layer Collaboration in
the Forward-Forward Algorithm," NeurIPS 2024 (Eq. 6-8 in Sec. 3.2).

FFA design
----------
Each block's post-activation tensor is reduced to a (B, C_ell) vector v_ell
via global average pool (GAP), and its **goodness** is
    G_ell(x, y) = || v_ell ||^2 / C_ell        (mean-squared goodness).

The collaboration offset gamma_{<t} is the standard Lorberbom sum of earlier
layers' goodnesses (Eq. 7 in the paper):
    gamma_{<t}(x, y) = sum_{t' < t} G_{t'}(x, y).

This is accumulated in the forward pass with .detach() so no gradient crosses
block boundaries (local update rule).

Per-layer loss (Eq. 8 of the paper, rewritten for the symmetric sigmoid form):
    L_t = -log sigma( G_t^+ + gamma_{<t}^+ - theta_t )
          -log sigma( theta_t - G_t^- - gamma_{<t}^- )
with a dynamic theta_t set to the running midpoint of positive and negative
goodness-plus-offset (matching our vanilla_ffa convention).

Positive and negative samples use label-overlay (same protocol as vanilla_ffa):
the correct one-hot label is written into the first 10 flattened pixels for
positives; a random wrong label for negatives. Evaluation argmaxes the sum of
(G_ell + gamma_{<ell}) across all blocks when the candidate label c = 0..9 is
overlaid.

Fair-comparison contract
------------------------
Adam(lr=1e-3) per-block, batch_size=128, 200 epochs, no augmentation. Each
block has its own optimizer; the collaboration offset is a detached input,
so each block's update only touches its own parameters.
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


# -------------------------------------------------------------------
# Label-overlay (identical to vanilla_ffa; FFA protocol)
# -------------------------------------------------------------------

def overlay_label(x_flat: torch.Tensor, y: torch.Tensor,
                  num_classes: int = FAIR_NUM_CLASSES) -> torch.Tensor:
    out = x_flat.clone()
    one_hot = torch.zeros(x_flat.shape[0], num_classes,
                          device=x_flat.device, dtype=x_flat.dtype)
    one_hot.scatter_(1, y.unsqueeze(1), 1.0)
    out[:, :num_classes] = one_hot
    return out


def _random_wrong_labels(y: torch.Tensor,
                         num_classes: int = FAIR_NUM_CLASSES) -> torch.Tensor:
    delta = torch.randint(1, num_classes, y.shape, device=y.device)
    return (y + delta) % num_classes


def _apply_overlay_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(x.shape[0], -1)
    x_flat = overlay_label(x_flat, y)
    return x_flat.reshape(x.shape)


# -------------------------------------------------------------------
# Goodness
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Model -- each "layer" in Lorberbom's construction is a conv block + a small
# FFA MLP head. The head takes the GAP(C_ell) vector through Linear + LN + ReLU
# + Linear + LN + ReLU into a hidden_dim vector z_ell, and the goodness is
# G_ell = || z_ell ||^2 / hidden_dim. The head is essential: without it the
# sigmoid-FFA loss saturates near the dynamic theta midpoint because GAP(C_ell)
# alone lacks the per-sample variance needed to separate pos/neg. This
# mirrors vanilla_ffa's FFAHead design; it also matches the role of Lorberbom's
# MLP layers, whose hidden features serve as both transformation and goodness
# source.
# -------------------------------------------------------------------

class _FFAHead(nn.Module):
    def __init__(self, in_ch: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        v = F.adaptive_avg_pool2d(h, 1).reshape(h.shape[0], -1)  # (B, C)
        z = torch.relu(self.ln1(self.fc1(v)))
        z = torch.relu(self.ln2(self.fc2(z)))
        return z                                                  # (B, hidden_dim)


def _goodness_vec(z: torch.Tensor) -> torch.Tensor:
    """Mean-squared goodness on the head output vector z (B, hidden_dim)."""
    return (z ** 2).mean(dim=1)                                   # (B,)


class LayerCollabModel(nn.Module):
    """Conv stack + per-block FFA MLP head. Each (block + head) pair has its
    own Adam; collaboration offset gamma_{<t} is computed externally in the
    training loop (detached scalar per sample)."""

    def __init__(self, arch: str, lr: float, hidden_dim: int = 256):
        super().__init__()
        specs = make_conv_blocks(arch)
        self.blocks = nn.ModuleList([ConvBlock(s) for s in specs])
        self.heads = nn.ModuleList([_FFAHead(s["out_ch"], hidden_dim) for s in specs])
        self.n_blocks = len(self.blocks)
        self.specs = specs
        self.hidden_dim = hidden_dim
        # Per-block independent Adam over (block + head).
        self.optimizers = [
            optim.Adam(list(b.parameters()) + list(h.parameters()), lr=lr)
            for b, h in zip(self.blocks, self.heads)
        ]


# -------------------------------------------------------------------
# Evaluation: 10-way label-overlay argmax on sum_ell (G_ell + gamma_<ell)
# -------------------------------------------------------------------

@torch.no_grad()
def _lc_eval(model: LayerCollabModel, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bsz = x.shape[0]
        best_score = torch.full((bsz,), -float("inf"), device=device)
        best_c = torch.zeros(bsz, dtype=torch.long, device=device)
        for c in range(FAIR_NUM_CLASSES):
            y_c = torch.full((bsz,), c, dtype=torch.long, device=device)
            x_c = _apply_overlay_image(x, y_c)
            h = x_c
            gamma = torch.zeros(bsz, device=device)
            score = torch.zeros(bsz, device=device)
            for block, head in zip(model.blocks, model.heads):
                h = block(h)
                g = _goodness_vec(head(h))
                score = score + (g + gamma)
                gamma = gamma + g
            mask = score > best_score
            best_score[mask] = score[mask]
            best_c[mask] = c
        correct += (best_c == y).sum().item()
        total += bsz
    return correct / max(total, 1)


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------

@register("layer_collab")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    batch_size = cfg.get("batch_size", 128)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 recipe: LR decay only (per-block FFA heads already carry
    # LayerNorm inside the Linear+LN+ReLU stack; no extra BN).
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, augment=False,
    )
    model = LayerCollabModel(arch=arch, lr=lr).to(device)

    acc_curve: List[float] = []
    loss_curve: List[float] = []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        apply_lr_decay(model.optimizers, epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor,
                       verbose=(epoch == lr_decay_epoch))
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_pos = _apply_overlay_image(x, y)
            y_neg = _random_wrong_labels(y)
            x_neg = _apply_overlay_image(x, y_neg)

            bsz = x.shape[0]
            # Collaboration offset accumulators (DETACHED - gradient does not
            # cross block boundaries; this is the hallmark of local updates).
            gamma_pos = torch.zeros(bsz, device=device)
            gamma_neg = torch.zeros(bsz, device=device)
            h_pos = x_pos
            h_neg = x_neg

            batch_loss_sum = 0.0
            for ell, (block, head) in enumerate(zip(model.blocks, model.heads)):
                # Forward with detached input so only (block ell, head ell)'s
                # parameters receive gradient from this block's local loss.
                h_pos_out = block(h_pos.detach())
                h_neg_out = block(h_neg.detach())
                z_pos = head(h_pos_out)                  # (B, hidden_dim)
                z_neg = head(h_neg_out)
                g_pos = _goodness_vec(z_pos)             # (B,), attached
                g_neg = _goodness_vec(z_neg)

                # Compose score = G_ell + gamma_{<ell}. gamma_<ell is detached.
                score_pos = g_pos + gamma_pos            # attached to block ell only
                score_neg = g_neg + gamma_neg
                # Dynamic theta: running midpoint (detached so it's a constant).
                theta = ((score_pos.mean() + score_neg.mean()) / 2).detach()

                loss_pos = torch.log1p(torch.exp(-(score_pos - theta))).mean()
                loss_neg = torch.log1p(torch.exp(score_neg - theta)).mean()
                loss_ell = loss_pos + loss_neg

                opt = model.optimizers[ell]
                opt.zero_grad()
                loss_ell.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(block.parameters()) + list(head.parameters()), 1.0
                )
                opt.step()
                batch_loss_sum += float(loss_ell.detach())

                # Propagate detached activations + updated gamma to next block.
                h_pos = h_pos_out.detach()
                h_neg = h_neg_out.detach()
                gamma_pos = gamma_pos + g_pos.detach()
                gamma_neg = gamma_neg + g_neg.detach()

            running += batch_loss_sum / max(model.n_blocks, 1)
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        loss_curve.append(avg_loss)

        acc = _lc_eval(model, test_loader, device)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [LayerCollab/{arch}] epoch {epoch + 1}/{epochs}  "
                  f"loss={avg_loss:.4f}  test_acc={acc:.4f}  best={best_acc:.4f}",
                  flush=True)

    return TrainResult(
        algo="layer_collab", status="ok", arch=arch,
        test_acc_final=acc_curve[-1],
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=count_params(model),
        elapsed_s=time.time() - t0,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "Adam(per-block)",
            "arch": arch, "n_blocks": model.n_blocks,
            "block_channels": [s["out_ch"] for s in model.specs],
            "hidden_dim": model.hidden_dim,
            "head": "Linear+LN+ReLU + Linear+LN+ReLU (per block)",
            "goodness": "mean-squared of head output (||z_ell||^2 / hidden_dim)",
            "gamma_rule": "gamma_{<t} = sum_{t'<t} G_{t'} (Lorberbom Eq. 7)",
            "loss": "-log sigma(G+gamma-theta) - log sigma(theta-G-gamma)",
            "theta": "dynamic-midpoint",
            "overlay": "label-overlay (first 10 pixels)",
            "negative": "random-wrong-label",
            "eval": "10-way argmax over sum_ell (G_ell + gamma_{<ell})",
            "seed": cfg.get("seed", 0),
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": False,
            "bn_status": "native_ln_kept (per-block FFA head has Linear+LN+ReLU)",
        },
        gpu=str(device), timestamp=now_iso(),
    )
