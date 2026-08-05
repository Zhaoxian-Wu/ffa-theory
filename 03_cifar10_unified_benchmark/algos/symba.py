"""
SymBa (Lee & Song, NeurIPS 2023): Symmetric contrastive variant of FFA.

Key difference from vanilla FFA:
  - Vanilla FFA: -log sigma(G_pos - theta) - log sigma(theta - G_neg)
                 (asymmetric, threshold theta needed)
  - SymBa:       softplus(-alpha * (G_pos - G_neg))
                 (symmetric, depends only on pos-neg goodness gap,
                  temperature alpha absorbs the role of theta)

Architecture and data protocol match vanilla_ffa.py exactly for fair comparison:
  - Shared backbone (CNN3/CNN6, via cfg["arch"]) + FFAHead (same as vanilla FFA)
  - Label-overlay on first 10 pixels of flattened image
  - Goodness g(h) = mean(h^2)
  - Negative sample: random wrong label
  - Evaluation: 10-way argmax over goodness with each candidate label overlaid

Reference: Lee, H. & Song, J. (2023). "SymBa: Symmetric Backpropagation-free
Contrastive Learning with Forward-Forward Algorithm." NeurIPS 2023.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    FAIR_DEFAULT_ARCH, FAIR_FEAT_DIM, FAIR_LR, FAIR_NUM_CLASSES, TrainResult,
    apply_lr_decay, count_params, get_cifar10_loaders, make_backbone, now_iso,
    register, set_seed,
)


# SymBa default temperature (paper uses alpha=4.0)
DEFAULT_ALPHA = 4.0


def overlay_label(x_flat: torch.Tensor, y: torch.Tensor, num_classes: int = FAIR_NUM_CLASSES) -> torch.Tensor:
    """Overlay one-hot label in first num_classes pixels of flattened image."""
    out = x_flat.clone()
    one_hot = torch.zeros(x_flat.shape[0], num_classes, device=x_flat.device, dtype=x_flat.dtype)
    one_hot.scatter_(1, y.unsqueeze(1), 1.0)
    out[:, :num_classes] = one_hot
    return out


def _random_wrong_labels(y: torch.Tensor, num_classes: int = FAIR_NUM_CLASSES) -> torch.Tensor:
    delta = torch.randint(1, num_classes, y.shape, device=y.device)
    return (y + delta) % num_classes


class FFAHead(nn.Module):
    """2-layer MLP FFA head over backbone features. Goodness = mean(h^2)."""

    def __init__(self, feat_dim: int = FAIR_FEAT_DIM, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.ln1(self.fc1(x)))
        h = torch.relu(self.ln2(self.fc2(h)))
        return h

    @staticmethod
    def goodness(h: torch.Tensor) -> torch.Tensor:
        return (h ** 2).mean(dim=1)


def _apply_overlay_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(x.shape[0], -1)
    x_flat = overlay_label(x_flat, y)
    return x_flat.reshape(x.shape)


@torch.no_grad()
def _ffa_eval(backbone: nn.Module, head: FFAHead, loader, device) -> float:
    """10-way label-overlay argmax evaluation (same protocol as vanilla FFA)."""
    backbone.eval(); head.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bsz = x.shape[0]
        best_g = torch.full((bsz,), -float("inf"), device=device)
        best_c = torch.zeros(bsz, dtype=torch.long, device=device)
        for c in range(FAIR_NUM_CLASSES):
            y_c = torch.full((bsz,), c, dtype=torch.long, device=device)
            x_c = _apply_overlay_image(x, y_c)
            g = head.goodness(head(backbone(x_c)))
            mask = g > best_g
            best_g[mask] = g[mask]
            best_c[mask] = c
        correct += (best_c == y).sum().item()
        total += bsz
    return correct / max(total, 1)


def _symba_loss(g_pos: torch.Tensor, g_neg: torch.Tensor, alpha: float) -> torch.Tensor:
    """SymBa symmetric contrastive loss.

    L = softplus(-alpha * (g_pos - g_neg))
      = log(1 + exp(-alpha * (g_pos - g_neg)))

    This is the core difference from vanilla FFA. Unlike vanilla FFA which uses
    two independent sigmoid losses against a threshold theta, SymBa uses one
    symmetric loss on the goodness gap only. Implemented via F.softplus for
    numerical stability (no manual log1p+exp).
    """
    return F.softplus(-alpha * (g_pos - g_neg)).mean()


def _legacy_global_train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    alpha = cfg.get("alpha", DEFAULT_ALPHA)
    batch_size = cfg.get("batch_size", 128)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 recipe: LR decay only (FFAHead carries LayerNorm; no extra BN).
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, augment=False,
    )
    backbone = make_backbone(arch).to(device)
    head = FFAHead(feat_dim=backbone.out_dim, hidden_dim=512).to(device)
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), lr=lr
    )

    acc_curve, loss_curve = [], []
    t0 = time.time()
    best_acc = 0.0
    for epoch in range(epochs):
        apply_lr_decay(optimizer, epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor,
                       verbose=(epoch == lr_decay_epoch))
        backbone.train(); head.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_pos = _apply_overlay_image(x, y)
            y_neg = _random_wrong_labels(y)
            x_neg = _apply_overlay_image(x, y_neg)

            g_pos = head.goodness(head(backbone(x_pos)))
            g_neg = head.goodness(head(backbone(x_neg)))
            loss = _symba_loss(g_pos, g_neg, alpha)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()), 1.0
            )
            optimizer.step()
            running += loss.item()
            n_batches += 1
        avg_loss = running / max(n_batches, 1)
        loss_curve.append(avg_loss)

        acc = _ffa_eval(backbone, head, test_loader, device)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [SymBa] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  "
                  f"test_acc={acc:.4f}  best={best_acc:.4f}", flush=True)

    return TrainResult(
        algo="symba", status="ok", arch=arch,
        test_acc_final=acc_curve[-1],
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=count_params(backbone) + count_params(head),
        elapsed_s=time.time() - t0,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": batch_size,
                     "optimizer": "Adam", "goodness": "mean_sq",
                     "overlay": "label-overlay", "negative": "random-wrong",
                     "alpha": alpha, "loss": "softplus(-alpha*(g_pos-g_neg))",
                     "arch": arch, "seed": cfg.get("seed", 0),
                     "lr_decay_epoch": lr_decay_epoch,
                     "lr_decay_factor": lr_decay_factor,
                     "use_bn": False,
                     "bn_status": "native_ln_kept (FFAHead has 2 LayerNorm)"},
        gpu=str(device), timestamp=now_iso(),
    )


@register("symba")
def train(cfg: dict) -> TrainResult:
    """Strict block-local SymBa with one loss and optimizer per CNN block."""
    from algos.strict_local_ffa import StrictLocalFFAModel, evaluate
    from algos.vanilla_ffa import _apply_overlay_image, _random_wrong_labels

    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = int(cfg.get("epochs", 200))
    lr = float(cfg.get("lr", FAIR_LR))
    alpha = float(cfg.get("alpha", DEFAULT_ALPHA))
    batch_size = int(cfg.get("batch_size", 128))
    arch = str(cfg.get("arch", FAIR_DEFAULT_ARCH))
    lr_decay_epoch = int(cfg.get("lr_decay_epoch", 100))
    lr_decay_factor = float(cfg.get("lr_decay_factor", 0.1))
    trunk_norm = str(cfg.get("trunk_norm", "none"))

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, augment=False)
    model = StrictLocalFFAModel(arch=arch, lr=lr, trunk_norm=trunk_norm).to(device)
    best_acc, acc_curve, loss_curve = 0.0, [], []
    start = time.time()
    for epoch in range(epochs):
        apply_lr_decay(model.optimizers, epoch, lr, at_epoch=lr_decay_epoch,
                       factor=lr_decay_factor, verbose=(epoch == lr_decay_epoch))
        model.train()
        running, batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            h_pos = _apply_overlay_image(x, y)
            h_neg = _apply_overlay_image(x, _random_wrong_labels(y))
            layer_losses = []
            for block, head, optimizer in zip(model.blocks, model.heads, model.optimizers):
                h_pos_out = block(h_pos.detach())
                h_neg_out = block(h_neg.detach())
                loss = _symba_loss(head.goodness(head(h_pos_out)),
                                   head.goodness(head(h_neg_out)), alpha)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(block.parameters()) + list(head.parameters()), 1.0)
                optimizer.step()
                layer_losses.append(float(loss.detach()))
                h_pos, h_neg = h_pos_out.detach(), h_neg_out.detach()
            running += sum(layer_losses) / max(len(layer_losses), 1)
            batches += 1
        average_loss = running / max(batches, 1)
        accuracy = evaluate(model, test_loader, device)
        loss_curve.append(average_loss)
        acc_curve.append(accuracy)
        best_acc = max(best_acc, accuracy)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [StrictLocalSymBa/{arch}] epoch {epoch + 1}/{epochs} "
                  f"loss={average_loss:.4f} test_acc={accuracy:.4f} best={best_acc:.4f}", flush=True)
    return TrainResult(
        algo="symba", status="ok", arch=arch, test_acc_final=acc_curve[-1],
        test_acc_best=best_acc, test_acc_curve=acc_curve, train_loss_curve=loss_curve,
        n_params=count_params(model), elapsed_s=time.time() - start,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": batch_size,
                     "optimizer": "Adam(per-block)", "alpha": alpha,
                     "overlay": "label-overlay", "negative": "random-wrong-label",
                     "loss": "softplus(-alpha*(g_pos-g_neg))", "arch": arch,
                     "locality": "strict; detached inter-block activations; independent block/head optimizers",
                     "trunk_norm": trunk_norm, "use_bn": False,
                     "lr_decay_epoch": lr_decay_epoch, "lr_decay_factor": lr_decay_factor,
                     "implementation": "strict_local_repair", "seed": cfg.get("seed", 0)},
        gpu=str(device), timestamp=now_iso())
