"""
SCFF: Self-Contrastive Forward-Forward Algorithm (Chen et al., Nature Comm. 2025).

Reference: Chen et al., "Self-Contrastive Forward-Forward Algorithm", 2024.
Paper PDF: papers/chen2024_selfcontrastive_ffa.pdf

Core idea:
  Traditional FFA needs "negative samples" — images paired with wrong labels.
  SCFF removes this requirement by using the sample itself contrasted against
  a shuffled/mismatched version of itself. In this simplified variant we use
  in-batch image shuffling:

    Positive:   overlay(x_i, y_i)               correct label on the image
    Negative:   overlay(x_pi, y_i)              where pi is an in-batch
                                                permutation of images — the
                                                image no longer matches its
                                                overlaid label, forming an
                                                image-label inconsistency
                                                drawn from the same batch
                                                without requiring a separate
                                                wrong-label lookup.

Goodness:   g(h) = mean(h^2)     (same as vanilla FFA)
Loss:       -log sigma(g_pos - theta) - log sigma(theta - g_neg)
Evaluation: 10-way argmax over g(h | overlaid label c) for c=0..9 (matches
            vanilla FFA so the comparison is apples-to-apples).

Architecture: shared backbone (CNN3/CNN6 via cfg["arch"]) + FFAHead (shared
              2-layer MLP head).
Fair-comparison contract: Adam(lr=1e-3), batch_size=128, no augmentation.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.optim as optim

from common import (
    FAIR_DEFAULT_ARCH, FAIR_FEAT_DIM, FAIR_LR, FAIR_NUM_CLASSES, TrainResult, make_backbone,
    apply_lr_decay, count_params, get_cifar10_loaders, now_iso, register, set_seed,
)


def overlay_label(x_flat: torch.Tensor, y: torch.Tensor,
                  num_classes: int = FAIR_NUM_CLASSES) -> torch.Tensor:
    """Overlay one-hot label in first num_classes pixels of flattened image."""
    out = x_flat.clone()
    one_hot = torch.zeros(x_flat.shape[0], num_classes,
                          device=x_flat.device, dtype=x_flat.dtype)
    one_hot.scatter_(1, y.unsqueeze(1), 1.0)
    out[:, :num_classes] = one_hot
    return out


def _apply_overlay_image(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(x.shape[0], -1)
    x_flat = overlay_label(x_flat, y)
    return x_flat.reshape(x.shape)


def _shuffled_permutation(bsz: int, y: torch.Tensor) -> torch.Tensor:
    """Return an in-batch permutation where no index is fixed when possible.

    We draw a random permutation; if any index happens to land on itself we
    derangement-patch by rotating those positions by 1. This keeps the
    operation O(B) and guarantees image != its own label-pairing with high
    probability (perfect derangement not required for the contrastive signal).
    """
    device = y.device
    perm = torch.randperm(bsz, device=device)
    fixed = (perm == torch.arange(bsz, device=device))
    if fixed.any() and bsz > 1:
        idx = fixed.nonzero(as_tuple=False).flatten()
        # Rotate fixed points by one position within the fixed set.
        rotated = torch.roll(idx, shifts=1, dims=0)
        perm[idx] = perm[rotated]
    return perm


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


@torch.no_grad()
def _ffa_eval(backbone: nn.Module, head: FFAHead, loader, device) -> float:
    """10-way label-overlay argmax evaluation (identical protocol to vanilla FFA)."""
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


def _legacy_global_train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
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
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=lr)

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
            bsz = x.shape[0]

            # Positive: image paired with its correct label.
            x_pos = _apply_overlay_image(x, y)
            # Negative (self-contrastive): shuffled image paired with
            # the positions' original labels, producing an image-label
            # mismatch drawn from the same batch — no wrong-label lookup.
            perm = _shuffled_permutation(bsz, y)
            x_neg = _apply_overlay_image(x[perm], y)

            g_pos = head.goodness(head(backbone(x_pos)))
            g_neg = head.goodness(head(backbone(x_neg)))
            theta = ((g_pos.mean() + g_neg.mean()) / 2).detach()

            loss_pos = torch.log1p(torch.exp(-(g_pos - theta))).mean()
            loss_neg = torch.log1p(torch.exp(g_neg - theta)).mean()
            loss = loss_pos + loss_neg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            running += loss.item()
            n_batches += 1
        avg_loss = running / max(n_batches, 1)
        loss_curve.append(avg_loss)

        acc = _ffa_eval(backbone, head, test_loader, device)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [SCFF] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  "
                  f"test_acc={acc:.4f}  best={best_acc:.4f}", flush=True)

    return TrainResult(
        algo="scff", status="ok", arch=arch,
        test_acc_final=acc_curve[-1],
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=count_params(backbone) + count_params(head),
        elapsed_s=time.time() - t0,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": batch_size,
                     "optimizer": "Adam", "goodness": "mean_sq",
                     "overlay": "label-overlay",
                     "negative": "in-batch-shuffled-image",
                     "theta": "dynamic-midpoint",
                     "lr_decay_epoch": lr_decay_epoch,
                     "lr_decay_factor": lr_decay_factor,
                     "use_bn": False,
                     "bn_status": "native_ln_kept (FFAHead has 2 LayerNorm)",
                     "arch": arch, "seed": cfg.get("seed", 0)},
        gpu=str(device), timestamp=now_iso(),
    )


@register("scff")
def train(cfg: dict) -> TrainResult:
    """Strict block-local self-contrastive FFA."""
    from algos.strict_local_ffa import StrictLocalFFAModel, evaluate

    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = int(cfg.get("epochs", 200))
    lr = float(cfg.get("lr", FAIR_LR))
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
            h_neg = _apply_overlay_image(x[_shuffled_permutation(x.shape[0], y)], y)
            layer_losses = []
            for block, head, optimizer in zip(model.blocks, model.heads, model.optimizers):
                h_pos_out = block(h_pos.detach())
                h_neg_out = block(h_neg.detach())
                g_pos = head.goodness(head(h_pos_out))
                g_neg = head.goodness(head(h_neg_out))
                theta = ((g_pos.mean() + g_neg.mean()) / 2).detach()
                loss = torch.log1p(torch.exp(-(g_pos - theta))).mean() + \
                       torch.log1p(torch.exp(g_neg - theta)).mean()
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
            print(f"  [StrictLocalSCFF/{arch}] epoch {epoch + 1}/{epochs} "
                  f"loss={average_loss:.4f} test_acc={accuracy:.4f} best={best_acc:.4f}", flush=True)
    return TrainResult(
        algo="scff", status="ok", arch=arch, test_acc_final=acc_curve[-1],
        test_acc_best=best_acc, test_acc_curve=acc_curve, train_loss_curve=loss_curve,
        n_params=count_params(model), elapsed_s=time.time() - start,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": batch_size,
                     "optimizer": "Adam(per-block)", "overlay": "label-overlay",
                     "negative": "in-batch-shuffled-image", "theta": "dynamic-midpoint",
                     "arch": arch, "locality": "strict; detached inter-block activations; independent block/head optimizers",
                     "trunk_norm": trunk_norm, "use_bn": False,
                     "lr_decay_epoch": lr_decay_epoch, "lr_decay_factor": lr_decay_factor,
                     "implementation": "strict_local_repair", "seed": cfg.get("seed", 0)},
        gpu=str(device), timestamp=now_iso())
