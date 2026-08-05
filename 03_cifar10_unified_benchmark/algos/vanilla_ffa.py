"""
Vanilla FFA: Hinton 2022 Forward-Forward Algorithm with label-overlay protocol.

Positive:   overlay correct one-hot label on first 10 pixels (pre-flatten)
Negative:   overlay random wrong label
Goodness:   g(h) = ||h||^2 / dim
Loss:       -log sigma(g_pos - theta) - log sigma(theta - g_neg), dynamic theta
Evaluation: 10-way argmax over g(h | overlaid label c) for c=0..9

This replicates the training recipe in experiments/cnn_vit_ffa_vs_bp.py:171-251
but rewires the data path through the shared backbone (CNN3/CNN6 via cfg["arch"])
plus a 2-layer FFAHead.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.optim as optim

from common import (
    FAIR_DEFAULT_ARCH, FAIR_FEAT_DIM, FAIR_LR, FAIR_NUM_CLASSES, TrainResult,
    apply_lr_decay, count_params, get_cifar10_loaders, make_backbone, now_iso,
    register, set_seed,
)


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
    """2-layer MLP FFA head over backbone features. Goodness = ||h||^2 / dim."""

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


@register("vanilla_ffa")
def train(cfg: dict) -> TrainResult:
    """Train the locality-correct layerwise FFA implementation.

    The former benchmark implementation used one terminal FFA head and a
    single optimizer over the full backbone, which made its loss end-to-end.
    The strict implementation assigns one goodness head and optimizer to each
    convolutional block and detaches activations between consecutive blocks.
    It lives in ``strict_local_ffa`` so that the historical repair and this
    canonical ``vanilla_ffa`` entry share exactly the same tested code path.
    """
    from algos.strict_local_ffa import train as train_strict_local_ffa

    result = train_strict_local_ffa(cfg)
    result.algo = "vanilla_ffa"
    result.hyperparams["implementation"] = "strict_local_repair"
    result.hyperparams["legacy_terminal_head"] = False
    return result


def _legacy_global_train(cfg: dict) -> TrainResult:
    """Archived end-to-end implementation retained only for code provenance."""
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 recipe: LR decay only. BN not added because the FFAHead
    # already carries two LayerNorm layers -- stacking BN on top would be
    # redundant / destabilising.
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.get("batch_size", 128), augment=False,
    )
    backbone = make_backbone(arch).to(device)
    head = FFAHead(feat_dim=backbone.out_dim, hidden_dim=512).to(device)
    optimizer = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)

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
            theta = ((g_pos.mean() + g_neg.mean()) / 2).detach()

            loss_pos = torch.log1p(torch.exp(-(g_pos - theta))).mean()
            loss_neg = torch.log1p(torch.exp(g_neg - theta)).mean()
            loss = loss_pos + loss_neg

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
            print(f"  [FFA] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  test_acc={acc:.4f}  best={best_acc:.4f}",
                  flush=True)

    return TrainResult(
        algo="vanilla_ffa", status="ok", arch=arch,
        test_acc_final=acc_curve[-1],
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=count_params(backbone) + count_params(head),
        elapsed_s=time.time() - t0,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": cfg.get("batch_size", 128),
                     "optimizer": "Adam", "goodness": "mean_sq", "overlay": "label-overlay",
                     "negative": "random-wrong", "theta": "dynamic-midpoint",
                     "arch": arch, "seed": cfg.get("seed", 0),
                     "lr_decay_epoch": lr_decay_epoch,
                     "lr_decay_factor": lr_decay_factor,
                     "use_bn": False,
                     "bn_status": "native_ln_kept (FFAHead has 2 LayerNorm)"},
        gpu=str(device), timestamp=now_iso(),
    )
