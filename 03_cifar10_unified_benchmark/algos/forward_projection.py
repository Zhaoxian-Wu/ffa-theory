"""
Forward Projection (FP-fixed): frozen random conv features + per-layer ridge heads.

Reference: O'Shea & Rajendran, "Forward Projection" (Nat. Comm. 2026 / arXiv 2023).

Steps:
  1. Instantiate the shared backbone (CNN3/CNN6 via cfg["arch"]) with random
     weights; freeze all parameters.
  2. Accumulate per-layer (H^T H, H^T Y) over the training set.
  3. Solve W_l = (H^T H + lambda_eff * I)^{-1} H^T Y in closed form per layer.
  4. Test: ensemble per-layer logits W_l^T h_l by summation, argmax.

Conv feature maps are adaptive-avg-pooled to (C, 4, 4) before ridge solve to
bound Gram matrix size. Closed-form means epochs do not affect accuracy; the
acc curve is filled with the single solution for interface uniformity.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

from common import (
    FAIR_DEFAULT_ARCH, FAIR_NUM_CLASSES, TrainResult, make_backbone,
    count_params, get_cifar10_loaders, now_iso, register, set_seed,
)


_POOL_SIZE = 4  # conv feature maps pooled to (C, 4, 4) before flatten


def _reduce_features(feats: List[torch.Tensor]) -> List[torch.Tensor]:
    """Project each intermediate tensor to a 2D (B, D) flattened form.

    For 4D conv outputs, adaptive-avg-pool to (_POOL_SIZE, _POOL_SIZE).
    For 2D outputs (FC), leave unchanged.
    """
    reduced = []
    for h in feats:
        if h.dim() == 4:
            h = F.adaptive_avg_pool2d(h, _POOL_SIZE)
            h = h.reshape(h.size(0), -1)
        elif h.dim() == 2:
            pass
        else:
            h = h.reshape(h.size(0), -1)
        reduced.append(h)
    return reduced


# ================================================================
# Per-layer ridge accumulators
# ================================================================

class _RidgeAccumulator:
    """Streaming accumulator for (H^T H, H^T Y) without materializing full H."""

    def __init__(self, feat_dim: int, n_classes: int, device: torch.device,
                 dtype: torch.dtype = torch.float64):
        self.feat_dim = feat_dim
        self.n_classes = n_classes
        self.device = device
        self.dtype = dtype
        self.AtA = torch.zeros(feat_dim + 1, feat_dim + 1, device=device, dtype=dtype)
        self.AtY = torch.zeros(feat_dim + 1, n_classes, device=device, dtype=dtype)
        self.n = 0

    def update(self, h: torch.Tensor, y_onehot: torch.Tensor) -> None:
        # Append bias column of ones for intercept.
        h64 = h.to(self.dtype)
        ones = torch.ones(h64.size(0), 1, device=self.device, dtype=self.dtype)
        h_aug = torch.cat([h64, ones], dim=1)
        y64 = y_onehot.to(self.dtype)
        self.AtA += h_aug.T @ h_aug
        self.AtY += h_aug.T @ y64
        self.n += h64.size(0)

    def solve(self, ridge_lambda: float) -> torch.Tensor:
        """Return W of shape (feat_dim + 1, n_classes)."""
        # Scale lambda relative to trace (standard FP convention: lam = lam * tr(A)/d).
        tr = self.AtA[: self.feat_dim, : self.feat_dim].diagonal().sum()
        lam = ridge_lambda * tr / max(self.feat_dim, 1)
        reg = lam * torch.eye(self.feat_dim + 1, device=self.device, dtype=self.dtype)
        reg[-1, -1] = 0.0  # do not regularize bias
        W = torch.linalg.solve(self.AtA + reg, self.AtY)
        return W


def _apply_head(h: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Apply a solved ridge head W (feat_dim+1, n_classes) to features h (B, feat_dim)."""
    ones = torch.ones(h.size(0), 1, device=h.device, dtype=W.dtype)
    h_aug = torch.cat([h.to(W.dtype), ones], dim=1)
    return h_aug @ W


# ================================================================
# Registered entry point
# ================================================================

@register("forward_projection")
def train(cfg: dict) -> TrainResult:
    """Forward Projection (FP-fixed): random frozen conv features + per-layer ridge heads."""
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)  # cosmetic only -- closed-form solution
    ridge_lambda = cfg.get("ridge_lambda", 1e-2)
    fp_mode = cfg.get("fp_mode", "fixed")
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    assert fp_mode == "fixed", f"Only FP-fixed supported, got fp_mode={fp_mode!r}"

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.get("batch_size", 128), augment=False,
    )

    # Random-frozen backbone.
    backbone = make_backbone(arch).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    t0 = time.time()

    # ---- Pass 1: build ridge accumulators for every intermediate layer ----
    accumulators: List[_RidgeAccumulator] = []
    feat_dims: List[int] = []

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feats = _reduce_features(backbone.forward_with_intermediates(x))
            y_oh = F.one_hot(y, FAIR_NUM_CLASSES).float()
            if not accumulators:
                for h in feats:
                    feat_dims.append(h.size(1))
                    accumulators.append(_RidgeAccumulator(h.size(1), FAIR_NUM_CLASSES, device))
            for acc, h in zip(accumulators, feats):
                acc.update(h, y_oh)

    # ---- Solve ridge per layer ----
    heads: List[torch.Tensor] = [acc.solve(ridge_lambda) for acc in accumulators]

    # ---- Evaluate: ensemble logits across layers ----
    @torch.no_grad()
    def _eval(loader) -> Tuple[float, float]:
        correct_ens = correct_last = total = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            feats = _reduce_features(backbone.forward_with_intermediates(x))
            logits_sum = None
            for h, W in zip(feats, heads):
                logit = _apply_head(h, W).float()
                logits_sum = logit if logits_sum is None else logits_sum + logit
            pred_ens = logits_sum.argmax(dim=1)
            pred_last = _apply_head(feats[-1], heads[-1]).float().argmax(dim=1)
            correct_ens += (pred_ens == y).sum().item()
            correct_last += (pred_last == y).sum().item()
            total += y.size(0)
        return correct_ens / total, correct_last / total

    acc_ens, acc_last = _eval(test_loader)
    elapsed = time.time() - t0

    # Interface uniformity: repeat the single closed-form accuracy `epochs` times.
    test_acc_curve = [acc_ens] * max(epochs, 1)
    train_loss_curve = [0.0] * max(epochs, 1)

    return TrainResult(
        algo="forward_projection",
        status="ok",
        arch=arch,
        test_acc_final=acc_ens,
        test_acc_best=acc_ens,
        test_acc_curve=test_acc_curve,
        train_loss_curve=train_loss_curve,
        n_params=count_params(backbone),  # 0 trainable under FP-fixed
        elapsed_s=elapsed,
        hyperparams={
            "fp_mode": fp_mode,
            "arch": arch,
            "ridge_lambda": ridge_lambda,
            "feat_dims": feat_dims,
            "pool_size": _POOL_SIZE,
            "closed_form": True,
            "epochs_cosmetic": epochs,
            "seed": cfg.get("seed", 0),
            "batch_size": cfg.get("batch_size", 128),
            "note": "no iterative training -- closed-form ridge regression",
            "ensemble_acc": acc_ens,
            "last_layer_only_acc": acc_last,
            "lr_decay_epoch": None,
            "lr_decay_factor": None,
            "use_bn": False,
            "bn_status": "exempt_closed_form (no gradient training; random frozen features + closed-form ridge)",
        },
        gpu=str(device),
        timestamp=now_iso(),
        tag="closed_form",
    )
