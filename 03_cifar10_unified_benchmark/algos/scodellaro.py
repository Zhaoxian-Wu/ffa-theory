"""
Scodellaro CNN-FFA (Sci. Reports 2025), truncated to 3 layers.

Key ideas preserved from the paper:
  - Structured label embedding via Fourier frequency/orientation patterns,
    blended with image: X_c = (1 - K) * X + K * T_c
  - BN before each conv (not LayerNorm)
  - Fixed threshold theta_l = C_l * H * W (number of neurons in the layer)
  - FFA loss accumulated over layers, excluding layer 0
  - Linear probe evaluation on pooled features from the supervised layers

Deviations for fair benchmark:
  - FILTERS = [32, 64, 128] (3 layers, not 6)
  - lr = 1e-3, batch_size = 128, 200 epochs (fair contract)
  - Fourier labels only (morph is costly and the fourier variant is already
    the paper's baseline 60.9%)
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, TrainResult,
    apply_lr_decay, block_channels, count_params, get_cifar10_loaders,
    now_iso, register, set_seed,
)


K_BLEND = 0.30


def _build_fourier_patterns(n_classes: int = 10, size: int = 32) -> torch.Tensor:
    freqs = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    angles = [0, 45, 90, 135, 0, 45, 90, 0, 45, 90]
    xs = np.linspace(0, 1, size, dtype=np.float32)
    ys = np.linspace(0, 1, size, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    patterns = []
    for f, a in zip(freqs, angles):
        rad = np.deg2rad(a)
        raw = np.sin(2 * np.pi * f * (XX * np.cos(rad) + YY * np.sin(rad)))
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        patterns.append(raw.astype(np.float32))
    return torch.from_numpy(np.stack(patterns))


class ScodellaroCNN(nn.Module):
    """N-layer Scodellaro CNN (arch-parameterised).

    Follows Scodellaro's design choices: 5x5 convs, BN-before-conv, no pooling
    (spatial 32x32 preserved throughout). The number of conv layers and their
    channel widths come from common.block_channels(arch) -- cnn3 gives 3 layers
    [32,64,128]; cnn6 gives 6 layers [32,32,64,64,128,128]. theta per layer is
    N_l = C_l * 32 * 32 (paper convention; spatial is always 32x32 here).
    """

    def __init__(self, arch: str = FAIR_DEFAULT_ARCH):
        super().__init__()
        self.arch = arch
        self.filters = block_channels(arch)
        in_ch = 3
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for out_ch in self.filters:
            self.bns.append(nn.BatchNorm2d(in_ch))
            self.convs.append(nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2))
            in_ch = out_ch
        self.thetas = [float(f * 32 * 32) for f in self.filters]
        # Linear-probe input: concatenate GAP features of layers 1..L-1 (skip layer 0).
        total_feat = sum(self.filters[1:])
        self.classifier = nn.Linear(total_feat, FAIR_NUM_CLASSES)

    def forward_layers(self, x):
        acts = []
        h = x
        for bn, conv in zip(self.bns, self.convs):
            h = F.relu(conv(bn(h)))
            acts.append(h)
        return acts

    @staticmethod
    def goodness(acts):
        return [(a ** 2).sum(dim=(1, 2, 3)) for a in acts]

    def forward_repr(self, x):
        acts = self.forward_layers(x)
        feats = [a.mean(dim=(2, 3)) for a in acts[1:]]
        return torch.cat(feats, dim=1)


def _apply_blending(x, labels, patterns):
    T = patterns[labels].unsqueeze(1).expand(-1, 3, -1, -1)
    return (1 - K_BLEND) * x + K_BLEND * T


def _make_fourier_pair(x, y, patterns, n_classes: int = FAIR_NUM_CLASSES):
    x_pos = _apply_blending(x, y, patterns)
    y_neg = (y + torch.randint(1, n_classes, y.shape, device=y.device)) % n_classes
    x_neg = _apply_blending(x, y_neg, patterns)
    return x_pos, x_neg


@torch.no_grad()
def _linear_probe_eval(model, train_loader, test_loader, patterns, device):
    model.eval()
    def _extract(loader):
        zs, ys = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            xp, _ = _make_fourier_pair(x, y, patterns)
            zs.append(model.forward_repr(xp).cpu())
            ys.append(y.cpu())
        return torch.cat(zs).to(device), torch.cat(ys).to(device)

    tr_z, tr_y = _extract(train_loader)
    te_z, te_y = _extract(test_loader)

    clf = nn.Linear(tr_z.shape[1], FAIR_NUM_CLASSES).to(device)
    opt = optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    with torch.enable_grad():
        for _ in range(300):
            logits = clf(tr_z)
            loss = F.cross_entropy(logits, tr_y)
            opt.zero_grad(); loss.backward(); opt.step()
    pred = clf(te_z).argmax(1)
    return (pred == te_y).float().mean().item()


def _legacy_global_train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 recipe: LR decay only (ScodellaroCNN already has BN before each conv).
    # Note: linear-probe classifier has its own Adam with fixed lr (post-hoc eval,
    # not part of the feature-learning trajectory); we do not decay it.
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.get("batch_size", 128), augment=False,
    )
    patterns = _build_fourier_patterns().to(device)
    model = ScodellaroCNN(arch=arch).to(device)

    # Exclude classifier from FFA optimizer (linear probe is trained separately)
    ffa_params = [p for p in model.parameters()
                  if p is not model.classifier.weight and p is not model.classifier.bias]
    opt = optim.Adam(ffa_params, lr=lr)

    loss_curve, acc_curve = [], []
    t0 = time.time()
    best_acc = 0.0
    eval_every = max(1, epochs // 20)
    for epoch in range(epochs):
        apply_lr_decay(opt, epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor,
                       verbose=(epoch == lr_decay_epoch))
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            xp, xn = _make_fourier_pair(x, y, patterns)
            acts_p = model.forward_layers(xp)
            acts_n = model.forward_layers(xn)
            g_p = model.goodness(acts_p)
            g_n = model.goodness(acts_n)
            loss = torch.tensor(0.0, device=device)
            # Paper recipe: skip layer 0, accumulate FFA goodness loss on layers 1..L-1.
            for ell in range(1, len(model.filters)):
                theta = model.thetas[ell]
                loss = loss - F.logsigmoid(g_p[ell] - theta).mean() \
                            - F.logsigmoid(theta - g_n[ell]).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            n_batches += 1
        loss_curve.append(running / max(n_batches, 1))

        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            acc = _linear_probe_eval(model, train_loader, test_loader, patterns, device)
            acc_curve.append(acc)
            best_acc = max(best_acc, acc)
            print(f"  [Scodellaro] epoch {epoch + 1}/{epochs}  loss={loss_curve[-1]:.4f}  "
                  f"probe_acc={acc:.4f}  best={best_acc:.4f}", flush=True)

    return TrainResult(
        algo="scodellaro", status="ok", arch=arch,
        test_acc_final=acc_curve[-1] if acc_curve else None,
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=count_params(model),
        elapsed_s=time.time() - t0,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": cfg.get("batch_size", 128),
                     "optimizer": "Adam", "filters": model.filters, "kernel": 5,
                     "label": "fourier", "K_blend": K_BLEND, "theta": "N_l fixed",
                     "arch": arch, "n_layers": len(model.filters),
                     "eval": "linear-probe", "seed": cfg.get("seed", 0),
                     "lr_decay_epoch": lr_decay_epoch,
                     "lr_decay_factor": lr_decay_factor,
                     "use_bn": True,
                     "bn_status": "native_bn_kept (paper: BN before each conv)"},
        gpu=str(device), timestamp=now_iso(),
    )


@register("scodellaro")
def train(cfg: dict) -> TrainResult:
    """Strict block-local Fourier-label CNN-FFA training.

    Each BN--convolution pair owns its own optimizer and receives only its
    own goodness loss.  The previous implementation summed losses across all
    layers and differentiated that total through the full convolution stack.
    """
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = int(cfg.get("epochs", 200))
    lr = float(cfg.get("lr", FAIR_LR))
    batch_size = int(cfg.get("batch_size", 128))
    arch = str(cfg.get("arch", FAIR_DEFAULT_ARCH))
    lr_decay_epoch = int(cfg.get("lr_decay_epoch", 100))
    lr_decay_factor = float(cfg.get("lr_decay_factor", 0.1))
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, augment=False)
    patterns = _build_fourier_patterns().to(device)
    model = ScodellaroCNN(arch=arch).to(device)
    optimizers = [optim.Adam(list(bn.parameters()) + list(conv.parameters()), lr=lr)
                  for bn, conv in zip(model.bns, model.convs)]
    loss_curve, acc_curve, best_acc = [], [], 0.0
    start = time.time()
    eval_every = max(1, epochs // 20)
    for epoch in range(epochs):
        apply_lr_decay(optimizers, epoch, lr, at_epoch=lr_decay_epoch,
                       factor=lr_decay_factor, verbose=(epoch == lr_decay_epoch))
        model.train()
        running, batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            h_pos, h_neg = _make_fourier_pair(x, y, patterns)
            layer_losses = []
            for ell, (bn, conv, optimizer) in enumerate(zip(model.bns, model.convs, optimizers)):
                h_pos_out = F.relu(conv(bn(h_pos.detach())))
                h_neg_out = F.relu(conv(bn(h_neg.detach())))
                g_pos = h_pos_out.square().sum(dim=(1, 2, 3))
                g_neg = h_neg_out.square().sum(dim=(1, 2, 3))
                theta = model.thetas[ell]
                loss = -F.logsigmoid(g_pos - theta).mean() - F.logsigmoid(theta - g_neg).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(bn.parameters()) + list(conv.parameters()), 1.0)
                optimizer.step()
                layer_losses.append(float(loss.detach()))
                h_pos, h_neg = h_pos_out.detach(), h_neg_out.detach()
            running += sum(layer_losses) / max(len(layer_losses), 1)
            batches += 1
        average_loss = running / max(batches, 1)
        loss_curve.append(average_loss)
        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            accuracy = _linear_probe_eval(model, train_loader, test_loader, patterns, device)
            acc_curve.append(accuracy)
            best_acc = max(best_acc, accuracy)
            print(f"  [StrictLocalScodellaro/{arch}] epoch {epoch + 1}/{epochs} "
                  f"loss={average_loss:.4f} probe_acc={accuracy:.4f} best={best_acc:.4f}", flush=True)
    return TrainResult(
        algo="scodellaro", status="ok", arch=arch,
        test_acc_final=acc_curve[-1] if acc_curve else None, test_acc_best=best_acc,
        test_acc_curve=acc_curve, train_loss_curve=loss_curve, n_params=count_params(model),
        elapsed_s=time.time() - start,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": batch_size,
                     "optimizer": "Adam(per-BN-conv block)", "filters": model.filters,
                     "kernel": 5, "label": "fourier", "K_blend": K_BLEND,
                     "theta": "N_l fixed", "arch": arch, "n_layers": len(model.filters),
                     "eval": "linear-probe", "locality": "strict; detached inter-block activations; independent BN-conv optimizers",
                     "lr_decay_epoch": lr_decay_epoch, "lr_decay_factor": lr_decay_factor,
                     "use_bn": True, "bn_status": "native_bn_kept (per-block)",
                     "implementation": "strict_local_repair", "seed": cfg.get("seed", 0)},
        gpu=str(device), timestamp=now_iso())
