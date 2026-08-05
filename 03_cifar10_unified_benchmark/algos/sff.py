"""
SFF / Scalable Forward-Forward (Krutsylo et al. 2025, arXiv:2501.03176).

SFF is a block-wise local classification variant of the Forward-Forward Algorithm:
  - Each conv block is followed by an auxiliary classifier head trained with
    independent cross-entropy (no label-overlay, no positive/negative pairs).
  - Activations are detached between blocks, so each block's gradient only
    flows through its own parameters + its aux head.
  - Final prediction = argmax over the ensemble (mean) of the per-block
    aux head logits.

Aux head (per task spec):
    conv(C, C, k=5, pad=2) -> LayerNorm(C) -> ReLU -> global avg pool -> Linear(C, 10)
where LayerNorm is channel-only (applied via permute trick on the (B, C, H, W) tensor).

Arch-parameterised: the number of blocks and per-block specs are taken from
`make_conv_blocks(arch)` in common.  Each block uses the exact same Conv+ReLU(+pool)
layout as CNN3Backbone / CNN6Backbone (via common.ConvBlock).  An aux head is
instantiated per block with `channels = spec["out_ch"]`.

Fair-comparison contract:
    Adam(lr=1e-3), batch_size=128, epochs=200, no augmentation, seed=0.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, ConvBlock, TrainResult,
    apply_lr_decay, block_channels, count_params, get_cifar10_loaders,
    make_conv_blocks, now_iso, register, set_seed,
)


AUX_KERNEL = 5


class _AuxHead(nn.Module):
    """SFF auxiliary classifier head: conv(k=5) -> LN(C, channel-only) -> relu -> GAP -> linear.

    LayerNorm is applied over the channel dimension only, following the paper's
    channel-only normalisation convention. We implement this by permuting to
    (B, H, W, C), running nn.LayerNorm(C), and permuting back.
    """

    def __init__(self, channels: int, num_classes: int = FAIR_NUM_CLASSES):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=AUX_KERNEL, padding=AUX_KERNEL // 2)
        self.ln = nn.LayerNorm(channels)
        self.linear = nn.Linear(channels, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, C, H, W)
        z = self.conv(h)
        # Channel-only LayerNorm via permute.
        z = z.permute(0, 2, 3, 1).contiguous()   # (B, H, W, C)
        z = self.ln(z)
        z = z.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W)
        z = torch.relu(z)
        z = F.adaptive_avg_pool2d(z, 1).reshape(z.shape[0], -1)  # (B, C)
        return self.linear(z)


class SFFModel(nn.Module):
    """Arch-parameterised SFF CNN with independent per-block aux heads.

    Blocks are built from `make_conv_blocks(arch)` via `common.ConvBlock`,
    so their Conv+ReLU(+pool) layout is 1:1 with the corresponding backbone
    (CNN3Backbone / CNN6Backbone). One `_AuxHead` per block.
    """

    def __init__(self, arch: str = FAIR_DEFAULT_ARCH,
                 num_classes: int = FAIR_NUM_CLASSES):
        super().__init__()
        block_specs = make_conv_blocks(arch)
        self.blocks = nn.ModuleList([ConvBlock(spec) for spec in block_specs])
        self.aux_heads = nn.ModuleList([
            _AuxHead(spec["out_ch"], num_classes) for spec in block_specs
        ])
        self.arch = arch
        self.n_blocks = len(block_specs)

    def forward_train(self, x: torch.Tensor):
        """Run forward with inter-block detach. Returns list of per-block aux logits.

        Each logits tensor is attached only to its own block + aux head,
        so backprop on each element updates only that block's parameters.
        """
        h = x
        all_logits = []
        for block, aux in zip(self.blocks, self.aux_heads):
            h_in = h.detach()                 # block-local gradient boundary
            h = block(h_in)
            all_logits.append(aux(h))
        return all_logits

    @torch.no_grad()
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """Return ensemble logits: mean of aux-head logits across all blocks."""
        h = x
        ensemble = None
        for block, aux in zip(self.blocks, self.aux_heads):
            h = block(h)
            logits = aux(h)
            ensemble = logits if ensemble is None else ensemble + logits
        return ensemble / self.n_blocks


def _make_block_optimizers(model: SFFModel, lr: float):
    """One independent Adam optimizer per (block + aux head) pair."""
    opts = []
    for block, aux in zip(model.blocks, model.aux_heads):
        params = list(block.parameters()) + list(aux.parameters())
        opts.append(optim.Adam(params, lr=lr))
    return opts


@torch.no_grad()
def _evaluate(model: SFFModel, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model.forward_eval(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    return correct / max(total, 1)


@register("sff")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    batch_size = cfg.get("batch_size", 128)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 recipe: LR decay only (SFF aux-heads carry channel-only
    # LayerNorm per paper; no extra BN added).
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, augment=False)
    model = SFFModel(arch=arch, num_classes=FAIR_NUM_CLASSES).to(device)
    opts = _make_block_optimizers(model, lr)

    loss_curve, acc_curve = [], []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        apply_lr_decay(opts, epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor,
                       verbose=(epoch == lr_decay_epoch))
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits_per_block = model.forward_train(x)
            block_losses = []
            for logits, opt in zip(logits_per_block, opts):
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                block_losses.append(loss.item())
            running += sum(block_losses) / len(block_losses)
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        loss_curve.append(avg_loss)

        acc = _evaluate(model, test_loader, device)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [SFF/{arch}] epoch {epoch + 1}/{epochs}  mean_block_loss={avg_loss:.4f}  "
                  f"test_acc={acc:.4f}  best={best_acc:.4f}", flush=True)

    return TrainResult(
        algo="sff", status="ok", arch=arch,
        test_acc_final=acc_curve[-1],
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=count_params(model),
        elapsed_s=time.time() - t0,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "Adam(per-block)", "aux_kernel": AUX_KERNEL,
            "arch": arch,
            "n_blocks": model.n_blocks,
            "block_channels": block_channels(arch),
            "layernorm": "channel-only",
            "eval": "mean-ensemble-argmax",
            "seed": cfg.get("seed", 0),
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": False,
            "bn_status": "native_ln_kept (paper-native channel-only LN)",
        },
        gpu=str(device), timestamp=now_iso(),
    )
