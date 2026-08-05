"""
Nokland & Eidnes 2019 — L_pred variant.

Each of the 3 conv blocks is followed by a prediction head (LPredHead) trained
with independent cross-entropy. Gradients do NOT cross block boundaries:
activations are .detach()-ed between blocks, and each (block + head) pair has
its own Adam optimizer.

Evaluation: argmax of the LAST block's LPredHead logits (deepest feature head),
matching the Nokland paper's "last layer" prediction protocol.

Fair-comparison contract:
    Adam(lr=1e-3) per (block + head), batch_size=128, epochs=200,
    no data augmentation, seed=0.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from common import (
    FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, TrainResult,
    apply_lr_decay, count_params, get_cifar10_loaders, now_iso,
    register, set_seed,
)
from algos.nokland_base import LPredHead, NoklandCNN


def _build_heads(channels, num_classes: int):
    return torch.nn.ModuleList([LPredHead(c, num_classes) for c in channels])


def _make_block_optimizers(model: NoklandCNN, heads, lr: float):
    """One independent Adam optimizer per (block + its LPredHead)."""
    opts = []
    for block, head in zip(model.blocks, heads):
        params = list(block.parameters()) + list(head.parameters())
        opts.append(optim.Adam(params, lr=lr))
    return opts


@torch.no_grad()
def _evaluate(model: NoklandCNN, heads, loader, device) -> float:
    model.eval()
    heads.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        acts = model.forward_eval(x)
        logits = heads[-1](acts[-1])  # use deepest block's head
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    return correct / max(total, 1)


@register("nokland_lpred")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    batch_size = cfg.get("batch_size", 128)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 recipe: LR decay + BN inside ConvBlock (applied equally to all algos).
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)
    use_bn = cfg.get("use_bn", True)

    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, augment=False)
    model = NoklandCNN(arch=arch, use_bn=use_bn).to(device)
    heads = _build_heads(model.channels, FAIR_NUM_CLASSES).to(device)
    opts = _make_block_optimizers(model, heads, lr)

    loss_curve, acc_curve = [], []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        apply_lr_decay(opts, epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor,
                       verbose=(epoch == lr_decay_epoch))
        model.train()
        heads.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward each block with detached input; train (block, head) pair locally.
            h = x
            block_losses = []
            for block, head, opt in zip(model.blocks, heads, opts):
                h = block(h.detach())
                logits = head(h)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                block_losses.append(loss.item())
                # Further forward passes must start from the post-update activation;
                # re-detach happens at the next iteration via `h.detach()`.
            running += sum(block_losses) / len(block_losses)
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        loss_curve.append(avg_loss)

        acc = _evaluate(model, heads, test_loader, device)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [Nokland-Lpred] epoch {epoch + 1}/{epochs}  "
                  f"mean_block_loss={avg_loss:.4f}  test_acc={acc:.4f}  "
                  f"best={best_acc:.4f}", flush=True)

    n_params = count_params(model) + count_params(heads)
    return TrainResult(
        algo="nokland_lpred", status="ok", arch=arch,
        test_acc_final=acc_curve[-1],
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=n_params,
        elapsed_s=time.time() - t0,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "Adam(per-block)",
            "arch": arch,
            "block_channels": list(model.channels),
            "n_blocks": model.n_blocks,
            "head": "conv3x3(no-bias) -> relu -> GAP -> linear",
            "loss": "local CE at every block",
            "eval": "argmax of deepest block's LPredHead",
            "seed": cfg.get("seed", 0),
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": use_bn,
            "bn_status": "added_to_convblock",
        },
        gpu=str(device), timestamp=now_iso(),
    )
