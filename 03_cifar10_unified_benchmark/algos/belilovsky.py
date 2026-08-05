"""
Belilovsky Greedy Layer-wise Learning (ICML 2019).

Ref: Belilovsky, Eickenberg, Oyallon. "Greedy Layerwise Learning Can Scale
to ImageNet." ICML 2019.

Train one conv block at a time with an auxiliary classifier. Once a block
converges, freeze it, detach its features, and train the next block + aux.
The final test accuracy is read off the top block's auxiliary classifier.

Arch-parameterised: the number of blocks and per-block (in_ch, out_ch, pool)
specs are taken from `make_conv_blocks(arch)` in common. Auxiliary head per
block:
  Conv(C, C, k=3, p=1) -> ReLU -> AdaptiveAvgPool(1) -> Flatten -> Linear(C, 10)

Each block gets `epochs // n_blocks` epochs; the remainder (0 .. n_blocks-1)
is allocated to the final stage so the total test_acc_curve length equals
`epochs` exactly.

Contrast with SFF / Nokland L_pred: here blocks are trained *sequentially*
(block_k starts only after block_{k-1} has fully converged and is frozen).
"""
from __future__ import annotations

import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from common import (
    FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, ConvBlock, TrainResult,
    apply_lr_decay, count_params, evaluate_classifier, get_cifar10_loaders,
    make_conv_blocks, now_iso, register, set_seed,
)


# ================================================================
# Auxiliary classifier head (unchanged; spec stays identical across archs)
# ================================================================

class AuxHead(nn.Module):
    """Auxiliary classifier head on top of a conv block.

    Conv(C, C, k=3, p=1) + ReLU + AdaptiveAvgPool(1) + Flatten + Linear(C, num_classes)
    """

    def __init__(self, channels: int, num_classes: int = FAIR_NUM_CLASSES):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(feat))
        h = self.avgpool(h).flatten(1)
        return self.fc(h)


# ================================================================
# Helpers
# ================================================================

def _forward_through_frozen(frozen_blocks: List[nn.Module],
                            x: torch.Tensor) -> torch.Tensor:
    """Pass `x` through an already-frozen chain of blocks without tracking gradients."""
    with torch.no_grad():
        h = x
        for b in frozen_blocks:
            h = b(h)
    return h.detach()


@torch.no_grad()
def _eval_stage(frozen_blocks: List[nn.Module], current_block: nn.Module,
                aux: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Standard top-1 accuracy through (frozen_blocks | current_block) + aux."""
    for b in frozen_blocks:
        b.eval()
    current_block.eval()
    aux.eval()
    return evaluate_classifier(
        lambda x: aux(current_block(_forward_through_frozen(frozen_blocks, x))),
        loader, device,
    )


def _build_epoch_schedule(epochs: int, n_blocks: int) -> List[int]:
    """Split `epochs` across `n_blocks` stages. Remainder lands on the last stage."""
    per = epochs // n_blocks
    schedule = [per] * n_blocks
    schedule[-1] = epochs - per * (n_blocks - 1)
    return schedule


# ================================================================
# Main training routine
# ================================================================

@register("belilovsky")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    batch_size = cfg.get("batch_size", 128)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)
    use_bn = cfg.get("use_bn", True)

    # Per-block (in_ch, out_ch, kernel, padding, pool) specs for the chosen arch.
    block_specs = make_conv_blocks(arch)
    n_blocks = len(block_specs)

    # Epoch budget split across stages. Remainder goes to the final stage so
    # test_acc_curve length matches `epochs` exactly.
    epoch_schedule = _build_epoch_schedule(epochs, n_blocks)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, augment=False,
    )

    blocks: List[nn.Module] = []
    auxes: List[AuxHead] = []
    for spec in block_specs:
        blocks.append(ConvBlock(spec, use_bn=use_bn).to(device))
        auxes.append(AuxHead(spec["out_ch"]).to(device))

    criterion = nn.CrossEntropyLoss()
    acc_curve: List[float] = []
    loss_curve: List[float] = []
    t0 = time.time()
    best_acc = 0.0
    global_epoch = 0  # counts across all stages for recipe LR decay
    current_lr = lr    # may drop to lr*lr_decay_factor for stages started post-decay

    for stage_idx, n_ep in enumerate(epoch_schedule):
        frozen_blocks = blocks[:stage_idx]  # earlier blocks, already trained + frozen
        current_block = blocks[stage_idx]
        aux = auxes[stage_idx]

        # Make sure all frozen blocks are actually frozen
        for b in frozen_blocks:
            for p in b.parameters():
                p.requires_grad_(False)
            b.eval()

        optimizer = optim.Adam(
            list(current_block.parameters()) + list(aux.parameters()),
            lr=current_lr,
        )

        for ep in range(n_ep):
            # Phase II.1 recipe: decay current stage's LR at the global epoch
            # crossing `lr_decay_epoch` (0-based). Only the active optimizer
            # needs decaying; earlier stages are frozen, later stages not yet
            # instantiated. If the crossing falls inside an earlier stage, the
            # decay has already been applied then; stages started after that
            # point will be rebuilt below with decayed lr.
            apply_lr_decay(optimizer, global_epoch, lr,
                           at_epoch=lr_decay_epoch, factor=lr_decay_factor)
            current_block.train()
            aux.train()
            running = 0.0
            n_batches = 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                h = _forward_through_frozen(frozen_blocks, x)
                h = current_block(h)
                logits = aux(h)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running += loss.item()
                n_batches += 1

            avg_loss = running / max(n_batches, 1)
            loss_curve.append(avg_loss)

            acc = _eval_stage(frozen_blocks, current_block, aux, test_loader, device)
            acc_curve.append(acc)
            best_acc = max(best_acc, acc)

            if (ep + 1) % 10 == 0 or ep == 0 or ep == n_ep - 1:
                print(
                    f"  [Belilovsky/{arch}] stage {stage_idx + 1}/{n_blocks}  "
                    f"epoch {ep + 1}/{n_ep}  "
                    f"loss={avg_loss:.4f}  test_acc={acc:.4f}  best={best_acc:.4f}",
                    flush=True,
                )
            global_epoch += 1

        # If the next stage will start after the LR decay threshold, make
        # sure it is initialised at the decayed learning rate rather than the
        # initial one. (The next stage instantiates a fresh Adam from `current_lr`.)
        if global_epoch > lr_decay_epoch:
            current_lr = lr * lr_decay_factor

        # Freeze this stage's block before moving on.
        for p in current_block.parameters():
            p.requires_grad_(False)
        current_block.eval()

    # Total param count: all blocks + all aux heads (only the final aux is used
    # at inference, but the method trains every stage's aux).
    total_params = sum(count_params(m) for m in blocks) + sum(count_params(m) for m in auxes)

    return TrainResult(
        algo="belilovsky", status="ok", arch=arch,
        test_acc_final=acc_curve[-1] if acc_curve else None,
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=total_params,
        elapsed_s=time.time() - t0,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "Adam", "seed": cfg.get("seed", 0),
            "arch": arch,
            "n_blocks": n_blocks,
            "n_stages": n_blocks,
            "epoch_schedule": epoch_schedule,
            "aux_head": "Conv+ReLU+AvgPool1+Linear",
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": use_bn,
            "bn_status": "added_to_convblock",
        },
        gpu=str(device), timestamp=now_iso(),
    )
