"""
AugLocal (Ma et al., ICLR 2024) — "Scaling Supervised Local Learning with
Augmented Auxiliary Networks".

Each backbone block is trained with local CE + a detach-based gradient cut
between blocks. AugLocal's distinguishing feature is a *linearly decreasing*
aux-net depth schedule: the aux net at layer ell has remaining depth
`L - ell - 1`, so early layers get deeper aux nets (mirroring the remaining
backbone) and the last layer's aux is a pure linear classifier.

At L=3 this schedule collapses to {2, 1, 0} -- a very short ladder, barely
distinguishable from simpler local-CE baselines. The result is therefore
tagged `"degenerate"` on cnn3 to flag in the paper that AugLocal is not
meaningfully exercised at L=3. On cnn6 the schedule becomes {5, 4, 3, 2, 1, 0},
which is the paper's intended regime, so the tag is cleared.

Fair-comparison contract: Adam(lr=1e-3) per (block + aux net), batch_size=128,
epochs=200, no data augmentation, seed=0.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    ConvBlock, FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, TrainResult,
    apply_lr_decay, count_params, get_cifar10_loaders, make_conv_blocks,
    now_iso, register, set_seed,
)
from algos.nokland_base import NoklandCNN


class AugLocalAux(nn.Module):
    """Auxiliary network with `remaining_depth` conv stages + GAP + Linear.

    The conv stages mirror the backbone's remaining blocks (ell+1, ..., L-1),
    reusing common.ConvBlock so the channel and pool schedule matches the
    backbone exactly for the active arch. Terminated by AdaptiveAvgPool(1)
    and a linear classifier.

    Example L=3 backbone (cnn3, pools=max,max,none; channels 32,64,128):
        ell=0 -> depth 2: Conv(32->64)+max + Conv(64->128)   + GAP + Linear
        ell=1 -> depth 1: Conv(64->128)                       + GAP + Linear
        ell=2 -> depth 0: GAP + Linear                        (pure linear)
    Example L=6 backbone (cnn6, pools=none,max,none,max,none,none;
                          channels 32,32,64,64,128,128):
        ell=0 -> depth 5  ...  ell=5 -> depth 0
    """

    def __init__(self, ell: int, backbone_specs, num_classes: int,
                 use_bn: bool = False):
        super().__init__()
        L = len(backbone_specs)
        assert 0 <= ell < L, "ell out of range"
        stages = []
        in_ch = backbone_specs[ell]["out_ch"]
        # Mirror the backbone's remaining blocks (ell+1, ..., L-1).
        for j in range(ell + 1, L):
            spec = dict(backbone_specs[j])          # copy
            spec["in_ch"] = in_ch                    # rewire for aux net chaining
            stages.append(ConvBlock(spec, use_bn=use_bn))
            in_ch = spec["out_ch"]
        self.stages = nn.Sequential(*stages) if stages else nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_ch, num_classes)
        self.remaining_depth = L - ell - 1

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.stages(h)
        z = self.gap(z).reshape(z.shape[0], -1)
        return self.linear(z)


def _build_aux_nets(backbone_specs, num_classes: int, use_bn: bool = False):
    L = len(backbone_specs)
    return nn.ModuleList([
        AugLocalAux(ell, backbone_specs, num_classes, use_bn=use_bn)
        for ell in range(L)
    ])


def _make_block_optimizers(model: NoklandCNN, aux_nets: nn.ModuleList, lr: float):
    opts = []
    for block, aux in zip(model.blocks, aux_nets):
        params = list(block.parameters()) + list(aux.parameters())
        opts.append(optim.Adam(params, lr=lr))
    return opts


@torch.no_grad()
def _evaluate(model: NoklandCNN, aux_nets: nn.ModuleList, loader, device) -> float:
    model.eval()
    aux_nets.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        acts = model.forward_eval(x)
        logits = aux_nets[-1](acts[-1])
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.shape[0]
    return correct / max(total, 1)


@register("auglocal")
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

    # Aux nets follow the backbone block specs exactly (channels & pools).
    specs = make_conv_blocks(arch)
    aux_nets = _build_aux_nets(specs, FAIR_NUM_CLASSES, use_bn=use_bn).to(device)
    aux_depths = [aux.remaining_depth for aux in aux_nets]

    opts = _make_block_optimizers(model, aux_nets, lr)

    loss_curve, acc_curve = [], []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        apply_lr_decay(opts, epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor,
                       verbose=(epoch == lr_decay_epoch))
        model.train()
        aux_nets.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward each backbone block on the detached activation from the
            # previous block and train (block, aux) jointly with local CE.
            h = x
            block_losses = []
            for block, aux, opt in zip(model.blocks, aux_nets, opts):
                h = block(h.detach())
                logits = aux(h)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                block_losses.append(loss.item())
            running += sum(block_losses) / len(block_losses)
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        loss_curve.append(avg_loss)

        acc = _evaluate(model, aux_nets, test_loader, device)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [AugLocal] epoch {epoch + 1}/{epochs}  "
                  f"mean_block_loss={avg_loss:.4f}  test_acc={acc:.4f}  "
                  f"best={best_acc:.4f}", flush=True)

    n_params = count_params(model) + count_params(aux_nets)
    tag = "degenerate" if arch == "cnn3" else None
    return TrainResult(
        algo="auglocal", status="ok", arch=arch,
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
            "block_channels": [s["out_ch"] for s in specs],
            "backbone_pools": [s["pool"] for s in specs],
            "aux_net_depths": aux_depths,
            "aux_structure": (
                "depth-d stages mirror backbone blocks (ell+1..L-1); "
                "each stage = conv3x3 + relu (+ maxpool if the backbone pools); "
                "terminated by AdaptiveAvgPool(1) + Linear(C_last, 10)"
            ),
            "loss": "local CE at every (block + aux) pair",
            "eval": "argmax of final aux (pure linear head on deepest features)",
            "seed": cfg.get("seed", 0),
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": use_bn,
            "bn_status": "added_to_convblock",
        },
        gpu=str(device), timestamp=now_iso(),
        tag=tag,  # L=3 shrinks AugLocal's depth schedule to {2,1,0}; L=6 clears tag
    )
