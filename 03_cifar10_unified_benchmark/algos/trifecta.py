"""
Trifecta: SymBa loss + BatchNorm + Online-Layer-Update (OLU).

Reference: "Trifecta: Three simple techniques for training deeper
Forward-Forward networks" (2023).

The original trifecta_optimized.py uses a 6- to 12-block VGG-style net and
augmentation — both violate our fair-comparison contract. Here we port the
three key techniques onto the CNN3 backbone:

  * Channels 4->32->64->128 (backbone pattern + 1 label-embedding channel)
  * BN -> Conv -> ReLU -> MaxPool/AvgPool (same block structure)
  * SymBa loss: softplus(-alpha * (g_pos - g_neg))
  * Each block has its own Adam optimizer; output is detached to the next.
  * OLU: odd/even iterations detach different blocks.

Label injection via Embedding(10, 32*32) -> reshape (1, 32, 32), concat to
image as 4th channel (exact replication of TrifectaModel.forward_train).
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from common import (
    FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, TrainResult,
    block_channels, count_params, get_cifar10_loaders, make_conv_blocks,
    now_iso, register, set_seed,
)

DIMS = 32


def _should_detach(olu: bool, index: int, iteration: int) -> bool:
    return (index % 2 == 0) ^ (iteration % 2 == 0) if olu else True


class TrifectaBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, index: int,
                 prev_params, lr: float, alpha: float, olu: bool,
                 pool: str = "max"):
        super().__init__()
        layers = [
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        ]
        if pool == "max":
            layers.append(nn.MaxPool2d(2))
        elif pool == "avg":
            layers.append(nn.AdaptiveAvgPool2d(4))
        elif pool == "none":
            pass
        self.inner = nn.Sequential(*layers)
        params = list(prev_params) + list(self.inner.parameters())
        self.optimizer = Adam(params, lr=lr)
        self.index = index
        self.alpha = alpha
        self.olu = olu

    def set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def forward(self, x, iteration):
        out = self.inner(x)
        activations = out.pow(2).flatten(start_dim=1)
        if not self.training:
            return out, activations.mean(1)
        if _should_detach(self.olu, self.index, iteration):
            pos, neg = activations.chunk(2)
            pos_g, neg_g = pos.mean(1), neg.mean(1)
            loss = F.softplus(-self.alpha * (pos_g - neg_g)).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return out.detach(), loss.item()
        return out, None


def _trifecta_pool_schedule(arch: str):
    """Derive trifecta's per-block pool schedule from make_conv_blocks.

    Trifecta's original design pools spatially between blocks (MaxPool2d) and
    terminates with an AvgPool(4) on the last block. We map common's
    per-block pool spec {"max","none","avg4"} to trifecta's {"max","none","avg"}
    and force the LAST block to use "avg" so the predicted goodness is computed
    over a canonical 4x4 footprint (Phase I convention for trifecta).
    """
    specs = make_conv_blocks(arch)
    pools = []
    for i, spec in enumerate(specs):
        if i == len(specs) - 1:
            pools.append("avg")                           # final avgpool(4)
        elif spec["pool"] == "max":
            pools.append("max")
        else:
            pools.append("none")
    return pools


class TrifectaModel(nn.Module):
    """N-block CNN Trifecta with 1 extra label-embedding channel prepended.

    Channels follow [4] + block_channels(arch): the +1 is the label-embedding
    channel (Trifecta's signature design). Pools follow the backbone layout
    with the final block forced to `avg` for a 4x4 goodness footprint.
    """

    def __init__(self, alpha: float, olu: bool, lr: float,
                 arch: str = FAIR_DEFAULT_ARCH):
        super().__init__()
        self.arch = arch
        self.embedding = nn.Embedding(FAIR_NUM_CLASSES, DIMS * DIMS)
        chs = block_channels(arch)
        pools = _trifecta_pool_schedule(arch)
        in_chs = [4] + list(chs[:-1])   # first block input: image(3)+label_emb(1)=4
        cfgs = list(zip(in_chs, chs, pools))
        self.blocks = nn.ModuleList()
        prev_params = list(self.embedding.parameters())
        for i, (c_in, c_out, pool) in enumerate(cfgs):
            block = TrifectaBlock(c_in, c_out, i, prev_params, lr=lr,
                                  alpha=alpha, olu=olu, pool=pool)
            self.blocks.append(block)
            prev_params = []  # already handed to the previous block
        self.n_blocks = len(self.blocks)
        self.channels = list(chs)
        self.pools = pools

    def set_lr(self, lr: float):
        for block in self.blocks:
            block.set_lr(lr)

    def forward_train(self, x, y_pos, y_neg, iteration):
        emb_pos = self.embedding(y_pos).view(-1, 1, DIMS, DIMS)
        emb_neg = self.embedding(y_neg).view(-1, 1, DIMS, DIMS)
        x_pos = torch.cat([x, emb_pos], dim=1)
        x_neg = torch.cat([x, emb_neg], dim=1)
        h = torch.cat([x_pos, x_neg], dim=0)
        for block in self.blocks:
            h, _ = block(h, iteration)

    @torch.no_grad()
    def predict(self, x, ensemble_last_n: int = 1):
        bs = x.size(0)
        layer_g = []
        for c in range(FAIR_NUM_CLASSES):
            y_try = torch.full((bs,), c, dtype=torch.long, device=x.device)
            emb = self.embedding(y_try).view(-1, 1, DIMS, DIMS)
            h = torch.cat([x, emb], dim=1)
            gs = []
            for block in self.blocks:
                h, g = block(h, iteration=0)
                gs.append(g)
            layer_g.append(torch.stack(gs))  # (L, B)
        all_g = torch.stack(layer_g)           # (K, L, B)
        ens_g = all_g[:, -ensemble_last_n:, :].mean(dim=1)  # (K, B)
        return ens_g.argmax(0)


@torch.no_grad()
def _evaluate(model, loader, device, ensemble_n):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model.predict(x, ensemble_last_n=ensemble_n)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


@register("trifecta")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    alpha = cfg.get("alpha", 4.0)
    olu = cfg.get("olu", True)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Trifecta evaluates by averaging goodness across the last N blocks; the
    # default `min(3, n_blocks)` keeps Phase-I cnn3 behavior (ens=3) and caps at
    # the available depth for cnn6.
    ensemble_n = cfg.get("ensemble_n", min(3, len(block_channels(arch))))
    # Phase II.1 recipe: LR decay only (TrifectaBlock already has BN before conv).
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.get("batch_size", 128), augment=False,
    )
    model = TrifectaModel(alpha=alpha, olu=olu, lr=lr, arch=arch).to(device)

    acc_curve = []
    t0 = time.time()
    best_acc = 0.0
    for epoch in range(epochs):
        if epoch == lr_decay_epoch:
            new_lr = lr * lr_decay_factor
            model.set_lr(new_lr)
            print(f"  [Trifecta] epoch {epoch}: lr -> {new_lr:g}", flush=True)
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_neg = torch.randint(0, FAIR_NUM_CLASSES - 1, y.shape, device=device)
            same = y_neg >= y
            y_neg[same] += 1
            model.forward_train(x, y, y_neg, iteration=epoch + 1)

        acc = _evaluate(model, test_loader, device, ensemble_n)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [Trifecta] epoch {epoch + 1}/{epochs}  test_acc={acc:.4f}  best={best_acc:.4f}",
                  flush=True)

    return TrainResult(
        algo="trifecta", status="ok", arch=arch,
        test_acc_final=acc_curve[-1],
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=[],
        n_params=count_params(model),
        elapsed_s=time.time() - t0,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": cfg.get("batch_size", 128),
                     "optimizer": "Adam-per-block", "alpha": alpha, "olu": olu,
                     "ensemble_n": ensemble_n,
                     "arch": arch, "n_blocks": model.n_blocks,
                     "channels": [4] + model.channels,
                     "pools": model.pools,
                     "seed": cfg.get("seed", 0),
                     "lr_decay_epoch": lr_decay_epoch,
                     "lr_decay_factor": lr_decay_factor,
                     "use_bn": True,
                     "bn_status": "native_bn_kept (TrifectaBlock has BN before conv)"},
        gpu=str(device), timestamp=now_iso(),
    )
