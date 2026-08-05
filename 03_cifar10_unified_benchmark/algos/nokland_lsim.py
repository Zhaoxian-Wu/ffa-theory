"""Nokland L_sim: similarity-matching local learning (Nokland & Eidnes, ICML 2019).

Per-block local loss: L_sim = || S - Y ||_F^2 / B^2, where
    S[i,j] = cos(phi(h_l[i]), phi(h_l[j])),  Y[i,j] = 1{y_i == y_j}.
Activations are detached between blocks. A separate linear classifier trained
on detached last-block features provides test-time logits (no gradient flows
back into the backbone). Self-contained: no import from nokland_base.py.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    ConvBlock, FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, TrainResult,
    apply_lr_decay, count_params, evaluate_classifier, final_feat_channels,
    get_cifar10_loaders, make_conv_blocks, now_iso, register, set_seed,
)


class SimProjector(nn.Module):
    """Standardization head (Nokland 2019 Sec.3.2): GAP -> linear -> l2-normalize."""

    def __init__(self, in_ch: int, proj_dim: int = 128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, proj_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.gap(h).flatten(1)
        z = self.proj(z)
        return F.normalize(z, dim=1)


def sim_loss(feat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L_sim = || S - Y ||_F^2 / B^2, where S = feat @ feat.T (feat is l2-normalized),
    Y[i,j] = 1 iff y_i == y_j."""
    S = feat @ feat.t()
    B = y.shape[0]
    Y = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    return ((S - Y) ** 2).sum() / (B * B)


# ================================================================
# Training
# ================================================================

@register("nokland_lsim")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    proj_dim = cfg.get("proj_dim", 128)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 recipe: LR decay + BN inside ConvBlock (applied equally to all algos).
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)
    use_bn = cfg.get("use_bn", True)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.get("batch_size", 128), augment=False,
    )

    # Arch-parameterised conv blocks (same layout as CNN3/CNN6 backbone).
    specs = make_conv_blocks(arch)
    blocks: List[nn.Module] = [ConvBlock(s, use_bn=use_bn).to(device) for s in specs]
    sim_heads: List[SimProjector] = [
        SimProjector(s["out_ch"], proj_dim).to(device) for s in specs
    ]
    last_ch = final_feat_channels(arch)

    # Eval-only linear classifier on detached last-block features.
    eval_gap = nn.AdaptiveAvgPool2d(1)
    eval_head = nn.Linear(last_ch, FAIR_NUM_CLASSES).to(device)

    # One optimizer per block (block conv + its sim head); one for eval head.
    block_opts = [
        optim.Adam(list(blocks[i].parameters()) + list(sim_heads[i].parameters()), lr=lr)
        for i in range(len(blocks))
    ]
    eval_opt = optim.Adam(eval_head.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    def forward_classify(x: torch.Tensor) -> torch.Tensor:
        """Used at eval time: features are detached from backbone."""
        with torch.no_grad():
            h = x
            for blk in blocks:
                h = blk(h)
            feat = eval_gap(h).flatten(1)
        return eval_head(feat)

    def n_total_params() -> int:
        n = 0
        for m in blocks + sim_heads + [eval_head]:
            n += count_params(m)
        return n

    acc_curve, loss_curve = [], []
    t0 = time.time()
    best_acc = 0.0
    for epoch in range(epochs):
        apply_lr_decay(block_opts + [eval_opt], epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor,
                       verbose=(epoch == lr_decay_epoch))
        for m in blocks + sim_heads:
            m.train()
        eval_head.train()

        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            h = x
            total_loss_val = 0.0
            # Per-block local training: detach activations between blocks.
            for i, blk in enumerate(blocks):
                h_in = h.detach().requires_grad_(False)
                h_out = blk(h_in)
                feat = sim_heads[i](h_out)
                loss_i = sim_loss(feat, y)
                block_opts[i].zero_grad()
                loss_i.backward()
                block_opts[i].step()
                total_loss_val += loss_i.item()
                h = h_out.detach()   # pass detached activation to next block

            # Eval-head training on fully detached last-block features.
            feat_eval = eval_gap(h).flatten(1)
            logits = eval_head(feat_eval)
            eval_loss = ce(logits, y)
            eval_opt.zero_grad()
            eval_loss.backward()
            eval_opt.step()

            running += total_loss_val + eval_loss.item()
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        loss_curve.append(avg_loss)

        for m in blocks + sim_heads:
            m.eval()
        eval_head.eval()
        acc = evaluate_classifier(forward_classify, test_loader, device)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [Nokland-Lsim] epoch {epoch + 1}/{epochs}  "
                  f"loss={avg_loss:.4f}  test_acc={acc:.4f}  best={best_acc:.4f}",
                  flush=True)

    return TrainResult(
        algo="nokland_lsim", status="ok", arch=arch,
        test_acc_final=acc_curve[-1],
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=n_total_params(),
        elapsed_s=time.time() - t0,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": cfg.get("batch_size", 128),
                     "optimizer": "Adam", "proj_dim": proj_dim, "seed": cfg.get("seed", 0),
                     "arch": arch, "n_blocks": len(blocks),
                     "block_channels": [s["out_ch"] for s in specs],
                     "loss": "L_sim (||S-Y||_F^2 / B^2)",
                     "lr_decay_epoch": lr_decay_epoch,
                     "lr_decay_factor": lr_decay_factor,
                     "use_bn": use_bn,
                     "bn_status": "added_to_convblock"},
        gpu=str(device), timestamp=now_iso(),
    )
