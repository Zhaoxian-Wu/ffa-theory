"""Greedy InfoMax (Loewe, O'Connor, Veeling, NeurIPS 2019, Eq. 1-4, Sec. 4.1).

Per-module InfoNCE on a patch grid with k-step ahead bilinear prediction:
    f_k(z_{t+k}, c_t) = exp(z_{t+k}^T W_k c_t)                        (Eq. 1)
    L_N = -E[ log f_k(z_pos, c) / sum_{z in {pos,neg}} f_k(z, c) ]    (Eq. 2)
Negatives: all patches of OTHER images in the same batch (paper Sec. 3.2).

CNN6: split CNN6Backbone into two gradient-isolated modules.
  Module 1 (conv1..conv3): image -> h3 (B, 64, 16, 16), 4x4 patch grid.
  Module 2 (conv4..conv5, fed h3.detach()): h3 -> h5 (B, 128, 8, 8), 2x2 grid.
patch_size=4 both; context-to-target shift k=1 horizontal. Per-module encoder
C -> D=64 + bilinear predictor W_k (D, D). Eval: linear probe on GAP(h5).

CNN3: L=3 RF saturates 32x32 after second maxpool (h3 is 8x8); patch hierarchy
collapses to global pixel-vs-pixel contrast -- not a regime the paper covers.
Returned as N/A.

Fair contract: Adam(lr=1e-3) per module, batch_size=128, cfg.epochs total
(split equally between self-sup and linear-probe phases), no aug.

Downgrade (quick mode): dual best_acc <= 20% -> single-module on h3 with
tag="single_module"; if still <= 15% -> tag="degenerate_patch_grid".
"""
from __future__ import annotations

import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from common import (
    FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, TrainResult,
    apply_lr_decay, count_params, get_cifar10_loaders, make_backbone, now_iso,
    na_result, register, set_seed,
)

def extract_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, nh*nw, C): mean-pool inside each patch_size area."""
    B, C, H, W = x.shape
    nh, nw = H // patch_size, W // patch_size
    x = x.reshape(B, C, nh, patch_size, nw, patch_size).permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, nh * nw, C, patch_size, patch_size)
    return x.mean(dim=(3, 4))

class InfoNCEModule(nn.Module):
    """Per-patch encoder C -> D plus learnable bilinear predictor W_k (D, D)."""

    def __init__(self, in_ch: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_ch, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.W_k = nn.Parameter(
            torch.eye(embed_dim) + 0.01 * torch.randn(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim

    def embed(self, patches: torch.Tensor) -> torch.Tensor:
        B, P, C = patches.shape
        return self.encoder(patches.reshape(B * P, C)).reshape(B, P, self.embed_dim)

def _shift_pairs(nh: int, nw: int, k: int = 1) -> Tuple[List[int], List[int]]:
    """Flat indices (i*nw+j) for ctx (i, j) and tgt (i, j+k), j+k < nw."""
    ctx, tgt = [], []
    for i in range(nh):
        for j in range(nw - k):
            ctx.append(i * nw + j)
            tgt.append(i * nw + (j + k))
    return ctx, tgt

def infonce_loss(module: InfoNCEModule, patches: torch.Tensor,
                 ctx_idx: List[int], tgt_idx: List[int]) -> torch.Tensor:
    """Paper Eq. (2): InfoNCE across patch pairs; negatives = OTHER images' patches."""
    B, P, _ = patches.shape
    D = module.embed_dim
    z = module.embed(patches)                                  # (B, P, D)
    c_all = torch.einsum("de,bpe->bpd", module.W_k, z)         # (B, P, D)
    ctx_t = torch.tensor(ctx_idx, device=patches.device, dtype=torch.long)
    tgt_t = torch.tensor(tgt_idx, device=patches.device, dtype=torch.long)
    c = c_all.index_select(1, ctx_t)                           # (B, N, D)
    pos_z = z.index_select(1, tgt_t)                           # (B, N, D)
    pos_logits = (c * pos_z).sum(dim=-1)                       # (B, N)
    z_flat = z.reshape(B * P, D)                               # (B*P, D)
    neg_logits = torch.einsum("bnd,md->bnm", c, z_flat)        # (B, N, B*P)
    col_img = torch.arange(B * P, device=patches.device) // P
    same = (col_img.unsqueeze(0) == torch.arange(B, device=patches.device).unsqueeze(1))
    neg_logits = neg_logits.masked_fill(same.unsqueeze(1), float("-inf"))
    all_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
    return -F.log_softmax(all_logits, dim=-1)[..., 0].mean()

class _Slice1(nn.Module):
    """conv1 + conv2(+pool) + conv3 of CNN6 -> h3 (B, 64, 16, 16).

    Phase II.1 recipe: optional BatchNorm2d inserted between each conv and its
    ReLU (use_bn=True).
    """
    def __init__(self, bb: nn.Module, use_bn: bool = False):
        super().__init__()
        self.conv1, self.conv2, self.conv3, self.pool = bb.conv1, bb.conv2, bb.conv3, bb.pool
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels) if use_bn else nn.Identity()
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels) if use_bn else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.bn1(self.conv1(x)))
        h2 = self.pool(torch.relu(self.bn2(self.conv2(h1))))
        return torch.relu(self.bn3(self.conv3(h2)))

class _Slice2(nn.Module):
    """conv4(+pool) + conv5 of CNN6, fed h3.detach() -> h5 (B, 128, 8, 8).

    Phase II.1 recipe: optional BatchNorm2d inserted between each conv and its
    ReLU (use_bn=True).
    """
    def __init__(self, bb: nn.Module, use_bn: bool = False):
        super().__init__()
        self.conv4, self.conv5, self.pool = bb.conv4, bb.conv5, bb.pool
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels) if use_bn else nn.Identity()
        self.bn5 = nn.BatchNorm2d(self.conv5.out_channels) if use_bn else nn.Identity()
    def forward(self, h3: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn5(self.conv5(self.pool(torch.relu(self.bn4(self.conv4(h3)))))))

class _Identity(nn.Module):
    def forward(self, h: torch.Tensor) -> torch.Tensor: return h

def _extract_feats(enc1: nn.Module, enc2: nn.Module, loader: DataLoader,
                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    enc1.eval(); enc2.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            v = F.adaptive_avg_pool2d(enc2(enc1(x)), 1).flatten(1)
            feats.append(v.cpu()); labels.append(y)
    return torch.cat(feats), torch.cat(labels)

def _linear_probe(enc1: nn.Module, enc2: nn.Module,
                  train_loader: DataLoader, test_loader: DataLoader,
                  device: torch.device, probe_epochs: int, lr: float,
                  feat_dim: int) -> Tuple[float, float, List[float]]:
    tr_f, tr_y = _extract_feats(enc1, enc2, train_loader, device)
    te_f, te_y = _extract_feats(enc1, enc2, test_loader, device)
    tr_f = tr_f.to(device); tr_y = tr_y.to(device)
    te_f = te_f.to(device); te_y = te_y.to(device)
    probe = nn.Linear(feat_dim, FAIR_NUM_CLASSES).to(device)
    opt = optim.Adam(probe.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    bsz, n = 128, tr_f.shape[0]
    curve: List[float] = []; best = 0.0
    for _ in range(probe_epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bsz):
            idx = perm[s:s + bsz]
            loss = crit(probe(tr_f[idx]), tr_y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            acc = (probe(te_f).argmax(dim=1) == te_y).float().mean().item()
        curve.append(acc); best = max(best, acc)
    return curve[-1], best, curve

def _run_selfsup(slice1: nn.Module, slice2: nn.Module,
                 mod1: InfoNCEModule, mod2: InfoNCEModule,
                 opt1: optim.Optimizer, opt2: optim.Optimizer,
                 loader: DataLoader, device: torch.device,
                 epochs: int, use_module2: bool, arch: str,
                 lr_init: float = FAIR_LR,
                 lr_decay_epoch: int = 100,
                 lr_decay_factor: float = 0.1) -> Tuple[List[float], bool]:
    """Run the self-supervised patch-InfoNCE phase.

    Phase II.1 recipe: LR decay (new_lr = lr_init * lr_decay_factor) is applied
    at the start of `lr_decay_epoch` (0-based). Returns (loss_curve, decay_applied).
    """
    loss_curve: List[float] = []
    ctx1, tgt1 = _shift_pairs(4, 4, k=1)
    ctx2, tgt2 = _shift_pairs(2, 2, k=1)
    decay_applied = False
    for ep in range(epochs):
        opts = [opt1, opt2] if use_module2 else [opt1]
        if apply_lr_decay(opts, ep, lr_init,
                          at_epoch=lr_decay_epoch, factor=lr_decay_factor,
                          verbose=True):
            decay_applied = True
        slice1.train(); mod1.train()
        if use_module2:
            slice2.train(); mod2.train()
        running, n_b = 0.0, 0
        for x, _y in loader:
            x = x.to(device, non_blocking=True)
            # Module 1 update: h3 patch-InfoNCE on (conv1..conv3).
            h3 = slice1(x)
            patches1 = extract_patches(h3, patch_size=4)          # (B, 16, 64)
            loss1 = infonce_loss(mod1, patches1, ctx1, tgt1)
            opt1.zero_grad(); loss1.backward(); opt1.step()
            total = float(loss1.detach())
            if use_module2:
                # Module 2 update: h5 patch-InfoNCE on (conv4..conv5), fed h3.detach().
                with torch.no_grad():
                    h3_det = slice1(x).detach()
                h5 = slice2(h3_det)
                patches2 = extract_patches(h5, patch_size=4)      # (B, 4, 128)
                loss2 = infonce_loss(mod2, patches2, ctx2, tgt2)
                opt2.zero_grad(); loss2.backward(); opt2.step()
                total = 0.5 * (total + float(loss2.detach()))
            running += total; n_b += 1
        avg = running / max(n_b, 1)
        loss_curve.append(avg)
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            t = "dual" if use_module2 else "single"
            print(f"  [GIM/{arch}/{t}] epoch {ep + 1}/{epochs}  ssl_loss={avg:.4f}",
                  flush=True)
    return loss_curve, decay_applied

@register("gim")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    batch_size = cfg.get("batch_size", 128)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 fair-regularised recipe knobs.
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)
    use_bn = cfg.get("use_bn", True)

    if arch == "cnn3":
        return na_result(
            "gim",
            "GIM is deployed with M=3 gradient-isolated modules where each module "
            "is itself a ResNet-50 residual block group (tens of conv layers); "
            "InfoNCE predicts k=5 future rows on a 7x7 patch grid of 64x64 inputs "
            "(Loewe 2019 Sec. 4.1, App. A.1). Under our L=3 single-conv-layer "
            "fair-comparison contract on 32x32 CIFAR-10, the combined receptive "
            "field saturates the image after the second maxpool and the patch "
            "hierarchy collapses, reducing InfoNCE to a global pixel-vs-pixel "
            "contrast --- a regime not evaluated in the original paper.",
            arch=arch,
        )

    # --- CNN6 branch: dual-module patch-InfoNCE. ---
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, augment=False)
    bb = make_backbone("cnn6").to(device)
    slice1 = _Slice1(bb, use_bn=use_bn).to(device)
    slice2 = _Slice2(bb, use_bn=use_bn).to(device)
    mod1 = InfoNCEModule(in_ch=64, embed_dim=64).to(device)
    mod2 = InfoNCEModule(in_ch=128, embed_dim=64).to(device)
    opt1 = optim.Adam(list(slice1.parameters()) + list(mod1.parameters()), lr=lr)
    opt2 = optim.Adam(list(slice2.parameters()) + list(mod2.parameters()), lr=lr)

    ssl_epochs = max(1, epochs // 2)
    probe_epochs = max(1, epochs - ssl_epochs)
    t0 = time.time(); tag = None

    ssl_loss, decay_applied = _run_selfsup(
        slice1, slice2, mod1, mod2, opt1, opt2,
        train_loader, device, ssl_epochs, True, arch,
        lr_init=lr, lr_decay_epoch=lr_decay_epoch,
        lr_decay_factor=lr_decay_factor,
    )
    final_acc, best_acc, acc_curve = _linear_probe(
        slice1, slice2, train_loader, test_loader, device,
        probe_epochs=probe_epochs, lr=lr, feat_dim=128,
    )

    # Downgrade protocol (smoke-test regime).
    if epochs <= 5 and best_acc <= 0.20:
        tag = "single_module"
        # Fresh reinit: single-module on h3 alone.
        bb = make_backbone("cnn6").to(device)
        slice1 = _Slice1(bb, use_bn=use_bn).to(device)
        slice2 = _Slice2(bb, use_bn=use_bn).to(device)
        mod1 = InfoNCEModule(in_ch=64, embed_dim=64).to(device)
        opt1 = optim.Adam(list(slice1.parameters()) + list(mod1.parameters()), lr=lr)
        ssl_loss, decay_applied = _run_selfsup(
            slice1, slice2, mod1, mod2, opt1, opt2,
            train_loader, device, ssl_epochs, False, arch,
            lr_init=lr, lr_decay_epoch=lr_decay_epoch,
            lr_decay_factor=lr_decay_factor,
        )
        final_acc, best_acc, acc_curve = _linear_probe(
            slice1, _Identity().to(device), train_loader, test_loader, device,
            probe_epochs=probe_epochs, lr=lr, feat_dim=64,
        )
        if best_acc <= 0.15:
            tag = "degenerate_patch_grid"

    total_params = (count_params(slice1) + count_params(slice2)
                    + count_params(mod1) + count_params(mod2))
    n_modules = 1 if tag in ("single_module", "degenerate_patch_grid") else 2

    hp = {
        "lr": lr, "epochs": epochs, "batch_size": batch_size,
        "ssl_epochs": ssl_epochs, "probe_epochs": probe_epochs,
        "optimizer": "Adam(per-module)", "arch": arch, "n_modules": n_modules,
        "module1": "h3 (B,64,16,16) -> 4x4 patch grid, D=64, k=1 horizontal",
        "module2": "h5 (B,128,8,8) -> 2x2 patch grid, D=64, k=1 horizontal",
        "loss": "InfoNCE (Loewe 2019 Eq. 2) with bilinear W_k predictor",
        "negatives": "in-batch: all patches of OTHER images",
        "eval": "linear probe on GAP(h5); GAP(h3) under single_module",
        "seed": cfg.get("seed", 0),
        "lr_decay_epoch": lr_decay_epoch,
        "lr_decay_factor": lr_decay_factor,
        "lr_decay_applied": decay_applied,
        "use_bn": use_bn,
        "bn_status": "added_to_convblock" if use_bn else "skipped_unavailable",
    }
    return TrainResult(
        algo="gim", status="ok", arch=arch,
        test_acc_final=final_acc, test_acc_best=best_acc,
        test_acc_curve=acc_curve, train_loss_curve=ssl_loss,
        n_params=total_params, elapsed_s=time.time() - t0,
        hyperparams=hp, gpu=str(device), timestamp=now_iso(), tag=tag,
    )
