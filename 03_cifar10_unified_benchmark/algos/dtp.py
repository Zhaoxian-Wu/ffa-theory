"""
Difference Target Propagation (DTP).

Refs:
  Lee et al. "Difference Target Propagation." ECML 2015.
  Ernoult et al. "Towards Scaling Difference Target Propagation by Learning
  Backprop Targets." ICML 2022 (arXiv 2201.12403).

Each forward layer f_l is paired with an inverse g_l trained to satisfy
g_l(f_l(h_{l-1})) ~= h_{l-1}. Targets propagate backward via the difference
rule  t_{l-1} = g_l(t_l) + h_{l-1} - g_l(h_l); every forward layer minimises
||f_l(h_{l-1}) - t_l||^2 locally (no cross-layer gradients). The top target
is one gradient step on CE w.r.t. h_L.

Arch-parameterised: forward blocks and per-block inverses are derived from
`make_conv_blocks(arch)`. Each inverse g_l must map h_l back to the shape of
h_{l-1}:
  - pool == "max":  forward halves spatial; inverse is ConvTranspose2d with
                    stride 2, kernel 4, padding 1 (exact 2x upsample for even sides).
  - pool == "none": forward preserves spatial; inverse is same-resolution
                    Conv2d(3x3, padding 1).
  - pool == "avg4": not used in the fair block specs (avg4 belongs to the
                    classifier head). Guarded with a NotImplementedError.
Inverses use tanh on the output to keep targets bounded.

Classifier: AdaptiveAvgPool(head_avgpool_size) + Linear(final_fc_in_dim, 256)
+ ReLU + Linear(256, 10). Per-block Adam. DTP is fragile; we clip grads and
fail fast on NaN.
"""
from __future__ import annotations

import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from common import (
    FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, ConvBlock, TrainResult,
    apply_lr_decay, count_params, evaluate_classifier, final_fc_in_dim,
    final_feat_channels, get_cifar10_loaders, head_avgpool_size,
    make_conv_blocks, now_iso, num_blocks, register, set_seed,
)


# ================================================================
# Forward / inverse / classifier modules
# ================================================================

class InverseBlock(nn.Module):
    """g_l: maps h_l back to h_{l-1} shape. Trailing tanh keeps it bounded.

    Derived from the corresponding forward block's spec:
      - pool == "max":  ConvTranspose2d(out_ch -> in_ch, k=4, s=2, p=1)  # 2x upsample
      - pool == "none": Conv2d(out_ch -> in_ch, k=3, p=1)                # same spatial
    """

    def __init__(self, spec: dict):
        super().__init__()
        pool = spec["pool"]
        if pool == "max":
            self.op: nn.Module = nn.ConvTranspose2d(
                spec["out_ch"], spec["in_ch"], kernel_size=4, stride=2, padding=1,
            )
        elif pool == "none":
            self.op = nn.Conv2d(
                spec["out_ch"], spec["in_ch"], kernel_size=3, padding=1,
            )
        else:
            # avg4 is reserved for the classifier head and should not appear
            # in per-block specs produced by make_conv_blocks(cnn3|cnn6).
            raise NotImplementedError(
                f"InverseBlock does not support pool='{pool}' "
                f"(expected 'max' or 'none' in fair block specs)."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.op(x))


class Classifier(nn.Module):
    """AdaptiveAvgPool(s) + Linear(final_fc_in_dim -> feat_dim) + ReLU + Linear(feat_dim -> C).

    The pool size s and fc input dim adapt to the chosen arch via
    head_avgpool_size(arch) / final_fc_in_dim(arch), so the classifier head
    is numerically identical to CNN3Backbone / CNN6Backbone heads.
    """

    def __init__(self, arch: str, feat_dim: int = 256,
                 num_classes: int = FAIR_NUM_CLASSES):
        super().__init__()
        s = head_avgpool_size(arch)
        self.avgpool = nn.AdaptiveAvgPool2d(s)
        self.fc1 = nn.Linear(final_fc_in_dim(arch), feat_dim)
        self.fc2 = nn.Linear(feat_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.avgpool(h).flatten(1)
        return self.fc2(torch.relu(self.fc1(z)))


# ================================================================
# Failure helper
# ================================================================

def _failed(t0: float, cfg: dict, device: torch.device, reason: str,
            arch: str) -> TrainResult:
    return TrainResult(
        algo="dtp", status="failed", arch=arch,
        elapsed_s=time.time() - t0,
        hyperparams={
            "lr": cfg.get("lr", FAIR_LR), "epochs": cfg.get("epochs", 200),
            "batch_size": cfg.get("batch_size", 128),
            "optimizer": "Adam", "seed": cfg.get("seed", 0),
            "arch": arch,
        },
        gpu=str(device), timestamp=now_iso(), error_msg=reason,
    )


# ================================================================
# Main training routine
# ================================================================

@register("dtp")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    batch_size = cfg.get("batch_size", 128)
    eta = cfg.get("dtp_eta", 0.1)
    grad_clip = cfg.get("grad_clip", 1.0)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 recipe knobs. DTP's target-prop rule depends on raw activation
    # scales; BN inside ConvBlock rescales them and may destabilise inverse
    # reconstruction. We still try BN by default (first preference); if the
    # forward MSE diverges or produces NaN, _failed() returns with a tagged
    # bn_status so the orchestration layer can opt to disable BN next run.
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)
    use_bn = cfg.get("use_bn", True)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, augment=False,
    )

    # Arch-parameterised forward / inverse stack.
    block_specs = make_conv_blocks(arch)
    n_layers = num_blocks(arch)

    f_blocks: List[nn.Module] = [ConvBlock(spec, use_bn=use_bn).to(device) for spec in block_specs]
    g_blocks: List[nn.Module] = [InverseBlock(spec).to(device) for spec in block_specs]
    classifier = Classifier(arch=arch).to(device)

    f_opts = [optim.Adam(b.parameters(), lr=lr) for b in f_blocks]
    g_opts = [optim.Adam(b.parameters(), lr=lr) for b in g_blocks]
    clf_opt = optim.Adam(classifier.parameters(), lr=lr)

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    acc_curve: List[float] = []
    loss_curve: List[float] = []
    t0 = time.time()
    best_acc = 0.0

    def eval_logits(x: torch.Tensor) -> torch.Tensor:
        h = x
        for b in f_blocks:
            h = b(h)
        return classifier(h)

    for epoch in range(epochs):
        # Recipe: lr decay on all three optimizer groups (forward, inverse, clf).
        apply_lr_decay([*f_opts, *g_opts, clf_opt], epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor)
        for b in f_blocks:
            b.train()
        for b in g_blocks:
            b.train()
        classifier.train()

        running = 0.0
        n_batches = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # (1) Forward pass, no autograd: collect activations h_0 .. h_L.
            with torch.no_grad():
                h = x
                acts: List[torch.Tensor] = [h]
                for b in f_blocks:
                    h = b(h)
                    acts.append(h)

            # (2) Classifier CE + derive top target  h_top - eta * d(CE)/d(h_top).
            h_top = acts[-1].clone().detach().requires_grad_(True)
            logits = classifier(h_top)
            clf_loss = ce(logits, y)
            clf_opt.zero_grad()
            clf_loss.backward()
            grad_h_top = h_top.grad.detach() if h_top.grad is not None else torch.zeros_like(h_top)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
            clf_opt.step()
            target_top = acts[-1] - eta * grad_h_top

            # (3) Auto-encoder losses for inverses:
            #     g_l(f_l(h_{l-1}+noise)) ~= h_{l-1}+noise.
            for l in range(n_layers):
                h_prev = acts[l].detach()
                h_noisy = h_prev + 0.1 * torch.randn_like(h_prev)
                with torch.no_grad():
                    h_cur = f_blocks[l](h_noisy)
                recon = g_blocks[l](h_cur)
                g_loss = mse(recon, h_noisy)
                g_opts[l].zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(g_blocks[l].parameters(), grad_clip)
                g_opts[l].step()

            # (4) Difference-target backward propagation.
            with torch.no_grad():
                targets: List[torch.Tensor] = [None] * (n_layers + 1)  # type: ignore
                targets[n_layers] = target_top
                for l in range(n_layers - 1, -1, -1):
                    t_l = targets[l + 1]
                    targets[l] = g_blocks[l](t_l) + acts[l] - g_blocks[l](acts[l + 1])

            # (5) Forward-layer local MSE updates: ||f_l(h_{l-1}) - t_l||^2.
            batch_loss = 0.0
            for l in range(n_layers):
                h_prev = acts[l].detach()
                target_l = targets[l + 1].detach()
                out = f_blocks[l](h_prev)
                if not torch.isfinite(out).all() or not torch.isfinite(target_l).all():
                    return _failed(t0, cfg, device,
                                   reason=f"NaN at layer {l + 1}, epoch {epoch}",
                                   arch=arch)
                f_loss = mse(out, target_l)
                f_opts[l].zero_grad()
                f_loss.backward()
                torch.nn.utils.clip_grad_norm_(f_blocks[l].parameters(), grad_clip)
                f_opts[l].step()
                batch_loss += f_loss.item()

            running += batch_loss + clf_loss.item()
            n_batches += 1

        avg_loss = running / max(n_batches, 1)
        if not (avg_loss == avg_loss):
            return _failed(t0, cfg, device, reason=f"Loss NaN at epoch {epoch}",
                           arch=arch)
        loss_curve.append(avg_loss)

        for b in f_blocks:
            b.eval()
        classifier.eval()
        acc = evaluate_classifier(eval_logits, test_loader, device)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(
                f"  [DTP/{arch}] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  "
                f"test_acc={acc:.4f}  best={best_acc:.4f}",
                flush=True,
            )

    total_params = (
        sum(count_params(m) for m in f_blocks)
        + sum(count_params(m) for m in g_blocks)
        + count_params(classifier)
    )
    return TrainResult(
        algo="dtp", status="ok", arch=arch,
        test_acc_final=acc_curve[-1] if acc_curve else None,
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=total_params,
        elapsed_s=time.time() - t0,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "optimizer": "Adam", "seed": cfg.get("seed", 0),
            "dtp_eta": eta, "grad_clip": grad_clip,
            "arch": arch,
            "n_layers": n_layers,
            "final_feat_channels": final_feat_channels(arch),
            "inverse_noise_std": 0.1,
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": use_bn,
            "bn_status": "added_to_convblock" if use_bn else "disabled",
        },
        gpu=str(device), timestamp=now_iso(),
    )
