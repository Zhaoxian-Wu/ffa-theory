"""
Counter-Current Learning (Kao & Hariharan, NeurIPS 2024).

Ref: Kao, C.-H. and Hariharan, B., "Counter-Current Learning: A Biologically
Plausible Dual Network Approach for Deep Learning," NeurIPS 2024.
arXiv:2409.19841 / OpenReview: L3RYBqzRmF

Two parallel pathways flow in opposite directions (cortex-cerebellum analogue):
  Primary  (bottom-up): image    -> f_1 -> f_2 -> ... -> f_L -> h_L_p
  Feedback (top-down):  y_onehot -> g_L -> g_{L-1} -> ... -> g_1 -> h_1_f

Per-layer local signal: the paper's batch-normalised CCL objective. It
aligns the primary/feedback cross-sample correlation to the same-class target
matrix, and regularises each within-path correlation toward the identity. The
two paths are still explicitly stop-graded between adjacent layers, so each
local loss updates only its paired primary/feedback blocks. A
linear classifier on stop-graded final-primary features (AvgPool(head_size) +
Flatten + Linear) is trained via CE.

Arch-parameterised: primary uses `make_conv_blocks(arch)` + classifier head
AdaptiveAvgPool(head_avgpool_size) + Linear(final_fc_in_dim, num_classes).
Feedback pathway is built as a mirrored stack: starting from y_onehot we
produce a tensor matching the shape of h_L_p, then iteratively upsample /
conv to match the shape of each earlier primary activation.

There is intentionally no end-to-end fallback: a failed dual-path run is
reported as failed rather than being substituted with a BP result.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, ConvBlock, TrainResult,
    apply_lr_decay, block_channels, count_params, evaluate_classifier,
    final_fc_in_dim, final_feat_channels, get_cifar10_loaders,
    head_avgpool_size, make_conv_blocks, now_iso, num_blocks, register,
    set_seed,
)


# ================================================================
# Primary-side helpers
# ================================================================

def _primary_output_shapes(arch: str, in_spatial: int = 32) -> List[Tuple[int, int, int]]:
    """Compute (C, H, W) of each primary block's output for the given arch.

    Assumes CIFAR-10 input 3x32x32 and that make_conv_blocks(arch) only uses
    pool in {"max", "none"} (which is the case for cnn3/cnn6; avg4 is reserved
    for the classifier head and never appears in fair block specs).
    """
    shapes: List[Tuple[int, int, int]] = []
    sp = in_spatial
    for spec in make_conv_blocks(arch):
        if spec["pool"] == "max":
            sp = sp // 2
        elif spec["pool"] == "none":
            pass
        else:
            raise NotImplementedError(
                f"counter_current does not handle pool='{spec['pool']}' "
                f"(expected 'max' or 'none' in fair block specs)."
            )
        shapes.append((spec["out_ch"], sp, sp))
    return shapes


class PrimaryHead(nn.Module):
    """Classifier head on stop-graded final primary features.

    AdaptiveAvgPool(head_size) -> Flatten -> Linear(final_fc_in_dim, num_classes).
    Structurally equivalent to CNN3Backbone/CNN6Backbone heads, so the
    numerical contract remains fair across archs.
    """

    def __init__(self, arch: str, num_classes: int = FAIR_NUM_CLASSES):
        super().__init__()
        s = head_avgpool_size(arch)
        self.avgpool = nn.AdaptiveAvgPool2d(s)
        self.fc = nn.Linear(final_fc_in_dim(arch), num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(self.avgpool(h).flatten(1))


# ================================================================
# Feedback-side modules
# ================================================================

class FeedbackTop(nn.Module):
    """y_onehot -> tensor of shape (B, C, H, W) matching the last primary output."""

    def __init__(self, num_classes: int, ch: int, h: int, w: int):
        super().__init__()
        self.ch, self.h, self.w = ch, h, w
        self.fc = nn.Linear(num_classes, ch * h * w)

    def forward(self, y_oh: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(y_oh).view(-1, self.ch, self.h, self.w))


class FeedbackBlock(nn.Module):
    """Bilinear resize to (out_h, out_w) + Conv(in_ch -> out_ch) + ReLU.

    Same spatial target as the matching earlier primary activation.
    """

    def __init__(self, in_ch: int, out_ch: int, out_h: int, out_w: int):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(x, size=(self.out_h, self.out_w),
                          mode="bilinear", align_corners=False)
        return torch.relu(self.conv(h))


def _build_feedback_stack(arch: str, device: torch.device,
                          num_classes: int = FAIR_NUM_CLASSES) -> List[nn.Module]:
    """Return feedback[0..L-1] where feedback[0] is the top (y_oh -> last-primary-shape)
    and feedback[k] for k>=1 maps from the shape of primary_out[L-k] down to the
    shape of primary_out[L-k-1]. I.e. feedback[k] ultimately aligns with primary[L-1-k].
    """
    shapes = _primary_output_shapes(arch)  # [(C_1,H_1,W_1), ..., (C_L,H_L,W_L)]
    L = len(shapes)
    blocks: List[nn.Module] = []
    # Top: y_oh -> shape of primary[L-1]
    c_top, h_top, w_top = shapes[-1]
    blocks.append(FeedbackTop(num_classes, c_top, h_top, w_top).to(device))
    # Middle feedbacks: k in 1..L-1, mapping shape[L-k] -> shape[L-k-1]
    for k in range(1, L):
        in_c, _, _ = shapes[L - k]
        out_c, out_h, out_w = shapes[L - k - 1]
        blocks.append(FeedbackBlock(in_c, out_c, out_h, out_w).to(device))
    return blocks


def _ccl_local_loss(primary: torch.Tensor, feedback: torch.Tensor,
                    target: torch.Tensor, alignment_scale: float,
                    decorrelation_scale: float) -> torch.Tensor:
    """Official CNN CCL loss with per-sample L2 normalisation.

    The cross-path correlation is supervised by the same-class matrix. Each
    within-path correlation is simultaneously regularised toward the batch
    identity, preventing a collapsed representation from minimising the loss.
    """
    if primary.shape != feedback.shape:
        raise ValueError(
            "Counter-Current alignment needs matching activation shapes, got "
            f"{tuple(primary.shape)} and {tuple(feedback.shape)}."
        )
    p = F.normalize(primary.flatten(start_dim=1), dim=1, eps=1e-8)
    f = F.normalize(feedback.flatten(start_dim=1), dim=1, eps=1e-8)
    same_class = target[:, None].eq(target[None, :]).to(dtype=p.dtype)
    identity = torch.eye(p.shape[0], dtype=p.dtype, device=p.device)
    cross = F.mse_loss(p @ f.transpose(0, 1), same_class)
    primary_self = F.mse_loss(p @ p.transpose(0, 1), identity)
    feedback_self = F.mse_loss(f @ f.transpose(0, 1), identity)
    return alignment_scale * cross + decorrelation_scale * (
        primary_self + feedback_self
    )


# ================================================================
# Dual-path training
# ================================================================

def _train_dual(device: torch.device, epochs: int, lr: float, bs: int,
                seed: int, arch: str, lr_decay_epoch: int, lr_decay_factor: float,
                use_bn: bool, ccl_loss_scale_c: float,
                ccl_loss_scale_ssl: float) -> TrainResult:
    train_loader, test_loader = get_cifar10_loaders(batch_size=bs, augment=False)

    block_specs = make_conv_blocks(arch)
    L = num_blocks(arch)

    primary: List[nn.Module] = [ConvBlock(spec, use_bn=use_bn).to(device) for spec in block_specs]
    feedback: List[nn.Module] = _build_feedback_stack(arch, device, FAIR_NUM_CLASSES)
    # feedback[k] corresponds to primary[L-1-k]. Group them into per-pair Adam.
    classifier = PrimaryHead(arch=arch).to(device)

    opts: List[optim.Optimizer] = []
    for ell in range(L):
        fb_idx = L - 1 - ell  # index into feedback stack for this primary layer
        opts.append(optim.Adam(
            list(primary[ell].parameters()) + list(feedback[fb_idx].parameters()),
            lr=lr,
        ))
    clf_opt = optim.Adam(classifier.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _readout(x: torch.Tensor) -> torch.Tensor:
        """Evaluate the linear readout without changing module train/eval modes."""
        h = x
        for block in primary:
            h = block(h)
        return classifier(h)

    acc_curve: List[float] = []
    loss_curve: List[float] = []
    t0 = time.time()
    best_acc = 0.0

    for epoch in range(epochs):
        # Recipe: lr decay on all active optimizers (per-layer + classifier).
        apply_lr_decay([*opts, clf_opt], epoch, lr,
                       at_epoch=lr_decay_epoch, factor=lr_decay_factor)
        for b in primary:
            b.train()
        for b in feedback:
            b.train()
        classifier.train()

        running, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_oh = F.one_hot(y, FAIR_NUM_CLASSES).float()

            # Primary forward with inter-block detach (locality).
            primary_acts: List[torch.Tensor] = []
            h = x
            for b in primary:
                h_in = h.detach() if primary_acts else h  # first block takes raw x
                h = b(h_in)
                primary_acts.append(h)

            # Feedback forward (top-down) with analogous detach.
            feedback_acts: List[torch.Tensor] = []  # indexed 0..L-1 aligned to feedback list
            h = feedback[0](y_oh)
            feedback_acts.append(h)
            for k in range(1, L):
                h = feedback[k](h.detach())
                feedback_acts.append(h)

            # Per-layer normalised CCL loss; feedback[k] aligns with primary[L-1-k].
            layer_losses: List[torch.Tensor] = []
            for ell in range(L):
                fb_idx = L - 1 - ell
                layer_losses.append(_ccl_local_loss(
                    primary_acts[ell], feedback_acts[fb_idx], y,
                    ccl_loss_scale_c, ccl_loss_scale_ssl,
                ))

            for o in opts:
                o.zero_grad()
            total_local = sum(layer_losses)
            total_local.backward()
            for o in opts:
                o.step()

            # CE updates only the output readout, as prescribed by the
            # stop-gradient at the input of the final primary layer.
            logits = classifier(primary_acts[-1].detach())
            clf_loss = ce(logits, y)
            clf_opt.zero_grad()
            clf_loss.backward()
            clf_opt.step()

            tot = float(sum(l.item() for l in layer_losses)) + clf_loss.item()
            if not (tot == tot) or tot == float("inf"):
                return TrainResult(
                    algo="counter_current", status="failed", arch=arch,
                    error_msg="NaN/Inf loss in dual-path training",
                    hyperparams={
                        "lr": lr, "epochs": epochs, "batch_size": bs,
                        "optimizer": "Adam", "seed": seed, "arch": arch,
                        "lr_decay_epoch": lr_decay_epoch,
                        "lr_decay_factor": lr_decay_factor,
                        "ccl_loss_scale_c": ccl_loss_scale_c,
                        "ccl_loss_scale_ssl": ccl_loss_scale_ssl,
                        "use_bn": use_bn,
                        "bn_status": "added_to_convblock" if use_bn else "disabled",
                    },
                    timestamp=now_iso(), tag="dual_path_failed",
                )
            running += tot
            n += 1

        avg_loss = running / max(n, 1)
        loss_curve.append(avg_loss)
        for b in primary:
            b.eval()
        classifier.eval()
        acc = evaluate_classifier(_readout, test_loader, device)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [CC dual/{arch}] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  "
                  f"test_acc={acc:.4f}  best={best_acc:.4f}", flush=True)

    total_params = (sum(count_params(b) for b in primary) +
                    sum(count_params(b) for b in feedback) +
                    count_params(classifier))
    return TrainResult(
        algo="counter_current", status="ok", arch=arch,
        test_acc_final=acc_curve[-1] if acc_curve else None,
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=total_params,
        elapsed_s=time.time() - t0,
        hyperparams={
            "lr": lr, "epochs": epochs, "batch_size": bs, "optimizer": "Adam",
            "seed": seed,
            "local_loss": "normalized_class_alignment_plus_identity_decorrelation",
            "ccl_loss_scale_c": ccl_loss_scale_c,
            "ccl_loss_scale_ssl": ccl_loss_scale_ssl,
            "arch": arch,
            "n_layers": L,
            "primary_channels": block_channels(arch),
            "final_feat_channels": final_feat_channels(arch),
            "classifier": f"AvgPool({head_avgpool_size(arch)}) + "
                          f"Linear({final_fc_in_dim(arch)}, {FAIR_NUM_CLASSES}) "
                          "on stop-graded h_L_p",
            "lr_decay_epoch": lr_decay_epoch,
            "lr_decay_factor": lr_decay_factor,
            "use_bn": use_bn,
            "bn_status": "added_to_convblock" if use_bn else "disabled",
        },
        gpu=str(device), timestamp=now_iso(), tag="dual_path",
    )


# ================================================================
# Registered entry
# ================================================================

@register("counter_current")
def train(cfg: dict) -> TrainResult:
    seed = cfg.get("seed", 0)
    set_seed(seed)
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    bs = cfg.get("batch_size", 128)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Phase II.1 recipe knobs.
    lr_decay_epoch = cfg.get("lr_decay_epoch", 100)
    lr_decay_factor = cfg.get("lr_decay_factor", 0.1)
    use_bn = cfg.get("use_bn", True)
    # Native CCL CNN loss scales from the authors' published configuration.
    ccl_loss_scale_c = cfg.get("ccl_loss_scale_c", 1.0)
    ccl_loss_scale_ssl = cfg.get("ccl_loss_scale_ssl", 0.5)

    return _train_dual(device, epochs, lr, bs, seed, arch,
                       lr_decay_epoch, lr_decay_factor, use_bn,
                       ccl_loss_scale_c, ccl_loss_scale_ssl)
