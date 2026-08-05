"""
Muon Optimizer — Newton-Schulz Orthogonalized Gradient Descent

Based on Keller Jordan's implementation. Core idea:
  For a 2D weight matrix W ∈ ℝ^{m×n}, instead of Adam's element-wise scaling,
  orthogonalize the Nesterov momentum G (approximating G / ||G||_op) so that the
  update matrix satisfies approximate orthogonality (row/column normalization).

Theoretical background (connection to FFA):
  - Muon's orthogonalization is equivalent to projected gradient descent on the spectral norm ball
  - This makes each layer's update direction "maximally exploit the parameter's degrees of freedom"
  - Hypothesis for FFA: locally orthogonalized gradients may align more closely with global BP gradient directions,
    thereby reducing the residual gap C₁ ≈ 1.35×

Usage:
  from muon_optimizer import Muon, make_muon_optimizer
  optimizer = make_muon_optimizer(model, lr=0.01)
"""

import torch
import torch.nn as nn


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Computes X such that X ≈ G (G^T G)^{-1/2}, i.e., X has orthonormal rows
    (or columns if m > n). Iteration:
        X ← a*X + b*(X X^T) X + c*(X X^T)^2 X
    with (a, b, c) chosen for fast convergence in [0, 1.5] spectral norm range.

    Reference: Keller Jordan, "Muon: An optimizer for hidden layers in neural networks" (2024)
    """
    assert len(G.shape) == 2, f"Expected 2D tensor, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750,  2.0315)

    X = G.to(torch.float32)
    # Normalize to ensure spectral norm ≈ 1 (required for NS convergence)
    X = X / (X.norm() + eps)

    # If m > n, work on transpose (cheaper)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum + Nesterov + Orthogonalization Update

    Algorithm:
        1. Nesterov momentum update: G_t = β G_{t-1} + ∇L_t
        2. Orthogonalize: Ĝ_t = NS(G_t)  (Newton-Schulz)
        3. Update: W ← W - lr * scale * Ĝ_t, with optional decoupled weight decay.

    Applied to 2D weight matrices only. Use AdamW for embeddings/biases/LN.

    Args:
        params: 2D weight tensors only
        lr: learning rate
        momentum: Nesterov momentum coefficient (default 0.95)
        ns_steps: Newton-Schulz iterations (default 5, more = more accurate)
        scaling_factor: multiplicative factor applied to sqrt(max(m, n))
        weight_decay: decoupled weight decay for Muon-managed parameters
    """
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.95,
        ns_steps: int = 5,
        scaling_factor: float = 1.0,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            scaling_factor=scaling_factor,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            scaling_factor = group['scaling_factor']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                if weight_decay:
                    p.mul_(1.0 - lr * weight_decay)

                # Only apply Muon to 2D weight matrices
                if g.ndim != 2:
                    # Fall back to SGD with momentum for non-2D
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(p)
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    p.add_(buf, alpha=-lr)
                    continue

                # Nesterov momentum
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = torch.zeros_like(g)
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                # Nesterov "lookahead" gradient
                g_nesterov = g + momentum * buf

                # Newton-Schulz orthogonalization
                g_orth = zeropower_via_newtonschulz5(g_nesterov, steps=ns_steps)

                # Scale factor: preserves similar magnitude to Adam
                # sqrt(max(m,n)) compensates for NS output being O(1/sqrt(n))
                m, n = g.shape
                scale = scaling_factor * max(m, n) ** 0.5

                p.add_(g_orth, alpha=-lr * scale)


def make_muon_optimizer(model: nn.Module, muon_lr: float = 0.01, adam_lr: float = 6e-4,
                        momentum: float = 0.95, ns_steps: int = 5,
                        adam_weight_decay: float = 0.1,
                        muon_scaling_factor: float = 1.0,
                        muon_weight_decay: float = 0.0):
    """
    Create optimizer group:
    - Muon for all 2D weight matrices (Linear.weight, etc.)
    - AdamW for embeddings, biases, LayerNorm parameters

    This follows the standard Muon usage pattern.
    """
    muon_params = []
    adam_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and 'embedding' not in name and 'wte' not in name and 'wpe' not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    print(f"Muon params: {sum(p.numel() for p in muon_params)/1e6:.1f}M  "
          f"(2D weight matrices)")
    print(f"AdamW params: {sum(p.numel() for p in adam_params)/1e6:.1f}M  "
          f"(embeddings, biases, LN)")

    muon_opt = Muon(
        muon_params,
        lr=muon_lr,
        momentum=momentum,
        ns_steps=ns_steps,
        scaling_factor=muon_scaling_factor,
        weight_decay=muon_weight_decay,
    )
    adam_opt = torch.optim.AdamW(adam_params, lr=adam_lr, weight_decay=adam_weight_decay)

    return muon_opt, adam_opt


def make_muon_optimizer_ffa(model: nn.Module, block_idx: int,
                             muon_lr: float = 0.01, adam_lr: float = 6e-4,
                             momentum: float = 0.95):
    """
    Create optimizer for a single FFA block:
    - Muon for 2D weight matrices within the block
    - AdamW for biases/LN within the block
    Returns a single combined step function.
    """
    block = model.blocks[block_idx]
    muon_params = []
    adam_params = []

    for name, p in block.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            muon_params.append(p)
        else:
            adam_params.append(p)

    opts = []
    if muon_params:
        opts.append(Muon(muon_params, lr=muon_lr, momentum=momentum))
    if adam_params:
        opts.append(torch.optim.AdamW(adam_params, lr=adam_lr))

    return opts
