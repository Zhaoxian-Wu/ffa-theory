"""Muon optimizer utilities for local CIFAR-10 training units.

The Muon update is applied to every trainable tensor with at least two
dimensions. Convolution kernels are viewed as matrices with the output
channel as the row dimension; the orthogonalized matrix update is reshaped
back to the original kernel shape. Scalar, bias, and normalization parameters
remain in a companion AdamW optimizer.
"""
from __future__ import annotations

from typing import Iterable, List

import torch


def zeropower_via_newtonschulz5(
    matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7,
) -> torch.Tensor:
    """Return a Newton--Schulz approximation to the matrix zeroth power."""
    if matrix.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape {tuple(matrix.shape)}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    work = matrix.float()
    work = work / (work.norm() + eps)
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.T
    for _ in range(steps):
        gram = work @ work.T
        work = a * work + (b * gram + c * (gram @ gram)) @ work
    if transposed:
        work = work.T
    return work.to(dtype=matrix.dtype)


class Muon(torch.optim.Optimizer):
    """Nesterov-momentum Muon for matrices and flattened convolution kernels."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("Muon does not support closures.")
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad_matrix = parameter.grad.reshape(parameter.shape[0], -1)
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad_matrix)
                buffer = state["momentum_buffer"]
                buffer.mul_(group["momentum"]).add_(grad_matrix)
                nesterov_grad = grad_matrix + group["momentum"] * buffer
                update = zeropower_via_newtonschulz5(
                    nesterov_grad, steps=group["ns_steps"],
                )
                scale = max(update.shape) ** 0.5
                parameter.add_(update.reshape_as(parameter), alpha=-group["lr"] * scale)


class LocalMuonOptimizer:
    """Optimizer pair for one strictly local FFA block and its goodness head."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        muon_lr: float,
        adam_lr: float,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ) -> None:
        matrix_params: List[torch.nn.Parameter] = []
        auxiliary_params: List[torch.nn.Parameter] = []
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            if parameter.ndim >= 2:
                matrix_params.append(parameter)
            else:
                auxiliary_params.append(parameter)
        self.optimizers: List[torch.optim.Optimizer] = []
        if matrix_params:
            self.optimizers.append(Muon(
                matrix_params, lr=muon_lr, momentum=momentum, ns_steps=ns_steps,
            ))
        if auxiliary_params:
            self.optimizers.append(torch.optim.AdamW(
                auxiliary_params, lr=adam_lr, weight_decay=0.0,
            ))

    @property
    def param_groups(self):
        """Expose groups so the common learning-rate scheduler can update both parts."""
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()
