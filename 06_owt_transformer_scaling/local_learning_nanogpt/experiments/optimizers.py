"""Optimizer registry and adapter bundles."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from local_learning_nanogpt.core.muon_optimizer import Muon, make_muon_optimizer
from local_learning_nanogpt.experiments.specs import OptimizerSpec


OPTIMIZER_REGISTRY: dict[str, OptimizerSpec] = {
    "adam": OptimizerSpec("adam", "AdamW baseline optimizer"),
    "muon": OptimizerSpec("muon", "Muon for 2D weights plus AdamW for the remainder"),
    "muon_paper": OptimizerSpec(
        "muon_paper",
        "Muon with paper-recommended 0.2 shape scaling and decoupled weight decay",
    ),
}


def list_optimizers() -> list[str]:
    return list(OPTIMIZER_REGISTRY)


def get_optimizer_spec(name: str) -> OptimizerSpec:
    if name not in OPTIMIZER_REGISTRY:
        raise KeyError(f"Unknown optimizer: {name}")
    return OPTIMIZER_REGISTRY[name]


@dataclass
class BPAdamBundle:
    optimizer: torch.optim.Optimizer

    def set_lr(self, new_lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def zero_grad_bp(self) -> None:
        self.optimizer.zero_grad()

    def step_bp(self, model: torch.nn.Module) -> None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()


@dataclass
class BPMuonBundle:
    muon_optimizer: torch.optim.Optimizer
    adam_optimizer: torch.optim.Optimizer

    def set_lr(self, new_lr: float) -> None:
        for optimizer in (self.muon_optimizer, self.adam_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = new_lr

    def zero_grad_bp(self) -> None:
        self.muon_optimizer.zero_grad()
        self.adam_optimizer.zero_grad()

    def step_bp(self, model: torch.nn.Module) -> None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.muon_optimizer.step()
        self.adam_optimizer.step()


@dataclass
class LocalAdamBundle:
    block_optimizers: list[torch.optim.Optimizer]
    emb_optimizer: torch.optim.Optimizer
    emb_params: list[torch.nn.Parameter]

    def set_lr(self, new_lr: float) -> None:
        for optimizer in self.block_optimizers:
            for group in optimizer.param_groups:
                group["lr"] = new_lr
        for group in self.emb_optimizer.param_groups:
            group["lr"] = new_lr

    def zero_grad_embeddings(self) -> None:
        self.emb_optimizer.zero_grad()

    def zero_grad_unit(self, index: int) -> None:
        self.block_optimizers[index].zero_grad()

    def step_unit(self, index: int) -> None:
        params = [param for group in self.block_optimizers[index].param_groups for param in group["params"]]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.block_optimizers[index].step()

    def step_embeddings(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.emb_params, 1.0)
        self.emb_optimizer.step()


@dataclass
class LocalMuonBundle:
    block_muon_optimizers: list[Muon | None]
    block_adam_optimizers: list[torch.optim.Optimizer | None]
    block_muon_params: list[list[torch.nn.Parameter]]
    block_adam_params: list[list[torch.nn.Parameter]]
    emb_optimizer: torch.optim.Optimizer
    emb_params: list[torch.nn.Parameter]

    def set_lr(self, new_lr: float) -> None:
        for optimizer in self.block_muon_optimizers:
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                group["lr"] = new_lr
        for optimizer in self.block_adam_optimizers:
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                group["lr"] = new_lr
        for group in self.emb_optimizer.param_groups:
            group["lr"] = new_lr

    def zero_grad_embeddings(self) -> None:
        self.emb_optimizer.zero_grad()

    def zero_grad_unit(self, index: int) -> None:
        muon_optimizer = self.block_muon_optimizers[index]
        adam_optimizer = self.block_adam_optimizers[index]
        if muon_optimizer is not None:
            muon_optimizer.zero_grad()
        if adam_optimizer is not None:
            adam_optimizer.zero_grad()

    def step_unit(self, index: int) -> None:
        if self.block_muon_params[index]:
            torch.nn.utils.clip_grad_norm_(self.block_muon_params[index], 1.0)
        if self.block_adam_params[index]:
            torch.nn.utils.clip_grad_norm_(self.block_adam_params[index], 1.0)
        muon_optimizer = self.block_muon_optimizers[index]
        adam_optimizer = self.block_adam_optimizers[index]
        if muon_optimizer is not None:
            muon_optimizer.step()
        if adam_optimizer is not None:
            adam_optimizer.step()

    def step_embeddings(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.emb_params, 1.0)
        self.emb_optimizer.step()


def build_optimizer_bundle(
    *,
    model: torch.nn.Module,
    algorithm_name: str,
    optimizer_name: str,
    lr: float,
) -> BPAdamBundle | BPMuonBundle | LocalAdamBundle | LocalMuonBundle:
    if algorithm_name == "bp":
        if optimizer_name == "adam":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
            return BPAdamBundle(optimizer=optimizer)
        use_paper_muon = optimizer_name == "muon_paper"
        muon_optimizer, adam_optimizer = make_muon_optimizer(
            model,
            muon_lr=lr,
            adam_lr=lr,
            momentum=0.95,
            muon_scaling_factor=0.2 if use_paper_muon else 1.0,
            muon_weight_decay=0.1 if use_paper_muon else 0.0,
        )
        return BPMuonBundle(muon_optimizer=muon_optimizer, adam_optimizer=adam_optimizer)

    emb_params = list(model.wte.parameters()) + list(model.wpe.parameters())
    emb_optimizer = torch.optim.AdamW(emb_params, lr=lr, weight_decay=0.01)

    if optimizer_name == "adam":
        block_optimizers = [
            torch.optim.AdamW(block.parameters(), lr=lr, weight_decay=0.1)
            for block in model.blocks
        ]
        return LocalAdamBundle(
            block_optimizers=block_optimizers,
            emb_optimizer=emb_optimizer,
            emb_params=emb_params,
        )

    if algorithm_name != "lce":
        raise ValueError(f"Optimizer {optimizer_name} is only supported for bp or lce, got {algorithm_name}.")

    use_paper_muon = optimizer_name == "muon_paper"
    block_muon_optimizers: list[Muon | None] = []
    block_adam_optimizers: list[torch.optim.Optimizer | None] = []
    block_muon_params: list[list[torch.nn.Parameter]] = []
    block_adam_params: list[list[torch.nn.Parameter]] = []
    for block in model.blocks:
        muon_params = [param for _, param in block.named_parameters() if param.ndim == 2 and param.requires_grad]
        adam_params = [param for _, param in block.named_parameters() if param.ndim != 2 and param.requires_grad]
        block_muon_params.append(muon_params)
        block_adam_params.append(adam_params)
        block_muon_optimizers.append(
            Muon(
                muon_params,
                lr=lr,
                momentum=0.95,
                scaling_factor=0.2 if use_paper_muon else 1.0,
                weight_decay=0.1 if use_paper_muon else 0.0,
            )
            if muon_params
            else None
        )
        block_adam_optimizers.append(torch.optim.AdamW(adam_params, lr=lr, weight_decay=0.1) if adam_params else None)

    return LocalMuonBundle(
        block_muon_optimizers=block_muon_optimizers,
        block_adam_optimizers=block_adam_optimizers,
        block_muon_params=block_muon_params,
        block_adam_params=block_adam_params,
        emb_optimizer=emb_optimizer,
        emb_params=emb_params,
    )
