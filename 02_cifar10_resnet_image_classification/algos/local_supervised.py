"""Stage-local supervised methods: Nøkland, SFF, and Distance-Forward.

The paper's LCE row uses Nøkland's 0.99 CE + 0.01 similarity loss.
Each of the five stages has its own optimizer; gradients stop at boundaries.
"""
from typing import List, Type
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from models import build_blocks, STAGE_CHANNELS
from .common import (NativeTrainer, EpochMetrics, apply_lr_decay,
                     LPredHead, _sim_loss, SFFAuxHead, DFHead)


class BlockHeadTrainer(NativeTrainer):
    head_cls: Type[nn.Module] = LPredHead
    algo = "block_head"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.blocks = [b.to(self.device) for b in build_blocks(self.arch, in_channels=3)]
        self.heads = [self._make_head(ch).to(self.device) for ch in STAGE_CHANNELS]
        self.opts = [
            optim.Adam(list(b.parameters()) + list(h.parameters()), lr=self.lr)
            for b, h in zip(self.blocks, self.heads)
        ]

    def _make_head(self, channels: int) -> nn.Module:
        return self.head_cls(channels, self.num_classes)

    def modules_for_mode(self, train: bool) -> None:
        for module in [*self.blocks, *self.heads]:
            module.train(train)

    def _loss_i(self, idx: int, h_out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ce(self.heads[idx](h_out), y)

    def train_epoch(self, loader, epoch: int) -> EpochMetrics:
        apply_lr_decay(self.opts, epoch, self.lr,
                       at_epoch=self.cfg.get("lr_decay_epoch", 100),
                       factor=self.cfg.get("lr_decay_factor", 0.1))
        self.modules_for_mode(True)
        running = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            h_in = x
            batch_loss = 0.0
            for i, block in enumerate(self.blocks):
                h_out = block(h_in.detach())
                loss_i = self._loss_i(i, h_out, y)
                self.opts[i].zero_grad()
                loss_i.backward()
                self.opts[i].step()
                batch_loss += float(loss_i.detach())
                h_in = h_out.detach()
            running += batch_loss / len(self.blocks)
            n_batches += 1
        return EpochMetrics(running / max(n_batches, 1))

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        self.modules_for_mode(False)
        feats = []
        h = x
        for block in self.blocks:
            h = block(h)
            feats.append(h)
        return feats

    def native_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        return self.heads[-1](h)


class NoklandTrainer(BlockHeadTrainer):
    algo = "nokland_lpredsim"

    def _loss_i(self, idx: int, h_out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        beta = self.cfg.get("beta", 0.99)
        y_oh = F.one_hot(y, self.num_classes).float()
        return beta * self.ce(self.heads[idx](h_out), y) + (1.0 - beta) * _sim_loss(h_out, y_oh)


class SFFTrainer(BlockHeadTrainer):
    algo = "sff"

    def _make_head(self, channels: int) -> nn.Module:
        return SFFAuxHead(channels, self.num_classes)

    def native_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        logits = []
        for block, head in zip(self.blocks, self.heads):
            h = block(h)
            logits.append(head(h))
        return torch.stack(logits, 0).mean(0)


class DistanceForwardTrainer(BlockHeadTrainer):
    algo = "distance_forward"

    def _make_head(self, channels: int) -> nn.Module:
        return DFHead(
            channels,
            embed_dim=self.cfg.get("embed_dim", 64),
            num_classes=self.num_classes,
            tau=self.cfg.get("tau", 10.0),
        )
