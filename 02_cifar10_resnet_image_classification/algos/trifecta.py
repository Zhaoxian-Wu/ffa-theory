"""Symmetric goodness loss with alternating odd/even stage updates."""

import torch


from .common import EpochMetrics, apply_lr_decay
from .symba import SymBaTrainer


class TrifectaTrainer(SymBaTrainer):
    algo = "trifecta"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.iteration = 0
        self.olu = cfg.get("olu", True)

    def train_epoch(self, loader, epoch: int) -> EpochMetrics:
        apply_lr_decay(self.opts, epoch, self.lr,
                       at_epoch=self.cfg.get("lr_decay_epoch", 100),
                       factor=self.cfg.get("lr_decay_factor", 0.1))
        self.modules_for_mode(True)
        running = 0.0
        n_batches = 0
        n_blocks = len(self.blocks)
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            h_pos, h_neg = self._positive_negative_inputs(x, y)
            active = set(range(self.iteration % 2, n_blocks, 2)) if self.olu else set(range(n_blocks))
            batch_loss = 0.0
            for i, block in enumerate(self.blocks):
                if i in active:
                    h_pos_out = block(h_pos.detach())
                    h_neg_out = block(h_neg.detach())
                    loss_i = self._loss_i(h_pos_out, h_neg_out, i)
                    self.opts[i].zero_grad()
                    loss_i.backward()
                    self.opts[i].step()
                    batch_loss += float(loss_i.detach())
                    h_pos, h_neg = h_pos_out.detach(), h_neg_out.detach()
                else:
                    with torch.no_grad():
                        h_pos = block(h_pos.detach())
                        h_neg = block(h_neg.detach())
            running += batch_loss / max(len(active), 1)
            n_batches += 1
            self.iteration += 1
        return EpochMetrics(running / max(n_batches, 1))
