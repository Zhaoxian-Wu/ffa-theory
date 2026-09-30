"""Accumulate detached earlier-stage goodness as a collaboration offset."""

import torch

from torch.nn import functional as F

from .common import EpochMetrics, apply_lr_decay
from .vanilla_ffa import LabelEnumTrainer, _gap_goodness


class LayerCollabTrainer(LabelEnumTrainer):
    algo = "layer_collab"

    def train_epoch(self, loader, epoch: int) -> EpochMetrics:
        apply_lr_decay(self.opts, epoch, self.lr,
                       at_epoch=self.cfg.get("lr_decay_epoch", 100),
                       factor=self.cfg.get("lr_decay_factor", 0.1))
        self.modules_for_mode(True)
        running = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            h_pos, h_neg = self._positive_negative_inputs(x, y)
            gamma_pos = torch.zeros(x.shape[0], device=self.device)
            gamma_neg = torch.zeros(x.shape[0], device=self.device)
            batch_loss = 0.0
            for i, block in enumerate(self.blocks):
                h_pos_out = block(h_pos.detach())
                h_neg_out = block(h_neg.detach())
                g_pos = _gap_goodness(h_pos_out)
                g_neg = _gap_goodness(h_neg_out)
                score_pos = g_pos + gamma_pos
                score_neg = g_neg + gamma_neg
                theta = ((score_pos.mean() + score_neg.mean()) / 2).detach()
                loss_i = F.softplus(-(score_pos - theta)).mean() + F.softplus(score_neg - theta).mean()
                self.opts[i].zero_grad()
                loss_i.backward()
                self.opts[i].step()
                batch_loss += float(loss_i.detach())
                h_pos, h_neg = h_pos_out.detach(), h_neg_out.detach()
                gamma_pos = gamma_pos + g_pos.detach()
                gamma_neg = gamma_neg + g_neg.detach()
            running += batch_loss / len(self.blocks)
            n_batches += 1
        return EpochMetrics(running / max(n_batches, 1))

    def native_logits(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        scores = []
        for c in range(self.num_classes):
            y_c = torch.full((bsz,), c, dtype=torch.long, device=self.device)
            h = self.label_emb(x, y_c)
            gamma = torch.zeros(bsz, device=self.device)
            total_score = torch.zeros(bsz, device=self.device)
            for block in self.blocks:
                h = block(h)
                g = _gap_goodness(h)
                total_score += g + gamma
                gamma = gamma + g
            scores.append(total_score)
        return torch.stack(scores, dim=1)
