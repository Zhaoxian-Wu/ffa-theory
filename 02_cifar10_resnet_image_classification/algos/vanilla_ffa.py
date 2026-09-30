"""Positive/negative goodness learning and label-enumeration inference.

The embedding channel is detached before the first stage, preserving the
experimental implementation: the embedding table is not updated by the loss.
Readout features use its mean channel, never the ground-truth test label.
"""
from typing import List, Tuple
import torch

from torch.nn import functional as F
from torch import optim
from models import LabelEmbedding, build_blocks
from .common import NativeTrainer, EpochMetrics, apply_lr_decay


def _gap_goodness(h: torch.Tensor) -> torch.Tensor:
    z = F.adaptive_avg_pool2d(h, 1).flatten(1)
    return z.pow(2).mean(dim=1)


def _random_wrong_labels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    delta = torch.randint(1, num_classes, y.shape, device=y.device)
    return (y + delta) % num_classes


class LabelEnumTrainer(NativeTrainer):
    algo = "label_enum"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.label_spatial = cfg.get("tiny_image_size", 32) if self.dataset == "tiny_imagenet" else 32
        self.label_emb = LabelEmbedding(num_classes=self.num_classes, spatial=self.label_spatial).to(self.device)
        self.blocks = [b.to(self.device) for b in build_blocks(self.arch, in_channels=4)]
        self.opts = []
        for i, block in enumerate(self.blocks):
            params = list(block.parameters())
            if i == 0:
                params += list(self.label_emb.parameters())
            self.opts.append(optim.Adam(params, lr=self.lr))

    def modules_for_mode(self, train: bool) -> None:
        self.label_emb.train(train)
        for block in self.blocks:
            block.train(train)

    def _positive_negative_inputs(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.label_emb(x, y), self.label_emb(x, _random_wrong_labels(y, self.num_classes))

    def _loss_i(self, h_pos_out: torch.Tensor, h_neg_out: torch.Tensor, idx: int) -> torch.Tensor:
        g_pos = _gap_goodness(h_pos_out)
        g_neg = _gap_goodness(h_neg_out)
        theta = ((g_pos.mean() + g_neg.mean()) / 2).detach()
        return F.softplus(-(g_pos - theta)).mean() + F.softplus(g_neg - theta).mean()

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
            batch_loss = 0.0
            for i, block in enumerate(self.blocks):
                h_pos_out = block(h_pos.detach())
                h_neg_out = block(h_neg.detach())
                loss_i = self._loss_i(h_pos_out, h_neg_out, i)
                self.opts[i].zero_grad()
                loss_i.backward()
                self.opts[i].step()
                batch_loss += float(loss_i.detach())
                h_pos, h_neg = h_pos_out.detach(), h_neg_out.detach()
            running += batch_loss / len(self.blocks)
            n_batches += 1
        return EpochMetrics(running / max(n_batches, 1))

    def _neutral_input(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        neutral = self.label_emb.emb.weight.mean(dim=0).view(1, 1, self.label_spatial, self.label_spatial)
        neutral = neutral.expand(bsz, -1, -1, -1)
        return torch.cat([x, neutral], dim=1)

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        self.modules_for_mode(False)
        feats = []
        h = self._neutral_input(x)
        for block in self.blocks:
            h = block(h)
            feats.append(h)
        return feats

    def native_logits(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        scores = []
        for c in range(self.num_classes):
            y_c = torch.full((bsz,), c, dtype=torch.long, device=self.device)
            h = self.label_emb(x, y_c)
            total_g = torch.zeros(bsz, device=self.device)
            for block in self.blocks:
                h = block(h)
                total_g += _gap_goodness(h)
            scores.append(total_g)
        return torch.stack(scores, dim=1)
