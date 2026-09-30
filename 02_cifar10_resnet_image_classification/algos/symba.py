"""Symmetric softplus loss on the positive-negative goodness margin."""

import torch

from torch.nn import functional as F

from .vanilla_ffa import LabelEnumTrainer, _gap_goodness


class SymBaTrainer(LabelEnumTrainer):
    algo = "symba"

    def _loss_i(self, h_pos_out: torch.Tensor, h_neg_out: torch.Tensor, idx: int) -> torch.Tensor:
        alpha = self.cfg.get("alpha", 4.0)
        return F.softplus(-alpha * (_gap_goodness(h_pos_out) - _gap_goodness(h_neg_out))).mean()
