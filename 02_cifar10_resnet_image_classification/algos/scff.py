"""SCFF adaptation: pair each label with a permuted image for negatives."""
from typing import Tuple
import torch


from .vanilla_ffa import LabelEnumTrainer


def _negative_permutation(bsz: int, device: torch.device) -> torch.Tensor:
    perm = torch.randperm(bsz, device=device)
    fixed = perm == torch.arange(bsz, device=device)
    if fixed.any() and bsz > 1:
        idx = fixed.nonzero(as_tuple=False).flatten()
        perm[idx] = perm[torch.roll(idx, 1, 0)]
    return perm


class SCFFTrainer(LabelEnumTrainer):
    algo = "scff"

    def _positive_negative_inputs(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        perm = _negative_permutation(x.shape[0], self.device)
        return self.label_emb(x, y), self.label_emb(x[perm], y)
