"""End-to-end cross-entropy with SGD and momentum."""
from typing import List
import torch


from torch import optim
from models import ResNetClassifier
from .common import NativeTrainer, EpochMetrics


class BPTrainer(NativeTrainer):
    algo = "bp"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model = ResNetClassifier(self.arch, self.num_classes).to(self.device)
        self.opt = optim.SGD(
            self.model.parameters(),
            lr=cfg.get("sgd_lr", 0.1),
            momentum=0.9,
            weight_decay=cfg.get("weight_decay", 5e-4),
            nesterov=True,
        )
        if cfg.get("bp_scheduler", "multistep") == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.get("epochs", 200))
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=[100, 150], gamma=0.1)

    def modules_for_mode(self, train: bool) -> None:
        self.model.train(train)

    def train_epoch(self, loader, epoch: int) -> EpochMetrics:
        self.modules_for_mode(True)
        loss_sum = 0.0
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.ce(logits, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_sum += loss.item() * y.shape[0]
            correct += (logits.detach().argmax(1) == y).sum().item()
            total += y.shape[0]
        self.scheduler.step()
        return EpochMetrics(loss_sum / max(total, 1), correct / max(total, 1))

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        self.modules_for_mode(False)
        feats = []
        h = x
        for block in self.model.blocks:
            h = block(h)
            feats.append(h)
        return feats

    def native_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
