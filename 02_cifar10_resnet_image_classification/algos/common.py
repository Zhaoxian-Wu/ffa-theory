"""Shared interfaces, local heads, and learning-rate schedule."""
from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import nn
from torch.nn import functional as F

NUM_CLASSES = {"mnist": 10, "cifar10": 10, "cifar100": 100, "tiny_imagenet": 200}

@dataclass
class EpochMetrics:
    loss: float
    acc: Optional[float] = None


def apply_lr_decay(optimizers, epoch, lr_init, at_epoch=100, factor=0.1):
    if epoch == at_epoch:
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = lr_init * factor

class NativeTrainer:
    algo = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.arch = cfg.get("arch", "resnet18")
        self.device = torch.device(cfg.get("device", "cuda"))
        self.dataset = cfg.get("dataset", "cifar10")
        self.num_classes = NUM_CLASSES[self.dataset]
        self.lr = cfg.get("lr", 1e-3)
        self.ce = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    def train_epoch(self, loader, epoch: int) -> EpochMetrics:
        raise NotImplementedError

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def native_logits(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def modules_for_mode(self, train: bool) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def evaluate_native(self, loader) -> dict:
        self.modules_for_mode(False)
        correct = total = 0
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.native_logits(x)
            loss_sum += self.ce(logits, y).item() * y.shape[0]
            correct += (logits.argmax(1) == y).sum().item()
            total += y.shape[0]
        return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1), "n": total}


class LPredHead(nn.Module):
    """Conv(C,C,3) + ReLU + GAP + Linear(C, num_classes)."""
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(self.gap(torch.relu(self.conv(h))).flatten(1))


def _sim_loss(h: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
    """L_sim = ||S(GAP(h)) - S(Y)||_F^2 / B^2  (adjusted cosine similarity matching)."""
    feats = F.adaptive_avg_pool2d(h, 1).flatten(1)
    def _acs(m):
        m_c = m - m.mean(dim=1, keepdim=True)
        m_n = F.normalize(m_c, p=2, dim=1, eps=1e-8)
        return m_n @ m_n.t()
    b = h.shape[0]
    return ((_acs(feats) - _acs(y_onehot.float())) ** 2).sum() / (b * b)


class SFFAuxHead(nn.Module):
    """Conv(C,C,k=5) -> channel-LayerNorm -> ReLU -> GAP -> Linear(C, num_classes)."""
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.ln = nn.LayerNorm(channels)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.conv(h)
        z = z.permute(0, 2, 3, 1).contiguous()
        z = self.ln(z)
        z = z.permute(0, 3, 1, 2).contiguous()
        z = torch.relu(z)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z)


class DFHead(nn.Module):
    """GAP -> proj -> L2-norm -> tau * cosine(h, prototype_k)."""
    def __init__(self, channels: int, embed_dim: int = 64,
                 num_classes: int = 10, tau: float = 10.0):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels, embed_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.1)
        self.tau = tau

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        feat = F.normalize(self.proj(self.gap(h).flatten(1)), dim=1, eps=1e-8)
        proto = F.normalize(self.prototypes, dim=1, eps=1e-8)
        return self.tau * feat @ proto.t()
