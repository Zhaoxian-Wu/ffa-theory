"""
BP: Standard end-to-end backpropagation with cross-entropy.

Reference: LeCun et al., 1989; modern baseline.
Architecture: shared backbone (CNN3Backbone or CNN6Backbone, via cfg["arch"])
              + Linear classifier head (out_dim -> 10).
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.optim as optim

from common import (
    FAIR_DEFAULT_ARCH, FAIR_LR, FAIR_NUM_CLASSES, TrainResult,
    count_params, evaluate_classifier, get_cifar10_loaders,
    make_backbone, now_iso, register, set_seed,
)


class BPModel(nn.Module):
    def __init__(self, arch: str = FAIR_DEFAULT_ARCH,
                 num_classes: int = FAIR_NUM_CLASSES):
        super().__init__()
        self.backbone = make_backbone(arch)
        self.head = nn.Linear(self.backbone.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@register("bp")
def train(cfg: dict) -> TrainResult:
    set_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cuda"))
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", FAIR_LR)
    arch = cfg.get("arch", FAIR_DEFAULT_ARCH)
    # Default weight_decay=0 preserves the fair-comparison contract. Callers
    # may override via cfg["weight_decay"] for BP-specific regularisation
    # ablations (out-of-contract; record in hyperparams).
    weight_decay = cfg.get("weight_decay", 0.0)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=cfg.get("batch_size", 128), augment=False,
    )
    model = BPModel(arch=arch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    acc_curve, loss_curve = [], []
    t0 = time.time()
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        avg_loss = running / max(n_batches, 1)
        loss_curve.append(avg_loss)

        model.eval()
        acc = evaluate_classifier(lambda x: model(x), test_loader, device)
        acc_curve.append(acc)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [BP] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  test_acc={acc:.4f}  best={best_acc:.4f}",
                  flush=True)

    return TrainResult(
        algo="bp", status="ok", arch=arch,
        test_acc_final=acc_curve[-1],
        test_acc_best=best_acc,
        test_acc_curve=acc_curve,
        train_loss_curve=loss_curve,
        n_params=count_params(model),
        elapsed_s=time.time() - t0,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": cfg.get("batch_size", 128),
                     "optimizer": "Adam", "arch": arch, "weight_decay": weight_decay,
                     "seed": cfg.get("seed", 0)},
        gpu=str(device), timestamp=now_iso(),
    )
