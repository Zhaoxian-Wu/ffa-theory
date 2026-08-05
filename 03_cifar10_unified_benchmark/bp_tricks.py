"""Out-of-contract BP ablation: BatchNorm + cosine-like LR decay at epoch 100.

Usage:
    python bp_tricks.py --arch cnn3 --trick ab --seed 0 --epochs 200

Tricks:
    a   = LR decay to lr/10 after epoch 100
    b   = BatchNorm2d before each ReLU
    ab  = both
    none = plain BP (sanity check; should match the contract BP number)

Outputs:
    results/CIFAR10-CNN/bp_tricks/{arch}_trick{trick}_seed{seed}/bp_tricks.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    FAIR_DEFAULT_ARCH, FAIR_FEAT_DIM, FAIR_NUM_CLASSES,
    evaluate_classifier, get_cifar10_loaders, head_avgpool_size,
    make_conv_blocks, now_iso, set_seed,
)


# --- Backbone with optional BatchNorm ---

class BPBackbone(nn.Module):
    """Shared backbone using common.make_conv_blocks but with optional BN.

    Architecture (identical to CNN3/6/9 Backbone when use_bn=False):
        for each block spec:  Conv(in_ch, out_ch) -> [BN(out_ch)] -> ReLU -> [Pool]
    Final head: AdaptiveAvgPool(head_avgpool_size) -> Flatten -> Linear(fc_in, out_dim).
    """

    def __init__(self, arch: str, use_bn: bool, out_dim: int = FAIR_FEAT_DIM):
        super().__init__()
        specs = make_conv_blocks(arch)
        layers: List[nn.Module] = []
        for spec in specs:
            layers.append(nn.Conv2d(spec["in_ch"], spec["out_ch"],
                                    kernel_size=spec["kernel"],
                                    padding=spec["padding"]))
            if use_bn:
                layers.append(nn.BatchNorm2d(spec["out_ch"]))
            layers.append(nn.ReLU(inplace=True))
            if spec["pool"] == "max":
                layers.append(nn.MaxPool2d(2))
            elif spec["pool"] == "avg4":
                layers.append(nn.AdaptiveAvgPool2d(4))
            # "none": no pool layer
        self.features = nn.Sequential(*layers)
        s = head_avgpool_size(arch)
        self.avgpool = nn.AdaptiveAvgPool2d(s)
        final_ch = specs[-1]["out_ch"]
        self.fc = nn.Linear(final_ch * s * s, out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.avgpool(z).reshape(z.shape[0], -1)
        return self.fc(z)


class BPModel(nn.Module):
    def __init__(self, arch: str, use_bn: bool,
                 num_classes: int = FAIR_NUM_CLASSES):
        super().__init__()
        self.backbone = BPBackbone(arch, use_bn=use_bn)
        self.head = nn.Linear(self.backbone.out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


@dataclass
class TrickResult:
    algo: str = "bp_tricks"
    arch: str = ""
    trick: str = ""
    status: str = "ok"
    test_acc_final: Optional[float] = None
    test_acc_best: Optional[float] = None
    test_acc_curve: List[float] = field(default_factory=list)
    train_loss_curve: List[float] = field(default_factory=list)
    n_params: Optional[int] = None
    elapsed_s: Optional[float] = None
    hyperparams: dict = field(default_factory=dict)
    gpu: Optional[str] = None
    timestamp: Optional[str] = None

    def to_dict(self): return asdict(self)


def train(arch: str, trick: str, seed: int, epochs: int, lr: float,
          batch_size: int, device: torch.device, out_dir: str) -> TrickResult:
    assert trick in ("none", "a", "b", "ab")
    use_bn = trick in ("b", "ab")
    lr_decay = trick in ("a", "ab")

    set_seed(seed)
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, augment=False)
    model = BPModel(arch=arch, use_bn=use_bn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[bp_tricks] arch={arch} trick={trick} use_bn={use_bn} lr_decay={lr_decay} "
          f"n_params={n_params:,}", flush=True)

    acc_curve, loss_curve = [], []
    best_acc = 0.0
    t0 = time.time()
    for epoch in range(epochs):
        # Trick (a): decay learning rate by 10x after epoch 100.
        if lr_decay and epoch == 100:
            for g in optimizer.param_groups:
                g["lr"] = lr / 10.0
            print(f"  [bp_tricks] epoch {epoch + 1}: lr -> {lr / 10.0}", flush=True)

        model.train()
        running, n_batches = 0.0, 0
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
            print(f"  [BP/{arch}/trick={trick}] epoch {epoch + 1}/{epochs}  "
                  f"loss={avg_loss:.4f}  test_acc={acc:.4f}  best={best_acc:.4f}",
                  flush=True)

    result = TrickResult(
        arch=arch, trick=trick, status="ok",
        test_acc_final=acc_curve[-1], test_acc_best=best_acc,
        test_acc_curve=acc_curve, train_loss_curve=loss_curve,
        n_params=n_params, elapsed_s=time.time() - t0,
        hyperparams={"lr": lr, "epochs": epochs, "batch_size": batch_size,
                     "optimizer": "Adam", "arch": arch, "trick": trick,
                     "use_bn": use_bn, "lr_decay_at_epoch_100": lr_decay,
                     "lr_decay_factor": 10 if lr_decay else None,
                     "seed": seed},
        gpu=str(device), timestamp=now_iso(),
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "bp_tricks.json"
    with open(p, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"[bp_tricks] saved {p}  best={best_acc:.4f}  final={acc_curve[-1]:.4f}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["cnn3", "cnn6", "cnn9"])
    ap.add_argument("--trick", required=True, choices=["none", "a", "b", "ab"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (
        f"results/CIFAR10-CNN/bp_tricks/{args.arch}_trick{args.trick}_seed{args.seed}"
    )
    device = torch.device(args.device)
    train(args.arch, args.trick, args.seed, args.epochs, args.lr,
          args.batch_size, device, out_dir)


if __name__ == "__main__":
    main()
