"""Compare BP and strict-local FFA error-signal-kernel effective ranks."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve()
ROOT = HERE.parent
sys.path.insert(0, str(HERE.parent))

from algos.strict_local_ffa import StrictLocalFFAModel  # noqa: E402
from algos.vanilla_ffa import _apply_overlay_image, _random_wrong_labels  # noqa: E402
from bp_tricks import BPModel  # noqa: E402
from common import FAIR_NUM_CLASSES, get_cifar10_loaders, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffa-checkpoint", type=Path, default=None,
                        help="Override the strict-local FFA checkpoint for a single architecture.")
    parser.add_argument("--ffa-trunk-norm", choices=("none", "channel_ln"), default="none",
                        help="Trunk normalization used by the supplied FFA checkpoint.")
    parser.add_argument("--archs", nargs="+", default=["cnn3", "cnn6", "cnn9"])
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results" / "bp_strict_local_ffa_gamma_erank")
    return parser.parse_args()


def balanced_test_loader(samples_per_class: int, batch_size: int) -> DataLoader:
    _, test_loader = get_cifar10_loaders(batch_size=batch_size, augment=False)
    targets = list(test_loader.dataset.targets)
    chosen: List[int] = []
    counts = [0] * FAIR_NUM_CLASSES
    for index, target in enumerate(targets):
        if counts[target] < samples_per_class:
            chosen.append(index)
            counts[target] += 1
        if all(count == samples_per_class for count in counts):
            break
    if any(count != samples_per_class for count in counts):
        raise RuntimeError(f"Could not build balanced subset: counts={counts}")
    return DataLoader(Subset(test_loader.dataset, chosen), batch_size=batch_size,
                      shuffle=False, num_workers=2, pin_memory=True)


def erank_from_signal_chunks(chunks: Sequence[Sequence[torch.Tensor]], device: torch.device) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for layer, layer_chunks in enumerate(chunks, start=1):
        signals = torch.cat(list(layer_chunks), dim=0).to(device, non_blocking=True)
        sample_gram = signals @ signals.T
        trace = torch.diagonal(sample_gram).sum()
        denom = sample_gram.square().sum()
        value = 1.0 if denom <= 1e-12 else float((trace.square() / denom).item())
        rows.append({
            "layer": layer,
            "feature_dim": int(signals.shape[1]),
            "n_signals": int(signals.shape[0]),
            "gamma_erank_pr": value,
        })
        del signals, sample_gram
        torch.cuda.empty_cache()
    return rows


def bp_block_outputs(model: BPModel, x: torch.Tensor) -> tuple[List[torch.Tensor], List[int]]:
    features = list(model.backbone.features.children())
    specs = model.backbone.features
    del specs
    h = x
    outputs: List[torch.Tensor] = []
    boundaries: List[int] = []
    index = 0
    # Each architecture block is Conv -> BN -> ReLU -> optional pool.
    for block_index, spec in enumerate(model.backbone.features):
        del block_index, spec
        break
    # Recover block boundaries from modules. A max/avg pool closes a block; a
    # ReLU followed by a non-pool closes a block without pooling.
    for index, layer in enumerate(features):
        h = layer(h)
        next_layer = features[index + 1] if index + 1 < len(features) else None
        if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            outputs.append(h)
            boundaries.append(index + 1)
        elif isinstance(layer, nn.ReLU) and not isinstance(next_layer, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            outputs.append(h)
            boundaries.append(index + 1)
    return outputs, boundaries


def bp_logits_from_block(model: BPModel, h: torch.Tensor, boundary: int) -> torch.Tensor:
    features = list(model.backbone.features.children())
    z = h
    for layer in features[boundary:]:
        z = layer(z)
    z = model.backbone.avgpool(z).reshape(z.shape[0], -1)
    return model.head(model.backbone.fc(z))


def collect_bp(model: BPModel, loader: DataLoader, device: torch.device) -> List[Dict[str, float]]:
    model.eval()
    chunks: List[List[torch.Tensor]] | None = None
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        outputs, boundaries = bp_block_outputs(model, x)
        if chunks is None:
            chunks = [[] for _ in outputs]
        for index, (h, boundary) in enumerate(zip(outputs, boundaries)):
            logits = bp_logits_from_block(model, h, boundary)
            loss = F.cross_entropy(logits, y, reduction="sum")
            delta = torch.autograd.grad(loss, h, retain_graph=index + 1 < len(outputs))[0]
            chunks[index].append(delta.reshape(delta.shape[0], -1).detach().cpu())
    assert chunks is not None
    return erank_from_signal_chunks(chunks, device)


def collect_ffa(model: StrictLocalFFAModel, loader: DataLoader, device: torch.device) -> List[Dict[str, float]]:
    model.eval()
    chunks: List[List[torch.Tensor]] = [[] for _ in model.blocks]
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        h_pos = _apply_overlay_image(x, y)
        h_neg = _apply_overlay_image(x, _random_wrong_labels(y))
        for index, (block, head) in enumerate(zip(model.blocks, model.heads)):
            h_pos_out = block(h_pos.detach())
            h_neg_out = block(h_neg.detach())
            g_pos = head.goodness(head(h_pos_out))
            g_neg = head.goodness(head(h_neg_out))
            theta = ((g_pos.mean() + g_neg.mean()) / 2).detach()
            pos_loss = F.softplus(-(g_pos - theta)).sum()
            neg_loss = F.softplus(g_neg - theta).sum()
            delta_pos = torch.autograd.grad(pos_loss, h_pos_out, retain_graph=True)[0]
            delta_neg = torch.autograd.grad(neg_loss, h_neg_out)[0]
            chunks[index].append(torch.cat([
                delta_pos.reshape(delta_pos.shape[0], -1),
                delta_neg.reshape(delta_neg.shape[0], -1),
            ], dim=0).detach().cpu())
            h_pos, h_neg = h_pos_out.detach(), h_neg_out.detach()
    return erank_from_signal_chunks(chunks, device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loader = balanced_test_loader(args.samples_per_class, args.batch_size)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for arch in args.archs:
        bp_path = ROOT / "experiments" / "CIFAR10-CNN" / "output" / "recipe_checkpoint_diagnose" / "checkpoints" / "bp" / arch / "end.pt"
        if args.ffa_checkpoint is not None:
            if len(args.archs) != 1:
                raise ValueError("--ffa-checkpoint requires exactly one architecture.")
            ffa_path = args.ffa_checkpoint
        else:
            ffa_path = ROOT / "experiments" / "CIFAR10-CNN" / "output" / "strict_local_ffa_sigma_profile" / f"{arch}_end.pt"
        print(f"=== Gamma erank: {arch} on {device} ===", flush=True)
        bp = BPModel(arch=arch, use_bn=True).to(device).eval()
        bp.load_state_dict(torch.load(bp_path, map_location="cpu")["model_state_dict"], strict=True)
        ffa = StrictLocalFFAModel(arch=arch, lr=1e-3, hidden_dim=256,
                                  trunk_norm=args.ffa_trunk_norm).to(device).eval()
        ffa.load_state_dict(torch.load(ffa_path, map_location="cpu")["model_state_dict"], strict=True)
        bp_rows = collect_bp(bp, loader, device)
        ffa_rows = collect_ffa(ffa, loader, device)
        for method, rows, path, n_base in (
            ("bp", bp_rows, bp_path, args.samples_per_class * FAIR_NUM_CLASSES),
            ("strict_local_ffa", ffa_rows, ffa_path, args.samples_per_class * FAIR_NUM_CLASSES),
        ):
            payload = {
                "arch": arch,
                "method": method,
                "checkpoint": str(path),
                "device": str(device),
                "base_examples": n_base,
                "signals_per_base_example": 1 if method == "bp" else 2,
                "gamma": "Gamma^(ell) = mean_i delta_i delta_i^T",
                "delta": "BP: task CE derivative; FFA: current block local positive/negative goodness-loss derivative",
                "rows": rows,
                "elc_sample_sum": sum(row["gamma_erank_pr"] for row in rows),
            }
            (args.out_dir / f"{method}_{arch}_end.json").write_text(json.dumps(payload, indent=2))
        summary.append({
            "arch": arch,
            "bp_elc_sample_sum": sum(row["gamma_erank_pr"] for row in bp_rows),
            "ffa_elc_sample_sum": sum(row["gamma_erank_pr"] for row in ffa_rows),
            "bp_last_gamma_erank": bp_rows[-1]["gamma_erank_pr"],
            "ffa_last_gamma_erank": ffa_rows[-1]["gamma_erank_pr"],
        })
        del bp, ffa
        torch.cuda.empty_cache()
    (args.out_dir / "summary.json").write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "archs": args.archs,
        "samples_per_class": args.samples_per_class,
        "rows": summary,
    }, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
