"""
CIFAR10-CNN Benchmark: main entry point.

Runs one or more algorithms sequentially, writes TrainResult JSON per algo.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from pathlib import Path

# Ensure the package path is importable (so `from common import ...` works from algos/)
sys.path.insert(0, str(Path(__file__).parent))
# Ensure repo root is on path for src/* imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common import ALGO_REGISTRY, TrainResult, save_result, now_iso, FAIR_ARCHES  # noqa: E402


ALL_ALGOS = [
    "bp", "vanilla_ffa", "symba", "scff", "trifecta", "scodellaro",
    "sff", "forward_projection", "distance_forward",
    "belilovsky", "nokland_lpred", "nokland_lsim", "nokland_lpredsim",
    "auglocal", "gim", "dtp", "counter_current", "dll",
    "predictive_coding", "layer_collab",
]


def _autoimport_algos():
    """Import every algos/*.py so @register decorators fire."""
    algos_dir = Path(__file__).parent / "algos"
    for p in sorted(algos_dir.glob("*.py")):
        if p.stem.startswith("_") or p.stem == "nokland_base":
            continue
        mod = f"algos.{p.stem}"
        try:
            importlib.import_module(mod)
        except Exception as e:
            print(f"[WARN] import {mod} failed: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", type=str, required=True,
                    help="Comma-separated list, or 'all'.")
    ap.add_argument("--arch", type=str, default="cnn3", choices=list(FAIR_ARCHES),
                    help="Shared backbone architecture (cnn3 | cnn6).")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output directory for JSON results. Defaults to "
                         "results/CIFAR10-CNN/{arch}_seed{seed}/.")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--quick", action="store_true",
                    help="Short-run mode (epochs=3) for smoke testing.")
    ap.add_argument("--weight_decay", type=float, default=None,
                    help="Out-of-contract ablation hyperparameter; only some "
                         "algorithms respect it (currently: bp).")
    args = ap.parse_args()

    _autoimport_algos()

    if args.algos == "all":
        names = ALL_ALGOS
    else:
        names = [s.strip() for s in args.algos.split(",") if s.strip()]

    epochs = 3 if args.quick else args.epochs
    out_dir = args.out_dir or f"results/CIFAR10-CNN/{args.arch}_seed{args.seed}"

    for name in names:
        if name not in ALGO_REGISTRY:
            print(f"[SKIP] algo '{name}' not registered "
                  f"(available: {sorted(ALGO_REGISTRY.keys())})", file=sys.stderr)
            continue

        cfg = {
            "seed": args.seed,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "device": args.device,
            "out_dir": out_dir,
            "arch": args.arch,
        }
        if args.weight_decay is not None:
            cfg["weight_decay"] = args.weight_decay
        print(f"\n{'=' * 70}\n[{now_iso()}] Running: {name}  "
              f"(arch={args.arch}, epochs={epochs}, seed={args.seed}, device={args.device})\n"
              f"{'=' * 70}", flush=True)

        t0 = time.time()
        try:
            result: TrainResult = ALGO_REGISTRY[name](cfg)
        except Exception as e:
            traceback.print_exc()
            result = TrainResult(
                algo=name, status="failed", arch=args.arch,
                elapsed_s=time.time() - t0,
                error_msg=f"{type(e).__name__}: {e}",
                timestamp=now_iso(),
            )
        # Ensure arch is recorded even if the algo forgot to set it
        if result.arch is None:
            result.arch = args.arch
        if result.timestamp is None:
            result.timestamp = now_iso()
        if result.elapsed_s is None:
            result.elapsed_s = time.time() - t0

        path = save_result(result, out_dir)
        print(f"[{now_iso()}] {name} -> arch={result.arch}, status={result.status}, "
              f"test_acc_best={result.test_acc_best}, saved={path}", flush=True)


if __name__ == "__main__":
    main()
