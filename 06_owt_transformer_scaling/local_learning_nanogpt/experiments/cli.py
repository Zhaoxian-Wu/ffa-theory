"""Public CLI for the parameterized Local_Learning_Nanogpt experiment framework."""

from __future__ import annotations

import argparse

from local_learning_nanogpt.experiments.algorithms import list_algorithms
from local_learning_nanogpt.experiments.datasets import list_datasets
from local_learning_nanogpt.experiments.depth_scan import run_depth_scan
from local_learning_nanogpt.experiments.optimizers import list_optimizers
from local_learning_nanogpt.experiments.profiles import list_profiles
from local_learning_nanogpt.experiments.robustness import run_robustness
from local_learning_nanogpt.experiments.scaling import run_scaling
from local_learning_nanogpt.experiments.scales import list_scales


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m local_learning_nanogpt.experiments.cli",
        description="Unified CLI for Local_Learning_Nanogpt experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scaling_parser = subparsers.add_parser("scaling", help="Run standard scaling-law experiments.")
    scaling_parser.add_argument("--dataset", choices=list_datasets(), required=True)
    scaling_parser.add_argument("--profile", choices=list_profiles(), required=True)
    scaling_parser.add_argument("--algorithms", nargs="+", choices=list_algorithms(), required=True)
    scaling_parser.add_argument("--optimizers", nargs="+", choices=list_optimizers(), required=True)
    scaling_parser.add_argument("--scales", nargs="+", choices=list_scales(), required=True)
    scaling_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    scaling_parser.add_argument("--gpu", type=int, default=0)
    scaling_parser.add_argument("--batch-size", type=int, default=None)
    scaling_parser.add_argument("--block-size", type=int, default=None)
    scaling_parser.add_argument("--lr", type=float, default=None)
    scaling_parser.add_argument("--min-lr", type=float, default=None)
    scaling_parser.add_argument("--n-neg", type=int, default=None)
    scaling_parser.add_argument("--lce-rank", type=int, default=64)
    scaling_parser.add_argument("--max-iters", type=int, default=None)
    scaling_parser.add_argument("--token-multiplier", type=float, default=None)
    scaling_parser.add_argument("--probe-iters", type=int, default=None)
    scaling_parser.add_argument("--eval-interval", type=int, default=None)
    scaling_parser.add_argument("--output-name", type=str, default=None)

    depth_parser = subparsers.add_parser("depth-scan", help="Run the strict OWT depth scan.")
    depth_parser.add_argument("--dataset", choices=["owt_small"], default="owt_small")
    depth_parser.add_argument("--profile", choices=["quick"], default="quick")
    depth_parser.add_argument("--algorithm", choices=["lce"], default="lce")
    depth_parser.add_argument("--optimizer", choices=["adam"], default="adam")
    depth_parser.add_argument("--depths", nargs="+", type=int, required=True)
    depth_parser.add_argument("--width", type=int, default=512)
    depth_parser.add_argument("--heads", type=int, default=8)
    depth_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    depth_parser.add_argument("--gpu", type=int, default=0)
    depth_parser.add_argument("--max-iters", type=int, default=5000)
    depth_parser.add_argument("--batch-size", type=int, default=32)
    depth_parser.add_argument("--block-size", type=int, default=256)
    depth_parser.add_argument("--lr", type=float, default=6e-4)
    depth_parser.add_argument("--min-lr", type=float, default=6e-5)
    depth_parser.add_argument("--warmup-iters", type=int, default=200)
    depth_parser.add_argument("--eval-interval", type=int, default=500)
    depth_parser.add_argument("--n-neg", type=int, default=128)
    depth_parser.add_argument("--output-root", type=str, default="")
    depth_parser.add_argument("--overwrite", action="store_true")
    depth_parser.add_argument("--summarize-only", action="store_true")

    robustness_parser = subparsers.add_parser("robustness", help="Run the noise-injection robustness study.")
    robustness_parser.add_argument("--dataset", choices=["shakespeare"], default="shakespeare")
    robustness_parser.add_argument("--algorithms", nargs="+", choices=["bp", "nce"], default=["bp", "nce"])
    robustness_parser.add_argument("--optimizer", choices=["adam"], default="adam")
    robustness_parser.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.05, 0.1])
    robustness_parser.add_argument("--depths", nargs="+", type=int, default=[4, 8])
    robustness_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    robustness_parser.add_argument("--gpu", type=int, default=0)
    robustness_parser.add_argument("--max-iters", type=int, default=3000)
    robustness_parser.add_argument("--batch-size", type=int, default=None)
    robustness_parser.add_argument("--block-size", type=int, default=None)
    robustness_parser.add_argument("--lr", type=float, default=None)
    robustness_parser.add_argument("--min-lr", type=float, default=None)
    robustness_parser.add_argument("--n-neg", type=int, default=None)
    robustness_parser.add_argument("--warmup-iters", type=int, default=100)
    robustness_parser.add_argument("--resume", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "scaling":
        return run_scaling(args)
    if args.command == "depth-scan":
        return run_depth_scan(args)
    if args.command == "robustness":
        return run_robustness(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
