"""Aggregate the monotone spectral-interpolation sweep with reused endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


POINTS = (
    (0, "baseline", "baseline"),
    (20, "interp20", "new"),
    (40, "interp40", "new"),
    (60, "interp60", "new"),
    (80, "interp80", "new"),
    (100, "flat100", "full_rank"),
)
ALGORITHMS = ("fa", "ffa")


def result_path(
    percentage: int,
    source: str,
    roots: dict[str, Path],
    arch: str,
    seed: int,
    algorithm: str,
    condition: str,
) -> Path:
    del percentage
    return roots[source] / arch / f"seed{seed}" / algorithm / f"{condition}.json"


def load_cell(path: Path, require_checkpoint: bool):
    if not path.exists():
        raise FileNotFoundError(f"Missing result: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise RuntimeError(f"Invalid result status in {path}")
    if payload["config"]["epochs"] != 200:
        raise RuntimeError(f"Expected 200 epochs in {path}")
    checkpoint = path.with_suffix(".pt")
    if require_checkpoint and not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    return payload, checkpoint if checkpoint.exists() else None


def best_epoch(payload: dict) -> int:
    recorded = payload.get("best_epoch")
    if recorded is not None:
        return int(recorded)
    curve = payload.get("test_acc_curve", [])
    if not curve:
        raise RuntimeError("Cannot recover best epoch from an empty accuracy curve.")
    return max(range(len(curve)), key=curve.__getitem__) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-root", default="results/CIFAR10-rank-interpolation-sweep"
    )
    parser.add_argument(
        "--baseline-root", default="results/CIFAR10-rank-control-200ep"
    )
    parser.add_argument(
        "--full-root", default="results/CIFAR10-rank-fraction-sweep"
    )
    parser.add_argument("--arch", default="cnn3")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    roots = {
        "baseline": Path(args.baseline_root),
        "new": Path(args.new_root),
        "full_rank": Path(args.full_root),
    }
    cells = {}
    for percentage, condition, source in POINTS:
        cells[percentage] = {}
        for algorithm in ALGORITHMS:
            path = result_path(
                percentage,
                source,
                roots,
                args.arch,
                args.seed,
                algorithm,
                condition,
            )
            payload, checkpoint = load_cell(path, require_checkpoint=percentage != 0)
            spectral_summary = payload.get("spectral_summary", {})
            cells[percentage][algorithm] = {
                "condition": condition,
                "source": source,
                "test_acc_final": payload["test_acc_final"],
                "test_acc_best": payload["test_acc_best"],
                "best_epoch": best_epoch(payload),
                "mean_raw_gram_participation_ranks": [
                    values.get("mean_raw_gram_participation_rank")
                    for _, values in sorted(spectral_summary.items())
                ],
                "mean_applied_gram_participation_ranks": [
                    values.get("mean_applied_gram_participation_rank")
                    for _, values in sorted(spectral_summary.items())
                ],
                "result_path": str(path),
                "checkpoint_path": None if checkpoint is None else str(checkpoint),
            }

    output_root = Path(args.new_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "arch": args.arch,
        "seed": args.seed,
        "epochs": 200,
        "interpolation": "lambda_i(p)=(1-p)*lambda_i+p*mean(lambda)",
        "endpoint_reuse": {
            "0_percent": str(roots["baseline"]),
            "100_percent": str(roots["full_rank"]),
        },
        "cells": cells,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    rows = [
        "| Spectral interpolation | FA final | FA best | FA best epoch | "
        "FFA final | FFA best | FFA best epoch |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for percentage, _, _ in POINTS:
        fa = cells[percentage]["fa"]
        ffa = cells[percentage]["ffa"]
        rows.append(
            f"| {percentage}% | {100 * fa['test_acc_final']:.2f}% | "
            f"{100 * fa['test_acc_best']:.2f}% | {fa['best_epoch']} | "
            f"{100 * ffa['test_acc_final']:.2f}% | "
            f"{100 * ffa['test_acc_best']:.2f}% | {ffa['best_epoch']} |"
        )
    markdown = "\n".join(rows) + "\n"
    table_path = output_root / "accuracy_table.md"
    table_path.write_text(markdown)

    print(markdown)
    print(f"Saved {summary_path}")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
