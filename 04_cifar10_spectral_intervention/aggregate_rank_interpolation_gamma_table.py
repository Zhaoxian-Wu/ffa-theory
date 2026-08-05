"""Aggregate FA/FFA interpolation accuracy and terminal Gamma eRank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


POINTS = (
    (0, "baseline", "new"),
    (20, "interp20", "new"),
    (40, "interp40", "new"),
    (60, "interp60", "new"),
    (80, "interp80", "new"),
    (100, "flat100", "endpoint"),
)
ALGORITHMS = ("fa", "ffa")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing result: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise RuntimeError(f"Invalid result status: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new-root", default="results/CIFAR10-rank-interpolation-sweep"
    )
    parser.add_argument(
        "--endpoint-root", default="results/CIFAR10-rank-fraction-sweep"
    )
    parser.add_argument("--arch", default="cnn3")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    roots = {"new": Path(args.new_root), "endpoint": Path(args.endpoint_root)}
    cells = {}
    for percentage, condition, source in POINTS:
        cells[percentage] = {}
        for algorithm in ALGORITHMS:
            base = (
                roots[source]
                / args.arch
                / f"seed{args.seed}"
                / algorithm
                / condition
            )
            result = load_json(base.with_suffix(".json"))
            if result["config"]["epochs"] != 200:
                raise RuntimeError(f"Expected 200 epochs: {base}.json")
            if source == "new":
                probe = result.get("terminal_gamma_probe")
            else:
                probe = load_json(base.with_suffix(".gamma.json"))[
                    "terminal_gamma_probe"
                ]
            if probe is None:
                raise RuntimeError(f"Missing terminal Gamma probe: {base}")
            cells[percentage][algorithm] = {
                "condition": condition,
                "source": source,
                "test_acc_final": result["test_acc_final"],
                "test_acc_best": result["test_acc_best"],
                "best_epoch": result.get("best_epoch"),
                "layer_gamma_erank_pr": probe["layer_gamma_erank_pr"],
                "mean_layer_gamma_erank_pr": probe[
                    "mean_layer_gamma_erank_pr"
                ],
                "last_layer_gamma_erank_pr": probe[
                    "last_layer_gamma_erank_pr"
                ],
                "result_path": str(base.with_suffix(".json")),
            }

    output_root = Path(args.new_root)
    summary = {
        "arch": args.arch,
        "seed": args.seed,
        "epochs": 200,
        "accuracy_metric": "best test accuracy",
        "gamma_metric": "terminal mean layerwise participation-ratio eRank",
        "cells": cells,
    }
    summary_path = output_root / "gamma_accuracy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    rows = [
        "| $p$ | FA best acc. | FA mean $\\operatorname{erank}(\\Gamma)$ | "
        "FFA best acc. | FFA mean $\\operatorname{erank}(\\Gamma)$ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for percentage, _, _ in POINTS:
        fa = cells[percentage]["fa"]
        ffa = cells[percentage]["ffa"]
        rows.append(
            f"| ${percentage}\\%$ | {100 * fa['test_acc_best']:.2f}% | "
            f"{fa['mean_layer_gamma_erank_pr']:.3f} | "
            f"{100 * ffa['test_acc_best']:.2f}% | "
            f"{ffa['mean_layer_gamma_erank_pr']:.3f} |"
        )
    markdown = "\n".join(rows) + "\n"
    table_path = output_root / "gamma_accuracy_table.md"
    table_path.write_text(markdown)
    print(markdown)
    print(f"Saved {summary_path}")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
