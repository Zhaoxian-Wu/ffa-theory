"""Aggregate and plot the BP/FFA spectral-interpolation intervention."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


PERCENTAGES = (0, 20, 40, 60, 80, 100)
ALGORITHMS = ("bp", "ffa")
COLORS = {"bp": "#0072B2", "ffa": "#D55E00"}
LABELS = {"bp": "BP", "ffa": "FFA"}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing result: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise RuntimeError(f"Invalid result status: {path}")
    return payload


def result_base(
    sweep_root: Path,
    endpoint_root: Path,
    algorithm: str,
    seed: int,
    percentage: int,
) -> Path:
    if algorithm == "ffa" and seed == 0 and percentage == 100:
        return endpoint_root / "cnn3" / "seed0" / "ffa" / "flat100"
    condition = "baseline" if percentage == 0 else f"interp{percentage}"
    return sweep_root / "cnn3" / f"seed{seed}" / algorithm / condition


def load_cell(
    sweep_root: Path,
    endpoint_root: Path,
    algorithm: str,
    seed: int,
    percentage: int,
) -> dict:
    base = result_base(sweep_root, endpoint_root, algorithm, seed, percentage)
    result = load_json(base.with_suffix(".json"))
    if result["config"]["epochs"] != 200:
        raise RuntimeError(f"Expected 200 epochs: {base}.json")
    probe = result.get("terminal_gamma_probe")
    if probe is None:
        probe = load_json(base.with_suffix(".gamma.json"))["terminal_gamma_probe"]
    return {
        "result_path": str(base.with_suffix(".json")),
        "best_accuracy_percent": 100.0 * result["test_acc_best"],
        "mean_layer_gamma_erank_pr": probe["mean_layer_gamma_erank_pr"],
    }


def aggregate(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-root", type=Path, default=Path("results/CIFAR10-rank-interpolation-sweep")
    )
    parser.add_argument(
        "--endpoint-root", type=Path, default=Path("results/CIFAR10-rank-fraction-sweep")
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/CIFAR10-rank-interpolation-sweep/"
            "bp_ffa_multiseed_summary.json"
        ),
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=Path("tex/main/figures/spectral_intervention_bp_ffa.pdf"),
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=Path("tex/main/figures/spectral_intervention_bp_ffa.png"),
    )
    args = parser.parse_args()

    cells: dict[str, dict[str, dict]] = {}
    for algorithm in ALGORITHMS:
        cells[algorithm] = {}
        for percentage in PERCENTAGES:
            seed_rows = [
                load_cell(
                    args.sweep_root,
                    args.endpoint_root,
                    algorithm,
                    seed,
                    percentage,
                )
                for seed in range(3)
            ]
            cells[algorithm][str(percentage)] = {
                "seeds": seed_rows,
                "best_accuracy_percent": aggregate(
                    [row["best_accuracy_percent"] for row in seed_rows]
                ),
                "mean_layer_gamma_erank_pr": aggregate(
                    [row["mean_layer_gamma_erank_pr"] for row in seed_rows]
                ),
            }

    summary = {
        "dataset": "CIFAR-10",
        "architecture": "cnn3",
        "epochs": 200,
        "seeds": [0, 1, 2],
        "p_percent": list(PERCENTAGES),
        "accuracy_metric": "best test accuracy (%)",
        "gamma_metric": (
            "terminal mean layerwise participation-ratio effective rank"
        ),
        "cells": cells,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.65))
    metrics = (
        ("best_accuracy_percent", "Best test accuracy (%)", "(a) Accuracy"),
        (
            "mean_layer_gamma_erank_pr",
            r"Mean layerwise $\operatorname{erank}(\Gamma)$",
            "(b) Error-signal effective rank",
        ),
    )
    for axis, (metric, ylabel, title) in zip(axes, metrics):
        for algorithm in ALGORITHMS:
            rows = [cells[algorithm][str(p)][metric] for p in PERCENTAGES]
            axis.errorbar(
                PERCENTAGES,
                [row["mean"] for row in rows],
                yerr=[row["sample_sd"] for row in rows],
                color=COLORS[algorithm],
                marker="o" if algorithm == "bp" else "s",
                markersize=4,
                linewidth=1.5,
                capsize=2.5,
                label=LABELS[algorithm],
            )
        axis.set_xlabel(r"Spectral interpolation $p$ (\%)")
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=5)
        axis.set_xticks(PERCENTAGES)
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.legend(frameon=False)

    figure.tight_layout(w_pad=2.0)
    args.pdf_output.parent.mkdir(parents=True, exist_ok=True)
    args.png_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.pdf_output, bbox_inches="tight")
    figure.savefig(args.png_output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.summary_output}")
    print(f"Saved {args.pdf_output}")
    print(f"Saved {args.png_output}")


if __name__ == "__main__":
    main()
