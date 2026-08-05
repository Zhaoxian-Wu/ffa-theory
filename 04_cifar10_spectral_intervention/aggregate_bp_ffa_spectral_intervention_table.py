"""Aggregate the BP/FFA spectral intervention into a grouped LaTeX table."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


PERCENTAGES = (0, 20, 40, 60, 80, 100)
ALGORITHMS = ("bp", "ffa")


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


def format_mean_sd(summary: dict, digits: int, suffix: str = "") -> str:
    mean = summary["mean"]
    sample_sd = summary["sample_sd"]
    return "$" + f"{mean:.{digits}f}{{\\pm}}{sample_sd:.{digits}f}{suffix}" + "$"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("results/CIFAR10-rank-interpolation-sweep"),
    )
    parser.add_argument(
        "--endpoint-root",
        type=Path,
        default=Path("results/CIFAR10-rank-fraction-sweep"),
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
        "--table-output",
        type=Path,
        default=Path(
            "results/CIFAR10-rank-interpolation-sweep/"
            "bp_ffa_multiseed_table.tex"
        ),
    )
    args = parser.parse_args()

    cells: dict[str, dict[str, dict]] = {}
    for percentage in PERCENTAGES:
        cells[str(percentage)] = {}
        for algorithm in ALGORITHMS:
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
            cells[str(percentage)][algorithm] = {
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

    rows = [
        r"\begin{tabular}{rcc cc}",
        r"\toprule",
        (
            r"& \multicolumn{2}{c}{Best accuracy} "
            r"& \multicolumn{2}{c}{$\erank(\Gamma)$} \\"
        ),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"$p$ & BP & FFA & BP & FFA \\",
        r"\midrule",
    ]
    for percentage in PERCENTAGES:
        bp = cells[str(percentage)]["bp"]
        ffa = cells[str(percentage)]["ffa"]
        bp_accuracy = format_mean_sd(bp["best_accuracy_percent"], 2, r"\%")
        ffa_accuracy = format_mean_sd(ffa["best_accuracy_percent"], 2, r"\%")
        bp_erank = format_mean_sd(bp["mean_layer_gamma_erank_pr"], 3)
        ffa_erank = format_mean_sd(ffa["mean_layer_gamma_erank_pr"], 3)
        p_value = percentage / 100.0
        rows.append(
            "$" + f"{p_value:.1f}$ & "
            f"{bp_accuracy} & {ffa_accuracy} & "
            f"{bp_erank} & {ffa_erank} \\\\"
        )
    rows.extend((r"\bottomrule", r"\end{tabular}"))
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    args.table_output.write_text("\n".join(rows) + "\n")
    print(f"Saved {args.summary_output}")
    print(f"Saved {args.table_output}")


if __name__ == "__main__":
    main()
