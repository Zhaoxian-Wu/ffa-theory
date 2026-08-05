"""Tabulate final accuracy against mean, maximum, and last-layer Gamma eRank."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Sequence


LAYERS_PER_BLOCK = (1, 2, 3, 4, 6, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def mean_std(values: Sequence[float]) -> Dict[str, object]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": list(values),
    }


def load_complete_settings(
    root: Path,
    seeds: Sequence[int],
    allow_partial: bool,
) -> tuple[List[Dict], List[str]]:
    rows: List[Dict] = []
    missing: List[str] = []
    for layers_per_block in LAYERS_PER_BLOCK:
        runs = []
        for seed in seeds:
            path = root / f"lpb_{layers_per_block}" / f"seed_{seed}" / "run.json"
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text())
            if payload.get("status") != "completed":
                missing.append(f"{path} (status={payload.get('status')})")
                continue
            runs.append(payload)
        if len(runs) != len(seeds):
            if runs and not allow_partial:
                raise RuntimeError(
                    f"Setting layers_per_block={layers_per_block} is incomplete."
                )
            continue
        final_profiles = [
            run["gamma_probes"][-1]["layer_gamma_erank_pr"] for run in runs
        ]
        rows.append({
            "num_blocks": 12 // layers_per_block,
            "layers_per_block": layers_per_block,
            "seeds": list(seeds),
            "final_test_accuracy": mean_std(
                [run["summary"]["test_accuracy_final"] for run in runs]
            ),
            "mean_layer_gamma_erank_pr": mean_std(
                [statistics.fmean(profile) for profile in final_profiles]
            ),
            "max_layer_gamma_erank_pr": mean_std(
                [max(profile) for profile in final_profiles]
            ),
            "last_layer_gamma_erank_pr": mean_std(
                [profile[-1] for profile in final_profiles]
            ),
        })
    if missing and not allow_partial:
        raise RuntimeError("Missing or incomplete runs:\n" + "\n".join(missing))
    return rows, missing


def write_outputs(root: Path, rows: Sequence[Dict], missing: Sequence[str],
                  partial: bool) -> None:
    suffix = "_partial" if partial else ""
    payload = {
        "scope": (
            "final epoch; statistics across seeds; mean/max are across 12 layers "
            "within each seed before seed aggregation"
        ),
        "partial": partial,
        "missing_or_incomplete": list(missing),
        "rows": list(rows),
    }
    (root / f"erank_metric_table{suffix}.json").write_text(
        json.dumps(payload, indent=2)
    )

    with (root / f"erank_metric_table{suffix}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "num_blocks", "layers_per_block",
            "final_acc_mean", "final_acc_std",
            "mean_layer_erank_mean", "mean_layer_erank_std",
            "max_layer_erank_mean", "max_layer_erank_std",
            "last_layer_erank_mean", "last_layer_erank_std",
        ])
        for row in rows:
            writer.writerow([
                row["num_blocks"], row["layers_per_block"],
                row["final_test_accuracy"]["mean"],
                row["final_test_accuracy"]["std"],
                row["mean_layer_gamma_erank_pr"]["mean"],
                row["mean_layer_gamma_erank_pr"]["std"],
                row["max_layer_gamma_erank_pr"]["mean"],
                row["max_layer_gamma_erank_pr"]["std"],
                row["last_layer_gamma_erank_pr"]["mean"],
                row["last_layer_gamma_erank_pr"]["std"],
            ])

    lines = [
        "# CNN12 locality eRank metric comparison",
        "",
        "Final-epoch statistics. Mean/max eRank are computed across the 12 "
        "convolutional layers within each seed, then aggregated across seeds.",
        "",
        "| Blocks | Layers/block | Final acc (%) | Mean-layer eRank | "
        "Max-layer eRank | Last-layer eRank |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['num_blocks']} | {row['layers_per_block']} | "
            f"{100 * row['final_test_accuracy']['mean']:.2f} ± "
            f"{100 * row['final_test_accuracy']['std']:.2f} | "
            f"{row['mean_layer_gamma_erank_pr']['mean']:.3f} ± "
            f"{row['mean_layer_gamma_erank_pr']['std']:.3f} | "
            f"{row['max_layer_gamma_erank_pr']['mean']:.3f} ± "
            f"{row['max_layer_gamma_erank_pr']['std']:.3f} | "
            f"{row['last_layer_gamma_erank_pr']['mean']:.3f} ± "
            f"{row['last_layer_gamma_erank_pr']['std']:.3f} |"
        )
    (root / f"erank_metric_table{suffix}.md").write_text(
        "\n".join(lines) + "\n"
    )


def main() -> None:
    args = parse_args()
    rows, missing = load_complete_settings(
        args.results_root, args.seeds, args.allow_partial
    )
    if not rows:
        raise RuntimeError("No complete three-seed settings found.")
    partial = bool(missing)
    write_outputs(args.results_root, rows, missing, partial)
    print(
        f"Wrote {'partial' if partial else 'complete'} eRank table with "
        f"{len(rows)} settings.",
        flush=True,
    )


if __name__ == "__main__":
    main()

