"""Aggregate and plot the CIFAR-10 CNN12 locality block sweep."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


LAYERS_PER_BLOCK = (1, 2, 3, 4, 6, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": list(values),
    }


def mean_curve(curves: Sequence[Sequence[float]]) -> List[float]:
    lengths = {len(curve) for curve in curves}
    if len(lengths) != 1:
        raise ValueError(f"Curve lengths differ: {sorted(lengths)}")
    return [statistics.fmean(values) for values in zip(*curves)]


def std_curve(curves: Sequence[Sequence[float]]) -> List[float]:
    lengths = {len(curve) for curve in curves}
    if len(lengths) != 1:
        raise ValueError(f"Curve lengths differ: {sorted(lengths)}")
    return [
        statistics.stdev(values) if len(values) > 1 else 0.0
        for values in zip(*curves)
    ]


def load_runs(root: Path, seeds: Sequence[int], allow_partial: bool):
    runs: Dict[int, List[Dict]] = {}
    missing: List[str] = []
    for layers_per_block in LAYERS_PER_BLOCK:
        current = []
        for seed in seeds:
            path = root / f"lpb_{layers_per_block}" / f"seed_{seed}" / "run.json"
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text())
            if payload.get("status") != "completed":
                missing.append(f"{path} (status={payload.get('status')})")
                continue
            current.append(payload)
        if current:
            runs[layers_per_block] = current
    if missing and not allow_partial:
        raise RuntimeError("Missing or incomplete runs:\n" + "\n".join(missing))
    if not runs:
        raise RuntimeError("No completed runs found.")
    return runs, missing


def aggregate(runs: Dict[int, List[Dict]], missing: Sequence[str]) -> Dict:
    output = {"missing_or_incomplete": list(missing), "settings": {}}
    for layers_per_block, items in runs.items():
        first = items[0]
        epochs = first["curves"]["epoch"]
        probe_epochs = [row["epoch"] for row in first["gamma_probes"]]
        layer_erank_curves = []
        for layer_index in range(12):
            curves = [
                [probe["layer_gamma_erank_pr"][layer_index]
                 for probe in item["gamma_probes"]]
                for item in items
            ]
            layer_erank_curves.append({
                "mean": mean_curve(curves),
                "std": std_curve(curves),
            })
        terminal_train_curves = [
            item["curves"]["train_terminal_group_ce"] for item in items
        ]
        mean_train_curves = [
            item["curves"]["train_mean_group_ce"] for item in items
        ]
        accuracy_curves = [item["curves"]["test_accuracy"] for item in items]
        output["settings"][str(layers_per_block)] = {
            "layers_per_block": layers_per_block,
            "num_blocks": 12 // layers_per_block,
            "seeds": [item["config"]["seed"] for item in items],
            "epochs": epochs,
            "probe_epochs": probe_epochs,
            "test_accuracy_final": mean_std(
                [item["summary"]["test_accuracy_final"] for item in items]
            ),
            "test_accuracy_best": mean_std(
                [item["summary"]["test_accuracy_best"] for item in items]
            ),
            "train_mean_group_ce_final": mean_std(
                [item["summary"]["train_mean_group_ce_final"] for item in items]
            ),
            "train_terminal_group_ce_final": mean_std(
                [item["summary"]["train_terminal_group_ce_final"] for item in items]
            ),
            "last_layer_gamma_erank_pr_final": mean_std(
                [item["summary"]["last_layer_gamma_erank_pr_final"] for item in items]
            ),
            "curves": {
                "test_accuracy_mean": mean_curve(accuracy_curves),
                "test_accuracy_std": std_curve(accuracy_curves),
                "train_mean_group_ce_mean": mean_curve(mean_train_curves),
                "train_mean_group_ce_std": std_curve(mean_train_curves),
                "train_terminal_group_ce_mean": mean_curve(terminal_train_curves),
                "train_terminal_group_ce_std": std_curve(terminal_train_curves),
                "layer_gamma_erank_pr": layer_erank_curves,
            },
        }
    return output


def setting_label(layers_per_block: int) -> str:
    return f"{12 // layers_per_block} blocks × {layers_per_block} layers"


def plot_summary(payload: Dict, root: Path) -> None:
    colors = plt.cm.viridis_r([index / 5 for index in range(6)])
    available = [
        value for key in map(str, LAYERS_PER_BLOCK)
        if (value := payload["settings"].get(key)) is not None
    ]

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    labels = [setting_label(row["layers_per_block"]) for row in available]
    positions = list(range(len(available)))
    final_acc = [100 * row["test_accuracy_final"]["mean"] for row in available]
    final_acc_std = [100 * row["test_accuracy_final"]["std"] for row in available]
    axes[0].errorbar(
        positions, final_acc, yerr=final_acc_std, marker="o", capsize=3, linewidth=2
    )
    axes[0].set_xticks(positions, labels, rotation=35, ha="right")
    axes[0].set_ylabel("Final test accuracy (%)")
    axes[0].grid(alpha=0.25)

    for color, row in zip(colors, available):
        axes[1].plot(
            row["epochs"],
            row["curves"]["train_terminal_group_ce_mean"],
            color=color,
            linewidth=1.8,
            label=setting_label(row["layers_per_block"]),
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Terminal-block train CE")
    axes[1].grid(alpha=0.25)

    for color, row in zip(colors, available):
        axes[2].plot(
            row["probe_epochs"],
            row["curves"]["layer_gamma_erank_pr"][-1]["mean"],
            marker="o",
            color=color,
            linewidth=1.8,
            label=setting_label(row["layers_per_block"]),
        )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel(r"Last-layer $\mathrm{erank}(\Gamma)$")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(root / "locality_accuracy_loss_erank.pdf", bbox_inches="tight")
    figure.savefig(root / "locality_accuracy_loss_erank.png", dpi=220,
                   bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(3, 4, figsize=(12, 8), sharex=True)
    for layer_index, axis in enumerate(axes.flat):
        for color, row in zip(colors, available):
            axis.plot(
                row["probe_epochs"],
                row["curves"]["layer_gamma_erank_pr"][layer_index]["mean"],
                color=color,
                linewidth=1.4,
            )
        axis.set_title(f"Layer {layer_index + 1}")
        axis.grid(alpha=0.2)
    figure.supxlabel("Epoch")
    figure.supylabel(r"$\mathrm{erank}(\Gamma)$")
    figure.tight_layout()
    figure.savefig(root / "all_layer_gamma_erank_trajectory.pdf", bbox_inches="tight")
    figure.savefig(root / "all_layer_gamma_erank_trajectory.png", dpi=220,
                   bbox_inches="tight")
    plt.close(figure)


def write_tables(payload: Dict, root: Path) -> None:
    rows = [
        payload["settings"][key]
        for key in map(str, LAYERS_PER_BLOCK)
        if key in payload["settings"]
    ]
    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "num_blocks", "layers_per_block", "n_seeds",
            "final_test_acc_mean", "final_test_acc_std",
            "best_test_acc_mean", "best_test_acc_std",
            "final_mean_group_train_ce_mean", "final_mean_group_train_ce_std",
            "final_terminal_train_ce_mean", "final_terminal_train_ce_std",
            "final_last_layer_gamma_erank_mean",
            "final_last_layer_gamma_erank_std",
        ])
        for row in rows:
            writer.writerow([
                row["num_blocks"], row["layers_per_block"], len(row["seeds"]),
                row["test_accuracy_final"]["mean"],
                row["test_accuracy_final"]["std"],
                row["test_accuracy_best"]["mean"],
                row["test_accuracy_best"]["std"],
                row["train_mean_group_ce_final"]["mean"],
                row["train_mean_group_ce_final"]["std"],
                row["train_terminal_group_ce_final"]["mean"],
                row["train_terminal_group_ce_final"]["std"],
                row["last_layer_gamma_erank_pr_final"]["mean"],
                row["last_layer_gamma_erank_pr_final"]["std"],
            ])

    lines = [
        "# CNN12 locality block sweep",
        "",
        "| Blocks | Layers/block | Final acc (%) | Best acc (%) | "
        "Mean-group train CE | Terminal train CE | Last-layer Gamma eRank |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['num_blocks']} | {row['layers_per_block']} | "
            f"{100 * row['test_accuracy_final']['mean']:.2f} ± "
            f"{100 * row['test_accuracy_final']['std']:.2f} | "
            f"{100 * row['test_accuracy_best']['mean']:.2f} ± "
            f"{100 * row['test_accuracy_best']['std']:.2f} | "
            f"{row['train_mean_group_ce_final']['mean']:.4f} ± "
            f"{row['train_mean_group_ce_final']['std']:.4f} | "
            f"{row['train_terminal_group_ce_final']['mean']:.4f} ± "
            f"{row['train_terminal_group_ce_final']['std']:.4f} | "
            f"{row['last_layer_gamma_erank_pr_final']['mean']:.3f} ± "
            f"{row['last_layer_gamma_erank_pr_final']['std']:.3f} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    runs, missing = load_runs(args.results_root, args.seeds, args.allow_partial)
    payload = aggregate(runs, missing)
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "summary.json").write_text(json.dumps(payload, indent=2))
    write_tables(payload, args.results_root)
    plot_summary(payload, args.results_root)
    print(
        f"Aggregated {sum(len(items) for items in runs.values())} runs into "
        f"{args.results_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
