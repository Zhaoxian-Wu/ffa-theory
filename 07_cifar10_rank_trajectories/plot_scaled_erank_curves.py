"""
Plot scaled-layer sigma/gram effective-rank curves from exported JSON data.

Each input JSON is expected to come from extract_curve_data_from_checkpoint.py
or any file that follows the same schema. Unlike the earlier averaged plotting
style, this script keeps different total depths separate, so each depth gets
its own curve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve-data", nargs="+", required=True)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/CIFAR10-CNN/output-figure",
    )
    parser.add_argument(
        "--data-out-dir",
        type=str,
        default="experiments/CIFAR10-CNN/output",
    )
    parser.add_argument("--tag", type=str, default="bp_vs_ffa")
    parser.add_argument("--title-prefix", type=str, default="")
    parser.add_argument("--method-order", nargs="*", default=None)
    parser.add_argument("--y-scale", type=str, default="linear", choices=["linear", "log"])
    return parser.parse_args()


def load_rows(paths: List[str]) -> List[Dict]:
    rows = []
    for path in paths:
        obj = json.loads(Path(path).read_text())
        rows.append(obj)
    return rows


def method_order(rows: List[Dict], cli_order: List[str] | None) -> List[str]:
    methods = sorted({row["method_label"] for row in rows})
    if not cli_order:
        return methods
    ordered = [m for m in cli_order if m in methods]
    ordered.extend([m for m in methods if m not in ordered])
    return ordered


def infer_depth_label(row: Dict) -> str:
    if row.get("depth") is not None:
        return f"L{row['depth']}"
    if row.get("total_layers") is not None:
        return f"L{row['total_layers']}"
    if row.get("arch"):
        return str(row["arch"])
    return "unknown"


def curve_key(row: Dict) -> str:
    return f"{row['method_label']}-{infer_depth_label(row)}"


def build_curve_payload(rows: List[Dict], metric: str, method_names: List[str]) -> Dict:
    curves = []
    for method in method_names:
        method_rows = [row for row in rows if row["method_label"] == method]
        method_rows.sort(key=lambda r: (r.get("total_layers", r.get("depth", 0)), r.get("arch", "")))
        for row in method_rows:
            xs = [float(p["scaled_layer"]) for p in row["profile"]]
            ys = [float(p[metric]) for p in row["profile"]]
            curves.append(
                {
                    "curve_key": curve_key(row),
                    "method_label": row["method_label"],
                    "depth_label": infer_depth_label(row),
                    "arch": row.get("arch"),
                    "source": row.get("checkpoint", row.get("source", "unknown")),
                    "x": xs,
                    "y": ys,
                }
            )
    return {"curves": curves}


def color_map(methods: List[str]) -> Dict[str, str]:
    palette = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    return {m: palette[i % len(palette)] for i, m in enumerate(methods)}


def line_style_map() -> Dict[str, str]:
    return {
        "L4": "-",
        "L8": "--",
        "L16": "-.",
        "L32": ":",
        "cnn3": "-",
        "cnn6": "--",
        "cnn9": "-.",
    }


def plot_metric(
    curve_obj: Dict,
    metric: str,
    out_prefix: Path,
    methods: List[str],
    title_prefix: str,
    y_scale: str,
) -> None:
    y_label = metric
    title_map = {
        "sigma_erank_pr": "Scaled-Layer Sigma Effective Rank",
        "gram_erank_pr": "Scaled-Layer Gram Effective Rank",
    }
    colors = color_map(methods)
    styles = line_style_map()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for payload in curve_obj["curves"]:
        method = payload["method_label"]
        depth_label = payload["depth_label"]
        xs = np.array(payload["x"], dtype=float)
        ys = np.array(payload["y"], dtype=float)
        style = styles.get(depth_label, "-")
        label = f"{method}-{depth_label}"
        ax.plot(xs, ys, linewidth=2.0, linestyle=style, color=colors[method], label=label)
        ax.scatter(
            xs,
            ys,
            color=colors[method],
            alpha=0.32,
            s=18,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Scaled layer index")
    ax.set_ylabel(y_label)
    ax.set_yscale(y_scale)
    full_title = title_map[metric] if not title_prefix else f"{title_prefix} | {title_map[metric]}"
    ax.set_title(full_title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    data_out_dir = Path(args.data_out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.curve_data)
    methods = method_order(rows, args.method_order)

    sigma_obj = build_curve_payload(rows, "sigma_erank_pr", methods)
    gram_obj = build_curve_payload(rows, "gram_erank_pr", methods)

    (data_out_dir / f"sigma_curve_data_{args.tag}.json").write_text(json.dumps(sigma_obj, indent=2))
    (data_out_dir / f"gram_curve_data_{args.tag}.json").write_text(json.dumps(gram_obj, indent=2))

    plot_metric(
        sigma_obj,
        "sigma_erank_pr",
        out_dir / f"scaled_sigma_erank_pr_{args.tag}",
        methods,
        args.title_prefix,
        args.y_scale,
    )
    plot_metric(
        gram_obj,
        "gram_erank_pr",
        out_dir / f"scaled_gram_erank_pr_{args.tag}",
        methods,
        args.title_prefix,
        args.y_scale,
    )
    print(f"Saved figures to {out_dir}", flush=True)
    print(f"Saved curve data to {data_out_dir}", flush=True)


if __name__ == "__main__":
    main()
