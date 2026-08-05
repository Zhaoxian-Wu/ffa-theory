"""Generate the locality-audited CIFAR-10 CNN benchmark table.

The legacy consolidated table mixed end-to-end fallback runs, methods with
strict block-local gradients, and parameter-local methods whose targets or
feedback carry information across depth.  This generator reads the formal
audit manifest and includes only evidence-supported rows. The Category
column preserves the original mechanism taxonomy; the audit is conveyed by
the panel headings and table notes. The repaired strict-local FFA and
alternating-PC summaries are separate result sources, because they were
rerun after the audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARCHS = ("cnn3", "cnn6", "cnn9")
ROW_END = r" \\"

GRADIENT_ISOLATED_ROWS = (
    ("vanilla_ffa", "Vanilla FFA \\cite{hinton2022forward}", "FFA"),
    ("symba", "SymBa \\cite{lee2023symba}", "FFA-contrastive"),
    ("scff", "SCFF \\cite{chen2024selfcontrastive}", "FFA-contrastive"),
    ("trifecta", "Trifecta \\cite{dooms2023trifecta}", "FFA-extended"),
    ("scodellaro", "Scodellaro CNN-FFA \\cite{scodellaro2025cnn}", "FFA-extended"),
    ("distance_forward", "Distance-Forward \\cite{wu2024distance}", "FFA-extended"),
    ("sff", "SFF \\cite{krutsylo2025scalable}", "Local-CE"),
    ("belilovsky", "Greedy Layerwise \\cite{belilovsky2019greedy}", "Local-CE"),
    ("nokland_lpred", "N{\\o}kland $L_{\\mathrm{pred}}$ \\cite{nokland2019local}", "Local-CE"),
    ("nokland_lsim", "N{\\o}kland $L_{\\mathrm{sim}}$ \\cite{nokland2019local}", "Local-sim"),
    ("nokland_lpredsim", "N{\\o}kland $L_{\\mathrm{pred}+\\mathrm{sim}}$ \\cite{nokland2019local}", "Local-combined"),
    ("gim", "Greedy InfoMax \\cite{lowe2019greedy}", "InfoNCE-local"),
    ("dll", r"Dendritic Localized \cite{lv2025dendritic}", "Dendritic-local"),
)
REPAIRED_FFA = {"vanilla_ffa", "symba", "scff", "scodellaro"}

AUXILIARY_ROWS = (
    ("auglocal", "AugLocal \\cite{ma2024auglocal}", "Aux-net"),
    ("dtp", "Difference Target Prop \\cite{lee2015difference}", "Target-Prop"),
)

CROSS_LAYER_ROWS = (
    ("layer_collab", "Layer Collaboration \\cite{lorberbom2024layer}", "Multi-layer-coord"),
)

COUNTER_CURRENT_ROW = (
    "counter_current", r"Counter-Current \cite{kao2024countercurrent}", "Dual-path feedback"
)

def fmt(mean: float | None, std: float | None, status: str = "ok") -> str:
    if status == "na":
        return "N/A"
    if mean is None:
        return "--"
    if std is None or std == 0:
        return f"{100 * mean:.2f}"
    return f"{100 * mean:.2f} $\\pm$ {100 * std:.2f}"


def recipe_entries(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for arch in ARCHS:
        payload = json.loads((root / f"{arch}_summary.json").read_text())
        result[arch] = {row["algo"]: row for row in payload["algos"]}
    return result


def cells(entries: dict[str, dict[str, dict[str, Any]]], algo: str) -> list[str]:
    output = []
    for arch in ARCHS:
        row = entries[arch].get(algo)
        output.append("--" if row is None else fmt(
            row.get("test_acc_mean"), row.get("test_acc_std"), row.get("status", "ok")))
    return output


def build(recipe_root: Path, repair_root: Path, counter_current_root: Path,
          pc_summary_paths: dict[str, Path], out: Path) -> None:
    entries = recipe_entries(recipe_root)
    repaired_entries = recipe_entries(repair_root)
    counter_current_entries = recipe_entries(counter_current_root)
    pc_summaries = {
        arch: json.loads(path.read_text())
        for arch, path in pc_summary_paths.items()
        if path.exists()
    }
    if "cnn3" not in pc_summaries:
        raise RuntimeError(f"Missing required CNN3 PC summary: {pc_summary_paths['cnn3']}")

    rows: list[str] = []
    bp = cells(entries, "bp")
    rows.append("  Standard BP & Baseline & " + " & ".join(bp) + ROW_END)
    rows.append("  \\midrule")
    for algo, label, regime in GRADIENT_ISOLATED_ROWS:
        values = cells(repaired_entries, algo) if algo in REPAIRED_FFA else cells(entries, algo)
        rows.append(f"  {label} & {regime} & " + " & ".join(values) + ROW_END)
    rows.append("  \\midrule")
    for algo, label, regime in AUXILIARY_ROWS:
        rows.append(f"  {label} & {regime} & " + " & ".join(cells(entries, algo)) + ROW_END)
    rows.append("  \\midrule")
    algo, label, regime = COUNTER_CURRENT_ROW
    rows.append(f"  {label} & {regime} & "
                + " & ".join(cells(counter_current_entries, algo)) + ROW_END)
    for algo, label, regime in CROSS_LAYER_ROWS:
        rows.append(f"  {label} & {regime} & " + " & ".join(cells(entries, algo)) + ROW_END)
    rows.append("  \\midrule")
    rows.append("  Forward Projection \\cite{oshea2025forwardprojection} & Random-features & "
                + " & ".join(cells(entries, "forward_projection")) + ROW_END)

    rows.append("  \\midrule")
    for key in sorted(pc_summaries["cnn3"]["settings"], key=int):
        values = []
        for arch in ARCHS:
            summary = pc_summaries.get(arch)
            setting = None if summary is None else summary["settings"].get(key)
            values.append("--" if setting is None else fmt(
                setting["test_accuracy_best"]["mean"],
                setting["test_accuracy_best"]["std"],
            ))
        rows.append("  PC ($K=" + key + "$) & Energy-based & "
                    + " & ".join(values) + ROW_END)

    tex = "\n".join([
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        "Algorithm & Category & CNN3 & CNN6 & CNN9" + ROW_END,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-root", "--results-root", "--results_root", dest="recipe_root", type=Path,
                        default=ROOT / "results" / "CIFAR10-CNN_recipe2")
    parser.add_argument("--repair-root", type=Path,
                        default=ROOT / "results" / "CIFAR10-strict_local_family_repair")
    parser.add_argument("--counter-current-root", type=Path,
                        default=ROOT / "results" / "CIFAR10-counter-current-repair")
    parser.add_argument("--pc-root", type=Path,
                        default=ROOT / "results" / "CIFAR10-PC_fixed_prediction_v2_channel_ln_200epoch")
    parser.add_argument("--out", "--out-tex", "--out_tex", dest="out", type=Path,
                        default=ROOT / "tex" / "main" / "appendix" / "CIFAR10-CNN_bench.tex")
    args = parser.parse_args()
    pc_summary_paths = {
        arch: (args.pc_root / "summary.json" if arch == "cnn3" else args.pc_root / "aggregates" / arch / "summary.json")
        for arch in ARCHS
    }
    build(args.recipe_root, args.repair_root, args.counter_current_root,
          pc_summary_paths, args.out)


if __name__ == "__main__":
    main()
