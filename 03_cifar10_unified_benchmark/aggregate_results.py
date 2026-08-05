"""
Aggregate CIFAR10-CNN benchmark JSONs across seeds for a given arch.

Usage:
    python aggregate_results.py --arch cnn3
    python aggregate_results.py --arch cnn6

Inputs:
    results/CIFAR10-CNN/{arch}_seed{0,1,2}/{algo}.json

Outputs:
    results/CIFAR10-CNN/{arch}_summary.json
    tex/main/appendix/CIFAR10-CNN_{arch}_bench.tex
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

# Algorithm labels used in the generated LaTeX table. Entries using \cite{...}
# must match an existing \bibitem key in tex/main/body.tex. For the five
# algorithms whose papers were identified post-hoc (see the README "References
# for algorithms without in-tex bibkey" section), we include the arXiv ID as
# superscript footnote-style text instead, to avoid introducing undefined \cite
# keys into the paper build.
CATEGORY = {
    "bp":                 ("Standard BP", "Baseline"),
    "vanilla_ffa":        ("Vanilla FFA \\cite{hinton2022forward}", "FFA"),
    "symba":              ("SymBa \\cite{lee2023symba}", "FFA-contrastive"),
    "scff":               ("SCFF \\cite{chen2024selfcontrastive}", "FFA-contrastive"),
    "trifecta":           ("Trifecta \\cite{dooms2023trifecta}", "FFA-extended"),
    "scodellaro":         ("Scodellaro CNN-FFA \\cite{scodellaro2025cnn}", "FFA-extended"),
    "sff":                ("SFF \\cite{krutsylo2025scalable}", "Local-CE"),
    "forward_projection": ("Forward Projection \\cite{oshea2025forwardprojection}", "Random-features"),
    "distance_forward":   ("Distance-Forward \\cite{wu2024distance}", "FFA-extended"),
    "belilovsky":         ("Greedy Layerwise \\cite{belilovsky2019greedy}", "Local-CE"),
    "nokland_lpred":      ("N{\\o}kland $L_{\\text{pred}}$ \\cite{nokland2019local}", "Local-CE"),
    "nokland_lsim":       ("N{\\o}kland $L_{\\text{sim}}$ \\cite{nokland2019local}", "Local-sim"),
    "nokland_lpredsim":   ("N{\\o}kland $L_{\\text{pred}+\\text{sim}}$ \\cite{nokland2019local}", "Local-combined"),
    "auglocal":           ("AugLocal \\cite{ma2024auglocal}", "Aux-net"),
    "gim":                ("Greedy InfoMax \\cite{lowe2019greedy}", "InfoNCE-local"),
    "dtp":                ("Difference Target Prop \\cite{lee2015difference}", "Target-Prop"),
    "counter_current":    ("Counter-Current \\cite{kao2024countercurrent}", "Bio-plausible"),
    "dll":                ("Dendritic Localized \\cite{lv2025dendritic}", "Bio-plausible"),
    "predictive_coding":  ("Predictive Coding \\cite{salvatori2026predictive}", "Energy-based"),
    "layer_collab":       ("Layer Collaboration \\cite{lorberbom2024layer}", "Multi-layer-coord"),
}

ORDER = list(CATEGORY.keys())


def _load_seed(dir_: Path) -> Dict[str, dict]:
    out = {}
    for p in sorted(dir_.glob("*.json")):
        with open(p) as f:
            out[p.stem] = json.load(f)
    return out


def _fmt_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "--"
    if std is None or std == 0:
        return f"{mean * 100:.2f}"
    return f"{mean * 100:.2f} $\\pm$ {std * 100:.2f}"


def _fmt_delta(ref: Optional[float], val: Optional[float]) -> str:
    if ref is None or val is None:
        return "--"
    return f"{(val - ref) * 100:+.2f}"


def aggregate(in_dirs: List[Path]) -> dict:
    per_seed: List[Dict[str, dict]] = [_load_seed(d) for d in in_dirs]
    summary = {"algos": [], "n_seeds": len(in_dirs), "in_dirs": [str(d) for d in in_dirs]}

    # BP mean for delta calculation
    bp_vals = [s.get("bp", {}).get("test_acc_best") for s in per_seed
               if s.get("bp", {}).get("status") == "ok"]
    bp_vals = [v for v in bp_vals if v is not None]
    bp_mean = statistics.mean(bp_vals) if bp_vals else None

    for name in ORDER:
        entries = [s.get(name) for s in per_seed]
        entries = [e for e in entries if e is not None]
        if not entries:
            continue
        statuses = {e.get("status") for e in entries}
        if statuses == {"na"}:
            entry = entries[0]
            summary["algos"].append({
                "algo": name, "category": CATEGORY[name][1],
                "status": "na", "n_seeds_ok": 0,
                "test_acc_mean": None, "test_acc_std": None,
                "na_reason": entry.get("na_reason"),
                "tag": entry.get("tag"),
                "n_params": entry.get("n_params"),
                "per_seed_best": [None] * len(entries),
            })
            continue
        ok_accs = [e["test_acc_best"] for e in entries
                   if e.get("status") == "ok" and e.get("test_acc_best") is not None]
        mean = statistics.mean(ok_accs) if ok_accs else None
        std = statistics.stdev(ok_accs) if len(ok_accs) >= 2 else 0.0
        tags = [e.get("tag") for e in entries if e.get("tag")]
        nparams = next((e.get("n_params") for e in entries if e.get("n_params")), None)
        summary["algos"].append({
            "algo": name, "category": CATEGORY[name][1],
            "status": "ok" if ok_accs else "failed",
            "n_seeds_ok": len(ok_accs),
            "test_acc_mean": mean, "test_acc_std": std,
            "per_seed_best": [e.get("test_acc_best") if e.get("status") == "ok" else None
                              for e in entries],
            "tag": tags[0] if tags else None,
            "n_params": nparams,
            "na_reason": None,
        })
    summary["bp_mean"] = bp_mean
    return summary


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate] summary -> {path}")


def write_latex(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bp_mean = summary.get("bp_mean")
    rows: List[str] = []
    for entry in summary["algos"]:
        label = CATEGORY[entry["algo"]][0]
        category = entry["category"]
        status = entry["status"]
        if status == "na":
            acc_cell = "N/A"
            delta_cell = "--"
        else:
            acc_cell = _fmt_mean_std(entry["test_acc_mean"], entry["test_acc_std"])
            delta_cell = _fmt_delta(bp_mean, entry["test_acc_mean"])
        tag = entry.get("tag")
        tag_escaped = tag.replace("_", "\\_") if tag else ""
        tag_note = f"\\textsuperscript{{{tag_escaped}}}" if tag else ""
        nparam = entry.get("n_params")
        param_cell = f"{nparam / 1e3:.0f}K" if (nparam and nparam > 0) else "--"
        rows.append(
            f"  {label}{tag_note} & {category} & {param_cell} & {acc_cell} & {delta_cell} \\\\"
        )

    n_seeds = summary["n_seeds"]
    arch = summary.get("arch", "cnn3")
    seed_note = f"$n_{{\\text{{seed}}}}{{=}}{n_seeds}$, mean $\\pm$ std"
    if arch == "cnn3":
        arch_desc = "a shared 3-layer CNN backbone"
        arch_label_suffix = "_cnn3"
        arch_label_phrase = ""
    elif arch == "cnn6":
        arch_desc = "a shared 6-layer CNN backbone (three preserved spatial scales 32/16/8)"
        arch_label_suffix = "_cnn6"
        arch_label_phrase = "CNN6: "
    elif arch == "cnn9":
        arch_desc = "a shared 9-layer CNN backbone (four preserved spatial scales 32/16/8/4; VGG-style channel growth $32{\\to}64{\\to}128{\\to}256$)"
        arch_label_suffix = "_cnn9"
        arch_label_phrase = "CNN9: "
    else:
        arch_desc = f"a shared {arch} backbone"
        arch_label_suffix = f"_{arch}"
        arch_label_phrase = f"{arch.upper()}: "
    tex = (
        f"% Auto-generated by experiments/CIFAR10-CNN/aggregate_results.py (arch={arch})\n"
        "% Unified local-learning benchmark on CIFAR-10.\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{"
        f"{arch_label_phrase}Unified CIFAR-10 benchmark across local learning algorithms on "
        f"{arch_desc} (Adam, lr $= 10^{{-3}}$, batch $= 128$, 200 epochs, no augmentation; "
        + seed_note + "). "
        "N/A entries are algorithms whose design assumptions degenerate under this arch; see Appendix~\\ref{app:cifar10_cnn_bench}. "
        "\\textsuperscript{\\emph{degenerate}}: auxiliary-net depth schedule collapses. "
        "\\textsuperscript{\\emph{single\\_path}}: dual-path fallback after initial non-convergence. "
        "\\textsuperscript{\\emph{closed\\_form}}: no iterative training (ridge closed-form on random features)."
        "}\n"
        f"\\label{{tab:cifar10_cnn_bench{arch_label_suffix}}}\n"
        "\\begin{tabular}{@{}llrcr@{}}\n"
        "\\toprule\n"
        "Algorithm & Category & Params & Top-1 Acc (\\%) & $\\Delta$ vs.\\ BP (pp) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(path, "w") as f:
        f.write(tex)
    print(f"[aggregate] LaTeX -> {path}")


def aggregate_arch(arch: str, results_root: Path = Path("results/CIFAR10-CNN"),
                   n_seeds: int = 3):
    """Convenience wrapper for `python aggregate_results.py --arch X`."""
    in_dirs = [results_root / f"{arch}_seed{s}" for s in range(n_seeds)]
    existing = [d for d in in_dirs if d.exists()]
    print(f"[aggregate/{arch}] loading from {len(existing)} seed dir(s): "
          f"{[str(d) for d in existing]}")
    summary = aggregate(existing)
    summary["arch"] = arch
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default=None,
                    help="Aggregate results for this arch (cnn3|cnn6). If omitted, "
                         "aggregates ALL archs (cnn3, cnn6, cnn9).")
    ap.add_argument("--results_root", type=str, default="results/CIFAR10-CNN")
    ap.add_argument("--tex_dir", type=str, default="tex/main/appendix")
    args = ap.parse_args()

    archs = [args.arch] if args.arch else ["cnn3", "cnn6", "cnn9"]
    results_root = Path(args.results_root)

    for arch in archs:
        summary = aggregate_arch(arch, results_root=results_root)
        summary_path = results_root / f"{arch}_summary.json"
        tex_path = Path(args.tex_dir) / f"CIFAR10-CNN_{arch}_bench.tex"
        write_summary(summary, summary_path)
        write_latex(summary, tex_path)
        print(f"[aggregate/{arch}] {len(summary['algos'])} algorithms  "
              f"n_seeds={summary['n_seeds']}  "
              f"BP mean={summary['bp_mean'] * 100:.2f}%" if summary["bp_mean"]
              else f"[aggregate/{arch}] {len(summary['algos'])} algorithms  "
                   f"n_seeds={summary['n_seeds']}  BP missing")


if __name__ == "__main__":
    main()
