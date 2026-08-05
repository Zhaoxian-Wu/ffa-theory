"""Strict depth scan built on top of the unified scaling runner."""

from __future__ import annotations

import json
import os
from pathlib import Path

from local_learning_nanogpt.experiments.datasets import load_dataset_bundle
from local_learning_nanogpt.experiments.profiles import resolve_erank_interval
from local_learning_nanogpt.experiments.runtime import resolve_torch_device
from local_learning_nanogpt.experiments.scaling import run_single_scaling_experiment
from local_learning_nanogpt.experiments.scales import ScaleSpec
from local_learning_nanogpt.paths import resolve_results_dir


def summarize_result(result: dict) -> dict:
    best_entry = min(result["val_ppls"], key=lambda item: item["ppl"])
    last_eranks = result["gram_eranks"][-1]["eranks"] if result["gram_eranks"] else []
    return {
        "name": result["name"],
        "mode": result["mode"],
        "algorithm": result["algorithm"],
        "optimizer": result["optimizer"],
        "n_layer": result["n_layer"],
        "n_embd": result["n_embd"],
        "n_head": result["n_head"],
        "n_params": result["n_params"],
        "best_ppl": result["best_ppl"],
        "best_step": best_entry["step"],
        "final_ppl": result["final_ppl"],
        "time_sec": result["time"],
        "last_gram_eranks": last_eranks,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def aggregate_raw(raw_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(raw_dir.glob("L*.json"), key=lambda item: int(item.stem[1:])):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def run_depth_scan(args) -> int:
    if args.dataset != "owt_small":
        raise ValueError("depth-scan currently only supports dataset=owt_small.")
    if args.profile != "quick":
        raise ValueError("depth-scan currently only supports profile=quick.")
    if args.algorithm != "lce" or args.optimizer != "adam":
        raise ValueError("depth-scan currently only supports algorithm=lce and optimizer=adam.")

    output_root = Path(args.output_root or os.path.join(resolve_results_dir(), "r13_exp3_owt_lce_depth"))
    raw_dir = output_root / "raw"
    subset_path = output_root / f"subset_gpu{args.gpu}.json"
    summary_path = output_root / "summary.json"

    if args.summarize_only:
        summary = aggregate_raw(raw_dir)
        write_json(summary_path, summary)
        print(f"Summary saved to {summary_path}")
        for row in summary:
            print(
                f"L={row['n_layer']:>2d}  best_ppl={row['best_ppl']:.2f}  "
                f"best_step={row['best_step']:>4d}  time={row['time_sec']:.0f}s"
            )
        return 0

    dataset_bundle = load_dataset_bundle(args.dataset)
    device = resolve_torch_device(args.device, args.gpu)
    print(f"Device: {device}")
    print(f"Data dir: {dataset_bundle.data_dir}")
    print(f"Depths: {args.depths}")
    print(f"Fixed config: d={args.width}, h={args.heads}, algorithm=lce, optimizer=adam")

    subset_rows = []
    for depth in args.depths:
        out_file = raw_dir / f"L{depth}.json"
        if out_file.exists() and not args.overwrite:
            with open(out_file) as f:
                cached = json.load(f)
            print(f"Skipping L={depth}: found existing result at {out_file}")
            subset_rows.append(cached)
            continue

        scale = ScaleSpec(
            name=f"L{depth}",
            n_layer=depth,
            n_embd=args.width,
            n_head=args.heads,
            approx_params=None,
        )
        result = run_single_scaling_experiment(
            scale=scale,
            algorithm_name=args.algorithm,
            optimizer_name=args.optimizer,
            dataset_name=args.dataset,
            profile_name=args.profile,
            train_data=dataset_bundle.train_data,
            val_data=dataset_bundle.val_data,
            vocab_size=dataset_bundle.vocab_size,
            device=device,
            max_iters=args.max_iters,
            batch_size=args.batch_size,
            block_size=args.block_size,
            lr=args.lr,
            min_lr=args.min_lr,
            n_neg=args.n_neg,
            warmup_iters=args.warmup_iters,
            eval_interval=args.eval_interval,
            eval_batches=5,
            erank_interval=resolve_erank_interval(profile_name="quick", eval_interval=args.eval_interval),
            probe_iters=0,
            token_multiplier=None,
        )
        row = summarize_result(result)
        write_json(out_file, row)
        subset_rows.append(row)
        print(
            f"Saved L{depth}: best_ppl={row['best_ppl']:.2f}, "
            f"best_step={row['best_step']}, final_ppl={row['final_ppl']:.2f}"
        )

    write_json(subset_path, subset_rows)

    if all((raw_dir / f"L{depth}.json").exists() for depth in args.depths):
        full_summary = aggregate_raw(raw_dir)
        write_json(summary_path, full_summary)
        print(f"\nFull summary saved to {summary_path}")
    return 0
