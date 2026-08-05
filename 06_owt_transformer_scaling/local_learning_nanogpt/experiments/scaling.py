"""Unified scaling-law runner across algorithm / optimizer / scale / dataset / profile axes."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import torch

from local_learning_nanogpt.core.ffa_gpt import FFAGPT, FFAGPTConfig
from local_learning_nanogpt.experiments.algorithms import (
    compute_eranks,
    evaluate_perplexity,
    get_algorithm_spec,
    run_probe,
    select_algorithms,
    train_step,
    prepare_batch,
)
from local_learning_nanogpt.experiments.datasets import get_dataset_spec, load_dataset_bundle
from local_learning_nanogpt.experiments.optimizers import build_optimizer_bundle
from local_learning_nanogpt.experiments.profiles import (
    TOKENS_PER_STEP,
    resolve_erank_interval,
    resolve_eval_batches,
    resolve_eval_interval,
    resolve_max_iters,
    resolve_probe_iters,
    resolve_token_multiplier,
    resolve_warmup_iters,
)
from local_learning_nanogpt.experiments.runtime import resolve_torch_device
from local_learning_nanogpt.experiments.scales import ScaleSpec, select_scales
from local_learning_nanogpt.experiments.specs import AlgorithmSpec, ScalingPreset
from local_learning_nanogpt.paths import prepare_results_path


LEGACY_PRESETS = [
    ScalingPreset(
        name="scaling_law",
        dataset="shakespeare",
        profile="quick",
        algorithms=("bp", "nce"),
        optimizers=("adam",),
        scales=("tiny", "small", "medium", "large", "xlarge"),
        output_filename="scaling_law.json",
        default_max_iters=3000,
        default_probe_iters=0,
    ),
    ScalingPreset(
        name="scaling_law_owt",
        dataset="owt_small",
        profile="quick",
        algorithms=("bp", "nce"),
        optimizers=("adam",),
        scales=("tiny", "small", "medium", "large"),
        output_filename="scaling_law_owt.json",
        default_max_iters=5000,
        default_probe_iters=0,
    ),
    ScalingPreset(
        name="scaling_law_owt_lce",
        dataset="owt_small",
        profile="quick",
        algorithms=("bp", "nce", "lce"),
        optimizers=("adam",),
        scales=("tiny", "small", "medium", "large"),
        output_filename="scaling_law_owt_lce.json",
        default_max_iters=5000,
        default_probe_iters=1000,
    ),
    ScalingPreset(
        name="scaling_law_owt_msp",
        dataset="owt_small",
        profile="quick",
        algorithms=("bp", "nce", "msp"),
        optimizers=("adam",),
        scales=("tiny", "small", "medium", "large"),
        output_filename="scaling_law_owt_msp.json",
        default_max_iters=5000,
        default_probe_iters=1000,
    ),
    ScalingPreset(
        name="scaling_law_owt_muon",
        dataset="owt_small",
        profile="quick",
        algorithms=("bp", "lce"),
        optimizers=("adam", "muon"),
        scales=("tiny", "small", "medium"),
        output_filename="scaling_law_owt_muon.json",
        default_max_iters=5000,
        default_probe_iters=0,
    ),
    ScalingPreset(
        name="scaling_law_owt_xlarge",
        dataset="owt_small",
        profile="quick",
        algorithms=("bp", "lce"),
        optimizers=("adam",),
        scales=("xlarge",),
        output_filename="scaling_law_owt_xlarge.json",
        default_max_iters=10000,
        default_probe_iters=0,
    ),
    ScalingPreset(
        name="scaling_law_chinchilla",
        dataset="owt_small",
        profile="chinchilla",
        algorithms=("bp", "lce"),
        optimizers=("adam",),
        scales=("tiny", "small", "medium", "large", "xlarge"),
        output_filename="scaling_law_chinchilla.json",
        default_probe_iters=0,
    ),
    ScalingPreset(
        name="scaling_law_owt_full_bp",
        dataset="owt_full",
        profile="chinchilla",
        algorithms=("bp",),
        optimizers=("adam",),
        scales=("tiny", "small", "medium", "large", "xlarge"),
        output_filename="scaling_law_owt_full_bp.json",
        default_probe_iters=0,
    ),
    ScalingPreset(
        name="scaling_law_fair_xlarge",
        dataset="owt_full",
        profile="chinchilla",
        algorithms=("bp", "lce"),
        optimizers=("adam",),
        scales=("large", "xlarge"),
        output_filename="scaling_law_fair_xlarge.json",
        default_probe_iters=0,
    ),
]


def infer_legacy_preset(
    dataset: str,
    profile: str,
    algorithms: list[str],
    optimizers: list[str],
    scales: list[str],
) -> ScalingPreset | None:
    normalized_algorithms = tuple(sorted(algorithms))
    normalized_optimizers = tuple(sorted(optimizers))
    normalized_scales = tuple(scales)
    for preset in LEGACY_PRESETS:
        if (
            preset.dataset == dataset
            and preset.profile == profile
            and tuple(sorted(preset.algorithms)) == normalized_algorithms
            and tuple(sorted(preset.optimizers)) == normalized_optimizers
            and preset.scales == normalized_scales
        ):
            return preset
    return None


def validate_scaling_selection(
    *,
    dataset_name: str,
    profile_name: str,
    algorithms: list[AlgorithmSpec],
    optimizers: list[str],
) -> None:
    if profile_name == "chinchilla" and dataset_name not in {"owt_small", "owt_full"}:
        raise ValueError("profile=chinchilla only supports dataset in {owt_small, owt_full}.")
    if dataset_name == "shakespeare" and profile_name != "quick":
        raise ValueError("dataset=shakespeare only supports profile=quick.")

    for algorithm in algorithms:
        for optimizer in optimizers:
            if optimizer not in algorithm.supports_optimizers:
                raise ValueError(
                    f"algorithm={algorithm.name} does not support optimizer={optimizer}."
                )


def _legacy_or_generic_output(
    *,
    dataset_name: str,
    profile_name: str,
    algorithms: list[str],
    optimizers: list[str],
    scales: list[str],
    output_name: str | None,
    preset: ScalingPreset | None,
) -> str:
    if output_name:
        filename = output_name if output_name.endswith(".json") else f"{output_name}.json"
        return prepare_results_path(filename)
    if preset is not None:
        return prepare_results_path(preset.output_filename)
    filename = (
        f"scaling__{dataset_name}__{profile_name}__alg-{'-'.join(algorithms)}"
        f"__opt-{'-'.join(optimizers)}__scale-{'-'.join(scales)}.json"
    )
    return prepare_results_path(filename)


def _should_log_eranks(step: int, max_iters: int, erank_interval: int) -> bool:
    return step % erank_interval == 0 or step == max_iters - 1


def run_single_scaling_experiment(
    *,
    scale: ScaleSpec,
    algorithm_name: str,
    optimizer_name: str,
    dataset_name: str,
    profile_name: str,
    train_data,
    val_data,
    vocab_size: int,
    device: str,
    max_iters: int,
    batch_size: int,
    block_size: int,
    lr: float,
    min_lr: float,
    n_neg: int,
    warmup_iters: int,
    eval_interval: int,
    eval_batches: int,
    erank_interval: int,
    probe_iters: int,
    token_multiplier: float | None,
    lce_rank: int,
) -> dict[str, Any]:
    algorithm = get_algorithm_spec(algorithm_name)
    config = FFAGPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=scale.n_layer,
        n_head=scale.n_head,
        n_embd=scale.n_embd,
        dropout=0.1,
        n_neg=n_neg,
        lce_rank=lce_rank,
    )
    model = FFAGPT(config, mode=algorithm.model_mode).to(device)
    optimizer_bundle = build_optimizer_bundle(
        model=model,
        algorithm_name=algorithm.name,
        optimizer_name=optimizer_name,
        lr=lr,
    )
    n_params = sum(parameter.numel() for parameter in model.parameters())
    total_tokens = max_iters * batch_size * block_size
    t0 = time.time()

    local_ce_algorithms = {"lce", "pd_lc", "plce", "lrva_lce"}
    if algorithm.name in local_ce_algorithms:
        lce_extra_params = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".goodness_head." in name
        )
        print(
            f"  [{scale.name}/{algorithm.name}/{optimizer_name}] total={n_params/1e6:.2f}M "
            f"base≈{(n_params-lce_extra_params)/1e6:.2f}M local_extra={lce_extra_params/1e6:.2f}M"
        )
    elif algorithm.name == "msp":
        horizons = [min(2**index, model.K_max) for index in range(scale.n_layer)]
        print(f"  [{scale.name}/{algorithm.name}/{optimizer_name}] K_per_layer={horizons}")

    best_ppl = float("inf")
    val_ppls: list[dict[str, Any]] = []
    eranks_log: list[dict[str, Any]] = []

    def get_lr(step: int) -> float:
        if step < warmup_iters:
            return lr * step / max(1, warmup_iters)
        ratio = (step - warmup_iters) / max(1, max_iters - warmup_iters)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)

    for step in range(max_iters):
        model.train()
        batch = prepare_batch(algorithm, train_data, block_size, batch_size, device, model)
        optimizer_bundle.set_lr(get_lr(step))
        train_loss = train_step(algorithm, model, optimizer_bundle, batch)

        if step % eval_interval == 0 or step == max_iters - 1:
            model.eval()
            val_ppl = evaluate_perplexity(
                model,
                val_data,
                block_size,
                batch_size,
                device,
                num_batches=eval_batches,
            )
            best_ppl = min(best_ppl, val_ppl)
            val_ppls.append({"step": step, "ppl": float(val_ppl)})

            if _should_log_eranks(step, max_iters, erank_interval):
                eranks_log.append({"step": step, "eranks": compute_eranks(model, val_data, block_size, batch_size, device)})

            elapsed = time.time() - t0
            print(
                f"  {scale.name:>8} {algorithm.model_mode:<15} {optimizer_name:<5} "
                f"step {step:7d}/{max_iters} | loss {train_loss:.4f} | "
                f"ppl {val_ppl:8.1f} (best {best_ppl:.1f}) | {elapsed:.0f}s"
            )

    probe_best_ppl, probe_entries = run_probe(
        algorithm,
        model,
        train_data,
        val_data,
        block_size=block_size,
        batch_size=batch_size,
        device=device,
        probe_iters=probe_iters,
        eval_batches=eval_batches,
        start_step=max_iters,
    )
    if probe_entries:
        best_ppl = min(best_ppl, probe_best_ppl)
        val_ppls.extend(probe_entries)

    elapsed = time.time() - t0
    result: dict[str, Any] = {
        "name": scale.name,
        "scale": scale.name,
        "algorithm": algorithm.name,
        "optimizer": optimizer_name,
        "dataset": dataset_name,
        "profile": profile_name,
        "mode": algorithm.model_mode,
        "exp_key": f"{algorithm.model_mode}_{optimizer_name}",
        "n_params": n_params,
        "approx_params": scale.approx_params,
        "n_layer": scale.n_layer,
        "n_embd": scale.n_embd,
        "n_head": scale.n_head,
        "max_iters": max_iters,
        "total_tokens": total_tokens,
        "probe_iters": probe_iters,
        "best_ppl": best_ppl,
        "final_ppl": val_ppls[-1]["ppl"] if val_ppls else None,
        "val_ppls": val_ppls,
        "gram_eranks": eranks_log,
        "time": elapsed,
        "dataset_fraction": total_tokens / len(train_data),
    }
    if profile_name == "chinchilla" and scale.approx_params and token_multiplier is not None:
        result["token_multiplier"] = token_multiplier
        result["chinchilla_ratio"] = total_tokens / (token_multiplier * scale.approx_params)
    if algorithm.name in {"lce", "pd_lc", "plce", "lrva_lce"}:
        result["local_ce_extra_params"] = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".goodness_head." in name
        )
    if algorithm.name == "lrva_lce":
        result["lce_rank"] = lce_rank
    if algorithm.name == "msp":
        result["K_max"] = model.K_max
        result["K_per_layer"] = [min(2**index, model.K_max) for index in range(scale.n_layer)]
    return result


def _load_existing_results(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _result_key(result: dict[str, Any]) -> tuple[str, str, str, int | None]:
    return result["name"], result["algorithm"], result["optimizer"], result.get("lce_rank")


def run_scaling(args) -> int:
    dataset_spec = get_dataset_spec(args.dataset)
    dataset_bundle = load_dataset_bundle(args.dataset)
    algorithms = select_algorithms(args.algorithms)
    optimizers = args.optimizers
    scales = select_scales(args.scales)
    validate_scaling_selection(
        dataset_name=args.dataset,
        profile_name=args.profile,
        algorithms=algorithms,
        optimizers=optimizers,
    )

    preset = infer_legacy_preset(args.dataset, args.profile, args.algorithms, args.optimizers, args.scales)
    out_path = _legacy_or_generic_output(
        dataset_name=args.dataset,
        profile_name=args.profile,
        algorithms=args.algorithms,
        optimizers=args.optimizers,
        scales=args.scales,
        output_name=args.output_name,
        preset=preset,
    )
    all_results = _load_existing_results(out_path)
    done = {_result_key(result) for result in all_results}
    device = resolve_torch_device(args.device, args.gpu)

    batch_size = args.batch_size or dataset_spec.default_batch_size
    block_size = args.block_size or dataset_spec.default_block_size
    lr = args.lr if args.lr is not None else dataset_spec.default_lr
    min_lr = args.min_lr if args.min_lr is not None else dataset_spec.default_min_lr
    n_neg = args.n_neg if args.n_neg is not None else dataset_spec.default_n_neg
    token_multiplier = resolve_token_multiplier(args.profile, args.token_multiplier)
    probe_iters = resolve_probe_iters(probe_iters_override=args.probe_iters, preset=preset)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} ({dataset_bundle.data_dir})")
    print(f"Profile: {args.profile}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Optimizers: {optimizers}")
    print(f"Scales: {args.scales}")
    if "lrva_lce" in args.algorithms:
        print(f"LCE rank: {args.lce_rank}")
    print(f"Output: {out_path}")
    if token_multiplier is not None:
        print(f"Token multiplier: {token_multiplier}×params")
    print()

    for optimizer_name in optimizers:
        for algorithm in algorithms:
            for scale in scales:
                key = (
                    scale.name,
                    algorithm.name,
                    optimizer_name,
                    args.lce_rank if algorithm.name == "lrva_lce" else None,
                )
                if key in done:
                    print(f"Skipping {scale.name}/{algorithm.name}/{optimizer_name}")
                    continue

                max_iters = resolve_max_iters(
                    profile_name=args.profile,
                    dataset_name=args.dataset,
                    scale=scale,
                    max_iters_override=args.max_iters,
                    token_multiplier_override=args.token_multiplier,
                    preset=preset,
                )
                warmup_iters = resolve_warmup_iters(
                    profile_name=args.profile,
                    dataset_name=args.dataset,
                    scale_name=scale.name,
                    max_iters=max_iters,
                    preset=preset,
                )
                eval_interval = resolve_eval_interval(
                    profile_name=args.profile,
                    max_iters=max_iters,
                    eval_interval_override=args.eval_interval,
                    preset=preset,
                )
                eval_batches = resolve_eval_batches(
                    profile_name=args.profile,
                    dataset_name=args.dataset,
                )
                erank_interval = resolve_erank_interval(
                    profile_name=args.profile,
                    eval_interval=eval_interval,
                )

                print(f"=== {scale.name} / {algorithm.name} / {optimizer_name} ===")
                result = run_single_scaling_experiment(
                    scale=scale,
                    algorithm_name=algorithm.name,
                    optimizer_name=optimizer_name,
                    dataset_name=args.dataset,
                    profile_name=args.profile,
                    train_data=dataset_bundle.train_data,
                    val_data=dataset_bundle.val_data,
                    vocab_size=dataset_bundle.vocab_size,
                    device=device,
                    max_iters=max_iters,
                    batch_size=batch_size,
                    block_size=block_size,
                    lr=lr,
                    min_lr=min_lr,
                    n_neg=n_neg,
                    warmup_iters=warmup_iters,
                    eval_interval=eval_interval,
                    eval_batches=eval_batches,
                    erank_interval=erank_interval,
                    probe_iters=probe_iters,
                    token_multiplier=token_multiplier,
                    lce_rank=args.lce_rank,
                )
                all_results.append(result)
                with open(out_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"  -> best_ppl={result['best_ppl']:.2f} saved to {out_path}\n")

    print("\n=== Scaling Summary ===")
    print(f"{'Scale':<8} {'Algorithm':<8} {'Opt':<6} {'Best PPL':>10} {'Steps':>8}")
    print("-" * 50)
    for result in all_results:
        print(
            f"{result['name']:<8} {result['algorithm']:<8} {result['optimizer']:<6} "
            f"{result['best_ppl']:>10.2f} {result['max_iters']:>8d}"
        )
    return 0
