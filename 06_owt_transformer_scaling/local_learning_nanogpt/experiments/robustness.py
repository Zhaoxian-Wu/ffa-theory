"""Noise-injection robustness experiments under the unified CLI."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch

from local_learning_nanogpt.core.ffa_gpt import FFAGPT, FFAGPTConfig
from local_learning_nanogpt.experiments.algorithms import get_algorithm_spec, prepare_batch, train_step
from local_learning_nanogpt.experiments.datasets import get_dataset_spec, load_dataset_bundle
from local_learning_nanogpt.experiments.optimizers import build_optimizer_bundle
from local_learning_nanogpt.experiments.runtime import resolve_torch_device
from local_learning_nanogpt.paths import prepare_results_path


DEPTH_CONFIGS = {
    4: {"n_layer": 4, "n_embd": 256, "n_head": 4},
    8: {"n_layer": 8, "n_embd": 256, "n_head": 4},
}


def inject_noise(model: torch.nn.Module, sigma: float) -> None:
    if sigma == 0:
        return
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dim() >= 2:
                parameter.add_(torch.randn_like(parameter) * sigma)


def run_one(
    *,
    depth: int,
    algorithm_name: str,
    optimizer_name: str,
    sigma: float,
    dataset_name: str,
    train_data,
    val_data,
    vocab_size: int,
    device: str,
    max_iters: int,
    batch_size: int,
    block_size: int,
    lr: float,
    min_lr: float,
    warmup_iters: int,
    n_neg: int,
) -> dict:
    cfg = DEPTH_CONFIGS[depth]
    algorithm = get_algorithm_spec(algorithm_name)
    model = FFAGPT(
        FFAGPTConfig(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            dropout=0.1,
            n_neg=n_neg,
        ),
        mode=algorithm.model_mode,
    ).to(device)
    optimizer_bundle = build_optimizer_bundle(
        model=model,
        algorithm_name=algorithm_name,
        optimizer_name=optimizer_name,
        lr=lr,
    )
    n_params = sum(parameter.numel() for parameter in model.parameters())

    def get_lr(step: int) -> float:
        if step < warmup_iters:
            return lr * step / max(1, warmup_iters)
        ratio = (step - warmup_iters) / max(1, max_iters - warmup_iters)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)

    best_ppl = float("inf")
    ppls = []
    step_times = []
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(max_iters):
        t_step = time.time()
        model.train()
        batch = prepare_batch(algorithm, train_data, block_size, batch_size, device, model)
        optimizer_bundle.set_lr(get_lr(step))
        train_step(algorithm, model, optimizer_bundle, batch)
        inject_noise(model, sigma)
        step_times.append(time.time() - t_step)

        if step % 500 == 0 or step == max_iters - 1:
            model.eval()
            losses = []
            for _ in range(3):
                val_batch = prepare_batch(algorithm, val_data, block_size, batch_size, device, model)
                losses.append(model.evaluate_perplexity(val_batch["x"], val_batch["y"]))
            ppl = math.exp(min(float(np.mean(losses)), 20.0))
            best_ppl = min(best_ppl, ppl)
            ppls.append({"step": step, "ppl": float(ppl)})

    if str(device).startswith("cuda"):
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
    else:
        peak_mem_mb = 0.0
    avg_step_ms = float(np.mean(step_times) * 1000)

    return {
        "depth": f"L{depth}",
        "algorithm": algorithm_name,
        "optimizer": optimizer_name,
        "dataset": dataset_name,
        "mode": algorithm.model_mode,
        "sigma": sigma,
        "n_params": n_params,
        "n_layer": cfg["n_layer"],
        "best_ppl": best_ppl,
        "final_ppl": ppls[-1]["ppl"],
        "ppls": ppls,
        "peak_memory_mb": peak_mem_mb,
        "avg_step_ms": avg_step_ms,
        "total_time_s": float(sum(step_times)),
    }


def run_robustness(args) -> int:
    if args.dataset != "shakespeare":
        raise ValueError("robustness currently only supports dataset=shakespeare.")
    if args.optimizer != "adam":
        raise ValueError("robustness currently only supports optimizer=adam.")

    dataset_spec = get_dataset_spec(args.dataset)
    dataset_bundle = load_dataset_bundle(args.dataset)
    device = resolve_torch_device(args.device, args.gpu)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset} ({dataset_bundle.data_dir})")
    print(f"Algorithms: {args.algorithms}")
    print(f"Sigmas: {args.sigmas}")
    print(f"Depths: {args.depths}")
    print()

    out_path = prepare_results_path("noise_injection.json")
    all_results = []
    if args.resume and out_path:
        try:
            with open(out_path) as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = []

    batch_size = args.batch_size or dataset_spec.default_batch_size
    block_size = args.block_size or dataset_spec.default_block_size
    lr = args.lr if args.lr is not None else dataset_spec.default_lr
    min_lr = args.min_lr if args.min_lr is not None else dataset_spec.default_min_lr
    n_neg = args.n_neg if args.n_neg is not None else dataset_spec.default_n_neg

    for depth in args.depths:
        if depth not in DEPTH_CONFIGS:
            raise ValueError(f"Unsupported robustness depth: {depth}")
        for algorithm_name in args.algorithms:
            for sigma in args.sigmas:
                print(f"--- L{depth}/{algorithm_name}/sigma={sigma} ---")
                result = run_one(
                    depth=depth,
                    algorithm_name=algorithm_name,
                    optimizer_name=args.optimizer,
                    sigma=sigma,
                    dataset_name=args.dataset,
                    train_data=dataset_bundle.train_data,
                    val_data=dataset_bundle.val_data,
                    vocab_size=dataset_bundle.vocab_size,
                    device=device,
                    max_iters=args.max_iters,
                    batch_size=batch_size,
                    block_size=block_size,
                    lr=lr,
                    min_lr=min_lr,
                    warmup_iters=args.warmup_iters,
                    n_neg=n_neg,
                )
                print(
                    f"  best_ppl={result['best_ppl']:.2f} mem={result['peak_memory_mb']:.0f}MB "
                    f"step={result['avg_step_ms']:.1f}ms total={result['total_time_s']:.0f}s"
                )
                all_results.append(result)
                with open(out_path, "w") as f:
                    json.dump(all_results, f, indent=2)
    return 0
