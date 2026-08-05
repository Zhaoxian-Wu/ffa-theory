"""
Gradient Alignment Measurement: FFA vs BP under Three Metrics

Measures per-layer cosine similarity between FFA-local and BP-global gradients
under three metrics:
  1. Frobenius (raw): cos(g_FFA, g_BP) in Euclidean space
  2. Adam-preconditioned: cos(P^{-1/2} g_FFA, P^{-1/2} g_BP) where P = diag(sqrt(v) + eps)
  3. Muon-orthogonalized: cos(NS(g_FFA), NS(g_BP)) for 2D weight matrices

Theory predictions (Remark rem:metric_alignment in the paper):
  - Per-sample rank-1 gradients: Muon does NOT change alignment (invariance)
  - Mini-batch: Muon alignment depends on left singular subspace overlap
  - Depth interaction: Muon helps FFA more at shallow depth, helps BP more at large depth

Usage:
  python measure_gradient_alignment.py [--device cuda] [--data_dir PATH]
"""

import os, math, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from local_learning_nanogpt.core.ffa_gpt import FFAGPT, FFAGPTConfig, get_batch
from local_learning_nanogpt.core.muon_optimizer import zeropower_via_newtonschulz5
from local_learning_nanogpt.paths import prepare_results_path, resolve_shakespeare_data_dir


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between flattened tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    denom = a_flat.norm() * b_flat.norm()
    if denom < 1e-12:
        return 0.0
    return float((a_flat @ b_flat) / denom)


def adam_precondition(grad: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Apply Adam-style preconditioning: grad / sqrt(v + eps)."""
    return grad / (torch.sqrt(v + eps))


def collect_bp_gradients(model, x, y):
    """Compute BP global gradients for each block's parameters.
    Returns dict: block_idx -> {param_name: grad_tensor}
    """
    model.zero_grad()
    _, loss = model.forward_bp(x, y)
    loss.backward()

    bp_grads = {}
    for i, block in enumerate(model.blocks):
        bp_grads[i] = {}
        for name, p in block.named_parameters():
            if p.grad is not None:
                bp_grads[i][name] = p.grad.clone()
    return bp_grads, loss.item()


def collect_ffa_gradients(model, x, y):
    """Compute FFA local gradients for each block's parameters.
    Returns dict: block_idx -> {param_name: grad_tensor}
    """
    model.zero_grad()
    # Run FFA forward to get per-layer losses
    layer_losses = model.forward_ffa_lce_untied(x, y)

    ffa_grads = {}
    for i, (block, ll) in enumerate(zip(model.blocks, layer_losses)):
        # Zero grads for this block specifically
        for p in block.parameters():
            if p.grad is not None:
                p.grad.zero_()
        ll.backward(retain_graph=(i < len(layer_losses) - 1))

        ffa_grads[i] = {}
        for name, p in block.named_parameters():
            if p.grad is not None:
                ffa_grads[i][name] = p.grad.clone()

    return ffa_grads, [ll.item() for ll in layer_losses]


def measure_alignment_one_batch(model, x, y, adam_v_state=None):
    """
    For one mini-batch, compute BP and FFA gradients, then measure alignment
    under three metrics for each layer.

    adam_v_state: dict of {(block_idx, param_name): running_v} for Adam preconditioning.
                  If None, uses squared gradient as single-step estimate.

    Returns dict with per-layer metrics.
    """
    # 1. Collect FFA gradients first (uses detach, doesn't affect BP computation)
    ffa_grads, ffa_losses = collect_ffa_gradients(model, x, y)

    # 2. Collect BP gradients (needs fresh forward pass)
    bp_grads, bp_loss = collect_bp_gradients(model, x, y)

    results = {}
    for layer_idx in range(len(model.blocks)):
        ffa_g = ffa_grads.get(layer_idx, {})
        bp_g = bp_grads.get(layer_idx, {})

        # Collect all parameter names present in both
        common_names = set(ffa_g.keys()) & set(bp_g.keys())
        if not common_names:
            continue

        # ── Metric 1: Frobenius (raw) ──────────────────────────────────────
        # Concatenate all params for this layer
        ffa_vec = torch.cat([ffa_g[n].flatten() for n in sorted(common_names)])
        bp_vec = torch.cat([bp_g[n].flatten() for n in sorted(common_names)])
        frob_align = cosine_sim(ffa_vec, bp_vec)

        # ── Metric 2: Adam-preconditioned ──────────────────────────────────
        # Use squared gradient as single-step v estimate (or running v if available)
        ffa_adam_parts = []
        bp_adam_parts = []
        for n in sorted(common_names):
            if adam_v_state and (layer_idx, n) in adam_v_state:
                v = adam_v_state[(layer_idx, n)]
            else:
                # Single-step estimate: v ≈ average of both squared grads
                v = 0.5 * (ffa_g[n] ** 2 + bp_g[n] ** 2)
            ffa_adam_parts.append(adam_precondition(ffa_g[n], v).flatten())
            bp_adam_parts.append(adam_precondition(bp_g[n], v).flatten())
        ffa_adam_vec = torch.cat(ffa_adam_parts)
        bp_adam_vec = torch.cat(bp_adam_parts)
        adam_align = cosine_sim(ffa_adam_vec, bp_adam_vec)

        # ── Metric 3: Muon-orthogonalized (2D weights only) ───────────────
        ffa_muon_parts = []
        bp_muon_parts = []
        n_muon_params = 0
        for n in sorted(common_names):
            if ffa_g[n].ndim == 2:
                ffa_orth = zeropower_via_newtonschulz5(ffa_g[n])
                bp_orth = zeropower_via_newtonschulz5(bp_g[n])
                ffa_muon_parts.append(ffa_orth.flatten())
                bp_muon_parts.append(bp_orth.flatten())
                n_muon_params += 1
            else:
                # Non-2D: use raw gradient (Muon falls back to SGD)
                ffa_muon_parts.append(ffa_g[n].flatten())
                bp_muon_parts.append(bp_g[n].flatten())

        if ffa_muon_parts:
            ffa_muon_vec = torch.cat(ffa_muon_parts)
            bp_muon_vec = torch.cat(bp_muon_parts)
            muon_align = cosine_sim(ffa_muon_vec, bp_muon_vec)
        else:
            muon_align = frob_align

        # ── Per-weight-matrix Muon alignment (for detailed analysis) ──────
        per_weight_muon = {}
        for n in sorted(common_names):
            if ffa_g[n].ndim == 2:
                ffa_orth = zeropower_via_newtonschulz5(ffa_g[n])
                bp_orth = zeropower_via_newtonschulz5(bp_g[n])
                per_weight_muon[n] = cosine_sim(ffa_orth, bp_orth)

        results[layer_idx] = {
            'frobenius': frob_align,
            'adam': adam_align,
            'muon': muon_align,
            'n_params': sum(ffa_g[n].numel() for n in common_names),
            'n_muon_params_2d': n_muon_params,
            'per_weight_muon': per_weight_muon,
            'ffa_loss': ffa_losses[layer_idx] if layer_idx < len(ffa_losses) else None,
        }

    return results, bp_loss


def run_experiment(n_layer=4, n_embd=256, n_head=4, block_size=256,
                   batch_size=32, n_measurements=10, max_train_steps=2000,
                   measure_every=200, device='cuda', data_dir=None):
    """
    Train a model and periodically measure gradient alignment.

    Returns list of measurement dicts at each checkpoint.
    """
    data_dir = resolve_shakespeare_data_dir(data_dir)

    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # Vocab size from meta
    import pickle
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta.get('vocab_size', 65)
    else:
        vocab_size = 65

    config = FFAGPTConfig(
        block_size=block_size, vocab_size=vocab_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=0.0, n_neg=128,  # no dropout for clean gradient comparison
    )
    # Use ffa_lce_untied mode so we have both BP and FFA forward paths
    model = FFAGPT(config, mode='ffa_lce_untied').to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: L={n_layer}, d={n_embd}, params={n_params/1e6:.1f}M")

    # Use Adam for actual training (so we can also collect running v stats)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Track Adam's running v for preconditioning
    adam_v = {}

    lr = 3e-4
    min_lr = 3e-5
    warmup_iters = 100

    def get_lr(step):
        if step < warmup_iters:
            return lr * step / max(1, warmup_iters)
        ratio = (step - warmup_iters) / max(1, max_train_steps - warmup_iters)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (lr - min_lr)

    all_measurements = []
    t0 = time.time()

    for step in range(max_train_steps + 1):
        # ── Measurement checkpoint ──────────────────────────────────────
        if step % measure_every == 0:
            model.eval()
            print(f"\n[Step {step}/{max_train_steps}] Measuring gradient alignment...")

            # Average over n_measurements batches for stability
            agg = {}
            bp_losses = []
            for _ in range(n_measurements):
                x, y = get_batch(train_data, block_size, batch_size, device)
                with torch.enable_grad():
                    results, bp_loss = measure_alignment_one_batch(model, x, y, adam_v)
                bp_losses.append(bp_loss)

                for li, metrics in results.items():
                    if li not in agg:
                        agg[li] = {'frobenius': [], 'adam': [], 'muon': []}
                    agg[li]['frobenius'].append(metrics['frobenius'])
                    agg[li]['adam'].append(metrics['adam'])
                    agg[li]['muon'].append(metrics['muon'])

            # Summarize
            measurement = {
                'step': step,
                'bp_loss': np.mean(bp_losses),
                'time': time.time() - t0,
                'layers': {},
            }
            for li in sorted(agg.keys()):
                measurement['layers'][li] = {
                    'frobenius_mean': float(np.mean(agg[li]['frobenius'])),
                    'frobenius_std': float(np.std(agg[li]['frobenius'])),
                    'adam_mean': float(np.mean(agg[li]['adam'])),
                    'adam_std': float(np.std(agg[li]['adam'])),
                    'muon_mean': float(np.mean(agg[li]['muon'])),
                    'muon_std': float(np.std(agg[li]['muon'])),
                }
                print(f"  Layer {li}: Frob={measurement['layers'][li]['frobenius_mean']:.4f} "
                      f"Adam={measurement['layers'][li]['adam_mean']:.4f} "
                      f"Muon={measurement['layers'][li]['muon_mean']:.4f}")

            all_measurements.append(measurement)
            model.train()

        if step >= max_train_steps:
            break

        # ── Training step (BP mode for shared weight updates) ───────────
        x, y = get_batch(train_data, block_size, batch_size, device)
        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        optimizer.zero_grad()
        _, loss = model.forward_bp(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update running v for Adam preconditioning
        beta2 = 0.999
        for i, block in enumerate(model.blocks):
            for name, p in block.named_parameters():
                if p.grad is not None:
                    key = (i, name)
                    g2 = p.grad.data ** 2
                    if key in adam_v:
                        adam_v[key] = beta2 * adam_v[key] + (1 - beta2) * g2
                    else:
                        adam_v[key] = (1 - beta2) * g2

        if step % 200 == 0:
            print(f"  [Train] step={step}, loss={loss.item():.4f}, lr={current_lr:.2e}")

    return all_measurements, config


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_train_steps', type=int, default=2000)
    parser.add_argument('--measure_every', type=int, default=200)
    parser.add_argument('--n_measurements', type=int, default=10)
    args = parser.parse_args()

    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'

    print(f"=== Gradient Alignment Measurement ===")
    print(f"Device: {device}")
    print(f"Config: L={args.n_layer}, d={args.n_embd}, h={args.n_head}")

    measurements, config = run_experiment(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        batch_size=args.batch_size,
        max_train_steps=args.max_train_steps,
        measure_every=args.measure_every,
        n_measurements=args.n_measurements,
        device=device,
        data_dir=args.data_dir,
    )

    # Save results
    out_path = prepare_results_path('gradient_alignment.json')

    output = {
        'config': {
            'n_layer': config.n_layer,
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'block_size': config.block_size,
            'vocab_size': config.vocab_size,
        },
        'args': vars(args),
        'measurements': measurements,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n=== Summary: Mean alignment across training ===")
    print(f"{'Step':>6} | {'Layer':>5} | {'Frobenius':>10} | {'Adam':>10} | {'Muon':>10}")
    print("-" * 55)
    for m in measurements:
        for li in sorted(m['layers'].keys(), key=int):
            l = m['layers'][li]
            print(f"{m['step']:>6} | {li:>5} | {l['frobenius_mean']:>10.4f} | "
                  f"{l['adam_mean']:>10.4f} | {l['muon_mean']:>10.4f}")
        print("-" * 55)


if __name__ == '__main__':
    main()
