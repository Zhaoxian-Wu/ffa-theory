"""
Exp 2 (T=200 000) -- multi-layer convergence, 4 depths in parallel on GPU 1.

Replaces figures/exp2_multi_layer.pdf with a higher-iteration version.

Usage:
  python experiments/exp2_25k_parallel.py              # run if no cache, then plot
  python experiments/exp2_25k_parallel.py --run-only   # run and cache, skip plot
  python experiments/exp2_25k_parallel.py --plot-only  # plot from existing cache
  python experiments/exp2_25k_parallel.py --force      # re-run even if cache exists, then plot
"""

import argparse
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
CACHEDIR = os.path.join(os.path.dirname(__file__), "cache")

# T = 25000
T = 200000
if T == 25000:
    CACHE_FILE = os.path.join(CACHEDIR, "exp2_25k_results.npz")
elif T == 200000:
    CACHE_FILE = os.path.join(CACHEDIR, "exp2_200k_results.npz")
else:
    raise ValueError(f"Unsupported T={T}")
SEED = 42

depths = [2, 4, 8, 16]

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.figsize": (3.8, 3.),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


class FFALayer(nn.Module):
    def __init__(self, d_in, d_out, use_ln=False):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(d_out, elementwise_affine=False)
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")

    def forward(self, x):
        z = self.linear(x)
        if self.use_ln:
            z = self.ln(z)
        return F.relu(z)


def goodness(h):
    return (h ** 2).sum(dim=1)


def ffa_loss(h_pos, h_neg, theta):
    g_pos = goodness(h_pos)
    g_neg = goodness(h_neg)
    return -F.logsigmoid(g_pos - theta).mean() - F.logsigmoid(theta - g_neg).mean()


def run_one_depth(L, result_queue):
    """Worker: train L-layer FFA for T steps, put loss list into queue."""
    device = torch.device("cuda:1")
    torch.manual_seed(SEED)

    d0, d, n = 50, 500, 100
    mu = torch.zeros(d0, device=device)
    mu[0] = 1.0
    x_pos = mu.unsqueeze(0) + torch.randn(n // 2, d0, device=device)
    x_neg = -mu.unsqueeze(0) + torch.randn(n // 2, d0, device=device)

    dims = [d0] + [d] * L
    layers = nn.ModuleList(
        [FFALayer(dims[i], dims[i + 1], use_ln=True).to(device) for i in range(L)]
    )
    thetas = [float(d)] * L
    c_lr, c_prime = 5.0, 50.0

    losses = []
    for t in range(T):
        eta_t = c_lr / (c_prime + t)
        total = 0.0
        h_pos, h_neg = x_pos, x_neg
        for ell, layer in enumerate(layers):
            layer.zero_grad()
            hp_out = layer(h_pos)
            hn_out = layer(h_neg)
            loss = ffa_loss(hp_out, hn_out, thetas[ell])
            loss.backward(retain_graph=True)
            total += loss.item()
            with torch.no_grad():
                for p in layer.parameters():
                    if p.grad is not None:
                        p -= eta_t * p.grad
                        p.grad = None
            h_pos = layer(h_pos).detach()
            h_neg = layer(h_neg).detach()
        losses.append(total)

        if (t + 1) % 5000 == 0:
            print(f"  L={L}: step {t+1}/{T}, V={total:.6f}", flush=True)

    result_queue.put((L, losses))


def run_experiment():
    """Run training for all depths, return results dict {L: losses_list}."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    procs = []
    for L in depths:
        p = ctx.Process(target=run_one_depth, args=(L, result_queue))
        p.start()
        procs.append(p)
        print(f"Launched worker for L={L}", flush=True)

    results = {}
    for _ in depths:
        L, losses = result_queue.get()
        results[L] = losses
        print(f"  Collected L={L}: V^0={losses[0]:.3f}, V^T={losses[-1]:.6f}", flush=True)

    for p in procs:
        p.join()

    return results


def save_cache(results):
    os.makedirs(CACHEDIR, exist_ok=True)
    np.savez(CACHE_FILE, **{f"L{L}": np.array(losses) for L, losses in results.items()})
    print(f"Cached results to {CACHE_FILE}", flush=True)


def load_cache():
    data = np.load(CACHE_FILE)
    results = {int(key[1:]): data[key].tolist() for key in data.files}
    print(f"Loaded cached results from {CACHE_FILE}", flush=True)
    return results

def plot_results(results):
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(1, 1)

    for L in depths:
        plotted_list = [r / L for r in results[L]]
        ax.loglog(range(1, T + 1), plotted_list, label=f"$L={L}$", linewidth=0.8)
        # ax.semilogy(range(1, T + 1), plotted_list, label=f"$L={L}$", linewidth=0.8)
        # ax.plot(range(1, T + 1), plotted_list, label=f"$L={L}$", linewidth=0.8)
    ts = np.arange(100, T + 1)
    
    ax.vlines(1000, ymin=1e0, ymax=1e2, colors="grey", linestyles=":", linewidth=1.0, alpha=0.6)
    
    # ax.loglog(ts, results[2][99] * 100 / (ts), "k--", alpha=0.5, label="$O(1/K)$")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Lyapunov function $V_k$")
    # ax.set_title("Multi-layer convergence")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGDIR, "exp2_multi_layer.pdf")
    plt.savefig(out)
    plt.close()
    print(f"Saved {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Exp 2: multi-layer FFA convergence")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-only", action="store_true", help="run training and cache, skip plotting")
    group.add_argument("--plot-only", action="store_true", help="load cache and plot, skip training")
    parser.add_argument("--force", action="store_true", help="re-run training even if cache exists")
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(CACHE_FILE):
            print(f"ERROR: cache not found at {CACHE_FILE}. Run without --plot-only first.", flush=True)
            sys.exit(1)
        results = load_cache()
        plot_results(results)
        return

    cache_exists = os.path.exists(CACHE_FILE)
    if cache_exists and not args.force:
        print(f"Cache found at {CACHE_FILE}, loading...", flush=True)
        results = load_cache()
    else:
        if args.force and cache_exists:
            print("--force: ignoring existing cache, re-running.", flush=True)
        results = run_experiment()
        save_cache(results)

    if not args.run_only:
        plot_results(results)


if __name__ == "__main__":
    main()
