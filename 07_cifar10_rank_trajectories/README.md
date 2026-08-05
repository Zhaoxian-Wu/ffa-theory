# CIFAR-10 representation and signal effective-rank trajectories

**Paper output:** Appendix Figure 3.

This directory contains the BP/FFA trajectory runners, signal-rank measurement, renderer, and curve plotter.
Shared CIFAR-10 code is linked from `../03_cifar10_unified_benchmark/`.

```bash
python train_bp_ffa_gamma_erank_trajectory.py --archs cnn3 --epochs 1 --probe-epochs 0 1 --device cuda:0 --out-dir results/smoke
```

Use the full 200-epoch, multi-architecture run before rendering a paper-scale trajectory.
