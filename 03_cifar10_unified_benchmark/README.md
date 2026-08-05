# Unified CIFAR-10 local-learning benchmark

**Paper output:** Appendix Tables 5–6.

This directory contains the benchmark runner, shared model utilities, BP baseline, and the `algos/` registry.
Example smoke test:

```bash
python run_benchmark.py --algos bp --arch cnn3 --quick --device cuda --out_dir results/smoke
```

Use the full seed and algorithm matrix rather than `--quick` to reproduce the appendix tables.
Generated JSON result files belong under `results/`.
