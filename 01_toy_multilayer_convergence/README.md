# Toy multi-layer convergence

**Paper output:** Figure 1 (left); protocol details appear in Appendix D.

`exp2_multi_layer.py` is the recovered historical runner for the two-Gaussian depth sweep.
It trains depths 2, 4, 8, and 16, caches trajectories in `cache/`, and writes the rendered PDF to `figures/`.

```bash
python exp2_multi_layer.py --force
```

The historical script hard-codes `cuda:1`, starts four worker processes, and runs 200,000 iterations per depth.
