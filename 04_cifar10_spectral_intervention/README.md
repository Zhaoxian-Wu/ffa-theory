# Controlled spectral intervention

**Paper output:** Table 3.
The left panel is the CIFAR-10 CNN spectral interpolation experiment; the right panel is the finite-sample Jacobian concentration diagnostic.

`rank_control_intervention.py` implements the left-panel sweep and its aggregation/plotting scripts are colocated here.
The right-panel code lives in `jacobian_concentration/`.
Both reuse only the shared benchmark code in `../03_cifar10_unified_benchmark/` through relative links.

```bash
python rank_control_intervention.py --algorithms bp --conditions baseline --epochs 1 --device cuda --max-train-batches 1 --max-test-batches 1 --output-root results/smoke
```
