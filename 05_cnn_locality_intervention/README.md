# CNN locality intervention

**Paper output:** Table 1, controlled CIFAR-10 locality intervention with CNN12.

`locality_block_sweep.py` runs the grouped-local cross-entropy sweep; `aggregate_locality_block_sweep.py` and `summarize_locality_erank_metrics.py` aggregate the resulting JSON files.

```bash
python locality_block_sweep.py --layers-per-block 1 --epochs 1 --device cuda:0 --max-train-batches 1 --max-test-batches 1 --max-probe-batches 1 --out-root results/smoke
```

The formal comparison sweeps block sizes 1, 2, 3, 4, 6, and 12 across the paper's seed schedule.
