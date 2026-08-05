# The Price of Locality: Why Forward-Forward Underperforms Backpropagation?

Official code repository for the paper *The Price of Locality: Why Forward-Forward Underperforms Backpropagation?*.

Reproducibility code for the experiments in the accompanying Forward-Forward Algorithm (FFA) theory manuscript.
The repository is organized in manuscript order: one top-level directory per experimental group.
Figure and table numbers refer to the reader-visible numbering in the compiled paper.

## Scope and reproducibility status

Every compiled-paper experiment has a corresponding directory.
Where the exact historical implementation was not recovered, that limitation is stated explicitly rather than represented as reproducible code.

| Directory | Paper output | Status |
| --- | --- | --- |
| `01_toy_multilayer_convergence` | Figure 1 (left) | Historical runner is included. |
| `02_cifar10_resnet_image_classification` | Figure 1 (right), Figure 2, Appendix Figures 4–5, Appendix Tables 7–10 | Figure 2 plotting code is included; the training/readout code has not been recovered. |
| `03_cifar10_unified_benchmark` | Appendix Tables 5–6 | Benchmark runners and algorithm registry are included. |
| `04_cifar10_spectral_intervention` | Table 3 (left and right panels) | Intervention and Jacobian-diagnostic code are included. |
| `05_cnn_locality_intervention` | Table 1 | CNN12 locality-sweep code is included. |
| `06_owt_transformer_scaling` | Table 2 | Packaged Transformer scaling implementation is included. |
| `07_cifar10_rank_trajectories` | Appendix Figure 3 | Trajectory runners and renderers are included. |
| `08_cifar10_cnn_vit` | Appendix Table 4 | Candidate historical CNN/ViT implementations are included; exact provenance remains to be audited. |

## Repository layout

```text
ffa-theory/
├── 01_toy_multilayer_convergence/
├── 02_cifar10_resnet_image_classification/
├── 03_cifar10_unified_benchmark/
├── 04_cifar10_spectral_intervention/
├── 05_cnn_locality_intervention/
├── 06_owt_transformer_scaling/
├── 07_cifar10_rank_trajectories/
└── 08_cifar10_cnn_vit/
```

Each directory is self-contained: executable scripts are placed directly in the experiment directory, not behind an additional `src/` layer.
The spectral, locality, and rank-trajectory experiments reuse the shared CIFAR-10 benchmark implementation through relative symbolic links to `03_cifar10_unified_benchmark`.

## Environment setup

The repository was validated with Python 3.11.15, PyTorch 2.10.0, torchvision 0.25.0, NumPy 2.4.3, and Matplotlib 3.10.8.
Formal CIFAR-10 and Transformer reproductions require a CUDA GPU; CPU is appropriate only for brief smoke tests.

### 1. Create the Conda environment

```bash
git clone https://github.com/Zhaoxian-Wu/ffa-theory.git
cd ffa-theory
conda env create -f environment.yml
conda activate ffa-theory
```

`environment.yml` pins the Python, NumPy, Matplotlib, `datasets`, and `tiktoken` versions used for validation.
PyTorch is deliberately installed in the next step because its wheel must match the local CUDA runtime.

### 2. Install PyTorch and the package

For CUDA 12.8, the validated configuration is:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e 06_owt_transformer_scaling
```

For another CUDA version or CPU-only execution, install the matching `torch`/`torchvision` pair from the official PyTorch installation selector, then run the final editable-install command above.

### 3. Verify the installation

```bash
python -c "import torch, torchvision, numpy, matplotlib; print(torch.__version__); print(torch.cuda.is_available())"
python -m local_learning_nanogpt.experiments.cli --help
```

Set `CUDA_VISIBLE_DEVICES` before running GPU jobs when needed.
The historical Figure 1 toy runner is the exception: it explicitly selects `cuda:1`.

## Data and generated artifacts

CIFAR-10 scripts use `torchvision`.
The unified benchmark and spectral intervention can download CIFAR-10 on first use; the Jacobian diagnostic expects an existing dataset under its `data/cifar10` directory.
Use the data-preparation utilities under `06_owt_transformer_scaling/local_learning_nanogpt/data_prep/` and obey the dataset's terms of use.

Store generated artifacts in the corresponding experiment's `data/`, `results/`, `figures/`, or `cache/` directory, as applicable.

## Running experiments

Run commands below from the repository root.
First inspect `--help` for the full configuration surface.
Commands marked **smoke test** intentionally use small budgets and do not reproduce a paper result.

### 01. Toy multi-layer convergence — Figure 1 (left)

```bash
python 01_toy_multilayer_convergence/exp2_multi_layer.py --force
```

The historical runner launches four processes and currently selects `cuda:1` internally.
It performs 200,000 iterations per depth and writes its cache and PDF to `01_toy_multilayer_convergence/cache/` and `01_toy_multilayer_convergence/figures/`, respectively.

### 02. CIFAR-10 ResNet image classification — Figure 2

```bash
cd 02_cifar10_resnet_image_classification
python plot_cifar10_resnet_bench.py
```

This regenerates Figure 2 from the reported mean accuracies embedded in the plotting script.
It is not a training implementation.
The exact code for Figure 1 (right), Appendix Figures 4–5, and Appendix Tables 7–10 has not yet been located.

### 03. Unified CIFAR-10 benchmark — Appendix Tables 5–6

```bash
python 03_cifar10_unified_benchmark/run_benchmark.py \
  --algos bp --arch cnn3 --quick --device cuda \
  --out_dir 03_cifar10_unified_benchmark/results/smoke
```

Replace `--quick` with `--epochs 200` and provide the paper's full algorithm, architecture, and seed matrix for a formal reproduction.
Aggregate result JSON files with `aggregate_results.py` or `aggregate_consolidated.py`.

### 04. Spectral intervention — Table 3

Left panel, short smoke test:

```bash
python 04_cifar10_spectral_intervention/rank_control_intervention.py \
  --algorithms bp --conditions baseline --epochs 1 --device cuda \
  --max-train-batches 1 --max-test-batches 1 \
  --output-root 04_cifar10_spectral_intervention/results/smoke
```

Right panel (requires prepared CIFAR-10 data and CUDA):

```bash
python 04_cifar10_spectral_intervention/jacobian_concentration/measure_init_resnet18_tau_j.py \
  --data-dir 04_cifar10_spectral_intervention/jacobian_concentration/data/cifar10 \
  --out 04_cifar10_spectral_intervention/jacobian_concentration/results/init_resnet18_tau_j.json
```

The aggregation and plotting scripts in `04_cifar10_spectral_intervention/` consume the JSON outputs from the complete seed sweep.

### 05. CNN locality intervention — Table 1

```bash
python 05_cnn_locality_intervention/locality_block_sweep.py \
  --layers-per-block 1 --epochs 1 --device cuda:0 \
  --max-train-batches 1 --max-test-batches 1 --max-probe-batches 1 \
  --out-root 05_cnn_locality_intervention/results/smoke
```

For the formal sweep, evaluate each allowed `--layers-per-block` value (`1, 2, 3, 4, 6, 12`) over the paper's seed schedule, then run `aggregate_locality_block_sweep.py` and `summarize_locality_erank_metrics.py`.

### 06. OpenWebText Transformer scaling — Table 2

```bash
python -m local_learning_nanogpt.experiments.cli scaling \
  --dataset shakespeare --profile quick --algorithms bp \
  --optimizers adam --scales tiny --device cuda --gpu 0 \
  --max-iters 1 --batch-size 1 --block-size 8 --eval-interval 1 \
  --probe-iters 0 --output-name smoke
```

This is a CLI smoke test after installing the package above.
Table 2 instead uses the OpenWebText configuration and Chinchilla-style budget specified in the manuscript.
Consult the package's `data_prep/` utilities before running an OpenWebText job.

### 07. CIFAR-10 rank trajectories — Appendix Figure 3

```bash
python 07_cifar10_rank_trajectories/train_bp_ffa_gamma_erank_trajectory.py \
  --archs cnn3 --epochs 1 --probe-epochs 0 1 --device cuda:0 \
  --out-dir 07_cifar10_rank_trajectories/results/smoke
```

The `render_bp_ffa_gamma_erank_trajectory.py` and `plot_scaled_erank_curves.py` scripts render trajectories after the full training and probe sweep has finished.

### 08. CNN/ViT comparison — Appendix Table 4

The two included scripts are candidate historical implementations and download CIFAR-10 to `/tmp/cifar10_data` when run.
Their exact relation to the final Table 4 protocol has not yet been verified, so this directory is provided for audit rather than as a certified reproduction command.

## Reproducibility notes

- Use explicit `--out-dir` or `--output-root` paths to keep artifacts local to
the experiment directory.
- Set `--device`, `--gpu`, seeds, epochs, and dataset paths explicitly for a
reproducible run; defaults are historical and may be machine-specific.
- A successful smoke test validates the command path only.  It does not
validate agreement with a reported paper number.

## Citation and license

Citation metadata will be added when the accompanying manuscript is publicly released.
A license has not yet been added; until then, reuse requires permission from the authors.
