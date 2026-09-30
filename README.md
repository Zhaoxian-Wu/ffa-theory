# The Price of Locality: Why Forward-Forward Underperforms Backpropagation?

Official code for **The Price of Locality: Why Forward-Forward Underperforms
Backpropagation?** by Zhaoxian Wu, Haichuan Liu, and Tianyi Chen.

[Paper](https://arxiv.org/abs/2609.33240) ·
[PDF](https://arxiv.org/pdf/2609.33240)

The paper studies the optimization and representation limitations of
Forward-Forward learning relative to backpropagation. The experiments cover
multi-layer convergence, image classification, controlled locality and spectral
interventions, representation and error-signal ranks, and Transformer scaling.

## Experiments

The repository follows manuscript order, with one numbered directory per
experimental group. Figure and table numbers refer to the paper.

| Directory | Experiment | Paper output |
| --- | --- | --- |
| [01_toy_multilayer_convergence](01_toy_multilayer_convergence/) | Multi-layer FFA convergence and depth-dependent error floors. | Figure 1 (left) |
| [02_cifar10_resnet_image_classification](02_cifar10_resnet_image_classification/) | ResNet training, detached readout, depth and task-difficulty scaling, and training-recipe comparisons. | Figure 1 (right), Figure 2, Appendix Figures 4–5 and Tables 7–10 |
| [03_cifar10_unified_benchmark](03_cifar10_unified_benchmark/) | Local-learning algorithms on shared CNN3/6/9 backbones. | Appendix Tables 5–6 |
| [04_cifar10_spectral_intervention](04_cifar10_spectral_intervention/) | Spectral interpolation and Jacobian-concentration diagnostics. | Table 3 |
| [05_cnn_locality_intervention](05_cnn_locality_intervention/) | Grouped-goodness FFA with different gradient horizons on a fixed CNN12. | Table 1 |
| [06_owt_transformer_scaling](06_owt_transformer_scaling/) | Transformer pre-training on OpenWebText under Chinchilla-style budgets. | Table 2 |
| [07_cifar10_rank_trajectories](07_cifar10_rank_trajectories/) | Representation and error-signal effective ranks during CNN training. | Appendix Figure 3 |
| [08_cifar10_cnn_vit](08_cifar10_cnn_vit/) | BP and FFA on CNN and Vision Transformer architectures. | Appendix Table 4 |

Each directory provides its own README and experiment entry points. The spectral,
locality, and rank-trajectory experiments share the CNN benchmark implementation
through relative links to `03_cifar10_unified_benchmark`.

## Installation

Create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate ffa-theory
```

Install PyTorch and torchvision for your CUDA runtime. The existing environment
configuration was used with PyTorch 2.10.0 and torchvision 0.25.0. For CUDA 12.8:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

For the Transformer experiments, also install the package:

```bash
python -m pip install -e 06_owt_transformer_scaling
```

The ResNet module uses PyTorch, torchvision, NumPy, and Pillow; these dependencies
are also listed in `requirements.txt`. Plotting scripts use Matplotlib. Full
training experiments are intended for a CUDA GPU.

## ResNet image-classification experiments

The ResNet module includes native training, detached readout, dataset
preprocessing, baseline and hardened recipes, and result aggregation.

| Dimension | Settings |
| --- | --- |
| Architecture | ResNet18, ResNet24, ResNet56, ResNet108 |
| Dataset | MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet |
| Methods | BP, LCE, SFF, Distance-Forward, Vanilla FFA, SCFF, SymBa, LayerCollab, Trifecta |
| Readout | Linear classifier on detached, pooled stem and residual-stage outputs |
| Paper training budget | 200 epochs, batch size 128, seed 0 |

MNIST and CIFAR-10 use all nine methods. CIFAR-100 and Tiny ImageNet compare BP,
LCE, SFF, and Distance-Forward under both training recipes. The hardened recipe
adds stronger augmentation and regularization; on Tiny ImageNet it also changes
the input resolution from 32 × 32 to 64 × 64.

**LCE is the paper's display name for `nokland_lpredsim`.** The implementation
uses the original `0.99 * CE + 0.01 * similarity_loss` objective. See the
[ResNet README](02_cifar10_resnet_image_classification/README.md) for the complete
algorithm mapping, optimizer settings, and evaluation protocol.

Run from the ResNet experiment directory:

```bash
cd 02_cifar10_resnet_image_classification

python train.py --dataset cifar10 --arch resnet18 --algo bp \
  --data-dir "<dataset-directory>" --output "<result-file.json>"

python train.py --dataset cifar100 --arch resnet56 --algo nokland_lpredsim \
  --recipe hardened \
  --data-dir "<dataset-directory>" --output "<result-file.json>"
```

Replace the angle-bracket placeholders with your dataset directory and a separate
result filename for each configuration. Each result contains native and readout
accuracy curves, best/final accuracies, and the training configuration.

Aggregate result files to compare accuracy, readout-minus-native differences, and
baseline-to-hardened gains:

```bash
python aggregate_results.py "<baseline-result.json>" "<hardened-result.json>"
python aggregate_results.py --metric final "<baseline-result.json>" "<hardened-result.json>"
```

The existing `plot_cifar10_resnet_bench.py` renders Figure 2 from the reported
values embedded in that script. `train.py` and `aggregate_results.py` operate on
new training runs.

## Other experiment entry points

| Experiment | Main scripts or package |
| --- | --- |
| Toy convergence | `01_toy_multilayer_convergence/exp2_multi_layer.py` |
| Unified CNN benchmark | `03_cifar10_unified_benchmark/run_benchmark.py` and `aggregate_results.py` |
| Spectral intervention | `04_cifar10_spectral_intervention/rank_control_intervention.py` and `jacobian_concentration/` |
| Locality intervention | `05_cnn_locality_intervention/locality_block_sweep.py` and `aggregate_locality_block_sweep.py` |
| Transformer scaling | `local_learning_nanogpt.experiments.cli` in `06_owt_transformer_scaling/` |
| Rank trajectories | `07_cifar10_rank_trajectories/train_bp_ffa_sigma_gamma_matched_trajectory.py` and the accompanying renderers |
| CNN/ViT comparison | `08_cifar10_cnn_vit/cnn_vit_ffa_vs_bp.py` and `b_r3a_cnn_vit.py` |

Use each experiment's README for its training and evaluation protocol. The
CNN/ViT directory contains candidate historical implementations; their exact
correspondence to the final Table 4 protocol remains to be verified.

## Data

Image-classification experiments use torchvision datasets. The ResNet loader
downloads MNIST and CIFAR datasets if needed. Tiny ImageNet must be supplied in
its original extracted layout, including the validation annotations. OpenWebText
preparation utilities are provided in the Transformer package's `data_prep/`
directory. Dataset and result locations are supplied by the reader.

## Citation

```bibtex
@misc{wu2026pricelocality,
  title         = {The Price of Locality: Why Forward-Forward Underperforms Backpropagation?},
  author        = {Zhaoxian Wu and Haichuan Liu and Tianyi Chen},
  year          = {2026},
  eprint        = {2609.33240},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.33240}
}
```
