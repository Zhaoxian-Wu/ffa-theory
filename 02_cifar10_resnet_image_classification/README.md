# ResNet image classification and detached readout

**Paper outputs:** Figure 1 (right), Figure 2, Appendix Figures 4–5, and Tables 7–10.

This module contains the nine native training methods and the detached linear
readout used to compare their representations across ResNet depths and datasets.
The code preserves the experimental implementation, including its method-specific
losses and optimizer settings.

## Reading the code

| File | Contents |
| --- | --- |
| `models.py` | CIFAR-style ResNet18/24/56/108 and the label embedding channel. |
| `algos/bp.py` | End-to-end cross-entropy training. |
| `algos/local_supervised.py` | Nøkland, SFF, and Distance-Forward stage-local updates. |
| `algos/vanilla_ffa.py` | Positive/negative goodness loss and native label enumeration. |
| `algos/scff.py` | Permuted-image negative pairs. |
| `algos/symba.py` | Symmetric goodness-margin loss. |
| `algos/layer_collab.py` | Detached accumulated goodness across stages. |
| `algos/trifecta.py` | Alternating odd/even stage updates. |
| `algos/common.py` | Local classifier heads, similarity loss, and evaluation. |
| `readout.py` | Feature pooling, detached readout training, and evaluation. |
| `data.py` | Dataset loading and the original preprocessing operations. |
| `recipes.py` | Baseline and hardened training configurations. |
| `train.py` | Native training and readout training in their original epoch order. |
| `aggregate_results.py` | Best/final accuracy, native/readout differences, and hardening gains. |

## Architectures and readout

| Architecture | BasicBlocks in the four residual stages |
| --- | --- |
| ResNet18 | `[2, 2, 2, 2]` |
| ResNet24 | `[3, 3, 3, 2]` |
| ResNet56 | `[7, 7, 7, 6]` |
| ResNet108 | `[13, 13, 13, 14]` |

The five local training units are the stem and the four residual stages. A local
loss differentiates all convolutions inside its stage; gradients stop between
stages. Their channel counts are `[64, 64, 128, 256, 512]`.

The paper uses `readout_mode="all"`: global-average-pool each of the five outputs,
concatenate the resulting 1024 features, and apply a linear classifier. Features
are detached and the backbone is in evaluation mode while training the readout.
Label-injection methods use the mean embedding channel for readout features;
ground-truth labels are not supplied to feature extraction. Native evaluation
instead enumerates all candidate labels and scores their goodness.

ResNet18 retains its original torchvision construction. The larger models retain
their original explicit BasicBlock construction, including their initialization.

## Algorithm names

| Paper label | `--algo` | Implemented objective or update |
| --- | --- | --- |
| BP | `bp` | End-to-end cross-entropy. |
| LCE | `nokland_lpredsim` | `0.99 * CE + 0.01 * similarity_loss`. |
| SFF | `sff` | Local CE with auxiliary convolutional heads; native logits average all heads. |
| DF | `distance_forward` | CE over scaled cosine similarities to class prototypes. |
| Vanilla FFA | `vanilla_ffa` | Positive/negative softplus goodness with a detached batch threshold. |
| SCFF | `scff` | Goodness training using permuted-image negative pairs. |
| SymBa | `symba` | Softplus of the negative positive-minus-negative goodness margin. |
| LayerCollab | `layer_collab` | Goodness loss with accumulated detached earlier-stage goodness. |
| Trifecta | `trifecta` | SymBa loss with alternating odd/even stage updates. |

**LCE is the paper's display name for the existing Nøkland implementation.** Its
similarity term is retained. Local optimizers operate on whole stages, not on
individual convolutions.

The label-injection implementation detaches its input before the first stage as
well as at later boundaries. Consequently the embedding table receives no loss
gradient, even though it is included in the first optimizer. This behavior is
retained. SCFF also retains the original permutation and fixed-point repair rule;
that rule can leave a fixed point when only one fixed point is present.

## Training recipes

All paper runs in this module use seed 0, 200 epochs, and batch size 128. MNIST
and CIFAR-10 use all nine algorithms. CIFAR-100 and Tiny ImageNet use BP,
Nøkland/LCE, SFF, and DF under both recipes.

| Setting | Baseline | Hardened |
| --- | --- | --- |
| BP optimizer | SGD, lr 0.1, Nesterov momentum 0.9 | Same |
| BP weight decay | 0.0005 | 0.001 |
| BP learning-rate schedule | ×0.1 at epochs 100 and 150 | Cosine |
| Local optimizer | Adam per stage, lr 0.001, no weight decay | Same |
| Local learning-rate schedule | ×0.1 at epoch 100 | Same |
| Native CE label smoothing | 0 | 0.1 |
| Readout optimizer | AdamW, lr 0.001 | Same |
| Readout weight decay | 0.0001 | 0.001 |
| Readout label smoothing / dropout | 0 / 0 | 0.1 / 0.2 |
| Additional augmentation | None | RandAugment and RandomErasing |
| Tiny ImageNet image size | 32 × 32 | 64 × 64 |

The hardened recipe applies to CIFAR-100 and Tiny ImageNet. The `weight_decay`
setting affects BP; the local Adam optimizers retain their original settings.
For Tiny ImageNet, hardening changes resolution together with regularization.

Each epoch preserves this sequence:

1. Train the native algorithm over the training loader.
2. Train the detached readout over the training loader.
3. Evaluate native test accuracy.
4. Evaluate the readout on the first ten training batches.
5. Evaluate readout test accuracy.

The fourth pass is retained because iterating shuffled, augmented data changes
the random-number state used by later epochs.

## Usage

From this experiment directory, supply your own dataset directory and result file:

```bash
python train.py --dataset cifar10 --arch resnet18 --algo bp \
  --data-dir "<dataset-directory>" --output "<result-file.json>"

python train.py --dataset cifar10 --arch resnet56 --algo symba \
  --data-dir "<dataset-directory>" --output "<result-file.json>"

python train.py --dataset cifar100 --arch resnet108 --algo nokland_lpredsim \
  --recipe hardened --data-dir "<dataset-directory>" --output "<result-file.json>"

python train.py --dataset tiny_imagenet --arch resnet24 --algo distance_forward \
  --recipe hardened --data-dir "<dataset-directory>" --output "<result-file.json>"
```

Replace angle-bracket placeholders with your own locations. MNIST and CIFAR
datasets are downloaded if needed. For Tiny ImageNet, provide the extracted
`tiny-imagenet-200` dataset with its original training directory and validation
annotations under the dataset directory.

The JSON result contains the training configuration, native/readout accuracy
curves, and best/final accuracies. Accuracy values in JSON are fractions; the
aggregation script reports percentages and percentage-point differences:

```bash
python aggregate_results.py "<baseline-result.json>" "<hardened-result.json>"
python aggregate_results.py --metric final "<baseline-result.json>" "<hardened-result.json>"
```

Paper figure/table mapping:

| Output | Data to use |
| --- | --- |
| Figure 1 (right) | ResNet18/CIFAR-10 accuracy trajectories. |
| Figure 2 | CIFAR-10 best readout accuracy, nine algorithms, ResNet18/24/56. |
| Figures 4–5 / Table 7 | BP and three supervised local methods across datasets, depths, and recipes. |
| Table 8 | Native versus readout best accuracy under the hardened recipe. |
| Table 9 | Five label-injection methods on MNIST and CIFAR-10. |
| Table 10 | Hardened minus baseline readout accuracy for matching configurations. |

This directory supplies the implementation and result aggregation. Historical
paper curves and result archives are separate from the code; running this module
generates new results rather than loading embedded paper numbers.
