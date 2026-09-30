"""Baseline and hardened configurations used in the ResNet experiments."""


def training_config(dataset, arch="resnet18", recipe="baseline"):
    if recipe not in ("baseline", "hardened"):
        raise ValueError("Choose baseline or hardened.")
    if recipe == "hardened" and dataset not in ("cifar100", "tiny_imagenet"):
        raise ValueError("The hardened recipe is reported for CIFAR-100 and Tiny ImageNet.")
    config = dict(
        dataset=dataset, arch=arch, epochs=200, batch_size=128, seed=0,
        lr=1e-3, sgd_lr=0.1, weight_decay=5e-4, bp_scheduler="multistep",
        label_smoothing=0.0, readout_lr=1e-3, readout_weight_decay=1e-4,
        readout_label_smoothing=0.0, readout_dropout=0.0, readout_mode="all",
        augment=True, strong_augment=False, tiny_image_size=32,
        train_eval_batches=10, recipe_name=recipe,
    )
    if recipe == "hardened":
        config.update(
            strong_augment=True, label_smoothing=0.1,
            readout_label_smoothing=0.1, readout_dropout=0.2,
            weight_decay=1e-3, readout_weight_decay=1e-3,
            bp_scheduler="cosine",
            tiny_image_size=64 if dataset == "tiny_imagenet" else 32,
        )
    return config
