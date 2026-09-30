"""Train a native ResNet method together with a detached linear readout."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from algos import TRAINERS, build_trainer
from algos.common import NUM_CLASSES
from data import DATASETS, make_loaders
from models import STAGE_DEPTHS
from readout import LinearReadout, train_readout_epoch, evaluate_readout
from recipes import training_config


def run_one(algorithm, config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    train_loader, test_loader = make_loaders(
        dataset=config["dataset"], data_dir=config["data_dir"],
        batch_size=config["batch_size"], augment=config["augment"],
        tiny_image_size=config["tiny_image_size"],
        strong_augment=config["strong_augment"],
    )
    trainer = build_trainer(algorithm, config)
    readout = LinearReadout(
        config["readout_mode"], NUM_CLASSES[config["dataset"]],
        dropout=config["readout_dropout"],
    ).to(config["device"])
    optimizer = torch.optim.AdamW(
        readout.parameters(), lr=config["readout_lr"],
        weight_decay=config["readout_weight_decay"],
    )
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=config["readout_label_smoothing"],
    )
    native_curve, readout_curve, loss_curve = [], [], []
    for epoch in range(config["epochs"]):
        native_train = trainer.train_epoch(train_loader, epoch)
        train_readout_epoch(
            trainer, readout, optimizer, train_loader, config["device"], criterion,
        )
        native_test = trainer.evaluate_native(test_loader)
        # Preserve this pass: shuffled/augmented data consume random numbers,
        # affecting the subsequent test pass and the next training epoch.
        evaluate_readout(
            trainer, readout, train_loader, config["device"], criterion,
            max_batches=config["train_eval_batches"],
        )
        readout_test = evaluate_readout(
            trainer, readout, test_loader, config["device"], criterion,
        )
        native_curve.append(native_test["acc"])
        readout_curve.append(readout_test["acc"])
        loss_curve.append(native_train.loss)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: native={native_test['acc']:.4f} "
                  f"readout={readout_test['acc']:.4f}", flush=True)

    return dict(
        algo=algorithm,
        config={key: value for key, value in config.items()
                if key not in ("data_dir", "device")},
        native_test_acc_best=max(native_curve),
        native_test_acc_final=native_curve[-1],
        readout_test_acc_best=max(readout_curve),
        readout_test_acc_final=readout_curve[-1],
        native_train_loss_curve=loss_curve,
        native_test_acc_curve=native_curve,
        readout_test_acc_curve=readout_curve,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="cifar10")
    parser.add_argument("--arch", choices=STAGE_DEPTHS, default="resnet18")
    parser.add_argument("--algo", choices=TRAINERS, default="bp")
    parser.add_argument("--recipe", choices=("baseline", "hardened"), default="baseline")
    parser.add_argument("--data-dir", required=True, help="Reader-supplied dataset directory")
    parser.add_argument("--output", required=True, help="Result JSON file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    config = training_config(args.dataset, args.arch, args.recipe)
    config.update(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                  device=args.device, data_dir=args.data_dir)
    result = run_one(args.algo, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
