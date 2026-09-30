"""Dataset preprocessing used by the ResNet experiments."""
from pathlib import Path
from typing import Dict
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

DATASETS = ("mnist", "cifar10", "cifar100", "tiny_imagenet")
NORMALIZE = {
    "mnist": ((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tiny_imagenet": ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
}

class TinyImageNetVal(torch.utils.data.Dataset):
    """Tiny ImageNet validation split using val_annotations.txt labels."""

    def __init__(self, root: Path, class_to_idx: Dict[str, int], transform=None):
        self.root = root
        self.transform = transform
        ann_path = root / "val" / "val_annotations.txt"
        image_dir = root / "val" / "images"
        if not ann_path.exists():
            raise FileNotFoundError(f"Missing Tiny ImageNet validation annotations: {ann_path}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing Tiny ImageNet validation images: {image_dir}")

        samples = []
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                image_name, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    continue
                samples.append((image_dir / image_name, class_to_idx[wnid]))
        if not samples:
            raise RuntimeError(f"No Tiny ImageNet validation samples found in {ann_path}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def _tiny_imagenet_root(data_dir: str) -> Path:
    root = Path(data_dir)
    candidates = [
        root / "tiny-imagenet-200",
        root / "tiny_imagenet",
        root / "TinyImageNet",
    ]
    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "val").exists():
            return candidate
    raise FileNotFoundError(
        "Tiny ImageNet data not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def make_loaders(
    dataset: str,
    data_dir: str,
    batch_size: int = 128,
    augment: bool = True,
    num_workers: int = 2,
    tiny_image_size: int = 32,
    strong_augment: bool = False,
):
    """Return training and test loaders with the experimental transforms."""
    assert dataset in DATASETS
    mean, std = NORMALIZE[dataset]

    if dataset == "mnist":
        train_tf = T.Compose([
            T.Resize(32),
            T.Lambda(lambda img: img.convert("RGB")),
            T.ToTensor(),
            T.Normalize(*NORMALIZE["mnist"]),
        ])
        test_tf = train_tf
    elif dataset == "tiny_imagenet":
        image_size = int(tiny_image_size)
        if augment:
            train_ops = [
                T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
            ]
            if strong_augment:
                train_ops.append(T.RandAugment(num_ops=2, magnitude=9))
            train_ops.extend([
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
            if strong_augment:
                train_ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
            train_tf = T.Compose(train_ops)
        else:
            train_tf = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean, std),
            ])
        test_tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    elif augment:
        train_ops = [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
        ]
        if strong_augment:
            train_ops.append(T.RandAugment(num_ops=2, magnitude=9))
        train_ops.extend([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        if strong_augment:
            train_ops.append(T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)))
        train_tf = T.Compose(train_ops)
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    else:
        train_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    if dataset == "mnist":
        cls = torchvision.datasets.MNIST
        train_set = cls(root=data_dir, train=True,  download=True, transform=train_tf)
        test_set  = cls(root=data_dir, train=False, download=True, transform=test_tf)
    elif dataset == "tiny_imagenet":
        tiny_root = _tiny_imagenet_root(data_dir)
        train_set = torchvision.datasets.ImageFolder(tiny_root / "train", transform=train_tf)
        test_set = TinyImageNetVal(tiny_root, train_set.class_to_idx, transform=test_tf)
    else:
        cls = (torchvision.datasets.CIFAR10 if dataset == "cifar10"
               else torchvision.datasets.CIFAR100)
        train_set = cls(root=data_dir, train=True,  download=True, transform=train_tf)
        test_set  = cls(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
