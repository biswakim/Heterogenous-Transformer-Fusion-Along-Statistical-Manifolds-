"""
src/data/cifar.py

CIFAR-100 dataloaders matching the pilot report's spec exactly:

    "All experiments in this section use CIFAR-100 (100 classes, 32x32
     RGB images), split into the standard 50,000-image training set and
     10,000-image test set." (Sec 6.1)

    "Standard CIFAR-100 augmentation was applied during training: random
     32x32 crops with 4-pixel padding, random horizontal flip, random
     rotation (+-15 deg), and colour jitter (brightness/contrast +-0.2)."
     (Sec 6.2)

INTERFACE CONTRACT -- every other file in this codebase assumes this
exact shape, so changing it would break them all:
    get_cifar100_dataloaders(batch_size=128) -> (train_loader, test_loader)
The TRAIN loader is also used as the calibration-batch source
(src/utils/seeding.seeded_calibration_batch takes the FIRST returned
loader), so it must be shuffled -- an unshuffled loader would make every
"fusion seed" draw the identical calibration batch, silently defeating
the whole point of the multi-seed protocol.

If a different cifar.py already exists elsewhere on disk with a
DIFFERENT interface, use that one instead of this file and tell me its
signature so downstream code can be matched to it rather than the other
way around.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standard CIFAR-100 per-channel statistics.
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def _build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_transform, test_transform


def get_cifar100_dataloaders(batch_size: int = 128, num_workers: int = 4,
                             data_root: str = "./data", download: bool = True,
                             pin_memory: bool = True):
    """
    Returns (train_loader, test_loader).

    train_loader: shuffle=True (required both for normal training AND for
    calibration-batch sampling elsewhere in the codebase -- see module
    docstring), drop_last=True so every training batch is full-sized,
    which matters for the gradient-accumulation step size used in
    train_one_epoch (accum_steps=2 assumes a consistent batch size).

    test_loader: shuffle=False, drop_last=False -- evaluation must cover
    every one of the 10,000 test images exactly once, every time, so
    accuracy numbers are exactly comparable across runs and epochs.
    """
    train_transform, test_transform = _build_transforms()

    train_set = datasets.CIFAR100(root=data_root, train=True, download=download, transform=train_transform)
    test_set = datasets.CIFAR100(root=data_root, train=False, download=download, transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
