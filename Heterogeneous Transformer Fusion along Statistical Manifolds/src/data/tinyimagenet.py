"""
src/data/tinyimagenet.py

Tiny-ImageNet-200 dataloaders (64x64, 200 classes, 100k train images).

Mirrors the interface of src/data/cifar.py exactly:
    get_tinyimagenet_dataloaders(batch_size=128) -> (train_loader, test_loader)
so the same experiment runners work with a one-line loader swap.

TWO THINGS ABOUT THIS DATASET THAT ARE EASY TO GET WRONG
------------------------------------------------------------------------
1. THE TEST SET HAS NO LABELS. tiny-imagenet-200/test/ ships without
   annotations, so it cannot be used for evaluation. Standard practice --
   and what this loader does -- is to use the VALIDATION set (10,000
   labelled images, 50 per class) as the test set. The paper must say
   "evaluated on the Tiny-ImageNet validation set", not "test set".

2. THE VALIDATION SET IS NOT IN CLASS FOLDERS. It ships as a flat
   val/images/ directory plus val_annotations.txt mapping filename ->
   wnid. ImageFolder cannot read that layout, so this module reorganises
   it into val/<wnid>/*.JPEG on first use. The reorganisation is done once
   and detected on subsequent runs.

Class ordering is taken from sorted wnid, identically for train and val,
so label indices match across the two splits.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DIRNAME = "tiny-imagenet-200"

# Tiny-ImageNet channel statistics (not the same as ImageNet-1K's).
TIN_MEAN = (0.4802, 0.4481, 0.3975)
TIN_STD = (0.2770, 0.2691, 0.2821)


def _download_and_extract(data_root: str):
    target = os.path.join(data_root, DIRNAME)
    if os.path.isdir(target):
        return target

    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading Tiny-ImageNet (~248MB) to {zip_path} ...")
        urllib.request.urlretrieve(URL, zip_path)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)
    print(f"Extracted to {target}")
    return target


def _reorganize_val(root: str):
    """
    Convert val/images/*.JPEG + val_annotations.txt into val/<wnid>/*.JPEG
    so torchvision's ImageFolder can read it. Idempotent: if val/images no
    longer exists, the reorganisation already happened.
    """
    val_dir = os.path.join(root, "val")
    img_dir = os.path.join(val_dir, "images")
    ann_path = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.isdir(img_dir):
        return  # already reorganised

    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"Expected {ann_path}. The archive layout may differ from the "
            "standard cs231n release; inspect the val/ directory manually."
        )

    print("Reorganising Tiny-ImageNet validation set into class folders (one-time) ...")
    mapping = {}
    with open(ann_path) as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]

    for fname, wnid in mapping.items():
        cls_dir = os.path.join(val_dir, wnid)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(img_dir, fname)
        if os.path.exists(src):
            shutil.move(src, os.path.join(cls_dir, fname))

    # Drop the now-empty images/ dir so the check above short-circuits next time.
    try:
        os.rmdir(img_dir)
    except OSError:
        print(f"  note: {img_dir} not empty after reorganisation; leaving it in place.")
    print(f"  done: {len(mapping)} validation images placed into class folders.")


def _build_transforms(img_size: int = 64):
    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=img_size // 8),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(TIN_MEAN, TIN_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TIN_MEAN, TIN_STD),
    ])
    return train_transform, test_transform


def get_tinyimagenet_dataloaders(batch_size: int = 128, num_workers: int = 4,
                                 data_root: str = "./data", download: bool = True,
                                 pin_memory: bool = True, img_size: int = 64):
    """
    Returns (train_loader, test_loader), where test_loader is built from the
    labelled VALIDATION split (see module docstring).

    Interface note: `num_workers=0` must be honoured, because
    src/utils/seeding.seeded_calibration_batch relies on it to avoid the
    persistent-worker file-descriptor leak that previously killed a sweep.
    """
    root = _download_and_extract(data_root) if download else os.path.join(data_root, DIRNAME)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"{root} not found. Run with download=True, or place the extracted "
            "tiny-imagenet-200 directory there."
        )
    _reorganize_val(root)

    train_transform, test_transform = _build_transforms(img_size)

    train_set = datasets.ImageFolder(os.path.join(root, "train"), transform=train_transform)
    test_set = datasets.ImageFolder(os.path.join(root, "val"), transform=test_transform)

    if train_set.classes != test_set.classes:
        raise RuntimeError(
            "Train and validation class orderings differ; label indices would not "
            "correspond. This should not happen with the standard release."
        )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader
