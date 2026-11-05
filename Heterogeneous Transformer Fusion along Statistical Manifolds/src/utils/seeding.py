"""
src/utils/seeding.py

Reproducible seeding, per protocol Sec. 5.4 ("Seeds").

Two DISTINCT kinds of seed exist in this program and they must not be
conflated:

  - PARENT SEED: controls parent weight initialization + training data
    order. Different parent seeds = genuinely different trained parents.
    Protocol: 5 on CIFAR-100, 3 on GLUE/MIMIC, 2 on ImageNet-1K.

  - FUSION-INTERNAL SEED: controls only the randomness INSIDE a fusion
    run given fixed parents -- i.e. which calibration batch is drawn, and
    any stochastic initialization inside the transport solvers. Protocol
    Sec. 9.1 asks for 3 of these per parent-seed pair.

Reporting is always mean +/- std over the seed count specified for that
tier (Sec. 11, point 1), which is why every runner loops over both.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """
    Seed python, numpy, and torch (CPU + all CUDA devices).

    `deterministic=True` additionally forces cuDNN into deterministic mode.
    Left OFF by default because it materially slows training and the
    protocol's variance analysis is over seeds, not over bitwise-identical
    reruns -- turn it on only if you need exact reproduction of a single
    run for debugging.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.benchmark = True


def derive_seed(*parts) -> int:
    """
    Deterministically derive a sub-seed from arbitrary components, e.g.
    derive_seed("tier1", "A_vs_B-12", "msot", parent_seed, fusion_seed).

    Using a derived seed rather than reusing the raw parent seed for the
    calibration batch avoids accidentally correlating "parent seed 3" with
    "always the same calibration batch", which would make the 3
    fusion-internal repeats less independent than intended.
    """
    h = 0
    for p in parts:
        s = str(p)
        for ch in s:
            h = (h * 31 + ord(ch)) % (2 ** 31 - 1)
        h = (h * 131 + 7) % (2 ** 31 - 1)
    return h


def seeded_calibration_batch(dataloader_fn, batch_size: int, seed: int, device):
    """
    Draw a calibration batch (protocol Sec. 5.1: 32 images) in a way that
    actually depends on `seed`.

    NOTE: the existing experiment scripts do
        calib_loader, _ = get_cifar100_dataloaders(batch_size=32)
        calibration_batch, _ = next(iter(calib_loader))
    which returns the SAME batch every time if the loader isn't shuffled,
    or an unseeded-random one if it is -- either way the 3 "fusion-internal
    seeds" the protocol asks for would not actually differ. This function
    reseeds immediately before constructing the loader so the drawn batch
    is a real function of the fusion seed.

    FIXED (file-descriptor leak): this used to call
    `dataloader_fn(batch_size=batch_size)`, inheriting the default
    num_workers=4 + persistent_workers=True. Because a NEW loader is built
    on every fusion run and abandoned after a single batch, each call
    leaked 4 worker processes that were never joined -- after roughly a
    hundred fusion runs the process exhausted its file-descriptor limit and
    died with "OSError: [Errno 24] Too many open files" mid-sweep.

    num_workers=0 is now forced: the calibration batch is a single batch of
    32 images loaded exactly once, so worker processes buy nothing here and
    cost a leak. The iterator and loader are also explicitly dropped before
    returning, so nothing lingers even if the caller holds the tensor.
    """
    set_seed(seed)
    loader, _ = dataloader_fn(batch_size=batch_size, num_workers=0)
    it = iter(loader)
    images, _ = next(it)
    images = images.to(device)
    del it, loader
    return images
