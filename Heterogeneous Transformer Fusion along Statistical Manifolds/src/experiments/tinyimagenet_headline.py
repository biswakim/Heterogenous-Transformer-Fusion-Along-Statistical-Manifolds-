"""
src/experiments/tinyimagenet_headline.py

TINY-IMAGENET REPLICATION of the two headline CIFAR-100 findings.

Deliberately NOT a full tier structure. A second dataset needs to
replicate the claims the paper actually makes, not re-run the whole
protocol -- so this covers exactly two things:

  1. ZERO-SHOT RETENTION: does alignment-preserving fusion retain a large
     fraction of parent accuracy with no fine-tuning, while naive
     averaging collapses to near chance?

  2. MECHANISM GRID (B1 x C1): is the retention attributable to the
     Bures-Wasserstein interpolation rather than to permutation
     alignment, as the CIFAR-100 results indicate?

Skipped on purpose: depth-gap sweep, hyperparameter ablations, the full
baseline table. Those established the claims; this tests whether they
hold on a second, harder dataset.

ARCHITECTURE NOTE
------------------------------------------------------------------------
Tiny-ImageNet is 64x64, so patch_size is raised from the CIFAR setting
of 2 to 8. Attention is O(n^2) in patch count: patch_size=2 at 64px
would give 1024 patches versus CIFAR's 256, roughly 16x the attention
cost. patch_size=8 gives 64 patches (cheaper than CIFAR); patch_size=4
gives 256, matching CIFAR's sequence length exactly. Default here is 8
for speed; pass --patch-size 4 if you have the compute and want the
sequence length held constant across datasets.

EVALUATION NOTE
------------------------------------------------------------------------
Tiny-ImageNet's test split is unlabelled, so all "test" numbers here are
computed on the labelled VALIDATION split (10,000 images, 50/class).
Report it as such.

Usage:
    python -m src.experiments.tinyimagenet_headline --seeds 2
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os

import torch
import torch.nn as nn

from src.data.tinyimagenet import get_tinyimagenet_dataloaders
from src.models.vitparents import build_vit_parent
from src.utils.seeding import set_seed, derive_seed, seeded_calibration_batch
from src.utils.registry import log_run, already_logged, query
from src.stats.analysis import format_mean_std, paired_comparison, summarize_family
from src.fusion.pipeline import MSOTFusionPipeline
from src.baselines.naive_averaging import fuse_naive_averaging
from src.experiments.tier1depthsweep import train_parent, evaluate, checkpoint_is_complete

TIER = "tinyimagenet"
DATASET = "tinyimagenet"
ARCH_PAIR = "A_vs_B-12_tin"

SYNTHESIS = ["bures", "euclidean", "identity_pad"]
ALIGNMENT = ["sinkhorn", "hungarian", "none"]


def tin_parent(config_id, num_classes=200, img_size=64, patch_size=8):
    return build_vit_parent(config_id, num_classes=num_classes,
                            img_size=img_size, patch_size=patch_size)


def get_or_train_tin_parent(config_id, seed, train_loader, test_loader, device,
                            epochs, patch_size=8):
    """Same completion-marker logic as the CIFAR runs, with a tin_ prefix so
    Tiny-ImageNet checkpoints never collide with CIFAR-100 ones."""
    name = f"tin_Parent_{config_id}_seed{seed}"
    set_seed(seed)
    model = tin_parent(config_id, patch_size=patch_size).to(device)

    if checkpoint_is_complete(name):
        print(f"  Loading completed checkpoint for {name}")
        model.load_state_dict(torch.load(f"checkpoints/{name}_best.pth", map_location=device))
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        _, acc = evaluate(model, test_loader, criterion, device)
        print(f"  {name} loaded, verified val acc {acc:.2f}%")
        return model, acc

    if os.path.exists(f"checkpoints/{name}_best.pth"):
        print(f"  {name} has a checkpoint but no completion marker (interrupted) -- retraining.")
    model, acc = train_parent(model, name, train_loader, test_loader, device, epochs=epochs)
    return model, acc


def run(n_seeds=2, epochs_a=60, epochs_b=100, patch_size=8, batch_size=128,
        registry_path="results/raw/runs.jsonl"):
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TINY-IMAGENET HEADLINE REPLICATION === device={device}")
    print(f"    patch_size={patch_size} -> {(64 // patch_size) ** 2} patches/image\n")

    train_loader, test_loader = get_tinyimagenet_dataloaders(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for ps in range(n_seeds):
        print(f"\n--- Parent seed {ps} ---")
        parent_A, acc_A = get_or_train_tin_parent("A", ps, train_loader, test_loader,
                                                  device, epochs_a, patch_size)
        parent_A.cpu(); gc.collect(); torch.cuda.empty_cache()
        parent_B, acc_B = get_or_train_tin_parent("B-12", ps, train_loader, test_loader,
                                                  device, epochs_b, patch_size)
        parent_A.to(device)
        better = max(acc_A, acc_B)

        # --- naive averaging: the collapse baseline, no calibration batch needed ---
        key = dict(tier=TIER, dataset=DATASET, arch_pair=ARCH_PAIR,
                   method="naive_averaging", parent_seed=ps, fusion_seed=0)
        if not already_logged(key, path=registry_path):
            child = fuse_naive_averaging(parent_A, parent_B).to(device)
            zs_loss, zs_acc = evaluate(child, test_loader, criterion, device)
            log_run({**key, "zeroshot_accuracy": zs_acc, "zeroshot_loss": zs_loss,
                     "parent_A_accuracy": acc_A, "parent_B_accuracy": acc_B,
                     "better_parent_accuracy": better, "patch_size": patch_size},
                    path=registry_path)
            print(f"  [naive_averaging          ] zs={zs_acc:5.2f}%  "
                  f"({100 * zs_acc / better:.1f}% of better parent)")
            del child; gc.collect(); torch.cuda.empty_cache()
        else:
            print("  [naive_averaging          ] (cached)")

        # --- the 3x3 mechanism grid ---
        for syn, ali in itertools.product(SYNTHESIS, ALIGNMENT):
            method_tag = f"mech_{syn}_{ali}"
            key = dict(tier=TIER, dataset=DATASET, arch_pair=ARCH_PAIR,
                       method=method_tag, parent_seed=ps, fusion_seed=0)
            if already_logged(key, path=registry_path):
                print(f"  [{syn:<13} {ali:<10}] (cached)")
                continue

            fusion_seed = derive_seed(TIER, syn, ali, ps)
            set_seed(fusion_seed)
            calib = seeded_calibration_batch(get_tinyimagenet_dataloaders, 32,
                                             fusion_seed, device)

            pipeline = MSOTFusionPipeline(parent_A, parent_B, alpha=0.4, beta=0.5,
                                          epsilon_fgw=0.02, epsilon_sink=0.02,
                                          synthesis=syn, alignment=ali, verbose=False)
            child = pipeline.fuse(calib).to(device)
            zs_loss, zs_acc = evaluate(child, test_loader, criterion, device)

            log_run({**key, "synthesis": syn, "alignment": ali,
                     "zeroshot_accuracy": zs_acc, "zeroshot_loss": zs_loss,
                     "parent_A_accuracy": acc_A, "parent_B_accuracy": acc_B,
                     "better_parent_accuracy": better, "patch_size": patch_size,
                     "n_split_slots": getattr(pipeline, "n_split_slots", None)},
                    path=registry_path)
            print(f"  [{syn:<13} {ali:<10}] zs={zs_acc:5.2f}%  "
                  f"({100 * zs_acc / better:.1f}% of better parent)")

            del child, pipeline; gc.collect(); torch.cuda.empty_cache()

        parent_A.cpu(); parent_B.cpu(); del parent_A, parent_B
        gc.collect(); torch.cuda.empty_cache()

    print_table(registry_path)


def print_table(registry_path="results/raw/runs.jsonl"):
    print(f"\n{'='*76}\nTINY-IMAGENET: ZERO-SHOT ACCURACY (validation split, no fine-tuning)\n{'='*76}")

    naive = [r["zeroshot_accuracy"] for r in
             query(path=registry_path, tier=TIER, method="naive_averaging")
             if "zeroshot_accuracy" in r]
    parents_a = [r["parent_A_accuracy"] for r in query(path=registry_path, tier=TIER)
                 if "parent_A_accuracy" in r]
    parents_b = [r["parent_B_accuracy"] for r in query(path=registry_path, tier=TIER)
                 if "parent_B_accuracy" in r]
    if parents_a:
        print(f"  Parent A (6L)      : {format_mean_std(sorted(set(parents_a)))}")
        print(f"  Parent B (12L)     : {format_mean_std(sorted(set(parents_b)))}")
    print(f"  naive_averaging    : {format_mean_std(naive) if naive else '--'}")

    corner = "synthesis \\ alignment"
    print(f"\n{corner:<24}" + "".join(f"{a:>17}" for a in ALIGNMENT))
    print("-" * 76)
    cells = {}
    for syn in SYNTHESIS:
        row = f"{syn:<24}"
        for ali in ALIGNMENT:
            vals = [r["zeroshot_accuracy"] for r in
                    query(path=registry_path, tier=TIER, method=f"mech_{syn}_{ali}")
                    if "zeroshot_accuracy" in r]
            cells[(syn, ali)] = vals
            row += f"{format_mean_std(vals) if vals else '--':>17}"
        print(row)

    print(f"\n{'='*76}\nKEY COMPARISONS\n{'='*76}")
    family = []
    bures = cells.get(("bures", "sinkhorn"), [])
    euclid = cells.get(("euclidean", "sinkhorn"), [])
    n = min(len(bures), len(euclid))
    if n >= 3:
        family.append(paired_comparison(bures[:n], euclid[:n], label="bures vs euclidean (interpolation)"))
    n = min(len(bures), len(naive))
    if n >= 3:
        family.append(paired_comparison(bures[:n], naive[:n], label="bures vs naive_averaging"))
    for syn in SYNTHESIS:
        a, b = cells.get((syn, "sinkhorn"), []), cells.get((syn, "none"), [])
        n = min(len(a), len(b))
        if n >= 3:
            family.append(paired_comparison(a[:n], b[:n], label=f"{syn}: alignment vs none"))
    if family:
        print(summarize_family(family))
    else:
        print("  (need >= 3 seeds for significance tests; with fewer, report descriptively)")


def main():
    ap = argparse.ArgumentParser(description="Tiny-ImageNet headline replication")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs-a", type=int, default=60)
    ap.add_argument("--epochs-b", type=int, default=100)
    ap.add_argument("--patch-size", type=int, default=8, choices=[2, 4, 8, 16])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--registry", type=str, default="results/raw/runs.jsonl")
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    if args.table_only:
        print_table(args.registry)
        return
    run(n_seeds=args.seeds, epochs_a=args.epochs_a, epochs_b=args.epochs_b,
        patch_size=args.patch_size, batch_size=args.batch_size,
        registry_path=args.registry)


if __name__ == "__main__":
    main()
