"""
src/experiments/width_mechanisms.py

B1 x C1 MECHANISM GRID ON A WIDTH-HETEROGENEOUS PAIR.

Every result reported so far uses parents of identical width
(d_model=128), differing only in depth. This script asks whether the
central finding -- that the interpolation rule accounts for the effect
and the alignment mechanism does not -- still holds when the parents
also differ in width.

Pair: Parent A (6 layers, d_model=128) vs Parent Bw-12 (12 layers,
d_model=224).

WHY THE MECHANISM GRID RATHER THAN A BASELINE TABLE
------------------------------------------------------------------------
On width-heterogeneous pairs the external baselines (naive averaging,
identity padding, CLAFusion-style DP) cannot run natively: they average
or permute weight matrices elementwise, which is undefined when the
shapes differ. Running them requires first projecting Parent A into
Parent B's width using this framework's own Phase-5 machinery, so any
such comparison is partly "baseline plus a component of our method" and
is correspondingly hard to interpret.

The mechanism grid has no such problem. It varies only the interpolation
rule and the alignment mechanism, both internal to the pipeline, with the
width projection held fixed across every cell. It therefore answers the
question the paper actually needs answered.

Zero-shot only by default: that is where the CIFAR-100 and Tiny-ImageNet
effects were clearest, and it avoids spending hours of fine-tuning on a
comparison that fine-tuning is known to compress.

Usage:
    python -m src.experiments.width_mechanisms --seeds 3
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os

import torch
import torch.nn as nn

from src.data.cifar import get_cifar100_dataloaders
from src.utils.seeding import set_seed, derive_seed, seeded_calibration_batch
from src.utils.registry import log_run, already_logged, query
from src.stats.analysis import format_mean_std, paired_comparison, summarize_family
from src.fusion.pipelinewidth import MSOTWidthPipeline
from src.experiments.tier0_anchor_gate import get_or_train_parent
from src.experiments.tier1depthsweep import evaluate

TIER = "widthmech"
DATASET = "cifar100"
ARCH_PAIR = "A_128d_vs_Bw-12_224d"

SYNTHESIS = ["bures", "euclidean", "identity_pad"]
ALIGNMENT = ["sinkhorn", "hungarian", "none"]


def run(n_seeds=3, registry_path="results/raw/runs.jsonl"):
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== WIDTH-HETEROGENEOUS MECHANISM GRID === device={device}")
    print("    Parent A: 6L, d_model=128   |   Parent Bw-12: 12L, d_model=224\n")

    train_loader, test_loader = get_cifar100_dataloaders(batch_size=128)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for ps in range(n_seeds):
        print(f"\n--- Parent seed {ps} ---")
        parent_A, acc_A = get_or_train_parent("A", ps, train_loader, test_loader, device, 60)
        parent_A.cpu(); gc.collect(); torch.cuda.empty_cache()
        parent_B, acc_B = get_or_train_parent("Bw-12", ps, train_loader, test_loader, device, 100)
        parent_A.to(device)
        better = max(acc_A, acc_B)
        print(f"  parents: A={acc_A:.2f}%  Bw-12={acc_B:.2f}%")

        for syn, ali in itertools.product(SYNTHESIS, ALIGNMENT):
            key = dict(tier=TIER, dataset=DATASET, arch_pair=ARCH_PAIR,
                       method=f"wmech_{syn}_{ali}", parent_seed=ps, fusion_seed=0)
            if already_logged(key, path=registry_path):
                prior = query(path=registry_path, **key)[0]
                print(f"  [{syn:<13} {ali:<10}] zs={prior.get('zeroshot_accuracy')}  (cached)")
                continue

            fusion_seed = derive_seed(TIER, syn, ali, ps)
            set_seed(fusion_seed)
            calib = seeded_calibration_batch(get_cifar100_dataloaders, 32, fusion_seed, device)

            pipeline = MSOTWidthPipeline(
                parent_A, parent_B, alpha=0.4, beta=0.5,
                epsilon_fgw=0.02, epsilon_sink=0.02,
                width_alignment_method="zero_pad",
                synthesis=syn, alignment=ali, verbose=False,
            )
            child = pipeline.fuse(calib).to(device)
            zs_loss, zs_acc = evaluate(child, test_loader, criterion, device)

            log_run({**key, "synthesis": syn, "alignment": ali,
                     "zeroshot_accuracy": zs_acc, "zeroshot_loss": zs_loss,
                     "parent_A_accuracy": acc_A, "parent_B_accuracy": acc_B,
                     "better_parent_accuracy": better,
                     "n_split_slots": getattr(pipeline, "n_split_slots", None)},
                    path=registry_path)
            print(f"  [{syn:<13} {ali:<10}] zs={zs_acc:5.2f}%  "
                  f"({100 * zs_acc / better:.1f}% of better parent)")

            del child, pipeline
            gc.collect(); torch.cuda.empty_cache()

        parent_A.cpu(); parent_B.cpu(); del parent_A, parent_B
        gc.collect(); torch.cuda.empty_cache()

    print_table(registry_path)


def print_table(registry_path="results/raw/runs.jsonl"):
    print(f"\n{'='*76}\nWIDTH-HETEROGENEOUS MECHANISM GRID -- ZERO-SHOT ACCURACY\n{'='*76}")
    corner = "synthesis \\ alignment"
    print(f"{corner:<24}" + "".join(f"{a:>17}" for a in ALIGNMENT))
    print("-" * 76)

    cells = {}
    for syn in SYNTHESIS:
        row = f"{syn:<24}"
        for ali in ALIGNMENT:
            vals = [r["zeroshot_accuracy"] for r in
                    query(path=registry_path, tier=TIER, method=f"wmech_{syn}_{ali}")
                    if "zeroshot_accuracy" in r]
            cells[(syn, ali)] = vals
            row += f"{format_mean_std(vals) if vals else '--':>17}"
        print(row)

    print(f"\n{'='*76}\nKEY COMPARISONS\n{'='*76}")
    family = []
    for ali in ALIGNMENT:
        b, e = cells.get(("bures", ali), []), cells.get(("euclidean", ali), [])
        n = min(len(b), len(e))
        if n >= 3:
            family.append(paired_comparison(b[:n], e[:n],
                          label=f"bures vs euclidean ({ali} arm)"))
    for syn in SYNTHESIS:
        a, no = cells.get((syn, "sinkhorn"), []), cells.get((syn, "none"), [])
        n = min(len(a), len(no))
        if n >= 3:
            family.append(paired_comparison(a[:n], no[:n],
                          label=f"{syn}: sinkhorn vs none"))
    if family:
        print(summarize_family(family))
    else:
        print("  (need >= 3 seeds per cell for significance tests)")


def main():
    ap = argparse.ArgumentParser(description="Width-heterogeneous mechanism grid")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--registry", type=str, default="results/raw/runs.jsonl")
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    if args.table_only:
        print_table(args.registry)
        return
    run(n_seeds=args.seeds, registry_path=args.registry)


if __name__ == "__main__":
    main()
