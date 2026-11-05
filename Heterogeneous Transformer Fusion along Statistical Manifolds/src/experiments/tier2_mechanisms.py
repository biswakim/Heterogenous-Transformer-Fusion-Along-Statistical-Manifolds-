"""
src/experiments/tier2_mechanisms.py

B1 x C1 MECHANISM ABLATION (protocol Sec. 8.2 and 8.3).

Runs the full 3x3 grid at the fixed 6-vs-12 anchor:

    B1 (meso-scale synthesis)   x   C1 (micro-scale alignment)
    -------------------------       ---------------------------
    bures        (proposed)         sinkhorn   (proposed)
    euclidean                       hungarian
    identity_pad                    none

WHY THE FULL GRID AND NOT TWO SEPARATE 1-D SWEEPS
------------------------------------------------------------------------
Because the two axes INTERACT, and the interaction is the whole point.
Bures-Wasserstein interpolation was verified to be invariant to row
permutation of the block being merged: it consumes that block only via
its mean and covariance, both permutation-invariant. Concretely,

    bures(W_ref, W_other)  ==  bures(W_ref, permute_rows(W_other))

holds to 0.00e+00. So under synthesis='bures' the ENTIRE C1 axis is a
no-op -- sinkhorn, hungarian and none must produce bit-identical models.
Euclidean interpolation does not have this property and does respond to
alignment.

That predicts a specific, falsifiable pattern in the results table:

    row 'bures':      all three C1 arms IDENTICAL  (sanity check --
                      if they differ, the invariance analysis is wrong)
    row 'euclidean':  C1 arms DIFFER, and the gap between 'none' and the
                      aligned arms measures what permutation alignment is
                      actually worth once the interpolation rule preserves it

This is the experiment that distinguishes "alignment does not help" from
"alignment was never being used". Those are very different claims and the
paper needs to know which one is true.

Usage:  python -m src.experiments.tier2_mechanisms
"""

from __future__ import annotations

import argparse
import gc
import itertools

import torch
import torch.nn as nn

from src.data.cifar import get_cifar100_dataloaders
from src.utils.seeding import set_seed, derive_seed, seeded_calibration_batch
from src.utils.registry import log_run, already_logged, query
from src.stats.analysis import format_mean_std, paired_comparison, summarize_family
from src.fusion.pipeline import MSOTFusionPipeline
from src.experiments.tier0_anchor_gate import fine_tune_with_history, get_or_train_parent
from src.experiments.tier1depthsweep import evaluate

TIER = "tier2mech"
DATASET = "cifar100"
ARCH_PAIR = "A_vs_B-12"

SYNTHESIS = ["bures", "euclidean", "identity_pad"]
ALIGNMENT = ["sinkhorn", "hungarian", "none"]


def run_mechanisms(n_seeds=3, ft_epochs=15, zeroshot_only=False,
                   registry_path="results/raw/runs.jsonl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "ZERO-SHOT ONLY" if zeroshot_only else "zero-shot + 15-epoch fine-tune"
    print(f"=== B1 x C1 MECHANISM ABLATION ({mode}) === device={device}\n")

    train_loader, test_loader = get_cifar100_dataloaders(batch_size=128)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for ps in range(n_seeds):
        print(f"\n--- Parent seed {ps} ---")
        parent_A, acc_A = get_or_train_parent("A", ps, train_loader, test_loader, device, 60)
        parent_A.cpu(); gc.collect(); torch.cuda.empty_cache()
        parent_B, acc_B = get_or_train_parent("B-12", ps, train_loader, test_loader, device, 100)
        parent_A.to(device)
        better = max(acc_A, acc_B)

        for syn, ali in itertools.product(SYNTHESIS, ALIGNMENT):
            method_tag = f"mech_{syn}_{ali}"
            key = dict(tier=TIER, dataset=DATASET, arch_pair=ARCH_PAIR,
                       method=method_tag, parent_seed=ps, fusion_seed=0)
            if already_logged(key, path=registry_path):
                prior = query(path=registry_path, **key)[0]
                print(f"  [{syn:<13} {ali:<10}] zs={prior.get('zeroshot_accuracy')}  (cached)")
                continue

            fusion_seed = derive_seed(TIER, syn, ali, ps)
            set_seed(fusion_seed)
            calib = seeded_calibration_batch(get_cifar100_dataloaders, 32, fusion_seed, device)

            pipeline = MSOTFusionPipeline(parent_A, parent_B, alpha=0.4, beta=0.5,
                                          epsilon_fgw=0.02, epsilon_sink=0.02,
                                          synthesis=syn, alignment=ali, verbose=False)
            child = pipeline.fuse(calib).to(device)

            # Zero-shot first -- this is where the alignment effect, if any,
            # should be most visible, before fine-tuning washes it out.
            zs_loss, zs_acc = evaluate(child, test_loader, criterion, device)

            record = {**key, "synthesis": syn, "alignment": ali,
                      "zeroshot_accuracy": zs_acc, "zeroshot_loss": zs_loss,
                      "parent_A_accuracy": acc_A, "parent_B_accuracy": acc_B,
                      "better_parent_accuracy": better,
                      "n_split_slots": getattr(pipeline, "n_split_slots", None)}

            if not zeroshot_only:
                res = fine_tune_with_history(child, train_loader, test_loader, device,
                                             epochs=ft_epochs, recovery_target=better)
                record.update({k: v for k, v in res.items() if k != "history"})
                record["history"] = res["history"]
                print(f"  [{syn:<13} {ali:<10}] zs={zs_acc:5.2f}%  final={res['final_accuracy']:5.2f}%")
            else:
                print(f"  [{syn:<13} {ali:<10}] zs={zs_acc:5.2f}%")

            log_run(record, path=registry_path)
            del child, pipeline
            gc.collect(); torch.cuda.empty_cache()

        parent_A.cpu(); parent_B.cpu(); del parent_A, parent_B
        gc.collect(); torch.cuda.empty_cache()

    print_mechanism_table(registry_path)


def print_mechanism_table(registry_path="results/raw/runs.jsonl", metric="zeroshot_accuracy"):
    label = "ZERO-SHOT" if metric == "zeroshot_accuracy" else "FINAL (15-epoch)"
    print(f"\n{'='*76}\nB1 x C1 MECHANISM GRID -- {label} ACCURACY\n{'='*76}")
    # NOTE: the header is built outside the f-string because Python < 3.12
    # forbids backslashes inside f-string expression parts (the container
    # runs 3.10), and this label contains one.
    corner = "synthesis \\ alignment"
    print(f"{corner:<24}" + "".join(f"{a:>17}" for a in ALIGNMENT))
    print("-" * 76)

    cells = {}
    for syn in SYNTHESIS:
        row = f"{syn:<24}"
        for ali in ALIGNMENT:
            vals = [r[metric] for r in
                    query(path=registry_path, tier=TIER, arch_pair=ARCH_PAIR,
                          method=f"mech_{syn}_{ali}") if metric in r]
            cells[(syn, ali)] = vals
            row += f"{format_mean_std(vals) if vals else '--':>17}"
        print(row)

    # The falsifiable prediction: under 'bures', the C1 arms must coincide.
    print(f"\n{'='*76}\nPERMUTATION-INVARIANCE CHECK\n{'='*76}")
    for syn in SYNTHESIS:
        base = cells.get((syn, "sinkhorn"), [])
        none = cells.get((syn, "none"), [])
        if len(base) >= 1 and len(none) >= 1 and len(base) == len(none):
            identical = all(abs(a - b) < 1e-9 for a, b in zip(base, none))
            verdict = ("C1 axis is a NO-OP (alignment discarded)" if identical
                       else "C1 axis has an effect (alignment preserved)")
            print(f"  synthesis='{syn}': sinkhorn vs none -> {verdict}")

    # Where alignment IS preserved, quantify what it buys.
    print(f"\n{'='*76}\nVALUE OF ALIGNMENT, where the interpolation rule preserves it\n{'='*76}")
    family = []
    for syn in SYNTHESIS:
        for aligned_arm in ("sinkhorn", "hungarian"):
            a, b = cells.get((syn, aligned_arm), []), cells.get((syn, "none"), [])
            n = min(len(a), len(b))
            if n >= 3:
                family.append(paired_comparison(a[:n], b[:n],
                              label=f"{syn}: {aligned_arm} vs none"))
    if family:
        print(summarize_family(family))
    else:
        print("  (need >= 3 seeds per cell for significance testing)")


def main():
    ap = argparse.ArgumentParser(description="B1 x C1 mechanism ablation")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ft-epochs", type=int, default=15)
    ap.add_argument("--zeroshot-only", action="store_true",
                    help="Skip fine-tuning. ~9x faster, and zero-shot is where "
                         "the alignment effect should be clearest.")
    ap.add_argument("--registry", type=str, default="results/raw/runs.jsonl")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--metric", type=str, default="zeroshot_accuracy",
                    choices=["zeroshot_accuracy", "final_accuracy"])
    args = ap.parse_args()

    if args.table_only:
        print_mechanism_table(args.registry, metric=args.metric)
        return
    run_mechanisms(n_seeds=args.seeds, ft_epochs=args.ft_epochs,
                   zeroshot_only=args.zeroshot_only, registry_path=args.registry)


if __name__ == "__main__":
    main()
