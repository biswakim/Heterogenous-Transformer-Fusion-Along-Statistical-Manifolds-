"""
src/experiments/large_parents.py

SCALE TEST: does the decomposition hold at larger parameter count?

Every result so far uses parents of 0.85M (6L) and 1.64M (12L)
parameters. The most predictable reviewer objection is that permutation
alignment may simply be a non-issue at that scale. This script repeats
the mechanism grid on parents roughly 13x larger, at the width used by
Imfeld et al. (2024) -- d_model=384, 12 heads, FFN=1536 -- so the
comparison is against the architecture of the prior work this paper
engages with.

    small (existing):  6L/12L, d_model=128, FFN=256   ->  0.85M / 1.64M
    large (this file): 6L/12L, d_model=384, FFN=1536  -> ~10.7M / 21.3M

DESIGN NOTE: the depth gap is held at 6-vs-12, identical to the anchor
used everywhere else in the paper. Parameter count is therefore the ONLY
variable that changes relative to the main results, which is what makes
"the finding holds across a 13x parameter range" a clean claim rather
than a confounded one. Matching Imfeld's depth (7 layers) instead would
have changed two things at once.

No edits to vitparents.py are required: build_vit_parent already accepts
**overrides, so the existing depth configs are reused with the width
fields overridden.

COMPUTE COST -- READ BEFORE LAUNCHING
------------------------------------------------------------------------
Width 384 vs 128 is 3x wider, and both attention and FFN cost scale
roughly quadratically in width, so expect ~9x the per-epoch time of the
existing CIFAR-100 runs. Three seeds means six parent trainings. Budget
in days, not hours. If that does not fit, --seeds 2 still supports the
claim descriptively, and --epochs-a/--epochs-b can be reduced: this
experiment tests whether the MECHANISM ranking survives at scale, not
whether these parents reach peak accuracy, so somewhat undertrained
parents are acceptable here in a way they would not be for a headline
accuracy claim. State the reduced schedule if you use one.

If you hit CUDA OOM, lower --batch-size to 64.

Usage:
    python -m src.experiments.large_parents --seeds 3
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os

import torch
import torch.nn as nn

from src.data.cifar import get_cifar100_dataloaders
from src.models.vitparents import build_vit_parent
from src.utils.seeding import set_seed, derive_seed, seeded_calibration_batch
from src.utils.registry import log_run, already_logged, query
from src.stats.analysis import format_mean_std, paired_comparison, summarize_family
from src.fusion.pipeline import MSOTFusionPipeline
from src.baselines.naive_averaging import fuse_naive_averaging
from src.experiments.tier1depthsweep import train_parent, evaluate, checkpoint_is_complete

TIER = "largeparents"
DATASET = "cifar100"
ARCH_PAIR = "LA_384d_vs_LB12_384d"

# Imfeld et al. (2024) width: d_model=384, 12 heads, FFN = 4*d_model.
LARGE = {"d_model": 384, "num_heads": 12, "ffn_dim": 1536}

SYNTHESIS = ["bures", "euclidean", "identity_pad"]
ALIGNMENT = ["sinkhorn", "hungarian", "none"]


def large_parent(config_id):
    return build_vit_parent(config_id, **LARGE)


def get_or_train_large(config_id, seed, train_loader, test_loader, device, epochs):
    """Per-seed checkpoints under an 'L_' prefix so they cannot collide with
    the small-parent or Tiny-ImageNet checkpoints."""
    name = f"L_Parent_{config_id}_seed{seed}"
    set_seed(seed)
    model = large_parent(config_id).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if checkpoint_is_complete(name):
        print(f"  Loading completed checkpoint for {name} ({n_params:,} params)")
        model.load_state_dict(torch.load(f"checkpoints/{name}_best.pth", map_location=device))
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        _, acc = evaluate(model, test_loader, criterion, device)
        print(f"  {name} loaded, verified test acc {acc:.2f}%")
        return model, acc, n_params

    if os.path.exists(f"checkpoints/{name}_best.pth"):
        print(f"  {name} has a checkpoint but no completion marker (interrupted) -- retraining.")
    print(f"  {name}: {n_params:,} parameters")
    model, acc = train_parent(model, name, train_loader, test_loader, device, epochs=epochs)
    return model, acc, n_params


def run(n_seeds=3, epochs_a=60, epochs_b=100, batch_size=128,
        registry_path="results/raw/runs.jsonl"):
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== LARGE-PARENT SCALE TEST === device={device}")
    print(f"    d_model={LARGE['d_model']}, heads={LARGE['num_heads']}, "
          f"FFN={LARGE['ffn_dim']}  (Imfeld et al. width)\n")

    train_loader, test_loader = get_cifar100_dataloaders(batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for ps in range(n_seeds):
        print(f"\n--- Parent seed {ps} ---")
        parent_A, acc_A, np_A = get_or_train_large("A", ps, train_loader, test_loader, device, epochs_a)
        parent_A.cpu(); gc.collect(); torch.cuda.empty_cache()
        parent_B, acc_B, np_B = get_or_train_large("B-12", ps, train_loader, test_loader, device, epochs_b)
        parent_A.to(device)
        better = max(acc_A, acc_B)
        print(f"  parents: A={acc_A:.2f}% ({np_A:,}p)  B-12={acc_B:.2f}% ({np_B:,}p)")

        # naive averaging: the collapse reference, no calibration batch needed
        key = dict(tier=TIER, dataset=DATASET, arch_pair=ARCH_PAIR,
                   method="naive_averaging", parent_seed=ps, fusion_seed=0)
        if not already_logged(key, path=registry_path):
            child = fuse_naive_averaging(parent_A, parent_B).to(device)
            zs_loss, zs_acc = evaluate(child, test_loader, criterion, device)
            log_run({**key, "zeroshot_accuracy": zs_acc, "zeroshot_loss": zs_loss,
                     "parent_A_accuracy": acc_A, "parent_B_accuracy": acc_B,
                     "better_parent_accuracy": better,
                     "n_params_A": np_A, "n_params_B": np_B}, path=registry_path)
            print(f"  [naive_averaging          ] zs={zs_acc:5.2f}%")
            del child; gc.collect(); torch.cuda.empty_cache()
        else:
            print("  [naive_averaging          ] (cached)")

        for syn, ali in itertools.product(SYNTHESIS, ALIGNMENT):
            key = dict(tier=TIER, dataset=DATASET, arch_pair=ARCH_PAIR,
                       method=f"lmech_{syn}_{ali}", parent_seed=ps, fusion_seed=0)
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
            zs_loss, zs_acc = evaluate(child, test_loader, criterion, device)

            log_run({**key, "synthesis": syn, "alignment": ali,
                     "zeroshot_accuracy": zs_acc, "zeroshot_loss": zs_loss,
                     "parent_A_accuracy": acc_A, "parent_B_accuracy": acc_B,
                     "better_parent_accuracy": better,
                     "n_params_A": np_A, "n_params_B": np_B,
                     "n_split_slots": getattr(pipeline, "n_split_slots", None)},
                    path=registry_path)
            print(f"  [{syn:<13} {ali:<10}] zs={zs_acc:5.2f}%  "
                  f"({100 * zs_acc / better:.1f}% of better parent)")

            del child, pipeline; gc.collect(); torch.cuda.empty_cache()

        parent_A.cpu(); parent_B.cpu(); del parent_A, parent_B
        gc.collect(); torch.cuda.empty_cache()

    print_table(registry_path)


def print_table(registry_path="results/raw/runs.jsonl"):
    print(f"\n{'='*76}\nLARGE PARENTS (d_model=384) -- ZERO-SHOT ACCURACY\n{'='*76}")

    pa = [r["parent_A_accuracy"] for r in query(path=registry_path, tier=TIER) if "parent_A_accuracy" in r]
    pb = [r["parent_B_accuracy"] for r in query(path=registry_path, tier=TIER) if "parent_B_accuracy" in r]
    nv = [r["zeroshot_accuracy"] for r in query(path=registry_path, tier=TIER, method="naive_averaging")
          if "zeroshot_accuracy" in r]
    npa = {r.get("n_params_A") for r in query(path=registry_path, tier=TIER) if r.get("n_params_A")}
    npb = {r.get("n_params_B") for r in query(path=registry_path, tier=TIER) if r.get("n_params_B")}
    if pa:
        print(f"  Parent A (6L) : {format_mean_std(sorted(set(pa)))}"
              f"{'  (' + f'{min(npa):,}' + ' params)' if npa else ''}")
        print(f"  Parent B (12L): {format_mean_std(sorted(set(pb)))}"
              f"{'  (' + f'{min(npb):,}' + ' params)' if npb else ''}")
    print(f"  naive_averaging: {format_mean_std(nv) if nv else '--'}")

    corner = "synthesis \\ alignment"
    print(f"\n{corner:<24}" + "".join(f"{a:>17}" for a in ALIGNMENT))
    print("-" * 76)
    cells = {}
    for syn in SYNTHESIS:
        row = f"{syn:<24}"
        for ali in ALIGNMENT:
            v = [r["zeroshot_accuracy"] for r in
                 query(path=registry_path, tier=TIER, method=f"lmech_{syn}_{ali}")
                 if "zeroshot_accuracy" in r]
            cells[(syn, ali)] = v
            row += f"{format_mean_std(v) if v else '--':>17}"
        print(row)

    print(f"\n{'='*76}\nKEY COMPARISONS\n{'='*76}")
    family = []
    for ali in ALIGNMENT:
        b, e = cells.get(("bures", ali), []), cells.get(("euclidean", ali), [])
        n = min(len(b), len(e))
        if n >= 3:
            family.append(paired_comparison(b[:n], e[:n], label=f"bures vs euclidean ({ali})"))
    for syn in SYNTHESIS:
        a, no = cells.get((syn, "sinkhorn"), []), cells.get((syn, "none"), [])
        n = min(len(a), len(no))
        if n >= 3:
            family.append(paired_comparison(a[:n], no[:n], label=f"{syn}: sinkhorn vs none"))
    if family:
        print(summarize_family(family))
    else:
        print("  (need >= 3 seeds per cell for significance tests)")


def main():
    ap = argparse.ArgumentParser(description="Large-parent scale test")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs-a", type=int, default=60)
    ap.add_argument("--epochs-b", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--registry", type=str, default="results/raw/runs.jsonl")
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    if args.table_only:
        print_table(args.registry)
        return
    run(n_seeds=args.seeds, epochs_a=args.epochs_a, epochs_b=args.epochs_b,
        batch_size=args.batch_size, registry_path=args.registry)


if __name__ == "__main__":
    main()
