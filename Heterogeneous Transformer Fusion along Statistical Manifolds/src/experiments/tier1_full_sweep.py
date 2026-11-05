"""
src/experiments/tier1_full_sweep.py

TIER 1 -- DEPTH-GAP SWEEP AND MAIN COMPARISON TABLE (protocol Sec. 9.2).

    "For each of the 8 architecture pairs (4 depth gaps x {Variation 1,
     Variation 2}), run all 7 methods from Section 7 (2 parents-alone +
     ensemble + 4 fusion methods). To bound cost, use 3 parent-seed
     pairings (not the full 5x5 cross) x 2 fusion-internal seeds per
     pairing = 6 fusion runs per (architecture pair, method) cell for the
     4 fusion methods, and 5 seeds each for the parents-alone/ensemble
     rows. Output: the main result table of the eventual paper (Table 2)."

This is a LONG job (the protocol budgets ~345 runs for this tier). Two
properties make that survivable:

  - Resumable: every completed cell is logged to the registry and skipped
    on a rerun, so an interrupted sweep picks up where it stopped.
  - Ordered cheapest-first: Variation 1 before Variation 2, small gaps
    before large, so partial results are still a usable (if incomplete)
    table if you run out of time or compute.

PREREQUISITE: Tier 0 must have PASSED (`tier0_anchor_gate.py`). Per
Algorithm 1 step 7, advance only if the previous tier's results were
non-degenerate and statistically analyzed. This script warns but does not
hard-block, since you may deliberately want a partial Tier 1 preview.
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import torch
import torch.nn as nn

from src.data.cifar import get_cifar100_dataloaders
from src.models.vitparents import build_vit_parent
from src.utils.seeding import set_seed, derive_seed, seeded_calibration_batch
from src.utils.registry import log_run, already_logged, query
from src.stats.analysis import format_mean_std, paired_comparison, holm_bonferroni, summarize_family
from src.experiments.methods import build_method, ALL_METHODS, NEEDS_FINETUNE, method_display_name
from src.experiments.tier0_anchor_gate import fine_tune_with_history, get_or_train_parent
from src.experiments.tier1depthsweep import evaluate

TIER = "tier1"
DATASET = "cifar100"

# (gap, Variation-1 config, Variation-2 config) -- Sec. 5.1 architecture tables
ARCH_PAIRS = [
    (2, "B-8", "Bw-8"),
    (4, "B-10", "Bw-10"),
    (6, "B-12", "Bw-12"),   # the pilot's anchor operating point
    (8, "B-14", "Bw-14"),
]

PARENT_EPOCHS = {"A": 60, "B-8": 80, "B-10": 90, "B-12": 100, "B-14": 110,
                 "Bw-8": 80, "Bw-10": 90, "Bw-12": 100, "Bw-14": 110}


def run_cell(method, parent_A, parent_B, acc_A, acc_B, arch_pair, ps, fs,
             train_loader, test_loader, device, ft_epochs, registry_path, **hp):
    """Run one (architecture pair, method, seed) cell, or skip it if already logged."""
    key = dict(tier=TIER, dataset=DATASET, arch_pair=arch_pair, method=method,
               parent_seed=ps, fusion_seed=fs)
    if already_logged(key, path=registry_path):
        prior = query(path=registry_path, **key)[0]
        return float(prior.get("final_accuracy", "nan")), True

    better_parent = max(acc_A, acc_B)
    fusion_seed = derive_seed(TIER, arch_pair, method, ps, fs)
    set_seed(fusion_seed)
    calib = seeded_calibration_batch(get_cifar100_dataloaders, 32, fusion_seed, device)

    t0 = time.time()
    model, diag = build_method(method, parent_A, parent_B, calib, **hp)
    fusion_seconds = time.time() - t0
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    if method in NEEDS_FINETUNE:
        result = fine_tune_with_history(model, train_loader, test_loader, device,
                                        epochs=ft_epochs, recovery_target=better_parent)
    else:
        # Parents and the ensemble are fixed reference points -- fine-tuning
        # them would make them different (better-trained) models and destroy
        # their role as the comparison floor/ceiling.
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        result = {"final_accuracy": test_acc, "final_test_loss": test_loss,
                  "best_accuracy_reference_only": test_acc, "epochs_to_recovery": None,
                  "history": []}

    record = {**key, "parent_A_accuracy": acc_A, "parent_B_accuracy": acc_B,
              "better_parent_accuracy": better_parent, "fusion_seconds": fusion_seconds,
              "n_params": sum(p.numel() for p in model.parameters()),
              "ft_epochs": ft_epochs if method in NEEDS_FINETUNE else 0,
              **hp, **diag,
              **{k: v for k, v in result.items() if k != "history"},
              "history": result["history"]}
    log_run(record, path=registry_path)

    del model
    gc.collect(); torch.cuda.empty_cache()
    return result["final_accuracy"], False


def run_tier_1(variations=("v1", "v2"), gaps=(2, 4, 6, 8), n_parent_seeds=3,
               n_fusion_seeds=2, n_reference_seeds=5, ft_epochs=15,
               registry_path="results/raw/runs.jsonl", **hp):
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TIER 1 DEPTH-GAP SWEEP === device={device}")

    if not query(path=registry_path, tier="tier0"):
        print("\nWARNING: no Tier 0 runs found in the registry. Protocol Algorithm 1 gates\n"
              "Tier 1 behind a passing Tier 0 anchor. Proceeding anyway, but the variance\n"
              "floor these comparisons should be judged against has not been established.\n")

    train_loader, test_loader = get_cifar100_dataloaders(batch_size=128)

    for variation in variations:
        for gap, cfg_v1, cfg_v2 in ARCH_PAIRS:
            if gap not in gaps:
                continue
            cfg_B = cfg_v1 if variation == "v1" else cfg_v2
            arch_pair = f"A_vs_{cfg_B}"
            print(f"\n{'='*64}\n{variation.upper()} | gap {gap} | A vs {cfg_B}\n{'='*64}")

            max_seeds = max(n_parent_seeds, n_reference_seeds)
            for ps in range(max_seeds):
                parent_A, acc_A = get_or_train_parent("A", ps, train_loader, test_loader,
                                                      device, PARENT_EPOCHS["A"])
                parent_A.cpu(); gc.collect(); torch.cuda.empty_cache()
                parent_B, acc_B = get_or_train_parent(cfg_B, ps, train_loader, test_loader,
                                                      device, PARENT_EPOCHS[cfg_B])
                parent_A.to(device)

                for method in ALL_METHODS:
                    # reference rows get n_reference_seeds and no fusion seeds;
                    # fusion rows get n_parent_seeds x n_fusion_seeds
                    if method in NEEDS_FINETUNE:
                        if ps >= n_parent_seeds:
                            continue
                        seed_list = range(n_fusion_seeds)
                    else:
                        if ps >= n_reference_seeds:
                            continue
                        seed_list = [0]

                    for fs in seed_list:
                        acc, skipped = run_cell(method, parent_A, parent_B, acc_A, acc_B,
                                                arch_pair, ps, fs, train_loader, test_loader,
                                                device, ft_epochs, registry_path, **hp)
                        tag = "cached" if skipped else "run"
                        print(f"  [{method:<18} ps={ps} fs={fs}] {acc:.2f}%  ({tag})")

                parent_A.cpu(); parent_B.cpu(); del parent_A, parent_B
                gc.collect(); torch.cuda.empty_cache()

    print_main_table(registry_path, variations, gaps)


def print_main_table(registry_path="results/raw/runs.jsonl", variations=("v1", "v2"), gaps=(2, 4, 6, 8)):
    """Render the Table-II-shaped summary, and run the corrected significance family."""
    for variation in variations:
        print(f"\n{'='*84}\nTIER 1 MAIN TABLE -- Variation {variation.upper()}\n{'='*84}")
        header = f"{'Method':<22}" + "".join(f"{'Gap '+str(g):>15}" for g in gaps)
        print(header); print("-" * len(header))

        for method in ALL_METHODS:
            row = f"{method_display_name(method):<22}"
            for gap in gaps:
                cfg_B = next(c1 if variation == "v1" else c2 for g, c1, c2 in ARCH_PAIRS if g == gap)
                vals = [r["final_accuracy"] for r in
                        query(path=registry_path, tier=TIER, arch_pair=f"A_vs_{cfg_B}", method=method)
                        if "final_accuracy" in r]
                row += f"{format_mean_std(vals) if vals else '--':>15}"
            print(row)

        # Significance: MSOT vs each baseline, corrected within this variation's family
        print(f"\nPaired comparisons vs MSOT (Holm-Bonferroni corrected within Variation {variation.upper()}):")
        family = []
        for gap in gaps:
            cfg_B = next(c1 if variation == "v1" else c2 for g, c1, c2 in ARCH_PAIRS if g == gap)
            ap = f"A_vs_{cfg_B}"
            msot = {(r["parent_seed"], r["fusion_seed"]): r["final_accuracy"]
                    for r in query(path=registry_path, tier=TIER, arch_pair=ap, method="msot")
                    if "final_accuracy" in r}
            for baseline in ALL_METHODS:
                if baseline == "msot":
                    continue
                base = {(r["parent_seed"], r["fusion_seed"]): r["final_accuracy"]
                        for r in query(path=registry_path, tier=TIER, arch_pair=ap, method=baseline)
                        if "final_accuracy" in r}
                shared = sorted(set(msot) & set(base))
                if len(shared) >= 3:
                    family.append(paired_comparison([msot[k] for k in shared],
                                                    [base[k] for k in shared],
                                                    label=f"gap{gap} msot vs {baseline}"))
        if family:
            print(summarize_family(family))
        else:
            print("  (not enough paired runs yet for significance testing)")


def main():
    ap = argparse.ArgumentParser(description="Tier 1 depth-gap sweep (protocol Sec. 9.2)")
    ap.add_argument("--variations", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    ap.add_argument("--gaps", nargs="+", type=int, default=[2, 4, 6, 8])
    ap.add_argument("--parent-seeds", type=int, default=3)
    ap.add_argument("--fusion-seeds", type=int, default=2)
    ap.add_argument("--reference-seeds", type=int, default=5)
    ap.add_argument("--ft-epochs", type=int, default=15)
    ap.add_argument("--registry", type=str, default="results/raw/runs.jsonl")
    ap.add_argument("--table-only", action="store_true",
                    help="Skip running; just re-render the table from already-logged results.")
    args = ap.parse_args()

    if args.table_only:
        print_main_table(args.registry, tuple(args.variations), tuple(args.gaps))
        return

    run_tier_1(variations=tuple(args.variations), gaps=tuple(args.gaps),
               n_parent_seeds=args.parent_seeds, n_fusion_seeds=args.fusion_seeds,
               n_reference_seeds=args.reference_seeds, ft_epochs=args.ft_epochs,
               registry_path=args.registry)


if __name__ == "__main__":
    main()
