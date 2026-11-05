"""
src/experiments/tier2_ablations.py

TIER 2 -- ABLATION BATTERY (protocol Sec. 8), the achievable-now slice.

Covers the four hyperparameter sweeps that already flow through the real
pipeline with zero new fusion code (confirmed against current
phase2macro.py / methods.py):

    A1 -- alpha (FGW feature/topology trade-off), Phase II
    A2 -- beta  (adjacency mix), Phase I
    A3 -- eps_fgw (FGW entropic regularization), Phase II
    C2 -- eps_sink (Sinkhorn entropic regularization), Phase IV

NOT included yet (need small code changes to fuse_blocks_ot itself, not
just parameter sweeps -- deliberately deferred rather than rushed):
    A4 -- similarity metric for functional adjacency (needs CKA implemented)
    B1 -- meso-scale synthesis mechanism (bures vs euclidean vs identity)
    B2 -- fixed vs. transport-derived lambda
    C1 -- micro-scale alignment mechanism (Sinkhorn vs Hungarian vs none)

Per Sec. 8: "All ablations in this section are run on CIFAR-100 at the
fixed 6-vs-12 operating point... reusing the same 5 parent seeds across
every ablation so that only the fusion mechanism varies." This reuses
whatever Tier-0 parent checkpoints already exist for A_vs_B-12 rather than
retraining.
"""

from __future__ import annotations

import argparse
import gc
import os

import torch

from src.data.cifar import get_cifar100_dataloaders
from src.utils.seeding import set_seed, derive_seed, seeded_calibration_batch
from src.utils.registry import log_run, already_logged, query, collect_metric
from src.stats.analysis import format_mean_std, paired_comparison, coefficient_of_variation
from src.experiments.methods import build_method
from src.experiments.tier0_anchor_gate import fine_tune_with_history, get_or_train_parent

TIER = "tier2"
DATASET = "cifar100"
ARCH_PAIR = "A_vs_B-12"  # the fixed anchor, per Sec. 8

PROPOSED_DEFAULTS = {"alpha": 0.4, "beta": 0.5, "eps_fgw": 0.02, "eps_sink": 0.02}

# (family, sweep_param, values) -- protocol Sec. 8.5's ablation summary matrix
SWEEPS = [
    ("A1_alpha", "alpha", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    ("A2_beta", "beta", [0.0, 0.25, 0.5, 0.75, 1.0]),
    ("A3_eps_fgw", "eps_fgw", [0.005, 0.01, 0.02, 0.05, 0.1]),
    ("C2_eps_sink", "eps_sink", [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]),
]


def run_ablation_cell(family, param_name, value, parent_A, parent_B, acc_A, acc_B,
                      ps, fs, train_loader, test_loader, device, ft_epochs, registry_path):
    hp = {**PROPOSED_DEFAULTS, param_name: value}
    method_tag = f"msot_{family}_{value}"
    key = dict(tier=TIER, dataset=DATASET, arch_pair=ARCH_PAIR, method=method_tag,
               parent_seed=ps, fusion_seed=fs)
    if already_logged(key, path=registry_path):
        prior = query(path=registry_path, **key)[0]
        return float(prior.get("final_accuracy", "nan")), True

    better_parent = max(acc_A, acc_B)
    fusion_seed = derive_seed(TIER, family, value, ps, fs)
    set_seed(fusion_seed)
    calib = seeded_calibration_batch(get_cifar100_dataloaders, 32, fusion_seed, device)

    model, diag = build_method("msot", parent_A, parent_B, calib, **hp)
    model = model.to(device)
    result = fine_tune_with_history(model, train_loader, test_loader, device,
                                    epochs=ft_epochs, recovery_target=better_parent)

    record = {**key, "ablation_family": family, "swept_param": param_name, "swept_value": value,
              **hp, "parent_A_accuracy": acc_A, "parent_B_accuracy": acc_B,
              **{k: v for k, v in result.items() if k != "history"}, "history": result["history"]}
    log_run(record, path=registry_path)

    del model
    gc.collect(); torch.cuda.empty_cache()
    return result["final_accuracy"], False


def run_tier_2(n_seeds=5, ft_epochs=15, registry_path="results/raw/runs.jsonl"):
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TIER 2 ABLATION BATTERY (achievable slice: A1, A2, A3, C2) === device={device}")

    train_loader, test_loader = get_cifar100_dataloaders(batch_size=128)

    for ps in range(n_seeds):
        print(f"\n--- Parent seed {ps} ---")
        parent_A, acc_A = get_or_train_parent("A", ps, train_loader, test_loader, device, 60)
        parent_A.cpu(); gc.collect(); torch.cuda.empty_cache()
        parent_B, acc_B = get_or_train_parent("B-12", ps, train_loader, test_loader, device, 100)
        parent_A.to(device)

        for family, param_name, values in SWEEPS:
            for value in values:
                acc, skipped = run_ablation_cell(family, param_name, value, parent_A, parent_B,
                                                 acc_A, acc_B, ps, 0, train_loader, test_loader,
                                                 device, ft_epochs, registry_path)
                tag = "cached" if skipped else "run"
                print(f"  [{family:<12} {param_name}={value:<6} ps={ps}] {acc:.2f}%  ({tag})")

        parent_A.cpu(); parent_B.cpu(); del parent_A, parent_B
        gc.collect(); torch.cuda.empty_cache()

    print_ablation_table(registry_path)


def print_ablation_table(registry_path="results/raw/runs.jsonl"):
    """Render the Table-III-shaped summary: best value per family, full sweep curve, and
    a paired comparison of the best value against the proposed default (alpha=0.4 etc.)."""
    print(f"\n{'='*90}\nTIER 2 ABLATION TABLE\n{'='*90}")
    print(f"{'Family':<14}{'Param':<10}{'Value':<10}{'Accuracy':<22}{'CoV':>8}")
    print("-" * 64)

    best_per_family = {}
    for family, param_name, values in SWEEPS:
        family_results = []
        for value in values:
            method_tag = f"msot_{family}_{value}"
            vals = collect_metric("final_accuracy", path=registry_path, tier=TIER,
                                  arch_pair=ARCH_PAIR, method=method_tag)
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            cov = coefficient_of_variation(vals)
            family_results.append((value, vals, mean))
            cov_str = "--" if cov != cov else f"{cov:.3f}"  # nan check
            print(f"{family:<14}{param_name:<10}{str(value):<10}{format_mean_std(vals):<22}{cov_str:>8}")
        if family_results:
            best_value, best_vals, _ = max(family_results, key=lambda x: x[2])
            best_per_family[family] = (best_value, best_vals)
        print()

    print(f"{'='*90}\nBest value vs. proposed default, per family\n{'='*90}")
    for family, param_name, values in SWEEPS:
        if family not in best_per_family:
            continue
        best_value, best_vals = best_per_family[family]
        default_value = PROPOSED_DEFAULTS[param_name]
        default_tag = f"msot_{family}_{default_value}"
        default_vals = collect_metric("final_accuracy", path=registry_path, tier=TIER,
                                      arch_pair=ARCH_PAIR, method=default_tag)
        if best_value == default_value:
            print(f"{family}: best value ({best_value}) IS the proposed default -- no comparison needed.")
            continue
        if len(default_vals) >= 3 and len(best_vals) >= 3:
            n = min(len(default_vals), len(best_vals))
            cmp = paired_comparison(best_vals[:n], default_vals[:n],
                                    label=f"{family} best({best_value}) vs default({default_value})")
            print(f"{family}: best={best_value} vs default={default_value} -> "
                  f"diff {cmp.mean_difference:+.3f}, {cmp.test_used}, p={cmp.p_value}, d={cmp.cohens_d:.3f}")
        else:
            print(f"{family}: not enough paired seeds yet for a significance test.")


def main():
    ap = argparse.ArgumentParser(description="Tier 2 ablation battery (protocol Sec. 8)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ft-epochs", type=int, default=15)
    ap.add_argument("--registry", type=str, default="results/raw/runs.jsonl")
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    if args.table_only:
        print_ablation_table(args.registry)
        return

    run_tier_2(n_seeds=args.seeds, ft_epochs=args.ft_epochs, registry_path=args.registry)


if __name__ == "__main__":
    main()
