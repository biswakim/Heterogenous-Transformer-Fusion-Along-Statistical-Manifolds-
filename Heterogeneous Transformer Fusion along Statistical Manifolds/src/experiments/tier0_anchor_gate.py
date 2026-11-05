"""
src/experiments/tier0_anchor_gate.py

TIER 0 -- ANCHOR REPLICATION (protocol Sec. 9.1), the real one.

    "Re-run the pilot's exact configuration (Parent A: 6L, Parent B: 12L,
     CIFAR-100) with 5 seeds for each parent and 3 fusion-internal seeds
     (calibration-batch sampling, Sinkhorn initialization) per parent-seed
     pair, i.e. 15 fusion runs at this one operating point. Purpose:
     confirm the 49.74% pilot result is not a lucky single seed, and
     establish the variance floor that every later comparison is judged
     against."

NOTE ON NAMING: the repo already contains `tier0anchor.py`, which is a
random-weights smoke test, not this gate. That file is left untouched;
this is a new one rather than an overwrite, so nothing you may already be
relying on breaks. Consider renaming the old one to `smoke_test.py` to
avoid confusion later.

This is a GATE (Algorithm 1, step 7): no later tier should run until this
one reports PASS. The specific risk it exists to catch is spelled out in
protocol Sec. 17:

    "If Tier-0 variance across seeds is larger than the pilot's single-run
     margin (49.74% vs 49.18/49.24%), the paper's headline claim must be
     revised from 'outperforms both parents' to whatever the
     corrected-for-variance comparison actually supports."

The pilot's margin was +0.50 points. That is a SMALL margin, so this gate
genuinely may not pass -- and if it doesn't, that is a real finding to
report (Sec. 11's negative-results commitment), not a reason to re-roll
seeds until it does.

COMPUTE COST -- READ BEFORE LAUNCHING
------------------------------------------------------------------------
Full defaults train 5 x (60-epoch 6L parent) + 5 x (100-epoch 12L parent)
= 800 parent-epochs, plus 15 x 15 = 225 fine-tuning epochs. On a single
consumer GPU this is a multi-day job. Do a cheap smoke run FIRST:

    python -m src.experiments.tier0_anchor_gate --smoke

which uses 2 parent seeds x 2 fusion seeds and tiny epoch budgets purely
to verify the plumbing end-to-end. Only launch the full run once that
passes.
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.data.cifar import get_cifar100_dataloaders
from src.models.vitparents import build_vit_parent
from src.fusion.pipeline import MSOTFusionPipeline
from src.utils.seeding import set_seed, derive_seed, seeded_calibration_batch
from src.utils.registry import log_run, already_logged, query
from src.stats.analysis import mean_std, format_mean_std, paired_comparison
from src.experiments.tier1depthsweep import (
    train_parent, evaluate, train_one_epoch, checkpoint_is_complete,
    build_warmup_cosine_scheduler,
)

TIER = "tier0"
DATASET = "cifar100"
ARCH_PAIR = "A_6L_vs_B-12_12L"


def fine_tune_with_history(fused_model, train_loader, test_loader, device,
                           epochs=15, lr=3e-4, recovery_target=None):
    """
    Fine-tune and return the FULL per-epoch history, not just one number.

    Differs from tier1depthsweep.fine_tune_fused_child only in what it
    returns: this reports final accuracy plus `epochs_to_recovery`
    (Sec. 10: "first epoch at which the fused child matches or exceeds the
    better parent's final accuracy"), which cannot be recovered after the
    fact from a single scalar.
    """
    fused_model = fused_model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(fused_model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler()

    history, epochs_to_recovery = [], None
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            fused_model, train_loader, optimizer, criterion, scaler, device, accum_steps=2
        )
        test_loss, test_acc = evaluate(fused_model, test_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                        "test_loss": test_loss, "test_acc": test_acc})
        if recovery_target is not None and epochs_to_recovery is None and test_acc >= recovery_target:
            epochs_to_recovery = epoch + 1
        print(f"    Epoch [{epoch+1}/{epochs}] test acc {test_acc:.2f}%")

    return {
        "final_accuracy": history[-1]["test_acc"],
        "final_test_loss": history[-1]["test_loss"],
        "best_accuracy_reference_only": max(h["test_acc"] for h in history),
        "epochs_to_recovery": epochs_to_recovery,
        "history": history,
    }


def get_or_train_parent(config_id, seed, train_loader, test_loader, device, epochs):
    """
    Train (or reuse) one parent for a specific seed. Checkpoints are named
    per-seed so the 5 seeds never overwrite each other.

    Existing un-seeded checkpoints (`Parent_A_best.pth`) are deliberately
    NOT reused as "seed 0": they were trained with an unrecorded seed, so
    counting them as one of the 5 would break the premise that the 5 are
    independent, identically-produced samples.
    """
    name = f"Parent_{config_id}_seed{seed}"
    set_seed(seed)
    model = build_vit_parent(config_id).to(device)

    if checkpoint_is_complete(name):
        print(f"  Loading completed checkpoint for {name}")
        model.load_state_dict(torch.load(f"checkpoints/{name}_best.pth", map_location=device))
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        _, acc = evaluate(model, test_loader, criterion, device)
        print(f"  {name} loaded, verified test acc {acc:.2f}%")
        return model, acc

    if os.path.exists(f"checkpoints/{name}_best.pth"):
        print(f"  {name} has a checkpoint but no completion marker (interrupted run) -- retraining.")
    model, acc = train_parent(model, name, train_loader, test_loader, device, epochs=epochs)
    return model, acc


def run_tier_0(n_parent_seeds=5, n_fusion_seeds=3, epochs_A=60, epochs_B=100,
               ft_epochs=15, alpha=0.4, beta=0.5, eps_fgw=0.02, eps_sink=0.02,
               registry_path="results/raw/runs.jsonl"):
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TIER 0 ANCHOR GATE === device={device}")
    print(f"{n_parent_seeds} parent seeds x {n_fusion_seeds} fusion seeds = "
          f"{n_parent_seeds * n_fusion_seeds} fusion runs\n")

    train_loader, test_loader = get_cifar100_dataloaders(batch_size=128)
    per_parent_seed = []

    for ps in range(n_parent_seeds):
        print(f"\n--- Parent seed {ps} ---")
        parent_A, acc_A = get_or_train_parent("A", ps, train_loader, test_loader, device, epochs_A)
        parent_A.cpu(); gc.collect(); torch.cuda.empty_cache()
        parent_B, acc_B = get_or_train_parent("B-12", ps, train_loader, test_loader, device, epochs_B)
        parent_A.to(device)
        better_parent = max(acc_A, acc_B)

        fused_accs = []
        for fs in range(n_fusion_seeds):
            key = dict(tier=TIER, dataset=DATASET, arch_pair=ARCH_PAIR, method="msot",
                       parent_seed=ps, fusion_seed=fs)
            if already_logged(key, path=registry_path):
                prior = query(path=registry_path, **key)[0]
                print(f"  [seed {ps}.{fs}] already logged ({prior.get('final_accuracy')}%), skipping")
                fused_accs.append(float(prior["final_accuracy"]))
                continue

            print(f"  [seed {ps}.{fs}] fusing...")
            fusion_seed = derive_seed(TIER, ARCH_PAIR, "msot", ps, fs)
            set_seed(fusion_seed)
            calib = seeded_calibration_batch(get_cifar100_dataloaders, 32, fusion_seed, device)

            t0 = time.time()
            pipeline = MSOTFusionPipeline(parent_A, parent_B, alpha=alpha, beta=beta,
                                          epsilon_fgw=eps_fgw, epsilon_sink=eps_sink)
            fused_child = pipeline.fuse(calib)
            fusion_seconds = time.time() - t0

            result = fine_tune_with_history(fused_child, train_loader, test_loader, device,
                                            epochs=ft_epochs, recovery_target=better_parent)
            record = {
                **key,
                "parent_A_accuracy": acc_A,
                "parent_B_accuracy": acc_B,
                "better_parent_accuracy": better_parent,
                "fusion_seconds": fusion_seconds,
                "n_params": sum(p.numel() for p in fused_child.parameters()),
                "alpha": alpha, "beta": beta, "eps_fgw": eps_fgw, "eps_sink": eps_sink,
                "ft_epochs": ft_epochs,
                **{k: v for k, v in result.items() if k != "history"},
                "history": result["history"],
            }
            log_run(record, path=registry_path)
            fused_accs.append(result["final_accuracy"])
            print(f"  [seed {ps}.{fs}] final {result['final_accuracy']:.2f}% "
                  f"(better parent {better_parent:.2f}%, fusion {fusion_seconds:.1f}s)")

            del fused_child; gc.collect(); torch.cuda.empty_cache()

        per_parent_seed.append({"parent_seed": ps, "acc_A": acc_A, "acc_B": acc_B,
                                "better_parent": better_parent,
                                "fused_mean": float(np.mean(fused_accs))})
        parent_A.cpu(); parent_B.cpu(); del parent_A, parent_B
        gc.collect(); torch.cuda.empty_cache()

    report_gate(per_parent_seed, registry_path)


def report_gate(per_parent_seed, registry_path="results/raw/runs.jsonl"):
    """
    Aggregate and decide PASS/FAIL.

    Pairing is done at the PARENT-SEED level: each parent seed contributes
    one fused number (the mean over its fusion seeds) paired against that
    same seed's parents. Treating all 15 fusion runs as independent would
    inflate n threefold, since 3 of them share one pair of parents -- that
    is pseudo-replication, and it would make the test look far more
    confident than the data supports.
    """
    accs_A = [r["acc_A"] for r in per_parent_seed]
    accs_B = [r["acc_B"] for r in per_parent_seed]
    accs_F = [r["fused_mean"] for r in per_parent_seed]
    better = [r["better_parent"] for r in per_parent_seed]

    print("\n" + "=" * 64)
    print("                TIER 0 ANCHOR GATE RESULTS")
    print("=" * 64)
    print(f"Parent A (6L)           : {format_mean_std(accs_A)}")
    print(f"Parent B (12L)          : {format_mean_std(accs_B)}")
    print(f"Fused Child (12L, MSOT) : {format_mean_std(accs_F)}")
    print(f"(n = {len(per_parent_seed)} parent seeds; fused value per seed is the mean over its fusion seeds)")

    cmp_better = paired_comparison(accs_F, better, label="MSOT vs better parent")
    print(f"\nMSOT vs better parent: mean diff {cmp_better.mean_difference:+.3f} pts, "
          f"{cmp_better.test_used}, p={cmp_better.p_value:.4f}, d={cmp_better.cohens_d:.3f}")
    if cmp_better.note:
        print(f"  note: {cmp_better.note}")

    _, std_F = mean_std(accs_F)
    print("\n--- GATE DECISION ---")
    passed = True
    if np.isnan(std_F):
        print("INCONCLUSIVE: need >= 2 parent seeds to estimate variance.")
        passed = False
    else:
        print(f"Fused-child std across parent seeds: {std_F:.3f} pts")
        print(f"Pilot's single-run margin over the better parent was +0.50 pts.")
        if std_F > 0.50:
            print(f"  -> Seed variance ({std_F:.3f}) EXCEEDS the pilot's margin (0.50).")
            print("     Per protocol Sec. 17, the headline claim must be narrowed: a +0.50pt")
            print("     single-run gap is not distinguishable from seed noise at this variance.")
            passed = False
        else:
            print(f"  -> Seed variance ({std_F:.3f}) is within the pilot's margin (0.50).")
        if not np.isnan(cmp_better.p_value) and cmp_better.p_value >= 0.05:
            print(f"  -> MSOT does NOT significantly beat the better parent (p={cmp_better.p_value:.4f}).")
            passed = False
        elif not np.isnan(cmp_better.p_value):
            print(f"  -> MSOT significantly beats the better parent (p={cmp_better.p_value:.4f}).")

    print(f"\nGATE: {'PASS -- proceed to Tier 1.' if passed else 'FAIL / NARROW CLAIM -- see above before proceeding.'}")
    print("Either outcome is a reportable result. Do not re-roll seeds to change it (Sec. 11, point 5).")
    print("=" * 64)
    return passed


def main():
    ap = argparse.ArgumentParser(description="Tier 0 anchor gate (protocol Sec. 9.1)")
    ap.add_argument("--parent-seeds", type=int, default=5)
    ap.add_argument("--fusion-seeds", type=int, default=3)
    ap.add_argument("--epochs-a", type=int, default=60)
    ap.add_argument("--epochs-b", type=int, default=100)
    ap.add_argument("--ft-epochs", type=int, default=15)
    ap.add_argument("--registry", type=str, default="results/raw/runs.jsonl")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run to verify plumbing only: 2x2 seeds, 2/2/2 epochs. Results are meaningless.")
    args = ap.parse_args()

    if args.smoke:
        print("*** SMOKE MODE: verifying plumbing only. These numbers are NOT results. ***\n")
        run_tier_0(n_parent_seeds=2, n_fusion_seeds=2, epochs_A=2, epochs_B=2, ft_epochs=2,
                   registry_path="results/raw/smoke_runs.jsonl")
    else:
        run_tier_0(n_parent_seeds=args.parent_seeds, n_fusion_seeds=args.fusion_seeds,
                   epochs_A=args.epochs_a, epochs_B=args.epochs_b,
                   ft_epochs=args.ft_epochs, registry_path=args.registry)


if __name__ == "__main__":
    main()
