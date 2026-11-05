"""
src/experiments/width_depth_sweep.py

MSOT ZERO-SHOT ACCURACY ACROSS DEPTH GAPS, EACH ALSO VARYING WIDTH.

WHY THIS SCRIPT EXISTS
------------------------------------------------------------------------
The earlier width-heterogeneous experiment (width_mechanisms.py) fixed
the depth gap at the anchor value (6-vs-12) and varied only width
(d_model 128 -> 224). That is a legitimate one-factor-at-a-time design,
but the paper described the setting as "width- and depth-heterogeneous"
without being explicit that depth was held fixed at a single value.
This script fills the gap: it runs MSOT across all four depth gaps
already used everywhere else in the paper (2, 4, 6, 8), and at EVERY
gap the width also increases, using the Bw-8/Bw-10/Bw-12/Bw-14 configs
already defined in VIT_CONFIGS:

    gap 2: Parent A (6L, d=128)  vs  Bw-8  (8L,  d=160)
    gap 4: Parent A (6L, d=128)  vs  Bw-10 (10L, d=192)
    gap 6: Parent A (6L, d=128)  vs  Bw-12 (12L, d=224)
    gap 8: Parent A (6L, d=128)  vs  Bw-14 (14L, d=256)

So both depth gap AND width gap increase together across the sweep,
which is what "varying both" means. This is still not a full
depth-by-width GRID (that would need e.g. gap-2-with-d=256 as well,
which VIT_CONFIGS does not define and this script does not add) -- it is
a diagonal through that grid. State it as such.

METHOD: MSOT only (synthesis='bures', alignment='sinkhorn',
width_alignment_method='zero_pad'), i.e. the pipeline's proposed
configuration throughout. Naive averaging is included as the reference
point that motivates zero-shot evaluation at all (it needs no width
reconciliation to define -- see fuse_naive_averaging -- so it runs
natively on these mismatched pairs by padding before averaging in the
same way the width-mechanisms baseline does).

Parent A is reused directly from the existing small-parent checkpoints
(checkpoints/Parent_A_seed{0..4}_best.pth) -- same config, same seeds,
nothing new to train there. Only the four Bw-* parents are new.

Usage:
    python -m src.experiments.width_depth_sweep --seeds 5
"""

from __future__ import annotations

import argparse
import gc
import os

import torch
import torch.nn as nn

from src.data.cifar import get_cifar100_dataloaders
from src.utils.seeding import set_seed, derive_seed, seeded_calibration_batch
from src.utils.registry import log_run, already_logged, query
from src.stats.analysis import format_mean_std, paired_comparison, summarize_family
from src.fusion.pipelinewidth import MSOTWidthPipeline
from src.baselines.naive_averaging import fuse_naive_averaging
from src.experiments.tier0_anchor_gate import get_or_train_parent
from src.experiments.tier1depthsweep import evaluate, train_parent, checkpoint_is_complete
from src.experiments.tier1_full_sweep import PARENT_EPOCHS

TIER = "widthdepth"
DATASET = "cifar100"

# gap -> (config_id, depth, d_model). Each gap pairs with a WIDER config
# than the last, so width increases alongside depth across the sweep.
GAPS = {
    2: "Bw-8",
    4: "Bw-10",
    6: "Bw-12",
    8: "Bw-14",
}


def get_or_train_width_parent(config_id, seed, train_loader, test_loader, device, epochs):
    """Per-seed checkpoint, same completion-marker discipline as everywhere
    else. Bw-* checkpoints are already named per-seed from the earlier
    width_mechanisms.py run, so Bw-12 seeds 0-4 should already be cached."""
    from src.models.vitparents import build_vit_parent

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
        print(f"  {name} has a checkpoint but no completion marker (interrupted) -- retraining.")
    model, acc = train_parent(model, name, train_loader, test_loader, device, epochs=epochs)
    return model, acc


def run(n_seeds=5, registry_path="results/raw/runs.jsonl"):
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== MSOT ACROSS DEPTH GAPS, WIDTH ALSO VARYING === device={device}\n")
    for gap, cfg in GAPS.items():
        print(f"    gap {gap}: Parent A (6L, d=128) vs {cfg}")
    print()

    train_loader, test_loader = get_cifar100_dataloaders(batch_size=128)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for ps in range(n_seeds):
        print(f"\n{'='*60}\nParent seed {ps}\n{'='*60}")
        parent_A, acc_A = get_or_train_parent("A", ps, train_loader, test_loader,
                                              device, PARENT_EPOCHS["A"])
        parent_A.cpu(); gc.collect(); torch.cuda.empty_cache()

        for gap, cfg in GAPS.items():
            arch_pair = f"A_vs_{cfg}_gap{gap}"
            print(f"\n--- gap {gap} ({cfg}) ---")

            epochs_b = PARENT_EPOCHS.get(cfg, 100)
            parent_B, acc_B = get_or_train_width_parent(cfg, ps, train_loader, test_loader,
                                                         device, epochs_b)
            parent_A.to(device)
            better = max(acc_A, acc_B)
            print(f"  parents: A={acc_A:.2f}%  {cfg}={acc_B:.2f}%")

            # --- naive averaging reference (needs width reconciliation too) ---
            key = dict(tier=TIER, dataset=DATASET, arch_pair=arch_pair,
                       method="naive_averaging", parent_seed=ps, fusion_seed=0)
            if not already_logged(key, path=registry_path):
                from src.fusion.phase5width import project_block_dimensions
                # Naive averaging on mismatched widths: zero-pad Parent A's
                # blocks to Parent B's width, then average -- the direct
                # width-analogue of depth-match-by-duplication.
                import copy
                child = copy.deepcopy(parent_B)
                d_model = parent_B.d_model
                nhead = parent_B.blocks[0].attn.num_heads
                dim_ff = parent_B.blocks[0].ff[0].out_features
                n_a, n_b = len(parent_A.blocks), len(parent_B.blocks)
                for j in range(n_b):
                    i = min(int(j * n_a / n_b), n_a - 1)
                    a_block = project_block_dimensions(parent_A.blocks[i], d_model, nhead, dim_ff)
                    sd_a, sd_b = a_block.state_dict(), parent_B.blocks[j].state_dict()
                    avg = {k: 0.5 * sd_a[k] + 0.5 * sd_b[k] for k in sd_b}
                    child.blocks[j].load_state_dict(avg)
                child = child.to(device)
                zs_loss, zs_acc = evaluate(child, test_loader, criterion, device)
                log_run({**key, "zeroshot_accuracy": zs_acc, "zeroshot_loss": zs_loss,
                         "parent_A_accuracy": acc_A, "parent_B_accuracy": acc_B,
                         "better_parent_accuracy": better}, path=registry_path)
                print(f"  [naive_averaging] zs={zs_acc:5.2f}%  ({100*zs_acc/better:.1f}% of better parent)")
                del child; gc.collect(); torch.cuda.empty_cache()
            else:
                print("  [naive_averaging] (cached)")

            # --- MSOT (proposed configuration) ---
            key = dict(tier=TIER, dataset=DATASET, arch_pair=arch_pair,
                       method="msot", parent_seed=ps, fusion_seed=0)
            if already_logged(key, path=registry_path):
                prior = query(path=registry_path, **key)[0]
                print(f"  [msot] zs={prior.get('zeroshot_accuracy')}  (cached)")
            else:
                fusion_seed = derive_seed(TIER, cfg, ps)
                set_seed(fusion_seed)
                calib = seeded_calibration_batch(get_cifar100_dataloaders, 32, fusion_seed, device)

                pipeline = MSOTWidthPipeline(parent_A, parent_B, alpha=0.4, beta=0.5,
                                             epsilon_fgw=0.02, epsilon_sink=0.02,
                                             width_alignment_method="zero_pad",
                                             synthesis="bures", alignment="sinkhorn",
                                             verbose=False)
                child = pipeline.fuse(calib).to(device)
                zs_loss, zs_acc = evaluate(child, test_loader, criterion, device)

                log_run({**key, "zeroshot_accuracy": zs_acc, "zeroshot_loss": zs_loss,
                         "parent_A_accuracy": acc_A, "parent_B_accuracy": acc_B,
                         "better_parent_accuracy": better,
                         "n_split_slots": getattr(pipeline, "n_split_slots", None)},
                        path=registry_path)
                print(f"  [msot] zs={zs_acc:5.2f}%  ({100*zs_acc/better:.1f}% of better parent)")
                del child, pipeline; gc.collect(); torch.cuda.empty_cache()

            parent_B.cpu(); del parent_B
            gc.collect(); torch.cuda.empty_cache()

        parent_A.cpu(); del parent_A
        gc.collect(); torch.cuda.empty_cache()

    print_table(registry_path)


def print_table(registry_path="results/raw/runs.jsonl"):
    print(f"\n{'='*80}\nMSOT ZERO-SHOT ACCURACY -- DEPTH GAP AND WIDTH BOTH INCREASING\n{'='*80}")
    print(f"{'Method':<20}" + "".join(f"{'Gap '+str(g):>15}" for g in GAPS))
    print("-" * 80)

    cells = {}
    for method in ["naive_averaging", "msot"]:
        row = f"{method:<20}"
        for gap, cfg in GAPS.items():
            vals = [r["zeroshot_accuracy"] for r in
                    query(path=registry_path, tier=TIER, arch_pair=f"A_vs_{cfg}_gap{gap}", method=method)
                    if "zeroshot_accuracy" in r]
            cells[(gap, method)] = vals
            row += f"{format_mean_std(vals) if vals else '--':>15}"
        print(row)

    print(f"\n{'='*80}\nMSOT vs naive_averaging, per gap (Holm-Bonferroni corrected)\n{'='*80}")
    family = []
    for gap in GAPS:
        m, n = cells.get((gap, "msot"), []), cells.get((gap, "naive_averaging"), [])
        k = min(len(m), len(n))
        if k >= 3:
            family.append(paired_comparison(m[:k], n[:k], label=f"gap{gap} msot vs naive_averaging"))
    if family:
        print(summarize_family(family))
    else:
        print("  (need >= 3 seeds per gap for significance tests)")


def main():
    ap = argparse.ArgumentParser(description="MSOT across depth gaps with width also varying")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--registry", type=str, default="results/raw/runs.jsonl")
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    if args.table_only:
        print_table(args.registry)
        return
    run(n_seeds=args.seeds, registry_path=args.registry)


if __name__ == "__main__":
    main()
