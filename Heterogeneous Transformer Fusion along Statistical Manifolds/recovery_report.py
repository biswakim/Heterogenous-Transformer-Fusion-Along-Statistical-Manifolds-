"""
analyze_recovery.py -- run from /workspace:  python analyze_recovery.py

Computes epochs-to-recovery (protocol Sec. 10: "first epoch at which the
fused child matches or exceeds the better parent's final accuracy") for
every fusion method at every depth gap, and runs the paired comparison
that the efficiency claim would rest on.

IMPORTANT CAVEAT this script surfaces explicitly: a run that NEVER reached
the better parent's accuracy within the 15-epoch budget has
epochs_to_recovery = None. Averaging only over runs that DID recover would
silently flatter whichever method fails most often -- so the
"never_recovered" column is printed alongside, and any method with
failures must have that stated in the paper rather than reporting a mean
over survivors only.
"""

import json
import statistics
from collections import defaultdict

try:
    from scipy import stats as st
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

REG = "results/raw/runs.jsonl"
FUSION = ["naive_averaging", "identity_pad_ot", "clafusion_dp", "msot"]

runs = [json.loads(l) for l in open(REG) if l.strip()]
t1 = [r for r in runs if r.get("tier") == "tier1"]

recovered = defaultdict(list)   # (arch_pair, method) -> [epochs]
never = defaultdict(int)
finals = defaultdict(list)

for r in t1:
    m = r.get("method")
    if m not in FUSION:
        continue
    key = (r.get("arch_pair"), m)
    e = r.get("epochs_to_recovery")
    if e is None:
        never[key] += 1
    else:
        recovered[key].append(e)
    if r.get("final_accuracy") is not None:
        finals[key].append(r["final_accuracy"])

print("=" * 86)
print("EPOCHS-TO-RECOVERY  (first epoch reaching the better parent's accuracy)")
print("=" * 86)
print(f"{'arch_pair':<18}{'method':<20}{'n_recovered':>12}{'mean_epochs':>13}"
      f"{'median':>8}{'never':>8}{'final_acc':>11}")
print("-" * 86)

pairs = sorted(set(list(recovered.keys()) + list(never.keys())))
for key in pairs:
    ap, m = key
    v = recovered.get(key, [])
    n_never = never.get(key, 0)
    mean_e = f"{statistics.mean(v):.2f}" if v else "--"
    med_e = f"{statistics.median(v):.1f}" if v else "--"
    fa = f"{statistics.mean(finals[key]):.2f}" if finals.get(key) else "--"
    print(f"{ap:<18}{m:<20}{len(v):>12}{mean_e:>13}{med_e:>8}{n_never:>8}{fa:>11}")

# ------------------------------------------------------------------ #
# The claim: MSOT recovers FASTER than each baseline. Paired by seed.
# ------------------------------------------------------------------ #
print()
print("=" * 86)
print("PAIRED COMPARISON OF RECOVERY SPEED (MSOT vs each baseline, per gap)")
print("negative diff = MSOT recovers in FEWER epochs = MSOT is faster")
print("=" * 86)

by_seed = defaultdict(dict)  # (arch_pair, ps, fs) -> {method: epochs}
for r in t1:
    m = r.get("method")
    if m not in FUSION:
        continue
    k = (r.get("arch_pair"), r.get("parent_seed"), r.get("fusion_seed"))
    by_seed[k][m] = r.get("epochs_to_recovery")

arch_pairs = sorted({k[0] for k in by_seed})
print(f"{'comparison':<44}{'n':>4}{'mean diff':>11}{'p':>10}")
print("-" * 70)
for ap in arch_pairs:
    for base in ["naive_averaging", "identity_pad_ot", "clafusion_dp"]:
        a, b = [], []
        skipped = 0
        for k, d in by_seed.items():
            if k[0] != ap:
                continue
            if "msot" in d and base in d:
                # A None on either side means that run never recovered;
                # it cannot be treated as a number, so the pair is dropped
                # and counted rather than silently imputed.
                if d["msot"] is None or d[base] is None:
                    skipped += 1
                    continue
                a.append(d["msot"])
                b.append(d[base])
        if len(a) < 3:
            print(f"{ap} msot vs {base:<22}{len(a):>4}  (too few pairs)")
            continue
        diff = statistics.mean([x - y for x, y in zip(a, b)])
        if HAVE_SCIPY and any(x != y for x, y in zip(a, b)):
            p = st.wilcoxon(a, b).pvalue
            p_str = f"{p:.4f}"
        else:
            p_str = "--"
        note = f"  ({skipped} pair(s) dropped: no recovery)" if skipped else ""
        print(f"{ap} msot vs {base:<22}{len(a):>4}{diff:>11.2f}{p_str:>10}{note}")

print()
print("Reminder: these p-values are UNCORRECTED. Apply Holm-Bonferroni across")
print("this family before claiming significance in the paper (protocol Sec. 11).")
