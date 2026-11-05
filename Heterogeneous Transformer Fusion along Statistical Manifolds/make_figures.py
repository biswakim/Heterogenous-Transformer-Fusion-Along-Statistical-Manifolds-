"""
make_figures.py -- run from /workspace:  python make_figures.py

Generates every figure for the paper from results/raw/runs.jsonl.
No GPU, no retraining -- everything is already logged.

Figures produced (each skipped with a message if its data is absent):

  fig1_crossover.pdf    Fine-tuning trajectories with the crossover epoch
                        marked. THE most important figure: it shows naive
                        averaging starting at chance and overtaking the
                        alignment-preserving methods at a specific epoch,
                        which reframes "naive averaging wins" from an
                        awkward result into a characterised trade-off.

  fig2_mechanism.pdf    Zero-shot accuracy by interpolation rule x
                        alignment mechanism. The headline decomposition:
                        the interpolation rule moves the bars, the
                        alignment mechanism does not.

  fig3_sweeps.pdf       Accuracy vs alpha, beta, eps_FGW, eps_Sink with
                        std bands. Supports the insensitivity claim --
                        flat lines are the point, so the y-axis is scaled
                        to seed noise rather than auto-scaled, which
                        would make noise look like signal.

  fig4_zeroshot_gap.pdf Zero-shot accuracy vs depth gap, all methods.

  fig5_datasets.pdf     CIFAR-100 vs Tiny-ImageNet side by side, showing
                        the finding replicates.

A note on fig3's y-axis: auto-scaling a flat curve makes ±0.4 points look
like a dramatic trend. Each panel is therefore drawn on a fixed window
centred on the sweep mean and at least +/-2 std wide, so "flat" reads as
flat.
"""

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REG = "results/raw/runs.jsonl"
OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

COLORS = {
    "naive_averaging": "#d62728",
    "identity_pad_ot": "#2ca02c",
    "clafusion_dp": "#ff7f0e",
    "msot": "#1f77b4",
    "parent": "#7f7f7f",
}
LABELS = {
    "naive_averaging": "Naive averaging",
    "identity_pad_ot": "Identity pad. + OT",
    "clafusion_dp": "CLAFusion-style DP",
    "msot": "MSOT (proposed)",
}


def load():
    if not os.path.exists(REG):
        raise SystemExit(f"{REG} not found -- run this from /workspace.")
    out = []
    for line in open(REG):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


RUNS = load()


def q(**f):
    return [r for r in RUNS if all(r.get(k) == v for k, v in f.items())]


# --------------------------------------------------------------------- #
# Figure 1: fine-tuning trajectories + crossover
# --------------------------------------------------------------------- #
def fig_crossover(arch_pair="A_vs_B-12"):
    curves = {}
    for method in ["naive_averaging", "identity_pad_ot", "clafusion_dp", "msot"]:
        hs = [r["history"] for r in q(tier="tier1", arch_pair=arch_pair, method=method)
              if r.get("history")]
        if not hs:
            continue
        n_ep = min(len(h) for h in hs)
        arr = np.array([[e["test_acc"] for e in h[:n_ep]] for h in hs])
        curves[method] = (arr.mean(axis=0), arr.std(axis=0))

    if len(curves) < 2:
        print("  skip fig1: need fine-tuning histories from tier1 (none found)")
        return

    parents = [r["better_parent_accuracy"] for r in q(tier="tier1", arch_pair=arch_pair)
               if r.get("better_parent_accuracy")]
    better = float(np.mean(parents)) if parents else None

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for m, (mu, sd) in curves.items():
        ep = np.arange(1, len(mu) + 1)
        ax.plot(ep, mu, label=LABELS[m], color=COLORS[m], lw=1.8)
        ax.fill_between(ep, mu - sd, mu + sd, color=COLORS[m], alpha=0.15, lw=0)

    # crossover: first epoch where naive averaging overtakes MSOT
    note = ""
    if "naive_averaging" in curves and "msot" in curves:
        nv, ms = curves["naive_averaging"][0], curves["msot"][0]
        n = min(len(nv), len(ms))
        cross = next((i + 1 for i in range(n) if nv[i] > ms[i]), None)
        if cross:
            ax.axvline(cross, color="k", ls=":", lw=1.2)
            ax.annotate(f"crossover\nepoch {cross}", xy=(cross, ax.get_ylim()[0] + 6),
                        xytext=(cross + 0.6, ax.get_ylim()[0] + 6),
                        fontsize=8, va="bottom")
            note = f" (crossover at epoch {cross})"

    if better:
        ax.axhline(better, color=COLORS["parent"], ls="--", lw=1.2)
        ax.text(len(next(iter(curves.values()))[0]) * 0.62, better + 0.7,
                "better parent", fontsize=8, color=COLORS["parent"])

    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Fine-tuning trajectories (CIFAR-100, 6L vs. 12L)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(f"{OUT}/fig1_crossover.pdf")
    plt.close(fig)
    print(f"  wrote fig1_crossover.pdf{note}")


# --------------------------------------------------------------------- #
# Figure 2: mechanism grid
# --------------------------------------------------------------------- #
def fig_mechanism():
    SYN = ["bures", "euclidean", "identity_pad"]
    ALI = ["sinkhorn", "hungarian", "none"]
    data = {}
    for s in SYN:
        for a in ALI:
            v = [r["zeroshot_accuracy"] for r in q(tier="tier2mech", method=f"mech_{s}_{a}")
                 if "zeroshot_accuracy" in r]
            if v:
                data[(s, a)] = (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0)
    if not data:
        print("  skip fig2: no tier2mech runs found")
        return

    naive = [r["zeroshot_accuracy"] for r in q(tier="zeroshot", method="naive_averaging")
             if "zeroshot_accuracy" in r]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    w, x = 0.25, np.arange(len(SYN))
    hatches = ["", "//", ".."]
    for i, a in enumerate(ALI):
        mus = [data.get((s, a), (0, 0))[0] for s in SYN]
        sds = [data.get((s, a), (0, 0))[1] for s in SYN]
        ax.bar(x + (i - 1) * w, mus, w, yerr=sds, capsize=3,
               label=f"align: {a}", hatch=hatches[i],
               color=plt.cm.Blues(0.4 + 0.2 * i), edgecolor="k", lw=0.6)

    if naive:
        nv = float(np.mean(naive))
        ax.axhline(nv, color=COLORS["naive_averaging"], ls="--", lw=1.3)
        # Annotate on the LEFT: the legend occupies the upper right, and the
        # tall Bures bars occupy the upper left, but the region just above
        # the naive line on the far left is empty.
        ax.text(-0.42, nv + 0.6, f"naive averaging ({nv:.1f}%)",
                fontsize=7.5, color=COLORS["naive_averaging"],
                ha="left", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(["Bures", "Euclidean", "Identity pad"])
    ax.set_xlabel("Interpolation rule (meso scale)")
    ax.set_ylabel("Zero-shot accuracy (%)")
    ax.set_title("Interpolation rule dominates; alignment does not")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.savefig(f"{OUT}/fig2_mechanism.pdf")
    plt.close(fig)
    print("  wrote fig2_mechanism.pdf")


# --------------------------------------------------------------------- #
# Figure 3: hyperparameter sweeps
# --------------------------------------------------------------------- #
def fig_sweeps():
    fams = [("A1_alpha", r"$\alpha$"), ("A2_beta", r"$\beta$"),
            ("A3_eps_fgw", r"$\epsilon_{FGW}$"), ("C2_eps_sink", r"$\epsilon_{Sink}$")]
    panels = []
    for fam, lab in fams:
        rs = [r for r in q(tier="tier2") if r.get("ablation_family") == fam
              and "final_accuracy" in r]
        if not rs:
            continue
        by = defaultdict(list)
        for r in rs:
            by[r["swept_value"]].append(r["final_accuracy"])
        xs = sorted(by)
        mus = [np.mean(by[v]) for v in xs]
        sds = [np.std(by[v], ddof=1) if len(by[v]) > 1 else 0.0 for v in xs]
        panels.append((lab, xs, mus, sds))

    if not panels:
        print("  skip fig3: no tier2 sweep runs found")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 2.6), sharey=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (lab, xs, mus, sds) in zip(axes, panels):
        ax.errorbar(range(len(xs)), mus, yerr=sds, marker="o", ms=4,
                    color=COLORS["msot"], capsize=3, lw=1.5)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(v) for v in xs], rotation=45, ha="right")
        ax.set_xlabel(lab)
        # Fixed window: flatness must LOOK flat, not be auto-scaled into a trend.
        c = np.mean(mus)
        half = max(2.0 * max(sds + [0.3]), 1.5)
        ax.set_ylim(c - half, c + half)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("Test accuracy (%)")
    fig.suptitle("Fused accuracy is insensitive to every hyperparameter", y=1.04)
    fig.savefig(f"{OUT}/fig3_sweeps.pdf")
    plt.close(fig)
    print("  wrote fig3_sweeps.pdf")


# --------------------------------------------------------------------- #
# Figure 4: zero-shot vs depth gap
# --------------------------------------------------------------------- #
def fig_zeroshot_gap():
    gaps = [(2, "B-8"), (4, "B-10"), (6, "B-12"), (8, "B-14")]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    plotted = False
    for m in ["msot", "clafusion_dp", "identity_pad_ot", "naive_averaging"]:
        xs, mus, sds = [], [], []
        for g, cfg in gaps:
            v = [r["zeroshot_accuracy"] for r in
                 q(tier="zeroshot", arch_pair=f"A_vs_{cfg}", method=m)
                 if "zeroshot_accuracy" in r]
            if v:
                xs.append(g); mus.append(np.mean(v))
                sds.append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
        if xs:
            ax.errorbar(xs, mus, yerr=sds, marker="o", ms=4, capsize=3,
                        label=LABELS[m], color=COLORS[m], lw=1.6)
            plotted = True
    if not plotted:
        print("  skip fig4: no zeroshot runs found")
        plt.close(fig)
        return
    ax.axhline(1.0, color="k", ls=":", lw=1.0)
    ax.text(2.1, 1.6, "chance", fontsize=7)
    ax.set_xticks([g for g, _ in gaps])
    ax.set_xlabel("Depth gap $|m-n|$")
    ax.set_ylabel("Zero-shot accuracy (%)")
    ax.set_title("Zero-shot retention vs. depth gap")
    ax.legend(fontsize=7.5, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(f"{OUT}/fig4_zeroshot_gap.pdf")
    plt.close(fig)
    print("  wrote fig4_zeroshot_gap.pdf")


# --------------------------------------------------------------------- #
# Figure 5: cross-dataset replication
# --------------------------------------------------------------------- #
def fig_datasets():
    def grab(tier, s, a):
        return [r["zeroshot_accuracy"] for r in q(tier=tier, method=f"mech_{s}_{a}")
                if "zeroshot_accuracy" in r]

    sets = [("CIFAR-100", "tier2mech", "zeroshot"), ("Tiny-ImageNet", "tinyimagenet", "tinyimagenet")]
    have = [(n, t, nt) for n, t, nt in sets if grab(t, "bures", "sinkhorn")]
    if len(have) < 2:
        print("  skip fig5: need both datasets' mechanism grids")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for ax, (name, tier, ntier) in zip(axes, have):
        bur = [v for a in ["sinkhorn", "hungarian", "none"] for v in grab(tier, "bures", a)]
        euc = [v for a in ["sinkhorn", "hungarian", "none"] for v in grab(tier, "euclidean", a)]
        nai = [r["zeroshot_accuracy"] for r in q(tier=ntier, method="naive_averaging")
               if "zeroshot_accuracy" in r]
        vals = [np.mean(bur), np.mean(euc), np.mean(nai) if nai else 0]
        errs = [np.std(bur, ddof=1), np.std(euc, ddof=1),
                np.std(nai, ddof=1) if len(nai) > 1 else 0]
        ax.bar(["Bures", "Euclidean", "Naive avg."], vals, yerr=errs, capsize=4,
               color=[COLORS["msot"], "#9edae5", COLORS["naive_averaging"]],
               edgecolor="k", lw=0.6)
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v + max(errs) + 0.4, f"{v:.1f}", ha="center", fontsize=8)
    axes[0].set_ylabel("Zero-shot accuracy (%)")
    fig.suptitle("The finding replicates across datasets", y=1.02)
    fig.savefig(f"{OUT}/fig5_datasets.pdf")
    plt.close(fig)
    print("  wrote fig5_datasets.pdf")


if __name__ == "__main__":
    print(f"Loaded {len(RUNS)} runs from {REG}\nWriting to {OUT}/\n")
    fig_crossover()
    fig_mechanism()
    fig_sweeps()
    fig_zeroshot_gap()
    fig_datasets()
    print("\nDone.")
