"""
src/stats/analysis.py

Implements the protocol's Statistical Analysis Plan (Sec. 11) end to end:

  1. every reported number is mean +/- std over seeds (never a single run);
  2. the primary comparison is a PAIRED test across matching parent seeds
     -- paired t-test if a Shapiro-Wilk check says the differences are
     approximately normal, otherwise Wilcoxon signed-rank -- with Cohen's
     d reported alongside the p-value;
  3. Holm-Bonferroni correction is applied WITHIN each tier's family of
     tests before any claim of significance;
  4. sweep-style ablations report the full curve, so this module also
     provides the coefficient of variation used as the Sec. 16
     sensitivity measure for RQ6.

Pairing note: `paired_comparison` requires the two input lists to be
ordered by the SAME parent seeds. Pairing is the entire reason this test
has power here -- parent-seed-to-parent-seed variation is large relative
to the method differences being measured, and an unpaired test would
mostly be measuring that shared noise. `compare_methods_from_registry`
below does the pairing by seed key explicitly rather than assuming two
lists happen to line up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


# --------------------------------------------------------------------- #
# Descriptive
# --------------------------------------------------------------------- #
def mean_std(values) -> tuple[float, float]:
    """Sample mean and standard deviation (ddof=1), the form reported in every table."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        # A single run has no variance estimate. Returning 0.0 here would
        # imply a precision that doesn't exist, so return nan and let the
        # caller decide how to render it.
        return float(arr[0]), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1))


def format_mean_std(values, decimals: int = 2) -> str:
    """Render as 'mean +/- std' for direct paste into a results table."""
    m, s = mean_std(values)
    if math.isnan(m):
        return "--"
    if math.isnan(s):
        return f"{m:.{decimals}f} (single run, no variance)"
    return f"{m:.{decimals}f} +/- {s:.{decimals}f}"


def coefficient_of_variation(values) -> float:
    """
    std / |mean|, the sensitivity measure Sec. 16 specifies for RQ6
    ("the paper reports the coefficient of variation of accuracy across
    each sweep as the sensitivity measure").
    """
    m, s = mean_std(values)
    if math.isnan(s) or m == 0:
        return float("nan")
    return float(s / abs(m))


# --------------------------------------------------------------------- #
# Effect size
# --------------------------------------------------------------------- #
def cohens_d_paired(a, b) -> float:
    """
    Cohen's d for paired samples: mean(difference) / std(difference).

    Reported alongside every p-value per Sec. 11 point 2, because with
    small seed counts a p-value alone conflates "large effect" with
    "lucky variance" -- the effect size is what tells you whether a
    significant difference is also a meaningful one.
    """
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Paired inputs must have equal length, got {a_arr.shape} and {b_arr.shape}")
    diff = a_arr - b_arr
    if diff.size < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd == 0:
        return float("inf") if diff.mean() != 0 else 0.0
    return float(diff.mean() / sd)


# --------------------------------------------------------------------- #
# Paired significance test
# --------------------------------------------------------------------- #
@dataclass
class ComparisonResult:
    label: str
    n_pairs: int
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float
    mean_difference: float
    test_used: str
    statistic: float
    p_value: float
    cohens_d: float
    normality_p: float
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def paired_comparison(a, b, label: str = "", normality_alpha: float = 0.05) -> ComparisonResult:
    """
    Compare paired measurements `a` (e.g. proposed) against `b` (baseline).

    Test selection follows Sec. 11 point 2: Shapiro-Wilk on the paired
    DIFFERENCES, then paired t-test if normal, Wilcoxon signed-rank
    otherwise.
    """
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Paired inputs must have equal length, got {a_arr.size} and {b_arr.size}")

    n = a_arr.size
    mean_a, std_a = mean_std(a_arr)
    mean_b, std_b = mean_std(b_arr)
    diff = a_arr - b_arr

    if n < 3:
        return ComparisonResult(
            label=label, n_pairs=n, mean_a=mean_a, std_a=std_a, mean_b=mean_b, std_b=std_b,
            mean_difference=float(diff.mean()) if n else float("nan"),
            test_used="none", statistic=float("nan"), p_value=float("nan"),
            cohens_d=cohens_d_paired(a_arr, b_arr) if n >= 2 else float("nan"),
            normality_p=float("nan"),
            note=f"Only {n} pair(s); too few for a meaningful significance test. Report descriptively only.",
        )

    if np.allclose(diff, 0):
        return ComparisonResult(
            label=label, n_pairs=n, mean_a=mean_a, std_a=std_a, mean_b=mean_b, std_b=std_b,
            mean_difference=0.0, test_used="none", statistic=float("nan"), p_value=1.0,
            cohens_d=0.0, normality_p=float("nan"),
            note="All paired differences are exactly zero; methods produced identical results.",
        )

    normality_p = float(stats.shapiro(diff).pvalue)

    if normality_p >= normality_alpha:
        res = stats.ttest_rel(a_arr, b_arr)
        test_used, statistic, p_value = "paired t-test", float(res.statistic), float(res.pvalue)
    else:
        res = stats.wilcoxon(a_arr, b_arr)
        test_used, statistic, p_value = "Wilcoxon signed-rank", float(res.statistic), float(res.pvalue)

    return ComparisonResult(
        label=label, n_pairs=n, mean_a=mean_a, std_a=std_a, mean_b=mean_b, std_b=std_b,
        mean_difference=float(diff.mean()), test_used=test_used, statistic=statistic,
        p_value=p_value, cohens_d=cohens_d_paired(a_arr, b_arr), normality_p=normality_p,
    )


# --------------------------------------------------------------------- #
# Multiple-comparison correction
# --------------------------------------------------------------------- #
def holm_bonferroni(results: list[ComparisonResult], alpha: float = 0.05) -> list[dict]:
    """
    Apply Holm-Bonferroni correction across a FAMILY of comparisons
    (Sec. 11 point 3: "within each tier's family of tests before any claim
    of statistical significance").

    Returns the results as dicts, sorted by ascending p-value, each with
    added 'adjusted_alpha', 'significant', and 'rank' fields.

    Holm proceeds in ascending p order, comparing the k-th smallest
    p-value against alpha/(K-k). Once one test fails, every remaining
    (larger) p-value is non-significant regardless of its own threshold --
    this step-down behaviour is why it's applied here rather than
    thresholding each test independently.
    """
    valid = [r for r in results if not math.isnan(r.p_value)]
    skipped = [r for r in results if math.isnan(r.p_value)]

    ordered = sorted(valid, key=lambda r: r.p_value)
    K = len(ordered)
    out, still_significant = [], True

    for k, r in enumerate(ordered):
        adjusted_alpha = alpha / (K - k) if K - k > 0 else alpha
        if still_significant and r.p_value <= adjusted_alpha:
            significant = True
        else:
            significant = False
            still_significant = False
        d = r.to_dict()
        d.update({"rank": k + 1, "adjusted_alpha": adjusted_alpha, "significant": significant})
        out.append(d)

    for r in skipped:
        d = r.to_dict()
        d.update({"rank": None, "adjusted_alpha": None, "significant": False})
        out.append(d)

    return out


# --------------------------------------------------------------------- #
# Registry integration
# --------------------------------------------------------------------- #
def compare_methods_from_registry(
    method_a: str, method_b: str, metric: str = "final_accuracy",
    path: str = "results/raw/runs.jsonl", **filters,
) -> ComparisonResult:
    """
    Pull two methods' runs from the registry and compare them PAIRED BY
    SEED, so pairing is by matching (parent_seed, fusion_seed) rather than
    by list position.

    Any seed present for one method but not the other is dropped from the
    comparison with a warning, since an unmatched run would break pairing.
    """
    from src.utils.registry import query

    runs_a = {(r.get("parent_seed"), r.get("fusion_seed")): r for r in query(path=path, method=method_a, **filters)}
    runs_b = {(r.get("parent_seed"), r.get("fusion_seed")): r for r in query(path=path, method=method_b, **filters)}

    shared = sorted(set(runs_a) & set(runs_b), key=lambda k: (str(k[0]), str(k[1])))
    only_a, only_b = set(runs_a) - set(runs_b), set(runs_b) - set(runs_a)
    if only_a or only_b:
        print(f"[stats] WARNING: dropping unpaired seeds -- {len(only_a)} only in '{method_a}', "
              f"{len(only_b)} only in '{method_b}'. Comparing the {len(shared)} shared seed(s).")

    a_vals = [float(runs_a[k][metric]) for k in shared if metric in runs_a[k]]
    b_vals = [float(runs_b[k][metric]) for k in shared if metric in runs_b[k]]

    return paired_comparison(a_vals, b_vals, label=f"{method_a} vs {method_b} ({metric})")


def summarize_family(results: list[ComparisonResult], alpha: float = 0.05) -> str:
    """Render a corrected family of comparisons as a readable text table."""
    corrected = holm_bonferroni(results, alpha=alpha)
    lines = [
        f"{'comparison':<44} {'n':>3} {'diff':>8} {'p':>9} {'adj_a':>8} {'d':>7}  sig",
        "-" * 92,
    ]
    for r in corrected:
        p = "nan" if r["p_value"] is None or math.isnan(r["p_value"]) else f"{r['p_value']:.4f}"
        adj = "--" if r["adjusted_alpha"] is None else f"{r['adjusted_alpha']:.4f}"
        d = "nan" if math.isnan(r["cohens_d"]) else f"{r['cohens_d']:.3f}"
        lines.append(
            f"{r['label'][:44]:<44} {r['n_pairs']:>3} {r['mean_difference']:>8.3f} "
            f"{p:>9} {adj:>8} {d:>7}  {'YES' if r['significant'] else 'no'}"
        )
        if r.get("note"):
            lines.append(f"    note: {r['note']}")
    return "\n".join(lines)
