"""
src/utils/registry.py

Persist every run's config + metrics, keyed by
(tier, dataset, arch_pair, method, parent_seed, fusion_seed) -- this is
Algorithm 2's final step ("Persist run config + all logged metrics keyed
by (tier, dataset, arch-pair, method, seed) for the statistical analysis
pass") and the input the whole of Sec. 11's statistics plan reads from.

Format is append-only JSON Lines (one JSON object per line):
  - append-only means a crashed or killed run never corrupts earlier
    results, which matters when Tier 1 is ~345 runs over multiple days;
  - one-object-per-line means you can inspect/grep results mid-run
    without waiting for the program to finish or holding it all in memory.

Nothing here imports torch, so results can be analyzed on any machine
(e.g. your laptop) without a GPU environment.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

DEFAULT_PATH = "results/raw/runs.jsonl"

# The fields that jointly identify a run. Any two records sharing all of
# these are repeats of the same cell and should not both exist unless you
# deliberately reran (see `already_logged`).
KEY_FIELDS = ("tier", "dataset", "arch_pair", "method", "parent_seed", "fusion_seed")


def log_run(record: dict, path: str = DEFAULT_PATH) -> dict:
    """
    Append one run record. `record` should contain the KEY_FIELDS plus
    whatever metrics that run produced (Sec. 10's metric set: accuracy,
    test loss, epochs-to-recovery, fusion wall-clock, latency, params,
    plus the fusion-quality diagnostics).

    A UTC timestamp is added automatically. Returns the record as written.
    """
    missing = [k for k in KEY_FIELDS if k not in record]
    if missing:
        raise ValueError(f"Run record is missing required key field(s): {missing}")

    record = dict(record)
    record["logged_at"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def load_runs(path: str = DEFAULT_PATH) -> list[dict]:
    """Read every logged run. Returns [] if the file doesn't exist yet."""
    if not os.path.exists(path):
        return []
    runs = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                # A partially-written final line can happen if a run was
                # killed mid-write. Skip it loudly rather than crashing
                # the whole analysis pass over one truncated record.
                print(f"[registry] WARNING: skipping malformed line {line_no} in {path}")
    return runs


def run_key(record: dict) -> tuple:
    return tuple(record.get(k) for k in KEY_FIELDS)


def already_logged(record: dict, path: str = DEFAULT_PATH) -> bool:
    """
    True if a run with these exact key fields is already recorded. Lets a
    runner skip cells it has already completed, so an interrupted Tier-1
    sweep can be resumed without redoing days of finished work.
    """
    target = run_key(record)
    return any(run_key(r) == target for r in load_runs(path))


def query(path: str = DEFAULT_PATH, **filters) -> list[dict]:
    """
    Return runs matching all given field==value filters, e.g.
        query(tier="tier0", method="msot")
    Filters may name any field, not only KEY_FIELDS.
    """
    out = []
    for r in load_runs(path):
        if all(r.get(k) == v for k, v in filters.items()):
            out.append(r)
    return out


def collect_metric(metric: str, path: str = DEFAULT_PATH, **filters) -> list[float]:
    """
    Pull one metric across all runs matching `filters`, as a plain list of
    floats -- the standard input shape for src/stats/analysis.py.

    Runs matching the filters but missing the metric are skipped with a
    warning, since a silently-short list would quietly change the
    denominator of a mean/std.
    """
    values, skipped = [], 0
    for r in query(path=path, **filters):
        if metric in r and r[metric] is not None:
            values.append(float(r[metric]))
        else:
            skipped += 1
    if skipped:
        print(f"[registry] WARNING: {skipped} run(s) matched {filters} but had no '{metric}' field")
    return values
