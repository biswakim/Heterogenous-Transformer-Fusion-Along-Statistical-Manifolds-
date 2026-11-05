"""
src/baselines/naive_averaging.py

Baseline: depth-match by truncation/duplication, then average weights
directly with no OT alignment at any scale (protocol Sec. 7).

This baseline isolates the TOTAL value of all three transport stages
combined: it is the same pipeline shape as MSOT (pair each target slot
with a Parent-A block, combine) with every transport component removed.
The gap between this row and the MSOT row in Table II is the headline
"does optimal transport buy us anything at all" number.
"""

from __future__ import annotations

from .common import build_child_from_blocks, depth_match_indices, average_blocks


def fuse_naive_averaging(parent_A, parent_B, weight_A: float = 0.5):
    """
    Returns a child of Parent B's depth whose block j is the plain
    elementwise average of Parent B's block j and the proportionally
    index-matched Parent A block -- no graph alignment, no barycentric
    synthesis, no neuron permutation alignment.
    """
    m, n = len(parent_A.blocks), len(parent_B.blocks)
    idx_map = depth_match_indices(m, n)

    fused_blocks = []
    for j in range(n):
        a_block = parent_A.blocks[idx_map[j]]
        b_block = parent_B.blocks[j]
        # weight_A weights Parent A's contribution, matching the `lam`
        # convention used by fuse_blocks_ot elsewhere in the codebase.
        fused_blocks.append(average_blocks(a_block, b_block, weight=weight_A))

    return build_child_from_blocks(parent_B, fused_blocks)
