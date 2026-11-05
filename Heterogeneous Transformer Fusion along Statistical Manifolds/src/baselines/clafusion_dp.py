"""
src/baselines/clafusion_dp.py

Baseline: CLAFusion-style dynamic-programming monotonic layer
correspondence (Nguyen et al., 2023) in place of FGW macro-alignment,
everything else unchanged (protocol Sec. 7).

This is the DIRECT competitor for RQ1 -- the paper's central claim is
that Transformer block correspondence is NOT monotonically tied to depth,
so a monotonic DP alignment should underperform FGW precisely because it
cannot match functionally similar blocks sitting at very different
depths. Holding Phases III-IV fixed and swapping only the macro stage is
what makes that comparison clean.

SCOPE NOTE: this is a faithful reimplementation of the monotonic
cross-layer alignment IDEA, not of the CLAFusion paper's full method
(which also handles its own balancing/normalization details). The paper
should describe it as "CLAFusion-style" -- as the protocol's own baseline
table already does -- rather than claiming a reproduction of their
published numbers.
"""

from __future__ import annotations

import numpy as np
import torch

from src.fusion.phase1graph import construct_computational_graph
from src.fusion.phase3meso import fuse_blocks_ot
from .common import build_child_from_blocks


def monotonic_dp_alignment(X_A: torch.Tensor, X_B: torch.Tensor) -> list[int]:
    """
    Assign each of Parent B's n blocks one Parent A block index, subject
    to the assignment being monotonically non-decreasing in slot index --
    the defining constraint of sequential cross-layer alignment.

    Cost is Euclidean distance between block feature embeddings. Endpoints
    are pinned: slot 0 aligns to A block 0 and slot n-1 to A block m-1
    (the standard DTW boundary condition), which guarantees the alignment
    spans A's full depth rather than collapsing onto a few blocks.

    Returns a list of length n mapping slot -> Parent A block index.
    """
    A = X_A.detach().cpu().double().numpy()
    B = X_B.detach().cpu().double().numpy()
    m, n = A.shape[0], B.shape[0]

    # C[i, j] = cost of assigning Parent A block i to target slot j
    C = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)

    INF = float("inf")
    D = np.full((m, n), INF)
    back = np.full((m, n), -1, dtype=int)

    # Boundary: slot 0 must use A block 0
    D[0, 0] = C[0, 0]

    for j in range(1, n):
        # running argmin over i' <= i of D[i', j-1], so an A block may
        # serve several consecutive slots (repeat) or the alignment may
        # advance past one or more A blocks (skip) -- both stay monotonic.
        best_val, best_idx = INF, -1
        for i in range(m):
            if D[i, j - 1] < best_val:
                best_val, best_idx = D[i, j - 1], i
            if best_val < INF:
                D[i, j] = C[i, j] + best_val
                back[i, j] = best_idx

    # Boundary: final slot must use A block m-1
    if D[m - 1, n - 1] == INF:
        raise RuntimeError(
            f"No monotonic alignment exists from {m} blocks to {n} slots under the pinned "
            "endpoint constraints. This should only happen if m > n."
        )

    # Backtrack. Cast to plain python int -- numpy integer types survive
    # json.dumps only via the default=str fallback, which would silently
    # turn logged indices into strings ("3" not 3) in the run registry.
    mapping = [0] * n
    i = m - 1
    for j in range(n - 1, -1, -1):
        mapping[j] = int(i)
        if j > 0:
            i = back[i, j]
    return mapping


def fuse_clafusion_dp(parent_A, parent_B, calibration_batch, beta: float = 0.5,
                      lam: float = 0.5, epsilon: float = 0.02):
    """
    Compute a monotonic DP correspondence from block features, then fuse
    each target slot with its assigned Parent-A block using the same
    micro-scale OT machinery as the proposed method.
    """
    X_A, _ = construct_computational_graph(parent_A, calibration_batch, beta=beta)
    X_B, _ = construct_computational_graph(parent_B, calibration_batch, beta=beta)

    mapping = monotonic_dp_alignment(X_A, X_B)

    n = len(parent_B.blocks)
    d_model = parent_B.d_model
    nhead = parent_B.blocks[0].attn.num_heads
    dim_ff = parent_B.blocks[0].ff[0].out_features

    fused_blocks = []
    for j in range(n):
        a_block = parent_A.blocks[mapping[j]]
        fused = fuse_blocks_ot(
            parent_B.blocks[j], a_block, d_model, nhead, dim_ff, lam=lam, epsilon=epsilon
        )
        fused_blocks.append(fused)

    child = build_child_from_blocks(parent_B, fused_blocks)
    diagnostics = {
        "dp_mapping": mapping,
        "n_distinct_A_blocks_used": len(set(mapping)),
    }
    return child, diagnostics


def fuse_clafusion_dp_width(parent_A_true, parent_A_projected, parent_B, calibration_batch,
                            beta: float = 0.5, lam: float = 0.5, epsilon: float = 0.02):
    """
    Width-heterogeneous variant of `fuse_clafusion_dp`.

    Splits the two roles Parent A plays: block FEATURES are extracted from
    the true (native-width) Parent A, because feature extraction needs a
    real forward pass that a projected stand-in cannot perform; block
    PARAMETERS come from the width-projected copy, because fusion needs
    shapes compatible with Parent B.

    Feature dimensions still differ between the parents here, so the DP
    cost matrix is computed after zero-padding the narrower feature set to
    the wider one -- the same naive reconciliation the width pipeline
    applies by default, kept identical so the baseline and the proposed
    method see the same feature geometry.
    """
    X_A, _ = construct_computational_graph(parent_A_true, calibration_batch, beta=beta)
    X_B, _ = construct_computational_graph(parent_B, calibration_batch, beta=beta)

    if X_A.shape[1] < X_B.shape[1]:
        X_A = torch.nn.functional.pad(X_A, (0, X_B.shape[1] - X_A.shape[1]))
    elif X_B.shape[1] < X_A.shape[1]:
        X_B = torch.nn.functional.pad(X_B, (0, X_A.shape[1] - X_B.shape[1]))

    mapping = monotonic_dp_alignment(X_A, X_B)

    n = len(parent_B.blocks)
    d_model = parent_B.d_model
    nhead = parent_B.blocks[0].attn.num_heads
    dim_ff = parent_B.blocks[0].ff[0].out_features

    fused_blocks = []
    for j in range(n):
        a_block = parent_A_projected.blocks[mapping[j]]
        fused = fuse_blocks_ot(
            parent_B.blocks[j], a_block, d_model, nhead, dim_ff, lam=lam, epsilon=epsilon
        )
        fused_blocks.append(fused)

    child = build_child_from_blocks(parent_B, fused_blocks)
    diagnostics = {
        "dp_mapping": mapping,
        "n_distinct_A_blocks_used": len(set(mapping)),
        "width_projected": True,
    }
    return child, diagnostics
