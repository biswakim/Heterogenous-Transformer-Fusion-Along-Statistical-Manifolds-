"""
src/baselines/identity_pad_ot.py

Baseline: pad the shallow parent to the target depth with identity
blocks, then apply homogeneous-depth layer-wise OT fusion (protocol
Sec. 7).

This isolates the value of Phases II-III specifically: micro-scale OT
(Phase IV) is held fixed and identical to the proposed method, while
graph-based macro alignment is replaced by naive positional pairing and
barycentric synthesis is replaced by identity padding. It is the direct
test of the pilot report's central claim that identity padding
"introduces functionally inactive layers that neither contribute new
representations nor capture the intermediate computations implied by the
transport plan."

CAVEAT ON WHAT "IDENTITY" MEANS FOR A POST-LN BLOCK
-----------------------------------------------------------------------
These parents use POST-layer-norm blocks:
    x = ln1(x + attn(x));  x = ln2(x + ff(x))
Zeroing the residual branches (attention output projection and the FFN's
second linear layer) gives block(x) = ln2(ln1(x)) -- NOT a true identity,
because the two LayerNorms still normalize. A genuine identity is not
achievable for a post-LN block without also neutralizing the norms, which
would change the block's type rather than its parameters.

This is the standard practical construction and is what "identity
padding" means in the post-LN setting, but it is worth stating explicitly
in the paper rather than implying a mathematically exact identity --
otherwise a reviewer who checks will find the discrepancy themselves.
"""

from __future__ import annotations

import copy

import torch

from src.fusion.phase3meso import fuse_blocks_ot
from .common import build_child_from_blocks


@torch.no_grad()
def make_identity_block(reference_block):
    """
    Build a residual-neutralized copy of `reference_block`: attention
    output projection and FFN output layer zeroed, LayerNorms left at
    their trained values. See the module docstring for why this is an
    approximate rather than exact identity.
    """
    blk = copy.deepcopy(reference_block)
    blk.attn.out_proj.weight.zero_()
    if blk.attn.out_proj.bias is not None:
        blk.attn.out_proj.bias.zero_()
    blk.ff[3].weight.zero_()
    if blk.ff[3].bias is not None:
        blk.ff[3].bias.zero_()
    return blk


def pad_to_depth(parent_A, target_depth: int):
    """
    Place Parent A's m real blocks at evenly spaced positions across
    `target_depth` slots, filling the gaps with identity blocks.

    Returns (padded_blocks, real_positions) where real_positions lists the
    slot indices holding genuine (non-identity) Parent-A blocks -- logged
    so the paper can report how many slots were functionally inactive
    under this baseline.
    """
    m = len(parent_A.blocks)
    if m > target_depth:
        raise ValueError(f"Cannot identity-pad a {m}-block parent up to {target_depth} slots")
    if m == target_depth:
        return [copy.deepcopy(b) for b in parent_A.blocks], list(range(m))

    if m == 1:
        positions = [0]
    else:
        positions = [round(i * (target_depth - 1) / (m - 1)) for i in range(m)]

    padded = [None] * target_depth
    for i, pos in enumerate(positions):
        padded[pos] = copy.deepcopy(parent_A.blocks[i])
    for j in range(target_depth):
        if padded[j] is None:
            padded[j] = make_identity_block(parent_A.blocks[0])

    return padded, positions


def fuse_identity_pad_ot(parent_A, parent_B, lam: float = 0.5, epsilon: float = 0.02):
    """
    Pad Parent A to Parent B's depth with identity blocks, then fuse slot
    j of the padded parent with Parent B's block j using the SAME
    micro-scale OT machinery the proposed method uses (fuse_blocks_ot),
    so the only difference from MSOT is macro alignment + synthesis.
    """
    n = len(parent_B.blocks)
    d_model = parent_B.d_model
    nhead = parent_B.blocks[0].attn.num_heads
    dim_ff = parent_B.blocks[0].ff[0].out_features

    padded_blocks, real_positions = pad_to_depth(parent_A, n)

    fused_blocks = []
    for j in range(n):
        fused = fuse_blocks_ot(
            parent_B.blocks[j], padded_blocks[j], d_model, nhead, dim_ff, lam=lam, epsilon=epsilon
        )
        fused_blocks.append(fused)

    child = build_child_from_blocks(parent_B, fused_blocks)
    diagnostics = {
        "n_identity_slots": n - len(real_positions),
        "real_block_positions": real_positions,
    }
    return child, diagnostics
