"""
src/baselines/common.py

Shared plumbing for the baseline fusion methods of protocol Sec. 7.

Every fusion baseline (and MSOT itself) produces a child with Parent B's
depth, width, and non-block parameters (patch embedding, positional
embedding, cls token, classification head), differing only in how the
per-block parameters are derived. Keeping that construction in one place
means a baseline can never accidentally "win" or "lose" because it
inherited a different head than the proposed method did.
"""

from __future__ import annotations

import copy

import torch


def build_child_from_blocks(parent_B, fused_blocks):
    """
    Deep-copy Parent B (so patch_embed / pos_embed / cls_token / head all
    come from the deeper parent, matching the pilot report's Sec. 6.3) and
    overwrite its blocks with `fused_blocks`.
    """
    if len(fused_blocks) != len(parent_B.blocks):
        raise ValueError(
            f"Expected {len(parent_B.blocks)} fused blocks to match Parent B's depth, got {len(fused_blocks)}"
        )
    child = copy.deepcopy(parent_B)
    for j, blk in enumerate(fused_blocks):
        child.blocks[j].load_state_dict(blk.state_dict())
    return child


def depth_match_indices(m: int, n: int) -> list[int]:
    """
    Map each of the n target slots to one of Parent A's m blocks by
    proportional nearest-index (duplication when m < n, truncation when
    m > n). This is the "depth-match by truncation/duplication" step named
    in the Naive-averaging baseline, deliberately containing no transport
    or alignment of any kind.
    """
    return [min(int(j * m / n), m - 1) for j in range(n)]


@torch.no_grad()
def average_blocks(block_1, block_2, weight: float = 0.5):
    """
    Plain elementwise weighted average of two blocks' parameters, with NO
    permutation alignment -- the thing OT-based fusion is supposed to
    improve on. Returns a new block (a copy of block_1's structure).
    """
    out = copy.deepcopy(block_1)
    sd_1 = block_1.state_dict()
    sd_2 = block_2.state_dict()
    if set(sd_1.keys()) != set(sd_2.keys()):
        raise ValueError("Blocks must share identical parameter names to be averaged")

    merged = {}
    for k in sd_1:
        if sd_1[k].shape != sd_2[k].shape:
            raise ValueError(f"Shape mismatch on '{k}': {sd_1[k].shape} vs {sd_2[k].shape}")
        merged[k] = weight * sd_1[k].float() + (1.0 - weight) * sd_2[k].float()
        merged[k] = merged[k].to(sd_1[k].dtype)
    out.load_state_dict(merged)
    return out
