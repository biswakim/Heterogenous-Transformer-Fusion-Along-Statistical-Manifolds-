"""
src/experiments/methods.py

Single dispatch point mapping a method name (+ its hyperparameters) to a
fused child model. Both the Tier-1 baseline sweep and the Tier-2 ablation
grid call through here, so a method can never be constructed two slightly
different ways in two different runners -- which would silently make the
main table and the ablation table incomparable.

Method names used throughout the registry and all result tables:

    parent_a          Parent A alone (no fusion)
    parent_b          Parent B alone (no fusion)
    logit_ensemble    Softmax averaging of both parents (upper reference)
    naive_averaging   Depth-match + weight average, no OT anywhere
    identity_pad_ot   Identity padding + layer-wise micro-OT
    clafusion_dp      Monotonic DP macro-alignment + micro-OT
    msot              The proposed full pipeline

Ablation variants are NOT separate method names: they are `msot` with
overridden hyperparameters (alpha/beta/eps/lambda/mechanism switches),
recorded in the run record's own fields. This keeps "which method" and
"which configuration of that method" as separate, independently
queryable dimensions in the registry.
"""

from __future__ import annotations

import copy

import torch

from src.fusion.pipeline import MSOTFusionPipeline
from src.fusion.pipelinewidth import MSOTWidthPipeline
from src.fusion.phase5width import project_block_dimensions
from src.baselines.naive_averaging import fuse_naive_averaging
from src.baselines.identity_pad_ot import fuse_identity_pad_ot
from src.baselines.clafusion_dp import fuse_clafusion_dp, fuse_clafusion_dp_width
from src.baselines.logit_ensemble import build_logit_ensemble

FUSION_METHODS = ["naive_averaging", "identity_pad_ot", "clafusion_dp", "msot"]
NON_FUSION_METHODS = ["parent_a", "parent_b", "logit_ensemble"]
ALL_METHODS = NON_FUSION_METHODS + FUSION_METHODS

# Methods that produce a trainable child and therefore get fine-tuned.
# parent_a/parent_b/logit_ensemble are evaluated directly -- fine-tuning a
# parent would make it a different (better-trained) model and destroy its
# role as a fixed reference point.
NEEDS_FINETUNE = set(FUSION_METHODS)


def _widths_differ(parent_A, parent_B) -> bool:
    return parent_A.d_model != parent_B.d_model


def _project_parent_to_width(parent_A, parent_B):
    """
    Lift every one of Parent A's blocks into Parent B's dimensions using
    the same Phase-5 mechanism the width pipeline uses, returning a
    shallow stand-in object exposing `.blocks` and `.d_model`.

    WHY THIS EXISTS: the three fusion baselines (naive averaging, identity
    padding, CLAFusion DP) are all defined for same-width parents -- they
    average or OT-align tensors elementwise/columnwise, which is undefined
    when d_model differs. To run them at all on Variation-2 pairs, Parent
    A must first be brought into Parent B's shape.

    IMPORTANT INTERPRETATION CAVEAT: this means every Variation-2 baseline
    number is "baseline applied AFTER zero-pad width projection", not the
    baseline as originally published. The projection is a component of the
    proposed framework (Phase 0/5), so a Variation-2 table comparing MSOT
    against these baselines is partly comparing MSOT against itself-plus-a
    -weaker-back-end. State this explicitly in the paper, or restrict the
    headline baseline comparison to Variation 1 (same width), where no
    projection is needed and the comparison is clean. Flagging rather than
    silently deciding, since it affects what Table II can claim.
    """
    class _ProjectedParent:
        pass

    proj = _ProjectedParent()
    d_model = parent_B.d_model
    nhead = parent_B.blocks[0].attn.num_heads
    dim_ff = parent_B.blocks[0].ff[0].out_features
    proj.blocks = torch.nn.ModuleList(
        [project_block_dimensions(b, d_model, nhead, dim_ff) for b in parent_A.blocks]
    )
    proj.d_model = d_model
    return proj


def build_method(method: str, parent_A, parent_B, calibration_batch, **hp):
    """
    Construct the model for `method`.

    Returns (model, diagnostics). `diagnostics` is a dict of
    method-specific mechanism info (transport sparsity, DP mapping,
    identity-slot count, ...) to be merged into the run record; it is
    empty for methods that expose none.

    Recognized hyperparameters (all optional, protocol defaults applied):
        alpha, beta, eps_fgw, eps_sink   -- MSOT phases I/II/III/IV
        lam                              -- micro-scale interpolation weight
    """
    alpha = hp.get("alpha", 0.4)
    beta = hp.get("beta", 0.5)
    eps_fgw = hp.get("eps_fgw", 0.02)
    eps_sink = hp.get("eps_sink", 0.02)
    lam = hp.get("lam", 0.5)

    if method == "parent_a":
        return copy.deepcopy(parent_A), {}
    if method == "parent_b":
        return copy.deepcopy(parent_B), {}
    if method == "logit_ensemble":
        return build_logit_ensemble(parent_A, parent_B), {}

    # MSOT handles width heterogeneity natively via its own Phase 0 +
    # Phase 5. The three fusion baselines do not, so Parent A is projected
    # into Parent B's width first -- see _project_parent_to_width for the
    # interpretation caveat this introduces for Variation-2 tables.
    width_mismatch = _widths_differ(parent_A, parent_B)
    extra_diag = {"width_projected_for_baseline": bool(width_mismatch)} if width_mismatch else {}
    parent_A_eff = _project_parent_to_width(parent_A, parent_B) if width_mismatch else parent_A

    if method == "naive_averaging":
        return fuse_naive_averaging(parent_A_eff, parent_B, weight_A=lam), dict(extra_diag)

    if method == "identity_pad_ot":
        child, diag = fuse_identity_pad_ot(parent_A_eff, parent_B, lam=lam, epsilon=eps_sink)
        return child, {**diag, **extra_diag}

    if method == "clafusion_dp":
        if width_mismatch:
            # construct_computational_graph needs a real forward pass, which
            # the projected stand-in cannot provide -- so features come from
            # the true Parent A, while fusion uses the projected blocks.
            child, diag = fuse_clafusion_dp_width(
                parent_A, parent_A_eff, parent_B, calibration_batch,
                beta=beta, lam=lam, epsilon=eps_sink
            )
        else:
            child, diag = fuse_clafusion_dp(parent_A, parent_B, calibration_batch,
                                            beta=beta, lam=lam, epsilon=eps_sink)
        return child, {**diag, **extra_diag}

    if method == "msot":
        if width_mismatch:
            pipeline = MSOTWidthPipeline(
                parent_A, parent_B, alpha=alpha, beta=beta,
                epsilon_fgw=eps_fgw, epsilon_sink=eps_sink,
                width_alignment_method=hp.get("width_alignment_method", "zero_pad"),
            )
        else:
            pipeline = MSOTFusionPipeline(parent_A, parent_B, alpha=alpha, beta=beta,
                                          epsilon_fgw=eps_fgw, epsilon_sink=eps_sink)
        child = pipeline.fuse(calibration_batch)
        return child, dict(extra_diag)

    raise ValueError(f"Unknown method '{method}'. Choose from {ALL_METHODS}")


def method_display_name(method: str) -> str:
    """Row labels matching the results-table template."""
    return {
        "parent_a": "Parent A alone",
        "parent_b": "Parent B alone",
        "logit_ensemble": "Logit ensemble",
        "naive_averaging": "Naive averaging",
        "identity_pad_ot": "Identity pad. + OT",
        "clafusion_dp": "CLAFusion-style DP",
        "msot": "MSOT (proposed)",
    }.get(method, method)
