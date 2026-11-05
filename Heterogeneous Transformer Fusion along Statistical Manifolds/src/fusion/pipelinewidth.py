import torch
import copy
from src.fusion.phase1graph import construct_computational_graph
from src.fusion.phase2macro import macro_scale_fgw_alignment, compute_interpolation_weights
from src.fusion.phase3meso import fuse_blocks_ot
from src.fusion.phase5width import project_block_dimensions
from src.fusion.phase0width import zero_pad_or_truncate, random_orthogonal_projection, LearnedLinearProjection


def align_feature_widths(X_A, X_B, method="zero_pad", seed=None):
    """
    Reconcile X_A (L_A, d_A) and X_B (L_B, d_B) calibration-feature matrices
    to a common last-dimension width, for Phase I/II only (Sec 6, Phase 0).

    Only the FEATURES used for graph construction/alignment are touched
    here -- the actual block parameters keep their native widths and are
    reconciled separately for fusion via phase5width.project_block_dimensions.
    """
    d_A, d_B = X_A.shape[-1], X_B.shape[-1]
    if d_A == d_B:
        return X_A, X_B

    if method == "zero_pad":
        target_dim = max(d_A, d_B)
        return zero_pad_or_truncate(X_A, target_dim), zero_pad_or_truncate(X_B, target_dim)

    if method == "random_orthogonal":
        target_dim = min(d_A, d_B)
        return (
            random_orthogonal_projection(X_A, target_dim, seed=seed),
            random_orthogonal_projection(X_B, target_dim, seed=seed),
        )

    if method == "learned_linear":
        # No ground-truth row correspondence exists yet between X_A's and
        # X_B's blocks (that's Phase II's job, which hasn't run yet at this
        # point) -- so, purely to fit a projection DIRECTION, pair rows by
        # rank order of mean activation as a cheap proxy target.
        if d_A <= d_B:
            narrow, wide, narrow_is_A = X_A, X_B, True
        else:
            narrow, wide, narrow_is_A = X_B, X_A, False

        n = min(narrow.shape[0], wide.shape[0])
        narrow_order = torch.argsort(narrow.mean(dim=1))
        wide_order = torch.argsort(wide.mean(dim=1))
        narrow_matched = narrow[narrow_order[:n]]
        wide_matched = wide[wide_order[:n]]

        proj = LearnedLinearProjection(source_dim=wide.shape[-1], target_dim=narrow.shape[-1]).to(wide.device)
        proj.fit(wide_matched, narrow_matched)
        wide_projected = proj(wide).detach()

        return (narrow, wide_projected) if narrow_is_A else (wide_projected, narrow)

    raise ValueError(f"Unknown width_alignment_method: '{method}'. Choose from 'zero_pad', 'random_orthogonal', 'learned_linear'.")


class MSOTWidthPipeline:
    """
    Width-heterogeneous counterpart to MSOTFusionPipeline.

    ADDED (matching MSOTFusionPipeline): the B1/C1 mechanism switches, so
    the same decomposition experiment can be run on width-heterogeneous
    pairs.

      synthesis -- B1 axis (Sec 8.2):
          'bures'        Bures-Wasserstein barycentric synthesis (proposed)
          'euclidean'    plain linear interpolation at the same split slots
          'identity_pad' no synthesis: reuse the dominant Parent-A block

      alignment -- C1 axis (Sec 8.3):
          'sinkhorn'     Hungarian heads + entropic Sinkhorn FFN (proposed)
          'hungarian'    hard assignment throughout
          'none'         no permutation alignment anywhere

    NOTE ON WHAT WIDTH RECONCILIATION IS AND IS NOT. The `width_alignment_method`
    argument selects how the calibration FEATURES are reconciled for graph
    construction (Phase 0). It is independent of, and must not be confused
    with, the weight-space zero-padding in phase5width.project_block_dimensions,
    which is always applied and is not ablatable here -- differently shaped
    parameter matrices have to be made compatible somehow before any
    interpolation rule can act on them.
    """

    def __init__(
        self, parent_A, parent_B, alpha=0.4, beta=0.5, epsilon_fgw=0.02, epsilon_sink=0.02,
        width_alignment_method="zero_pad", width_alignment_seed=None,
        synthesis="bures", alignment="sinkhorn", verbose=True,
    ):
        self.parent_A = parent_A
        self.parent_B = parent_B
        self.alpha = alpha
        self.beta = beta
        self.epsilon_fgw = epsilon_fgw
        self.epsilon_sink = epsilon_sink
        self.width_alignment_method = width_alignment_method
        self.width_alignment_seed = width_alignment_seed
        self.synthesis = synthesis
        self.alignment = alignment
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _micro_synthesis(self):
        """
        Interpolation rule for MICRO-scale fusion. B1's 'identity_pad' arm
        is a meso-stage decision ("do not synthesize"), not an interpolation
        rule, so it falls back to 'bures' here. Without this, 'identity_pad'
        would be passed to interpolate_weights and raise.
        """
        return "bures" if self.synthesis == "identity_pad" else self.synthesis

    def fuse(self, calibration_batch):
        self._log("Phase I: Constructing computational graphs...")
        X_A, A_A = construct_computational_graph(self.parent_A, calibration_batch, beta=self.beta)
        X_B, A_B = construct_computational_graph(self.parent_B, calibration_batch, beta=self.beta)

        self._log(f"Phase 0: Reconciling feature widths via '{self.width_alignment_method}'...")
        X_A, X_B = align_feature_widths(X_A, X_B, method=self.width_alignment_method,
                                        seed=self.width_alignment_seed)

        self._log("Phase II: Computing FGW macro-scale alignment...")
        T_star = macro_scale_fgw_alignment(X_A, A_A, X_B, A_B, alpha=self.alpha, epsilon=self.epsilon_fgw)
        target_slots = compute_interpolation_weights(T_star)

        self._log("Phase III & IV: Synthesizing meso-scale blocks and micro-aligning parameters...")
        d_model = self.parent_B.d_model
        nhead = self.parent_B.blocks[0].attn.num_heads
        dim_ff = self.parent_B.blocks[0].ff[0].out_features

        fused_child = copy.deepcopy(self.parent_B)
        n_split = 0

        for slot_info in target_slots:
            j = slot_info["slot"]
            block_B = self.parent_B.blocks[j]

            if slot_info["split"]:
                n_split += 1
                idx_1, idx_2 = slot_info["parents"]
                lam = slot_info["lambda"]

                # Weight-space width reconciliation (always applied, see class docstring)
                block_A_1 = project_block_dimensions(self.parent_A.blocks[idx_1], d_model, nhead, dim_ff)
                block_A_2 = project_block_dimensions(self.parent_A.blocks[idx_2], d_model, nhead, dim_ff)

                if self.synthesis == "identity_pad":
                    short_counterpart = block_A_1 if lam >= 0.5 else block_A_2
                    self._log(f"  Slot {j}: identity-pad -> reusing Parent A block "
                              f"{idx_1 if lam >= 0.5 else idx_2} (no synthesis)")
                else:
                    self._log(f"  Slot {j}: Synthesizing block from Parent A blocks "
                              f"{idx_1} & {idx_2} (lam={lam:.3f}, {self.synthesis})")
                    short_counterpart = fuse_blocks_ot(
                        block_A_1, block_A_2, d_model, nhead, dim_ff, lam=lam,
                        epsilon=self.epsilon_sink, synthesis=self.synthesis,
                        alignment=self.alignment,
                    )

                fused = fuse_blocks_ot(
                    block_B, short_counterpart, d_model, nhead, dim_ff, lam=0.5,
                    epsilon=self.epsilon_sink, synthesis=self._micro_synthesis(),
                    alignment=self.alignment,
                )
            else:
                idx = slot_info["parents"][0]
                short_counterpart = project_block_dimensions(self.parent_A.blocks[idx], d_model, nhead, dim_ff)
                self._log(f"  Slot {j}: 1-to-1 alignment with Parent A block {idx}")
                fused = fuse_blocks_ot(
                    block_B, short_counterpart, d_model, nhead, dim_ff, lam=0.5,
                    epsilon=self.epsilon_sink, synthesis=self._micro_synthesis(),
                    alignment=self.alignment,
                )

            fused_child.blocks[j].load_state_dict(fused.state_dict())

        self.n_split_slots = n_split
        self.n_single_slots = len(target_slots) - n_split
        self._log("Width Fusion Complete!")
        return fused_child
