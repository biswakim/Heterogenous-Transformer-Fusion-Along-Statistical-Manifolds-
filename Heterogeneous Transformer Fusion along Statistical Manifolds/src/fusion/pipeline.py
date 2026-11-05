import torch
import copy
from src.fusion.phase1graph import construct_computational_graph
from src.fusion.phase2macro import macro_scale_fgw_alignment, compute_interpolation_weights
from src.fusion.phase3meso import fuse_blocks_ot

class MSOTFusionPipeline:
    """
    synthesis -- B1 ablation axis (protocol Sec. 8.2):
        'bures'        Bures-Wasserstein barycentric synthesis (proposed)
        'euclidean'    plain linear interpolation at the same split slots
        'identity_pad' no synthesis at all: reuse the dominant Parent-A
                       block instead of building an intermediate one
    alignment -- C1 ablation axis (protocol Sec. 8.3):
        'sinkhorn'     Hungarian heads + entropic Sinkhorn FFN (proposed)
        'hungarian'    hard assignment throughout
        'none'         no permutation alignment anywhere
    """

    def __init__(self, parent_A, parent_B, alpha=0.4, beta=0.5, epsilon_fgw=0.02,
                 epsilon_sink=0.02, synthesis="bures", alignment="sinkhorn", verbose=True):
        self.parent_A = parent_A
        self.parent_B = parent_B
        self.alpha = alpha
        self.beta = beta
        self.epsilon_fgw = epsilon_fgw
        self.epsilon_sink = epsilon_sink
        self.synthesis = synthesis
        self.alignment = alignment
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _micro_synthesis(self):
        """
        The weight-interpolation rule used for MICRO-scale fusion (Parent B's
        block against its Parent-A counterpart).

        B1's 'identity_pad' arm is a meso-stage decision -- it means "don't
        synthesize an intermediate block at all" -- not an interpolation
        rule. The micro-scale fusion that follows still needs a valid rule,
        so identity_pad falls back to the proposed 'bures'. Without this,
        'identity_pad' would be passed to interpolate_weights and raise.
        """
        return "bures" if self.synthesis == "identity_pad" else self.synthesis

    def fuse(self, calibration_batch):
        self._log("Phase I: Constructing computational graphs...")
        X_A, A_A = construct_computational_graph(self.parent_A, calibration_batch, beta=self.beta)
        X_B, A_B = construct_computational_graph(self.parent_B, calibration_batch, beta=self.beta)

        self._log("Phase II: Computing FGW macro-scale alignment...")
        T_star = macro_scale_fgw_alignment(X_A, A_A, X_B, A_B, alpha=self.alpha, epsilon=self.epsilon_fgw)

        target_slots = compute_interpolation_weights(T_star)

        self._log("Phase III & IV: Synthesizing meso-scale blocks and micro-aligning parameters...")

        d_model = self.parent_B.d_model
        nhead = self.parent_B.blocks[0].attn.num_heads
        dim_ff = self.parent_B.blocks[0].ff[0].out_features

        fused_child = copy.deepcopy(self.parent_B)

        # For B1 reporting: how many slots actually went through synthesis.
        n_split = 0

        for slot_info in target_slots:
            j = slot_info["slot"]
            block_B = self.parent_B.blocks[j]

            if slot_info["split"]:
                n_split += 1
                idx_1, idx_2 = slot_info["parents"]
                lam = slot_info["lambda"]
                block_A_1 = self.parent_A.blocks[idx_1]
                block_A_2 = self.parent_A.blocks[idx_2]

                if self.synthesis == "identity_pad":
                    # B1 control arm: skip synthesis entirely, reuse whichever
                    # Parent-A block holds the majority of the transport mass.
                    # This is the "functionally inactive layer" heuristic the
                    # proposed method is meant to improve on.
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
                short_counterpart = self.parent_A.blocks[idx]
                self._log(f"  Slot {j}: 1-to-1 alignment with Parent A block {idx}")
                fused = fuse_blocks_ot(
                    block_B, short_counterpart, d_model, nhead, dim_ff, lam=0.5,
                    epsilon=self.epsilon_sink, synthesis=self._micro_synthesis(),
                    alignment=self.alignment,
                )

            fused_child.blocks[j].load_state_dict(fused.state_dict())

        self.n_split_slots = n_split
        self.n_single_slots = len(target_slots) - n_split
        self._log("Fusion Complete!")
        return fused_child
