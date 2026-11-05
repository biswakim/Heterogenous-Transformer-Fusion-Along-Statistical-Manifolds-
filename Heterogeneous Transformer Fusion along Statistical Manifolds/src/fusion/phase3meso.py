import torch
import ot
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.models.vitparents import TransformerBlock
from src.fusion.phase4micro import sinkhorn_neuron_alignment, align_attention_heads_to_reference

def align_ffn_to_reference(reference_block, other_block, epsilon=0.02, method="sinkhorn"):
    """
    Align `other_block`'s FFN hidden neurons onto `reference_block`'s ordering.

    `method` is the C1 ablation axis (protocol Sec. 8.3):
      'sinkhorn'  -- entropic soft assignment (the proposed mechanism)
      'hungarian' -- hard one-to-one permutation
      'none'      -- no alignment; weights returned unchanged

    FIXED (earlier): this used to call sinkhorn_neuron_alignment with no reg
    argument, silently falling back to that function's hardcoded reg=0.05
    regardless of what epsilon_sink was set on the pipeline.
    """
    W1_oth_t = other_block.ff[0].weight.detach().cpu()
    b1_oth_t = other_block.ff[0].bias.detach().cpu()
    W2_oth_t = other_block.ff[3].weight.detach().cpu()

    if method == "none":
        return W1_oth_t, b1_oth_t, W2_oth_t

    W1_ref = reference_block.ff[0].weight.detach().cpu().numpy()
    W1_oth = W1_oth_t.numpy()

    if method == "sinkhorn":
        P = sinkhorn_neuron_alignment(W1_oth, W1_ref, reg=epsilon)
    elif method == "hungarian":
        cost = ot.dist(W1_ref, W1_oth, metric="sqeuclidean")
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = col_ind[np.argsort(row_ind)]
        P = torch.zeros(W1_ref.shape[0], W1_oth.shape[0], dtype=torch.float32)
        P[torch.arange(len(perm)), torch.tensor(perm)] = 1.0
    else:
        raise ValueError(f"Unknown FFN alignment method: '{method}'. "
                         "Choose 'sinkhorn', 'hungarian', or 'none'.")

    W1_aligned = P @ W1_oth_t
    b1_aligned = P @ b1_oth_t
    W2_aligned = W2_oth_t @ P.t()

    return W1_aligned, b1_aligned, W2_aligned


def interpolate_weights(W_ref, W_other_aligned, lam, method="bures", reg=1e-3):
    """
    B1 ablation axis (protocol Sec. 8.2): how two aligned weight matrices
    are combined.

      'bures'     -- Bures-Wasserstein displacement interpolation (proposed)
      'euclidean' -- plain linear interpolation lam*W_ref + (1-lam)*W_other

    Identity padding, B1's third arm, is not a weight-combination rule --
    it means skipping synthesis entirely and reusing a parent block, so it
    is handled at the pipeline level rather than here.
    """
    if method == "euclidean":
        return lam * W_ref + (1.0 - lam) * W_other_aligned
    if method == "bures":
        return bures_wasserstein_interpolate(W_ref, W_other_aligned, lam, reg=reg)
    raise ValueError(f"Unknown synthesis method: '{method}'. Choose 'bures' or 'euclidean'.")


def bures_wasserstein_interpolate(W_ref, W_other_aligned, lam, reg=1e-3):
    device = W_ref.device
    n_rows = W_ref.shape[0]
    Wr = W_ref.detach().cpu().double().numpy()
    Wo = W_other_aligned.detach().cpu().double().numpy()
    t = 1.0 - lam

    if n_rows < Wr.shape[1] + 2:
        blended = (1 - t) * Wr + t * Wo
        return torch.tensor(blended, dtype=W_ref.dtype, device=device)

    try:
        A, b = ot.gaussian.empirical_bures_wasserstein_mapping(Wr, Wo, reg=reg)
        pushed = Wr @ A.T + b
        blended = (1 - t) * Wr + t * pushed
    except Exception:
        blended = (1 - t) * Wr + t * Wo

    return torch.tensor(blended, dtype=W_ref.dtype, device=device)

def fuse_blocks_ot(block_ref, block_other, d_model, nhead, dim_ff, lam=0.5, dropout=0.1,
                   epsilon=0.02, synthesis="bures", alignment="sinkhorn"):
    """
    `synthesis` -- B1 axis: 'bures' (proposed) | 'euclidean'
    `alignment` -- C1 axis: 'sinkhorn' (proposed) | 'hungarian' | 'none'

    Note the head-level default differs from the FFN default by design and
    matches the pilot report (Sec. 6.3): heads are aligned as atomic units
    via Hungarian assignment, FFN channels via entropic Sinkhorn. The C1
    ablation overrides BOTH together so the arm names mean "this mechanism
    everywhere", which is what makes the three arms cleanly comparable.
    """
    fused = TransformerBlock(d_model, nhead, dim_ff, dropout=dropout)

    # Head-level mechanism per C1 arm. Note the proposed configuration
    # ('sinkhorn') uses HUNGARIAN for heads and Sinkhorn for the FFN --
    # that asymmetry is the pilot report's own choice (Sec. 6.3: heads
    # aligned as atomic units), not an oversight.
    head_method = {
        "sinkhorn": "hungarian",   # proposed: Hungarian heads + Sinkhorn FFN
        "hungarian": "hungarian",
        "none": "none",
    }[alignment]

    in_w_o, in_b_o, out_w_o, out_b_o = align_attention_heads_to_reference(
        block_ref, block_other, nhead, d_model, method=head_method, reg=epsilon
    )
    
    in_proj_w_ref = block_ref.attn.in_proj_weight.detach().cpu()
    in_proj_b_ref = block_ref.attn.in_proj_bias.detach().cpu()
    out_proj_w_ref = block_ref.attn.out_proj.weight.detach().cpu()
    out_proj_b_ref = block_ref.attn.out_proj.bias.detach().cpu()

    with torch.no_grad():
        fused.attn.in_proj_weight.copy_(interpolate_weights(in_proj_w_ref, in_w_o, lam, synthesis))
        fused.attn.in_proj_bias.copy_(lam * in_proj_b_ref + (1 - lam) * in_b_o)
        fused.attn.out_proj.weight.copy_(interpolate_weights(out_proj_w_ref, out_w_o, lam, synthesis))
        fused.attn.out_proj.bias.copy_(lam * out_proj_b_ref + (1 - lam) * out_b_o)

        W1_aligned, b1_aligned, W2_aligned = align_ffn_to_reference(
            block_ref, block_other, epsilon=epsilon, method=alignment
        )
        fused.ff[0].weight.copy_(interpolate_weights(block_ref.ff[0].weight.detach().cpu(), W1_aligned, lam, synthesis))
        fused.ff[0].bias.copy_(lam * block_ref.ff[0].bias.detach().cpu() + (1 - lam) * b1_aligned)
        
        w2_ref_t = block_ref.ff[3].weight.detach().cpu().t()
        w2_aligned_t = W2_aligned.t()
        fused.ff[3].weight.copy_(interpolate_weights(w2_ref_t, w2_aligned_t, lam, synthesis).t())
        fused.ff[3].bias.copy_(lam * block_ref.ff[3].bias.detach().cpu() + (1 - lam) * block_other.ff[3].bias.detach().cpu())

        fused.ln1.weight.copy_(lam * block_ref.ln1.weight.detach().cpu() + (1 - lam) * block_other.ln1.weight.detach().cpu())
        fused.ln1.bias.copy_(lam * block_ref.ln1.bias.detach().cpu() + (1 - lam) * block_other.ln1.bias.detach().cpu())
        fused.ln2.weight.copy_(lam * block_ref.ln2.weight.detach().cpu() + (1 - lam) * block_other.ln2.weight.detach().cpu())
        fused.ln2.bias.copy_(lam * block_ref.ln2.bias.detach().cpu() + (1 - lam) * block_other.ln2.bias.detach().cpu())

    return fused
