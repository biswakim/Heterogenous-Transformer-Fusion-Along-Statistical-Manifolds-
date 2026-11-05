import torch
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment

def sinkhorn_neuron_alignment(source_rows, target_rows, reg=0.05):
    Mc = ot.dist(target_rows, source_rows, metric="sqeuclidean")
    Mc /= Mc.max() + 1e-9
    P = ot.sinkhorn(np.ones(target_rows.shape[0]) / target_rows.shape[0],
                     np.ones(source_rows.shape[0]) / source_rows.shape[0], Mc, reg)
    return torch.tensor(P * source_rows.shape[0], dtype=torch.float32)

def align_attention_heads_to_reference(ref_block, other_block, nhead, d_model, method="hungarian", reg=0.05):
    """
    Align `other_block`'s attention heads onto `ref_block`'s head ordering.

    `method` is the C1 ablation axis (protocol Sec. 8.3):
      'hungarian' -- hard one-to-one permutation (the proposed mechanism)
      'sinkhorn'  -- soft assignment, heads become weighted mixtures
      'none'      -- NO alignment; other_block's weights returned unchanged.
                     This is the arm that isolates whether permutation
                     alignment is doing any work at all.

    Whichever method is used, Q, K, V, and out_proj's input columns are
    transformed by the SAME correspondence -- aligning them independently
    would break attention geometry.
    """
    head_dim = d_model // nhead

    in_proj_w_o = other_block.attn.in_proj_weight.detach().cpu()
    in_proj_b_o = other_block.attn.in_proj_bias.detach().cpu()
    out_proj_w_o = other_block.attn.out_proj.weight.detach().cpu()
    out_proj_b_o = other_block.attn.out_proj.bias.detach().cpu().clone()

    if method == "none":
        # C1 control arm: return the other block's parameters untouched.
        return in_proj_w_o, in_proj_b_o, out_proj_w_o, out_proj_b_o

    Wq_r, Wk_r, Wv_r = ref_block.attn.in_proj_weight.detach().cpu().chunk(3, dim=0)
    Wq_o, Wk_o, Wv_o = in_proj_w_o.chunk(3, dim=0)

    def head_signature(Wq, Wk, Wv):
        sig = []
        for h in range(nhead):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            sig.append(torch.cat([Wq[sl], Wk[sl], Wv[sl]], dim=0).flatten())
        return torch.stack(sig)

    sig_r = head_signature(Wq_r, Wk_r, Wv_r).numpy()
    sig_o = head_signature(Wq_o, Wk_o, Wv_o).numpy()
    cost = ot.dist(sig_r, sig_o, metric="sqeuclidean")

    Wq_o2, Wk_o2, Wv_o2 = in_proj_w_o.chunk(3, dim=0)
    bq_o2, bk_o2, bv_o2 = in_proj_b_o.chunk(3, dim=0)

    if method == "hungarian":
        row_ind, col_ind = linear_sum_assignment(cost)
        perm = col_ind[np.argsort(row_ind)]
        idx = torch.cat([torch.arange(p * head_dim, (p + 1) * head_dim) for p in perm])

        in_proj_weight_aligned = torch.cat([Wq_o2[idx], Wk_o2[idx], Wv_o2[idx]], dim=0)
        in_proj_bias_aligned = torch.cat([bq_o2[idx], bk_o2[idx], bv_o2[idx]], dim=0)
        out_proj_w_aligned = out_proj_w_o[:, idx]

    elif method == "sinkhorn":
        # Soft head assignment: build an (nhead x nhead) plan, then expand it
        # to a (d_model x d_model) block-structured mixing matrix so whole
        # heads move together rather than individual rows.
        Mc = cost / (cost.max() + 1e-9)
        u = np.ones(nhead) / nhead
        P = ot.sinkhorn(u, u, Mc, reg)
        P = torch.tensor(P * nhead, dtype=torch.float32)
        S = torch.kron(P, torch.eye(head_dim))  # (d_model, d_model)

        in_proj_weight_aligned = torch.cat([S @ Wq_o2, S @ Wk_o2, S @ Wv_o2], dim=0)
        in_proj_bias_aligned = torch.cat([S @ bq_o2, S @ bk_o2, S @ bv_o2], dim=0)
        out_proj_w_aligned = out_proj_w_o @ S.t()

    else:
        raise ValueError(f"Unknown head alignment method: '{method}'. "
                         "Choose 'hungarian', 'sinkhorn', or 'none'.")

    return in_proj_weight_aligned, in_proj_bias_aligned, out_proj_w_aligned, out_proj_b_o
