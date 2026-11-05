import torch
import ot
import numpy as np

def macro_scale_fgw_alignment(X_A, A_A, X_B, A_B, alpha=0.4, epsilon=0.02):
    # FIXED: M was previously raw (unnormalized) Euclidean distance, whose
    # scale depends entirely on the feature extractor's output magnitude.
    # With entropic regularization, the Gibbs kernel is exp(-M/epsilon), so
    # an unnormalized M with entries large relative to epsilon causes severe
    # numerical peaking -- this is what was producing the "Sinkhorn did not
    # converge" warnings. Rescaling M to [0, 1] makes epsilon's scale
    # consistent regardless of the raw feature magnitude. Verified this
    # eliminates the warning entirely at the default epsilon=0.02 in
    # testing; very low epsilon values (the tail of the A3 ablation sweep,
    # eps_fgw <= 0.01) can still trigger it occasionally -- that's an
    # inherent property of entropic OT at low regularization, not a bug,
    # and the resulting transport plan remains finite and properly
    # marginal-normalized even when it fires (verified separately).
    M_raw = torch.cdist(X_A, X_B).cpu().numpy()
    M = M_raw / (M_raw.max() + 1e-8)
    C1 = A_A.cpu().numpy()
    C2 = A_B.cpu().numpy()
    p = np.ones(M.shape[0]) / M.shape[0]
    q = np.ones(M.shape[1]) / M.shape[1]

    try:
        T = ot.gromov.entropic_fused_gromov_wasserstein(
            M, C1, C2, p, q, loss_fun="square_loss", epsilon=epsilon, alpha=alpha
        )
        if not np.all(np.isfinite(T)) or T.sum() < 1e-6:
            raise RuntimeError("entropic FGW returned a degenerate plan")
    except Exception as e:
        print(f"  [macro_align] entropic FGW failed ({e}); falling back to exact FGW.")
        T = ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, loss_fun="square_loss", alpha=alpha)
        
    return torch.tensor(T, dtype=torch.float32, device=X_A.device)
    
def compute_feature_cost_matrix(X_A, X_B):
    X_A_flat = X_A.view(X_A.shape[0], -1)
    X_B_flat = X_B.view(X_B.shape[0], -1)
    
    M = torch.cdist(X_A_flat, X_B_flat, p=2) ** 2
    M = M / (M.max() + 1e-8)
    return M

def compute_interpolation_weights(T_star, mass_threshold=0.01):
    """
    mass_threshold: minimum fraction of a column's OWN total mass a row must
    hold to count as an "active" contributor to that target slot.

    FIXED: previously used an absolute cutoff (`top2_vals[1] > 1e-4`) to
    decide split-vs-single. Entropic FGW plans (Sec 3.4) have full support
    by construction -- every entry in T_star is nonzero, just often
    vanishingly small -- so an absolute 1e-4 cutoff was classifying almost
    every slot as "split" even when >99% of the mass sat on one row (this
    is what produced 12/12 split slots instead of the pilot's 6/12). Using
    a threshold relative to each column's own mass distinguishes a
    genuinely competitive second contributor from numerical residue.
    """
    m, n = T_star.shape
    lambdas = []

    for j in range(n):
        col = T_star[:, j]
        col_sum = col.sum()
        active_mask = col > (mass_threshold * col_sum)
        active_idx = torch.nonzero(active_mask, as_tuple=True)[0]

        if active_idx.numel() == 0:
            # shouldn't happen for a valid transport plan, but guard anyway
            active_idx = col.argmax().unsqueeze(0)

        if active_idx.numel() == 1:
            lambdas.append({"slot": j, "split": False, "parents": (active_idx.item(),), "lambda": 1.0})
            continue

        # two highest-mass ACTIVE rows (generalizes Eq. 2's "two adjacent
        # blocks" to the top-2 contributors by mass -- real entropic plans
        # can spread mass across non-adjacent rows too)
        active_vals = col[active_idx]
        top2 = torch.topk(active_vals, 2)
        idx_1 = active_idx[top2.indices[0]].item()
        idx_2 = active_idx[top2.indices[1]].item()
        val_1, val_2 = top2.values[0], top2.values[1]

        if idx_1 > idx_2:
            idx_1, idx_2 = idx_2, idx_1
            val_1, val_2 = val_2, val_1

        lam = val_1 / (val_1 + val_2)  # Eq. 2, weight on the lower-indexed block
        lambdas.append({"slot": j, "split": True, "parents": (idx_1, idx_2), "lambda": lam.item()})

    return lambdas
