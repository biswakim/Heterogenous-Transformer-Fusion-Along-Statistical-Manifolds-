import torch
import torch.nn as nn

def zero_pad_or_truncate(features, target_dim):
    """
    Naive baseline: Pads the narrower model's channel dimension with zeros, 
    or truncates the wider one to a common working dimension.
    """
    current_dim = features.shape[-1]
    if current_dim == target_dim:
        return features
    elif current_dim > target_dim:
        return features[..., :target_dim] # Truncate
    else:
        pad_size = target_dim - current_dim
        return torch.nn.functional.pad(features, (0, pad_size)) # Zero-pad

def random_orthogonal_projection(features, target_dim, seed=None):
    """
    Baseline: Projects representations into a shared dimension via a
    fixed random orthogonal matrix to preserve pairwise distances.

    FIXED: the old version built Q from QR of a (current_dim, target_dim)
    matrix regardless of direction. When projecting UP (current_dim <
    target_dim), torch.linalg.qr's default reduced mode returns Q with
    shape (current_dim, current_dim) -- not (current_dim, target_dim) --
    so `q.T` stayed square, and the following slice
    `[:current_dim, :target_dim]` silently clipped back down to
    (current_dim, current_dim) since target_dim exceeded the matrix's
    actual width. The output shape never reached target_dim; matmul
    quietly returned (..., current_dim), unchanged. Fixed by always
    building Q from a (max_dim, min_dim) matrix, then using Q directly
    when projecting down (orthonormal columns) or Q.T when projecting up
    (orthonormal rows), so the output reaches target_dim either way.
    """
    current_dim = features.shape[-1]
    if current_dim == target_dim:
        return features

    max_dim, min_dim = max(current_dim, target_dim), min(current_dim, target_dim)
    if seed is not None:
        gen = torch.Generator(device=features.device).manual_seed(seed)
        rand_mat = torch.randn(max_dim, min_dim, generator=gen, device=features.device)
    else:
        rand_mat = torch.randn(max_dim, min_dim, device=features.device)

    q, _ = torch.linalg.qr(rand_mat)  # (max_dim, min_dim), orthonormal columns
    projection_matrix = q if current_dim >= target_dim else q.T  # always ends up (current_dim, target_dim)

    return torch.matmul(features, projection_matrix)

class LearnedLinearProjection(nn.Module):
    """
    Proposed: A lightweight linear map fit on the calibration batch.
    Used to project B's features into A's dimensionality space.
    """
    def __init__(self, source_dim, target_dim):
        super().__init__()
        self.proj = nn.Linear(source_dim, target_dim, bias=False)
        
    def forward(self, features):
        return self.proj(features)
        
    def fit(self, source_features, target_features, epochs=50, lr=1e-3):
        """Fits the projection matrix using MSE on the calibration batch."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Flatten sequence/batch dims for alignment training
        src_flat = source_features.reshape(-1, source_features.shape[-1])
        tgt_flat = target_features.reshape(-1, target_features.shape[-1])
        
        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            projected = self(src_flat)
            loss = criterion(projected, tgt_flat)
            loss.backward()
            optimizer.step()
        self.eval()
