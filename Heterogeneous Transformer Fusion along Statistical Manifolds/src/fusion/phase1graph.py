import torch
import torch.nn.functional as F

def attention_entropy(attn_w):
    p = attn_w.clamp(min=1e-9)
    ent = -(p * p.log()).sum(dim=-1).mean()
    return ent.item()

@torch.no_grad()
def construct_computational_graph(model, calibration_batch, beta=0.5):
    """Extracts features and builds hybrid adjacency."""
    model.eval()
    _, hiddens = model(calibration_batch, return_blocks=True, need_weights=True)
    
    feats, pooled_states = [], []
    for i, h in enumerate(hiddens):
        pooled = h.mean(dim=(0, 1)) # Pool over batch and seq
        ent = attention_entropy(model.blocks[i].last_attn_weights)
        feat = torch.cat([pooled, torch.tensor([ent], device=pooled.device)])
        feats.append(feat)
        pooled_states.append(pooled)
        
    X = torch.stack(feats)
    pooled_tensor = torch.stack(pooled_states)
    
    # Structural Adjacency
    depth = pooled_tensor.shape[0]
    seq_adj = torch.zeros(depth, depth, device=pooled_tensor.device)
    for i in range(depth - 1):
        seq_adj[i, i + 1] = 1.0
        seq_adj[i + 1, i] = 1.0
        
    # Functional Adjacency via cosine similarity
    normed = F.normalize(pooled_tensor, dim=1)
    func_adj = (normed @ normed.T + 1.0) / 2.0
    
    A_hybrid = beta * seq_adj + (1.0 - beta) * func_adj
    return X, A_hybrid
