import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.last_attn_weights = None 

    def forward(self, x, need_weights=False):
        if need_weights:
            attn_out, attn_w = self.attn(x, x, x, need_weights=True, average_attn_weights=True)
            self.last_attn_weights = attn_w.detach()
        else:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, depth, d_model, num_heads, ffn_dim, num_classes=100, img_size=32, patch_size=2, in_channels=3):
        super().__init__()
        self.depth = depth
        self.d_model = d_model
        
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(0.1)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_dim) for _ in range(depth)
        ])
        
        self.head = nn.Linear(d_model, num_classes)
        self.patch_size = patch_size

    def forward(self, x, return_blocks=False, need_weights=False):
        B, C, H, W = x.shape
        p = self.patch_size
        
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * p * p)
        
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        block_representations = []
        
        for block in self.blocks:
            x = block(x, need_weights=need_weights)
            if return_blocks:
                block_representations.append(x)
                
        logits = self.head(x[:, 0])
        
        if return_blocks:
            return logits, block_representations
        return logits

VIT_CONFIGS = {
    "A":     {"depth": 6,  "d_model": 128, "num_heads": 8,  "ffn_dim": 256},
    "B-8":   {"depth": 8,  "d_model": 128, "num_heads": 8,  "ffn_dim": 256},
    "B-10":  {"depth": 10, "d_model": 128, "num_heads": 8,  "ffn_dim": 256},
    "B-12":  {"depth": 12, "d_model": 128, "num_heads": 8,  "ffn_dim": 256},
    "B-14":  {"depth": 14, "d_model": 128, "num_heads": 8,  "ffn_dim": 256},
    "Bw-8":  {"depth": 8,  "d_model": 160, "num_heads": 8,  "ffn_dim": 320},
    "Bw-10": {"depth": 10, "d_model": 192, "num_heads": 8,  "ffn_dim": 384},
    "Bw-12": {"depth": 12, "d_model": 224, "num_heads": 8,  "ffn_dim": 448},
    "Bw-14": {"depth": 14, "d_model": 256, "num_heads": 16, "ffn_dim": 512},
}

def build_vit_parent(config_id, **overrides):
    """
    Build a parent from the named depth/width config.

    `overrides` lets the same config table serve a second dataset without
    duplicating every entry -- e.g. for Tiny-ImageNet:
        build_vit_parent("A", num_classes=200, img_size=64, patch_size=8)
    Patch size matters a lot at 64px: keeping patch_size=2 would give
    (64/2)^2 = 1024 patches and attention is O(n^2) in patch count, so
    training would be ~16x slower than the 256-patch CIFAR setting.
    patch_size=8 gives 64 patches; patch_size=4 gives 256, matching CIFAR's
    sequence length exactly.
    """
    if config_id not in VIT_CONFIGS:
        raise ValueError(f"Unknown config_id: {config_id}. Must be one of {list(VIT_CONFIGS.keys())}")

    config = {**VIT_CONFIGS[config_id], **overrides}
    return VisionTransformer(**config)
