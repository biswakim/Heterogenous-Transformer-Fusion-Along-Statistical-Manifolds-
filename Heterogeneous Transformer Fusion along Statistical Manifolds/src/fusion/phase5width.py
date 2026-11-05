import torch
import torch.nn as nn

def pad_2d_weight(weight, target_out_dim, target_in_dim):
    """
    Pads a 2D weight matrix to match target dimensions.
    weight shape: (out_features, in_features)
    """
    out_diff = target_out_dim - weight.shape[0]
    in_diff = target_in_dim - weight.shape[1]
    
    # F.pad expects padding from last dimension to first: (left, right, top, bottom)
    return torch.nn.functional.pad(weight, (0, in_diff, 0, out_diff), mode='constant', value=0.0)

def pad_1d_bias(bias, target_dim):
    """
    Pads a 1D bias or LayerNorm parameter to match target dimensions.
    """
    diff = target_dim - bias.shape[0]
    return torch.nn.functional.pad(bias, (0, diff), mode='constant', value=0.0)

def project_block_dimensions(block_small, target_d_model, target_nhead, target_ffn_dim):
    """
    Creates a new TransformerBlock matching the target dimensions,
    populated with the zero-padded weights of the smaller block.
    """
    device = block_small.attn.in_proj_weight.device
    projected = type(block_small)(target_d_model, target_nhead, target_ffn_dim).to(device)
    
    with torch.no_grad():
        # Project LayerNorms
        projected.ln1.weight.copy_(pad_1d_bias(block_small.ln1.weight, target_d_model))
        projected.ln1.bias.copy_(pad_1d_bias(block_small.ln1.bias, target_d_model))
        projected.ln2.weight.copy_(pad_1d_bias(block_small.ln2.weight, target_d_model))
        projected.ln2.bias.copy_(pad_1d_bias(block_small.ln2.bias, target_d_model))
        
        # Project FFN
        projected.ff[0].weight.copy_(pad_2d_weight(block_small.ff[0].weight, target_ffn_dim, target_d_model))
        projected.ff[0].bias.copy_(pad_1d_bias(block_small.ff[0].bias, target_ffn_dim))
        projected.ff[3].weight.copy_(pad_2d_weight(block_small.ff[3].weight, target_d_model, target_ffn_dim))
        projected.ff[3].bias.copy_(pad_1d_bias(block_small.ff[3].bias, target_d_model))
        
        # Project Attention (in_proj is 3 * d_model in the out_features dimension)
        projected.attn.in_proj_weight.copy_(pad_2d_weight(block_small.attn.in_proj_weight, 3 * target_d_model, target_d_model))
        projected.attn.in_proj_bias.copy_(pad_1d_bias(block_small.attn.in_proj_bias, 3 * target_d_model))
        projected.attn.out_proj.weight.copy_(pad_2d_weight(block_small.attn.out_proj.weight, target_d_model, target_d_model))
        projected.attn.out_proj.bias.copy_(pad_1d_bias(block_small.attn.out_proj.bias, target_d_model))

    return projected