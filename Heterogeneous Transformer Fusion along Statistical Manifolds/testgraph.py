import torch
from src.models.vitparents import build_vit_parent
from src.fusion.phase1graph import construct_computational_graph
from src.fusion.phase0width import zero_pad_or_truncate

def test_pipeline():
    print("Testing Architecture and Graph Construction...")
    
    # 1. Instantiate Parent A (6L, 128d) and Parent B-12 (12L, 128d)
    parent_A = build_vit_parent("A")
    parent_B = build_vit_parent("B-12")
    
    # Dummy CIFAR-100 calibration batch (Batch=32, Channels=3, H=32, W=32)
    x = torch.randn(32, 3, 32, 32)
    
    # 2. Forward pass to get block representations
    _, blocks_A = parent_A(x, return_blocks=True)
    _, blocks_B = parent_B(x, return_blocks=True)
    
    print(f"Parent A blocks: {len(blocks_A)}, Shape: {blocks_A[0].shape}")
    print(f"Parent B blocks: {len(blocks_B)}, Shape: {blocks_B[0].shape}")
    
    # 3. Construct computational graphs (Phase 1)
    X_A, A_A = construct_computational_graph(blocks_A, beta=0.5)
    X_B, A_B = construct_computational_graph(blocks_B, beta=0.5)
    
    print(f"Graph A - Features: {X_A.shape}, Adjacency: {A_A.shape}")
    print(f"Graph B - Features: {X_B.shape}, Adjacency: {A_B.shape}")
    
    # Test Phase 0 (Width Alignment) with a mismatched model
    parent_Bw = build_vit_parent("Bw-12") # d_model = 224
    _, blocks_Bw = parent_Bw(x, return_blocks=True)
    aligned_Bw_block = zero_pad_or_truncate(blocks_Bw[0], target_dim=128)
    print(f"Width Aligned Block Shape: {aligned_Bw_block.shape} (Expected: B, Seq, 128)")

if __name__ == "__main__":
    test_pipeline()
