import torch
from src.models.vitparents import build_vit_parent
from src.fusion.pipeline import MSOTFusionPipeline

def run_tier_0():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    # 1. Initialize Parent A (6L) and Parent B (12L)[cite: 2]
    parent_A = build_vit_parent("A").to(device)
    parent_B = build_vit_parent("B-12").to(device)
    
    # Normally, you would load pre-trained weights here
    # parent_A.load_state_dict(torch.load("checkpoints/parent_A_seed_1.pth"))
    # parent_B.load_state_dict(torch.load("checkpoints/parent_B-12_seed_1.pth"))
    
    # 2. Generate a calibration batch (simulating CIFAR-100)
    calibration_batch = torch.randn(32, 3, 32, 32).to(device)
    
    # 3. Instantiate and run the fusion pipeline
    pipeline = MSOTFusionPipeline(
        parent_A=parent_A, 
        parent_B=parent_B, 
        alpha=0.4, 
        beta=0.5, 
        epsilon_fgw=0.02, 
        epsilon_sink=0.02
    )
    
    fused_model = pipeline.fuse(calibration_batch)
    
    # 4. Verification
    print(f"\nVerification:")
    print(f"Fused Model Depth: {len(fused_model.blocks)} (Expected: 12)")
    
    # Forward pass sanity check
    with torch.no_grad():
        test_out = fused_model(calibration_batch)
    print(f"Output shape: {test_out.shape} (Expected: 32, 100)")

if __name__ == "__main__":
    run_tier_0()
