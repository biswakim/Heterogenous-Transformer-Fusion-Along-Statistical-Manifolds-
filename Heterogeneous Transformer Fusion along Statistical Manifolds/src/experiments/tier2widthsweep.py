import os
import gc
import torch
import torch.nn as nn
from src.data.cifar import get_cifar100_dataloaders
from src.models.vitparents import build_vit_parent
from src.fusion.pipelinewidth import MSOTWidthPipeline
from src.experiments.tier1depthsweep import train_parent, fine_tune_fused_child, evaluate, load_legacy_checkpoint, checkpoint_is_complete

def run_tier_2():
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Tier 2 (Depth + Width) on device: {device}")
    
    # Effective batch size of 256 via accumulation
    train_loader, test_loader = get_cifar100_dataloaders(batch_size=128)
    
    # 1. Load the Anchor Parent A (6 Layers, 128 d_model)
    parent_A = build_vit_parent("A").to(device)
    if checkpoint_is_complete("Parent_A"):
        print("Found existing COMPLETED checkpoint for Parent A. Loading...")
        parent_A = load_legacy_checkpoint(parent_A, "checkpoints/Parent_A_best.pth", device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        _, acc_A = evaluate(parent_A, test_loader, criterion, device)
        print(f"Parent A Loaded. Verified Test Acc: {acc_A:.2f}%")
    else:
        if os.path.exists("checkpoints/Parent_A_best.pth"):
            print("Found a checkpoint for Parent A but no completion marker -- "
                  "this looks like it's from an INTERRUPTED run. Retraining from scratch "
                  "rather than silently using a partially-trained model.")
        parent_A, acc_A = train_parent(parent_A, "Parent_A", train_loader, test_loader, device, epochs=60)
    
    parent_A.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Train the Wider Target Parent Bw-12 (12 Layers, 224 d_model)
    parent_Bw = build_vit_parent("Bw-12").to(device)
    if checkpoint_is_complete("Parent_Bw-12"):
        print("Found existing COMPLETED checkpoint for Parent Bw-12. Translating keys and loading...")
        # load_legacy_checkpoint is a safe no-op on keys that already use
        # the new naming (replace() only touches substrings that are
        # actually present), so applying it here unconditionally is safe.
        parent_Bw = load_legacy_checkpoint(parent_Bw, "checkpoints/Parent_Bw-12_best.pth", device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        _, acc_Bw = evaluate(parent_Bw, test_loader, criterion, device)
        print(f"Parent Bw-12 Loaded. Verified Test Acc: {acc_Bw:.2f}%")
    else:
        if os.path.exists("checkpoints/Parent_Bw-12_best.pth"):
            print("Found a checkpoint for Parent Bw-12 but no completion marker -- "
                  "this looks like it's from an INTERRUPTED run (this is almost certainly "
                  "why it previously loaded at ~6% accuracy). Retraining from scratch "
                  "rather than silently using a partially-trained model.")
        parent_Bw, acc_Bw = train_parent(parent_Bw, "Parent_Bw-12", train_loader, test_loader, device, epochs=100)
    
    torch.cuda.empty_cache()
    parent_A.to(device)
    
    # 3. Extract Calibration Batch
    calib_loader, _ = get_cifar100_dataloaders(batch_size=32)
    calibration_batch, _ = next(iter(calib_loader))
    calibration_batch = calibration_batch.to(device)
    
    # 4. Fuse Models using the Width Pipeline
    # FIXED: epsilon_sink was 0.05 here (vs. the protocol default of 0.02
    # used everywhere else) -- reverted unless you specifically meant to
    # deviate here, in which case leave a comment saying so.
    # width_alignment_method is now an explicit, real switch (Sec 8.4):
    # 'zero_pad' (matches prior behavior) | 'random_orthogonal' | 'learned_linear' (proposed).
    pipeline = MSOTWidthPipeline(
        parent_A, parent_Bw, alpha=0.4, beta=0.5, epsilon_fgw=0.02, epsilon_sink=0.02,
        width_alignment_method="zero_pad",
    )
    fused_child = pipeline.fuse(calibration_batch)
    
    # 5. Fine-Tune Fused Child (15 epochs)
    print("\n--- Fine-Tuning Wide Fused Child ---")
    acc_fused = fine_tune_fused_child(fused_child, train_loader, test_loader, device, epochs=15, lr=3e-4)
    
    print("\n=========================================")
    print("         TIER 2 FINAL RESULTS            ")
    print("=========================================")
    print(f"Parent A (6L, 128d) Accuracy:       {acc_A:.2f}%")
    print(f"Parent Bw (12L, 224d) Accuracy:     {acc_Bw:.2f}%")
    print(f"Fused Wide Child (12L, 224d) Acc:   {acc_fused:.2f}%")
    print("=========================================")

if __name__ == "__main__":
    run_tier_2()
