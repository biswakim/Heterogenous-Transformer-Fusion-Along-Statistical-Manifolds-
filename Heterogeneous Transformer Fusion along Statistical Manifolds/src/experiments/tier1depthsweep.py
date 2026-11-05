import os
import gc
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from src.data.cifar import get_cifar100_dataloaders
from src.models.vitparents import build_vit_parent
from src.fusion.pipeline import MSOTFusionPipeline


def build_warmup_cosine_scheduler(optimizer, total_epochs, warmup_fraction=0.08):
    """
    FIXED: training previously used plain CosineAnnealingLR with no warmup
    phase at all. Both the pilot report (Sec 6.2) and the protocol
    (Sec 5.4) specify "cosine learning-rate schedule with linear warm-up
    over the first ~8% of epochs" -- this builds that schedule. Stepped
    once per epoch, same granularity as the rest of this file.
    """
    warmup_epochs = max(1, round(warmup_fraction * total_epochs))
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, max_norm=1.0, accum_steps=2):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mixed-precision forward pass
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Scale loss for gradient accumulation
            loss = loss / accum_steps
            
        scaler.scale(loss).backward()
        
        # Step optimizer every `accum_steps` batches
        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accum_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return total_loss / len(dataloader), 100.0 * correct / total
    
def checkpoint_is_complete(model_name):
    """
    True only if `model_name`'s checkpoint represents a FULLY COMPLETED
    training run, not just an interrupted one.

    FIXED: run_tier_1/run_tier_2 previously checked only
    `os.path.exists(checkpoint_path)` before deciding to skip training and
    load the checkpoint -- but train_parent saves a checkpoint on every
    epoch that improves on the running best, so an interrupted run (crash,
    timeout, manual stop) after as little as epoch 1 already leaves a
    checkpoint file sitting on disk. A later run then silently loads that
    barely-trained snapshot and reports its performance as if it were the
    real, fully-trained parent (this is almost certainly what happened
    with Parent Bw-12 loading at 6.12% test accuracy). This checks for a
    separate marker file that train_parent only writes after ALL epochs
    finish, so a checkpoint file's mere existence is no longer enough.
    """
    ckpt_path = f"checkpoints/{model_name}_best.pth"
    done_path = f"checkpoints/{model_name}_best.pth.done"
    return os.path.exists(ckpt_path) and os.path.exists(done_path)


def load_legacy_checkpoint(model, path, device):
    """Translates old checkpoint keys to the new architecture nomenclature."""
    state_dict = torch.load(path, map_location=device)
    new_state_dict = {}
    for key, value in state_dict.items():
        # Translate old sub-module names to the new Colab-aligned names
        new_key = key.replace(".attention.", ".attn.")
        new_key = new_key.replace(".norm1.", ".ln1.")
        new_key = new_key.replace(".norm2.", ".ln2.")
        new_key = new_key.replace(".ffn.", ".ff.")
        new_state_dict[new_key] = value
        
    model.load_state_dict(new_state_dict)
    return model
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return total_loss / len(dataloader), 100.0 * correct / total

def train_parent(model, model_name, train_loader, test_loader, device, epochs=60):
    print(f"\n--- Training {model_name} for {epochs} epochs ---")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs)
    scaler = GradScaler()
    
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, accum_steps=2)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pth")
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)")
            
    print(f"Finished {model_name}. Best Test Accuracy: {best_acc:.2f}%")
    model.load_state_dict(torch.load(f"checkpoints/{model_name}_best.pth"))
    # Mark this checkpoint as a COMPLETED run (only written here, after
    # every epoch has run) -- see checkpoint_is_complete() above.
    open(f"checkpoints/{model_name}_best.pth.done", "w").close()
    return model, best_acc

def fine_tune_fused_child(fused_model, train_loader, test_loader, device, epochs=15, lr=3e-4):
    fused_model = fused_model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(fused_model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = build_warmup_cosine_scheduler(optimizer, epochs)
    scaler = GradScaler()
    
    print(f"Starting Fine-Tuning for {epochs} epochs...")
    best_acc, best_epoch = 0.0, 0
    final_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(fused_model, train_loader, optimizer, criterion, scaler, device, accum_steps=2)
        test_loss, test_acc = evaluate(fused_model, test_loader, criterion, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch + 1
        final_acc = test_acc
            
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # FIXED: this used to return best_acc (the max test accuracy seen across
    # all `epochs` fine-tuning epochs), which is test-set model selection --
    # it silently inflates the reported number and isn't comparable to a
    # fixed-epoch protocol the way Table 2 of the pilot report is. The
    # headline number is now the FINAL epoch's accuracy; best_acc/best_epoch
    # are still logged below as a separate diagnostic (useful for
    # "epochs-to-recovery" style analysis, Sec 10 of the protocol) but are
    # no longer what gets returned/reported as the fused child's accuracy.
    print(f"Fine-Tuning Complete. Final (epoch {epochs}) Test Accuracy: {final_acc:.2f}% "
          f"[best during fine-tuning was {best_acc:.2f}% at epoch {best_epoch}, shown for reference only]")
    return final_acc

def run_tier_1():
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Tier 1 on device: {device}")
    
    # Batch size 128 x 2 accumulation steps = effective batch size 256
    train_loader, test_loader = get_cifar100_dataloaders(batch_size=128)
    
    # 1. Train Parent A (or load existing checkpoint)
    parent_A = build_vit_parent("A").to(device)
    if checkpoint_is_complete("Parent_A"):
        print("Found existing COMPLETED checkpoint for Parent A. Translating keys and loading...")
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
    
    # Clear VRAM before building and training Parent B
    parent_A.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Train Parent B-12 (or load existing checkpoint)
    parent_B = build_vit_parent("B-12").to(device)
    if checkpoint_is_complete("Parent_B-12"):
        print("Found existing COMPLETED checkpoint for Parent B-12. Translating keys and loading...")
        parent_B = load_legacy_checkpoint(parent_B, "checkpoints/Parent_B-12_best.pth", device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        _, acc_B = evaluate(parent_B, test_loader, criterion, device)
        print(f"Parent B-12 Loaded. Verified Test Acc: {acc_B:.2f}%")
    else:
        if os.path.exists("checkpoints/Parent_B-12_best.pth"):
            print("Found a checkpoint for Parent B-12 but no completion marker -- "
                  "this looks like it's from an INTERRUPTED run. Retraining from scratch "
                  "rather than silently using a partially-trained model.")
        parent_B, acc_B = train_parent(parent_B, "Parent_B-12", train_loader, test_loader, device, epochs=100)
    
    # Clear VRAM before fusion
    torch.cuda.empty_cache()
    parent_A.to(device)
    
    # 3. Extract 32-image calibration batch (per protocol)
    calib_loader, _ = get_cifar100_dataloaders(batch_size=32)
    calibration_batch, _ = next(iter(calib_loader))
    calibration_batch = calibration_batch.to(device)
    
    # 4. Fuse Models
    pipeline = MSOTFusionPipeline(parent_A, parent_B, alpha=0.4, beta=0.5, epsilon_fgw=0.02, epsilon_sink=0.02)
    fused_child = pipeline.fuse(calibration_batch)
    
    # 5. Fine-Tune Fused Child (15 epochs)
    print("\n--- Fine-Tuning Fused Child ---")
    acc_fused = fine_tune_fused_child(fused_child, train_loader, test_loader, device, epochs=15, lr=3e-4)
    
    print("\n=========================================")
    print("         TIER 1 FINAL RESULTS            ")
    print("=========================================")
    print(f"Parent A (6L) Accuracy:     {acc_A:.2f}%")
    print(f"Parent B (12L) Accuracy:    {acc_B:.2f}%")
    print(f"Fused Child (12L) Accuracy: {acc_fused:.2f}%")
    print("=========================================")

if __name__ == "__main__":
    run_tier_1()
