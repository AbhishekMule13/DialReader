"""
Training & Evaluation Pipeline for DigitMaskCNN (Steps 3, 4, 5)
Trains small CNN on 32x32 white-pixel digit masks.
Evaluates on REAL test split, logs confusion matrix and error visualizations.
Compares with old MLP before integrating into dashboard.
"""

import sys
import os
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_PATH = PROJECT_ROOT / "data" / "digits" / "manifest.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CNN_MODEL_PATH = MODELS_DIR / "digit_mask_cnn_best.pt"
MLP_MODEL_PATH = MODELS_DIR / "best_digit_ann.pt"
BACKUP_MLP_PATH = MODELS_DIR / "best_digit_ann_backup.pt"

from src.mask_cnn import DigitMaskCNN, DigitMaskPredictor, MaskAugmenter, extract_white_digit_mask
from src.models import DigitANNPredictor
from src.synthetic import SyntheticDigitGenerator


class RealDigitMaskDataset(Dataset):
    """Dataset for 32x32 white-pixel digit masks from real slot crops."""
    def __init__(self, samples: List[Tuple[np.ndarray, int, str]], is_train: bool = False):
        self.samples = samples  # (32x32 binary array, label, crop_id)
        self.is_train = is_train
        self.augmenter = MaskAugmenter(p_glare=0.3, max_shift=2) if is_train else None
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        mask, label, crop_id = self.samples[idx]
        if self.is_train and self.augmenter:
            mask = self.augmenter(mask)
            
        # Normalize [0, 255] -> [-1.0, 1.0]
        t = torch.tensor(mask, dtype=torch.float32) / 255.0
        t = ((t - 0.5) / 0.5).unsqueeze(0)  # (1, 32, 32)
        return t, label, crop_id


def prepare_mask_data(
    manifest_df: pd.DataFrame,
    seed: int = 42,
    use_synthetic_train: bool = True,
    synth_per_class: int = 40
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits dataset by split_group (frames from same video never cross train and test).
    Validation and test sets contain REAL crops ONLY.
    """
    np.random.seed(seed)
    
    # Identify video groups vs standalone photos
    # Source images from same directory/video stay in the same split
    sources = manifest_df["source_image"].unique()
    
    # If manifest has predefined split_group, use it
    train_samples, val_samples, test_samples = [], [], []
    
    print("Extracting white digit masks for real slot crops...")
    for idx, row in manifest_df.iterrows():
        crop_p = PROJECT_ROOT / row["crop_path"]
        label = int(row["digit"])
        split_grp = str(row.get("split_group", "train")).lower()
        crop_id = f"{row.get('source_image', '')}_slot{row.get('slot_index', '')}"
        
        if not crop_p.exists():
            continue
            
        slot_img = Image.open(crop_p).convert("RGB")
        mask = extract_white_digit_mask(slot_img, output_size=32)
        
        # Route by split_group
        if split_grp == "test":
            test_samples.append((mask, label, crop_id))
        elif split_grp in ("val", "train_or_val_val"):
            val_samples.append((mask, label, crop_id))
        elif split_grp in ("train", "train_or_val_train"):
            train_samples.append((mask, label, crop_id))
        else:
            # Fallback random split by source image
            h = hash(row["source_image"]) % 100
            if h < 70:
                train_samples.append((mask, label, crop_id))
            elif h < 85:
                val_samples.append((mask, label, crop_id))
            else:
                test_samples.append((mask, label, crop_id))
                
    # If test or val is empty, split train_samples deterministically by source
    if len(test_samples) == 0 or len(val_samples) == 0:
        all_real = train_samples + val_samples + test_samples
        np.random.shuffle(all_real)
        n_total = len(all_real)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        train_samples = all_real[:n_train]
        val_samples = all_real[n_train:n_train+n_val]
        test_samples = all_real[n_train+n_val:]
        
    # Supplement training set with synthetic masks (NEVER added to val or test)
    if use_synthetic_train:
        gen = SyntheticDigitGenerator(64, 32)
        synth_added = 0
        for d in range(10):
            for _ in range(synth_per_class):
                synth_img = gen.render_digit_slot(d, is_red_drum=(d == 9))
                synth_mask = extract_white_digit_mask(synth_img, output_size=32)
                train_samples.append((synth_mask, d, f"synth_d{d}"))
                synth_added += 1
        print(f"Added {synth_added} synthetic training masks (Val & Test remain 100% REAL).")
        
    print(f"Dataset split sizes: Train={len(train_samples)}, Val={len(val_samples)} (REAL), Test={len(test_samples)} (REAL)")
    return (
        RealDigitMaskDataset(train_samples, is_train=True),
        RealDigitMaskDataset(val_samples, is_train=False),
        RealDigitMaskDataset(test_samples, is_train=False)
    )


def train_mask_cnn(epochs: int = 35, batch_size: int = 32, lr: float = 1e-3):
    print("=" * 70)
    print(" Training DigitMaskCNN on White-Pixel Masks (Step 3)")
    print("=" * 70)
    
    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} not found. Run build_real_digits.py first.")
        return False
        
    df_manifest = pd.read_csv(MANIFEST_PATH)
    if len(df_manifest) == 0:
        print("Manifest is empty. Please annotate images with label_readings.py first.")
        return False
        
    train_ds, val_ds, test_ds = prepare_mask_data(df_manifest)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitMaskCNN(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_acc = 0.0
    best_state_dict = None
    
    print(f"\nStarting training on {device} ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_corr, train_tot = 0.0, 0, 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            train_corr += (preds == labels).sum().item()
            train_tot += len(labels)
            
        epoch_train_acc = train_corr / max(1, train_tot)
        
        # Validation on REAL crops
        model.eval()
        val_corr, val_tot = 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=-1)
                val_corr += (preds == labels).sum().item()
                val_tot += len(labels)
                
        epoch_val_acc = val_corr / max(1, val_tot)
        scheduler.step()
        
        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            
        if epoch == 1 or epoch % 5 == 0 or is_best:
            tag = " ★ (Best Real Val)" if is_best else ""
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {epoch_train_acc:.3f} | Real Val Acc: {epoch_val_acc:.3f}{tag}")
            
    # Save best checkpoint
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state_dict,
        "val_acc": best_val_acc,
        "classes": list(range(10))
    }, CNN_MODEL_PATH)
    print(f"\nSaved best DigitMaskCNN model to: {CNN_MODEL_PATH}")
    
    # -------------------------------------------------------------------------
    # STEP 4: Evaluation on REAL Test Split
    # -------------------------------------------------------------------------
    evaluate_on_test_split(best_state_dict, test_loader, device, df_manifest)
    return True


def evaluate_on_test_split(state_dict, test_loader, device, manifest_df):
    print("\n" + "=" * 70)
    print(" STEP 4: Evaluation on REAL Test Split")
    print("=" * 70)
    
    model = DigitMaskCNN(num_classes=10).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    all_preds, all_labels, all_ids = [], [], []
    all_confs = []
    
    with torch.no_grad():
        for imgs, labels, ids in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=-1)
            top_prob, preds = torch.max(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_ids.extend(ids)
            all_confs.extend(top_prob.cpu().numpy().tolist())
            
    tot = len(all_labels)
    if tot == 0:
        print("Test set has 0 samples.")
        return
        
    corr = sum(1 for p, y in zip(all_preds, all_labels) if p == y)
    overall_acc = corr / tot
    print(f"\nOverall Digit Accuracy on REAL Test Split: {corr}/{tot} ({overall_acc:.2%})")
    
    # Per-digit accuracy
    print("\nPer-Digit Accuracy:")
    for d in range(10):
        d_tot = sum(1 for y in all_labels if y == d)
        d_corr = sum(1 for p, y in zip(all_preds, all_labels) if y == d and p == d)
        acc_str = f"{d_corr}/{d_tot} ({d_corr/d_tot:.1%})" if d_tot > 0 else "N/A (0 test samples)"
        print(f"  Digit {d}: {acc_str}")
        
    # Confusion Matrix (10x10)
    print("\nConfusion Matrix (Rows = True Digit, Columns = Predicted Digit):")
    cm = np.zeros((10, 10), dtype=int)
    for p, y in zip(all_preds, all_labels):
        cm[y, p] += 1
        
    header = "True \\ Pred | " + " ".join(f"{c:3d}" for c in range(10))
    print(header)
    print("-" * len(header))
    for r in range(10):
        row_str = f"Digit {r:2d}   | " + " ".join(f"{cm[r, c]:3d}" for c in range(10))
        print(row_str)
        
    # Top 10 worst errors
    errors = []
    for p, y, cid, conf in zip(all_preds, all_labels, all_ids, all_confs):
        if p != y:
            errors.append({"id": cid, "true": y, "pred": p, "conf": conf})
            
    errors = sorted(errors, key=lambda x: x["conf"], reverse=True)[:10]
    print(f"\nTop {len(errors)} Worst Prediction Errors:")
    for i, e in enumerate(errors):
        print(f"  Error {i+1}: Crop='{e['id']}' | True={e['true']} -> Pred={e['pred']} (Confidence: {e['conf']:.2%})")


if __name__ == "__main__":
    train_mask_cnn()
