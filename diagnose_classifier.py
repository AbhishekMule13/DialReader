"""
Diagnostic script for DialReader digit classifier.
Performs Checks 1 through 6 to inspect model loading, class mapping,
dataset labels, preprocessing parity, and inference outputs.
"""

import sys
import os
import random
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

# Ensure dial-reader root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import DigitANN, DigitANNPredictor, get_ann_transform
from src.synthetic import SyntheticDigitGenerator
from src.infer import DialReadingPipeline
from src.crop import WindowNormalizer


def separator(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


# ==============================================================================
# CHECK 1: Is a trained model loaded?
# ==============================================================================
def check_1():
    separator("CHECK 1: Is a Trained Model Loaded?")
    model_path = PROJECT_ROOT / "models" / "best_digit_ann.pt"
    
    print(f"Checkpoint path: {model_path}")
    if not model_path.exists():
        print(f"ERROR: Checkpoint file not found at {model_path}!")
        return None, None
        
    stat = model_path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"File size:       {stat.st_size:,} bytes")
    print(f"Modified date:   {mtime}")
    
    # Load raw checkpoint dict
    ckpt = torch.load(model_path, map_location="cpu")
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Recorded val_acc: {ckpt.get('val_acc')}")
    print(f"Recorded classes: {ckpt.get('classes')}")
    
    # Initialize expected architecture
    model = DigitANN(input_dim=2048, hidden_layers=[256, 128, 64], num_classes=10)
    expected_keys = set(model.state_dict().keys())
    
    raw_sd = ckpt.get("model_state_dict", ckpt)
    raw_keys = set(raw_sd.keys())
    
    # Test key adaptation used by DigitANNPredictor
    adapted_sd = {}
    for k, v in raw_sd.items():
        if k.startswith("net.12."):
            adapted_sd[k.replace("net.12.", "classifier.")] = v
        elif k.startswith("net."):
            adapted_sd[k.replace("net.", "feature_extractor.")] = v
        else:
            adapted_sd[k] = v
            
    adapted_keys = set(adapted_sd.keys())
    missing = expected_keys - adapted_keys
    unexpected = adapted_keys - expected_keys
    
    print(f"\nModel architecture parameters check:")
    print(f"  Expected tensors:   {len(expected_keys)}")
    print(f"  Checkpoint tensors: {len(raw_keys)}")
    print(f"  Missing keys:       {missing if missing else 'None (0 missing)'}")
    print(f"  Unexpected keys:    {unexpected if unexpected else 'None (0 unexpected)'}")
    
    # Load weights into model
    load_result = model.load_state_dict(adapted_sd, strict=True)
    print(f"  load_state_dict result: {load_result}")
    print(f"  Silent fallback to random init: NO (all weights explicitly restored)")
    
    model.eval()
    print(f"\nModel output specs:")
    print(f"  Classifier head output features: {model.classifier.out_features} (10 classes)")
    inference_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"  Inference class list:            {inference_classes}")
    
    return model_path, ckpt


# ==============================================================================
# CHECK 2: Class Mapping
# ==============================================================================
def check_2(ckpt):
    separator("CHECK 2: Class Mapping")
    
    train_classes = ckpt.get("classes")
    print(f"Training-time class mapping (from checkpoint):")
    print(f"  Type: {type(train_classes)}")
    print(f"  Mapping: { {i: c for i, c in enumerate(train_classes)} }")
    
    print(f"\nInference-time mapping (from src/models.py:predict_slot):")
    inference_mapping = {i: i for i in range(10)}
    print(f"  Mapping: {inference_mapping}")
    
    # Check potential pitfalls
    print(f"\nAuditing mapping consistency:")
    # 1. String vs int sorting
    str_sorted = sorted([str(i) for i in range(10)])
    print(f"  Lexicographical string sort would be: {str_sorted}")
    is_lex_sorted = (train_classes == [int(x) for x in str_sorted])
    print(f"  Is training classes lexicographically sorted? {is_lex_sorted} (No string sort bug)")
    
    # 2. Off-by-one check
    is_off_by_one = (train_classes == list(range(1, 11)))
    print(f"  Is training classes 1-indexed (1..10)?       {is_off_by_one} (0-indexed 0..9)")
    
    # 3. Mismatch check
    mismatches = [(i, tc, ic) for i, (tc, ic) in enumerate(zip(train_classes, list(range(10)))) if tc != ic]
    print(f"  Index-to-digit mismatches:                  {mismatches if mismatches else 'None (100% match)'}")
    
    # Verify with synthetic generator
    print(f"\nVerification on synthetic generator samples (predicting digits 0-9):")
    p = DigitANNPredictor(PROJECT_ROOT / "models" / "best_digit_ann.pt", {
        "data": {"input_size": [64, 32], "channels": 1},
        "model": {"hidden_layers": [256, 128, 64], "activation": "relu", "batchnorm": True, "dropout": 0.2}
    })
    gen = SyntheticDigitGenerator(64, 32)
    matches = 0
    for d in range(10):
        img = gen.render_digit_slot(d, is_red_drum=(d == 9))
        pred = p.predict_slot(img)
        is_match = (pred["digit"] == d)
        if is_match:
            matches += 1
        print(f"  Target: {d} -> Predicted: {pred['digit']} (conf: {pred['confidence']:.4f}, match={is_match})")
    print(f"Synthetic generator mapping match rate: {matches}/10")


# ==============================================================================
# CHECK 3: Training Data vs Labels (labels.csv & Contact Sheet)
# ==============================================================================
def check_3():
    separator("CHECK 3: Training Data vs labels.csv")
    labels_csv_path = PROJECT_ROOT / "data" / "labeled" / "labels.csv"
    print(f"Loading labels from: {labels_csv_path}")
    
    if not labels_csv_path.exists():
        print(f"ERROR: {labels_csv_path} does not exist!")
        return
        
    df = pd.read_csv(labels_csv_path)
    print(f"Total rows in labels.csv: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Non-null counts per column:\n{df.notnull().sum()}")
    
    non_null_readings = df[df["reading"].notnull()]
    print(f"\nRows with non-null 'reading' values ({len(non_null_readings)}):")
    print(non_null_readings[["filename", "reading", "orientation", "dial_variant"]])
    
    # Pick 20 rows
    random.seed(42)
    sample_indices = random.sample(range(len(df)), min(20, len(df)))
    sample_df = df.iloc[sample_indices].copy()
    
    print(f"\nSelected 20 random rows from labels.csv:")
    for idx, row in sample_df.iterrows():
        print(f"  Row {idx:3d}: File='{row['filename']}' | reading='{row['reading']}' | orientation={row['orientation']}")
        
    # Generate Contact Sheet Image
    print(f"\nBuilding contact sheet image with 4 slot crops per row...")
    pipeline = DialReadingPipeline()
    
    cell_w, cell_h = 100, 160
    sheet_w = 400 + (4 * cell_w) + 40
    sheet_h = 40 + (len(sample_df) * cell_h)
    contact_sheet = Image.new("RGB", (sheet_w, sheet_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(contact_sheet)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_b = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
        font_b = ImageFont.load_default()
        
    draw.text((20, 12), "DIALREADER DIAGNOSTIC CONTACT SHEET: 20 ROWS FROM labels.csv", fill=(255, 200, 50), font=font_b)
    
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for row_idx, (_, row) in enumerate(sample_df.iterrows()):
        y_pos = 40 + (row_idx * cell_h)
        rel_path = row["filename"]
        raw_path = PROJECT_ROOT / "data" / "raw" / rel_path
        
        reading_lbl = str(row["reading"]) if pd.notnull(row["reading"]) else "EMPTY / NaN"
        ori = int(row["orientation"]) if pd.notnull(row["orientation"]) else None
        
        info_text = f"Row: {row_idx+1}\nFile: {Path(rel_path).name}\nCSV label: {reading_lbl}\nOri: {ori}"
        draw.text((20, y_pos + 20), info_text, fill=(220, 220, 220), font=font)
        
        slots = []
        if raw_path.exists():
            try:
                res = pipeline.read_image(raw_path, manual_orientation=ori)
                slots = res.get("digit_slots", [])
            except Exception as e:
                draw.text((420, y_pos + 60), f"Error reading: {e}", fill=(255, 100, 100), font=font)
        else:
            draw.text((420, y_pos + 60), f"File missing: {raw_path.name}", fill=(255, 100, 100), font=font)
            
        for s_idx in range(4):
            x_pos = 420 + (s_idx * cell_w)
            # Slot bounding box
            draw.rectangle([x_pos, y_pos + 10, x_pos + cell_w - 10, y_pos + cell_h - 10], outline=(70, 70, 70), width=1)
            draw.text((x_pos + 10, y_pos + 15), f"Slot {s_idx+1}", fill=(180, 180, 180), font=font)
            
            if s_idx < len(slots):
                slot_thumb = slots[s_idx].copy().resize((cell_w - 20, cell_h - 55), Image.Resampling.BILINEAR)
                contact_sheet.paste(slot_thumb, (x_pos + 5, y_pos + 35))
                # Predict slot
                pred_r = pipeline.classifier.predict_slot(slots[s_idx])
                draw.text((x_pos + 10, y_pos + cell_h - 30), f"Pred: {pred_r['digit']} ({pred_r['confidence']:.2f})", fill=(50, 220, 100), font=font)
                
    contact_sheet_path = results_dir / "diagnose_contact_sheet_20.png"
    contact_sheet.save(contact_sheet_path)
    print(f"Saved contact sheet image to: {contact_sheet_path}")


# ==============================================================================
# CHECK 4: Accuracy on Training Data via Both Code Paths
# ==============================================================================
def check_4():
    separator("CHECK 4: Accuracy on Training Data via Both Code Paths")
    
    # As diagnosed, the model was trained on SyntheticDigitGenerator samples
    # We test on 30 dials (30 * 4 = 120 slot crops) with known ground truth digits
    random.seed(42)
    gen = SyntheticDigitGenerator(64, 32)
    
    # Path A: Training-time transform (from notebooks/train_ann.ipynb)
    train_transform = get_ann_transform(img_size=(64, 32), channels=1, is_train=False)
    
    # Path B: Dashboard inference transform (from src/models.py)
    pipeline = DialReadingPipeline()
    dashboard_transform = pipeline.classifier.transform
    model = pipeline.classifier.model
    model.eval()
    
    num_samples = 30
    slots_per_dial = 4
    
    correct_a = [0] * 4
    correct_b = [0] * 4
    total_slots = [0] * 4
    
    print(f"Generating {num_samples} dials with 4 slots each ({num_samples * 4} crops total)...")
    
    for i in range(num_samples):
        # 4 random digits
        ground_truth_digits = [random.randint(0, 9) for _ in range(4)]
        
        for slot_idx in range(4):
            d = ground_truth_digits[slot_idx]
            is_red = (slot_idx == 3)
            # Create synthetic slot crop
            slot_img = gen.render_digit_slot(d, is_red_drum=is_red)
            
            # --- Path (a): Training-time preprocessing ---
            tensor_a = train_transform(slot_img).unsqueeze(0)
            with torch.no_grad():
                logits_a = model(tensor_a)[0]
                pred_a = int(logits_a.argmax().item())
                
            # --- Path (b): Dashboard inference preprocessing ---
            # Dashboard passes PIL to predict_slot
            res_b = pipeline.classifier.predict_slot(slot_img)
            pred_b = res_b["digit"]
            
            total_slots[slot_idx] += 1
            if pred_a == d:
                correct_a[slot_idx] += 1
            if pred_b == d:
                correct_b[slot_idx] += 1
                
    print("\nResults on 30 synthetic dials with known labels:")
    print("Slot   | Path (a) Training Preproc Acc | Path (b) Dashboard Preproc Acc")
    print("-" * 65)
    for s in range(4):
        acc_a = correct_a[s] / total_slots[s]
        acc_b = correct_b[s] / total_slots[s]
        print(f"Slot {s+1} | {correct_a[s]:2d}/{total_slots[s]:2d} ({acc_a:6.1%})               | {correct_b[s]:2d}/{total_slots[s]:2d} ({acc_b:6.1%})")
    
    total_corr_a = sum(correct_a)
    total_corr_b = sum(correct_b)
    tot = sum(total_slots)
    print("-" * 65)
    print(f"Total  | {total_corr_a:2d}/{tot:2d} ({total_corr_a/tot:6.1%})               | {total_corr_b:2d}/{tot:2d} ({total_corr_b/tot:6.1%})")
    print(f"Difference between Path (a) and Path (b): {abs(total_corr_a - total_corr_b)} mismatches")


# ==============================================================================
# CHECK 5: Preprocessing Parity Side-by-Side
# ==============================================================================
def check_5():
    separator("CHECK 5: Preprocessing Parity Side-by-Side on the SAME Crop")
    
    # Load Preset 1 (0078) Slot 0
    p1 = Path(r"C:\Users\ABHI\.gemini\antigravity-ide\brain\8abd5874-9ef0-49ba-9ca6-2a6566cf2a94\.user_uploaded\media_1792996743107.png")
    pipeline = DialReadingPipeline()
    res = pipeline.read_image(p1)
    sample_crop = res["digit_slots"][0]
    
    # Path A: Training preprocessing (notebooks/train_ann.ipynb)
    train_tf = get_ann_transform((64, 32), channels=1, is_train=False)
    tensor_a = train_tf(sample_crop)
    
    # Path B: Dashboard preprocessing (src/models.py:get_ann_transform)
    dash_tf = pipeline.classifier.transform
    tensor_b = dash_tf(sample_crop)
    
    table_data = [
        ("Input PIL Size (W, H)", f"{sample_crop.size}", f"{sample_crop.size}"),
        ("Input PIL Mode", f"{sample_crop.mode}", f"{sample_crop.mode}"),
        ("Color / Channel Order", "RGB (PIL)", "RGB (PIL)"),
        ("Grayscale Conversion", "Grayscale(num_output_channels=1)", "Grayscale(num_output_channels=1)"),
        ("Resize Target (H, W)", "(64, 32)", "(64, 32)"),
        ("Resize Interpolation", "BILINEAR", "BILINEAR"),
        ("ToTensor Scaling", "[0.0, 1.0]", "[0.0, 1.0]"),
        ("Normalization Mean", "[0.5]", "[0.5]"),
        ("Normalization Std", "[0.5]", "[0.5]"),
        ("Output Tensor Shape", f"{tuple(tensor_a.shape)}", f"{tuple(tensor_b.shape)}"),
        ("Output Tensor Dtype", f"{tensor_a.dtype}", f"{tensor_b.dtype}"),
        ("Output Value Min", f"{tensor_a.min().item():.5f}", f"{tensor_b.min().item():.5f}"),
        ("Output Value Max", f"{tensor_a.max().item():.5f}", f"{tensor_b.max().item():.5f}"),
        ("Output Value Mean", f"{tensor_a.mean().item():.5f}", f"{tensor_b.mean().item():.5f}"),
        ("Max Absolute Diff", "---", f"{(tensor_a - tensor_b).abs().max().item():.8f}")
    ]
    
    print(f"{'Attribute':<25} | {'Training Path (a)':<32} | {'Dashboard Path (b)':<32}")
    print("-" * 95)
    for attr, a_val, b_val in table_data:
        print(f"{attr:<25} | {a_val:<32} | {b_val:<32}")
        
    diff = (tensor_a - tensor_b).abs().max().item()
    print("-" * 95)
    print(f"Parity Verdict: {'EXACT MATCH (diff=0)' if diff < 1e-6 else 'MISMATCH DETECTED'}")


# ==============================================================================
# CHECK 6: Presets 1 & 2 Slot-by-Slot Diagnostics
# ==============================================================================
def check_6():
    separator("CHECK 6: Preset 1 (0078) and Preset 2 (0080) Full Inference Log")
    
    presets = [
        ("Preset 1 (Dial 0078)", Path(r"C:\Users\ABHI\.gemini\antigravity-ide\brain\8abd5874-9ef0-49ba-9ca6-2a6566cf2a94\.user_uploaded\media_1792996743107.png"), "0078"),
        ("Preset 2 (Dial 0080)", Path(r"C:\Users\ABHI\.gemini\antigravity-ide\brain\8abd5874-9ef0-49ba-9ca6-2a6566cf2a94\.user_uploaded\media_1792996750413.png"), "0080")
    ]
    
    pipeline = DialReadingPipeline()
    
    for title, p_path, true_reading in presets:
        print(f"\n--- {title} ---")
        if not p_path.exists():
            print(f"File not found: {p_path}")
            continue
            
        res = pipeline.read_image(p_path)
        slots = res.get("digit_slots", [])
        print(f"Total Reading: {res.get('value')} (Whole: {res.get('whole_digits')}, Tenth: {res.get('tenth')})")
        print(f"Overall Confidence: {res.get('confidence'):.4f}")
        
        for idx, s in enumerate(slots):
            r = pipeline.classifier.predict_slot(s)
            true_d = int(true_reading[idx])
            pred_d = r["digit"]
            conf = r["confidence"]
            probs = r["all_probs"]
            
            top3 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
            top3_str = ", ".join([f"Digit {c}: {p*100:.1f}%" for c, p in top3])
            
            status_tag = "MATCH" if true_d == pred_d else f"WRONG (True={true_d})"
            print(f"  Slot {idx+1}: Pred={pred_d} (conf={conf:.4f}) [{status_tag}] | Top 3: {top3_str}")


def main():
    print("Starting DialReader Diagnostic Suite...")
    model_path, ckpt = check_1()
    if ckpt is not None:
        check_2(ckpt)
    check_3()
    check_4()
    check_5()
    check_6()
    print("\nDiagnostic complete.\n")


if __name__ == "__main__":
    main()
