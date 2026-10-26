"""
Build Real Digit Dataset from readings.csv (Step 2)
Extracts per-slot crops with the existing slot cropper and saves them
to data/digits/<digit>/... with a manifest recording metadata.
Reports class balance and flags classes with < 30 samples.
"""

import sys
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

READINGS_CSV = PROJECT_ROOT / "data" / "labeled" / "readings.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "labeled" / "labels.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DIGITS_DIR = PROJECT_ROOT / "data" / "digits"
MANIFEST_PATH = DIGITS_DIR / "manifest.csv"

from src.infer import DialReadingPipeline


def build_dataset(clean_existing: bool = True):
    print("=" * 70)
    print(" Building Real Digit Dataset (Step 2)")
    print("=" * 70)
    
    if not READINGS_CSV.exists():
        print(f"ERROR: {READINGS_CSV} does not exist! Please label images first with label_readings.py.")
        return False, {}
        
    df_readings = pd.read_csv(READINGS_CSV)
    df_labels = pd.read_csv(LABELS_CSV) if LABELS_CSV.exists() else pd.DataFrame()
    
    # Merge split_group and orientation from labels.csv if present
    if not df_labels.empty and "filename" in df_labels.columns:
        meta_cols = [c for c in ["filename", "split_group", "orientation"] if c in df_labels.columns]
        merged_df = pd.merge(df_readings, df_labels[meta_cols], on="filename", how="left")
    else:
        merged_df = df_readings.copy()
        merged_df["split_group"] = "train"
        merged_df["orientation"] = 0

    # Filter strictly valid 4-digit numeric readings
    valid_mask = (
        merged_df["reading"].notnull() &
        merged_df["reading"].astype(str).str.strip().str.isdigit() &
        (merged_df["reading"].astype(str).str.strip().str.len() == 4)
    )
    valid_df = merged_df[valid_mask].copy()
    valid_df["reading"] = valid_df["reading"].astype(str).str.strip()
    
    print(f"Total rows in readings.csv:  {len(df_readings)}")
    print(f"Valid 4-digit dial readings: {len(valid_df)}")
    
    if len(valid_df) == 0:
        print("No valid 4-digit entries in readings.csv yet.")
        print("Please enter readings using: streamlit run label_readings.py --server.port 8503")
        return False, {}
        
    # Prepare directory structure
    if clean_existing and DIGITS_DIR.exists():
        # Remove old digit folders
        for d in range(10):
            d_dir = DIGITS_DIR / str(d)
            if d_dir.exists():
                shutil.rmtree(d_dir)
                
    for d in range(10):
        (DIGITS_DIR / str(d)).mkdir(parents=True, exist_ok=True)
        
    pipeline = DialReadingPipeline()
    manifest_rows = []
    class_counts = {d: 0 for d in range(10)}
    split_counts = {}
    
    print(f"\nProcessing {len(valid_df)} dial images into 4 slot crops each...")
    
    for idx, row in valid_df.iterrows():
        rel_fn = row["filename"]
        raw_path = RAW_DIR / rel_fn
        reading_str = row["reading"]
        tenth = row.get("tenth_optional", "")
        split_grp = row.get("split_group", "train")
        ori = int(row["orientation"]) if pd.notnull(row.get("orientation")) else None
        
        if not raw_path.exists():
            print(f"  Warning: Raw image not found: {raw_path}")
            continue
            
        try:
            res = pipeline.read_image(raw_path, manual_orientation=ori)
            slots = res.get("digit_slots", [])
        except Exception as e:
            print(f"  Error extracting slots for {rel_fn}: {e}")
            continue
            
        if res.get("status") == "no_window":
            print(f"  Skipping {rel_fn}: no window found (status='no_window').")
            continue

        if len(slots) != 4:
            print(f"  Warning: Expected 4 slots for {rel_fn}, got {len(slots)}. Skipping.")
            continue
            
        stem = Path(rel_fn).stem.replace(",", "_").replace(" ", "_")
        parent_tag = Path(rel_fn).parent.name
        
        for slot_idx in range(4):
            digit_char = reading_str[slot_idx]
            digit = int(digit_char)
            
            crop_filename = f"{parent_tag}_{stem}_slot{slot_idx}_d{digit}.png"
            crop_path = DIGITS_DIR / str(digit) / crop_filename
            rel_crop_path = f"data/digits/{digit}/{crop_filename}"
            
            # Save slot crop
            slots[slot_idx].save(crop_path, format="PNG")
            class_counts[digit] += 1
            split_counts[split_grp] = split_counts.get(split_grp, 0) + 1
            
            manifest_rows.append({
                "crop_path": rel_crop_path,
                "source_image": rel_fn,
                "slot_index": slot_idx,
                "digit": digit,
                "split_group": split_grp,
                "tenth_optional": tenth
            })
            
    # Save manifest
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(MANIFEST_PATH, index=False)
    print(f"\nSaved manifest with {len(df_manifest)} slot crops to: {MANIFEST_PATH}")
    
    # Class distribution report
    print("\n" + "=" * 50)
    print(f"{'Digit Class':<12} | {'Count':<8} | {'Status'}")
    print("-" * 50)
    flagged_classes = []
    for d in range(10):
        cnt = class_counts[d]
        if cnt < 30:
            status = f"⚠️ FLAGGED (< 30 examples)"
            flagged_classes.append((d, cnt))
        else:
            status = "✓ OK (>= 30)"
        print(f"Digit {d:<6} | {cnt:<8} | {status}")
    print("-" * 50)
    print(f"Total Crops: {sum(class_counts.values())}")
    print(f"Split distribution: {split_counts}")
    
    if flagged_classes:
        print(f"\n[ALERT] {len(flagged_classes)} classes have fewer than 30 real examples: {flagged_classes}")
        print("Synthetic rolling-drum augmentation will supplement underrepresented classes during training.")
    else:
        print("\nAll classes meet the 30-sample minimum threshold!")
        
    return True, class_counts


if __name__ == "__main__":
    build_dataset()
