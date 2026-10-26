"""
Crop Generation Script for DialReader

Extracts individual digit slot crops (0-9) from labeled display windows in labels.csv,
and sorts them into:
    data/digits/{train, val, test}/{0..9}/<crop_name>.png

Split Strategy (Deterministic by Source, NOT Random):
  1. Knife Photos:
     - Partitioned deterministically into 'train' and 'val' (e.g. 80% train, 20% val).
  2. Knife Video Frames:
     - Put into 'train' ONLY. Strict Rule: Video frames are NEVER included in val or test.
  3. Knife+Pressing:
     - Held-out 'test' set exclusively.
  4. Negative Images (116, 117):
     - Filtered out completely; not parsed as valid digits.
"""

import ast
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from PIL import Image

# Ensure dial-reader root is on path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crop import WindowNormalizer


def is_negative_example(filename: str) -> bool:
    """
    Checks if filename corresponds to negative examples (e.g., 116, 117).
    """
    f_lower = filename.lower()
    return "116" in f_lower or "117" in f_lower


def assign_split(filename: str, split_group_col: str, train_ratio: float = 0.8) -> str:
    """
    Determines whether an image belongs to 'train', 'val', or 'test'.
    """
    f_clean = filename.replace("\\", "/").lower()

    # Rule: Knife+Pressing is strictly the held-out test set
    if "knife+pressing" in f_clean or "knife_pressing" in f_clean:
        return "test"

    # Rule: Sparse video frames are NEVER allowed in val or test
    if "video" in f_clean:
        return "train"

    # For Knife static photos:
    # If explicitly flagged in CSV's split_group column
    if isinstance(split_group_col, str) and split_group_col.strip():
        val = split_group_col.strip().lower()
        if "val" in val:
            return "val"
        elif "train" in val:
            return "train"

    # Deterministic hash split for static knife photos
    # Using hash of filename ensures reproducibility without data leakage
    h = hash(f_clean) % 100
    if h < int(train_ratio * 100):
        return "train"
    else:
        return "val"


def generate_digit_crops(
    labels_csv: Path,
    raw_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8
) -> Dict[str, Any]:
    labels_csv = Path(labels_csv)
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    if not labels_csv.exists():
        print(f"Error: {labels_csv} does not exist. Please create/fill labels first.")
        return {"total_crops": 0, "status": "missing_csv"}

    df = pd.read_csv(labels_csv)
    normalizer = WindowNormalizer()

    # Create destination directories: data/digits/{train,val,test}/{0..9}
    splits = ["train", "val", "test"]
    for s in splits:
        for d in range(10):
            (output_dir / s / str(d)).mkdir(parents=True, exist_ok=True)

    counts: Dict[str, Dict[int, int]] = {
        s: {d: 0 for d in range(10)} for s in splits
    }

    total_windows_processed = 0
    total_crops_saved = 0
    skipped_negatives = 0

    print(f"[generate_digit_crops] Processing {len(df)} entries from {labels_csv}...")

    for _, row in df.iterrows():
        fname = str(row["filename"]).strip()
        reading = str(row["reading"]).strip() if pd.notna(row["reading"]) else ""
        bbox_str = str(row["bbox"]).strip() if pd.notna(row["bbox"]) else ""
        ori = int(row["orientation"]) if pd.notna(row.get("orientation")) and str(row["orientation"]).isdigit() else 0
        split_col = str(row.get("split_group", ""))

        # Check negative sample rule
        if is_negative_example(fname):
            skipped_negatives += 1
            continue

        # Skip unlabeled or incomplete entries
        if len(reading) != 4 or not reading.isdigit() or not bbox_str:
            continue

        try:
            bbox = ast.literal_eval(bbox_str)
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
        except Exception:
            continue

        img_path = raw_dir / fname
        if not img_path.exists():
            continue

        split = assign_split(fname, split_col, train_ratio=train_ratio)

        try:
            with Image.open(img_path) as full_img:
                w, h = full_img.size
                xmin, ymin, xmax, ymax = bbox
                px_box = (
                    int(round(xmin * w)),
                    int(round(ymin * h)),
                    int(round(xmax * w)),
                    int(round(ymax * h))
                )
                
                # Safeguard bounding box bounds
                px_box = (
                    max(0, px_box[0]),
                    max(0, px_box[1]),
                    min(w, px_box[2]),
                    min(h, px_box[3])
                )
                
                if px_box[2] <= px_box[0] or px_box[3] <= px_box[1]:
                    continue

                window_crop = full_img.crop(px_box)
                proc = normalizer.process(window_crop, orientation=ori)
                slot_crops = proc["digit_slots"]

                stem = Path(fname).stem.replace(" ", "_")

                for pos_idx, (d_char, slot_img) in enumerate(zip(reading, slot_crops)):
                    digit = int(d_char)
                    out_path = output_dir / split / str(digit) / f"{stem}_slot{pos_idx}.png"
                    slot_img.save(out_path)
                    counts[split][digit] += 1
                    total_crops_saved += 1

                total_windows_processed += 1

        except Exception as e:
            print(f"Warning: Failed processing {fname}: {e}")
            continue

    # Print summary table
    print("\n" + "=" * 50)
    print("DIGIT CROP EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total labeled windows parsed: {total_windows_processed}")
    print(f"Negative samples skipped (116/117): {skipped_negatives}")
    print(f"Total single digit crops saved: {total_crops_saved}")
    print("-" * 50)
    print("Digit | Train |  Val  | Test | Total")
    print("-" * 50)
    for d in range(10):
        t_c = counts["train"][d]
        v_c = counts["val"][d]
        te_c = counts["test"][d]
        tot = t_c + v_c + te_c
        print(f"  {d}   |  {t_c:4d} |  {v_c:4d} | {te_c:4d} | {tot:5d}")
    print("=" * 50 + "\n")

    return {
        "windows_processed": total_windows_processed,
        "crops_saved": total_crops_saved,
        "counts": counts,
        "status": "success"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate discrete digit crops from labels.csv")
    parser.add_argument("--labels_csv", type=str, default="data/labeled/labels.csv")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/digits")
    parser.add_argument("--train_ratio", type=float, default=0.8)

    args = parser.parse_args()
    generate_digit_crops(
        labels_csv=Path(args.labels_csv),
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio
    )
