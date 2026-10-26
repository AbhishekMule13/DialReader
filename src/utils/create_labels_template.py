"""
Labels Template Generator for DialReader
Scans raw image directories and creates or updates labels.csv:
Fields: filename, bbox, reading, orientation, mid_roll, dial_variant, split_group
Rule 1: Knife photos -> train/val
Rule 2: Knife video frames -> strictly train
Rule 3: Knife+Pressing -> strictly test
Rule 6: 116/117 marked as negative examples
"""

import argparse
from pathlib import Path
import pandas as pd


NEGATIVE_FILENAMES = {"116.jpeg", "117.jpeg", "116,117.jpeg"}


def create_template(
    raw_dir: Path,
    output_csv: Path,
    include_video: bool = True
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Existing records if already labeled
    existing_df = None
    if output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            print(f"Loaded {len(existing_df)} existing records from {output_csv}")
        except Exception as e:
            print(f"Notice: Could not parse existing CSV ({e}), creating fresh template.")

    records = []
    seen_files = set()
    if existing_df is not None and "filename" in existing_df.columns:
        seen_files = set(existing_df["filename"].dropna())
        records = existing_df.to_dict("records")

    # 1. Knife photos
    knife_dir = raw_dir / "knife"
    if knife_dir.exists():
        for p in sorted(knife_dir.glob("*.jpg")):
            rel_path = f"knife/{p.name}"
            if rel_path not in seen_files:
                records.append({
                    "filename": rel_path,
                    "bbox": "",
                    "reading": "",
                    "orientation": 0,
                    "mid_roll": False,
                    "dial_variant": "standard_3black_1red",
                    "split_group": "train_or_val"
                })
                seen_files.add(rel_path)

    # 2. Knife+Pressing photos (held out entirely as test set)
    kp_dir = raw_dir / "knife_pressing"
    if kp_dir.exists():
        for p in sorted(kp_dir.glob("*.jpeg")):
            rel_path = f"knife_pressing/{p.name}"
            if rel_path not in seen_files:
                is_neg = p.name in NEGATIVE_FILENAMES
                # Multi-dial note: 103,104.jpeg has 2 dials, add 2 slots
                num_slots = 2 if "103,104" in p.name else 1
                for slot in range(num_slots):
                    records.append({
                        "filename": rel_path,
                        "bbox": "" if not is_neg else "",
                        "reading": "negative" if is_neg else "",
                        "orientation": 0 if not is_neg else 0,
                        "mid_roll": False,
                        "dial_variant": "non_dial_negative" if is_neg else "standard_3black_1red",
                        "split_group": "test"
                    })
                seen_files.add(rel_path)

    # 3. Video frames (strictly train)
    if include_video:
        vid_dir = raw_dir / "knife_video_frames"
        if vid_dir.exists():
            for p in sorted(vid_dir.glob("*.jpg")):
                rel_path = f"knife_video_frames/{p.name}"
                if rel_path not in seen_files:
                    records.append({
                        "filename": rel_path,
                        "bbox": "",
                        "reading": "",
                        "orientation": 0,
                        "mid_roll": False,
                        "dial_variant": "standard_3black_1red",
                        "split_group": "train"
                    })
                    seen_files.add(rel_path)

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved template with {len(df)} rows to: {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labels.csv template")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=r"c:\Users\ABHI\OneDrive\Desktop\DialReader\dial-reader\data\raw",
        help="Raw data directory"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=r"c:\Users\ABHI\OneDrive\Desktop\DialReader\dial-reader\data\labeled\labels.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()
    create_template(Path(args.raw_dir), Path(args.output_csv))
