"""
YOLO Dataset Exporter for DialReader
Reads data/labeled/labels.csv and exports:
- YOLO txt labels (class 0: display_window)
- Train / Val / Test splits (strictly honoring hold-out of Knife+Pressing and train-only video frames)
- configs/detector.yaml configuration file for Ultralytics
"""

import argparse
import ast
from pathlib import Path
import pandas as pd
import yaml


def parse_bbox(bbox_str):
    if not isinstance(bbox_str, str) or not bbox_str.strip():
        return None
    try:
        val = ast.literal_eval(bbox_str.strip())
        if isinstance(val, (list, tuple)) and len(val) == 4:
            return [float(x) for x in val]
    except Exception:
        pass
    return None


def export_yolo_dataset(
    labels_csv: Path,
    raw_dir: Path,
    output_yolo_dir: Path,
    detector_config_path: Path
):
    labels_csv = Path(labels_csv)
    raw_dir = Path(raw_dir)
    output_yolo_dir = Path(output_yolo_dir)
    output_yolo_dir.mkdir(parents=True, exist_ok=True)

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found at {labels_csv}")

    df = pd.read_csv(labels_csv)
    print(f"Loaded {len(df)} entries from {labels_csv}")

    # Subdirectories for YOLO
    for split in ["train", "val", "test"]:
        (output_yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Group by filename
    grouped = df.groupby("filename")
    
    # We assign splits:
    # 1. Knife+Pressing -> strictly test
    # 2. Video frames -> strictly train
    # 3. Knife photos -> split 80% train, 20% val (session-based order, NOT random)
    knife_photos = sorted([f for f in grouped.groups.keys() if f.startswith("knife/")])
    val_count = max(1, int(len(knife_photos) * 0.2))
    knife_val_files = set(knife_photos[-val_count:]) # Last 20% session as val

    split_counts = {"train": 0, "val": 0, "test": 0}

    for filename, group in grouped:
        src_path = raw_dir / filename
        if not src_path.exists():
            continue

        # Determine split
        if filename.startswith("knife_pressing/"):
            split = "test"
        elif filename.startswith("knife_video_frames/"):
            split = "train"
        elif filename in knife_val_files:
            split = "val"
        else:
            split = "train"

        # Safe unique image name
        img_id = filename.replace("/", "_").replace("\\", "_")
        dest_img_path = output_yolo_dir / "images" / split / img_id
        dest_label_path = output_yolo_dir / "labels" / split / (Path(img_id).stem + ".txt")

        # Copy or symlink image
        if not dest_img_path.exists():
            import shutil
            shutil.copy2(src_path, dest_img_path)

        # Write labels (display_window is class 0)
        lines = []
        for _, row in group.iterrows():
            bbox = parse_bbox(row.get("bbox", ""))
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox
                # Clamp coordinates to [0, 1]
                xmin = max(0.0, min(1.0, xmin))
                ymin = max(0.0, min(1.0, ymin))
                xmax = max(0.0, min(1.0, xmax))
                ymax = max(0.0, min(1.0, ymax))

                w = xmax - xmin
                h = ymax - ymin
                if w > 0.005 and h > 0.005:
                    xc = xmin + w / 2.0
                    yc = ymin + h / 2.0
                    lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        with open(dest_label_path, "w", encoding="utf-8") as lf:
            lf.writelines(lines)

        split_counts[split] += 1

    print("YOLO Dataset Export Summary:")
    for s, c in split_counts.items():
        print(f"  {s}: {c} images")

    # Generate configs/detector.yaml
    detector_config_path = Path(detector_config_path)
    detector_config_path.parent.mkdir(parents=True, exist_ok=True)
    yolo_cfg = {
        "path": str(output_yolo_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "display_window"
        }
    }
    with open(detector_config_path, "w", encoding="utf-8") as yf:
        yaml.dump(yolo_cfg, yf, default_flow_style=False)
    print(f"Saved YOLO configuration to: {detector_config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO dataset from labels.csv")
    parser.add_argument(
        "--labels_csv",
        type=str,
        default=r"c:\Users\ABHI\OneDrive\Desktop\DialReader\dial-reader\data\labeled\labels.csv"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=r"c:\Users\ABHI\OneDrive\Desktop\DialReader\dial-reader\data\raw"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"c:\Users\ABHI\OneDrive\Desktop\DialReader\dial-reader\data\labeled\yolo_dataset"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=r"c:\Users\ABHI\OneDrive\Desktop\DialReader\dial-reader\configs\detector.yaml"
    )
    args = parser.parse_args()
    export_yolo_dataset(
        Path(args.labels_csv),
        Path(args.raw_dir),
        Path(args.output_dir),
        Path(args.config_path)
    )
