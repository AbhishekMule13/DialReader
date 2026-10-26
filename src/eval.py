"""
Evaluation Script for DialReader

Evaluates the end-to-end pipeline (YOLO detector + ANN digit classifier)
on the held-out test set (Knife+Pressing).

Outputs & Metrics:
------------------
- Full-number exact-match accuracy (4 out of 4 digits exactly matching)
- Per-digit position accuracy (slots 1, 2, 3, 4)
- Last-digit (mid-roll drum) error rate reported separately
- Negative rejection rate (images 116, 117 correctly marked unreadable/negative)
- 4 Confusion matrices (saved to runs/eval/confusion_matrices.png)
- Saves the 20 worst failure cases with visual banner annotations (predicted vs true)
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageDraw

# Ensure dial-reader root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import DialReadingPipeline


def evaluate_test_set(
    labels_csv: Path,
    raw_dir: Path,
    detector_weights: Path,
    classifier_weights: Path,
    config_path: str = "configs/digit_ann.yaml",
    output_dir: Path = Path("runs/eval"),
    min_digit_conf: float = 0.65
) -> Dict[str, Any]:
    labels_csv = Path(labels_csv)
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(parents=True, exist_ok=True)

    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_csv}")

    df = pd.read_csv(labels_csv)
    # Test set rows: knife_pressing or Knife+Pressing
    test_df = df[df["filename"].str.lower().str.contains("knife[+_]pressing")].copy()
    if len(test_df) == 0:
        raise RuntimeError("No test set records found in labels.csv (matching 'knife+pressing').")

    pipeline = DialReadingPipeline(
        detector_weights=detector_weights,
        classifier_weights=classifier_weights,
        config_path=config_path,
        min_digit_conf=min_digit_conf
    )

    y_true_by_pos: List[List[int]] = [[], [], [], []]
    y_pred_by_pos: List[List[int]] = [[], [], [], []]

    total_dial_samples = 0
    exact_match_count = 0
    negative_tested = 0
    negative_correct = 0

    failures = []

    print(f"[evaluate_test_set] Evaluating on {len(test_df)} test entries (Knife+Pressing)...")
    for idx, row in test_df.iterrows():
        fname = str(row["filename"]).strip()
        true_reading = str(row["reading"]).strip() if pd.notna(row["reading"]) else ""
        variant = str(row.get("dial_variant", ""))
        ori = int(row.get("orientation", 0)) if pd.notna(row.get("orientation", 0)) and str(row.get("orientation")).isdigit() else 0

        img_path = raw_dir / fname
        if not img_path.exists():
            continue

        result = pipeline.read_image(img_path, manual_orientation=ori)

        # Negative test handling (116, 117)
        if "116" in fname or "117" in fname or variant == "non_dial_negative" or true_reading == "negative":
            negative_tested += 1
            if result["status"] == "unreadable" or result["value"] is None:
                negative_correct += 1
            else:
                failures.append({
                    "filename": fname,
                    "true": "negative",
                    "pred": result["value"],
                    "confidence": result["confidence"],
                    "reason": "False Positive detection on negative non-dial image",
                    "image_path": img_path
                })
            continue

        if len(true_reading) != 4 or not true_reading.isdigit():
            continue

        total_dial_samples += 1
        pred_value = result.get("value")
        pred_status = result.get("status")

        is_exact = (pred_value == true_reading and pred_status == "ok")
        if is_exact:
            exact_match_count += 1
        else:
            failures.append({
                "filename": fname,
                "true": true_reading,
                "pred": pred_value or "unreadable",
                "confidence": result.get("confidence", 0.0),
                "per_conf": result.get("per_digit_confidence", []),
                "reason": result.get("reason", "Digit mismatch"),
                "image_path": img_path
            })

        # Track per-digit predictions
        if pred_value and len(pred_value) == 4 and pred_value.isdigit():
            for pos in range(4):
                y_true_by_pos[pos].append(int(true_reading[pos]))
                y_pred_by_pos[pos].append(int(pred_value[pos]))
        else:
            # If unreadable, penalize with -1
            for pos in range(4):
                y_true_by_pos[pos].append(int(true_reading[pos]))
                y_pred_by_pos[pos].append(-1)

    # Calculate metrics
    exact_acc = (exact_match_count / max(1, total_dial_samples)) * 100.0
    print("\n" + "=" * 55)
    print("HELD-OUT TEST SET (KNIFE+PRESSING) EVALUATION REPORT")
    print("=" * 55)
    print(f"Total Dial Test Images: {total_dial_samples}")
    print(f"Full 4-Digit Exact Match Accuracy: {exact_acc:.1f}% ({exact_match_count}/{total_dial_samples})")

    neg_acc = 0.0
    if negative_tested > 0:
        neg_acc = (negative_correct / negative_tested) * 100.0
        print(f"Negative Image Rejection Rate (116/117): {neg_acc:.1f}% ({negative_correct}/{negative_tested})")

    per_pos_acc = []
    for pos in range(4):
        yt = np.array(y_true_by_pos[pos])
        yp = np.array(y_pred_by_pos[pos])
        acc = float(np.mean(yt == yp)) * 100.0 if len(yt) > 0 else 0.0
        per_pos_acc.append(acc)
        print(f"  Digit Slot {pos+1} Accuracy: {acc:.1f}%")

    last_digit_err = 100.0 - per_pos_acc[3] if per_pos_acc else 0.0
    print(f"Last Digit (Drum 4 / Mid-Roll) Error Rate: {last_digit_err:.1f}%")
    print("=" * 55)

    # Plot Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    classes = list(range(10))
    for pos, ax in enumerate(axes.flat):
        yt = [y for y, p in zip(y_true_by_pos[pos], y_pred_by_pos[pos]) if p != -1]
        yp = [p for y, p in zip(y_true_by_pos[pos], y_pred_by_pos[pos]) if p != -1]
        if yt and yp:
            cm = confusion_matrix(yt, yp, labels=classes)
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"Slot {pos+1} (Acc: {per_pos_acc[pos]:.1f}%)")
            ax.set_xticks(classes)
            ax.set_yticks(classes)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(10):
                for j in range(10):
                    val = cm[i, j]
                    if val > 0:
                        ax.text(j, i, str(val), ha="center", va="center", color="white" if val > cm.max()/2 else "black")
        else:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center")

    plt.tight_layout()
    cm_path = output_dir / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrices to: {cm_path}")

    # Save top 20 worst failures
    failures_to_save = min(20, len(failures))
    print(f"Saving top {failures_to_save} worst failure visualizations to {failures_dir}...")
    failures.sort(key=lambda f: f.get("confidence", 0.0), reverse=True)  # Overconfident mistakes first

    for i, fail in enumerate(failures[:20]):
        src_p = fail.get("image_path")
        if not src_p or not src_p.exists():
            continue
        try:
            with Image.open(src_p) as orig_img:
                img = orig_img.copy()
                w, h = img.size
                banner_h = max(40, int(h * 0.08))
                banner = Image.new("RGB", (w, banner_h), color=(220, 20, 20))
                b_draw = ImageDraw.Draw(banner)
                text = f"True: {fail['true']} | Pred: {fail['pred']} | Conf: {fail['confidence']:.2f} | Reason: {fail['reason']}"
                b_draw.text((20, int(banner_h * 0.3)), text, fill=(255, 255, 255))

                composite = Image.new("RGB", (w, h + banner_h))
                composite.paste(banner, (0, 0))
                composite.paste(img, (0, banner_h))

                out_fail_path = failures_dir / f"fail_{i+1:02d}_{Path(fail['filename']).stem}.jpg"
                composite.save(out_fail_path, quality=90)
        except Exception as e:
            print(f"Warning: could not save failure visual for {fail['filename']}: {e}")

    print("Evaluation completed successfully.\n")
    return {
        "exact_accuracy": exact_acc,
        "per_digit_accuracy": per_pos_acc,
        "last_digit_error": last_digit_err,
        "negative_rejection": neg_acc,
        "failures_count": len(failures)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DialReader on Test Set")
    parser.add_argument("--labels_csv", type=str, default="data/labeled/labels.csv")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--detector", type=str, default="models/detector_best.pt")
    parser.add_argument("--classifier", type=str, default="models/digit_ann_best.pt")
    parser.add_argument("--config", type=str, default="configs/digit_ann.yaml")
    parser.add_argument("--out_dir", type=str, default="runs/eval")

    args = parser.parse_args()
    try:
        evaluate_test_set(
            Path(args.labels_csv),
            Path(args.raw_dir),
            Path(args.detector),
            Path(args.classifier),
            config_path=args.config,
            output_dir=Path(args.out_dir)
        )
    except Exception as e:
        print(f"[Notice] {e}")
