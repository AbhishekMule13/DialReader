"""
Stage 1: YOLO Detector Training Script for DialReader
Fine-tunes YOLOv8n on single class 'display_window'.
Adheres to reproducibility (seed 42), leak-free splits, and metrics logging.
"""

import argparse
import shutil
from pathlib import Path
import yaml
import torch
import numpy as np


def train_detector(
    config_path: Path,
    epochs: int = 50,
    batch_size: int = 8,
    img_size: int = 640,
    seed: int = 42,
    base_model: str = "yolov8n.pt",
    output_model_path: Path = Path("models/detector_best.pt")
):
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    config_path = Path(config_path)
    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Detector config not found at {config_path}. "
            "Please run the labeling UI and export YOLO dataset first."
        )

    # Verify dataset exists and contains labels
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dataset_root = Path(cfg["path"])
    train_labels = list((dataset_root / "labels" / "train").glob("*.txt"))
    non_empty = [f for f in train_labels if f.stat().st_size > 0]

    if len(non_empty) == 0:
        raise RuntimeError(
            "No labeled display_window annotations found in train set. "
            "Please label images using 'streamlit run label_ui.py' before training!"
        )

    print(f"Starting YOLO detector training on {len(non_empty)} labeled images...")
    print(f"Config: {config_path}, Epochs: {epochs}, Batch: {batch_size}, Seed: {seed}")

    from ultralytics import YOLO
    model = YOLO(base_model)

    results = model.train(
        data=str(config_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        seed=seed,
        plots=True,
        save=True,
        device="cpu",  # or "0" if cuda available
        project="runs/detect",
        name="dial_detector",
        exist_ok=True
    )

    best_weight = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if best_weight.exists():
        shutil.copy2(best_weight, output_model_path)
        print(f"Saved best detector weights to: {output_model_path}")
    else:
        print("Notice: Best weights not found at expected location.")

    # Run validation
    metrics = model.val(data=str(config_path), split="val")
    print(f"Validation mAP50: {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train display window detector")
    parser.add_argument("--config", type=str, default="configs/detector.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_model", type=str, default="yolov8n.pt")
    parser.add_argument("--out", type=str, default="models/detector_best.pt")

    args = parser.parse_args()
    train_detector(
        Path(args.config),
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        seed=args.seed,
        base_model=args.base_model,
        output_model_path=Path(args.out)
    )
