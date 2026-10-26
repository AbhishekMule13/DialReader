"""
Stage 2: ANN Digit Classifier Training Pipeline for DialReader

Trains a Fully-Connected Network (Multi-Layer Perceptron) on discrete digit
slot crops (0-9). The single-digit model is evaluated 4 times per dial window.

Key Features & Engineering Decisions:
-------------------------------------
1. All hyperparameter settings are centralized in `configs/digit_ann.yaml`.
2. Architecture: Fully-connected layers with BatchNorm1d, non-linear activation
   (ReLU/GELU/LeakyReLU), and Dropout to prevent co-adaptation.
3. Label Smoothing (e.g. epsilon=0.05): Prevents overconfidence on transition
   and blurred digits where the drum is partially between positions.
4. Class-Balanced Loss: Computes inverse frequency class weights to prevent
   majority digits (like '0') from dominating gradients.
5. Strict Augmentation Rules: Photometric jitter and small tilt (+-8 deg) only.
   CRITICAL: Under NO circumstances are horizontal or vertical flips applied,
   as digits are orientation-sensitive.
6. Validation Integrity: Validation and testing are performed exclusively on
   REAL crops. Video frames and synthetic crops are strictly restricted to
   training augmentation.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import random

# Ensure dial-reader project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from src.models import DigitANN, get_ann_transform
from src.synthetic import SyntheticDigitGenerator


# Default fallback configuration matching configs/digit_ann.yaml
DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "data": {
        "input_size": [64, 32],
        "channels": 1,
        "batch_size": 32,
        "num_workers": 0,
        "train_split_ratio": 0.8
    },
    "model": {
        "hidden_layers": [256, 128, 64],
        "activation": "relu",
        "batchnorm": True,
        "dropout": 0.2,
        "num_classes": 10
    },
    "training": {
        "epochs": 40,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "label_smoothing": 0.05,
        "class_weights": True,
        "scheduler": {
            "type": "cosine",
            "min_lr": 0.000005,
            "step_size": 15,
            "gamma": 0.5
        },
        "early_stopping": {
            "enabled": True,
            "patience": 10,
            "min_delta": 0.001
        },
        "grad_clip": 1.0,
        "use_synthetic_augmentation": True,
        "synthetic_samples_per_digit": 40
    },
    "augmentation": {
        "brightness": 0.25,
        "contrast": 0.25,
        "rotation_deg": 8.0,
        "noise_std": 0.02,
        "horizontal_flip": False,
        "vertical_flip": False
    },
    "paths": {
        "labels_csv": "data/labeled/labels.csv",
        "raw_dir": "data/raw",
        "digits_dir": "data/digits",
        "checkpoint_dir": "models",
        "best_model_name": "digit_ann_best.pt",
        "results_dir": "results"
    }
}


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Loads configuration from a YAML file, merging with defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f)
                if user_cfg and isinstance(user_cfg, dict):
                    # Deep merge user_cfg into cfg
                    for k, v in user_cfg.items():
                        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                            cfg[k].update(v)
                        else:
                            cfg[k] = v
            print(f"[Config] Loaded settings from: {path}")
        else:
            print(f"[Config] Warning: {path} not found. Using defaults.")
    return cfg


class InMemoryCropDataset(Dataset):
    """
    In-memory PyTorch dataset for digit slot crops with dynamic transformation.
    """

    def __init__(self, samples: List[Tuple[Image.Image, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_crops_from_directory(
    split_dir: Path,
    transform=None
) -> Optional[Dataset]:
    """
    Loads crops from data/digits/<split>/{0..9}/*.png if directory contains images.
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return None

    # Count total images in split
    image_files = list(split_dir.glob("*/*.png")) + list(split_dir.glob("*/*.jpg"))
    if not image_files:
        return None

    return datasets.ImageFolder(root=str(split_dir), transform=transform)


def compute_class_weights(dataset: Dataset, num_classes: int = 10, device: str = "cpu") -> torch.Tensor:
    """
    Computes normalized inverse-frequency class weights:
        w_c = Total / (num_classes * count_c)
    Prevents majority classes from overwhelming the gradient.
    """
    counts = [0] * num_classes

    if isinstance(dataset, datasets.ImageFolder):
        for _, label in dataset.samples:
            if 0 <= label < num_classes:
                counts[label] += 1
    elif isinstance(dataset, InMemoryCropDataset):
        for _, label in dataset.samples:
            if 0 <= label < num_classes:
                counts[label] += 1

    total = sum(counts)
    if total == 0:
        return torch.ones(num_classes, dtype=torch.float32, device=device)

    weights = []
    for c in counts:
        if c > 0:
            w = total / (num_classes * c)
        else:
            w = 1.0
        weights.append(w)

    w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    # Normalize so mean weight is 1.0
    w_tensor = w_tensor / w_tensor.mean()
    return w_tensor


def train_digit_ann(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Union[str, Path]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main training execution function.
    Can be invoked programmatically by run_experiments.py or notebook.
    
    Returns:
        Dict with metrics: 'train_acc', 'val_acc', 'val_loss', 'param_count', 'best_epoch'
    """
    cfg = config if config is not None else load_config(config_path)
    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"[train_digit_ann] Running on device: {device.upper()}")

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    aug_cfg = cfg.get("augmentation", {})
    paths_cfg = cfg.get("paths", {})

    img_size = tuple(data_cfg.get("input_size", [64, 32]))
    channels = int(data_cfg.get("channels", 1))
    batch_size = int(data_cfg.get("batch_size", 32))
    input_dim = img_size[0] * img_size[1] * channels

    # 1. Prepare Transforms
    train_transform = get_ann_transform(img_size, channels=channels, is_train=True, aug_config=aug_cfg)
    val_transform = get_ann_transform(img_size, channels=channels, is_train=False)

    # 2. Load Datasets
    digits_dir = Path(paths_cfg.get("digits_dir", "data/digits"))
    train_dir = digits_dir / "train"
    val_dir = digits_dir / "val"

    train_ds = load_crops_from_directory(train_dir, transform=train_transform)
    val_ds = load_crops_from_directory(val_dir, transform=val_transform)

    # Check if crops exist on disk; if not, raise informative error
    if train_ds is None or len(train_ds) == 0:
        raise RuntimeError(
            f"No digit crops found in '{train_dir}'.\n"
            "Please label dial images in labels.csv and run:\n"
            "    python src/utils/generate_digit_crops.py\n"
            "before training the ANN model."
        )

    # Check synthetic augmentation option
    if train_cfg.get("use_synthetic_augmentation", False):
        synth_samples_count = int(train_cfg.get("synthetic_samples_per_digit", 40))
        if verbose:
            print(f"[Data] Adding synthetic augmentation ({synth_samples_count} per digit 0-9)...")
        gen = SyntheticDigitGenerator()
        synth_raw = gen.generate_batch(num_samples_per_digit=synth_samples_count)
        
        # Merge with existing train set
        existing_samples = []
        if isinstance(train_ds, datasets.ImageFolder):
            for path_str, lbl in train_ds.samples:
                existing_samples.append((Image.open(path_str).convert("RGB"), lbl))
        for item in synth_raw:
            existing_samples.append((item["image"], item["digit"]))

        train_ds = InMemoryCropDataset(existing_samples, transform=train_transform)

    if verbose:
        val_count = len(val_ds) if val_ds else 0
        print(f"[Data] Training samples: {len(train_ds)} | Validation samples: {val_count}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0)
    ) if val_ds and len(val_ds) > 0 else None

    # 3. Instantiate DigitANN Model
    model = DigitANN(
        input_dim=input_dim,
        hidden_layers=model_cfg.get("hidden_layers", [256, 128, 64]),
        num_classes=int(model_cfg.get("num_classes", 10)),
        activation=str(model_cfg.get("activation", "relu")),
        batchnorm=bool(model_cfg.get("batchnorm", True)),
        dropout=float(model_cfg.get("dropout", 0.2))
    ).to(device)

    params_info = model.count_parameters()
    if verbose:
        print(f"[Model] DigitANN initialized: {params_info['trainable_parameters']:,} trainable parameters.")

    # 4. Loss Function, Optimizer & Regularization
    class_weights = None
    if train_cfg.get("class_weights", True):
        class_weights = compute_class_weights(train_ds, num_classes=10, device=device)

    label_smoothing = float(train_cfg.get("label_smoothing", 0.05))
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    lr = float(train_cfg.get("lr", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 5. Learning Rate Scheduler
    sched_cfg = train_cfg.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine").lower()
    epochs = int(train_cfg.get("epochs", 40))

    if sched_type == "cosine":
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif sched_type == "step":
        step_sz = int(sched_cfg.get("step_size", 15))
        gamma = float(sched_cfg.get("gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_sz, gamma=gamma)
    else:
        scheduler = None

    # 6. Checkpoint & Early Stopping Setup
    chk_dir = Path(paths_cfg.get("checkpoint_dir", "models"))
    chk_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = chk_dir / paths_cfg.get("best_model_name", "digit_ann_best.pt")

    es_cfg = train_cfg.get("early_stopping", {})
    use_es = es_cfg.get("enabled", True)
    es_patience = int(es_cfg.get("patience", 10))
    es_min_delta = float(es_cfg.get("min_delta", 0.001))
    patience_counter = 0

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    # 7. Training Loop
    if verbose:
        print(f"\nStarting training for {epochs} epochs...\n" + "-" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping to prevent gradient explosion
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_acc = correct / max(1, total)
        train_loss = total_loss / max(1, total)

        # Validation step
        val_acc, val_loss = 0.0, float("inf")
        if val_loader:
            model.eval()
            v_loss, v_corr, v_tot = 0.0, 0, 0
            with torch.no_grad():
                for v_imgs, v_labels in val_loader:
                    v_imgs = v_imgs.to(device)
                    v_labels = v_labels.to(device)
                    v_logits = model(v_imgs)
                    loss_v = criterion(v_logits, v_labels)

                    v_loss += loss_v.item() * len(v_labels)
                    v_preds = v_logits.argmax(dim=-1)
                    v_corr += (v_preds == v_labels).sum().item()
                    v_tot += len(v_labels)

            val_acc = v_corr / max(1, v_tot)
            val_loss = v_loss / max(1, v_tot)

        # Update scheduler
        curr_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        # Track history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(curr_lr)

        # Check for improvement (prioritizing validation accuracy)
        improved = False
        if val_acc > best_val_acc + es_min_delta or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            improved = True
            
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "param_count": params_info["trainable_parameters"]
            }, best_model_path)
        else:
            patience_counter += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs or improved):
            imp_mark = " ★ (Best)" if improved else ""
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
                f"LR: {curr_lr:.6f}{imp_mark}"
            )

        # Early stopping condition
        if use_es and patience_counter >= es_patience:
            if verbose:
                print(f"[EarlyStopping] Early stopping triggered at epoch {epoch}. No validation improvement for {es_patience} epochs.")
            break

    if verbose:
        print("-" * 60)
        print(f"Training Completed. Best Val Acc: {best_val_acc:.3f} (Epoch {best_epoch}).")
        print(f"Best model weights saved to: {best_model_path}\n")

    return {
        "train_acc": train_acc,
        "val_acc": best_val_acc,
        "val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "param_count": params_info["trainable_parameters"],
        "checkpoint_path": str(best_model_path),
        "history": history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ANN Digit Classifier for DialReader")
    parser.add_argument("--config", type=str, default="configs/digit_ann.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    
    args = parser.parse_args()
    loaded_cfg = load_config(args.config)
    
    if args.epochs is not None:
        loaded_cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        loaded_cfg["training"]["lr"] = args.lr
    if args.batch is not None:
        loaded_cfg["data"]["batch_size"] = args.batch

    try:
        train_digit_ann(config=loaded_cfg)
    except RuntimeError as err:
        print(f"\n[Notice] {err}")
