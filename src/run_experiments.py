"""
Systematic Experiment Runner for DialReader Digit ANN

Trains several architectural and hyperparameter configurations, varying ONE
setting at a time against a defined baseline (One-Factor-at-a-Time / OFAT).

Logs performance metrics into `results/experiments.csv`:
  - experiment_name
  - hidden_layers
  - activation
  - batchnorm
  - dropout
  - lr
  - param_count
  - train_acc
  - val_acc
  - val_loss
  - best_epoch
  - status

Supports `--dry-run` to verify architectures, parameter counts, and pipeline
integrity before full training begins.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure dial-reader project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.models import DigitANN
from src.train_digits import load_config, train_digit_ann


# Systematic experiments list (varying one hyperparameter from baseline)
EXPERIMENT_SUITE: List[Dict[str, Any]] = [
    {
        "name": "baseline",
        "desc": "Reference baseline architecture",
        "overrides": {}  # uses configs/digit_ann.yaml defaults
    },
    {
        "name": "arch_wide",
        "desc": "Wider hidden capacity [512, 256, 128]",
        "overrides": {
            "model": {"hidden_layers": [512, 256, 128]}
        }
    },
    {
        "name": "arch_deep",
        "desc": "Deeper network [256, 256, 128, 64]",
        "overrides": {
            "model": {"hidden_layers": [256, 256, 128, 64]}
        }
    },
    {
        "name": "arch_compact",
        "desc": "Lightweight 2-layer network [128, 64]",
        "overrides": {
            "model": {"hidden_layers": [128, 64]}
        }
    },
    {
        "name": "act_gelu",
        "desc": "GELU activation function",
        "overrides": {
            "model": {"activation": "gelu"}
        }
    },
    {
        "name": "act_leaky_relu",
        "desc": "LeakyReLU (slope 0.1) activation",
        "overrides": {
            "model": {"activation": "leaky_relu"}
        }
    },
    {
        "name": "dropout_none",
        "desc": "No dropout (p=0.0)",
        "overrides": {
            "model": {"dropout": 0.0}
        }
    },
    {
        "name": "dropout_high",
        "desc": "Strong regularization (p=0.4)",
        "overrides": {
            "model": {"dropout": 0.4}
        }
    },
    {
        "name": "no_batchnorm",
        "desc": "Ablation without Batch Normalization",
        "overrides": {
            "model": {"batchnorm": False}
        }
    },
    {
        "name": "lr_half",
        "desc": "Lower learning rate (0.0005)",
        "overrides": {
            "training": {"lr": 0.0005}
        }
    }
]


def apply_overrides(base_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Applies nested overrides to a base dictionary."""
    cfg = copy.deepcopy(base_cfg)
    for section, values in overrides.items():
        if isinstance(values, dict) and section in cfg and isinstance(cfg[section], dict):
            cfg[section].update(values)
        else:
            cfg[section] = values
    return cfg


def run_experiments(
    config_path: str = "configs/digit_ann.yaml",
    results_csv: str = "results/experiments.csv",
    dry_run: bool = False,
    quick_epochs: int = None
) -> pd.DataFrame:
    """
    Executes all experiments sequentially and saves results to CSV.
    """
    base_cfg = load_config(config_path)
    res_path = Path(results_csv)
    res_path.parent.mkdir(parents=True, exist_ok=True)

    results_records: List[Dict[str, Any]] = []

    print("=" * 75)
    print("DIALREADER ANN HYPERPARAMETER EXPERIMENT RUNNER")
    print("=" * 75)
    print(f"Total experiments queued: {len(EXPERIMENT_SUITE)}")
    print(f"Mode: {'DRY RUN (Model verification only)' if dry_run else 'FULL TRAINING'}")
    print(f"Results output: {res_path}\n")

    for i, exp in enumerate(EXPERIMENT_SUITE, start=1):
        name = exp["name"]
        desc = exp["desc"]
        overrides = exp["overrides"]

        exp_cfg = apply_overrides(base_cfg, overrides)
        if quick_epochs is not None:
            exp_cfg["training"]["epochs"] = quick_epochs

        # Query model details
        data_cfg = exp_cfg.get("data", {})
        model_cfg = exp_cfg.get("model", {})
        train_cfg = exp_cfg.get("training", {})

        img_size = tuple(data_cfg.get("input_size", [64, 32]))
        channels = int(data_cfg.get("channels", 1))
        input_dim = img_size[0] * img_size[1] * channels

        h_layers = model_cfg.get("hidden_layers", [256, 128, 64])
        act = model_cfg.get("activation", "relu")
        bn = model_cfg.get("batchnorm", True)
        drop = model_cfg.get("dropout", 0.2)
        lr = train_cfg.get("lr", 0.001)

        # Build model and calculate parameter count
        model_instance = DigitANN(
            input_dim=input_dim,
            hidden_layers=h_layers,
            num_classes=10,
            activation=act,
            batchnorm=bn,
            dropout=drop
        )
        param_count = model_instance.count_parameters()["trainable_parameters"]

        print(f"[{i:02d}/{len(EXPERIMENT_SUITE):02d}] Experiment '{name}': {desc}")
        print(f"     Layers: {h_layers} | Act: {act} | BN: {bn} | Drop: {drop} | LR: {lr}")
        print(f"     Trainable Parameters: {param_count:,}")

        record: Dict[str, Any] = {
            "experiment_name": name,
            "description": desc,
            "hidden_layers": str(h_layers),
            "activation": act,
            "batchnorm": bn,
            "dropout": drop,
            "lr": lr,
            "param_count": param_count,
            "train_acc": None,
            "val_acc": None,
            "val_loss": None,
            "best_epoch": None,
            "status": "pending"
        }

        if dry_run:
            # Verify forward pass on dummy batch
            dummy_x = torch.randn(4, channels, img_size[0], img_size[1])
            logits = model_instance(dummy_x)
            assert logits.shape == (4, 10), f"Unexpected output shape: {logits.shape}"
            record["status"] = "dry_run_verified"
            print("     Verification: SUCCESS (forward pass validated)\n")
        else:
            try:
                # Custom checkpoint name for this experiment
                exp_cfg["paths"]["best_model_name"] = f"digit_ann_{name}.pt"
                train_res = train_digit_ann(config=exp_cfg, verbose=False)

                record["train_acc"] = round(train_res["train_acc"], 4)
                record["val_acc"] = round(train_res["val_acc"], 4)
                record["val_loss"] = round(train_res["val_loss"], 4)
                record["best_epoch"] = train_res["best_epoch"]
                record["status"] = "completed"

                print(f"     Result: Val Acc: {record['val_acc']:.3f} | Train Acc: {record['train_acc']:.3f} | Val Loss: {record['val_loss']:.4f}\n")
            except Exception as e:
                record["status"] = f"error: {str(e)[:60]}"
                print(f"     Execution halted: {e}\n")

        results_records.append(record)

    df_results = pd.DataFrame(results_records)
    df_results.to_csv(res_path, index=False)
    print(f"Results successfully saved to: {res_path}")

    # Summary table display
    print("\n" + "=" * 95)
    print("EXPERIMENT SUITE SUMMARY TABLE")
    print("=" * 95)
    cols_to_print = ["experiment_name", "hidden_layers", "activation", "dropout", "param_count", "val_acc", "status"]
    print(df_results[cols_to_print].to_string(index=False))
    print("=" * 95 + "\n")

    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run systematic ANN experiments for DialReader")
    parser.add_argument("--config", type=str, default="configs/digit_ann.yaml", help="Path to base config YAML")
    parser.add_argument("--output", type=str, default="results/experiments.csv", help="Path to output CSV")
    parser.add_argument("--dry_run", action="store_true", help="Instantiate models and verify forward pass without training")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for quick sweeps")

    args = parser.parse_args()
    run_experiments(
        config_path=args.config,
        results_csv=args.output,
        dry_run=args.dry_run,
        quick_epochs=args.epochs
    )
