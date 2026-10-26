"""
Stage 2: ANN (Multi-Layer Perceptron) Digit Classifier for DialReader

Replaces the heavy CNN/MobileNet architecture with a clean, fully-connected
network that classifies ONE digit slot crop (0-9) at a time. For each dial
window, the pipeline runs this model 4 times (once per discrete slot).

Architecture:
  Flattened Crop -> [Linear -> (BatchNorm1d) -> Activation -> (Dropout)] * N -> Linear -> 10 classes
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def get_activation(name: str) -> nn.Module:
    """
    Returns the activation module corresponding to the given string name.
    """
    name = str(name).lower().strip()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: '{name}'. Choose from 'relu', 'leaky_relu', 'gelu', 'silu', 'tanh'.")


class DigitANN(nn.Module):
    """
    Fully-connected neural network (MLP) for single digit-slot classification.
    
    Parameters:
    -----------
    input_dim : int
        Flattened input dimension (e.g., 64 * 32 * 1 = 2048 for 64x32 grayscale).
    hidden_layers : List[int]
        Neuron counts for each hidden layer (e.g., [256, 128, 64]).
    num_classes : int
        Number of output classes (10 for digits 0-9).
    activation : str
        Activation function between linear layers.
    batchnorm : bool
        Whether to apply Batch Normalization after linear transformations.
    dropout : float
        Dropout probability to regularize hidden representations.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_layers: Optional[List[int]] = None,
        num_classes: int = 10,
        activation: str = "relu",
        batchnorm: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        self.input_dim = input_dim
        self.hidden_layers = list(hidden_layers)
        self.num_classes = num_classes
        self.activation_name = activation
        self.use_batchnorm = batchnorm
        self.dropout_rate = float(dropout)

        layers: List[nn.Module] = []
        curr_dim = input_dim

        # Construct each hidden stage
        for i, h_dim in enumerate(self.hidden_layers):
            layers.append(nn.Linear(curr_dim, h_dim))
            
            # Batch Normalization: standardizes activations across the batch,
            # allowing higher learning rates and reducing dependency on initialization.
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            
            # Non-linear activation: enables learning non-linear digit boundary features.
            layers.append(get_activation(self.activation_name))
            
            # Dropout: randomly drops neurons during training to prevent co-adaptation.
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))
            
            curr_dim = h_dim

        # Final classification head projecting to digit logits (0..9)
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(curr_dim, num_classes)

        # Initialize weights with proper scaling
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        Applies Kaiming (He) normal initialization for ReLU/GELU layers,
        or Xavier for Tanh, with zero bias initialization.
        """
        if isinstance(m, nn.Linear):
            if self.activation_name in ("relu", "leaky_relu"):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            else:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Accepts:
            x of shape (Batch, Channels, Height, Width) or (Batch, input_dim)
        Returns:
            Logits of shape (Batch, num_classes)
        """
        # Flatten any multi-dimensional spatial tensor: (B, C, H, W) -> (B, C*H*W)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        feats = self.feature_extractor(x)
        logits = self.classifier(feats)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """
        Returns the total and trainable parameter count of the model.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }


def get_ann_transform(
    img_size: Tuple[int, int] = (64, 32),
    channels: int = 1,
    is_train: bool = False,
    aug_config: Optional[Dict[str, Any]] = None
) -> transforms.Compose:
    """
    Constructs the preprocessing and augmentation pipeline for a single digit slot crop.
    
    CRITICAL RULE: Digits are not mirror-symmetric. Under NO circumstances should
    horizontal or vertical flips ever be added.
    """
    tf_list: List[Any] = []

    # 1. Resize to canonical crop size [height, width]
    tf_list.append(transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR))

    # 2. Convert to Grayscale if 1 channel (removes drum color bias between black and red)
    if channels == 1:
        tf_list.append(transforms.Grayscale(num_output_channels=1))

    # 3. Training augmentations (photometric and subtle geometric jitter)
    if is_train and aug_config:
        brightness = aug_config.get("brightness", 0.2)
        contrast = aug_config.get("contrast", 0.2)
        rot_deg = aug_config.get("rotation_deg", 8.0)

        if brightness > 0 or contrast > 0:
            tf_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))
        if rot_deg > 0:
            # Small rotation to account for slightly misaligned dial housings
            tf_list.append(transforms.RandomRotation(degrees=(-rot_deg, rot_deg)))

    # 4. Convert PIL to Tensor (scales pixel values to [0.0, 1.0])
    tf_list.append(transforms.ToTensor())

    # 5. Standard normalization: zero-centered with unit variance
    if channels == 1:
        tf_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        tf_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return transforms.Compose(tf_list)


class DigitANNPredictor:
    """
    High-level inference predictor for digit crops using the trained DigitANN model.
    Encapsulates slot classification and 4-slot dial reading.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.config = config or {}

        # Resolve architecture settings
        data_cfg = self.config.get("data", {})
        model_cfg = self.config.get("model", {})
        
        self.img_size = tuple(data_cfg.get("input_size", [64, 32]))
        self.channels = int(data_cfg.get("channels", 1))
        input_dim = self.img_size[0] * self.img_size[1] * self.channels

        hidden_layers = model_cfg.get("hidden_layers", [256, 128, 64])
        activation = model_cfg.get("activation", "relu")
        batchnorm = model_cfg.get("batchnorm", True)
        dropout = model_cfg.get("dropout", 0.2)

        self.model = DigitANN(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            num_classes=10,
            activation=activation,
            batchnorm=batchnorm,
            dropout=dropout
        ).to(self.device)

        self.transform = get_ann_transform(self.img_size, channels=self.channels, is_train=False)

        # Load weights if checkpoint exists
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            # Adapt legacy 'net.' keys if present
            if any(k.startswith("net.") for k in state_dict.keys()):
                adapted_sd = {}
                for k, v in state_dict.items():
                    if k.startswith("net.12."):
                        adapted_sd[k.replace("net.12.", "classifier.")] = v
                    elif k.startswith("net."):
                        adapted_sd[k.replace("net.", "feature_extractor.")] = v
                    else:
                        adapted_sd[k] = v
                state_dict = adapted_sd
            self.model.load_state_dict(state_dict)
            print(f"[DigitANNPredictor] Loaded weights from {model_path}")
        else:
            if model_path:
                print(f"[DigitANNPredictor] Checkpoint '{model_path}' not found. Initialized with random weights.")

        self.model.eval()

    def predict_slot(self, slot_crop: Image.Image) -> Dict[str, Any]:
        """
        Classifies ONE digit slot crop (0-9).
        
        Returns:
            Dict with 'digit', 'confidence', and 'all_probs'.
        """
        img_tensor = self.transform(slot_crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img_tensor)[0]
            probs = F.softmax(logits, dim=-1)

        top_prob, top_digit = torch.max(probs, dim=-1)
        top_prob_val = float(top_prob.cpu().item())
        top_digit_val = int(top_digit.cpu().item())

        return {
            "digit": top_digit_val,
            "confidence": round(top_prob_val, 4),
            "all_probs": [round(float(p), 4) for p in probs.cpu().numpy().tolist()]
        }

    def predict_dial(self, slot_crops: List[Image.Image]) -> Dict[str, Any]:
        """
        Runs the single-digit classifier 4 times, once per slot.
        Combines results into the 4-digit dial reading.
        """
        if len(slot_crops) != 4:
            raise ValueError(f"Expected 4 slot crops for dial reading, received {len(slot_crops)}.")

        per_slot_results = []
        digits_str = ""
        confs = []

        for i, crop in enumerate(slot_crops):
            res = self.predict_slot(crop)
            per_slot_results.append(res)
            digits_str += str(res["digit"])
            confs.append(res["confidence"])

        # Flag mid-roll or ambiguous last digit if confidence on slot 4 is lower
        # or top 2 probabilities are close
        last_slot_probs = per_slot_results[3]["all_probs"]
        sorted_last = sorted(last_slot_probs, reverse=True)
        mid_roll_detected = (sorted_last[0] - sorted_last[1]) < 0.25 if len(sorted_last) > 1 else False

        avg_conf = sum(confs) / 4.0

        return {
            "reading": digits_str,
            "confidence": round(avg_conf, 4),
            "per_digit_confidence": confs,
            "is_mid_roll": mid_roll_detected,
            "slots": per_slot_results
        }
