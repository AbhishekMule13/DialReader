"""
Stage 2: Digit Classifier Model Architecture for DialReader
Uses MobileNetV3-Small (pretrained) adapted for rolling-drum digit classification (0-9)
with explicit mid-roll transition detection and extensibility for continuous drum regression.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


def get_default_transform(img_size: Tuple[int, int] = (128, 64)):
    """
    Standard evaluation / inference transform.
    Input size: (height, width) = (128, 64)
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class DigitClassifier(nn.Module):
    """
    MobileNetV3-small backbone with:
    1. Primary classification head: 10 classes (digits 0-9)
    2. Mid-roll transition head: binary probability of being between two digits
    3. Extensible continuous offset head: predicts fractional vertical shift in [-0.5, +0.5]
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Retain feature extractor up to avgpool
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # MobileNetV3-small features output 576 channels
        in_features = 576

        # Digit class head (0-9)
        self.digit_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10)
        )

        # Mid-roll transition confidence head
        self.midroll_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Hardswish(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1)
        )

        # Continuous fractional offset head (for future fine-grained regression)
        self.offset_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Hardswish(),
            nn.Linear(64, 1),
            nn.Tanh()  # bounds output to [-1.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)

        digit_logits = self.digit_head(feat)
        midroll_logit = self.midroll_head(feat).squeeze(-1)
        offset = self.offset_head(feat).squeeze(-1) * 0.5  # scale to [-0.5, 0.5]

        return {
            "logits": digit_logits,
            "midroll_prob": torch.sigmoid(midroll_logit),
            "offset": offset
        }


class DigitPredictor:
    """
    Wrapper for model loading and single / batch inference.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        img_size: Tuple[int, int] = (128, 64)
    ):
        self.device = torch.device(device)
        self.transform = get_default_transform(img_size)
        self.model = DigitClassifier(pretrained=False).to(self.device)

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            print(f"Loaded DigitClassifier from {model_path}")
        self.model.eval()

    def predict_slot(
        self,
        slot_img: Image.Image,
        is_last_drum: bool = False,
        midroll_margin_thresh: float = 0.28
    ) -> Dict[str, Any]:
        """
        Classifies a single digit slot crop.
        Handles mid-roll detection via dual probability analysis and transition head.
        """
        tensor = self.transform(slot_img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            logits = outputs["logits"][0]
            probs = F.softmax(logits, dim=-1)
            midroll_score = float(outputs["midroll_prob"][0].cpu())
            offset_val = float(outputs["offset"][0].cpu())

        top_probs, top_indices = torch.topk(probs, k=2)
        top1_digit = int(top_indices[0].cpu())
        top1_conf = float(top_probs[0].cpu())
        top2_digit = int(top_indices[1].cpu())
        top2_conf = float(top_probs[1].cpu())

        # Check for mid-roll condition:
        # 1. Is last drum AND (midroll head triggers OR top1/top2 are close and adjacent)
        is_adjacent = (abs(top1_digit - top2_digit) == 1) or (set([top1_digit, top2_digit]) == {0, 9})
        prob_margin = top1_conf - top2_conf
        
        is_mid_roll = False
        if is_last_drum:
            if (prob_margin < midroll_margin_thresh and is_adjacent) or (midroll_score > 0.55):
                is_mid_roll = True

        return {
            "digit": top1_digit,
            "confidence": round(top1_conf, 4),
            "is_mid_roll": is_mid_roll,
            "second_digit": top2_digit if is_mid_roll else None,
            "second_confidence": round(top2_conf, 4) if is_mid_roll else None,
            "midroll_score": round(midroll_score, 4),
            "offset": round(offset_val, 4),
            "all_probs": [round(float(p), 4) for p in probs.cpu().numpy()]
        }
