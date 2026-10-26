"""
DialReader Stage 3: Small CNN on 32x32 White-Pixel Digit Masks
Replaces the flat MLP with a spatial Convolutional Neural Network trained on
segmented binary masks of the white digit glyphs.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageOps
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def extract_white_digit_mask(slot_img: Union[Image.Image, np.ndarray], output_size: int = 32) -> np.ndarray:
    """
    Extracts, segments, and standardizes white digit pixels from a slot crop.
    
    Pipeline:
      1. Low-saturation & high-value color segmentation (isolates white numerals on black/red).
      2. Otsu adaptive thresholding.
      3. Largest connected component extraction (filters borders and specular reflections).
      4. Tight bounding-box crop.
      5. Aspect-ratio preserving square padding.
      6. Standardized 32x32 binary mask output in [0, 255].
    """
    if isinstance(slot_img, Image.Image):
        arr = np.array(slot_img.convert("RGB"))
    else:
        arr = slot_img.copy()
        if len(arr.shape) == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            
    h, w, _ = arr.shape
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    
    # White digit condition: low saturation (< 110) AND high value (> 110)
    white_cand = (sat < 110) & (val > 110)
    
    # Otsu threshold on value
    _, otsu = cv2.threshold(val, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (white_cand & (otsu > 0)).astype(np.uint8) * 255
    
    # Exclude outer edge border margins (suppress window frame reflections)
    edge_px = max(2, int(min(h, w) * 0.05))
    mask[:edge_px, :] = 0
    mask[-edge_px:, :] = 0
    mask[:, :edge_px] = 0
    mask[:, -edge_px:] = 0
    
    # Find connected components to isolate numeral
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        # Sort components by area (excluding background label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_id = 1 + int(np.argmax(areas))
        x, y, comp_w, comp_h = stats[best_id, :4]
        
        # Validate minimum area to discard noise
        if comp_w >= 4 and comp_h >= 8:
            digit_crop = mask[y:y+comp_h, x:x+comp_w]
            max_dim = max(comp_w, comp_h)
            padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
            pad_y = (max_dim - comp_h) // 2
            pad_x = (max_dim - comp_w) // 2
            padded[pad_y:pad_y+comp_h, pad_x:pad_x+comp_w] = digit_crop
            
            resized = cv2.resize(padded, (output_size, output_size), interpolation=cv2.INTER_AREA)
            binary = (resized > 80).astype(np.uint8) * 255
            return binary
            
    # Fallback to direct otsu resize if component filtering yielded nothing
    padded_full = cv2.resize(mask, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return (padded_full > 80).astype(np.uint8) * 255


class DigitMaskCNN(nn.Module):
    """
    Lightweight, high-accuracy Convolutional Neural Network for 32x32 binary digit masks.
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        
        # Block 1: 32x32 -> 16x16
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 16x16 -> 8x8
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 8x8 -> 4x4
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Fully Connected Head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 32, 32)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits


class MaskAugmenter:
    """
    Augmentation pipeline for 32x32 binary digit masks:
    shifts, small scale jitter, edge blur, and random edge glare bars.
    """
    def __init__(self, p_glare: float = 0.25, max_shift: int = 2):
        self.p_glare = p_glare
        self.max_shift = max_shift
        
    def __call__(self, mask_arr: np.ndarray) -> np.ndarray:
        m = mask_arr.copy()
        h, w = m.shape
        
        # 1. Random shifts
        dx = np.random.randint(-self.max_shift, self.max_shift + 1)
        dy = np.random.randint(-self.max_shift, self.max_shift + 1)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        m = cv2.warpAffine(m, M, (w, h), borderValue=0)
        
        # 2. Random bright bar at edge (imitates residual window bezel glare)
        if np.random.random() < self.p_glare:
            edge = np.random.choice(["top", "bottom", "left", "right"])
            thickness = np.random.randint(1, 3)
            if edge == "top": m[:thickness, :] = 255
            elif edge == "bottom": m[-thickness:, :] = 255
            elif edge == "left": m[:, :thickness] = 255
            elif edge == "right": m[:, -thickness:] = 255
            
        # 3. Random blur
        if np.random.random() < 0.2:
            m = cv2.GaussianBlur(m, (3, 3), 0.5)
            
        return m


class DigitMaskPredictor:
    """
    Inference wrapper for DigitMaskCNN with built-in mask extraction.
    """
    def __init__(self, model_path: Union[str, Path], device: str = "cpu"):
        self.device = torch.device(device)
        self.model = DigitMaskCNN(num_classes=10).to(self.device)
        
        model_path = Path(model_path)
        if model_path.exists():
            ckpt = torch.load(model_path, map_location=self.device)
            sd = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(sd)
            print(f"[DigitMaskPredictor] Loaded DigitMaskCNN from {model_path}")
        else:
            print(f"[DigitMaskPredictor] Checkpoint not found: {model_path}")
            
        self.model.eval()
        
    def predict_slot(self, slot_crop: Image.Image) -> Dict[str, Any]:
        """Segments white digit mask and classifies numeral 0-9."""
        mask = extract_white_digit_mask(slot_crop, output_size=32)
        # Normalize to [-1.0, 1.0]
        tensor = torch.tensor(mask, dtype=torch.float32) / 255.0
        tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)[0]
            probs = F.softmax(logits, dim=-1)
            
        top_prob, top_digit = torch.max(probs, dim=-1)
        return {
            "digit": int(top_digit.cpu().item()),
            "confidence": round(float(top_prob.cpu().item()), 4),
            "all_probs": [round(float(p), 4) for p in probs.cpu().numpy().tolist()],
            "mask_preview": mask
        }
        
    def predict_dial(self, slot_crops: List[Image.Image]) -> Dict[str, Any]:
        """Runs mask CNN across 4 slots."""
        if len(slot_crops) != 4:
            raise ValueError(f"Expected 4 slot crops, got {len(slot_crops)}")
            
        per_slot = []
        digits_str = ""
        confs = []
        for s in slot_crops:
            r = self.predict_slot(s)
            per_slot.append(r)
            digits_str += str(r["digit"])
            confs.append(r["confidence"])
            
        return {
            "reading": digits_str,
            "confidence": round(sum(confs) / 4.0, 4),
            "per_digit_confidence": confs,
            "is_mid_roll": False,
            "slots": per_slot
        }
