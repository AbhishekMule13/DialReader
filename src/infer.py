"""
Stage 3: End-to-End Inference Pipeline for DialReader

Two-stage modular pipeline:
  1. Stage 1 (YOLO Detector): Locates and extracts the display_window crop.
  2. Normalizer: Canonical rectification and perspective correction.
  3. Stage 2 (Digit ANN): Runs the single-digit MLP 4 times (once per slot)
     to classify digits 0-9 and detect mid-roll transitions on drum 4.

Command:
    python src/infer.py --image path/to/photo.jpg [--min_conf 0.65]

Output JSON format:
{
  "value": "0021",
  "confidence": 0.96,
  "status": "ok" | "unreadable",
  "per_digit_confidence": [0.99, 0.98, 0.95, 0.92],
  "is_mid_roll": false,
  "detector_confidence": 0.95,
  "dial_variant": "standard_3black_1red"
}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import cv2
import numpy as np
from PIL import Image, ImageOps

# Ensure dial-reader root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detect import DialDetector
from src.crop import WindowNormalizer
from src.models import DigitANNPredictor
from src.train_digits import load_config


class DialReadingPipeline:
    """
    Two-stage pipeline:
    1. YOLO Detector -> crops display_window
    2. WindowNormalizer & DigitANNPredictor -> rectifies, extracts 4 slots, classifies digits
    """

    def __init__(
        self,
        detector_weights: Optional[Path] = None,
        classifier_weights: Optional[Path] = None,
        config_path: str = "configs/digit_ann.yaml",
        device: str = "cpu",
        min_digit_conf: float = 0.65,
        min_detector_conf: float = 0.40
    ):
        self.min_digit_conf = min_digit_conf
        self.min_detector_conf = min_detector_conf
        self.device = device
        self.cfg = load_config(config_path)

        # 1. Detector
        self.detector = None
        if detector_weights and Path(detector_weights).exists():
            self.detector = DialDetector(detector_weights, device=device)

        # 2. ANN Digit Classifier (evaluated 4 times per dial)
        self.classifier = None
        weights_candidates = [
            Path(classifier_weights) if classifier_weights else None,
            PROJECT_ROOT / "models" / "best_digit_ann.pt",
            PROJECT_ROOT / "models" / "digit_ann_best.pt",
            Path("models/best_digit_ann.pt"),
            Path("models/digit_ann_best.pt")
        ]
        for w in weights_candidates:
            if w and w.exists():
                self.classifier = DigitANNPredictor(
                    model_path=w,
                    config=self.cfg,
                    device=device
                )
                break

        # 3. Normalizer & Slot Splitter
        self.normalizer = WindowNormalizer()

    def read_image(
        self,
        image_input: Union[str, Path, Image.Image],
        manual_orientation: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end reading on a dial image.
        """

        if self.classifier is None:
            return {
                "value": None,
                "confidence": 0.0,
                "status": "unreadable",
                "reason": "Digit ANN classifier weights not loaded. Please train or provide model weights."
            }

        # Load and orient image
        try:
            if isinstance(image_input, (str, Path)):
                img_p = Path(image_input)
                if not img_p.exists():
                    return {
                        "value": None,
                        "confidence": 0.0,
                        "status": "unreadable",
                        "reason": f"File does not exist: {img_p}"
                    }
                pil_img = Image.open(img_p)
            elif isinstance(image_input, Image.Image):
                pil_img = image_input
            else:
                return {
                    "value": None,
                    "confidence": 0.0,
                    "status": "unreadable",
                    "reason": f"Unsupported image input type: {type(image_input)}"
                }
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as e:
            return {
                "value": None,
                "confidence": 0.0,
                "status": "unreadable",
                "reason": f"Could not decode image: {e}"
            }

        # Stage 1: Window Detection
        window_crop = None
        det_conf = 1.0

        if self.detector is not None and self.detector.is_loaded():
            detections = self.detector.detect(pil_img, conf_threshold=self.min_detector_conf)
            if detections:
                top_det = detections[0]
                window_crop = top_det["crop"]
                det_conf = top_det["confidence"]

        # Bezel aperture detector fallback if no YOLO detection
        if window_crop is None:
            # Check if image itself is already a cropped window (aspect ratio > 1.6)
            img_w, img_h = pil_img.size
            aspect = max(img_w, img_h) / max(1, min(img_w, img_h))
            if aspect > 1.6 and min(img_w, img_h) < 400:
                window_crop = pil_img
                det_conf = 0.90
            else:
                # Color bezel contour search
                np_img = np.array(pil_img)
                hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
                orange = (hsv[:,:,0] >= 5) & (hsv[:,:,0] <= 25) & (hsv[:,:,1] > 90) & (hsv[:,:,2] > 70)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
                orange_closed = cv2.morphologyEx(orange.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
                cnts, hier = cv2.findContours(orange_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                candidates = []
                if hier is not None:
                    for i, c in enumerate(cnts):
                        x, y, w, h = cv2.boundingRect(c)
                        c_aspect = w / float(max(1, h))
                        area = cv2.contourArea(c)
                        if hier[0][i][3] != -1 and 1.6 < c_aspect < 4.5 and area > 5000:
                            candidates.append((area, (x, y, w, h)))
                if candidates:
                    best_box = max(candidates, key=lambda c: c[0])[1]
                    bx, by, bw, bh = best_box
                    window_crop = pil_img.crop((bx, by, bx+bw, by+bh))
                    det_conf = 0.88
                elif aspect > 1.3:
                    window_crop = pil_img
                    det_conf = 0.80
                else:
                    return {
                        "value": None,
                        "confidence": 0.0,
                        "status": "unreadable",
                        "reason": "No display window detected in photo."
                    }

        # Stage 2: Normalization & Slot Extraction (including Method 2 decimal measurement)
        proc = self.normalizer.process(window_crop, orientation=manual_orientation)
        slots = proc["digit_slots"]
        layout_info = proc["layout"]
        decimal_info = proc.get("decimal_info", {})
        rectified_window = proc["rectified_window"]

        if len(slots) != 4:
            return {
                "value": None,
                "confidence": 0.0,
                "status": "unreadable",
                "reason": f"Unexpected slot extraction count: {len(slots)} (expected 4)"
            }

        # Stage 3: Classify each of the 4 slots with the Digit ANN
        ann_result = self.classifier.predict_dial(slots)
        value_str = ann_result["reading"]
        per_digit_conf = ann_result["per_digit_confidence"]
        is_mid_roll = ann_result["is_mid_roll"]

        min_d_conf = min(per_digit_conf)
        avg_d_conf = sum(per_digit_conf) / 4.0
        overall_conf = round(avg_d_conf * det_conf, 4)

        # Stage 4: Combine whole digits with Method 2 decimal fraction
        decimal_tenth = decimal_info.get("tenth", 0)
        decimal_fraction = decimal_info.get("decimal_fraction", 0.0)
        full_value_str = f"{value_str}.{decimal_tenth}"

        # Check readability threshold
        if min_d_conf < self.min_digit_conf:
            return {
                "value": full_value_str,
                "whole_digits": value_str,
                "fraction": decimal_fraction,
                "tenth": decimal_tenth,
                "decimal_info": decimal_info,
                "confidence": overall_conf,
                "status": "unreadable",
                "per_digit_confidence": per_digit_conf,
                "is_mid_roll": is_mid_roll,
                "detector_confidence": round(det_conf, 4),
                "dial_variant": layout_info["variant"],
                "window_crop": window_crop,
                "rectified_window": rectified_window,
                "digit_slots": slots,
                "reason": f"Digit confidence ({min_d_conf:.3f}) fell below threshold ({self.min_digit_conf})"
            }

        return {
            "value": full_value_str,
            "whole_digits": value_str,
            "fraction": decimal_fraction,
            "tenth": decimal_tenth,
            "decimal_info": decimal_info,
            "confidence": overall_conf,
            "status": "ok",
            "per_digit_confidence": per_digit_conf,
            "is_mid_roll": is_mid_roll,
            "detector_confidence": round(det_conf, 4),
            "dial_variant": layout_info["variant"],
            "window_crop": window_crop,
            "rectified_window": rectified_window,
            "digit_slots": slots
        }


def main():
    parser = argparse.ArgumentParser(description="End-to-End Dial Reading CLI with ANN Classifier")
    parser.add_argument("--image", type=str, required=True, help="Path to dial image (.jpg, .png, etc.)")
    parser.add_argument("--detector_weights", type=str, default="models/detector_best.pt")
    parser.add_argument("--classifier_weights", type=str, default="models/digit_ann_best.pt")
    parser.add_argument("--config", type=str, default="configs/digit_ann.yaml")
    parser.add_argument("--min_conf", type=float, default=0.65, help="Minimum per-digit confidence")
    parser.add_argument("--orientation", type=int, default=None, help="Manual rotation angle (0/90/180/270)")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    pipeline = DialReadingPipeline(
        detector_weights=Path(args.detector_weights),
        classifier_weights=Path(args.classifier_weights),
        config_path=args.config,
        device=args.device,
        min_digit_conf=args.min_conf
    )

    result = pipeline.read_image(Path(args.image), manual_orientation=args.orientation)
    # Filter out non-serializable PIL Image objects for CLI JSON output
    clean_result = {
        k: v for k, v in result.items()
        if not isinstance(v, Image.Image) and not (isinstance(v, list) and v and isinstance(v[0], Image.Image))
    }
    print(json.dumps(clean_result, indent=2))


if __name__ == "__main__":
    main()
