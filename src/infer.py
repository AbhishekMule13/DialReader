"""
Stage 3: End-to-End Inference Pipeline for DialReader

Three-stage modular pipeline:
  1. Stage 1 (Full-Resolution Window Detector): Locates and extracts the display_window crop.
     Strict: if no window is detected, returns status="no_window" and produces NO slot crops.
  2. Stage 2 (WindowNormalizer): Deskews, auto-orients, and dynamically derives 4 discrete slot
     crops based on the red drum position.
  3. Stage 3 (Method 2 Decimal & Digit ANN): Measures Tenth decimal fraction and classifies
     digits 0-9 for the 4 slots.
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
from src.digits import PixelDigitRecognizer


class DialReadingPipeline:
    """
    End-to-end pipeline:
    1. Full-Resolution Display Window Detector -> crops display window, produces overlay
    2. WindowNormalizer -> dynamically extracts 4 slots and Method 2 decimal measurement
    3. Classical PixelDigitRecognizer -> recognizes digits 0-9 with template matching & geometric analysis
    """

    def __init__(
        self,
        detector_weights: Optional[Path] = None,
        classifier_weights: Optional[Path] = None,
        config_path: Optional[str] = None,
        device: str = "cpu",
        min_digit_conf: float = 0.35,
        min_detector_conf: float = 0.40,
        recognizer: Optional[PixelDigitRecognizer] = None
    ):
        self.min_digit_conf = min_digit_conf
        self.min_detector_conf = min_detector_conf
        self.device = device

        # 1. Detector (pure CV full-resolution aperture detector)
        self.detector = DialDetector(detector_weights, device=device)

        # 2. Classical CV Digit Recognizer (zero neural network inference)
        if recognizer is not None:
            self.classifier = recognizer
        else:
            self.classifier = PixelDigitRecognizer(
                min_confidence=min_digit_conf,
                auto_load_verified=True,
                project_root=PROJECT_ROOT
            )

        # 3. Normalizer & Dynamic Slot Splitter
        self.normalizer = WindowNormalizer()

    def read_image(
        self,
        image_input: Union[str, Path, Image.Image],
        manual_orientation: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end reading on a dial image.
        If no display window is found, strictly returns status="no_window" and NO slot crops.
        """
        # Load and decode image
        try:
            if isinstance(image_input, (str, Path)):
                img_p = Path(image_input)
                if not img_p.exists():
                    return {
                        "value": None,
                        "confidence": 0.0,
                        "status": "unreadable",
                        "reason": f"File does not exist: {img_p}",
                        "window_crop": None,
                        "digit_slots": [],
                        "detected_overlay": None
                    }
                pil_img = Image.open(img_p)
            elif isinstance(image_input, Image.Image):
                pil_img = image_input
            else:
                return {
                    "value": None,
                    "confidence": 0.0,
                    "status": "unreadable",
                    "reason": f"Unsupported image input type: {type(image_input)}",
                    "window_crop": None,
                    "digit_slots": [],
                    "detected_overlay": None
                }
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        except Exception as e:
            return {
                "value": None,
                "confidence": 0.0,
                "status": "unreadable",
                "reason": f"Could not decode image: {e}",
                "window_crop": None,
                "digit_slots": [],
                "detected_overlay": None
            }

        # Stage 1: Window Detection (Full Resolution Original, NO fallback)
        detections = self.detector.detect(pil_img, conf_threshold=self.min_detector_conf)

        if not detections:
            # Rule 1: No fallback! Return status="no_window" and do NOT produce slot crops.
            overlay = self.detector.draw_overlay(pil_img, None)
            return {
                "value": None,
                "confidence": 0.0,
                "status": "no_window",
                "reason": "No display window detected in photo.",
                "window_crop": None,
                "rectified_window": None,
                "oriented_window": None,
                "digit_slots": [],
                "detected_overlay": overlay,
                "detection_box": None
            }

        top_det = detections[0]
        window_crop = top_det["crop"]
        det_conf = top_det["confidence"]
        overlay = self.detector.draw_overlay(pil_img, top_det)

        # Stage 2: Normalization & Dynamic Slot Extraction (Stage 2) + Method 2 Decimal (Stage 3)
        proc = self.normalizer.process(window_crop, orientation=manual_orientation)
        slots = proc["digit_slots"]
        layout_info = proc["layout"]
        decimal_info = proc.get("decimal_info", {})
        rectified_window = proc["rectified_window"]
        oriented_window = proc["oriented_window"]
        slot_ranges = proc["slot_ranges"]
        digit_positions = proc["digit_vertical_positions"]

        if len(slots) != 4:
            return {
                "value": None,
                "confidence": 0.0,
                "status": "unreadable",
                "reason": f"Unexpected slot extraction count: {len(slots)} (expected 4)",
                "window_crop": window_crop,
                "digit_slots": [],
                "detected_overlay": overlay
            }

        # Stage 3: Classify each of the 4 slots with the classical PixelDigitRecognizer
        slot_diagnostics = []
        if self.classifier is not None:
            recog_result = self.classifier.predict_dial(slots)
            value_str = recog_result["reading"]
            per_digit_conf = recog_result["per_digit_confidence"]
            is_mid_roll = recog_result["is_mid_roll"]
            min_d_conf = min(per_digit_conf) if per_digit_conf else 0.0
            avg_d_conf = sum(per_digit_conf) / len(per_digit_conf) if per_digit_conf else 0.0
            overall_conf = float(round(avg_d_conf * det_conf, 4))
            slot_diagnostics = recog_result.get("diagnostics", [])
        else:
            value_str = "0000"
            per_digit_conf = [0.0, 0.0, 0.0, 0.0]
            is_mid_roll = False
            min_d_conf = 0.0
            overall_conf = 0.0

        # Combine whole digits with Method 2 decimal fraction
        decimal_tenth = decimal_info.get("tenth", 0)
        decimal_fraction = decimal_info.get("decimal_fraction", 0.0)
        full_value_str = f"{value_str}.{decimal_tenth}"

        status = "ok"
        reason = None
        reasons_list = []

        for s in slot_diagnostics:
            s_stat = s.get("status", "ok")
            if s_stat != "ok":
                s_idx = s.get("slot_index", 0) + 1
                s_r = s.get("reason", f"Status: {s_stat}")
                reasons_list.append(f"Slot {s_idx} ({s_stat}): {s_r}")

        if any(s.get("status") in ("uncertain", "missing_template", "no_digit") for s in slot_diagnostics):
            status = "uncertain"
        if self.classifier is not None and min_d_conf < self.min_digit_conf:
            status = "unreadable" if min_d_conf < 0.20 else "uncertain"

        if status == "ok":
            reason = "Reading recognized with high confidence"
        elif reasons_list:
            reason = " | ".join(reasons_list)
        elif self.classifier is not None and min_d_conf < self.min_digit_conf:
            reason = f"Low confidence: minimum slot confidence ({min_d_conf:.2f}) < threshold ({self.min_digit_conf:.2f})"
        else:
            reason = f"Reading status: {status}"

        return {
            "value": full_value_str,
            "whole_digits": value_str,
            "fraction": decimal_fraction,
            "tenth": decimal_tenth,
            "decimal_info": decimal_info,
            "confidence": overall_conf,
            "status": status,
            "reason": reason,
            "per_digit_confidence": per_digit_conf,
            "is_mid_roll": is_mid_roll,
            "diagnostics": slot_diagnostics,
            "slot_diagnostics": slot_diagnostics,
            "detector_confidence": float(round(det_conf, 4)),
            "dial_variant": layout_info["variant"],
            "window_crop": window_crop,
            "oriented_window": oriented_window,
            "rectified_window": rectified_window,
            "digit_slots": slots,
            "slot_ranges": slot_ranges,
            "digit_vertical_positions": digit_positions,
            "detected_overlay": overlay,
            "detection_box": top_det.get("pixel_bbox"),
            "red_drum_crop": proc.get("red_drum_crop"),
            "red_drum_range": proc.get("red_drum_range"),
            "tick_crop": proc.get("tick_crop"),
            "tick_range": proc.get("tick_range"),
            "vernier_margin_px": proc.get("vernier_margin_px", 0)
        }


def main():
    parser = argparse.ArgumentParser(description="End-to-End Dial Reading CLI with Classical CV Recognizer")
    parser.add_argument("--image", type=str, required=True, help="Path to dial image (.jpg, .png, etc.)")
    parser.add_argument("--detector_weights", type=str, default=None)
    parser.add_argument("--classifier_weights", type=str, default="models/best_digit_ann.pt")
    parser.add_argument("--config", type=str, default="configs/digit_ann.yaml")
    parser.add_argument("--min_conf", type=float, default=0.35, help="Minimum per-digit confidence")
    parser.add_argument("--orientation", type=int, default=None, help="Manual rotation angle (0/90/180/270)")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    pipeline = DialReadingPipeline(
        detector_weights=Path(args.detector_weights) if args.detector_weights else None,
        classifier_weights=Path(args.classifier_weights) if args.classifier_weights else None,
        config_path=args.config,
        device=args.device,
        min_digit_conf=args.min_conf
    )

    result = pipeline.read_image(Path(args.image), manual_orientation=args.orientation)
    def default_serializer(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    clean_result = {
        k: v for k, v in result.items()
        if not isinstance(v, Image.Image) and not (isinstance(v, list) and v and isinstance(v[0], Image.Image))
    }
    print(json.dumps(clean_result, indent=2, default=default_serializer))


if __name__ == "__main__":
    main()
