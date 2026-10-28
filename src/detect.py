"""
Stage 1: Display Window Detector for DialReader
Detects the rectangular digit display window of mechanical dial counters
on the full-resolution original image without downscaling.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import cv2


class DialDetector:
    """
    Detects the rectangular digit display window of orange mechanical dial counters.
    Operates on full-resolution original images.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        margin_pct: float = 0.04,
        max_coarse_dim: int = 1280,
        enable_coarse_to_fine: bool = True
    ):
        self.device = device
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self.margin_pct = margin_pct
        self.max_coarse_dim = max_coarse_dim
        self.enable_coarse_to_fine = enable_coarse_to_fine

        if self.model_path and self.model_path.exists():
            self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            print(f"Loading YOLO detector from {self.model_path}")
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            print(f"Note: YOLO detector could not be loaded ({e}). Using full-resolution aperture detector.")
            self.model = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _to_pil_rgb(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = ImageOps.exif_transpose(image_input).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
        return pil_img

    def detect_aperture_fullres(
        self,
        pil_img: Image.Image
    ) -> Optional[Dict[str, Any]]:
        """
        Detects display window on full-resolution original image using color, geometry, and morphology.
        Returns None if no display window is found (NO hardcoded whole-image fallback).
        """
        arr = np.array(pil_img)
        img_h, img_w, _ = arr.shape
        aspect = max(img_w, img_h) / float(max(1, min(img_w, img_h)))
        tot_area = float(img_w * img_h)

        # 1. Check if the image itself is already a cropped window
        if 1.8 < aspect < 5.0 and min(img_w, img_h) < 400:
            hsv_check = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            orange_m = (hsv_check[:,:,0] >= 4) & (hsv_check[:,:,0] <= 25) & (hsv_check[:,:,1] > 70) & (hsv_check[:,:,2] > 60)
            if float(np.mean(orange_m)) < 0.40:
                return {
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "pixel_bbox": [0, 0, img_w, img_h],
                    "confidence": 0.95,
                    "crop": pil_img,
                    "is_already_crop": True
                }

        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        # 2. Orange casing detection (Elesa / Clayton orange)
        orange = (hsv[:,:,0] >= 4) & (hsv[:,:,0] <= 25) & (hsv[:,:,1] > 60) & (hsv[:,:,2] > 40)
        num_orange = int(np.sum(orange))
        if num_orange < int(tot_area * 0.0008):
            return None

        # Size constraints on candidate display windows
        min_area = tot_area * 0.0006
        max_area = tot_area * 0.05
        max_w = img_w * 0.60
        max_h = img_h * 0.40

        candidates = []

        # 3. Method A: Interior hole detection via cv2.RETR_CCOMP
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        orange_closed = cv2.morphologyEx(orange.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        cnts, hier = cv2.findContours(orange_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hier is not None:
            for i, c in enumerate(cnts):
                if hier[0][i][3] != -1:  # Interior hole inside casing
                    x, y, w, h = cv2.boundingRect(c)
                    asp = max(w, h) / float(max(1, min(w, h)))
                    area = cv2.contourArea(c)
                    if 1.7 < asp < 4.8 and min_area < area < max_area and w <= max_w and h <= max_h:
                        hole_hsv = hsv[y:y+h, x:x+w]
                        dark_ratio = float(np.mean(hole_hsv[:,:,2] < 80))
                        if dark_ratio > 0.12:
                            score = area * (1.0 - min(0.6, abs(asp - 2.8) / 2.8))
                            candidates.append((score, (x, y, w, h), 'hole'))

        # 4. Method B: Connected dark aperture search inside prominent orange casing blobs
        if not candidates:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(orange.astype(np.uint8) * 255)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                top_indices = np.argsort(areas)[::-1][:6]
                for tid in top_indices:
                    ox, oy, ow, oh, oarea = stats[tid + 1]
                    if oarea < 500:
                        continue
                    c_hsv = hsv[oy:oy+oh, ox:ox+ow]
                    dark = (c_hsv[:,:,2] < 80).astype(np.uint8) * 255
                    d_num, d_l, d_stats, d_cent = cv2.connectedComponentsWithStats(dark)
                    if d_num > 1:
                        d_areas = d_stats[1:, cv2.CC_STAT_AREA]
                        for did in np.argsort(d_areas)[::-1][:8]:
                            bx, by, bw, bh, ba = d_stats[did + 1]
                            asp = max(bw, bh) / float(max(1, min(bw, bh)))
                            gx, gy = ox + bx, oy + by
                            if 1.7 < asp < 4.8 and min_area < ba < max_area and bw <= max_w and bh <= max_h:
                                score = ba * (1.0 - min(0.6, abs(asp - 2.8) / 2.8))
                                candidates.append((score, (gx, gy, bw, bh), 'casing_dark'))

        if not candidates:
            return None

        # Pick highest scoring candidate
        candidates.sort(key=lambda c: c[0], reverse=True)
        best_box = candidates[0][1]
        bx, by, bw, bh = best_box

        # Apply small margin (3-5%)
        mx = int(round(bw * self.margin_pct))
        my = int(round(bh * self.margin_pct))
        x1 = max(0, bx - mx)
        y1 = max(0, by - my)
        x2 = min(img_w, bx + bw + mx)
        y2 = min(img_h, by + bh + my)

        crop = pil_img.crop((x1, y1, x2, y2))
        norm_bbox = [
            round(x1 / img_w, 4),
            round(y1 / img_h, 4),
            round(x2 / img_w, 4),
            round(y2 / img_h, 4)
        ]

        return {
            "bbox": norm_bbox,
            "pixel_bbox": [x1, y1, x2, y2],
            "confidence": 0.92,
            "crop": crop,
            "is_already_crop": False
        }

    def detect(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect display windows in full-resolution image.
        Uses trained YOLO detector if available, otherwise falls back to fullres aperture detector.
        Returns empty list [] if no window is detected (NO whole-image fallback).
        """
        pil_img = self._to_pil_rgb(image_input)
        w_img, h_img = pil_img.size

        # If trained YOLO model is loaded, run YOLO first
        if self.model is not None:
            try:
                results = self.model.predict(
                    pil_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=self.device,
                    verbose=False
                )
                detections = []
                for r in results:
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        x1, y1, x2, y2 = map(int, xyxy)
                        x1 = max(0, min(w_img - 1, x1))
                        y1 = max(0, min(h_img - 1, y1))
                        x2 = max(x1 + 1, min(w_img, x2))
                        y2 = max(y1 + 1, min(h_img, y2))

                        crop = pil_img.crop((x1, y1, x2, y2))
                        norm_bbox = [
                            round(x1 / w_img, 4),
                            round(y1 / h_img, 4),
                            round(x2 / w_img, 4),
                            round(y2 / h_img, 4)
                        ]
                        detections.append({
                            "bbox": norm_bbox,
                            "pixel_bbox": [x1, y1, x2, y2],
                            "confidence": conf,
                            "crop": crop
                        })
                if detections:
                    detections.sort(key=lambda d: d["confidence"], reverse=True)
                    return detections
            except Exception as e:
                print(f"YOLO predict error ({e}), falling back to fullres detector.")

        # Classical full-resolution window detector
        from src.window_detect import WindowDetector
        win_detector = WindowDetector(
            margin_black=self.margin_pct,
            margin_tick=self.margin_pct * 1.3,
            max_coarse_dim=self.max_coarse_dim,
            enable_coarse_to_fine=self.enable_coarse_to_fine
        )
        res = win_detector.detect(pil_img)
        if res.get("status") in ["ok", "too_far"]:
            return [{
                "bbox": res["bbox"],
                "pixel_bbox": res["pixel_bbox"],
                "confidence": res["confidence"],
                "crop": res["warped_window"],
                "polygon": res.get("polygon"),
                "overlay": res.get("detected_overlay"),
                "white_digit_count": res.get("white_digit_count", 0),
                "aspect_ratio": res.get("aspect_ratio", 0.0),
                "status": res.get("status"),
                "reason": res.get("reason"),
                "is_already_crop": False,
                "was_coarse_to_fine": res.get("was_coarse_to_fine", False),
                "used_fallback": res.get("used_fallback", False)
            }]

        # No window found! Return empty list (NO fallback)
        return []

    def draw_overlay(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        detection: Optional[Dict[str, Any]] = None
    ) -> Image.Image:
        """
        Draws bounding box overlay on full-resolution original image for visual verification.
        """
        if detection is not None and detection.get("overlay") is not None:
            return detection["overlay"]

        pil_img = self._to_pil_rgb(image_input).copy()
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size

        line_width = max(3, int(round(min(w, h) * 0.005)))

        if detection is not None and "pixel_bbox" in detection:
            x1, y1, x2, y2 = detection["pixel_bbox"]
            conf = detection.get("confidence", 1.0)

            # Draw bright green bounding box
            draw.rectangle([x1, y1, x2, y2], outline="#00FF66", width=line_width)

            # Draw banner label
            label = f"Display Window ({conf:.0%})"
            text_size = max(14, int(round(min(w, h) * 0.015)))
            banner_h = text_size + 10
            banner_y1 = max(0, y1 - banner_h)
            banner_y2 = y1 if y1 > banner_h else y1 + banner_h
            draw.rectangle([x1, banner_y1, min(w, x1 + text_size * 12), banner_y2], fill="#00FF66")
            draw.text((x1 + 6, banner_y1 + 4), label, fill="#000000")
        else:
            # Draw "No Window Detected" warning watermark
            msg = "NO DISPLAY WINDOW FOUND"
            draw.rectangle([20, 20, 360, 70], fill="#D32F2F")
            draw.text((30, 35), msg, fill="#FFFFFF")

        return pil_img
