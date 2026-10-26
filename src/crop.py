"""
Stage 2: Crop Normalizer & Digit Slot Extractor for DialReader
- Normalizes perspective and canonical horizontal orientation (0/90/180/270).
- Dynamically analyzes layout & drum colors (Rule 5: does not assume digit 4 is always red).
- Splits normalized window into 4 discrete digit slots (accounting for right-hand fractional scale).
"""

from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import cv2


class WindowNormalizer:
    """
    Rectifies, rotates, and splits the display window into 4 individual digit slots.
    """

    def __init__(
        self,
        target_height: int = 128,
        target_width: int = 320,
        fractional_scale_ratio: float = 0.12
    ):
        self.target_height = target_height
        self.target_width = target_width
        self.fractional_scale_ratio = fractional_scale_ratio

    def auto_orient(self, img: Image.Image, manual_orientation: Optional[int] = None) -> Image.Image:
        """
        Rotates image to canonical horizontal reading orientation (left-to-right).
        If manual_orientation is supplied (0, 90, 180, 270), applies it.
        Otherwise, if height > width * 1.15, auto-rotates by 90 degrees.
        """
        if manual_orientation is not None and manual_orientation in [90, 180, 270]:
            return img.rotate(manual_orientation, expand=True)

        w, h = img.size
        # If crop is vertical (common in knife+pressing), rotate to horizontal
        if h > w * 1.15:
            return img.rotate(90, expand=True)
        return img

    def rectify_and_resize(self, window_crop: Image.Image) -> Image.Image:
        """
        Resizes the window crop to standardized target dimensions (e.g. 128x320).
        """
        return window_crop.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)

    def detect_layout_variant(self, rectified_img: Image.Image) -> Dict[str, Any]:
        """
        Analyzes drum colors across slots (Rule 5: Detects layout per window,
        checks if digit 4 is red or black, or flags unknown variants).
        """
        np_img = np.array(rectified_img.convert("RGB"))
        h, w, _ = np_img.shape
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)

        # Region for digit 4 (approx 66% to 88% width)
        d4_x1 = int(w * 0.66)
        d4_x2 = int(w * (1.0 - self.fractional_scale_ratio))
        d4_crop_hsv = hsv[:, d4_x1:d4_x2]

        # Region for digit 1-3 (left half)
        d123_crop_hsv = hsv[:, :int(w * 0.60)]

        # Red mask in HSV: (H < 12 or H > 168) and S > 70 and V > 50
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([168, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask_red_d4 = cv2.inRange(d4_crop_hsv, lower_red1, upper_red1) | cv2.inRange(d4_crop_hsv, lower_red2, upper_red2)
        mask_red_d123 = cv2.inRange(d123_crop_hsv, lower_red1, upper_red1) | cv2.inRange(d123_crop_hsv, lower_red2, upper_red2)

        red_ratio_d4 = float(np.mean(mask_red_d4 > 0))
        red_ratio_d123 = float(np.mean(mask_red_d123 > 0))

        if red_ratio_d4 > 0.15 and red_ratio_d123 < 0.10:
            variant = "standard_3black_1red"
            is_digit4_red = True
        elif red_ratio_d4 < 0.08 and red_ratio_d123 < 0.08:
            variant = "all_black"
            is_digit4_red = False
        else:
            variant = "unknown_variant"
            is_digit4_red = (red_ratio_d4 > 0.10)

        return {
            "variant": variant,
            "is_digit4_red": is_digit4_red,
            "d4_red_ratio": round(red_ratio_d4, 3),
            "d123_red_ratio": round(red_ratio_d123, 3)
        }

    def split_into_slots(
        self,
        rectified_img: Image.Image
    ) -> List[Image.Image]:
        """
        Splits rectified window image into 4 digit slot crops.
        Reserves the rightmost margin (fractional_scale_ratio) for the vernier scale.

        Returns:
            List of 4 PIL Images (slot 1, slot 2, slot 3, slot 4)
        """
        w, h = rectified_img.size
        # Usable width for 4 digits
        usable_w = w * (1.0 - self.fractional_scale_ratio)
        slot_w = usable_w / 4.0

        slots = []
        for i in range(4):
            x1 = int(round(i * slot_w))
            x2 = int(round((i + 1) * slot_w))
            # Slight padding margin if needed
            slot_crop = rectified_img.crop((x1, 0, x2, h))
            slots.append(slot_crop)

        return slots

    def measure_red_drum_fraction(self, rectified_img: Image.Image) -> Dict[str, Any]:
        """
        Method 2: Continuous Displacement of the Digit Square ("Square Around the Number")
        Measures the vertical displacement of the white digit on the red drum (Slot 4)
        relative to the reading baseline (computed from Slots 1-3 or window centerline).
        """
        np_img = np.array(rectified_img.convert("RGB"))
        h, w, _ = np_img.shape
        default_baseline_y = h / 2.0  # 64.0 px in 128h crop

        # 1. Determine calibrated reading baseline from Slots 1-3 (black drums)
        usable_w = w * (1.0 - self.fractional_scale_ratio)
        black_region = np_img[:, :int(usable_w * 0.75)]
        gray_b = cv2.cvtColor(black_region, cv2.COLOR_RGB2GRAY)
        white_b_pts = np.where(gray_b > 150)[0]
        if len(white_b_pts) > 200:
            baseline_y = float(np.mean(white_b_pts))
        else:
            baseline_y = default_baseline_y

        # 2. Extract Drum 4 (the red drum)
        d4_x1 = int(usable_w * 0.75)
        d4_x2 = int(usable_w)
        d4_crop = np_img[:, d4_x1:d4_x2]
        gray_d4 = cv2.cvtColor(d4_crop, cv2.COLOR_RGB2GRAY)

        # Threshold white digit on red background
        _, thresh = cv2.threshold(gray_d4, 135, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            cx, cy = centroids[i]

            # Filter out dust, border glare, and sub-pixel noise
            if area > 100 and comp_h > 18 and comp_w > 10:
                dist = abs(cy - baseline_y)
                candidates.append({
                    "cy": cy,
                    "dist": dist,
                    "bbox": (d4_x1 + int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]), int(comp_w), int(comp_h)),
                    "area": int(area)
                })

        if not candidates:
            return {
                "decimal_fraction": 0.0,
                "tenth": 0,
                "raw_tenth": 0.0,
                "displacement_px": 0.0,
                "digit_cy": baseline_y,
                "baseline_y": round(baseline_y, 1),
                "status": "no_contour_found"
            }

        # Select primary digit closest to reading baseline
        primary = min(candidates, key=lambda c: c["dist"])
        digit_cy = primary["cy"]

        # When the dial advances (e.g. from 0 to 1), the drum rolls UPWARD (y decreases)
        # Displacement upward = baseline_y - digit_cy
        shift_up = baseline_y - digit_cy

        # In a standardized 128px height window:
        # Distance between consecutive digits is ~64px -> 6.4 to 8.0 px per 0.1 unit
        pitch_per_tenth = 8.0
        raw_tenth = shift_up / pitch_per_tenth

        # Clamp tenth between 0 and 9
        tenth = int(np.clip(round(raw_tenth), 0, 9))
        fraction = round(tenth * 0.1, 1)

        return {
            "decimal_fraction": fraction,
            "tenth": tenth,
            "raw_tenth": round(float(raw_tenth), 2),
            "displacement_px": round(float(shift_up), 1),
            "digit_cy": round(float(digit_cy), 1),
            "baseline_y": round(float(baseline_y), 1),
            "digit_bbox": primary["bbox"],
            "status": "ok"
        }

    def process(
        self,
        window_crop: Image.Image,
        orientation: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full rectification pipeline:
        orient -> resize -> detect variant -> split 4 slots -> measure decimal fraction (Method 2).
        """
        oriented = self.auto_orient(window_crop, manual_orientation=orientation)
        rectified = self.rectify_and_resize(oriented)
        layout_info = self.detect_layout_variant(rectified)
        slots = self.split_into_slots(rectified)
        decimal_info = self.measure_red_drum_fraction(rectified)

        return {
            "oriented_window": oriented,
            "rectified_window": rectified,
            "layout": layout_info,
            "digit_slots": slots,
            "decimal_info": decimal_info
        }