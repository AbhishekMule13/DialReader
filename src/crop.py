"""
Stage 2 & 3: Crop Normalizer & Digit Slot Extractor for DialReader
- Normalizes perspective and canonical horizontal reading orientation (left-to-right).
- Stage 2: Dynamically locates the red drum (HSV red) on the right end of the window.
  Derives the 4 slot boundaries from the window itself (3 black drums to the left + 1 red drum).
  Returns 4 discrete slot crops and the vertical centroid position of each digit.
- Stage 3: Method 2 Decimal Measurement:
  Boxes the white numeral on the red drum, measures its signed displacement from the reading
  baseline divided by the digit pitch measured dynamically in that image, rounds to nearest tenth.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps
import cv2


class WindowNormalizer:
    """
    Rectifies, rotates, dynamically extracts 4 discrete digit slots using the red drum position,
    and measures the tenth decimal fraction using Method 2.
    """

    def __init__(
        self,
        target_height: int = 128,
        target_width: int = 320
    ):
        self.target_height = target_height
        self.target_width = target_width

    def auto_orient(self, img: Image.Image, manual_orientation: Optional[int] = None) -> Image.Image:
        """
        Rotates image to canonical horizontal reading orientation (left-to-right, red drum on right).
        If manual_orientation is supplied (0, 90, 180, 270), applies it.
        Otherwise, auto-detects rotation from aspect ratio and red drum position.
        """
        if manual_orientation is not None and manual_orientation in [90, 180, 270]:
            img = img.rotate(manual_orientation, expand=True)

        w, h = img.size
        # If crop is vertical (h > w * 1.15), rotate 90 degrees clockwise to make it horizontal
        if h > w * 1.15:
            img = img.rotate(270, expand=True)
            w, h = img.size

        # Verify whether red drum is on right side
        arr = np.array(img.convert("RGB"))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        # True red wraps around 0 and 180 in HSV
        true_red = ((hsv[:,:,0] >= 170) | (hsv[:,:,0] <= 4)) & (hsv[:,:,1] > 75) & (hsv[:,:,2] > 50)
        mid_x = w // 2
        red_left = float(np.sum(true_red[:, :mid_x]))
        red_right = float(np.sum(true_red[:, mid_x:]))

        # If red drum is strongly on the left side, image is inverted 180 degrees
        if red_left > red_right * 2.0 and red_left > 100:
            img = img.rotate(180, expand=True)

        return img

    def rectify_and_resize(self, window_crop: Image.Image) -> Image.Image:
        """
        Resizes the window crop to standardized target dimensions (e.g. 128x320) for classification.
        """
        return window_crop.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)

    def detect_layout_variant(self, window_img: Image.Image) -> Dict[str, Any]:
        """
        Detects drum layout variant (standard 3 black + 1 red, or all black).
        """
        arr = np.array(window_img.convert("RGB"))
        h, w, _ = arr.shape
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        true_red = ((hsv[:,:,0] >= 170) | (hsv[:,:,0] <= 4)) & (hsv[:,:,1] > 75) & (hsv[:,:,2] > 50)
        mid_x = w // 2
        red_right = float(np.mean(true_red[:, mid_x:]))

        is_red = red_right > 0.08
        return {
            "variant": "standard_3black_1red" if is_red else "all_black",
            "is_digit4_red": is_red,
            "red_ratio_right": round(red_right, 4)
        }

    def split_into_slots(
        self,
        window_img: Image.Image
    ) -> Dict[str, Any]:
        """
        Stage 2: Derives slot boundaries from the window itself, NOT fixed fractions.
        - Locates the red drum (HSV red) on the right end of the window.
        - The three black drums are to its left.
        - Derives individual drum width W_drum = (red_x_start - black_x_start) / 3.0.
        - Splits window into 4 discrete equal-width slot crops for digit recognition.
        - Separately extracts the unclipped red drum and tick measurement region [red_x_start, W_window].
        - Computes the vertical position (cy) and bounding box of each digit numeral without tick interference.
        """
        arr = np.array(window_img.convert("RGB"))
        h, w, _ = arr.shape
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        # 1. Locate red drum on right half of window
        # True red: Hue in [170, 180] or [0, 4], Saturation > 75, Value > 50
        true_red = ((hsv[:,:,0] >= 170) | (hsv[:,:,0] <= 4)) & (hsv[:,:,1] > 75) & (hsv[:,:,2] > 50)
        red_cols = np.sum(true_red, axis=0)

        mid_x = w // 2
        # Look for where red drum starts in right half
        red_thresh = max(3, int(h * 0.15))
        red_indices = np.where(red_cols[mid_x:] >= red_thresh)[0]

        if len(red_indices) > 0:
            red_x_start = int(mid_x + red_indices[0])
        else:
            # Fallback if all-black variant: use 72% width
            red_x_start = int(round(w * 0.72))

        # 2. Locate black drums on the left
        # Black drums are dark (V < 75) or contain white digits
        dark_or_white = (hsv[:,:,2] < 75) | ((hsv[:,:,1] < 50) & (hsv[:,:,2] > 130))
        drum_cols = np.sum(dark_or_white, axis=0)
        black_thresh = max(3, int(h * 0.20))
        black_indices = np.where(drum_cols[:red_x_start] >= black_thresh)[0]

        black_x_start = int(black_indices[0]) if len(black_indices) > 0 else 0

        # Physical geometry: 3 black drums of equal width to the left of the red drum
        black_w = max(3, red_x_start - black_x_start)
        drum_w = float(black_w / 3.0)

        slot_ranges = []
        slot_crops = []

        # Slots 1, 2, 3 (Black drums) - strictly 1 drum_w wide each
        for i in range(3):
            x1 = int(round(black_x_start + i * drum_w))
            x2 = int(round(black_x_start + (i + 1) * drum_w))
            slot_ranges.append((x1, x2))
            slot_crops.append(window_img.crop((x1, 0, x2, h)))

        # Slot 4 (Red drum digit recognition slot) - strictly 1 drum_w wide, matching black drums
        x1_red = int(red_x_start)
        x2_red = int(min(w, round(red_x_start + drum_w)))
        slot_ranges.append((x1_red, x2_red))
        slot_crops.append(window_img.crop((x1_red, 0, x2_red, h)))

        # Unclipped Red Drum Region: from red_x_start to the complete right edge of the window
        # Contains numeral + graduation scale + right vernier margin
        red_drum_crop = window_img.crop((x1_red, 0, w, h))
        red_drum_range = (x1_red, int(w))

        # Tick measurement sub-region: right portion of red drum extending to window edge
        tick_x_start = int(min(w - 1, round(red_x_start + drum_w * 0.55)))
        tick_crop = window_img.crop((tick_x_start, 0, w, h))
        tick_range = (tick_x_start, int(w))
        vernier_margin_px = int(max(0, w - x2_red))

        # 3. Detect vertical centroid position of each digit
        digit_vertical_positions = []
        digit_boxes = []
        digit_heights = []

        for i, sc in enumerate(slot_crops):
            sc_np = np.array(sc.convert("RGB"))
            gray = cv2.cvtColor(sc_np, cv2.COLOR_BGR2GRAY)
            sc_w = slot_ranges[i][1] - slot_ranges[i][0]
            # Threshold white numeral on dark/red background
            white = gray > 140
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white.astype(np.uint8) * 255)

            candidates = []
            for j in range(1, num_labels):
                ba = int(stats[j, cv2.CC_STAT_AREA])
                bw = int(stats[j, cv2.CC_STAT_WIDTH])
                bh = int(stats[j, cv2.CC_STAT_HEIGHT])
                cx, cy = float(centroids[j][0]), float(centroids[j][1])

                # Filter out slot divider vertical lines at boundary
                if bw < 10 and (stats[j, cv2.CC_STAT_LEFT] <= 2 or stats[j, cv2.CC_STAT_LEFT] + bw >= sc_w - 2):
                    continue

                # In Slot 4 (red drum), ensure ticks are not mistakenly picked as the digit numeral
                if i == 3:
                    if bh <= 14 and ba < 80:
                        continue
                    if cx > sc_w * 0.65 and ba < 120:
                        continue

                if ba > 20 and bh > 10 and bw > 4:
                    box = (
                        int(slot_ranges[i][0] + stats[j, cv2.CC_STAT_LEFT]),
                        int(stats[j, cv2.CC_STAT_TOP]),
                        int(bw),
                        int(bh)
                    )
                    candidates.append((ba, cy, box, bh))

            if candidates:
                # Largest white numeral component
                best = max(candidates, key=lambda c: c[0])
                digit_vertical_positions.append(float(best[1]))
                digit_boxes.append(best[2])
                digit_heights.append(int(best[3]))
            else:
                digit_vertical_positions.append(float(h / 2.0))
                default_box = (int(slot_ranges[i][0]), int(h * 0.25), int(drum_w), int(h * 0.5))
                digit_boxes.append(default_box)
                digit_heights.append(int(h * 0.5))

        return {
            "slot_crops": slot_crops,
            "slot_ranges": slot_ranges,
            "drum_width": round(float(drum_w), 2),
            "red_x_start": int(red_x_start),
            "black_x_start": int(black_x_start),
            "digit_vertical_positions": [round(float(p), 2) for p in digit_vertical_positions],
            "digit_boxes": digit_boxes,
            "digit_heights": digit_heights,
            "red_drum_crop": red_drum_crop,
            "red_drum_range": red_drum_range,
            "tick_crop": tick_crop,
            "tick_range": tick_range,
            "vernier_margin_px": vernier_margin_px
        }

    # Public alias for backward compatibility and test conventions
    _split_slots = split_into_slots

    def measure_red_drum_fraction(
        self,
        slot_data: Dict[str, Any],
        window_height: int
    ) -> Dict[str, Any]:
        """
        Stage 3: Method 2 Decimal Measurement
        - Measures signed displacement of white digit on red drum from calibrated reading baseline.
        - Uses the MEDIAN of valid black-digit centers (Slots 1-3) to resist transitioning digits.
        - Digit pitch is measured dynamically in that image (pitch = 1.5 * avg_digit_height).
        - Analyzes unclipped tick region [red_x_start, W_window] for graduation verification.
        - tenth = round(displacement / (pitch / 10)).
        """
        cy_list = slot_data["digit_vertical_positions"]
        digit_boxes = slot_data["digit_boxes"]
        digit_heights = slot_data["digit_heights"]

        # Reading baseline: MEDIAN of valid black digits (Slots 1-3)
        # Using median ensures that a transitioning/mid-roll black digit does not skew the baseline.
        valid_black_cy = [
            float(cy) for cy in cy_list[:3]
            if cy is not None and not np.isnan(cy) and 0 < cy < window_height
        ]
        if valid_black_cy:
            baseline_y = float(np.median(valid_black_cy))
        else:
            baseline_y = float(window_height / 2.0)

        red_cy = float(cy_list[3]) if len(cy_list) >= 4 else float(window_height / 2.0)

        # Signed displacement: dial rolling upward as value increases -> baseline_y - red_cy
        disp_px = baseline_y - red_cy

        # Measure digit pitch dynamically in that image
        avg_digit_h = float(np.mean(digit_heights)) if digit_heights else float(window_height * 0.5)
        # Distance between consecutive digits on mechanical drum is ~1.5x digit height
        pitch = avg_digit_h * 1.5
        pitch_per_tenth = max(1.0, pitch / 10.0)

        raw_tenth = disp_px / pitch_per_tenth
        tenth = int(np.clip(round(raw_tenth), 0, 9))
        fraction = round(tenth * 0.1, 1)

        red_digit_box = digit_boxes[3] if len(digit_boxes) >= 4 else None

        # Analyze unclipped tick measurement region if provided
        tick_crop = slot_data.get("tick_crop")
        tick_info: Dict[str, Any] = {
            "ticks_detected": 0,
            "longest_tick_cy": None,
            "longest_tick_width": 0,
            "tick_pitch_px": None,
            "tick_offset_from_digit": None,
            "vernier_margin_px": int(slot_data.get("vernier_margin_px", 0))
        }

        if tick_crop is not None:
            try:
                tc_arr = np.array(tick_crop.convert("RGB"))
                min_rgb = np.minimum(np.minimum(tc_arr[:, :, 0], tc_arr[:, :, 1]), tc_arr[:, :, 2])
                white_ticks = (min_rgb > 120).astype(np.uint8) * 255
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_ticks)

                ticks = []
                for j in range(1, num_labels):
                    ba = int(stats[j, cv2.CC_STAT_AREA])
                    bw = int(stats[j, cv2.CC_STAT_WIDTH])
                    bh = int(stats[j, cv2.CC_STAT_HEIGHT])
                    cy = float(centroids[j][1])
                    if bw >= 4 and bh <= 14 and ba >= 6:
                        ticks.append((bw, bh, cy, ba))

                if ticks:
                    ticks.sort(key=lambda t: t[0], reverse=True)  # longest tick first
                    longest = ticks[0]
                    tick_info["ticks_detected"] = len(ticks)
                    tick_info["longest_tick_width"] = int(longest[0])
                    tick_info["longest_tick_cy"] = round(float(longest[2]), 2)
                    tick_info["tick_offset_from_digit"] = round(float(longest[2] - red_cy), 2)

                    if len(ticks) >= 3:
                        y_coords = sorted([t[2] for t in ticks])
                        diffs = np.diff(y_coords)
                        valid_diffs = [d for d in diffs if 4 <= d <= 25]
                        if valid_diffs:
                            tick_info["tick_pitch_px"] = round(float(np.median(valid_diffs)), 2)
            except Exception:
                pass

        return {
            "decimal_fraction": fraction,
            "tenth": tenth,
            "raw_tenth": round(float(raw_tenth), 2),
            "displacement_px": round(float(disp_px), 2),
            "baseline_y": round(float(baseline_y), 2),
            "digit_cy": round(float(red_cy), 2),
            "avg_digit_height": round(float(avg_digit_h), 2),
            "pitch_px": round(float(pitch), 2),
            "pitch_per_tenth": round(float(pitch_per_tenth), 2),
            "digit_box": red_digit_box,
            "digit_bbox": red_digit_box,
            "status": "ok",
            "tick_info": tick_info,
            "ticks_detected": tick_info["ticks_detected"],
            "vernier_margin_px": tick_info["vernier_margin_px"]
        }

    def process(
        self,
        window_crop: Image.Image,
        orientation: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full rectification, slot splitting, and decimal measurement pipeline.
        """
        oriented = self.auto_orient(window_crop, manual_orientation=orientation)
        w, h = oriented.size

        # Slot extraction on oriented native resolution
        slot_data = self.split_into_slots(oriented)
        layout_info = self.detect_layout_variant(oriented)
        decimal_info = self.measure_red_drum_fraction(slot_data, h)

        # Also provide standardized rectified window for UI / classifier
        rectified = self.rectify_and_resize(oriented)

        return {
            "oriented_window": oriented,
            "rectified_window": rectified,
            "layout": layout_info,
            "digit_slots": slot_data["slot_crops"],
            "slot_ranges": slot_data["slot_ranges"],
            "digit_vertical_positions": slot_data["digit_vertical_positions"],
            "digit_boxes": slot_data["digit_boxes"],
            "drum_width": slot_data["drum_width"],
            "decimal_info": decimal_info,
            "red_drum_crop": slot_data.get("red_drum_crop"),
            "red_drum_range": slot_data.get("red_drum_range"),
            "tick_crop": slot_data.get("tick_crop"),
            "tick_range": slot_data.get("tick_range"),
            "vernier_margin_px": slot_data.get("vernier_margin_px", 0)
        }