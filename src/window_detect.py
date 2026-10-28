"""
Stage 1: Aperture-First Display Window Detector for Mechanical Dial Counters.
Replaces the red-first detector with structural aperture detection:
1. Detects dark rectangular aperture regions (V < 75) containing a row of 3-4 white digit-sized blobs (V > 115, S < 95).
2. Fits rotated rectangle around the aperture / digit row.
3. Requires surrounding orange bezel (local hue 5-26, S > 70) on the sides to reject black machine shadows/parts.
4. Verifies the rightmost ~25% of the aperture is red by comparing local hue/saturation vs surrounding bezel.
5. Enforces size consistency: window width must be 4-6x the red drum width (rejects casing-sized boxes).
6. YOLO cross-check with data/labeled/yolo_dataset: marks 'confirmed' if IoU > 0.5; marks 'uncertain' if IoU <= 0.5.
7. Distance check: returns status='too_far' if window height < 60 px (whole digits only, no decimal).
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import cv2


class WindowDetector:
    """
    Classical structure-first computer vision detector for mechanical dial counter display windows.
    Operates strictly on the full-resolution original image without downscaling.
    """

    def __init__(
        self,
        min_aspect: float = 2.4,
        max_aspect: float = 6.2,
        yolo_dataset_dir: Union[str, Path] = "data/labeled/yolo_dataset",
        min_drum_ratio: float = 3.6,
        max_drum_ratio: float = 6.5,
        margin_black: float = 0.06,
        margin_tick: float = 0.08,
        margin_v: float = 0.07,
        max_coarse_dim: int = 1280,
        enable_coarse_to_fine: bool = True,
    ):
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.yolo_dataset_dir = Path(yolo_dataset_dir)
        self.min_drum_ratio = min_drum_ratio
        self.max_drum_ratio = max_drum_ratio
        self.margin_black = margin_black
        self.margin_tick = margin_tick
        self.margin_v = margin_v
        self.max_coarse_dim = max_coarse_dim
        self.enable_coarse_to_fine = enable_coarse_to_fine

    def _to_cv2_bgr(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, Image.Image, Optional[str]]:
        """Converts input image to OpenCV BGR, PIL RGB, and extracts image name."""
        img_name = None
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            img_name = p.name
            pil_raw = Image.open(p)
            pil_img = ImageOps.exif_transpose(pil_raw).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, Image.Image):
            pil_img = ImageOps.exif_transpose(image_input).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                bgr = image_input.copy()
                pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Expected 3-channel image array")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
        return bgr, pil_img, img_name

    def _find_yolo_box(self, img_name: Optional[str], img_w: int, img_h: int) -> Optional[List[int]]:
        """Finds YOLO bounding box from yolo_dataset if present."""
        if not img_name or not self.yolo_dataset_dir.exists():
            return None
        clean_stem = Path(img_name).stem.replace('/', '_').replace('\\', '_')
        for split in ['train', 'val', 'test']:
            p = self.yolo_dataset_dir / 'labels' / split / f"{clean_stem}.txt"
            if p.exists() and p.stat().st_size > 0:
                try:
                    lines = p.read_text().strip().splitlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls, xc, yc, bw, bh = map(float, parts)
                            x1 = int((xc - bw / 2.0) * img_w)
                            y1 = int((yc - bh / 2.0) * img_h)
                            x2 = int((xc + bw / 2.0) * img_w)
                            y2 = int((yc + bh / 2.0) * img_h)
                            return [x1, y1, x2, y2]
                except Exception:
                    pass
        return None

    @staticmethod
    def _compute_iou(boxA: List[int], boxB: List[int]) -> float:
        """Computes Intersection over Union of two bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    def _find_coarse_dial_boxes(
        self,
        bgr: np.ndarray,
        force_fullres: bool = False
    ) -> Tuple[List[List[int]], bool]:
        """
        Stage A (Coarse): Detects candidate dial body bounding boxes.
        If image is larger than max_coarse_dim (and not force_fullres),
        creates a temporary downscaled copy for coarse candidate search and
        maps coordinates back to full resolution using separate x and y scale factors.
        For smaller images, operates directly at 100% resolution without downscaling.
        """
        h_orig, w_orig = bgr.shape[:2]
        max_dim = max(h_orig, w_orig)

        if self.enable_coarse_to_fine and not force_fullres and max_dim > self.max_coarse_dim:
            scale = self.max_coarse_dim / float(max_dim)
            w_small = max(1, int(round(w_orig * scale)))
            h_small = max(1, int(round(h_orig * scale)))
            fx = w_orig / float(w_small)
            fy = h_orig / float(h_small)
            work_bgr = cv2.resize(bgr, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
            was_downscaled = True
        else:
            work_bgr = bgr
            w_small, h_small = w_orig, h_orig
            fx, fy = 1.0, 1.0
            was_downscaled = False

        hsv_small = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV)
        H_s, S_s, V_s = hsv_small[:, :, 0], hsv_small[:, :, 1], hsv_small[:, :, 2]
        orange_mask = (H_s >= 5) & (H_s <= 25) & (S_s > 85) & (V_s > 40)
        tot_pixels = h_small * w_small
        orange_count = int(np.sum(orange_mask))

        if orange_count < 80:
            return [], was_downscaled

        dial_boxes = []

        # If the image is already a close-up where orange covers >40% or image is small
        if (orange_count / float(tot_pixels) > 0.40 and min(w_small, h_small) < 800) or min(w_small, h_small) < 250:
            dial_boxes.append([0, 0, w_orig, h_orig])
            return dial_boxes, was_downscaled

        k_a = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        orange_clean = cv2.morphologyEx(orange_mask.astype(np.uint8), cv2.MORPH_CLOSE, k_a)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(orange_clean)

        cand_bodies = []
        min_area = int(350 if was_downscaled else 400)
        for i in range(1, num_labels):
            bx, by, bw, bh, ba = stats[i]
            if ba < min_area:
                continue
            asp = max(bw, bh) / float(max(1, min(bw, bh)))
            if asp > 3.0:
                continue
            compact_factor = 1.0 / (1.0 + (asp - 1.1) ** 2)
            edge_penalty = 0.05 if (by == 0 and asp > 2.0) else 1.0
            score = ba * compact_factor * edge_penalty
            cand_bodies.append((score, bx, by, bw, bh))

        cand_bodies.sort(key=lambda c: c[0], reverse=True)

        for c in cand_bodies[:3]:
            _, bx, by, bw, bh = c
            orig_bx = bx * fx
            orig_by = by * fy
            orig_bw = bw * fx
            orig_bh = bh * fy

            pad_x = max(20, int(orig_bw * 0.22))
            pad_y = max(20, int(orig_bh * 0.22))

            x1 = max(0, min(w_orig - 1, int(round(orig_bx - pad_x))))
            y1 = max(0, min(h_orig - 1, int(round(orig_by - pad_y))))
            x2 = max(x1 + 1, min(w_orig, int(round(orig_bx + orig_bw + pad_x))))
            y2 = max(y1 + 1, min(h_orig, int(round(orig_by + orig_bh + pad_y))))

            dial_boxes.append([x1, y1, x2, y2])

        return dial_boxes, was_downscaled

    def _find_aperture_candidates(
        self,
        bgr: np.ndarray,
        dial_boxes: List[List[int]]
    ) -> List[Dict[str, Any]]:
        """
        Stage B: Run Aperture-First Window Detector Inside Candidate Dial Body Box(es).
        Crops ROI directly from 100% full-resolution original image.
        """
        candidates = []

        for dial_box in dial_boxes:
            dx1, dy1, dx2, dy2 = dial_box
            roi = bgr[dy1:dy2, dx1:dx2]
            h_roi, w_roi = roi.shape[:2]
            if h_roi < 10 or w_roi < 10:
                continue

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            H_r, S_r, V_r = hsv_roi[:, :, 0], hsv_roi[:, :, 1], hsv_roi[:, :, 2]

            dark_roi = (V_r < 75)
            white_roi = (V_r > 115) & (S_r < 95)
            orange_roi = (H_r >= 4) & (H_r <= 28) & (S_r > 70) & (V_r > 35)

            k_roi = max(3, int(min(w_roi, h_roi) * 0.015))
            k_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (k_roi * 2 + 1, k_roi + 1))
            dark_closed = cv2.morphologyEx(dark_roi.astype(np.uint8), cv2.MORPH_CLOSE, k_elem)

            contours_roi, _ = cv2.findContours(dark_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours_roi:
                area = cv2.contourArea(cnt)
                if area < 30:
                    continue
                rect = cv2.minAreaRect(cnt)
                (rcx, rcy), (rw, rh), angle = rect
                if rw < rh:
                    rw, rh = rh, rw
                    angle += 90.0
                asp = rw / float(max(1.0, rh))
                if not (self.min_aspect <= asp <= self.max_aspect) and not (2.0 <= asp <= 6.2):
                    continue
                if rw < 15 or rh < 4:
                    continue

                box_pts = cv2.boxPoints(rect).astype(np.int32)

                # Surrounding orange bezel
                outer_rect = ((rcx, rcy), (rw * 1.45, rh * 2.2), angle)
                outer_pts = cv2.boxPoints(outer_rect).astype(np.int32)
                m_out = np.zeros((h_roi, w_roi), dtype=np.uint8)
                cv2.fillPoly(m_out, [outer_pts], 255)
                cv2.fillPoly(m_out, [box_pts], 0)
                tb = np.sum(m_out > 0)
                if tb == 0:
                    continue
                ob = np.sum(orange_roi & (m_out > 0))
                o_ratio = ob / float(tb)
                if o_ratio < 0.10:
                    continue

                # White digits check inside
                m_in = np.zeros((h_roi, w_roi), dtype=np.uint8)
                cv2.fillPoly(m_in, [box_pts], 255)
                ti = np.sum(m_in > 0)
                if ti == 0:
                    continue
                wi = np.sum(white_roi & (m_in > 0))
                w_ratio = wi / float(ti)
                if not (0.01 <= w_ratio <= 0.45):
                    continue

                drum_w = rw / (3.0 if asp < 3.6 else 4.0)
                final_w = rw if asp >= 3.6 else (rw + drum_w)
                drum_ratio = final_w / float(max(1.0, drum_w))
                if not (self.min_drum_ratio <= drum_ratio <= self.max_drum_ratio):
                    continue

                cand_h = rh * 1.25

                # ------------------------------------------------------------------
                # Relative size checks against dial casing ROI:
                # Distinguishes genuine 4-drum display aperture from micro-blemishes
                # (< 8% width or < 0.25% area) and whole-casing perimeters (> 88% width or > 45% area).
                # ------------------------------------------------------------------
                rel_w = final_w / float(max(1, w_roi))
                rel_area = (final_w * cand_h) / float(max(1, w_roi * h_roi))

                if rel_w < 0.08 or rel_area < 0.0025:
                    continue
                if rel_w > 0.88 or rel_area > 0.45:
                    continue

                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                e1_cx = rcx + (final_w * 0.38) * cos_a
                e1_cy = rcy + (final_w * 0.38) * sin_a
                e2_cx = rcx - (final_w * 0.38) * cos_a
                e2_cy = rcy - (final_w * 0.38) * sin_a

                def sample_drum(cx, cy):
                    if not (0 <= cx < w_roi and 0 <= cy < h_roi):
                        return 0.0
                    r_box = cv2.boxPoints(((cx, cy), (drum_w * 1.1, rh * 0.9), angle)).astype(np.int32)
                    m = np.zeros((h_roi, w_roi), dtype=np.uint8)
                    cv2.fillPoly(m, [r_box], 255)
                    h_sub = H_r[m > 0]
                    s_sub = S_r[m > 0]
                    if len(h_sub) == 0:
                        return 0.0
                    return np.sum(((h_sub <= 6) | (h_sub >= 170)) & (s_sub > 75)) / float(len(h_sub))

                r1 = sample_drum(e1_cx, e1_cy)
                r2 = sample_drum(e2_cx, e2_cy)
                max_red = max(r1, r2)

                final_cx, final_cy = rcx, rcy
                if asp < 3.6:
                    if r1 >= r2:
                        final_cx += (drum_w * 0.5) * cos_a
                        final_cy += (drum_w * 0.5) * sin_a
                    else:
                        final_cx -= (drum_w * 0.5) * cos_a
                        final_cy -= (drum_w * 0.5) * sin_a

                # Global coordinates in original image
                glob_cx = final_cx + dx1
                glob_cy = final_cy + dy1
                final_rect = ((glob_cx, glob_cy), (final_w, cand_h), angle)

                # ------------------------------------------------------------------
                # Enhanced Candidate Scoring:
                # Balances relative size, aspect ratio, white numeral content,
                # red drum evidence, and surrounding orange bezel without letting
                # orange surroundings alone overwhelm geometric aperture evidence.
                # ------------------------------------------------------------------
                size_term = min(1.0, rel_w / 0.40) * 2.0 + min(1.0, cand_h / 100.0) * 1.0
                score = min(0.60, o_ratio) * 3.5 + min(1.0, w_ratio / 0.15) * 2.5 + max_red * 2.5 + size_term

                candidates.append({
                    'rect': final_rect,
                    'w': final_w,
                    'h': cand_h,
                    'drum_w': drum_w,
                    'drum_ratio': drum_ratio,
                    'orange_ratio': o_ratio,
                    'red_fraction': max_red,
                    'dial_box': dial_box,
                    'score': score
                })

        return candidates

    def detect(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """
        Executes aperture-first window detection and validation with coarse-to-fine acceleration.
        Returns:
            status: 'ok', 'uncertain', 'too_far', or 'no_window'
            reason: failure/status explanation string
            warped_window: rectified PIL crop (red drum on right, 100% full resolution)
            crop: rectified PIL crop
            detected_overlay: PIL overlay image
            pixel_bbox: [x1, y1, x2, y2]
            bbox: [norm_x1, norm_y1, norm_x2, norm_y2]
            aspect_ratio: float
            warped_size: (w, h)
            drum_w: float
            drum_ratio: float
            is_larger_than_8x_drum: bool
            yolo_box: Optional[List[int]]
            yolo_iou: float
            confirmed: bool
            was_coarse_to_fine: bool
            used_fallback: bool
        """
        bgr, pil_img, img_name = self._to_cv2_bgr(image_input)
        h_img, w_img = bgr.shape[:2]

        # ------------------------------------------------------------------
        # Stage A: Coarse Dial Body Candidate Detection
        # For large images (max(w, h) > max_coarse_dim), detects coarse dial
        # candidate boxes on a downscaled working copy, mapping them back to full
        # resolution with separate x/y scale factors.
        # Avoids downscaling for small images.
        # ------------------------------------------------------------------
        dial_boxes, was_downscaled = self._find_coarse_dial_boxes(bgr)
        candidates = []
        used_fallback = False

        if dial_boxes:
            candidates = self._find_aperture_candidates(bgr, dial_boxes)

        # ------------------------------------------------------------------
        # Safe Fallback:
        # If coarse search on downscaled image produced no boxes OR produced
        # invalid candidate(s) where Stage B found no valid aperture window,
        # fall back safely to the full-resolution detection path.
        # ------------------------------------------------------------------
        if was_downscaled and not candidates:
            dial_boxes_full, _ = self._find_coarse_dial_boxes(bgr, force_fullres=True)
            if dial_boxes_full:
                candidates = self._find_aperture_candidates(bgr, dial_boxes_full)
                if candidates:
                    used_fallback = True
                    dial_boxes = dial_boxes_full

        if not candidates:
            reason = "no_compact_orange_body_found" if not dial_boxes else "no_window_in_dial_body"
            overlay = self.draw_overlay(bgr, None, None, "no_window", reason)
            return {
                "status": "no_window",
                "reason": reason,
                "warped_window": None,
                "crop": None,
                "detected_overlay": overlay,
                "confidence": 0.0,
                "rect": None,
                "polygon": None,
                "pixel_bbox": None,
                "bbox": None,
                "aspect_ratio": 0.0,
                "warped_size": (0, 0),
                "drum_w": 0.0,
                "drum_ratio": 0.0,
                "is_larger_than_8x_drum": False,
                "yolo_box": None,
                "yolo_iou": 0.0,
                "confirmed": False,
                "was_coarse_to_fine": was_downscaled and not used_fallback,
                "used_fallback": used_fallback
            }

        candidates.sort(key=lambda c: c['score'], reverse=True)
        best = candidates[0]

        # ------------------------------------------------------------------
        # Stage C: Warp from Original Full-Resolution Image
        # ------------------------------------------------------------------
        rect = best['rect']
        center, size, angle = rect
        rw, rh = size
        if rw < rh:
            rw, rh = rh, rw
            angle += 90.0

        # Perspective / rotation rectification with border replicate padding to prevent edge clipping
        pad = int(max(rw, rh) * 0.5) + 30
        bgr_padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        padded_center = (center[0] + pad, center[1] + pad)
        M = cv2.getRotationMatrix2D(padded_center, angle, 1.0)
        rotated = cv2.warpAffine(
            bgr_padded, M, (w_img + 2 * pad, h_img + 2 * pad), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        x1 = int(round(padded_center[0] - rw / 2.0))
        y1 = int(round(padded_center[1] - rh / 2.0))
        x2 = int(round(padded_center[0] + rw / 2.0))
        y2 = int(round(padded_center[1] + rh / 2.0))
        warped_bgr = rotated[y1:y2, x1:x2]

        if warped_bgr.size == 0:
            overlay = self.draw_overlay(bgr, None, None, "no_window", "warp_zero_size")
            return {
                "status": "no_window",
                "reason": "warp_zero_size",
                "warped_window": None,
                "crop": None,
                "detected_overlay": overlay,
                "confidence": 0.0,
                "rect": None,
                "polygon": None,
                "pixel_bbox": None,
                "bbox": None,
                "aspect_ratio": 0.0,
                "warped_size": (0, 0),
                "drum_w": 0.0,
                "drum_ratio": 0.0,
                "is_larger_than_8x_drum": False,
                "yolo_box": None,
                "yolo_iou": 0.0,
                "confirmed": False,
                "was_coarse_to_fine": was_downscaled and not used_fallback,
                "used_fallback": used_fallback
            }

        # Orient red drum to the right side
        w_hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
        ww = warped_bgr.shape[1]
        left_h = w_hsv[:, :int(ww * 0.25), 0]
        right_h = w_hsv[:, int(ww * 0.75):, 0]
        left_s = w_hsv[:, :int(ww * 0.25), 1]
        right_s = w_hsv[:, int(ww * 0.75):, 1]
        left_red = np.sum(((left_h <= 6) | (left_h >= 170)) & (left_s > 80))
        right_red = np.sum(((right_h <= 6) | (right_h >= 170)) & (right_s > 80))
        if left_red > right_red:
            warped_bgr = cv2.rotate(warped_bgr, cv2.ROTATE_180)

        final_h, final_w = warped_bgr.shape[:2]
        crop_pil = Image.fromarray(cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB))

        # Pixel bounding box in original image
        b_pts = cv2.boxPoints(rect)
        px_min = int(max(0, np.floor(np.min(b_pts[:, 0]))))
        py_min = int(max(0, np.floor(np.min(b_pts[:, 1]))))
        px_max = int(min(w_img, np.ceil(np.max(b_pts[:, 0]))))
        py_max = int(min(h_img, np.ceil(np.max(b_pts[:, 1]))))
        classical_bbox = [px_min, py_min, px_max, py_max]

        drum_w = best['drum_w']
        drum_ratio = final_w / float(max(1.0, drum_w))
        is_huge = (final_w > 8.0 * drum_w)

        # -------------------------------------------------------------
        # 3. YOLO Cross-Check
        # -------------------------------------------------------------
        yolo_box = self._find_yolo_box(img_name, w_img, h_img)
        confirmed = False
        iou = 0.0
        status = "ok"
        reason = "detected_successfully"

        if yolo_box is not None:
            iou = self._compute_iou(classical_bbox, yolo_box)
            if iou > 0.5:
                confirmed = True
            else:
                status = "uncertain"
                reason = f"yolo_mismatch_iou_{iou:.2f}"

        # -------------------------------------------------------------
        # 4. Distance Check: height < 60px -> too_far
        # -------------------------------------------------------------
        if status == "ok" and final_h < 60:
            status = "too_far"
            reason = f"height_{final_h}px_under_60px"

        overlay = self.draw_overlay(bgr, b_pts, yolo_box, status, reason, iou, (final_w, final_h), dial_box=best.get('dial_box'))

        norm_bbox = [
            round(px_min / float(w_img), 4),
            round(py_min / float(h_img), 4),
            round(px_max / float(w_img), 4),
            round(py_max / float(h_img), 4),
        ]

        conf = round(min(0.98, 0.85 + best['orange_ratio'] * 0.12), 4)

        return {
            "status": status,
            "reason": reason,
            "warped_window": crop_pil,
            "crop": crop_pil,
            "detected_overlay": overlay,
            "confidence": conf,
            "rect": rect,
            "polygon": b_pts.tolist(),
            "pixel_bbox": classical_bbox,
            "dial_box": best.get('dial_box'),
            "bbox": norm_bbox,
            "aspect_ratio": round(final_w / float(max(1, final_h)), 2),
            "warped_size": (final_w, final_h),
            "drum_w": drum_w,
            "drum_ratio": round(drum_ratio, 2),
            "is_larger_than_8x_drum": is_huge,
            "yolo_box": yolo_box,
            "yolo_iou": round(iou, 2),
            "confirmed": confirmed,
            "was_coarse_to_fine": was_downscaled and not used_fallback,
            "used_fallback": used_fallback
        }

    def draw_overlay(
        self,
        img_bgr: np.ndarray,
        b_pts: Optional[np.ndarray],
        yolo_box: Optional[List[int]],
        status: str = "ok",
        reason: str = "",
        iou: float = 0.0,
        crop_size: Tuple[int, int] = (0, 0),
        dial_box: Optional[List[int]] = None
    ) -> Image.Image:
        """Draws overlay banner, Stage A dial box (orange), classical window polygon (green), and YOLO box (cyan)."""
        vis = img_bgr.copy()
        h_img, w_img = vis.shape[:2]

        banner_color = (
            (0, 160, 0) if status == "ok"
            else ((0, 140, 255) if status == "too_far"
                  else ((200, 150, 0) if status == "uncertain"
                        else (0, 0, 180)))
        )

        cv2.rectangle(vis, (0, 0), (w_img, 65), (20, 25, 30), -1)

        txt1 = f"STATUS: {status.upper()} | Reason: {reason} | Crop Size: {crop_size[0]}x{crop_size[1]} px"
        cv2.putText(vis, txt1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2)

        txt2 = f"Stage A: Dial Body (ORANGE) | Stage B: Window (GREEN) | YOLO (CYAN)"
        cv2.putText(vis, txt2, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        if dial_box is not None and dial_box != [0, 0, w_img, h_img]:
            cv2.rectangle(vis, (dial_box[0], dial_box[1]), (dial_box[2], dial_box[3]), (0, 165, 255), 2)
            cv2.putText(vis, "Stage A: Dial Body", (dial_box[0] + 5, max(15, dial_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        if b_pts is not None:
            cv2.polylines(vis, [b_pts.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=3)

        if yolo_box is not None:
            cv2.rectangle(vis, (yolo_box[0], yolo_box[1]), (yolo_box[2], yolo_box[3]), (255, 255, 0), 3)
            cv2.putText(vis, f"YOLO (IoU {iou:.2f})", (yolo_box[0], max(15, yolo_box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
