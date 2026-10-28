"""
Step 3A Unit and Regression Tests: Coarse-to-Fine Display Window Detection

Validates:
1. Large images (> 1280 px) trigger coarse downscaled candidate detection (was_coarse_to_fine=True).
2. Small images (<= 1280 px) bypass downscaling completely (was_coarse_to_fine=False).
3. Coordinate mapping from downscaled candidate boxes back to original full resolution
   uses separate x and y scale factors and handles non-square aspect ratios.
4. Final warped display crop retains 100% full original image resolution.
5. Safe fallback: if coarse downscaled detection produces no candidates or invalid candidates,
   the detector safely falls back to the full-resolution detection path.
6. Graceful failure: images with no dial casing return status='no_window' without crashing.
7. Out-of-bounds bounding box protection: invalid or edge boxes never cause index errors.
8. DialDetector wrapper integration: parameters and result flags pass through correctly.
9. Preset regressions: dial_0078.png and dial_0080.png produce identical readings and structure.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.window_detect import WindowDetector
from src.detect import DialDetector
from src.infer import DialReadingPipeline


def create_synthetic_dial_image(
    img_w: int,
    img_h: int,
    cx: int,
    cy: int,
    radius: int
) -> np.ndarray:
    """Creates a synthetic dial image with an orange casing and dark digit aperture."""
    # Machine background
    img = np.full((img_h, img_w, 3), (120, 115, 110), dtype=np.uint8)

    # Orange circular bezel (Elesa orange: BGR ~ (20, 95, 225))
    cv2.circle(img, (cx, cy), radius, (20, 95, 225), -1)

    # Dark aperture
    win_w = int(radius * 1.2)
    win_h = int(radius * 0.35)
    win_x1 = cx - win_w // 2
    win_y1 = cy - int(radius * 0.25)
    win_x2 = win_x1 + win_w
    win_y2 = win_y1 + win_h
    cv2.rectangle(img, (win_x1, win_y1), (win_x2, win_y2), (18, 18, 18), -1)

    # Red drum on rightmost 25% of aperture
    drum_w = win_w // 4
    red_x1 = win_x2 - drum_w
    cv2.rectangle(img, (red_x1, win_y1), (win_x2, win_y2), (25, 25, 195), -1)

    # White digit strokes in each drum
    for i in range(4):
        dx = win_x1 + int((i + 0.5) * drum_w)
        dy = (win_y1 + win_y2) // 2
        cv2.rectangle(img, (dx - 5, dy - win_h // 3), (dx + 5, dy + win_h // 3), (245, 245, 245), -1)

    return img


# =============================================================================
# 1. Large Image Uses Coarse-to-Fine Path
# =============================================================================
def test_large_image_uses_coarse_to_fine_path():
    """Verify that images with max_dim > 1280 px use coarse-to-fine detection."""
    detector = WindowDetector(max_coarse_dim=1280, enable_coarse_to_fine=True)
    # 2400 x 1800 px image
    img = create_synthetic_dial_image(2400, 1800, 1200, 900, 450)

    res = detector.detect(img)
    assert res["status"] in ("ok", "too_far")
    assert res["was_coarse_to_fine"] is True
    assert res["used_fallback"] is False
    assert res["warped_window"] is not None
    # Rectified window should have high resolution
    w_crop, h_crop = res["warped_size"]
    assert w_crop > 400, f"Expected full-res crop width > 400, got {w_crop}"


# =============================================================================
# 2. Small Image Bypasses Downscaling
# =============================================================================
def test_small_image_bypasses_downscaling():
    """Verify that images with max_dim <= 1280 px bypass coarse downscaling."""
    detector = WindowDetector(max_coarse_dim=1280, enable_coarse_to_fine=True)
    # 800 x 600 px image
    img = create_synthetic_dial_image(800, 600, 400, 300, 150)

    res = detector.detect(img)
    assert res["status"] in ("ok", "too_far")
    assert res["was_coarse_to_fine"] is False
    assert res["used_fallback"] is False
    assert res["warped_window"] is not None


# =============================================================================
# 3. Non-Square Aspect Ratio Coordinate Mapping
# =============================================================================
def test_non_square_aspect_coordinate_mapping():
    """Verify coordinate mapping across non-square images with separate fx and fy factors."""
    detector = WindowDetector(max_coarse_dim=1280, enable_coarse_to_fine=True)
    # Non-square: 2500 x 1700 (aspect 1.4705)
    w_orig, h_orig = 2500, 1700
    dial_cx, dial_cy, dial_r = 1600, 800, 300
    img = create_synthetic_dial_image(w_orig, h_orig, dial_cx, dial_cy, dial_r)

    dial_boxes, was_downscaled = detector._find_coarse_dial_boxes(img)
    assert was_downscaled is True
    assert len(dial_boxes) > 0

    # Ensure mapped candidate bounding box in full resolution contains the dial center
    box = dial_boxes[0]
    bx1, by1, bx2, by2 = box
    assert bx1 <= dial_cx <= bx2, f"dial_cx {dial_cx} not within [{bx1}, {bx2}]"
    assert by1 <= dial_cy <= by2, f"dial_cy {dial_cy} not within [{by1}, {by2}]"

    # Bounds must strictly fit within original image bounds
    assert bx1 >= 0 and by1 >= 0
    assert bx2 <= w_orig and by2 <= h_orig


# =============================================================================
# 4. Final Display Crops Retain 100% Original Resolution
# =============================================================================
def test_display_crops_retain_full_original_resolution():
    """Verify that final warped crop retains full original resolution and is not downscaled."""
    detector = WindowDetector(max_coarse_dim=1280, enable_coarse_to_fine=True)
    w_orig, h_orig = 2000, 1500
    r = 350
    img = create_synthetic_dial_image(w_orig, h_orig, 1000, 750, r)

    # Detect with coarse-to-fine enabled
    res_coarse = detector.detect(img)
    assert res_coarse["was_coarse_to_fine"] is True
    crop_coarse = res_coarse["warped_window"]

    # Detect with coarse-to-fine disabled (baseline fullres)
    detector_full = WindowDetector(enable_coarse_to_fine=False)
    res_full = detector_full.detect(img)
    assert res_full["was_coarse_to_fine"] is False
    crop_full = res_full["warped_window"]

    # Both crops should have nearly identical full-resolution dimensions (within 2 px)
    assert abs(crop_coarse.size[0] - crop_full.size[0]) <= 2
    assert abs(crop_coarse.size[1] - crop_full.size[1]) <= 2
    # Verify crop is not downscaled by ~0.64 factor (which would yield ~260 px width instead of ~420)
    assert crop_coarse.size[0] > 380


# =============================================================================
# 5. Failed Coarse Candidate Falls Back Safely
# =============================================================================
def test_failed_coarse_candidate_falls_back_to_fullres():
    """Verify that if coarse search on downscaled image yields no candidate, fallback executes."""
    detector = WindowDetector(max_coarse_dim=1280, enable_coarse_to_fine=True)
    img = create_synthetic_dial_image(2400, 1800, 1200, 900, 450)

    orig_find_coarse = detector._find_coarse_dial_boxes

    # Simulate coarse detection returning empty on downscaled image, but succeeding on fullres
    def mock_find_coarse(bgr, force_fullres=False):
        if not force_fullres:
            return [], True
        return orig_find_coarse(bgr, force_fullres=True)

    detector._find_coarse_dial_boxes = mock_find_coarse

    res = detector.detect(img)
    assert res["status"] in ("ok", "too_far")
    assert res["used_fallback"] is True
    assert res["was_coarse_to_fine"] is False
    assert res["warped_window"] is not None


# =============================================================================
# 6. Graceful Failure on Images Without Dial Casing
# =============================================================================
def test_no_orange_casing_graceful_no_window():
    """Verify that an image with no orange dial returns 'no_window' safely."""
    detector = WindowDetector(max_coarse_dim=1280, enable_coarse_to_fine=True)
    # Plain metallic texture
    img = np.full((1600, 1200, 3), (128, 128, 128), dtype=np.uint8)

    res = detector.detect(img)
    assert res["status"] == "no_window"
    assert res["crop"] is None
    assert res["warped_window"] is None
    assert res["reason"] == "no_compact_orange_body_found"


# =============================================================================
# 7. Out-of-Bounds and Invalid Box Protection
# =============================================================================
def test_out_of_bounds_and_invalid_box_safety():
    """Verify that invalid or edge candidate boxes do not raise index errors."""
    detector = WindowDetector()
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Test extreme/invalid candidate boxes
    invalid_boxes = [
        [-50, -50, 10, 10],
        [0, 0, 5, 5],            # too small (< 10 px)
        [390, 390, 500, 500],    # out of image bounds
        [200, 200, 200, 200],    # zero area
    ]

    # Should run and return empty candidates list without crashing
    candidates = detector._find_aperture_candidates(img, invalid_boxes)
    assert isinstance(candidates, list)


# =============================================================================
# 8. DialDetector Integration and Parameter Pass-Through
# =============================================================================
def test_detector_wrapper_coarse_to_fine_integration():
    """Verify that DialDetector correctly exposes and passes coarse-to-fine parameters."""
    det = DialDetector(max_coarse_dim=1000, enable_coarse_to_fine=True)
    assert det.max_coarse_dim == 1000
    assert det.enable_coarse_to_fine is True

    img = create_synthetic_dial_image(1600, 1200, 800, 600, 300)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results = det.detect(pil_img)
    assert len(results) > 0
    top = results[0]
    assert "was_coarse_to_fine" in top
    assert top["was_coarse_to_fine"] is True
    assert top["used_fallback"] is False


# =============================================================================
# 9. Preset Regression Parity and Output Format Integrity
# =============================================================================
def test_pipeline_presets_and_output_structure_unchanged():
    """Verify that DialReadingPipeline preserves all public keys and verified readings on presets."""
    pipeline = DialReadingPipeline()

    expected_keys = {
        "value", "whole_digits", "fraction", "tenth", "decimal_info",
        "confidence", "status", "per_digit_confidence", "is_mid_roll",
        "diagnostics", "slot_diagnostics", "detector_confidence",
        "dial_variant", "window_crop", "oriented_window", "rectified_window",
        "digit_slots", "slot_ranges", "digit_vertical_positions",
        "detected_overlay", "detection_box", "red_drum_crop",
        "red_drum_range", "tick_crop", "tick_range", "vernier_margin_px"
    }

    # Test preset 78
    p78 = PROJECT_ROOT / "data" / "presets" / "dial_0078.png"
    if p78.exists():
        res78 = pipeline.read_image(p78)
        assert res78["value"] == "0078.0"
        assert res78["whole_digits"] == "0078"
        assert res78["tenth"] == 0
        assert res78["status"] == "ok"
        for k in expected_keys:
            assert k in res78, f"Missing key {k} in read_image output"

    # Test preset 80
    p80 = PROJECT_ROOT / "data" / "presets" / "dial_0080.png"
    if p80.exists():
        res80 = pipeline.read_image(p80)
        assert res80["value"] == "0080.0"
        assert res80["whole_digits"] == "0080"
        assert res80["tenth"] == 0
        assert res80["status"] == "ok"
        for k in expected_keys:
            assert k in res80, f"Missing key {k} in read_image output"
