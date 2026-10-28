"""
Step 3C Unit and Regression Tests: False Aperture Candidate Rejection and Ranking

Validates:
1. knife_083755.jpg selects the genuine ~842x234 px display aperture rather than the 65x30 px blemish.
2. knife_083758.jpg, knife_083809.jpg, and knife_083814.jpg continue to select complete display windows.
3. Existing presets preserve their verified readings and public result structure.
4. Small and distant images (80.jpeg) preserve safe status behavior and select the true aperture.
5. Candidate ranking rejects casing blemishes based on general geometry and relative dimensions.
6. Baseline full-resolution and coarse-to-fine detection behave consistently.
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
from src.infer import DialReadingPipeline

KNIFE_DIR = PROJECT_ROOT / "data" / "raw" / "knife"
P_083755 = KNIFE_DIR / "20261006_083755899_iOS.jpg"
P_083758 = KNIFE_DIR / "20261006_083758932_iOS.jpg"
P_083809 = KNIFE_DIR / "20261006_083809306_iOS.jpg"
P_083814 = KNIFE_DIR / "20261006_083814174_iOS.jpg"


# =============================================================================
# 1. knife_083755.jpg Selects Genuine Display Window (Not 65x30 Blemish)
# =============================================================================
def test_knife_083755_selects_genuine_display_window():
    """Verify knife_083755 selects genuine ~842x234 window instead of 65x30 blemish."""
    if not P_083755.exists():
        pytest.skip(f"Test image not found: {P_083755}")

    det = WindowDetector(enable_coarse_to_fine=True)
    res = det.detect(P_083755)

    assert res["status"] == "ok", f"Expected status 'ok', got {res['status']} ({res.get('reason')})"
    w, h = res["warped_size"]

    # Must be the genuine large aperture (~842x234), NOT the 65x30 micro blemish
    assert w > 700, f"Expected genuine aperture width > 700, got {w}"
    assert h > 180, f"Expected genuine aperture height > 180, got {h}"
    assert abs(w - 842) <= 30
    assert abs(h - 234) <= 20


# =============================================================================
# 2. Complete Display Windows Selected Across Knife Series
# =============================================================================
@pytest.mark.parametrize("img_path", [P_083758, P_083809, P_083814])
def test_knife_series_selects_complete_display_windows(img_path):
    """Verify complete high-resolution display windows are selected across knife series."""
    if not img_path.exists():
        pytest.skip(f"Test image not found: {img_path}")

    det = WindowDetector(enable_coarse_to_fine=True)
    res = det.detect(img_path)

    assert res["status"] == "ok"
    w, h = res["warped_size"]
    assert w > 700, f"Expected aperture width > 700, got {w}"
    assert h > 180, f"Expected aperture height > 180, got {h}"


# =============================================================================
# 3. Existing Presets Preserve Verified Readings
# =============================================================================
def test_existing_presets_preserve_verified_readings():
    """Verify that preset dials preserve verified readings and status."""
    pipeline = DialReadingPipeline()

    # Preset 78
    p78 = PROJECT_ROOT / "data" / "presets" / "dial_0078.png"
    if p78.exists():
        res78 = pipeline.read_image(p78)
        assert res78["value"] == "0078.0"
        assert res78["status"] == "ok"

    # Preset 80
    p80 = PROJECT_ROOT / "data" / "presets" / "dial_0080.png"
    if p80.exists():
        res80 = pipeline.read_image(p80)
        assert res80["value"] == "0080.0"
        assert res80["status"] == "ok"

    # 80.2.png
    p80_2 = PROJECT_ROOT / "data" / "raw" / "80.2.png"
    if p80_2.exists():
        res80_2 = pipeline.read_image(p80_2)
        assert res80_2["value"] == "0080.0"
        assert res80_2["status"] == "ok"


# =============================================================================
# 4. Small and Distant Images Preserve Safe Status Behavior
# =============================================================================
def test_distant_image_preserves_safe_status_and_true_aperture():
    """Verify 80.jpeg selects true ~110x42 dial aperture and flags too_far status safely."""
    p_80_jpeg = PROJECT_ROOT / "data" / "raw" / "80.jpeg"
    if not p_80_jpeg.exists():
        pytest.skip(f"Test image not found: {p_80_jpeg}")

    det = WindowDetector()
    res = det.detect(p_80_jpeg)

    # Must select the true ~110x42 dial aperture
    w, h = res["warped_size"]
    assert abs(w - 110) <= 15, f"Expected true aperture width ~110, got {w}"
    assert abs(h - 42) <= 10, f"Expected true aperture height ~42, got {h}"

    # Must correctly flag too_far because height < 60px
    assert res["status"] == "too_far"


# =============================================================================
# 5. Geometric Blemish Rejection (Synthetic)
# =============================================================================
def test_synthetic_casing_blemish_rejection_geometry():
    """Verify candidate ranking rejects casing blemishes and selects genuine aperture."""
    det = WindowDetector()

    # Create a synthetic 1000x800 orange casing ROI
    w_roi, h_roi = 1000, 800
    img = np.full((h_roi, w_roi, 3), (25, 95, 225), dtype=np.uint8)

    # 1. Genuine aperture: 550x160 px (rel_w = 0.55), dark, with white strokes
    win_cx, win_cy, win_w, win_h = 500, 300, 550, 160
    cv2.rectangle(
        img,
        (win_cx - win_w // 2, win_cy - win_h // 2),
        (win_cx + win_w // 2, win_cy + win_h // 2),
        (18, 18, 18),
        -1
    )
    # Red drum
    cv2.rectangle(
        img,
        (win_cx + win_w // 2 - win_w // 4, win_cy - win_h // 2),
        (win_cx + win_w // 2, win_cy + win_h // 2),
        (25, 25, 195),
        -1
    )
    # White strokes
    drum_w = win_w // 4
    for i in range(4):
        dx = (win_cx - win_w // 2) + int((i + 0.5) * drum_w)
        cv2.rectangle(img, (dx - 4, win_cy - 30), (dx + 4, win_cy + 30), (245, 245, 245), -1)

    # 2. Casing blemish: 45x20 px (rel_w = 0.045), surrounded entirely by orange
    blem_cx, blem_cy, blem_w, blem_h = 500, 600, 45, 20
    cv2.rectangle(
        img,
        (blem_cx - blem_w // 2, blem_cy - blem_h // 2),
        (blem_cx + blem_w // 2, blem_cy + blem_h // 2),
        (25, 25, 25),
        -1
    )
    cv2.rectangle(img, (blem_cx - 2, blem_cy - 5), (blem_cx + 2, blem_cy + 5), (245, 245, 245), -1)

    dial_box = [0, 0, w_roi, h_roi]
    cands = det._find_aperture_candidates(img, [dial_box])

    # The 45x20 blemish must be rejected by relative size check
    for c in cands:
        assert c["w"] > 200, f"Expected only genuine aperture, found blemish candidate of width {c['w']}"
    assert len(cands) >= 1
    assert abs(cands[0]["w"] - win_w) <= 20


# =============================================================================
# 6. Baseline and Coarse-to-Fine Consistency
# =============================================================================
def test_baseline_and_coarse_to_fine_consistency():
    """Verify that baseline and coarse-to-fine modes select the identical window on 083755."""
    if not P_083755.exists():
        pytest.skip(f"Test image not found: {P_083755}")

    det_base = WindowDetector(enable_coarse_to_fine=False)
    det_c2f = WindowDetector(enable_coarse_to_fine=True)

    res_base = det_base.detect(P_083755)
    res_c2f = det_c2f.detect(P_083755)

    assert res_base["status"] == "ok"
    assert res_c2f["status"] == "ok"
    assert abs(res_base["warped_size"][0] - res_c2f["warped_size"][0]) <= 30
    assert abs(res_base["warped_size"][1] - res_c2f["warped_size"][1]) <= 10
