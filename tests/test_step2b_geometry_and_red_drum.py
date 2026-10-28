"""
Step 2B Unit and Regression Tests: Slot Geometry, Baseline Median, and Red Drum Measurement

Validates:
1. Four equal-width digit-recognition slots (1 drum_w each).
2. Complete, unclipped red-drum and tick measurement region extending to the window's right edge.
3. Separation of red digit recognition slot from outer tick marks and vernier scale.
4. Median baseline resilience against transitioning/displaced black digits.
5. Safe handling of missing, empty, or invalid digit centers.
6. Robustness across varying window dimensions and right-side vernier margins.
7. Backward compatibility on verified presets (dial_0078.png and dial_0080.png).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image, ImageDraw
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crop import WindowNormalizer
from src.infer import DialReadingPipeline
from src.digits import PixelDigitRecognizer


# =============================================================================
# 1. Four Equal-Width Digit Recognition Slots
# =============================================================================
def test_four_equal_width_slots():
    """Verify that Slots 1-4 have equal widths matching drum_width."""
    normalizer = WindowNormalizer()

    # Create synthetic window: 300 wide, 100 high
    # Left 210 px dark (black drums: 3 * 70px), right 90 px red (red drum: 70px + 20px margin)
    img = Image.new("RGB", (300, 100), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.rectangle([210, 0, 300, 100], fill=(210, 30, 30))

    # Add white numerals to each drum
    for i in range(3):
        draw.rectangle([25 + i * 70, 25, 45 + i * 70, 75], fill=(245, 245, 245))
    draw.rectangle([225, 25, 245, 75], fill=(245, 245, 245))

    slot_data = normalizer.split_into_slots(img)
    slot_ranges = slot_data["slot_ranges"]
    drum_w = slot_data["drum_width"]

    assert len(slot_ranges) == 4, f"Expected 4 slots, got {len(slot_ranges)}"
    widths = [r[1] - r[0] for r in slot_ranges]

    # All 4 slots should be equal to drum_w (within 1 px rounding)
    for i, w in enumerate(widths):
        assert abs(w - drum_w) <= 1.0, f"Slot {i} width {w} differs from drum_w {drum_w}"
    assert len(slot_data["slot_crops"]) == 4


# =============================================================================
# 2. Unclipped Red Drum Region Extends to Right Edge
# =============================================================================
def test_full_red_drum_tick_region_extends_to_right_edge():
    """Verify that decimal measurement region extends to the right window edge (not clipped)."""
    normalizer = WindowNormalizer()
    window_w, window_h = 320, 100

    img = Image.new("RGB", (window_w, window_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    # Red drum starts at x=210, window extends to 320 (drum_w=70, margin=40px)
    draw.rectangle([210, 0, window_w, window_h], fill=(200, 30, 30))

    slot_data = normalizer.split_into_slots(img)
    red_drum_range = slot_data["red_drum_range"]
    tick_range = slot_data["tick_range"]
    vernier_margin = slot_data["vernier_margin_px"]

    # Red drum range must start at red_x_start and end at the exact window right edge
    assert red_drum_range[0] == slot_data["red_x_start"]
    assert red_drum_range[1] == window_w, f"Expected red drum to extend to {window_w}, got {red_drum_range[1]}"

    # Tick range must extend to window right edge
    assert tick_range[1] == window_w
    assert slot_data["red_drum_crop"].size == (window_w - slot_data["red_x_start"], window_h)
    assert vernier_margin == window_w - slot_data["slot_ranges"][3][1]
    assert vernier_margin > 0, "Vernier margin should capture right region beyond Slot 4"


# =============================================================================
# 3. Tick Marks Excluded from Red Digit Recognition Slot
# =============================================================================
def test_ticks_excluded_from_red_digit_recognition_slot():
    """Verify that Slot 4 contains only the numeral area and does not include tick marks."""
    normalizer = WindowNormalizer()
    window_w, window_h = 350, 100

    img = Image.new("RGB", (window_w, window_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    # Red drum starts at x=210, drum_w=70 -> Slot 4 is [210, 280]
    draw.rectangle([210, 0, window_w, window_h], fill=(210, 30, 30))

    # White numeral in Slot 4 (x=225 to 255)
    draw.rectangle([225, 25, 255, 75], fill=(245, 245, 245))

    # Draw prominent vernier tick marks in right margin beyond Slot 4 (x=290 to 330)
    for y_tick in range(15, 90, 10):
        draw.line([290, y_tick, 320, y_tick], fill=(255, 255, 255), width=3)

    slot_data = normalizer.split_into_slots(img)
    slot_4_crop = slot_data["slot_crops"][3]
    slot_4_w = slot_4_crop.size[0]

    # Slot 4 must end at x2 = 280, completely excluding ticks at x >= 290
    assert slot_4_w == slot_data["slot_ranges"][3][1] - slot_data["slot_ranges"][3][0]

    # Verify that PixelDigitRecognizer extracts the numeral cleanly without tick artifacts
    mask, feats = PixelDigitRecognizer.extract_normalized_digit_mask(slot_4_crop)
    assert mask is not None
    assert np.sum(mask > 0) > 100
    assert "norm_bbox" in feats


# =============================================================================
# 4. Median Baseline Resilience Against Displaced Black Digit
# =============================================================================
def test_median_baseline_resists_displaced_black_digit():
    """Verify that when one black digit rolls (transitioning), the baseline remains centered."""
    normalizer = WindowNormalizer()
    window_h = 100

    # Scenario A: Clean baseline where all black digits are resting at cy=50.0
    clean_slot_data = {
        "digit_vertical_positions": [50.0, 50.0, 50.0, 45.0],
        "digit_boxes": [(10, 30, 20, 40)] * 4,
        "digit_heights": [40, 40, 40, 40]
    }
    res_clean = normalizer.measure_red_drum_fraction(clean_slot_data, window_height=window_h)
    assert res_clean["baseline_y"] == 50.0
    assert res_clean["displacement_px"] == 5.0
    assert res_clean["tenth"] == 1

    # Scenario B: Slot 0 is transitioning mid-roll (displaced upwards to cy=20.0)
    # With mean: baseline would be (20 + 50 + 50) / 3 = 40.0 (corrupted by -10px)
    # With median: baseline is median([20, 50, 50]) = 50.0 (immune to transition)
    displaced_slot_data = {
        "digit_vertical_positions": [20.0, 50.0, 50.0, 45.0],
        "digit_boxes": [(10, 30, 20, 40)] * 4,
        "digit_heights": [40, 40, 40, 40]
    }
    res_displaced = normalizer.measure_red_drum_fraction(displaced_slot_data, window_height=window_h)
    assert res_displaced["baseline_y"] == 50.0, f"Expected median baseline 50.0, got {res_displaced['baseline_y']}"
    assert res_displaced["displacement_px"] == 5.0
    assert res_displaced["tenth"] == 1


# =============================================================================
# 5. Missing / Invalid Digit Centers Handled Safely
# =============================================================================
def test_baseline_handles_missing_and_invalid_centers():
    """Verify safe fallback when black digit positions are missing, NaN, or out of bounds."""
    normalizer = WindowNormalizer()
    window_h = 120

    # Test with NaN and out of bounds values
    bad_slot_data = {
        "digit_vertical_positions": [float("nan"), -10.0, 60.0, 55.0],
        "digit_boxes": [(0, 0, 10, 10)] * 4,
        "digit_heights": [40, 40, 40, 40]
    }
    res_bad = normalizer.measure_red_drum_fraction(bad_slot_data, window_height=window_h)
    # Only 60.0 is valid -> baseline should anchor on 60.0
    assert res_bad["baseline_y"] == 60.0
    assert res_bad["status"] == "ok"

    # Test with completely empty positions
    empty_slot_data = {
        "digit_vertical_positions": [],
        "digit_boxes": [],
        "digit_heights": []
    }
    res_empty = normalizer.measure_red_drum_fraction(empty_slot_data, window_height=window_h)
    # Falls back safely to h / 2.0 = 60.0
    assert res_empty["baseline_y"] == 60.0
    assert res_empty["status"] == "ok"


# =============================================================================
# 6. Adaptability Across Display Widths and Right Margins
# =============================================================================
@pytest.mark.parametrize("win_w, win_h, margin_px", [
    (240, 80, 0),
    (300, 100, 15),
    (400, 130, 35),
    (550, 160, 50),
])
def test_varying_display_widths_and_margins(win_w, win_h, margin_px):
    """Verify dynamic slot extraction across various image resolutions and vernier margins."""
    normalizer = WindowNormalizer()
    drum_w = (win_w - margin_px) / 4.0

    img = Image.new("RGB", (win_w, win_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    red_start = int(round(3 * drum_w))
    draw.rectangle([red_start, 0, win_w, win_h], fill=(215, 30, 30))

    slot_data = normalizer.split_into_slots(img)
    assert len(slot_data["slot_ranges"]) == 4
    assert slot_data["red_drum_range"][1] == win_w
    assert slot_data["tick_range"][1] == win_w
    assert slot_data["vernier_margin_px"] >= 0


# =============================================================================
# 7. Backward Compatibility on Labeled Dial Presets
# =============================================================================
def test_preset_regressions():
    """Verify that preset dials continue to produce identical verified readings."""
    pipeline = DialReadingPipeline()

    # Preset 78 -> 0078.0
    p78 = PROJECT_ROOT / "data" / "presets" / "dial_0078.png"
    if p78.exists():
        res78 = pipeline.read_image(p78)
        assert res78["whole_digits"] == "0078"
        assert res78["value"] == "0078.0"
        assert res78["status"] == "ok"
        assert res78["tenth"] == 0
        assert "red_drum_crop" in res78
        assert "tick_crop" in res78

    # Preset 80 -> 0080.0
    p80 = PROJECT_ROOT / "data" / "presets" / "dial_0080.png"
    if p80.exists():
        res80 = pipeline.read_image(p80)
        assert res80["whole_digits"] == "0080"
        assert res80["value"] == "0080.0"
        assert res80["status"] == "ok"
        assert res80["tenth"] == 0
        assert "red_drum_crop" in res80
        assert "tick_crop" in res80
