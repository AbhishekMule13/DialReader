"""
Unit and Integration Tests for Classical PixelDigitRecognizer (Step 2)

Verifies:
1. Binary-mask normalization and aspect ratio preservation.
2. Geometric feature extraction (holes, aspect ratio, crossbar, projections).
3. Template matching using synthetic test masks with explicitly known labels (0-9).
4. Rejection of unknown, blank, or ambiguous digits.
5. Correct handling of missing templates and template provenance tracking.
6. Compatibility with existing inference pipeline output structure.
7. Verification that the Method 2 decimal measurement function is completely unchanged.
8. End-to-end recognition on verified dial presets (dial_0078.png and dial_0080.png).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.digits import PixelDigitRecognizer, DigitTemplateLibrary, DigitTemplate
from src.infer import DialReadingPipeline
from src.crop import WindowNormalizer


# =============================================================================
# 1. Normalization Tests
# =============================================================================
def test_binary_mask_normalization():
    """Verify normalization scales digits into canonical size while preserving aspect ratio."""
    target_size = (48, 32)
    # Create a synthetic tall digit (resembling '1': h=60, w=18, aspect=0.3)
    tall_img = Image.new("RGB", (50, 80), color=(20, 20, 20))
    draw = ImageDraw.Draw(tall_img)
    draw.rectangle([16, 10, 34, 70], fill=(240, 240, 240))

    norm_mask, feats = PixelDigitRecognizer.extract_normalized_digit_mask(tall_img, target_size=target_size)
    assert norm_mask is not None
    assert norm_mask.shape == target_size
    assert norm_mask.dtype == np.uint8

    # Verify values are strictly binary (0 or 255)
    unique_vals = set(np.unique(norm_mask))
    assert unique_vals.issubset({0, 255})

    # Aspect ratio should be preserved: width must be smaller than height
    norm_bw, norm_bh = feats["norm_bbox"][2], feats["norm_bbox"][3]
    assert norm_bw < norm_bh
    # Canonical size padding: must not touch the extreme outer border
    assert norm_mask[0, :].sum() == 0
    assert norm_mask[-1, :].sum() == 0


def test_normalization_handles_red_and_black_backgrounds():
    """Verify that white numerals are cleanly segmented from both black and red drums."""
    target_size = (48, 32)

    # Black drum: black background (25, 25, 25), white numeral (240, 240, 240)
    blk_img = Image.new("RGB", (60, 90), color=(25, 25, 25))
    draw_blk = ImageDraw.Draw(blk_img)
    draw_blk.rectangle([15, 15, 45, 75], fill=(240, 240, 240))

    mask_blk, feats_blk = PixelDigitRecognizer.extract_normalized_digit_mask(blk_img, target_size=target_size)
    assert mask_blk is not None
    assert np.sum(mask_blk > 0) > 100

    # Red drum: red background (200, 35, 35), white numeral (245, 245, 245)
    red_img = Image.new("RGB", (60, 90), color=(200, 35, 35))
    draw_red = ImageDraw.Draw(red_img)
    draw_red.rectangle([15, 15, 45, 75], fill=(245, 245, 245))

    mask_red, feats_red = PixelDigitRecognizer.extract_normalized_digit_mask(red_img, target_size=target_size)
    assert mask_red is not None
    assert np.sum(mask_red > 0) > 100


# =============================================================================
# 2. Geometric Feature Extraction Tests
# =============================================================================
def test_geometric_feature_holes():
    """Verify hole detection accurately distinguishes digits 0, 8, and 7."""
    th, tw = 48, 32

    # Synthesize digit '0' (oval with 1 central hole)
    mask_0 = np.zeros((th, tw), dtype=np.uint8)
    cv2.ellipse(mask_0, (tw // 2, th // 2), (10, 18), 0, 0, 360, 255, -1)
    cv2.ellipse(mask_0, (tw // 2, th // 2), (4, 10), 0, 0, 360, 0, -1)
    feats_0 = PixelDigitRecognizer.extract_geometric_features(mask_0)
    assert feats_0["holes"] == 1
    assert feats_0["has_crossbar"] is False

    # Synthesize digit '8' (figure-8 with 2 holes and middle crossbar)
    mask_8 = np.zeros((th, tw), dtype=np.uint8)
    cv2.ellipse(mask_8, (tw // 2, th // 2), (11, 18), 0, 0, 360, 255, -1)
    cv2.ellipse(mask_8, (tw // 2, th // 4 + 2), (4, 5), 0, 0, 360, 0, -1)
    cv2.ellipse(mask_8, (tw // 2, 3 * th // 4 - 2), (4, 5), 0, 0, 360, 0, -1)
    feats_8 = PixelDigitRecognizer.extract_geometric_features(mask_8)
    assert feats_8["holes"] == 2
    assert feats_8["has_crossbar"] is True

    # Synthesize digit '7' (horizontal top bar + diagonal stem, 0 holes)
    mask_7 = np.zeros((th, tw), dtype=np.uint8)
    cv2.rectangle(mask_7, (4, 4), (28, 10), 255, -1)
    cv2.line(mask_7, (26, 8), (10, 42), 255, 4)
    feats_7 = PixelDigitRecognizer.extract_geometric_features(mask_7)
    assert feats_7["holes"] == 0
    assert feats_7["top_half_density"] > feats_7["bottom_half_density"]


def test_geometric_feature_projections():
    """Verify horizontal and vertical projection profiles."""
    th, tw = 48, 32
    mask = np.zeros((th, tw), dtype=np.uint8)
    mask[10:20, 5:15] = 255
    feats = PixelDigitRecognizer.extract_geometric_features(mask)

    assert len(feats["h_proj"]) == th
    assert len(feats["v_proj"]) == tw
    assert sum(feats["h_proj"]) == 100
    assert sum(feats["v_proj"]) == 100


# =============================================================================
# 3. Synthetic Template Matching Tests (All Digits 0-9)
# =============================================================================
def test_template_matching_synthetic_all_digits():
    """
    Renders synthetic test masks with known labels 0-9 and verifies that
    PixelDigitRecognizer matches them with correct labels.
    """
    library = DigitTemplateLibrary(canonical_size=(48, 32))
    registered_digits = library.load_synthetic_test_templates()
    assert len(registered_digits) == 10
    assert library.missing_digits() == []

    recognizer = PixelDigitRecognizer(
        template_library=library,
        min_confidence=0.30,
        ambiguity_margin=0.04,
        auto_load_verified=False
    )

    # For each digit 0-9, test recognition
    th, tw = (48, 32)
    font_names = ["arial.ttf", "calibri.ttf", "DejaVuSans.ttf", "tahoma.ttf"]
    font = None
    for fn in font_names:
        try:
            font = ImageFont.truetype(fn, int(th * 0.72))
            break
        except IOError:
            continue
    if font is None:
        font = ImageFont.load_default()

    correct_count = 0
    for expected_digit in range(10):
        img = Image.new("L", (tw * 2, th * 2), color=0)
        draw = ImageDraw.Draw(img)
        draw.text((tw, th), str(expected_digit), fill=255, font=font, anchor="mm")
        test_slot = Image.fromarray(np.array(img)).convert("RGB")

        pred = recognizer.predict_slot(test_slot)
        if pred["digit"] == expected_digit and pred["confidence"] >= 0.30:
            correct_count += 1

    # At least 9 out of 10 synthetic digits should match correctly
    assert correct_count >= 9, f"Expected >= 9 correct synthetic matches, got {correct_count}/10"


# =============================================================================
# 4. Unknown, Ambiguous, and Blank Digit Rejection Tests
# =============================================================================
def test_rejection_of_blank_image():
    """Verify that a blank slot image returns status='no_digit' and confidence=0.0."""
    recognizer = PixelDigitRecognizer(auto_load_verified=True, project_root=PROJECT_ROOT)
    blank_img = Image.new("RGB", (60, 100), color=(20, 20, 20))

    res = recognizer.predict_slot(blank_img)
    assert res["status"] in ("no_digit", "uncertain")
    assert res["confidence"] == 0.0


def test_rejection_of_random_noise():
    """Verify that pure random noise is rejected or flagged as uncertain."""
    recognizer = PixelDigitRecognizer(auto_load_verified=True, project_root=PROJECT_ROOT)
    np.random.seed(42)
    noise_arr = np.random.randint(0, 255, (100, 60, 3), dtype=np.uint8)
    noise_img = Image.fromarray(noise_arr)

    res = recognizer.predict_slot(noise_img)
    assert res["status"] in ("uncertain", "no_digit", "ambiguous")
    assert res["confidence"] < 0.50


# =============================================================================
# 5. Missing Template Handling Tests
# =============================================================================
def test_handling_of_missing_templates():
    """
    Verify behavior when only a subset of templates is available.
    When only verified templates [0, 7, 8] are loaded, digits [1, 2, 3, 4, 5, 6, 9]
    are correctly reported as missing.
    """
    library = DigitTemplateLibrary(canonical_size=(48, 32))
    # Register only templates for 0 and 8
    t0 = np.zeros((48, 32), dtype=np.uint8)
    cv2.circle(t0, (16, 24), 10, 255, 3)
    library.register_template(0, t0, source="test_0", verified=True)

    t8 = np.zeros((48, 32), dtype=np.uint8)
    cv2.circle(t8, (16, 14), 7, 255, 3)
    cv2.circle(t8, (16, 32), 8, 255, 3)
    library.register_template(8, t8, source="test_8", verified=True)

    assert library.available_digits() == [0, 8]
    assert set(library.missing_digits()) == {1, 2, 3, 4, 5, 6, 7, 9}
    assert library.has_template(0) is True
    assert library.has_template(1) is False

    recognizer = PixelDigitRecognizer(template_library=library, auto_load_verified=False)

    # Empty library test
    empty_library = DigitTemplateLibrary(canonical_size=(48, 32))
    empty_recog = PixelDigitRecognizer(template_library=empty_library, auto_load_verified=False)
    slot_sample = Image.new("RGB", (40, 60), color=(20, 20, 20))
    draw = ImageDraw.Draw(slot_sample)
    draw.rectangle([10, 10, 30, 50], fill=(240, 240, 240))
    res = empty_recog.predict_slot(slot_sample)
    assert res["status"] == "missing_template"
    assert res["confidence"] == 0.0


def test_verified_presets_template_registration():
    """
    Verify that loading preset templates extracts only verified real templates
    (digits 0, 7, 8) and correctly reports missing digits [1, 2, 3, 4, 5, 6, 9].
    """
    library = DigitTemplateLibrary(canonical_size=(48, 32))
    summary = library.load_verified_preset_templates(project_root=PROJECT_ROOT)

    available = library.available_digits()
    missing = library.missing_digits()

    assert 0 in available, "Expected digit 0 to be available from presets"
    assert 7 in available, "Expected digit 7 to be available from preset dial_0078"
    assert 8 in available, "Expected digit 8 to be available from presets"

    # Digits without verified ground-truth presets must be reported as missing
    for d in [1, 2, 3, 4, 5, 6, 9]:
        assert d in missing, f"Expected digit {d} to be reported missing without fabricated data"


# =============================================================================
# 6. Pipeline Compatibility & Output Structure Tests
# =============================================================================
def test_pipeline_compatibility_and_output_structure():
    """
    Verify that DialReadingPipeline.read_image() returns all required fields
    for app.py and CLI, preserving leading zeros, diagnostics, and decimal info.
    """
    pipeline = DialReadingPipeline()
    preset_path = PROJECT_ROOT / "data" / "presets" / "dial_0078.png"
    if not preset_path.exists():
        pytest.skip(f"Preset image not found: {preset_path}")

    res = pipeline.read_image(preset_path)

    # 1. Output keys check
    expected_keys = [
        "value", "whole_digits", "fraction", "tenth", "decimal_info",
        "confidence", "status", "per_digit_confidence", "is_mid_roll",
        "diagnostics", "slot_diagnostics", "detector_confidence",
        "dial_variant", "window_crop", "oriented_window", "rectified_window",
        "digit_slots", "slot_ranges", "digit_vertical_positions",
        "detected_overlay", "detection_box"
    ]
    for k in expected_keys:
        assert k in res, f"Missing expected output field in pipeline result: '{k}'"

    # 2. Reading values check
    assert res["whole_digits"] == "0078", f"Expected whole_digits '0078', got '{res['whole_digits']}'"
    assert len(res["whole_digits"]) == 4, "Must preserve 4-digit reading with leading zeros"
    assert res["whole_digits"].startswith("00"), "Must preserve leading zeros"
    assert res["status"] == "ok"
    assert res["tenth"] == 0
    assert res["value"] == "0078.0"

    # 3. Per-digit confidence and diagnostics
    assert len(res["per_digit_confidence"]) == 4
    for conf in res["per_digit_confidence"]:
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    diag = res["diagnostics"]
    assert len(diag) == 4
    for slot_idx, d in enumerate(diag):
        assert d["slot_index"] == slot_idx
        assert "best_candidate_digit" in d
        assert "best_template_score" in d
        assert "second_best_score" in d
        assert "ambiguity_margin" in d
        assert "status" in d
        assert "geometric_features" in d


def test_preset_dial_0080():
    """Verify that dial_0080.png reads 0080.0 with status ok."""
    pipeline = DialReadingPipeline()
    preset_path = PROJECT_ROOT / "data" / "presets" / "dial_0080.png"
    if not preset_path.exists():
        pytest.skip(f"Preset image not found: {preset_path}")

    res = pipeline.read_image(preset_path)
    assert res["whole_digits"] == "0080"
    assert res["status"] == "ok"
    assert res["value"] == "0080.0"


# =============================================================================
# 7. Decimal Measurement Function Parity Test
# =============================================================================
def test_decimal_measurement_function_unchanged():
    """
    Verifies that WindowNormalizer.measure_red_drum_fraction preserves the
    exact Method 2 algorithm: baseline_y - red_cy displacement, pitch formula,
    and clipping to [0, 9].
    """
    normalizer = WindowNormalizer()
    window_h = 100

    # Mock slot data: Slots 0-2 baseline at cy=50.0, Slot 3 (red drum) at cy=45.0
    # Expected displacement: baseline_y - red_cy = 50.0 - 45.0 = +5.0 px
    # Expected avg_digit_h = 40.0 -> pitch = 40.0 * 1.5 = 60.0 -> pitch_per_tenth = 6.0
    # raw_tenth = 5.0 / 6.0 = 0.83 -> rounded tenth = 1
    mock_slot_data = {
        "digit_vertical_positions": [50.0, 50.0, 50.0, 45.0],
        "digit_boxes": [
            (10, 30, 20, 40),
            (35, 30, 20, 40),
            (60, 30, 20, 40),
            (85, 25, 20, 40)
        ],
        "digit_heights": [40, 40, 40, 40]
    }

    dec_res = normalizer.measure_red_drum_fraction(mock_slot_data, window_height=window_h)

    assert dec_res["baseline_y"] == 50.0
    assert dec_res["digit_cy"] == 45.0
    assert dec_res["displacement_px"] == 5.0
    assert dec_res["pitch_px"] == 60.0
    assert dec_res["tenth"] == 1
    assert dec_res["decimal_fraction"] == 0.1
    assert dec_res["status"] == "ok"
    assert 0 <= dec_res["tenth"] <= 9


# =============================================================================
# 8. Step 4B: Template System 0-9 Expansion & Safety Tests
# =============================================================================
def test_all_ten_digit_labels_supported_by_template_format():
    """Verify that all ten digits (0-9) are valid and supported in template library."""
    lib = DigitTemplateLibrary(canonical_size=(48, 32))
    
    # Register synthetic template for each digit 0..9
    dummy_mask = np.zeros((48, 32), dtype=np.uint8)
    dummy_mask[10:38, 8:24] = 255
    
    for d in range(10):
        tpl = lib.register_template(
            digit=d,
            mask=dummy_mask,
            source=f"test_all_digits_d{d}",
            verified=True,
            condition="stationary"
        )
        assert tpl.digit == d
        assert lib.has_template(d) is True
        assert len(lib.get_templates(d)) == 1

    assert sorted(lib.available_digits()) == list(range(10))
    assert lib.missing_digits() == []

    # Out of range digits must raise ValueError
    with pytest.raises(ValueError):
        lib.register_template(-1, dummy_mask, source="invalid_neg")

    with pytest.raises(ValueError):
        lib.register_template(10, dummy_mask, source="invalid_10")


def test_missing_templates_remain_explicitly_missing():
    """Verify that missing digits are explicitly tracked and not falsely reported as available."""
    lib = DigitTemplateLibrary(canonical_size=(48, 32))
    lib.load_verified_preset_templates(project_root=PROJECT_ROOT)
    
    available = lib.available_digits()
    missing = lib.missing_digits()
    
    assert set(available) == {0, 7, 8}
    assert set(missing) == {1, 2, 3, 4, 5, 6, 9}
    
    for d in [1, 2, 3, 4, 5, 6, 9]:
        assert lib.has_template(d) is False
        assert lib.get_templates(d) == []


def test_unverified_templates_excluded_from_verified_bank():
    """Verify that unverified, rolling, or sparse crops cannot enter verified bank."""
    lib = DigitTemplateLibrary(canonical_size=(48, 32))
    
    valid_mask = np.zeros((48, 32), dtype=np.uint8)
    valid_mask[10:38, 8:24] = 255
    
    # 1. Rolling condition marked verified=True must be rejected
    with pytest.raises(ValueError, match="condition='stationary'"):
        lib.register_template(1, valid_mask, source="rolling_sample", verified=True, condition="rolling")

    # 2. Sparse / empty mask marked verified=True must be rejected
    sparse_mask = np.zeros((48, 32), dtype=np.uint8)
    sparse_mask[10, 10] = 255  # only 1 white pixel
    with pytest.raises(ValueError, match="too sparse"):
        lib.register_template(1, sparse_mask, source="empty_sample", verified=True, condition="stationary")

    # 3. Non-verified templates can be registered for testing but excluded from verified_only
    lib.register_template(2, valid_mask, source="synthetic_unverified", verified=False, condition="synthetic")
    assert len(lib.get_templates(2, verified_only=False)) == 1
    assert len(lib.get_templates(2, verified_only=True)) == 0


def test_multiple_verified_templates_for_one_digit():
    """Verify that multiple verified templates for the same digit are supported and accumulated."""
    lib = DigitTemplateLibrary(canonical_size=(48, 32))
    lib.load_verified_preset_templates(project_root=PROJECT_ROOT)
    
    # Digit 0 in presets has multiple templates (slot 0 and slot 1 in dial_0078, etc.)
    t0_list = lib.get_templates(0)
    assert len(t0_list) >= 2, f"Expected multiple verified templates for digit 0, got {len(t0_list)}"
    
    # Digit 8 in presets has templates from both dial_0078 and dial_0080
    t8_list = lib.get_templates(8)
    assert len(t8_list) >= 2, f"Expected multiple verified templates for digit 8, got {len(t8_list)}"
    
    # Sources must be distinct
    sources_0 = [t.source for t in t0_list]
    assert len(set(sources_0)) == len(sources_0)


def test_regression_preset_and_raw_ground_truth():
    """Verify existing preset and raw image ground truth readings remain 100% intact."""
    pipeline = DialReadingPipeline()

    # Preset 0078
    p78 = PROJECT_ROOT / "data" / "presets" / "dial_0078.png"
    if p78.exists():
        r78 = pipeline.read_image(p78)
        assert r78["whole_digits"] == "0078"
        assert r78["value"] == "0078.0"
        assert r78["status"] == "ok"
        assert r78["confidence"] >= 0.50

    # Preset 0080
    p80 = PROJECT_ROOT / "data" / "presets" / "dial_0080.png"
    if p80.exists():
        r80 = pipeline.read_image(p80)
        assert r80["whole_digits"] == "0080"
        assert r80["value"] == "0080.0"
        assert r80["status"] == "ok"

    # Raw 80.2.png
    raw80 = PROJECT_ROOT / "data" / "raw" / "80.2.png"
    if raw80.exists():
        r_raw80 = pipeline.read_image(raw80)
        assert r_raw80["whole_digits"] == "0080"
        assert r_raw80["value"] == "0080.0"
        assert r_raw80["status"] == "ok"


def test_missing_digit_candidate_rejected_from_forced_match():
    """
    Verify that when candidate geometry matches a missing digit (e.g. narrow digit 1),
    it is NOT forced into an accepted match with 0, 7, or 8.
    """
    recognizer = PixelDigitRecognizer(auto_load_verified=True, project_root=PROJECT_ROOT)
    assert 1 in recognizer.templates.missing_digits()

    # Create a synthetic tall narrow numeral (resembling digit 1: asp ~ 0.30, 0 holes)
    narrow_img = Image.new("RGB", (60, 100), color=(20, 20, 20))
    draw = ImageDraw.Draw(narrow_img)
    draw.rectangle([25, 15, 35, 85], fill=(240, 240, 240))  # narrow vertical bar

    res = recognizer.predict_slot(narrow_img)
    # Must NOT be accepted as status='ok' for digit 7!
    assert res["status"] in ("uncertain", "missing_template", "ambiguous")


def test_partial_rolling_slot_rejected():
    """Verify that a partial horizontal sliver from a rolling drum is marked uncertain."""
    recognizer = PixelDigitRecognizer(auto_load_verified=True, project_root=PROJECT_ROOT)

    # Create a horizontal sliver (aspect ratio > 1.4, h=10, w=35)
    roll_img = Image.new("RGB", (60, 100), color=(20, 20, 20))
    draw = ImageDraw.Draw(roll_img)
    draw.rectangle([10, 10, 50, 22], fill=(240, 240, 240))

    res = recognizer.predict_slot(roll_img)
    assert res["status"] in ("uncertain", "no_digit")

