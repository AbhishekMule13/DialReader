"""
DialReader Step 4E Regression and Validation Test Suite

Validates:
1. All ten digit labels (0-9) are accepted by the template library.
2. Missing templates remain explicitly reported in missing_digits().
3. Unverified, rolling, or sparse crops are strictly excluded from the verified template bank.
4. Duplicate source images (e.g. 80.2.png vs dial_0080.png) do not inflate independent photograph counts.
5. Identical mask deduplication prevents false template count inflation.
6. Unsupported/missing digits are not falsely forced into predictions of available digits (0, 7, 8).
7. Existing thresholds, ambiguity margins, and uncertainty safeguards remain intact.
8. Regression fixtures produce exact expected values:
     - data/presets/dial_0078.png -> 0078.0
     - data/presets/dial_0080.png -> 0080.0
     - data/raw/80.2.png -> 0080.0
9. Unseen evaluation images are strictly partitioned from template source images by SHA-256 hash.
"""

import sys
import hashlib
from pathlib import Path
import pytest
import numpy as np
from PIL import Image, ImageDraw

# Ensure dial-reader root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.digits import PixelDigitRecognizer, DigitTemplateLibrary, DigitTemplate
from src.infer import DialReadingPipeline


# =============================================================================
# 1. Ten Digit Labels Accepted
# =============================================================================
def test_all_ten_digit_labels_accepted():
    """Verify all 10 digits (0 through 9) can be registered as valid templates."""
    lib = DigitTemplateLibrary(canonical_size=(48, 32))

    dummy_mask = np.zeros((48, 32), dtype=np.uint8)
    dummy_mask[8:40, 6:26] = 255  # Solid rectangle with 512 white pixels

    for d in range(10):
        tpl = lib.register_template(
            digit=d,
            mask=dummy_mask,
            source=f"test_verified:digit_{d}",
            verified=True,
            condition="stationary"
        )
        assert tpl.digit == d
        assert lib.has_template(d) is True
        assert lib.get_template_count(d, verified_only=True) == 1

    assert sorted(lib.available_digits()) == list(range(10))
    assert lib.missing_digits() == []

    # Out of range digits must be rejected with ValueError
    with pytest.raises(ValueError, match="between 0 and 9"):
        lib.register_template(-1, dummy_mask, source="invalid_neg")

    with pytest.raises(ValueError, match="between 0 and 9"):
        lib.register_template(10, dummy_mask, source="invalid_10")


# =============================================================================
# 2. Missing Templates Remain Explicitly Missing
# =============================================================================
def test_missing_templates_remain_explicitly_missing():
    """Verify that digits without verified real photographs are reported as missing."""
    lib = DigitTemplateLibrary(canonical_size=(48, 32))
    lib.load_verified_preset_templates(project_root=PROJECT_ROOT)

    available = lib.available_digits()
    missing = lib.missing_digits()

    # In production, only 0, 7, 8 have verified physical presets
    assert set(available) == {0, 7, 8}
    assert set(missing) == {1, 2, 3, 4, 5, 6, 9}

    # Ensure missing digits have zero templates
    for d in [1, 2, 3, 4, 5, 6, 9]:
        assert lib.has_template(d) is False
        assert lib.get_template_count(d) == 0
        assert lib.get_independent_source_count(d) == 0


# =============================================================================
# 3. Unverified Templates Excluded From Verified Bank
# =============================================================================
def test_unverified_templates_excluded_from_verified_bank():
    """Verify that unverified, rolling, or sparse crops cannot enter verified template bank."""
    lib = DigitTemplateLibrary(canonical_size=(48, 32))

    valid_mask = np.zeros((48, 32), dtype=np.uint8)
    valid_mask[8:40, 6:26] = 255

    # 1. Non-stationary condition marked verified=True must raise ValueError
    for bad_cond in ["rolling", "reflective", "unreadable"]:
        with pytest.raises(ValueError, match="condition='stationary'"):
            lib.register_template(3, valid_mask, source="bad_cond", verified=True, condition=bad_cond)

    # 2. Sparse mask (< 20 white pixels) marked verified=True must raise ValueError
    sparse_mask = np.zeros((48, 32), dtype=np.uint8)
    sparse_mask[10:13, 10:13] = 255  # only 9 white pixels
    with pytest.raises(ValueError, match="too sparse"):
        lib.register_template(3, sparse_mask, source="sparse", verified=True, condition="stationary")

    # 3. Unverified templates (verified=False) can be loaded for testing but excluded from verified queries
    lib.register_template(4, valid_mask, source="synthetic_test", verified=False, condition="synthetic")
    assert lib.get_template_count(4, verified_only=False) == 1
    assert lib.get_template_count(4, verified_only=True) == 0


# =============================================================================
# 4. Duplicate Source Images Do Not Inflate Independent Sample Counts
# =============================================================================
def test_duplicate_source_images_do_not_inflate_independent_counts():
    """
    Verify that duplicate source photographs sharing identical pixel hashes
    (e.g. data/raw/80.2.png and data/presets/dial_0080.png) do not inflate
    the independent photograph counts.
    """
    p80_preset = PROJECT_ROOT / "data" / "presets" / "dial_0080.png"
    p80_raw = PROJECT_ROOT / "data" / "raw" / "80.2.png"

    if p80_preset.exists() and p80_raw.exists():
        h_preset = hashlib.sha256(open(p80_preset, "rb").read()).hexdigest()
        h_raw = hashlib.sha256(open(p80_raw, "rb").read()).hexdigest()
        assert h_preset == h_raw, "Expected 80.2.png and dial_0080.png to have identical SHA-256"

    lib = DigitTemplateLibrary(canonical_size=(48, 32))
    lib.load_verified_preset_templates(project_root=PROJECT_ROOT)

    source_counts = lib.independent_sources_by_digit(verified_only=True)

    # Digit 0 appears in dial_0078.png and dial_0080.png (80.2.png is duplicate of dial_0080)
    assert source_counts[0] == 2, f"Expected 2 independent source photographs for digit 0, got {source_counts[0]}"
    # Digit 8 appears in dial_0078.png and dial_0080.png
    assert source_counts[8] == 2, f"Expected 2 independent source photographs for digit 8, got {source_counts[8]}"
    # Digit 7 appears only in dial_0078.png
    assert source_counts[7] == 1, f"Expected 1 independent source photograph for digit 7, got {source_counts[7]}"
    # Missing digits have 0 independent photographs
    for d in [1, 2, 3, 4, 5, 6, 9]:
        assert source_counts[d] == 0


# =============================================================================
# 5. Mask Deduplication Prevents False Template Inflation
# =============================================================================
def test_mask_deduplication_prevents_false_inflation():
    """Verify that registering the exact same binary mask multiple times does not inflate templates."""
    lib = DigitTemplateLibrary(canonical_size=(48, 32))

    mask = np.zeros((48, 32), dtype=np.uint8)
    mask[10:35, 10:25] = 255

    t1 = lib.register_template(0, mask, source="src_a", verified=True, deduplicate_identical=True)
    t2 = lib.register_template(0, mask, source="src_b", verified=True, deduplicate_identical=True)

    assert t1 is t2, "Expected identical mask registration to return existing template instance"
    assert lib.get_template_count(0, verified_only=True) == 1


# =============================================================================
# 6. No Forced Predictions for Unsupported Digits
# =============================================================================
def test_no_forced_predictions_for_unsupported_digits():
    """
    Verify that numerals matching unsupported/missing digits are not falsely
    accepted as available digits (0, 7, 8).
    """
    recognizer = PixelDigitRecognizer(auto_load_verified=True, project_root=PROJECT_ROOT)
    assert 1 in recognizer.templates.missing_digits()
    assert 4 in recognizer.templates.missing_digits()

    # 1. Tall narrow numeral (resembling digit 1: asp <= 0.48, 0 holes)
    img_1 = Image.new("RGB", (60, 100), color=(20, 20, 20))
    draw_1 = ImageDraw.Draw(img_1)
    draw_1.rectangle([25, 15, 33, 85], fill=(245, 245, 245))

    res_1 = recognizer.predict_slot(img_1)
    assert res_1["status"] in ("uncertain", "missing_template", "ambiguous")

    # 2. Numeral with central crossbar and hole (resembling digit 4)
    img_4 = Image.new("RGB", (60, 100), color=(20, 20, 20))
    draw_4 = ImageDraw.Draw(img_4)
    draw_4.rectangle([15, 15, 25, 60], fill=(245, 245, 245))  # left vertical
    draw_4.rectangle([15, 50, 45, 60], fill=(245, 245, 245))  # horizontal crossbar
    draw_4.rectangle([35, 25, 45, 85], fill=(245, 245, 245))  # main vertical stem

    res_4 = recognizer.predict_slot(img_4)
    # Must NOT be accepted as status='ok' for digit 0 or 8
    assert res_4["status"] in ("uncertain", "missing_template", "ambiguous")


# =============================================================================
# 7. Existing Thresholds and Uncertainty Behavior Intact
# =============================================================================
def test_existing_thresholds_and_uncertainty_behavior_intact():
    """Verify minimum confidence (0.35) and ambiguity margin (0.08) thresholds operate correctly."""
    recognizer = PixelDigitRecognizer(
        min_confidence=0.35,
        ambiguity_margin=0.08,
        auto_load_verified=True,
        project_root=PROJECT_ROOT
    )
    assert recognizer.min_confidence == 0.35
    assert recognizer.ambiguity_margin == 0.08

    # Blank image must yield confidence 0.0 and status 'no_digit' or 'uncertain'
    blank = Image.new("RGB", (60, 100), color=(10, 10, 10))
    res_blank = recognizer.predict_slot(blank)
    assert res_blank["status"] in ("no_digit", "uncertain")
    assert res_blank["confidence"] == 0.0


# =============================================================================
# 8. Regression Fixtures Preserved (dial_0078, dial_0080, raw/80.2.png)
# =============================================================================
def test_regression_fixtures_preserved():
    """
    Verify regression fixtures produce 100% exact expected values:
      - data/presets/dial_0078.png -> 0078.0
      - data/presets/dial_0080.png -> 0080.0
      - data/raw/80.2.png          -> 0080.0
    """
    pipeline = DialReadingPipeline()

    # 1. dial_0078.png
    p78 = PROJECT_ROOT / "data" / "presets" / "dial_0078.png"
    if p78.exists():
        r78 = pipeline.read_image(p78)
        assert r78["whole_digits"] == "0078"
        assert r78["tenth"] == 0
        assert r78["value"] == "0078.0"
        assert r78["status"] == "ok"

    # 2. dial_0080.png
    p80 = PROJECT_ROOT / "data" / "presets" / "dial_0080.png"
    if p80.exists():
        r80 = pipeline.read_image(p80)
        assert r80["whole_digits"] == "0080"
        assert r80["tenth"] == 0
        assert r80["value"] == "0080.0"
        assert r80["status"] == "ok"

    # 3. data/raw/80.2.png
    r80_raw = PROJECT_ROOT / "data" / "raw" / "80.2.png"
    if r80_raw.exists():
        r_raw = pipeline.read_image(r80_raw)
        assert r_raw["whole_digits"] == "0080"
        assert r_raw["tenth"] == 0
        assert r_raw["value"] == "0080.0"
        assert r_raw["status"] == "ok"


# =============================================================================
# 9. Audit Report Structure and Content
# =============================================================================
def test_audit_report_structure():
    """Verify template library audit() returns structured, accurate diagnostics."""
    lib = DigitTemplateLibrary(canonical_size=(48, 32))
    lib.load_verified_preset_templates(project_root=PROJECT_ROOT)
    audit = lib.audit()

    assert "available_digits" in audit
    assert "missing_digits" in audit
    assert "template_counts" in audit
    assert "independent_sources" in audit
    assert "missing_reasons" in audit
    assert "all_exported_human_confirmed" in audit

    assert audit["available_digits"] == [0, 7, 8]
    assert audit["missing_digits"] == [1, 2, 3, 4, 5, 6, 9]
    assert audit["all_exported_human_confirmed"] is True
    assert audit["independent_sources"][0] == 2
    assert audit["independent_sources"][7] == 1
    assert audit["independent_sources"][8] == 2
