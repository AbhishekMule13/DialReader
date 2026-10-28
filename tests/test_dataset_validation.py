"""
Unit Tests for Dataset Validation and Labeling Safety Checks (Step 4C)

Verifies:
1. Malformed or invalid reading detection.
2. Missing raw image file detection.
3. Duplicate filename and duplicate pixel content detection.
4. Prevention of template-source flags on non-stationary, rolling, or unreadable samples.
5. Slot digit mismatch detection against 4-digit reading.
6. Target shortfall calculation for digits 0-9 (target >= 5 independent verified samples).
7. Non-destructive guarantee: validation never silently alters or deletes records.
8. Labeling workflow safety checks: repeated labeling updates existing rows rather than duplicating.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_validator import DatasetValidator, validate_labeled_dataset
from label_readings import save_reading, EXPECTED_COLS


# =============================================================================
# 1. Malformed & Invalid Reading Tests
# =============================================================================
def test_validation_detects_malformed_readings(tmp_path):
    """Verify validator catches malformed, non-numeric, or incorrect-length readings."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Create dummy image
    img_p = raw_dir / "test_dial.png"
    Image.new("RGB", (100, 50), color=(128, 128, 128)).save(img_p)

    # DataFrame with malformed readings
    df = pd.DataFrame([
        {"filename": "test_dial.png", "reading": "078", "condition": "stationary", "is_template_source": False},  # 3 digits
        {"filename": "test_dial.png", "reading": "00789", "condition": "stationary", "is_template_source": False},  # 5 digits
        {"filename": "test_dial.png", "reading": "007a", "condition": "stationary", "is_template_source": False},  # Non-numeric
    ])

    validator = DatasetValidator(raw_dir=raw_dir)
    report = validator.validate(df_override=df)

    assert not report.is_valid
    assert len(report.errors) >= 3
    error_types = [issue.category for issue in report.issues if issue.severity == "ERROR"]
    assert "malformed_reading" in error_types


# =============================================================================
# 2. Missing Image File Detection
# =============================================================================
def test_validation_detects_missing_image_file(tmp_path):
    """Verify validator catches records referencing image files that do not exist on disk."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    df = pd.DataFrame([
        {"filename": "non_existent_image.jpg", "reading": "0078", "condition": "stationary", "is_template_source": False}
    ])

    validator = DatasetValidator(raw_dir=raw_dir)
    report = validator.validate(df_override=df)

    assert not report.is_valid
    assert any("missing" in err.lower() for err in report.errors)
    assert any(issue.category == "image_missing" for issue in report.issues)


# =============================================================================
# 3. Duplicate Filename and Duplicate Content Detection
# =============================================================================
def test_validation_detects_duplicate_filenames(tmp_path):
    """Verify validator flags duplicate filenames in the dataset."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    img_p = raw_dir / "dup_dial.png"
    Image.new("RGB", (100, 50), color=(100, 100, 100)).save(img_p)

    df = pd.DataFrame([
        {"filename": "dup_dial.png", "reading": "0080", "condition": "stationary", "is_template_source": False},
        {"filename": "dup_dial.png", "reading": "0080", "condition": "stationary", "is_template_source": False},
    ])

    validator = DatasetValidator(raw_dir=raw_dir)
    report = validator.validate(df_override=df)

    assert not report.is_valid
    assert any(issue.category == "duplicate_filename" for issue in report.issues)


def test_validation_detects_duplicate_pixel_content(tmp_path):
    """Verify validator detects distinct filenames that share identical pixel data."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Create two different filenames with bit-identical image pixels
    base_img = Image.new("RGB", (80, 40), color=(50, 150, 200))
    base_img.save(raw_dir / "file_a.png")
    base_img.save(raw_dir / "file_b.png")

    df = pd.DataFrame([
        {"filename": "file_a.png", "reading": "0080", "condition": "stationary", "is_template_source": False},
        {"filename": "file_b.png", "reading": "0080", "condition": "stationary", "is_template_source": False},
    ])

    validator = DatasetValidator(raw_dir=raw_dir)
    report = validator.validate(df_override=df)

    # Duplicate content generates a warning (not fatal error, but must report duplicate group)
    assert len(report.duplicate_groups) == 1
    assert any(issue.category == "duplicate_content" for issue in report.issues)


# =============================================================================
# 4. Template Source Safety Flags
# =============================================================================
def test_validation_rejects_template_source_on_non_stationary(tmp_path):
    """Verify is_template_source=True is strictly rejected on rolling, reflective, or unreadable samples."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    img_p = raw_dir / "roll_dial.png"
    Image.new("RGB", (100, 50), color=(80, 80, 80)).save(img_p)

    df = pd.DataFrame([
        {"filename": "roll_dial.png", "reading": "0080", "condition": "rolling", "is_template_source": True},
        {"filename": "roll_dial.png", "reading": "0080", "condition": "reflective", "is_template_source": True},
        {"filename": "roll_dial.png", "reading": "unreadable", "condition": "unreadable", "is_template_source": True},
    ])

    validator = DatasetValidator(raw_dir=raw_dir)
    report = validator.validate(df_override=df)

    assert not report.is_valid
    template_errs = [issue for issue in report.issues if issue.category in ("template_not_stationary", "invalid_template_flag")]
    assert len(template_errs) >= 2


# =============================================================================
# 5. Slot Digit Mismatch Detection
# =============================================================================
def test_validation_detects_slot_digit_mismatches(tmp_path):
    """Verify validator flags mismatches between individual slot digits and the 4-digit reading."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    img_p = raw_dir / "slot_mismatch.png"
    Image.new("RGB", (100, 50), color=(90, 90, 90)).save(img_p)

    df = pd.DataFrame([
        {
            "filename": "slot_mismatch.png",
            "reading": "0078",
            "digit_0": 0,
            "digit_1": 0,
            "digit_2": 9,  # Mismatch: reading has 7
            "digit_3": 8,
            "condition": "stationary",
            "is_template_source": False
        }
    ])

    validator = DatasetValidator(raw_dir=raw_dir)
    report = validator.validate(df_override=df)

    assert not report.is_valid
    assert any(issue.category == "slot_mismatch" for issue in report.issues)


# =============================================================================
# 6. Class Balance and Shortfall Reporting
# =============================================================================
def test_validation_reports_accurate_shortfalls(tmp_path):
    """Verify validator accurately tracks independent verified counts and reports shortfalls against target (5)."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    img_p = raw_dir / "dial_valid.png"
    Image.new("RGB", (100, 50), color=(120, 120, 120)).save(img_p)

    # Valid reading 0078 gives independent counts: digit 0: 1 (counts once per image), digit 7: 1, digit 8: 1
    df = pd.DataFrame([
        {
            "filename": "dial_valid.png",
            "reading": "0078",
            "digit_0": 0,
            "digit_1": 0,
            "digit_2": 7,
            "digit_3": 8,
            "condition": "stationary",
            "is_template_source": True
        }
    ])

    validator = DatasetValidator(raw_dir=raw_dir, target_per_class=5)
    report = validator.validate(df_override=df)

    assert report.is_valid
    assert report.class_counts[0] == 1
    assert report.class_counts[7] == 1
    assert report.class_counts[8] == 1
    assert report.class_counts[1] == 0

    assert report.shortfalls[0] == 4
    assert report.shortfalls[7] == 4
    assert report.shortfalls[8] == 4
    assert report.shortfalls[1] == 5


# =============================================================================
# 7. Non-Destructive Guarantee
# =============================================================================
def test_validator_does_not_alter_data(tmp_path):
    """Verify validator operates in read-only mode and never mutates or drops records."""
    csv_file = tmp_path / "readings.csv"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    orig_content = "filename,reading,condition\nmissing.png,0078,stationary\n"
    csv_file.write_text(orig_content, encoding="utf-8")

    validator = DatasetValidator(readings_csv=csv_file, raw_dir=raw_dir)
    report = validator.validate()

    assert not report.is_valid
    # Verify file on disk is strictly unchanged
    assert csv_file.read_text(encoding="utf-8") == orig_content


# =============================================================================
# 8. Labeling Workflow Safety Checks
# =============================================================================
def test_save_reading_deduplicates_and_updates_in_place(tmp_path, monkeypatch):
    """Verify save_reading updates existing filename records in-place rather than duplicating."""
    csv_file = tmp_path / "readings.csv"
    monkeypatch.setattr("label_readings.READINGS_CSV", csv_file)

    # Initial save
    save_reading(filename="dial_01.png", reading="0070", condition="stationary")
    df1 = pd.read_csv(csv_file)
    assert len(df1) == 1
    assert df1.iloc[0]["reading"] == 70 or str(df1.iloc[0]["reading"]) == "0070"

    # Repeated labeling of same filename with corrected reading
    save_reading(filename="dial_01.png", reading="0078", condition="stationary")
    df2 = pd.read_csv(csv_file)
    assert len(df2) == 1  # Must NOT append a second row
    assert str(df2.iloc[0]["reading"]).zfill(4) == "0078"


def test_save_reading_forces_non_template_on_rolling(tmp_path, monkeypatch):
    """Verify save_reading safety gate automatically forces is_template_source=False if condition is rolling."""
    csv_file = tmp_path / "readings.csv"
    monkeypatch.setattr("label_readings.READINGS_CSV", csv_file)

    save_reading(
        filename="dial_roll.png",
        reading="0078",
        condition="rolling",
        is_template_source=True  # Intentionally attempt to flag rolling image as template
    )

    df = pd.read_csv(csv_file)
    assert len(df) == 1
    # Safety gate must have overridden is_template_source to False
    assert bool(df.iloc[0]["is_template_source"]) is False


# =============================================================================
# 9. Step 4C.1 Queue Discovery and Review Warnings Tests
# =============================================================================
def test_queue_discovers_all_raw_images_including_nested():
    """Verify load_dataset_and_progress discovers all 153 image files under data/raw including nested dirs."""
    from label_readings import load_dataset_and_progress
    eligible_df, _ = load_dataset_and_progress()

    assert len(eligible_df) == 153
    # Check that all nested directories are represented
    filenames = eligible_df["filename"].tolist()
    assert any(fn.startswith("knife/") for fn in filenames)
    assert any(fn.startswith("knife_pressing/") for fn in filenames)
    assert any(fn.startswith("knife_video_frames/") for fn in filenames)
    assert "80.2.png" in filenames
    assert "80.jpeg" in filenames


def test_queue_tags_review_warnings_for_risk_categories():
    """Verify collages, video frames, and negatives receive review warnings and safe default conditions."""
    from label_readings import load_dataset_and_progress
    eligible_df, _ = load_dataset_and_progress()

    fn_map = {row["filename"]: row for _, row in eligible_df.iterrows()}

    # Negative image
    neg_row = fn_map.get("knife_pressing/116.jpeg")
    assert neg_row is not None
    assert "negative" in neg_row["review_warning"].lower()
    assert neg_row["default_condition"] == "unreadable"

    # Sequential video frame
    vid_row = fn_map.get("knife_video_frames/knife_vid_f00000.jpg")
    assert vid_row is not None
    assert "video frame" in vid_row["review_warning"].lower() or "rolling" in vid_row["review_warning"].lower()
    assert vid_row["default_condition"] == "rolling"

    # Multi-dial collage
    col_row = fn_map.get("knife_pressing/103,104.jpeg")
    assert col_row is not None
    assert "collage" in col_row["review_warning"].lower()


def test_slot_level_vs_image_level_diversity_distinction(tmp_path):
    """
    Verify validator distinguishes between slot-level examples and image-level diversity:
    Reading '0080' has three '0' slots, but represents only 1 unique image for digit '0'.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    img_p = raw_dir / "dial_0080_test.png"
    Image.new("RGB", (100, 50), color=(100, 100, 100)).save(img_p)

    df = pd.DataFrame([
        {
            "filename": "dial_0080_test.png",
            "reading": "0080",
            "digit_0": 0,
            "digit_1": 0,
            "digit_2": 8,
            "digit_3": 0,
            "condition": "stationary",
            "is_template_source": True
        }
    ])

    validator = DatasetValidator(raw_dir=raw_dir)
    report = validator.validate(df_override=df)

    assert report.is_valid
    # Slot-level count for digit 0 must be 3
    assert report.human_slot_counts[0] == 3
    # Image-level diversity for digit 0 must be 1 (single unique photograph)
    assert report.class_counts[0] == 1
    # Shortfall for digit 0 against target (5) is 5 - 1 = 4 images
    assert report.shortfalls[0] == 4


def test_authoritative_accounting_separates_readings_csv_and_recognizer_templates():
    """Verify report separates human readings in readings.csv from active recognizer preset templates."""
    validator = DatasetValidator()
    report = validator.validate()

    # readings.csv tracks human-confirmed readings independently
    assert report.confirmed_full_readings >= 1
    assert report.verified_stationary_sources >= 1
    assert report.class_counts[0] >= 1
    assert report.human_slot_counts[0] >= 1

    # Recognizer active production template bank tracks total loaded templates (presets + verified file)
    assert report.recognizer_template_counts[0] >= 5
    assert report.recognizer_source_images[0] == 2
    assert report.recognizer_template_counts[7] >= 1
    assert report.recognizer_template_counts[8] >= 2
    assert report.unique_physical_images == 154

    # Separation test: verify with unconfirmed/empty dataframe that human counts drop to 0 while recognizer bank remains intact
    df_unconfirmed = pd.DataFrame([
        {"filename": "knife/20261006_083755899_iOS.jpg", "reading": "unreadable", "condition": "unreadable", "is_template_source": False}
    ])
    rep_unconfirmed = validator.validate(df_override=df_unconfirmed)
    assert rep_unconfirmed.confirmed_full_readings == 0
    assert rep_unconfirmed.verified_stationary_sources == 0
    assert rep_unconfirmed.class_counts[0] == 0
    assert rep_unconfirmed.recognizer_template_counts[0] >= 5

