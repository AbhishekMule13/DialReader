"""
DialReader Step 4F Automatic Reading Test Suite

Verifies:
1. Automated image upload/inference workflow accepting in-memory PIL Image, JPG, and PNG streams.
2. Complete pipeline execution:
     - Aperture detection
     - Rectification & deskew
     - Four-slot dynamic segmentation
     - Pixel-mask extraction & geometric feature analysis
     - Normalized template matching
     - Method 2 continuous decimal estimation
3. Full output contract:
     - Uploaded image / overlay
     - Detected display boundary
     - Four extracted digit slots
     - Predicted reading, confidence, status, and clear diagnostic reason
4. Zero manual digit entry requirement during normal automated inference.
5. Rejection of unsupported digits and clear reporting of missing templates (no false forced matches).
6. Clear explanatory reason when reading is uncertain, unreadable, or no window is found.
7. Optional correction workflow records ground truth safely without altering production templates.
8. Preservation of all canonical regression fixtures:
     - data/presets/dial_0078.png -> 0078.0
     - data/presets/dial_0080.png -> 0080.0
     - data/raw/80.2.png -> 0080.0
"""

import sys
import io
import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image, ImageDraw

# Ensure dial-reader root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import DialReadingPipeline
from src.digits import PixelDigitRecognizer, DigitTemplateLibrary
from label_readings import save_reading


# =============================================================================
# 1. Automated Inference from In-Memory Uploaded Images (JPG, JPEG, PNG)
# =============================================================================
def test_automated_reading_from_in_memory_png_upload():
    """Verify that an in-memory PNG upload (BytesIO stream) is automatically read without manual entry."""
    pipeline = DialReadingPipeline()
    preset_path = PROJECT_ROOT / "data" / "presets" / "dial_0078.png"
    if not preset_path.exists():
        pytest.skip(f"Preset image not found: {preset_path}")

    # Simulate in-memory file upload
    with open(preset_path, "rb") as f:
        file_bytes = f.read()
    uploaded_stream = io.BytesIO(file_bytes)
    in_memory_pil = Image.open(uploaded_stream)

    # Run automated inference
    res = pipeline.read_image(in_memory_pil)

    assert res["status"] == "ok"
    assert res["whole_digits"] == "0078"
    assert res["tenth"] == 0
    assert res["value"] == "0078.0"
    assert res["confidence"] >= 0.50
    assert "reason" in res
    assert len(res["digit_slots"]) == 4
    assert res["detected_overlay"] is not None


def test_automated_reading_from_in_memory_jpeg_upload():
    """Verify that an in-memory JPEG upload is automatically read without manual entry."""
    pipeline = DialReadingPipeline()
    preset_path = PROJECT_ROOT / "data" / "presets" / "dial_0080.png"
    if not preset_path.exists():
        pytest.skip(f"Preset image not found: {preset_path}")

    # Re-encode to JPEG in memory to simulate a smartphone JPEG upload
    orig_pil = Image.open(preset_path).convert("RGB")
    jpeg_buf = io.BytesIO()
    orig_pil.save(jpeg_buf, format="JPEG", quality=95)
    jpeg_buf.seek(0)
    uploaded_jpeg = Image.open(jpeg_buf)

    # Run automated inference
    res = pipeline.read_image(uploaded_jpeg)

    assert res["status"] == "ok"
    assert res["whole_digits"] == "0080"
    assert res["tenth"] == 0
    assert res["value"] == "0080.0"
    assert len(res["digit_slots"]) == 4


# =============================================================================
# 2. Automated Pipeline Output Contract
# =============================================================================
def test_automated_pipeline_output_contract():
    """
    Verify pipeline automatically produces all required visual and diagnostic components:
    - Detected boundary overlay
    - Rectified display window
    - Four discrete slot crops
    - Status, confidence, and explanatory reason
    - Method 2 decimal measurement details
    """
    pipeline = DialReadingPipeline()
    preset_path = PROJECT_ROOT / "data" / "presets" / "dial_0078.png"
    if not preset_path.exists():
        pytest.skip(f"Preset image not found: {preset_path}")

    res = pipeline.read_image(preset_path)

    # 1. Visual components
    assert isinstance(res["detected_overlay"], Image.Image)
    assert isinstance(res["rectified_window"], Image.Image)
    assert len(res["digit_slots"]) == 4
    for slot in res["digit_slots"]:
        assert isinstance(slot, Image.Image)

    # 2. Reading and confidence
    assert isinstance(res["value"], str)
    assert isinstance(res["confidence"], float)
    assert 0.0 <= res["confidence"] <= 1.0

    # 3. Status and reason
    assert res["status"] in ("ok", "uncertain", "unreadable", "no_window")
    assert isinstance(res["reason"], str)
    assert len(res["reason"]) > 0

    # 4. Method 2 decimal info
    assert "decimal_info" in res
    assert "displacement_px" in res["decimal_info"]
    assert "baseline_y" in res["decimal_info"]
    assert "tenth" in res


# =============================================================================
# 3. Unsupported Digits Handled Safely with Clear Reason
# =============================================================================
def test_unsupported_digits_rejected_with_diagnostic_reason():
    """
    Verify that slots containing numerals from unsupported digits (e.g. digit 1, 4)
    are NOT forced to available digits (0, 7, 8) and clearly report status='uncertain'
    with diagnostic explanation.
    """
    recognizer = PixelDigitRecognizer(auto_load_verified=True, project_root=PROJECT_ROOT)
    missing = recognizer.templates.missing_digits()
    assert 1 in missing, "Digit 1 should lack verified templates in baseline"

    # Create slot resembling digit 1 (tall narrow vertical stroke, aspect < 0.45, 0 holes)
    slot_1 = Image.new("RGB", (60, 100), color=(20, 20, 20))
    draw = ImageDraw.Draw(slot_1)
    draw.rectangle([25, 12, 33, 88], fill=(245, 245, 245))

    pred = recognizer.predict_slot(slot_1)

    # Must be held as uncertain, not forced to 7
    assert pred["status"] in ("uncertain", "missing_template", "ambiguous")
    assert "missing digit 1" in pred["reason"].lower() or "score" in pred["reason"].lower()


# =============================================================================
# 4. Unreadable and Non-Dial Images Rejected with Clear Reason
# =============================================================================
def test_unreadable_and_non_dial_image_rejection():
    """Verify that non-dial images or unreadable dials return status='no_window' or 'unreadable' with clear reason."""
    pipeline = DialReadingPipeline()

    # 1. Pure blank non-dial image
    blank_img = Image.new("RGB", (640, 480), color=(30, 30, 30))
    res_blank = pipeline.read_image(blank_img)
    assert res_blank["status"] == "no_window"
    assert "no display window" in res_blank["reason"].lower()
    assert len(res_blank["digit_slots"]) == 0
    assert res_blank["detected_overlay"] is not None

    # 2. Known raw unreadable machine sample from dataset
    unreadable_raw = PROJECT_ROOT / "data" / "raw" / "knife" / "20261006_083755899_iOS.jpg"
    if unreadable_raw.exists():
        res_raw = pipeline.read_image(unreadable_raw)
        assert res_raw["status"] in ("unreadable", "uncertain", "no_window")
        assert "reason" in res_raw
        assert len(res_raw["reason"]) > 0


# =============================================================================
# 5. Optional Human Correction Workflow Safety
# =============================================================================
def test_optional_correction_workflow_does_not_pollute_production_templates(tmp_path, monkeypatch):
    """
    Verify that saving a human correction through the optional review workflow
    records to readings.csv with is_template_source=False and does NOT alter
    active production templates.
    """
    csv_file = tmp_path / "readings.csv"
    monkeypatch.setattr("label_readings.READINGS_CSV", csv_file)

    # Save an optional correction from UI
    save_reading(
        filename="test_manual_entry.png",
        reading="0078",
        tenth=".0",
        condition="stationary",
        is_template_source=False,
        notes="streamlit_manual_correction"
    )

    # Check file on disk
    import pandas as pd
    df = pd.read_csv(csv_file)
    assert len(df) == 1
    assert str(df.iloc[0]["reading"]).zfill(4) == "0078"
    assert bool(df.iloc[0]["is_template_source"]) is False

    # Production templates in recognizer remain untouched
    lib = DigitTemplateLibrary(canonical_size=(48, 32))
    lib.load_verified_preset_templates(project_root=PROJECT_ROOT)
    assert lib.available_digits() == [0, 7, 8]
    assert set(lib.missing_digits()) == {1, 2, 3, 4, 5, 6, 9}


# =============================================================================
# 6. Preservation of Canonical Regression Fixtures
# =============================================================================
def test_regression_fixtures_exact_values():
    """
    Verify existing regression fixtures strictly return exact expected readings:
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
        assert r78["reason"] == "Reading recognized with high confidence"

    # 2. dial_0080.png
    p80 = PROJECT_ROOT / "data" / "presets" / "dial_0080.png"
    if p80.exists():
        r80 = pipeline.read_image(p80)
        assert r80["whole_digits"] == "0080"
        assert r80["tenth"] == 0
        assert r80["value"] == "0080.0"
        assert r80["status"] == "ok"

    # 3. data/raw/80.2.png
    raw80 = PROJECT_ROOT / "data" / "raw" / "80.2.png"
    if raw80.exists():
        r_raw80 = pipeline.read_image(raw80)
        assert r_raw80["whole_digits"] == "0080"
        assert r_raw80["tenth"] == 0
        assert r_raw80["value"] == "0080.0"
        assert r_raw80["status"] == "ok"
