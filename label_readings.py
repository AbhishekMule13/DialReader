"""
DialReader Interactive Reading Labeling Tool (Step 1)
Enables ground-truth labeling of 4-digit mechanical dial counter readings.

Usage:
    Streamlit Web UI (Recommended):
        streamlit run label_readings.py --server.port 8503

    OpenCV / CLI Terminal Mode:
        python label_readings.py --opencv
        python label_readings.py --cli
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict, Any, List
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Ensure dial-reader root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LABELS_CSV = PROJECT_ROOT / "data" / "labeled" / "labels.csv"
READINGS_CSV = PROJECT_ROOT / "data" / "labeled" / "readings.csv"
VERIFIED_TEMPLATES_JSON = PROJECT_ROOT / "data" / "labeled" / "verified_templates.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_COLS = [
    "filename", "reading", "tenth_optional", "condition",
    "digit_0", "digit_1", "digit_2", "digit_3",
    "is_template_source", "labeled_at", "notes"
]

from src.infer import DialReadingPipeline
from src.crop import WindowNormalizer
from src.digits import PixelDigitRecognizer


def load_dataset_and_progress():
    """Loads all image files from labels.csv and data/raw, tagging review warnings without excluding."""
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"labels.csv not found at {LABELS_CSV}")
        
    df_labels = pd.read_csv(LABELS_CSV)
    
    # Deduplicate filenames in labels.csv to prevent queue repetition (e.g. knife_pressing/103,104.jpeg)
    eligible = df_labels.drop_duplicates(subset=["filename"]).copy()
    
    # Also discover any unlisted real images in data/raw (e.g. 80.jpeg, 80.2.png, newly added dial photos)
    if RAW_DIR.exists():
        existing_filenames = set(df_labels["filename"].dropna().values)
        unlisted = []
        for p in sorted(RAW_DIR.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                rel_fn = p.relative_to(RAW_DIR).as_posix()
                if rel_fn not in existing_filenames:
                    unlisted.append({
                        "filename": rel_fn,
                        "bbox": "",
                        "reading": "",
                        "orientation": 0,
                        "mid_roll": False,
                        "dial_variant": "standard_3black_1red",
                        "split_group": "raw"
                    })
        if unlisted:
            eligible = pd.concat([eligible, pd.DataFrame(unlisted)], ignore_index=True)

    # Classify review warnings and suggested condition defaults
    def tag_image_review_warning(row):
        fn = str(row.get("filename", "")).strip()
        dv = str(row.get("dial_variant", "")).strip().lower()
        mid_roll = bool(row.get("mid_roll", False))
        
        if dv == "non_dial_negative" or "negative" in str(row.get("reading", "")).lower():
            return "Non-dial negative sample (casing / room background). Default condition: unreadable.", "unreadable"
        elif fn.startswith("knife_video_frames/") or mid_roll:
            return "Sequential video frame (cutter in motion / likely rolling drum). Default condition: rolling.", "rolling"
        elif "," in fn:
            return "Multi-dial collage (image contains multiple dials / crops). Verify boundaries carefully.", "stationary"
        return "", "stationary"

    warning_info = eligible.apply(tag_image_review_warning, axis=1)
    eligible["review_warning"] = [w[0] for w in warning_info]
    eligible["default_condition"] = [w[1] for w in warning_info]
    
    # Initialize readings.csv if missing
    if not READINGS_CSV.exists():
        READINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_readings = pd.DataFrame(columns=EXPECTED_COLS)
        df_readings.to_csv(READINGS_CSV, index=False)
    else:
        df_readings = pd.read_csv(READINGS_CSV, dtype=str)
        # Ensure all expected columns are present
        for c in EXPECTED_COLS:
            if c not in df_readings.columns:
                if c == "condition":
                    df_readings[c] = df_readings.apply(
                        lambda r: "unreadable" if str(r.get("reading", "")).strip().lower() in ("unreadable", "skipped") or "unreadable" in str(r.get("notes", "")).lower() else "stationary",
                        axis=1
                    )
                elif c == "is_template_source":
                    df_readings[c] = False
                else:
                    df_readings[c] = ""
                    
    return eligible, df_readings


def save_reading(
    filename: str,
    reading: str,
    tenth: str = "",
    condition: str = "stationary",
    digit_0: Optional[Union[int, str]] = None,
    digit_1: Optional[Union[int, str]] = None,
    digit_2: Optional[Union[int, str]] = None,
    digit_3: Optional[Union[int, str]] = None,
    is_template_source: bool = False,
    notes: str = ""
):
    """Appends or updates a verified reading with full condition and slot metadata."""
    READINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if READINGS_CSV.exists():
        df_readings = pd.read_csv(READINGS_CSV, dtype=str)
    else:
        df_readings = pd.DataFrame(columns=EXPECTED_COLS)
        
    for c in EXPECTED_COLS:
        if c not in df_readings.columns:
            df_readings[c] = ""
            
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tenth_val = str(tenth).strip() if str(tenth).strip() else ""
    
    # Derive slot digits if reading is 4 numeric digits and individual slots not provided
    clean_r = str(reading).strip()
    if len(clean_r) == 4 and clean_r.isdigit():
        d0 = int(clean_r[0]) if digit_0 is None or digit_0 == "" else int(digit_0)
        d1 = int(clean_r[1]) if digit_1 is None or digit_1 == "" else int(digit_1)
        d2 = int(clean_r[2]) if digit_2 is None or digit_2 == "" else int(digit_2)
        d3 = int(clean_r[3]) if digit_3 is None or digit_3 == "" else int(digit_3)
    else:
        d0, d1, d2, d3 = "", "", "", ""
        
    # Safety gate: template sources MUST be stationary and non-rolling
    if condition != "stationary":
        is_template_source = False
        
    row_data = {
        "filename": filename,
        "reading": clean_r,
        "tenth_optional": tenth_val,
        "condition": condition,
        "digit_0": d0,
        "digit_1": d1,
        "digit_2": d2,
        "digit_3": d3,
        "is_template_source": bool(is_template_source),
        "labeled_at": now_str,
        "notes": notes
    }
    
    if filename in df_readings["filename"].values:
        idx = df_readings.index[df_readings["filename"] == filename][0]
        for k, v in row_data.items():
            df_readings.at[idx, k] = v
    else:
        new_row = pd.DataFrame([row_data])
        df_readings = pd.concat([df_readings, new_row], ignore_index=True)
        
    df_readings.to_csv(READINGS_CSV, index=False)


def export_verified_templates(output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Exports verified real digit templates from readings.csv to verified_templates.json.
    Strictly gates quality:
      - condition must be 'stationary'
      - is_template_source must be True
      - reading must be valid 4 digits
      - each slot must have a non-empty, non-rolling binary mask
      - accumulates multiple real examples across runs without overwriting
    """
    if output_path is None:
        output_path = VERIFIED_TEMPLATES_JSON
        
    _, df = load_dataset_and_progress()
        
    # Filter candidate template sources (support both boolean True and string 'True')
    is_tpl_bool = df["is_template_source"].astype(str).str.strip().str.lower().isin(["true", "1"])
    eligible = df[
        is_tpl_bool &
        (df["condition"] == "stationary") &
        (df["reading"].notnull()) &
        (df["reading"].astype(str).str.len() == 4) &
        (df["reading"].astype(str).str.isdigit())
    ].copy()
    
    print(f"Eligible verified stationary template sources in readings.csv: {len(eligible)}")
    if len(eligible) == 0:
        print("No verified stationary template sources found in readings.csv.")
        return {"status": "empty", "exported": 0, "counts": {d: 0 for d in range(10)}}
        
    normalizer = WindowNormalizer()
    pipeline = DialReadingPipeline()
    
    # Load any pre-existing verified templates to accumulate without overwriting
    templates_by_source: Dict[str, Dict[str, Any]] = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                prev_items = json.load(f)
                if isinstance(prev_items, list):
                    for item in prev_items:
                        if isinstance(item, dict) and "source" in item:
                            templates_by_source[item["source"]] = item
        except Exception as e:
            print(f"  Warning: Could not read existing templates from {output_path}: {e}")

    import hashlib
    seen_image_hashes: Set[str] = set()

    for idx, row in eligible.iterrows():
        fn = row["filename"]
        raw_p = RAW_DIR / fn
        if not raw_p.exists():
            print(f"  Warning: Raw image not found: {raw_p}")
            continue

        # Prevent duplicate image content from exporting redundant templates
        try:
            with open(raw_p, "rb") as f:
                img_sha256 = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            img_sha256 = fn

        if img_sha256 in seen_image_hashes:
            print(f"  Notice: Skipping duplicate image content for '{fn}' (matches already exported photograph).")
            continue
        seen_image_hashes.add(img_sha256)

        try:
            res = pipeline.read_image(raw_p)
            slots = res.get("digit_slots", [])
            if len(slots) != 4:
                print(f"  Warning: {fn} did not yield 4 slots (got {len(slots)})")
                continue

            clean_r = str(row["reading"]).strip()
            d0 = int(row["digit_0"]) if pd.notnull(row.get("digit_0")) and str(row["digit_0"]).strip().isdigit() else int(clean_r[0])
            d1 = int(row["digit_1"]) if pd.notnull(row.get("digit_1")) and str(row["digit_1"]).strip().isdigit() else int(clean_r[1])
            d2 = int(row["digit_2"]) if pd.notnull(row.get("digit_2")) and str(row["digit_2"]).strip().isdigit() else int(clean_r[2])
            d3 = int(row["digit_3"]) if pd.notnull(row.get("digit_3")) and str(row["digit_3"]).strip().isdigit() else int(clean_r[3])
            slot_digits = [d0, d1, d2, d3]

            for s_idx in range(4):
                digit_val = slot_digits[s_idx]
                slot_crop = slots[s_idx]
                mask, feats = PixelDigitRecognizer.extract_normalized_digit_mask(slot_crop)

                if mask is None:
                    continue
                white_px = int(np.sum(mask > 0))
                asp = feats.get("aspect_ratio", 1.0)

                # Strict safety filters: exclude sparse, clipped, or rolling strokes
                if white_px < 20 or asp >= 1.30:
                    continue

                src_id = f"verified_real:{fn}:slot_{s_idx}:digit_{digit_val}"
                templates_by_source[src_id] = {
                    "digit": int(digit_val),
                    "source": src_id,
                    "filename": fn,
                    "slot_index": s_idx,
                    "condition": "stationary",
                    "verified": True,
                    "image_hash": img_sha256[:16],
                    "features": feats,
                    "mask": mask.tolist()
                }
        except Exception as e:
            print(f"  Error processing {fn}: {e}")

    exported_templates = list(templates_by_source.values())
    counts_by_digit = {d: 0 for d in range(10)}
    sources_by_digit = {d: set() for d in range(10)}

    for t in exported_templates:
        d = int(t["digit"])
        counts_by_digit[d] += 1
        sources_by_digit[d].add(t.get("image_hash", t.get("filename", "")))

    independent_source_counts = {d: len(sources_by_digit[d]) for d in range(10)}

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(exported_templates, f, indent=2)

    print(f"Successfully exported {len(exported_templates)} verified real templates to {output_path}")
    print(f"Counts per digit:           {counts_by_digit}")
    print(f"Independent source photos: {independent_source_counts}")
    missing = [d for d, c in counts_by_digit.items() if c == 0]
    print(f"Missing digits in exported real bank: {missing}")

    return {
        "status": "success",
        "exported": len(exported_templates),
        "counts": counts_by_digit,
        "independent_sources": independent_source_counts,
        "missing_digits": missing
    }


# ==============================================================================
# STREAMLIT UI IMPLEMENTATION
# ==============================================================================
def run_streamlit_app():
    import streamlit as st
    
    st.set_page_config(
        page_title="DialReader - Reading Labeler",
        page_icon="🏷️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS Styling
    st.markdown("""
    <style>
        .label-header { font-size: 1.8rem; font-weight: 700; color: #ff7b00; margin-bottom: 2px; }
        .label-sub { color: #8b949e; font-size: 0.95rem; margin-bottom: 16px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
        .slot-title { text-align: center; color: #58a6ff; font-weight: 600; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)
    
    eligible_df, readings_df = load_dataset_and_progress()
    labeled_filenames = set(readings_df[readings_df["reading"].notnull()]["filename"].values)
    
    st.sidebar.markdown("### 🏷️ Labeling Progress")
    total_count = len(eligible_df)
    labeled_count = len(labeled_filenames)
    progress_pct = labeled_count / max(1, total_count)
    st.sidebar.progress(progress_pct)
    st.sidebar.markdown(f"**Labeled:** `{labeled_count} / {total_count}` ({progress_pct:.1%})")
    st.sidebar.markdown(f"**Remaining:** `{total_count - labeled_count}`")
    
    # Filter mode
    view_mode = st.sidebar.radio(
        "Filter queue:",
        ["Unlabeled only (Sequential)", "All images", "Reviewed / Labeled only"],
        index=0
    )
    
    if view_mode == "Unlabeled only (Sequential)":
        queue_df = eligible_df[~eligible_df["filename"].isin(labeled_filenames)].reset_index(drop=True)
    elif view_mode == "Reviewed / Labeled only":
        queue_df = eligible_df[eligible_df["filename"].isin(labeled_filenames)].reset_index(drop=True)
    else:
        queue_df = eligible_df.reset_index(drop=True)
        
    if len(queue_df) == 0:
        st.success("🎉 All eligible images have been labeled! Check `data/labeled/readings.csv`.")
        st.dataframe(readings_df)
        return
        
    # Navigation state
    if "current_queue_idx" not in st.session_state:
        st.session_state.current_queue_idx = 0
        
    if st.session_state.current_queue_idx >= len(queue_df):
        st.session_state.current_queue_idx = 0
        
    curr_idx = st.session_state.current_queue_idx
    row = queue_df.iloc[curr_idx]
    filename = row["filename"]
    raw_path = RAW_DIR / filename
    ori = int(row["orientation"]) if pd.notnull(row["orientation"]) else None
    split_grp = str(row.get("split_group", "unknown"))
    review_warn = str(row.get("review_warning", ""))
    default_cond = str(row.get("default_condition", "stationary"))
    
    # Previous labeling info if exists
    existing_reading = ""
    existing_tenth = ""
    if filename in readings_df["filename"].values:
        sub = readings_df[readings_df["filename"] == filename].iloc[0]
        existing_reading = str(sub.get("reading", ""))
        existing_tenth = str(sub.get("tenth_optional", ""))
        existing_d0 = str(sub.get("digit_0", "")) if pd.notnull(sub.get("digit_0")) and str(sub.get("digit_0", "")).strip() != "nan" else ""
        existing_d1 = str(sub.get("digit_1", "")) if pd.notnull(sub.get("digit_1")) and str(sub.get("digit_1", "")).strip() != "nan" else ""
        existing_d2 = str(sub.get("digit_2", "")) if pd.notnull(sub.get("digit_2")) and str(sub.get("digit_2", "")).strip() != "nan" else ""
        existing_d3 = str(sub.get("digit_3", "")) if pd.notnull(sub.get("digit_3")) and str(sub.get("digit_3", "")).strip() != "nan" else ""
    else:
        existing_d0, existing_d1, existing_d2, existing_d3 = "", "", "", ""

    st.markdown('<div class="label-header">⏱️ DialReader Ground-Truth Labeler</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="label-sub">Image <b>{curr_idx + 1} of {len(queue_df)}</b> in queue &nbsp;|&nbsp; File: <code>{filename}</code> &nbsp;|&nbsp; Split: <code>{split_grp}</code></div>', unsafe_allow_html=True)
    if review_warn:
        st.warning(f"⚠️ **Review Notice:** {review_warn}")
    
    # Load and process image
    @st.cache_resource
    def get_pipeline():
        return DialReadingPipeline()
        
    pipeline = get_pipeline()
    
    if not raw_path.exists():
        st.error(f"Image file not found: {raw_path}")
        return
        
    # Run window crop & slots extraction
    with st.spinner("Extracting rectified window and slots..."):
        pipe_res = pipeline.read_image(raw_path, manual_orientation=ori)
        
    rectified_pil = pipe_res.get("rectified_window")
    slots = pipe_res.get("digit_slots", [])
    model_reading = pipe_res.get("whole_digits") or "None"
    model_conf = pipe_res.get("confidence", 0.0)
    model_tenth = pipe_res.get("tenth", "")
    overlay_pil = pipe_res.get("detected_overlay")
    status = pipe_res.get("status", "ok")
    slot_ranges = pipe_res.get("slot_ranges", [])

    if status == "no_window":
        st.error("⚠️ **No Display Window Found**: Window detection could not locate a mechanical dial display window in this image. This image is excluded from digit datasets.")
        col_img, col_act = st.columns([1.3, 1.0])
        with col_img:
            if overlay_pil is not None:
                st.image(overlay_pil, caption="Full Photo Detection Overlay (No window found)", use_container_width=True)
            else:
                st.image(Image.open(raw_path), caption="Raw Photo", use_container_width=True)
        with col_act:
            st.warning("Cannot label digits for images without a display window.")
            if st.button("⏭️ Next Image", key=f"next_no_win_{filename}"):
                st.session_state.current_queue_idx += 1
                st.rerun()
        return

    # Display images in 2 columns
    col_img, col_inputs = st.columns([1.3, 1.0])
    
    with col_img:
        # Full photo detection overlay
        st.markdown("**1. Full Photo Detection Overlay**")
        if overlay_pil is not None:
            st.image(overlay_pil, use_container_width=True, caption="Green box indicates detected display window.")
            
        st.markdown("**2. Rectified 4-Slot Display Window**")
        if rectified_pil is not None:
            # Draw slot boundary overlays for visual assistance using actual derived slot ranges
            annotated = rectified_pil.copy()
            draw = ImageDraw.Draw(annotated)
            w_img, h_img = annotated.size
            if slot_ranges and len(slot_ranges) == 4:
                orig_w = pipe_res.get("oriented_window", rectified_pil).size[0]
                for s_i in range(1, 4):
                    # Scale slot boundary to rectified width (320px)
                    x_scaled = int(round(slot_ranges[s_i][0] * (w_img / float(orig_w))))
                    draw.line([(x_scaled, 0), (x_scaled, h_img)], fill=(0, 255, 255), width=2)
            else:
                slot_w = w_img / 4.0
                for s_i in range(1, 4):
                    x = int(round(s_i * slot_w))
                    draw.line([(x, 0), (x, h_img)], fill=(0, 255, 255), width=2)
            st.image(annotated, use_container_width=True, caption="Cyan lines show the 4 dynamically derived slot boundaries.")
        else:
            st.warning("Display window could not be cropped automatically.")
            
        st.markdown("**3. Extracted Discrete Slot Crops**")
        if len(slots) == 4:
            s_cols = st.columns(4)
            for i in range(4):
                with s_cols[i]:
                    st.markdown(f"<div class='slot-title'>Slot {i+1}</div>", unsafe_allow_html=True)
                    st.image(slots[i], use_container_width=True)
                    
    with col_inputs:
        st.info("💡 **Rule:** The model's prediction is hidden until you submit your label, preventing bias.")
        
        with st.form(key=f"label_form_{filename}"):
            entered_reading = st.text_input(
                "4 Whole Digits (e.g. 0078):",
                value=existing_reading if existing_reading not in ("unreadable", "skipped") else "",
                max_chars=4,
                placeholder="4 digits (0000-9999)",
                help="Type the 4 visible whole digits from left to right."
            )

            st.markdown("**Confirm 4 Slot Numerals Individually:**")
            col_s0, col_s1, col_s2, col_s3 = st.columns(4)
            with col_s0:
                s0_val = st.text_input("Slot 1 (x1000)", value=existing_d0, max_chars=1, key=f"s0_{filename}")
            with col_s1:
                s1_val = st.text_input("Slot 2 (x100)", value=existing_d1, max_chars=1, key=f"s1_{filename}")
            with col_s2:
                s2_val = st.text_input("Slot 3 (x10)", value=existing_d2, max_chars=1, key=f"s2_{filename}")
            with col_s3:
                s3_val = st.text_input("Slot 4 (x1)", value=existing_d3, max_chars=1, key=f"s3_{filename}")

            cond_options = ["stationary", "rolling", "reflective", "unreadable"]
            default_cond_idx = cond_options.index(default_cond) if default_cond in cond_options else 0
            selected_condition = st.radio(
                "Physical Drum Condition:",
                cond_options,
                index=default_cond_idx,
                help="Mark stationary ONLY if wheels are stationary and digits are fully centered within slot aperture."
            )

            is_tpl = st.checkbox(
                "Confirm clear stationary sample for reference template library",
                value=False,
                help="Check only if all 4 digit numerals are clean, unclipped, and stationary."
            )
            
            entered_tenth = st.text_input(
                "Optional Method 2 Tenth (.0 to .9):",
                value=existing_tenth,
                max_chars=2,
                placeholder="e.g. .1 (leave empty if not entering)",
                help="Optional fractional tenth digit."
            )
            
            c_sub, c_skip, c_unr = st.columns(3)
            with c_sub:
                submit_clicked = st.form_submit_button("💾 Save (Enter)", type="primary")
            with c_skip:
                skip_clicked = st.form_submit_button("⏭️ Skip")
            with c_unr:
                unr_clicked = st.form_submit_button("⚠️ Unreadable")
                
        # Handle form actions
        if submit_clicked:
            clean_r = entered_reading.strip()
            
            # If 4 slots entered but clean_r is empty, compose clean_r from slots
            if not clean_r and s0_val.strip().isdigit() and s1_val.strip().isdigit() and s2_val.strip().isdigit() and s3_val.strip().isdigit():
                clean_r = f"{s0_val.strip()}{s1_val.strip()}{s2_val.strip()}{s3_val.strip()}"
                
            if len(clean_r) == 4 and clean_r.isdigit():
                d0 = int(s0_val.strip()) if s0_val.strip().isdigit() else int(clean_r[0])
                d1 = int(s1_val.strip()) if s1_val.strip().isdigit() else int(clean_r[1])
                d2 = int(s2_val.strip()) if s2_val.strip().isdigit() else int(clean_r[2])
                d3 = int(s3_val.strip()) if s3_val.strip().isdigit() else int(clean_r[3])

                if f"{d0}{d1}{d2}{d3}" != clean_r:
                    st.error(f"Slot mismatch: confirmed slots ({d0}{d1}{d2}{d3}) do not match 4-digit reading ({clean_r})")
                    return

                save_reading(
                    filename=filename,
                    reading=clean_r,
                    tenth=entered_tenth,
                    condition=selected_condition,
                    digit_0=d0,
                    digit_1=d1,
                    digit_2=d2,
                    digit_3=d3,
                    is_template_source=(is_tpl and selected_condition == "stationary"),
                    notes="verified_curator"
                )
                st.session_state["last_feedback"] = {
                    "type": "success",
                    "msg": f"✅ Saved `{filename}` $\to$ `{clean_r}` ({selected_condition})! (Model's blind guess was: `{model_reading}`, Conf: {model_conf:.1%})"
                }
                # Advance queue
                st.session_state.current_queue_idx = min(len(queue_df) - 1, curr_idx + 1)
                st.rerun()
            else:
                st.error("Please enter exactly 4 numeric digits (e.g. 0078), or click 'Unreadable'/'Skip'.")
                
        if skip_clicked:
            save_reading(filename, "skipped", "", condition="unreadable", notes="skipped_by_user")
            st.session_state["last_feedback"] = {"type": "info", "msg": f"Skipped `{filename}`."}
            st.session_state.current_queue_idx = min(len(queue_df) - 1, curr_idx + 1)
            st.rerun()
            
        if unr_clicked:
            save_reading(filename, "unreadable", "", condition="unreadable", notes="unreadable_dial")
            st.session_state["last_feedback"] = {"type": "warning", "msg": f"Marked `{filename}` as unreadable."}
            st.session_state.current_queue_idx = min(len(queue_df) - 1, curr_idx + 1)
            st.rerun()
            
        # Navigation buttons
        nav_prev, nav_next = st.columns(2)
        with nav_prev:
            if st.button("⬅️ Previous Image"):
                st.session_state.current_queue_idx = max(0, curr_idx - 1)
                st.rerun()
        with nav_next:
            if st.button("Next Image ➡️"):
                st.session_state.current_queue_idx = min(len(queue_df) - 1, curr_idx + 1)
                st.rerun()
                
        # Post-submission feedback banner
        if "last_feedback" in st.session_state:
            fb = st.session_state["last_feedback"]
            if fb["type"] == "success":
                st.success(fb["msg"])
            elif fb["type"] == "warning":
                st.warning(fb["msg"])
            else:
                st.info(fb["msg"])


# ==============================================================================
# OPENCV / TERMINAL CLI IMPLEMENTATION
# ==============================================================================
def run_cli_app(use_opencv: bool = False):
    print("=" * 70)
    print(" DialReader CLI / OpenCV Labeling Tool (Step 4B)")
    print("=" * 70)
    
    eligible_df, readings_df = load_dataset_and_progress()
    labeled_filenames = set(readings_df[readings_df["reading"].notnull()]["filename"].values)
    queue_df = eligible_df[~eligible_df["filename"].isin(labeled_filenames)].reset_index(drop=True)
    
    print(f"Total eligible images: {len(eligible_df)}")
    print(f"Already labeled:        {len(labeled_filenames)}")
    print(f"Remaining in queue:     {len(queue_df)}")
    
    if len(queue_df) == 0:
        print("All eligible images are already labeled in data/labeled/readings.csv!")
        return
        
    pipeline = DialReadingPipeline()
    
    for idx, row in queue_df.iterrows():
        filename = row["filename"]
        raw_path = RAW_DIR / filename
        ori = int(row["orientation"]) if pd.notnull(row["orientation"]) else None
        
        review_warn = str(row.get("review_warning", ""))
        default_cond = str(row.get("default_condition", "stationary"))
        cond_default_char = {"stationary": "s", "rolling": "r", "reflective": "f", "unreadable": "u"}.get(default_cond, "s")

        print("\n" + "-" * 70)
        print(f"[{idx+1}/{len(queue_df)}] Labeling: {filename}")
        if review_warn:
            print(f"  [REVIEW NOTICE] {review_warn}")
        
        if not raw_path.exists():
            print(f"File missing: {raw_path}")
            continue
            
        pipe_res = pipeline.read_image(raw_path, manual_orientation=ori)
        rectified_pil = pipe_res.get("rectified_window")
        model_guess = pipe_res.get("whole_digits") or "None"
        model_conf = pipe_res.get("confidence", 0.0)
        
        if rectified_pil is not None and use_opencv:
            cv_img = cv2.cvtColor(np.array(rectified_pil), cv2.COLOR_RGB2BGR)
            cv_disp = cv2.resize(cv_img, (640, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"Rectified Window - {filename}", cv_disp)
            cv2.waitKey(1)
            
        print("Commands: 4 digits (e.g. 0078) | 's' = Skip | 'u' = Unreadable | 'q' = Quit")
        user_input = input("Enter 4-digit reading: ").strip().lower()
        
        if user_input == "q":
            print("Exiting labeler. Progress saved to readings.csv.")
            break
        elif user_input == "s":
            save_reading(filename, "skipped", "", condition="unreadable", notes="skipped")
            print(f"-> Skipped {filename}.")
        elif user_input == "u":
            save_reading(filename, "unreadable", "", condition="unreadable", notes="unreadable")
            print(f"-> Marked {filename} as unreadable.")
        elif len(user_input) == 4 and user_input.isdigit():
            # Confirm each slot individually
            slot_prompt = f"Confirm 4 individual slots [{user_input[0]} {user_input[1]} {user_input[2]} {user_input[3]}] (press Enter to accept, or type 4 space-separated digits): "
            slot_input = input(slot_prompt).strip()
            if slot_input:
                parts = slot_input.split()
                if len(parts) == 4 and all(p.isdigit() for p in parts):
                    d0, d1, d2, d3 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    user_input = f"{d0}{d1}{d2}{d3}"
                else:
                    print("  Invalid slot input; defaulting to reading digits.")
                    d0, d1, d2, d3 = int(user_input[0]), int(user_input[1]), int(user_input[2]), int(user_input[3])
            else:
                d0, d1, d2, d3 = int(user_input[0]), int(user_input[1]), int(user_input[2]), int(user_input[3])

            cond_input = input(f"Condition ([s]tationary / [r]olling / [f]reflective / [u]nreadable) [default: {cond_default_char}]: ").strip().lower()
            cond_map = {"s": "stationary", "r": "rolling", "f": "reflective", "u": "unreadable"}
            cond = cond_map.get(cond_input, default_cond)
            is_tpl_input = input("Use as verified template source? (y/n) [default: n]: ").strip().lower()
            is_tpl = (is_tpl_input == "y")
            tenth = input("Optional tenth (leave empty or e.g. .1): ").strip()
            save_reading(
                filename=filename,
                reading=user_input,
                tenth=tenth,
                condition=cond,
                digit_0=d0,
                digit_1=d1,
                digit_2=d2,
                digit_3=d3,
                is_template_source=(is_tpl and cond == "stationary"),
                notes="cli_verified"
            )
            print(f"-> Saved: {user_input} ({cond}) | Slots: [{d0}, {d1}, {d2}, {d3}] | (Model blind guess: {model_guess}, Conf: {model_conf:.1%})")
        else:
            print("Invalid input. Skipping image.")
            
        if use_opencv:
            cv2.destroyAllWindows()
            
    if use_opencv:
        cv2.destroyAllWindows()


def print_status():
    """Prints current status of labeling progress and template bank."""
    print("=" * 70)
    print(" DialReader Ground-Truth Labeling & Template Status")
    print("=" * 70)
    if not READINGS_CSV.exists():
        print("readings.csv does not exist.")
        return
        
    df = pd.read_csv(READINGS_CSV)
    print(f"Total entries in readings.csv: {len(df)}")
    if "condition" in df.columns:
        print("Condition breakdown:")
        for cond, count in df["condition"].value_counts().items():
            print(f"  {cond}: {count}")
    if "is_template_source" in df.columns:
        tpl_count = df[df["is_template_source"] == True]
        print(f"Images flagged as verified template sources: {len(tpl_count)}")
        
    pipeline = DialReadingPipeline()
    tpl_lib = pipeline.classifier.templates
    print(f"Available digits in recognizer: {tpl_lib.available_digits()}")
    print(f"Missing digits in recognizer:   {tpl_lib.missing_digits()}")


def print_audit():
    """Prints authoritative Step 4E audit report for digit templates and source photos."""
    pipeline = DialReadingPipeline()
    tpl_lib = pipeline.classifier.templates
    audit = tpl_lib.audit()

    print("=" * 80)
    print(" DIALREADER STEP 4E: RECOGNIZER & TEMPLATE BANK AUDIT")
    print("=" * 80)
    print(f"Available Digits:                {audit['available_digits']}")
    print(f"Missing Digits:                  {audit['missing_digits']}")
    print(f"All Exported Human Confirmed?:   {'YES' if audit['all_exported_human_confirmed'] else 'NO'}")
    print(f"Duplicate Prevention Active?:    {'YES' if audit['deduplication_active'] else 'NO'}")
    print("\nDigit | Loaded Templates | Independent Photos | Shortfall (Target: 5) | Status")
    print("------+------------------+--------------------+-----------------------+-------")
    for d in range(10):
        t_cnt = audit["template_counts"].get(d, 0)
        s_cnt = audit["independent_sources"].get(d, 0)
        short = max(0, 5 - s_cnt)
        status = "AVAILABLE" if t_cnt > 0 else "MISSING"
        print(f"  {d}   |        {t_cnt:2d}        |         {s_cnt:2d}         |           {short:2d}          | {status}")

    if audit["missing_digits"]:
        print("\nMissing Digit Explanations:")
        for d in audit["missing_digits"]:
            print(f"  Digit {d}: {audit['missing_reasons'].get(d, 'No verified templates')}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="DialReader Ground-Truth Labeling Tool")
    parser.add_argument("--cli", action="store_true", help="Run in terminal CLI mode")
    parser.add_argument("--opencv", action="store_true", help="Run in OpenCV image popup mode")
    parser.add_argument("--export-templates", action="store_true", help="Export verified stationary templates to JSON")
    parser.add_argument("--status", action="store_true", help="Print labeling and template bank status")
    parser.add_argument("--audit", action="store_true", help="Print authoritative Step 4E template bank audit")
    parser.add_argument("--validate", action="store_true", help="Validate labeled dataset against quality and safety rules")
    args, _ = parser.parse_known_args()

    if args.audit:
        print_audit()
        return

    if args.validate:
        from src.dataset_validator import validate_labeled_dataset
        report = validate_labeled_dataset()
        report.print_report()
        return

    if args.status:
        print_status()
        return

    if args.export_templates:
        export_verified_templates()
        return
        
    if args.cli or args.opencv:
        run_cli_app(use_opencv=args.opencv)
    else:
        try:
            import streamlit as st
            run_streamlit_app()
        except Exception:
            run_cli_app(use_opencv=False)


if __name__ == "__main__":
    main()
