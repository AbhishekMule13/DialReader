"""
Interactive Streamlit Labeling Tool for DialReader
Enables visual bounding box annotation and 4-digit reading entry.
Saves directly to data/labeled/labels.csv and exports YOLO format.

Run with:
    streamlit run label_ui.py
"""

import ast
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st

# Compatibility patch for streamlit-drawable-canvas with modern Streamlit (>= 1.28)
try:
    import streamlit.elements.image as _st_image
    if not hasattr(_st_image, "image_to_url"):
        from streamlit.elements.lib.image_utils import image_to_url as _lib_image_to_url
        from streamlit.elements.image import create_layout_config
        _st_image.image_to_url = lambda image, width, clamp, channels, output_format, image_id: _lib_image_to_url(
            image, create_layout_config(width=width), clamp, channels, output_format, image_id
        )
except Exception:
    pass

from streamlit_drawable_canvas import st_canvas

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LABELED_DIR = PROJECT_ROOT / "data" / "labeled"
CSV_PATH = LABELED_DIR / "labels.csv"
YOLO_DIR = LABELED_DIR / "yolo_dataset"
CONFIG_PATH = PROJECT_ROOT / "configs" / "detector.yaml"

st.set_page_config(
    page_title="DialReader Labeling Tool",
    page_icon="🔍",
    layout="wide"
)

# Initialize / load CSV
if "labels_df" not in st.session_state or st.session_state.get("reload_csv", False):
    if not CSV_PATH.exists():
        from src.utils.create_labels_template import create_template
        create_template(RAW_DIR, CSV_PATH)
    st.session_state.labels_df = pd.read_csv(CSV_PATH)
    st.session_state.reload_csv = False

df = st.session_state.labels_df

# Ensure necessary columns
cols = ["filename", "bbox", "reading", "orientation", "mid_roll", "dial_variant", "split_group"]
for c in cols:
    if c not in df.columns:
        df[c] = ""


def save_and_export():
    df.to_csv(CSV_PATH, index=False)
    # Trigger YOLO export in background
    try:
        from src.utils.export_yolo import export_yolo_dataset
        export_yolo_dataset(CSV_PATH, RAW_DIR, YOLO_DIR, CONFIG_PATH)
    except Exception as e:
        st.warning(f"Note: YOLO export error: {e}")


def parse_bbox_val(val):
    if isinstance(val, str) and val.strip():
        try:
            b = ast.literal_eval(val.strip())
            if isinstance(b, (list, tuple)) and len(b) == 4:
                return [float(x) for x in b]
        except Exception:
            pass
    return None


# Sidebar controls & navigation
st.sidebar.title("🔍 DialReader Labeler")

# Filter options
filter_mode = st.sidebar.radio(
    "Filter View:",
    ["All Files", "Unlabeled Only", "Labeled Only", "Knife+Pressing (Test Set)", "Negatives Only"]
)

if filter_mode == "Unlabeled Only":
    active_indices = df[(df["reading"].isna()) | (df["reading"] == "") | (df["bbox"].isna()) | (df["bbox"] == "")].index.tolist()
elif filter_mode == "Labeled Only":
    active_indices = df[(df["reading"].notna()) & (df["reading"] != "") & (df["reading"] != "negative")].index.tolist()
elif filter_mode == "Knife+Pressing (Test Set)":
    active_indices = df[df["filename"].str.startswith("knife_pressing")].index.tolist()
elif filter_mode == "Negatives Only":
    active_indices = df[df["dial_variant"] == "non_dial_negative"].index.tolist()
else:
    active_indices = df.index.tolist()

if not active_indices:
    st.sidebar.warning("No records match the selected filter.")
    active_indices = df.index.tolist()

# Current index tracking
if "current_idx" not in st.session_state:
    st.session_state.current_idx = active_indices[0] if active_indices else 0

if st.session_state.current_idx not in active_indices:
    st.session_state.current_idx = active_indices[0]

curr_idx = st.session_state.current_idx

# Progress
total_rows = len(df)
labeled_count = len(df[(df["reading"].notna()) & (df["reading"] != "")])
st.sidebar.progress(labeled_count / max(1, total_rows))
st.sidebar.write(f"**Progress**: {labeled_count} / {total_rows} rows labeled ({labeled_count/max(1,total_rows)*100:.1f}%)")

# Jump to row
selected_idx = st.sidebar.selectbox(
    "Select item:",
    options=active_indices,
    index=active_indices.index(curr_idx),
    format_func=lambda i: f"[{i}] {df.loc[i, 'filename']} ({df.loc[i, 'reading'] or 'unlabeled'})"
)
if selected_idx != curr_idx:
    st.session_state.current_idx = selected_idx
    st.rerun()

# Current record
row = df.loc[curr_idx]
rel_filename = str(row["filename"])
img_path = RAW_DIR / rel_filename

# Main Area
col_left, col_right = st.columns([1.3, 1.0])

with col_left:
    st.subheader(f"🖼️ Image [{curr_idx}]: `{rel_filename}`")
    
    if not img_path.exists():
        st.error(f"File not found: {img_path}")
        image = None
    else:
        orig_img = Image.open(img_path)
        orig_img = ImageOps.exif_transpose(orig_img)
        orig_w, orig_h = orig_img.size

        # Canvas display scaling (max width 600)
        display_w = 600
        scale = display_w / orig_w
        display_h = int(orig_h * scale)

        # Parse existing bbox
        existing_bbox = parse_bbox_val(row["bbox"])
        init_rect = None
        if existing_bbox is not None:
            xmin, ymin, xmax, ymax = existing_bbox
            init_rect = {
                "type": "rect",
                "left": xmin * display_w,
                "top": ymin * display_h,
                "width": (xmax - xmin) * display_w,
                "height": (ymax - ymin) * display_h,
                "stroke": "#00FF00",
                "strokeWidth": 2,
                "fill": "rgba(0, 255, 0, 0.2)"
            }

        st.caption("Draw a rectangle around the `display_window` (digits + fractional scale) or adjust coordinates on the right:")
        
        # Interactive Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.25)",
            stroke_width=2,
            stroke_color="#FF4500",
            background_image=orig_img.resize((display_w, display_h), Image.Resampling.BILINEAR),
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="rect",
            initial_drawing={"objects": [init_rect]} if init_rect else None,
            key=f"canvas_{curr_idx}"
        )

with col_right:
    st.subheader("📝 Dial Window Details")

    # Detect bbox from canvas if drawn
    canvas_bbox = None
    if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
        rects = [obj for obj in canvas_result.json_data["objects"] if obj["type"] == "rect"]
        if rects:
            last_rect = rects[-1]
            l = last_rect["left"] / display_w
            t = last_rect["top"] / display_h
            w = last_rect["width"] / display_w
            h = last_rect["height"] / display_h
            canvas_bbox = [round(l, 4), round(t, 4), round(l + w, 4), round(t + h, 4)]

    active_bbox = canvas_bbox if canvas_bbox is not None else existing_bbox

    # Bounding Box inputs
    st.write("**Bounding Box** `[xmin, ymin, xmax, ymax]` (normalized 0.0 - 1.0):")
    bbox_str = str(active_bbox) if active_bbox is not None else ""
    bbox_input = st.text_input("BBox Coordinates", value=bbox_str, help="e.g. [0.3421, 0.4512, 0.5123, 0.5210]")

    # Reading input
    curr_reading = str(row["reading"]) if pd.notna(row["reading"]) and str(row["reading"]) != "nan" else ""
    reading_val = st.text_input(
        "4-Digit Reading",
        value=curr_reading,
        placeholder="e.g. 0045, 0080, negative",
        help="Enter the 4-digit reading. Type 'negative' if non-dial background image."
    )

    # Orientation
    curr_ori = int(row["orientation"]) if pd.notna(row["orientation"]) and str(row["orientation"]).isdigit() else 0
    ori_options = [0, 90, 180, 270]
    ori_idx = ori_options.index(curr_ori) if curr_ori in ori_options else 0
    orientation_val = st.selectbox(
        "Orientation (degrees counter-clockwise to upright)",
        options=ori_options,
        index=ori_idx,
        help="0 for normal upright dial, 90 for dial turned 90 deg, etc."
    )

    # Mid-roll flag
    curr_mid = bool(row["mid_roll"]) if pd.notna(row["mid_roll"]) else False
    mid_roll_val = st.checkbox("Mid-roll transition (last drum between two digits)", value=curr_mid)

    # Dial variant
    curr_variant = str(row["dial_variant"]) if pd.notna(row["dial_variant"]) and str(row["dial_variant"]) != "nan" else "standard_3black_1red"
    variant_options = ["standard_3black_1red", "all_black", "non_dial_negative", "unknown"]
    v_idx = variant_options.index(curr_variant) if curr_variant in variant_options else 0
    variant_val = st.selectbox("Dial Variant", options=variant_options, index=v_idx)

    # Crop preview
    parsed_curr_bbox = parse_bbox_val(bbox_input)
    if parsed_curr_bbox and img_path.exists():
        xmin, ymin, xmax, ymax = parsed_curr_bbox
        px_box = (
            int(xmin * orig_w),
            int(ymin * orig_h),
            int(xmax * orig_w),
            int(ymax * orig_h)
        )
        if px_box[2] > px_box[0] and px_box[3] > px_box[1]:
            crop_img = orig_img.crop(px_box)
            if orientation_val != 0:
                crop_img = crop_img.rotate(orientation_val, expand=True)
            st.write("**Extracted Normalized Crop Preview:**")
            st.image(crop_img, width=280)

    st.markdown("---")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

    with btn_col1:
        if st.button("💾 Save & Next (Enter)", use_container_width=True, type="primary"):
            df.loc[curr_idx, "bbox"] = bbox_input.strip()
            df.loc[curr_idx, "reading"] = reading_val.strip()
            df.loc[curr_idx, "orientation"] = orientation_val
            df.loc[curr_idx, "mid_roll"] = mid_roll_val
            df.loc[curr_idx, "dial_variant"] = variant_val
            save_and_export()

            # Advance to next index
            curr_pos = active_indices.index(curr_idx)
            next_idx = active_indices[(curr_pos + 1) % len(active_indices)]
            st.session_state.current_idx = next_idx
            st.rerun()

    with btn_col2:
        if st.button("⬅️ Previous", use_container_width=True):
            curr_pos = active_indices.index(curr_idx)
            prev_idx = active_indices[(curr_pos - 1) % len(active_indices)]
            st.session_state.current_idx = prev_idx
            st.rerun()

    with btn_col3:
        if st.button("➕ Add Another Dial for this Image", use_container_width=True):
            new_row = {
                "filename": rel_filename,
                "bbox": "",
                "reading": "",
                "orientation": orientation_val,
                "mid_roll": False,
                "dial_variant": variant_val,
                "split_group": row["split_group"]
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.session_state.labels_df = df
            save_and_export()
            st.session_state.current_idx = len(df) - 1
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("**Quick Guide:**")
st.sidebar.write("1. Click & drag on image to draw `display_window` rectangle.")
st.sidebar.write("2. Type 4-digit reading (e.g. `0045`).")
st.sidebar.write("3. If dial is sideways, select orientation (90/270).")
st.sidebar.write("4. Check 'Mid-roll' if drum 4 is between digits.")
st.sidebar.write("5. Click **Save & Next** to commit.")
