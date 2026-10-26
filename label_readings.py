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
import argparse
from pathlib import Path
from datetime import datetime
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
RAW_DIR = PROJECT_ROOT / "data" / "raw"

from src.infer import DialReadingPipeline


def load_dataset_and_progress():
    """Loads eligible images from labels.csv and merges with existing readings.csv."""
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"labels.csv not found at {LABELS_CSV}")
        
    df_labels = pd.read_csv(LABELS_CSV)
    # Filter eligible: dial_variant != 'non_dial_negative'
    eligible = df_labels[df_labels["dial_variant"] != "non_dial_negative"].copy()
    
    # Initialize readings.csv if missing
    if not READINGS_CSV.exists():
        READINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_readings = pd.DataFrame(columns=["filename", "reading", "tenth_optional", "labeled_at", "notes"])
        df_readings.to_csv(READINGS_CSV, index=False)
    else:
        df_readings = pd.read_csv(READINGS_CSV)
        # Ensure correct columns
        expected_cols = ["filename", "reading", "tenth_optional", "labeled_at", "notes"]
        for c in expected_cols:
            if c not in df_readings.columns:
                df_readings[c] = ""
                
    return eligible, df_readings


def save_reading(filename: str, reading: str, tenth: str = "", notes: str = ""):
    """Appends or updates a single reading in readings.csv."""
    READINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if READINGS_CSV.exists():
        df_readings = pd.read_csv(READINGS_CSV)
    else:
        df_readings = pd.DataFrame(columns=["filename", "reading", "tenth_optional", "labeled_at", "notes"])
        
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tenth_val = str(tenth).strip() if str(tenth).strip() else ""
    
    if filename in df_readings["filename"].values:
        idx = df_readings.index[df_readings["filename"] == filename][0]
        df_readings.at[idx, "reading"] = reading
        df_readings.at[idx, "tenth_optional"] = tenth_val
        df_readings.at[idx, "labeled_at"] = now_str
        df_readings.at[idx, "notes"] = notes
    else:
        new_row = pd.DataFrame([{
            "filename": filename,
            "reading": reading,
            "tenth_optional": tenth_val,
            "labeled_at": now_str,
            "notes": notes
        }])
        df_readings = pd.concat([df_readings, new_row], ignore_index=True)
        
    df_readings.to_csv(READINGS_CSV, index=False)


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
    
    # Previous labeling info if exists
    existing_reading = ""
    existing_tenth = ""
    if filename in readings_df["filename"].values:
        sub = readings_df[readings_df["filename"] == filename].iloc[0]
        existing_reading = str(sub.get("reading", ""))
        existing_tenth = str(sub.get("tenth_optional", ""))
        if existing_reading == "nan": existing_reading = ""
        if existing_tenth == "nan": existing_tenth = ""

    st.markdown('<div class="label-header">⏱️ DialReader Ground-Truth Labeler</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="label-sub">Image <b>{curr_idx + 1} of {len(queue_df)}</b> in queue &nbsp;|&nbsp; File: <code>{filename}</code> &nbsp;|&nbsp; Split: <code>{split_grp}</code></div>', unsafe_allow_html=True)
    
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
            st.image(overlay_pil, width="stretch", caption="Green box indicates detected display window.")
            
        st.markdown("**2. Rectified 4-Slot Display Window**")
        if rectified_pil is not None:
            # Draw slot boundary overlays for visual assistance using actual derived slot ranges
            annotated = rectified_pil.copy()
            draw = ImageDraw.Draw(annotated)
            w_img, h_img = annotated.size
            if slot_ranges and len(slot_ranges) == 4:
                for s_i in range(1, 4):
                    # Scale slot boundary to rectified width (320px)
                    orig_w = pipe_res.get("oriented_window").size[0]
                    x_scaled = int(round(slot_ranges[s_i][0] * (w_img / float(orig_w))))
                    draw.line([(x_scaled, 0), (x_scaled, h_img)], fill=(0, 255, 255), width=2)
            else:
                slot_w = w_img / 4.0
                for s_i in range(1, 4):
                    x = int(round(s_i * slot_w))
                    draw.line([(x, 0), (x, h_img)], fill=(0, 255, 255), width=2)
            st.image(annotated, width="stretch", caption="Cyan lines show the 4 dynamically derived slot boundaries.")
        else:
            st.warning("Display window could not be cropped automatically.")
            
        st.markdown("**3. Extracted Discrete Slot Crops**")
        if len(slots) == 4:
            s_cols = st.columns(4)
            for i in range(4):
                with s_cols[i]:
                    st.markdown(f"<div class='slot-title'>Slot {i+1}</div>", unsafe_allow_html=True)
                    st.image(slots[i], width="stretch")
                    
    with col_inputs:
        st.markdown("### ✍️ Enter Ground Truth")
        st.info("💡 **Rule:** The model's prediction is hidden until you submit your label, preventing bias.")
        
        with st.form(key=f"label_form_{filename}"):
            entered_reading = st.text_input(
                "4 Whole Digits (e.g. 0078):",
                value=existing_reading if existing_reading not in ("unreadable", "skipped") else "",
                max_chars=4,
                placeholder="4 digits (0000-9999)",
                help="Type the 4 visible whole digits from left to right."
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
            if len(clean_r) == 4 and clean_r.isdigit():
                save_reading(filename, clean_r, entered_tenth, notes="normal")
                st.session_state["last_feedback"] = {
                    "type": "success",
                    "msg": f"✅ Saved `{filename}` $\to$ `{clean_r}`! (Model's blind guess was: `{model_reading}`, Conf: {model_conf:.1%})"
                }
                # Advance queue
                st.session_state.current_queue_idx = min(len(queue_df) - 1, curr_idx + 1)
                st.rerun()
            else:
                st.error("Please enter exactly 4 numeric digits (e.g. 0078), or click 'Unreadable'/'Skip'.")
                
        if skip_clicked:
            save_reading(filename, "skipped", "", notes="skipped_by_user")
            st.session_state["last_feedback"] = {"type": "info", "msg": f"Skipped `{filename}`."}
            st.session_state.current_queue_idx = min(len(queue_df) - 1, curr_idx + 1)
            st.rerun()
            
        if unr_clicked:
            save_reading(filename, "unreadable", "", notes="unreadable_dial")
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
    print(" DialReader CLI / OpenCV Labeling Tool (Step 1)")
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
        
        print("\n" + "-" * 70)
        print(f"[{idx+1}/{len(queue_df)}] Labeling: {filename}")
        
        if not raw_path.exists():
            print(f"File missing: {raw_path}")
            continue
            
        pipe_res = pipeline.read_image(raw_path, manual_orientation=ori)
        rectified_pil = pipe_res.get("rectified_window")
        model_guess = pipe_res.get("whole_digits") or "None"
        model_conf = pipe_res.get("confidence", 0.0)
        
        if rectified_pil is not None and use_opencv:
            cv_img = cv2.cvtColor(np.array(rectified_pil), cv2.COLOR_RGB2BGR)
            # Resize for comfortable viewing
            cv_disp = cv2.resize(cv_img, (640, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"Rectified Window - {filename}", cv_disp)
            cv2.waitKey(1)
            
        print("Commands: 4 digits (e.g. 0078) | 's' = Skip | 'u' = Unreadable | 'q' = Quit")
        user_input = input("Enter 4-digit reading: ").strip().lower()
        
        if user_input == "q":
            print("Exiting labeler. Progress saved to readings.csv.")
            break
        elif user_input == "s":
            save_reading(filename, "skipped", "", notes="skipped")
            print(f"-> Skipped {filename}.")
        elif user_input == "u":
            save_reading(filename, "unreadable", "", notes="unreadable")
            print(f"-> Marked {filename} as unreadable.")
        elif len(user_input) == 4 and user_input.isdigit():
            tenth = input("Optional tenth (leave empty or e.g. .1): ").strip()
            save_reading(filename, user_input, tenth, notes="normal")
            print(f"-> Saved: {user_input} | (Model's blind guess was: {model_guess}, Conf: {model_conf:.1%})")
        else:
            print("Invalid input. Skipping image.")
            
        if use_opencv:
            cv2.destroyAllWindows()
            
    if use_opencv:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="DialReader Ground-Truth Labeling Tool")
    parser.add_argument("--cli", action="store_true", help="Run in terminal CLI mode")
    parser.add_argument("--opencv", action="store_true", help="Run in OpenCV image popup mode")
    args, _ = parser.parse_known_args()
    
    # If run via `streamlit run`, streamlit handles the environment
    is_streamlit = "streamlit" in sys.modules or os.environ.get("STREAMLIT_RUNNING", False)
    
    if args.cli or args.opencv:
        run_cli_app(use_opencv=args.opencv)
    else:
        # Default to streamlit runner
        try:
            import streamlit as st
            run_streamlit_app()
        except Exception:
            run_cli_app(use_opencv=False)


if __name__ == "__main__":
    main()
