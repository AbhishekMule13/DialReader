"""
DialReader: Interactive Web Application
End-to-end mechanical dial counter reader with pure classical computer vision
and Method 2 pixel-level decimal sub-division.

Supports automated reading of JPG, JPEG, and PNG images without requiring manual entry.
"""

from pathlib import Path
import sys
import io
import time
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import DialReadingPipeline
from src.digits import PixelDigitRecognizer, DigitTemplateLibrary
from label_readings import save_reading

# Page setup
st.set_page_config(
    page_title="DialReader - Automatic Mechanical Counter AI",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Industrial Glassmorphism with Elesa Orange Accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, #18202c 0%, #0d1117 100%);
    }
    
    /* Hero Title */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff8c33 0%, #ff5500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #8b949e;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }

    /* Template Bank Notice Pill */
    .tpl-notice {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.85rem;
        color: #c9d1d9;
        margin-bottom: 1.2rem;
    }

    /* Digital LCD Counter Display */
    .lcd-container {
        display: flex;
        align-items: center;
        justify-content: center;
        background: #05070a;
        border: 2px solid #30363d;
        border-radius: 14px;
        padding: 16px 26px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6), inset 0 0 20px rgba(0,0,0,0.8);
        margin: 10px 0 20px 0;
        gap: 10px;
    }
    
    .lcd-slot {
        background: #111418;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 8px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.2rem;
        font-weight: 800;
        color: #f0f6fc;
        text-shadow: 0 0 10px rgba(255,255,255,0.4);
        display: inline-block;
        min-width: 58px;
        text-align: center;
    }
    
    .lcd-slot-red {
        background: linear-gradient(180deg, #b31217 0%, #7d0d10 100%);
        border: 1px solid #ff4d4d;
        color: #ffffff;
        text-shadow: 0 0 12px rgba(255,100,100,0.8);
    }

    .lcd-slot-warn {
        background: #2d1800;
        border: 1px solid #ff9900;
        color: #ffcc00;
        text-shadow: 0 0 10px rgba(255,153,0,0.6);
    }
    
    .lcd-dot {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.2rem;
        font-weight: 800;
        color: #ff5500;
        margin: 0 -2px;
    }
    
    .lcd-slot-decimal {
        background: linear-gradient(180deg, #ff6600 0%, #b83d00 100%);
        border: 1px solid #ffaa55;
        color: #ffffff;
        text-shadow: 0 0 12px rgba(255,150,50,0.9);
    }
    
    /* Metric Card */
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 14px 18px;
        backdrop-filter: blur(10px);
        margin-bottom: 10px;
    }
    
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #58a6ff;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-lbl {
        color: #8b949e;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    
    /* Slot Card */
    .slot-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Initializes and caches the classical DialReadingPipeline."""
    return DialReadingPipeline(
        detector_weights=None,
        min_digit_conf=0.35
    )


def create_annotated_window(window_pil, decimal_info, whole_reading, full_value, slot_ranges=None):
    """Draws reading baseline, slot dividers, vernier line, and Method 2 red digit square overlay."""
    np_img = np.array(window_pil.convert("RGB"))
    h, w, _ = np_img.shape
    vis = np_img.copy()

    # 1. Vertical slot boundaries
    if slot_ranges and len(slot_ranges) == 4:
        for s_i in range(1, 4):
            x = int(round(slot_ranges[s_i][0]))
            if 0 <= x < w:
                cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
        # Vernier scale dividing line at the right edge of slot 4
        x_vernier = int(round(slot_ranges[3][1]))
        if 0 <= x_vernier < w:
            cv2.line(vis, (x_vernier, 0), (x_vernier, h), (255, 120, 0), 2)
    else:
        usable_w = w * (1.0 - 0.12)
        slot_w = usable_w / 4.0
        for i in range(1, 4):
            x = int(round(i * slot_w))
            cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
        cv2.line(vis, (int(usable_w), 0), (int(usable_w), h), (255, 120, 0), 2)

    # 2. Horizontal Reading Baseline
    baseline_y = int(round(decimal_info.get("baseline_y", h / 2.0)))
    baseline_y = int(np.clip(baseline_y, 0, h - 1))
    right_end = int(slot_ranges[2][1]) if (slot_ranges and len(slot_ranges) >= 3) else int(w * 0.75)
    cv2.line(vis, (0, baseline_y), (right_end, baseline_y), (0, 255, 255), 2)
    cv2.putText(vis, "Baseline", (8, max(12, baseline_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    # 3. Method 2 Red Drum Bounding Box ("Square Around the Digit")
    bbox = decimal_info.get("digit_box") or decimal_info.get("digit_bbox")
    if bbox:
        bx, by, bw, bh = [int(v) for v in bbox]
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        digit_cy = int(round(decimal_info.get("digit_cy", by + bh / 2.0)))
        digit_cy = int(np.clip(digit_cy, 0, h - 1))
        cv2.line(vis, (bx, digit_cy), (bx + bw, digit_cy), (0, 0, 255), 1)
        
        # Shift arrow & label
        arrow_x = min(w - 24, bx + bw + 6)
        cv2.arrowedLine(vis, (arrow_x, baseline_y), (arrow_x, digit_cy), (0, 255, 255), 1, tipLength=0.3)
        shift_txt = f"{decimal_info.get('displacement_px', 0.0):+.1f}px"
        mid_y = int(np.clip((baseline_y + digit_cy) / 2, 10, h - 5))
        cv2.putText(vis, shift_txt, (min(w - 50, arrow_x + 4), mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1)

    return Image.fromarray(vis)


# -------------------------------------------------------------
# Sidebar: Settings & Preloaded Presets
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⏱️ **DialReader Controls**")
    
    st.markdown("#### **Image Source**")
    input_mode = st.radio(
        "Choose mode:",
        ["📤 Upload New Photo", "🎯 Sample Presets & Dials"],
        index=0
    )
    
    preset_choice = None
    preset_map = {
        "Preset 1: dial_0078.png (0078.0)": PROJECT_ROOT / "data" / "presets" / "dial_0078.png",
        "Preset 2: dial_0080.png (0080.0)": PROJECT_ROOT / "data" / "presets" / "dial_0080.png",
        "Preset 3: 80.jpeg (Full Machine Photo)": PROJECT_ROOT / "data" / "raw" / "80.jpeg",
        "Sample Raw: knife_20261006_083755899_iOS.jpg (Unreadable Dial)": PROJECT_ROOT / "data" / "raw" / "knife" / "20261006_083755899_iOS.jpg",
        "Sample Raw: knife_20261006_083758932_iOS.jpg (Rolling Drums)": PROJECT_ROOT / "data" / "raw" / "knife" / "20261006_083758932_iOS.jpg"
    }

    if input_mode == "🎯 Sample Presets & Dials":
        preset_choice = st.selectbox(
            "Select sample dial:",
            list(preset_map.keys()),
            index=0
        )
    
    st.markdown("---")
    st.markdown("#### **Orientation & Thresholds**")
    manual_orient = st.selectbox(
        "Manual Rotation:",
        [None, 0, 90, 180, 270],
        format_func=lambda x: "Auto Detect" if x is None else f"{x}° Rotation"
    )
    
    min_conf = st.slider("Min Digit Confidence Threshold", 0.20, 0.80, 0.35, 0.05)
    
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.82rem; color:#8b949e; line-height:1.4;">
        <b>Classical CV Pipeline:</b><br>
        • Stage 1: Full-Resolution Display Window Detector<br>
        • Stage 2: Deskew & Dynamic 4-Slot Normalizer<br>
        • Stage 3: PixelDigitRecognizer (Geometric features + Templates)<br>
        • Stage 4: Method 2 Decimal Displacement Measurement<br>
        <br>
        <i>Zero neural network dependencies (no ANN, CNN, YOLO, or OCR).</i>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown('<div class="hero-title">⏱️ DialReader — Automatic Mechanical Counter Reader</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Automatic 4-digit reading with <b>Method 2 continuous decimal precision</b> using classical pixel computer vision.</div>',
    unsafe_allow_html=True
)

# Template bank status banner
pipeline = load_pipeline()
tpl_lib = pipeline.classifier.templates
avail_digits = tpl_lib.available_digits()
missing_digits = tpl_lib.missing_digits()

st.markdown(f"""
<div class="tpl-notice">
    <b>Template Bank Status:</b> Verified templates available for digits: <b style="color:#00e676;">{avail_digits}</b> &nbsp;|&nbsp; 
    Lacking verified templates: <b style="color:#ffa726;">{missing_digits}</b><br>
    <span style="font-size:0.80rem; color:#8b949e;">Unsupported digits are rejected with status <code>uncertain</code> / <code>missing_template</code> rather than falsely forced into 0, 7, or 8.</span>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Image Loading Workflow
# -------------------------------------------------------------
input_pil: Optional[Image.Image] = None
image_filename: str = "custom_upload.png"

if input_mode == "📤 Upload New Photo":
    uploaded_file = st.file_uploader(
        "Upload a dial photo (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        help="Upload any mechanical dial counter image. Reading will be performed automatically."
    )
    if uploaded_file is not None:
        input_pil = Image.open(uploaded_file).convert("RGB")
        image_filename = uploaded_file.name
    else:
        st.info("👆 Please upload a dial image (JPG, JPEG, or PNG) above to view its automatic reading, or select 'Sample Presets & Dials' in the sidebar.")
else:
    preset_path = preset_map.get(preset_choice)
    if preset_path and preset_path.exists():
        input_pil = Image.open(preset_path).convert("RGB")
        image_filename = preset_path.name
    else:
        st.warning(f"Preset file not found: {preset_path}")

# -------------------------------------------------------------
# Automatic Pipeline Inference
# -------------------------------------------------------------
if input_pil is not None:
    pipeline.min_digit_conf = min_conf
    
    with st.spinner("Automatically analyzing display window, segmenting slots, and estimating decimal..."):
        start_t = time.time()
        result = pipeline.read_image(input_pil, manual_orientation=manual_orient)
        latency = round((time.time() - start_t) * 1000, 1)

    # ---------------------------------------------------------
    # Extracted Metrics
    # ---------------------------------------------------------
    whole_digits = result.get("whole_digits") or "0000"
    tenth_val = result.get("tenth", 0)
    full_val = result.get("value") or f"{whole_digits}.{tenth_val}"
    status = result.get("status", "ok")
    reason = result.get("reason", "")
    confidence = result.get("confidence", 0.0)
    decimal_info = result.get("decimal_info", {})
    disp_px = decimal_info.get("displacement_px", 0.0)

    # ---------------------------------------------------------
    # Status Banner & Diagnostic Explanation
    # ---------------------------------------------------------
    if status == "no_window":
        st.error(f"❌ **No Display Window Detected**: {reason or 'The mechanical dial aperture could not be located in this photograph.'}")
        st.markdown("#### **Detection Overlay on Full Photo**")
        overlay_img = result.get("detected_overlay", input_pil)
        st.image(overlay_img, caption="Detection Overlay: No Display Window Found", use_container_width=True)
    else:
        # LCD Hero Display
        d1 = whole_digits[0] if len(whole_digits) > 0 else "-"
        d2 = whole_digits[1] if len(whole_digits) > 1 else "-"
        d3 = whole_digits[2] if len(whole_digits) > 2 else "-"
        d4 = whole_digits[3] if len(whole_digits) > 3 else "-"

        slot_css = "lcd-slot-warn" if status in ("uncertain", "unreadable") else "lcd-slot"
        red_css = "lcd-slot-warn" if status in ("uncertain", "unreadable") else "lcd-slot-red"

        st.markdown(f"""
        <div class="lcd-container">
            <div class="{slot_css}">{d1}</div>
            <div class="{slot_css}">{d2}</div>
            <div class="{slot_css}">{d3}</div>
            <div class="{red_css}">{d4}</div>
            <div class="lcd-dot">.</div>
            <div class="lcd-slot-decimal">{tenth_val}</div>
        </div>
        """, unsafe_allow_html=True)

        if status == "ok":
            st.success(f"✅ **Dial Reading Verified**: **`{full_val}`** &nbsp;|&nbsp; Status: `OK` (Confidence: {confidence:.0%})")
        elif status == "uncertain":
            st.warning(f"⚠️ **Reading Held as Uncertain**: {reason}")
        elif status == "unreadable":
            st.error(f"❌ **Unreadable Dial Reading**: {reason}")

        # Summary Metric Cards
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Total Reading</div>
                <div class="metric-val" style="color:#ffaa55;">{full_val}</div>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            conf_color = "#00e676" if confidence >= 0.60 else "#ffa726" if confidence >= 0.35 else "#ff5252"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Confidence / Status</div>
                <div class="metric-val" style="color:{conf_color};">{int(confidence * 100)}% ({status.upper()})</div>
            </div>
            """, unsafe_allow_html=True)
        with b3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Method 2 Shift (ΔY)</div>
                <div class="metric-val" style="color:#58a6ff;">{disp_px:+.1f} px</div>
            </div>
            """, unsafe_allow_html=True)
        with b4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Inference Latency</div>
                <div class="metric-val" style="color:#c9d1d9;">{latency} ms</div>
            </div>
            """, unsafe_allow_html=True)

        # Mid-roll transition warning
        if result.get("is_mid_roll"):
            st.info("ℹ️ **Mid-Roll Drum Transition:** The 4th red drum is rolling between numbers. Method 2 displacement continuously measured the fractional position.")

        st.markdown("---")

        # ---------------------------------------------------------
        # Visual Breakdown Columns (Side-by-side inspection)
        # ---------------------------------------------------------
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### **1. Uploaded Original Photo & Detected Boundary**")
            overlay_img = result.get("detected_overlay", input_pil)
            st.image(overlay_img, caption="Green box indicates detected display window", use_container_width=True)

        with col_right:
            st.markdown("#### **2. Rectified Window & Method 2 Decimal Analysis**")
            window_to_annotate = result.get("oriented_window") or result.get("rectified_window")
            slot_ranges = result.get("slot_ranges", [])
            if window_to_annotate is not None:
                annotated = create_annotated_window(
                    window_to_annotate,
                    decimal_info,
                    whole_digits,
                    full_val,
                    slot_ranges=slot_ranges
                )
                st.image(
                    annotated,
                    caption="Rectified Window (Cyan=Baseline, Green=Slot Dividers, Orange=Scale, Red=Digit Box, Cyan=Shift)",
                    use_container_width=True
                )

            # Method 2 metrics explanation card
            raw_tenth = decimal_info.get("raw_tenth", 0.0)
            base_y = decimal_info.get("baseline_y", 64.0)
            digit_cy = decimal_info.get("digit_cy", 64.0)
            bbox = decimal_info.get("digit_box") or decimal_info.get("digit_bbox") or [0, 0, 0, 0]
            bbox_vals = [int(v) for v in bbox]

            st.markdown(f"""
            <div style="background:#161b22; border-left:4px solid #ff6600; padding:12px 16px; border-radius:6px; font-size:0.88rem; color:#c9d1d9; margin-top:8px;">
                <b>Method 2 Continuous Decimal Measurement:</b><br>
                • <b>Reading Baseline:</b> <code>Y = {base_y:.1f} px</code><br>
                • <b>Red Drum Digit Centroid:</b> <code>Y = {digit_cy:.1f} px</code> (BBox: <code>[{bbox_vals[0]}, {bbox_vals[1]}, {bbox_vals[2]}, {bbox_vals[3]}]</code>)<br>
                • <b>Displacement (ΔY):</b> <code>{base_y:.1f} - {digit_cy:.1f} = {disp_px:+.1f} px</code><br>
                • <b>Fractional Tenth:</b> <code>round({raw_tenth:.2f}) = {tenth_val}</code> → <b><code>.{tenth_val}</code></b>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ---------------------------------------------------------
        # 4 Discrete Digit Slot Breakdown
        # ---------------------------------------------------------
        st.markdown("#### **3. Extracted 4 Digit Slots (Classical Pixel Recognition)**")
        digit_slots = result.get("digit_slots", [])
        per_digit_conf = result.get("per_digit_confidence", [1.0, 1.0, 1.0, 1.0])
        slot_diagnostics = result.get("slot_diagnostics", [])

        if len(digit_slots) == 4:
            s_cols = st.columns(4)
            titles = ["Slot 1 (x1000)", "Slot 2 (x100)", "Slot 3 (x10)", "Slot 4 (x1 Red Drum)"]

            for i in range(4):
                with s_cols[i]:
                    conf_i = per_digit_conf[i] if i < len(per_digit_conf) else 1.0
                    slot_diag = slot_diagnostics[i] if i < len(slot_diagnostics) else {}
                    slot_stat = slot_diag.get("status", "ok")
                    slot_reason = slot_diag.get("reason", "OK")

                    stat_color = "#00e676" if slot_stat == "ok" else ("#ffa726" if slot_stat == "ambiguous" else "#ff5252")

                    st.markdown(f"""
                    <div class="slot-card">
                        <div style="font-size:0.8rem; color:#8b949e; margin-bottom:4px;">{titles[i]}</div>
                        <div style="font-size:2.2rem; font-weight:800; color:{'#ff4d4d' if i==3 else '#f0f6fc'}; font-family:'JetBrains Mono';">
                            {whole_digits[i] if i < len(whole_digits) else '-'}
                        </div>
                        <div style="font-size:0.75rem; color:{stat_color}; font-weight:600;">Status: {slot_stat.upper()}</div>
                        <div style="font-size:0.75rem; color:#8b949e;">Conf: {int(conf_i * 100)}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(digit_slots[i], use_container_width=True)
                    st.progress(float(np.clip(conf_i, 0.0, 1.0)))

                    if slot_stat != "ok":
                        st.caption(f"⚠️ {slot_reason}")

                    # Detailed inspection expander
                    with st.expander(f"🔬 Slot {i+1} Pixel Details"):
                        norm_mask, feats = PixelDigitRecognizer.extract_normalized_digit_mask(digit_slots[i])
                        if norm_mask is not None:
                            st.image(norm_mask, caption="Canonical Binary Mask (48x32)", width=120)
                            st.write(f"• **Holes:** `{feats.get('holes', 0)}`")
                            st.write(f"• **Aspect Ratio:** `{feats.get('aspect_ratio', 0.0):.2f}`")
                            st.write(f"• **Crossbar:** `{feats.get('has_crossbar', False)}`")
                            st.write(f"• **Top Density:** `{feats.get('top_half_density', 0.0):.2f}`")
                            st.write(f"• **Bottom Density:** `{feats.get('bottom_half_density', 0.0):.2f}`")

    # ---------------------------------------------------------
    # Optional Human Correction / Review Workflow
    # ---------------------------------------------------------
    st.markdown("---")
    with st.expander("✏️ Optional Review: Correct Ground-Truth Reading (For Verified Operators)"):
        st.markdown(
            """
            <div style="font-size:0.85rem; color:#8b949e; margin-bottom:10px;">
            💡 <b>Note:</b> Normal automated inference never requires manual entry. Use this optional section ONLY if you know the true physical reading.
            Submitted corrections are stored in <code>data/labeled/readings.csv</code> with <code>is_template_source=False</code> and do <b>NOT</b> alter active production templates without explicit verified curation.
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form(key="correction_form"):
            c_r1, c_r2 = st.columns(2)
            with c_r1:
                corr_reading = st.text_input(
                    "4-Digit Whole Reading:",
                    value=whole_digits if status == "ok" else "",
                    max_chars=4,
                    placeholder="e.g. 0078",
                    help="Enter exactly 4 numeric digits."
                )
            with c_r2:
                corr_tenth = st.text_input(
                    "Optional Tenth (.0 to .9):",
                    value=f".{tenth_val}",
                    max_chars=2,
                    placeholder=".0"
                )

            st.markdown("**Individual Slot Numerals:**")
            col_s0, col_s1, col_s2, col_s3 = st.columns(4)
            with col_s0:
                s0 = st.text_input("Slot 1", value=whole_digits[0] if len(whole_digits) > 0 else "", max_chars=1)
            with col_s1:
                s1 = st.text_input("Slot 2", value=whole_digits[1] if len(whole_digits) > 1 else "", max_chars=1)
            with col_s2:
                s2 = st.text_input("Slot 3", value=whole_digits[2] if len(whole_digits) > 2 else "", max_chars=1)
            with col_s3:
                s3 = st.text_input("Slot 4", value=whole_digits[3] if len(whole_digits) > 3 else "", max_chars=1)

            corr_cond = st.selectbox(
                "Physical Drum Condition:",
                ["stationary", "rolling", "reflective", "unreadable"],
                index=0
            )

            corr_notes = st.text_input("Curator Notes:", value="streamlit_review_entry")
            submitted = st.form_submit_button("💾 Save Ground-Truth Record", type="primary")

            if submitted:
                clean_r = corr_reading.strip()
                if len(clean_r) == 4 and clean_r.isdigit():
                    d0 = int(s0.strip()) if s0.strip().isdigit() else int(clean_r[0])
                    d1 = int(s1.strip()) if s1.strip().isdigit() else int(clean_r[1])
                    d2 = int(s2.strip()) if s2.strip().isdigit() else int(clean_r[2])
                    d3 = int(s3.strip()) if s3.strip().isdigit() else int(clean_r[3])

                    save_reading(
                        filename=image_filename,
                        reading=clean_r,
                        tenth=corr_tenth.strip(),
                        condition=corr_cond,
                        digit_0=d0,
                        digit_1=d1,
                        digit_2=d2,
                        digit_3=d3,
                        is_template_source=False,  # Strictly False: manual corrections must not alter templates without explicit offline verification
                        notes=corr_notes.strip()
                    )
                    st.success(f"✅ Saved correction for `{image_filename}` $\to$ `{clean_r}{corr_tenth}` ({corr_cond}) into `readings.csv`.")
                else:
                    st.error("Please enter exactly 4 digits (e.g. 0078).")

    # ---------------------------------------------------------
    # JSON Debug View
    # ---------------------------------------------------------
    with st.expander("🔍 View Complete Pipeline JSON Output"):
        clean_res = {
            k: v for k, v in result.items()
            if not isinstance(v, Image.Image) and not (isinstance(v, list) and v and isinstance(v[0], Image.Image))
        }
        st.json(clean_res)
