"""
DialReader: Interactive Web Application
End-to-end mechanical dial counter reader with Method 2 decimal precision.
"""

from pathlib import Path
import sys
import io
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer import DialReadingPipeline

# Page setup
st.set_page_config(
    page_title="DialReader - Mechanical Dial Counter AI",
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
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ff8c33 0%, #ff5500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Digital LCD Counter Display */
    .lcd-container {
        display: flex;
        align-items: center;
        justify-content: center;
        background: #05070a;
        border: 2px solid #30363d;
        border-radius: 14px;
        padding: 20px 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6), inset 0 0 20px rgba(0,0,0,0.8);
        margin: 15px 0 25px 0;
        gap: 10px;
    }
    
    .lcd-slot {
        background: #111418;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 10px 18px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.4rem;
        font-weight: 800;
        color: #f0f6fc;
        text-shadow: 0 0 10px rgba(255,255,255,0.4);
        display: inline-block;
        min-width: 60px;
        text-align: center;
    }
    
    .lcd-slot-red {
        background: linear-gradient(180deg, #b31217 0%, #7d0d10 100%);
        border: 1px solid #ff4d4d;
        color: #ffffff;
        text-shadow: 0 0 12px rgba(255,100,100,0.8);
    }
    
    .lcd-dot {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.4rem;
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
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 12px;
    }
    
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #58a6ff;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-lbl {
        color: #8b949e;
        font-size: 0.85rem;
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
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Initializes and caches the DialReadingPipeline."""
    return DialReadingPipeline(
        detector_weights=None,
        classifier_weights=PROJECT_ROOT / "models" / "best_digit_ann.pt",
        config_path=str(PROJECT_ROOT / "configs" / "digit_ann.yaml"),
        min_digit_conf=0.55
    )


def create_annotated_window(rectified_pil, decimal_info, whole_reading, full_value):
    """Draws reading baseline and Method 2 red digit square overlay."""
    np_img = np.array(rectified_pil.convert("RGB"))
    h, w, _ = np_img.shape
    vis = np_img.copy()

    # 1. Vertical slot boundaries
    usable_w = w * (1.0 - 0.12)
    slot_w = usable_w / 4.0
    for i in range(1, 4):
        x = int(round(i * slot_w))
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
    # Vernier scale dividing line
    cv2.line(vis, (int(usable_w), 0), (int(usable_w), h), (255, 120, 0), 2)

    # 2. Horizontal Reading Baseline
    baseline_y = int(round(decimal_info.get("baseline_y", h / 2.0)))
    cv2.line(vis, (0, baseline_y), (int(usable_w * 0.75), baseline_y), (0, 255, 255), 2)
    cv2.putText(vis, "Baseline", (8, baseline_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # 3. Method 2 Red Drum Bounding Box ("Square Around the Digit")
    bbox = decimal_info.get("digit_bbox")
    if bbox:
        bx, by, bw, bh = bbox
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        digit_cy = int(round(decimal_info.get("digit_cy", by + bh / 2.0)))
        cv2.line(vis, (bx, digit_cy), (bx + bw, digit_cy), (0, 0, 255), 1)
        
        # Shift arrow
        cv2.arrowedLine(vis, (bx + bw + 6, baseline_y), (bx + bw + 6, digit_cy), (0, 255, 255), 1, tipLength=0.3)
        shift_txt = f"{decimal_info.get('displacement_px', 0.0):+.1f}px"
        cv2.putText(vis, shift_txt, (bx + bw + 10, int((baseline_y + digit_cy) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

    return Image.fromarray(vis)


# -------------------------------------------------------------
# Sidebar: Settings & Preloaded Presets
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/speedometer.png", width=64)
    st.markdown("### **DialReader Controls**")
    
    st.markdown("#### **Sample Dials**")
    preset_choice = st.radio(
        "Choose an image source:",
        ["Upload Custom Photo", "Preset 1: Dial 0078 (78.0)", "Preset 2: Dial 0080 (80.2)", "Dataset Sample 1", "Dataset Sample 2"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### **Orientation & Tuning**")
    manual_orient = st.selectbox(
        "Manual Rotation:",
        [None, 0, 90, 180, 270],
        format_func=lambda x: "Auto Detect" if x is None else f"{x}° Rotation"
    )
    
    min_conf = st.slider("Min Digit Confidence Threshold", 0.30, 0.90, 0.55, 0.05)
    
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.82rem; color:#8b949e; line-height:1.4;">
        <b>Technology Stack:</b><br>
        • Stage 1: Display Window Detection<br>
        • Stage 2: 4-Slot Normalizer<br>
        • Stage 3: PyTorch Digit ANN (MLP)<br>
        • Stage 4: <b>Method 2 Decimal Reader</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown('<div class="hero-title">⏱️ DialReader AI Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Automated 4-digit reading with <b>Method 2 pixel-level decimal sub-division</b> for Elesa mechanical position indicators.</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# Image Loading
# -------------------------------------------------------------
input_pil = None
preset_map = {
    "Preset 1: Dial 0078 (78.0)": Path(r"C:\Users\ABHI\.gemini\antigravity-ide\brain\8abd5874-9ef0-49ba-9ca6-2a6566cf2a94\.user_uploaded\media_1792996743107.png"),
    "Preset 2: Dial 0080 (80.2)": Path(r"C:\Users\ABHI\.gemini\antigravity-ide\brain\8abd5874-9ef0-49ba-9ca6-2a6566cf2a94\.user_uploaded\media_1792996750413.png"),
    "Dataset Sample 1": PROJECT_ROOT / "data" / "labeled" / "yolo_dataset" / "images" / "train" / "knife_20261006_083755899_iOS.jpg",
    "Dataset Sample 2": PROJECT_ROOT / "data" / "labeled" / "yolo_dataset" / "images" / "train" / "knife_20261006_083758932_iOS.jpg"
}

if preset_choice == "Upload Custom Photo":
    uploaded_file = st.file_uploader(
        "Upload a photo of your mechanical dial counter:",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded_file is not None:
        input_pil = Image.open(uploaded_file).convert("RGB")
else:
    preset_path = preset_map.get(preset_choice)
    if preset_path and preset_path.exists():
        input_pil = Image.open(preset_path).convert("RGB")
    else:
        st.warning(f"Preset file not found: {preset_path}")

# Fallback default if nothing loaded
if input_pil is None:
    fallback_p1 = Path(r"C:\Users\ABHI\.gemini\antigravity-ide\brain\8abd5874-9ef0-49ba-9ca6-2a6566cf2a94\.user_uploaded\media_1792996750413.png")
    if fallback_p1.exists():
        input_pil = Image.open(fallback_p1).convert("RGB")
        st.info("Loaded default test image (Dial 0080). You can upload your own photo or pick another preset from the left sidebar.")

# -------------------------------------------------------------
# Run Inference
# -------------------------------------------------------------
if input_pil is not None:
    pipeline = load_pipeline()
    pipeline.min_digit_conf = min_conf
    
    with st.spinner("Analyzing dial digits and measuring Method 2 decimal fraction..."):
        start_t = time.time()
        result = pipeline.read_image(input_pil, manual_orientation=manual_orient)
        latency = round((time.time() - start_t) * 1000, 1)

    # ---------------------------------------------------------
    # LCD Hero Display
    # ---------------------------------------------------------
    whole_digits = result.get("whole_digits") or "0000"
    tenth_val = result.get("tenth", 0)
    full_val = result.get("value") or f"{whole_digits}.{tenth_val}"
    status = result.get("status", "ok")
    confidence = result.get("confidence", 0.0)
    decimal_info = result.get("decimal_info", {})
    
    # Render LCD display
    d1 = whole_digits[0] if len(whole_digits) > 0 else "0"
    d2 = whole_digits[1] if len(whole_digits) > 1 else "0"
    d3 = whole_digits[2] if len(whole_digits) > 2 else "0"
    d4 = whole_digits[3] if len(whole_digits) > 3 else "0"
    
    st.markdown(f"""
    <div class="lcd-container">
        <div class="lcd-slot">{d1}</div>
        <div class="lcd-slot">{d2}</div>
        <div class="lcd-slot">{d3}</div>
        <div class="lcd-slot lcd-slot-red">{d4}</div>
        <div class="lcd-dot">.</div>
        <div class="lcd-slot lcd-slot-decimal">{tenth_val}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status badges
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Total Reading</div>
            <div class="metric-val" style="color:#ffaa55;">{full_val}</div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        conf_color = "#00e676" if confidence >= 0.70 else "#ffa726" if confidence >= 0.50 else "#ff5252"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Confidence</div>
            <div class="metric-val" style="color:{conf_color};">{int(confidence * 100)}%</div>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        disp_px = decimal_info.get("displacement_px", 0.0)
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

    # Mid-roll transition notice
    if result.get("is_mid_roll"):
        st.warning("⚠️ **Mid-Roll Drum Transition Detected:** The 4th red drum is rolling between numbers. Method 2 continuously measured the active digit displacement.")

    st.markdown("---")

    # ---------------------------------------------------------
    # Visual Breakdown Columns
    # ---------------------------------------------------------
    col_left, col_right = st.columns([1.1, 1.0])

    with col_left:
        st.markdown("#### **1. Full Input Photo & Detected Window**")
        st.image(input_pil, caption="Input Photo", use_container_width=True)

    with col_right:
        st.markdown("#### **2. Method 2 Inspection: Rectified Window**")
        rectified_window = result.get("rectified_window")
        if rectified_window is not None:
            annotated = create_annotated_window(
                rectified_window,
                decimal_info,
                whole_digits,
                full_val
            )
            st.image(
                annotated,
                caption="Rectified Window: Yellow=Baseline, Blue=Digit Square, Red=Digit Centroid, Green=Slot Dividers",
                use_container_width=True
            )
        
        # Method 2 metrics explanation card
        raw_tenth = decimal_info.get("raw_tenth", 0.0)
        base_y = decimal_info.get("baseline_y", 64.0)
        digit_cy = decimal_info.get("digit_cy", 64.0)
        bbox = decimal_info.get("digit_bbox", [0, 0, 0, 0])
        
        st.markdown(f"""
        <div style="background:#161b22; border-left:4px solid #ff6600; padding:12px 16px; border-radius:6px; font-size:0.9rem; color:#c9d1d9;">
            <b>Method 2 Pixel Calculations:</b><br>
            • <b>Calibrated Reading Baseline:</b> <code>Y = {base_y} px</code><br>
            • <b>White Digit 0 Bounding Box:</b> <code>X={bbox[0]}, Y={bbox[1]}, W={bbox[2]}, H={bbox[3]}</code><br>
            • <b>Digit Centroid:</b> <code>Y = {digit_cy} px</code><br>
            • <b>Displacement (ΔY):</b> <code>{base_y} - {digit_cy} = {disp_px:+.1f} px</code><br>
            • <b>Calculated Tenth:</b> <code>round({raw_tenth}) = {tenth_val}</code> → <b><code>.{tenth_val}</code></b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # 4 Discrete Digit Slot Breakdown
    # ---------------------------------------------------------
    st.markdown("#### **3. Individual Discrete Digit Slots (Stage 2: PyTorch ANN)**")
    digit_slots = result.get("digit_slots", [])
    per_digit_conf = result.get("per_digit_confidence", [1.0, 1.0, 1.0, 1.0])
    
    if len(digit_slots) == 4:
        sc1, sc2, sc3, sc4 = st.columns(4)
        cols = [sc1, sc2, sc3, sc4]
        titles = ["Slot 1 (Integer 1)", "Slot 2 (Integer 2)", "Slot 3 (Integer 3)", "Slot 4 (Red Drum)"]
        
        for i in range(4):
            with cols[i]:
                conf_i = per_digit_conf[i] if i < len(per_digit_conf) else 1.0
                st.markdown(f"""
                <div class="slot-card">
                    <div style="font-size:0.8rem; color:#8b949e; margin-bottom:6px;">{titles[i]}</div>
                    <div style="font-size:2.2rem; font-weight:800; color:{'#ff4d4d' if i==3 else '#f0f6fc'}; font-family:'JetBrains Mono';">
                        {whole_digits[i] if i < len(whole_digits) else '-'}
                    </div>
                    <div style="font-size:0.75rem; color:#8b949e; margin-top:4px;">Conf: {int(conf_i * 100)}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(digit_slots[i], use_container_width=True)
                st.progress(float(np.clip(conf_i, 0.0, 1.0)))

    # ---------------------------------------------------------
    # JSON Debug View
    # ---------------------------------------------------------
    with st.expander("🔍 View Complete Pipeline JSON Output"):
        # Create a serializable copy of result (without PIL images)
        clean_res = {k: v for k, v in result.items() if k not in ["window_crop", "rectified_window", "digit_slots"]}
        st.json(clean_res)

else:
    st.info("Please upload an image or choose one of the sample presets to begin.")
