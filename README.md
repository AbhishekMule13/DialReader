# DialReader: Orange Mechanical Dial Number Reader

An end-to-end computer vision and machine learning pipeline for reading 4-digit values from orange mechanical position-indicator dials (rolling-drum counters) with glare, perspective variations, and mid-roll drum transitions.

---

## 1. Project Architecture (Two Stages)

```
Input Photo ──> Stage 1: YOLO Detector (display_window) 
                  │
                  ▼
            Window Crop ──> Rectify Orientation & Extract 4 Discrete Slots
                              │
                              ▼
                     Stage 2: ANN Digit Classifier (MLP)
                     [Run 4 times: Slot 1, Slot 2, Slot 3, Slot 4]
                              │
                              ▼
                     Output: {"value": "0021", "confidence": 0.96, "status": "ok"}
```

1. **Stage 1: Detector (YOLOv8)**:
   - Fine-tuned on the single class `display_window` to find the rectangular window.
2. **Stage 2: Reader (ANN / Multi-Layer Perceptron)**:
   - **Crop Generation**: Splits rectified windows into 4 discrete digit slots (accounting for the fractional scale).
   - **Single-Digit ANN**: Classifies ONE slot crop (0–9) at a time. The pipeline runs it 4 times per dial.
   - **Configuration**: All architecture parameters (layers, activation, batchnorm, dropout, learning rate, scheduler, early stopping) live in `configs/digit_ann.yaml`.
   - **Augmentation**: Photometric jitter and small rotation (+-8°). **STRICT RULE**: No horizontal or vertical flips.
   - **Validation Integrity**: Validation and testing are performed strictly on real images. Video frames and synthetic crops are only used for training augmentation.

---

## 2. Directory Structure

```text
dial-reader/
├── configs/
│   ├── digit_ann.yaml           # Centralized ANN hyperparameters & architecture
│   └── detector.yaml            # YOLO dataset paths and class configuration
├── data/
│   ├── raw/
│   │   ├── knife/               # Standardized Knife photos (train/val pool)
│   │   ├── knife_pressing/      # Knife+Pressing photos (held-out test set & negatives)
│   │   └── knife_video_frames/  # Sparsely mined video keyframes (train only)
│   ├── labeled/
│   │   ├── labels.csv           # Master labels CSV (one row per dial window)
│   │   └── yolo_dataset/        # Auto-exported YOLO bounding box labels
│   ├── digits/                  # Generated discrete digit crops
│   │   ├── train/0..9/          # Training crops (Knife photos + video frames)
│   │   ├── val/0..9/            # Validation crops (Knife photos only)
│   │   └── test/0..9/           # Held-out test crops (Knife+Pressing only)
│   └── synthetic/               # Synthetic rolling-drum pretraining samples
├── src/
│   ├── __init__.py
│   ├── models.py                # DigitANN architecture, transforms, DigitANNPredictor
│   ├── train_digits.py          # ANN training pipeline (reads digit_ann.yaml)
│   ├── run_experiments.py       # Systematic hyperparameter comparison suite
│   ├── detect.py                # YOLOv8 display_window detector wrapper
│   ├── train_detector.py        # Detector training script
│   ├── crop.py                  # WindowNormalizer & 4-slot extractor
│   ├── infer.py                 # End-to-end inference CLI (Image -> JSON)
│   ├── eval.py                  # Held-out test evaluation, confusion matrices, failure gallery
│   └── utils/
│       ├── generate_digit_crops.py   # Turns labeled windows into data/digits/
│       ├── create_labels_template.py # Master CSV template generator
│       ├── export_yolo.py            # YOLO format dataset exporter
│       └── mine_video.py             # Sparse video keyframe extractor
├── notebooks/
│   └── dialreader_walkthrough.ipynb  # Step-by-step pipeline walkthrough notebook
├── results/
│   └── experiments.csv          # Architecture comparison table
├── models/                      # Saved PyTorch (.pt) weights
├── runs/eval/                   # Confusion matrices and failure image gallery
├── label_ui.py                  # Interactive Streamlit labeling helper UI
├── requirements.txt             # Pinned dependencies
└── README.md                    # Exact commands and instructions
```

---

## 3. Step-by-Step Workflow & Exact Commands

### Step 1: Label Images (Helper UI)
Launch the interactive Streamlit labeling interface:
```bash
cd dial-reader
py -3.10 -m streamlit run label_ui.py
```
- Draw bounding boxes around the glossy `display_window`.
- Type the 4-digit reading (e.g. `0021`).
- Multi-dial images (e.g. `103,104.jpeg`): Click **"Add Another Dial for this Image"** to record each window.
- Negative images (e.g. `116.jpeg`, `117.jpeg`): Pre-marked as negative; leave bbox empty.
- Labels are saved directly to `data/labeled/labels.csv`.

---

### Step 2: Extract Digit Crops
Once labeling is completed, turn labeled windows into discrete slot crops sorted into `data/digits/{train,val,test}/{0..9}/`:
```bash
py -3.10 src/utils/generate_digit_crops.py
```
*Splits by source (Knife -> train/val, Knife+Pressing -> held-out test). Video frames are strictly kept in train.*

---

### Step 3: Train YOLO Window Detector (Stage 1)
Train the YOLOv8 display window locator:
```bash
py -3.10 src/train_detector.py --config configs/detector.yaml --epochs 40 --batch 8
```
*Best detector weights saved to `models/detector_best.pt`.*

---

### Step 4: Train ANN Digit Classifier (Stage 2)
Train the Fully-Connected Network (MLP) on the extracted slot crops:
```bash
py -3.10 src/train_digits.py --config configs/digit_ann.yaml
```
- Configuration is loaded from `configs/digit_ann.yaml`.
- You can tune hidden layers, activation, dropout, and learning rate directly in YAML without modifying Python code.
- Best model weights saved to `models/digit_ann_best.pt`.

---

### Step 5: Run Systematic Architecture Experiments
Sweep across 10 configurations (varying one factor at a time against the baseline) to compare performance:
```bash
py -3.10 src/run_experiments.py
```
*Logs metrics (`train_acc`, `val_acc`, `val_loss`, `param_count`, `best_epoch`) into `results/experiments.csv`.*

To verify architectures in dry-run mode without training:
```bash
py -3.10 src/run_experiments.py --dry_run
```

---

### Step 6: Interactive Walkthrough Notebook
Open and execute the step-by-step walkthrough in Jupyter:
```bash
py -3.10 -m jupyter notebook notebooks/dialreader_walkthrough.ipynb
```
Explains each step for someone who knows basic ANNs: loads data, previews crops, walks through parameter counts, trains the ANN, plots loss/accuracy curves, plots confusion matrices, and runs sample inference.

---

### Step 7: Evaluate on Held-Out Test Set (Knife+Pressing)
Evaluate the complete end-to-end pipeline:
```bash
py -3.10 src/eval.py
```
**Evaluation outputs:**
1. Full 4-digit exact match accuracy on the Knife+Pressing test set.
2. Per-digit position accuracy (slots 1, 2, 3, 4).
3. Last-digit (mid-roll drum) error rate reported separately.
4. Negative sample (116/117) rejection rate.
5. Saves 4 confusion matrices to `runs/eval/confusion_matrices.png`.
6. Saves the 20 worst failure images with predicted vs true labels annotated to `runs/eval/failures/`.

---

### Step 8: End-to-End Single Image Inference
Run inference on an arbitrary dial photograph:
```bash
py -3.10 src/infer.py --image "data/raw/knife/20261006_083755899_iOS.jpg"
```
**JSON Output Format:**
```json
{
  "value": "0045",
  "confidence": 0.961,
  "status": "ok",
  "per_digit_confidence": [0.99, 0.98, 0.97, 0.91],
  "is_mid_roll": false,
  "detector_confidence": 0.97,
  "dial_variant": "standard_3black_1red"
}
```
