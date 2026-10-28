"""
Stage 3: Classical Computer Vision Digit Recognizer for DialReader

Replaces all ANN/CNN/deep-learning digit classification with a pure traditional
computer vision approach using:
  - Binarization and connected component segmentation
  - Aspect-preserving canonical normalization
  - Geometric pixel feature extraction (holes, projections, aspect ratio, crossbar, density)
  - Normalized template matching with binary IoU and pixel correlation
  - Ambiguity rejection and missing-template diagnostics
  - 100% reproducible template registration

Strictly ZERO neural network dependencies (no PyTorch, no torchvision, no OCR engines).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


@dataclass
class DigitTemplate:
    """
    Representation of a single digit template.
    """
    digit: int
    mask: np.ndarray  # Normalized binary mask (uint8: 0 or 255)
    features: Dict[str, Any]
    source: str
    verified: bool = True
    condition: str = "stationary"
    image_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "digit": self.digit,
            "source": self.source,
            "verified": self.verified,
            "condition": self.condition,
            "features": {k: v for k, v in self.features.items() if not isinstance(v, np.ndarray)}
        }
        if self.image_hash:
            d["image_hash"] = self.image_hash
        return d


class DigitTemplateLibrary:
    """
    Reproducible template bank for digits 0-9.
    Tracks provenance, verified status, and handles serialization.
    """

    def __init__(self, canonical_size: Tuple[int, int] = (48, 32)):
        self.canonical_size = canonical_size
        # Dictionary mapping digit (0..9) to a list of DigitTemplate instances
        self.templates: Dict[int, List[DigitTemplate]] = {d: [] for d in range(10)}

    def register_template(
        self,
        digit: int,
        mask: np.ndarray,
        source: str,
        features: Optional[Dict[str, Any]] = None,
        verified: bool = True,
        condition: str = "stationary",
        image_hash: Optional[str] = None,
        deduplicate_identical: bool = True
    ) -> DigitTemplate:
        """
        Registers a normalized binary mask as a template for a digit.
        Supports digits 0 through 9 and allows multiple verified templates per digit.
        If deduplicate_identical is True and an exact identical binary mask is already
        registered for this digit, returns the existing template without inflating counts.
        """
        if not (0 <= digit <= 9):
            raise ValueError(f"Digit must be between 0 and 9, got {digit}")

        # Ensure correct shape and dtype
        if mask.shape != self.canonical_size:
            mask = cv2.resize(mask, (self.canonical_size[1], self.canonical_size[0]), interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask > 0).astype(np.uint8) * 255

        # Strict safety gate: verified templates must be stationary and have valid non-empty strokes
        if verified:
            if condition != "stationary":
                raise ValueError(f"Verified template must have condition='stationary', got '{condition}'")
            white_px = int(np.sum(binary_mask > 0))
            if white_px < 20:
                raise ValueError(f"Verified template mask too sparse: {white_px} white pixels")

        # Deduplicate identical masks for the same digit to prevent false count inflation
        if deduplicate_identical:
            for existing in self.templates[digit]:
                if np.array_equal(existing.mask, binary_mask):
                    return existing

        if features is None:
            features = PixelDigitRecognizer.extract_geometric_features(binary_mask)

        tpl = DigitTemplate(
            digit=digit,
            mask=binary_mask,
            features=features,
            source=source,
            verified=verified,
            condition=condition,
            image_hash=image_hash
        )
        self.templates[digit].append(tpl)
        return tpl

    def available_digits(self) -> List[int]:
        """Returns list of digits that have at least one registered template."""
        return [d for d in range(10) if len(self.templates[d]) > 0]

    def missing_digits(self) -> List[int]:
        """Returns list of digits that have no registered template."""
        return [d for d in range(10) if len(self.templates[d]) == 0]

    def has_template(self, digit: int) -> bool:
        return len(self.templates.get(digit, [])) > 0

    def get_templates(self, digit: int, verified_only: bool = False) -> List[DigitTemplate]:
        tpls = self.templates.get(digit, [])
        if verified_only:
            return [t for t in tpls if t.verified]
        return tpls

    def get_template_count(self, digit: int, verified_only: bool = True) -> int:
        """Returns the number of registered templates for a digit."""
        return len(self.get_templates(digit, verified_only=verified_only))

    def get_independent_source_count(self, digit: int, verified_only: bool = True) -> int:
        """
        Returns the number of distinct physical source images for a digit.
        Resolves to file SHA-256 so duplicate files (e.g. 80.2.png vs dial_0080.png)
        do not falsely inflate independent photograph counts.
        """
        import hashlib
        tpls = self.get_templates(digit, verified_only=verified_only)
        hashes = set()
        for t in tpls:
            if t.image_hash:
                hashes.add(t.image_hash)
            else:
                parts = t.source.split(":")
                fn = parts[1] if len(parts) >= 2 else t.source
                resolved_hash = None
                for base_dir in [Path("data/presets"), Path("data/raw"), Path("../data/presets"), Path("../data/raw")]:
                    candidate = base_dir / fn
                    if candidate.exists():
                        try:
                            with open(candidate, "rb") as f:
                                resolved_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                            break
                        except Exception:
                            pass
                hashes.add(resolved_hash if resolved_hash else fn)
        return len(hashes)

    def independent_sources_by_digit(self, verified_only: bool = True) -> Dict[int, int]:
        """Returns mapping of digit -> count of independent source photographs."""
        return {d: self.get_independent_source_count(d, verified_only=verified_only) for d in range(10)}

    def audit(self) -> Dict[str, Any]:
        """
        Performs an authoritative audit of the template library:
        - Templates available for each digit 0-9
        - Number of independent source photographs for each digit
        - Missing digits and explicit reasons
        - Human confirmation verification
        - Duplicate prevention status
        """
        tpl_counts = {d: self.get_template_count(d, verified_only=True) for d in range(10)}
        source_counts = self.independent_sources_by_digit(verified_only=True)
        available = self.available_digits()
        missing = self.missing_digits()

        missing_reasons = {
            d: f"No human-verified stationary photographs containing digit {d} have been labeled in readings.csv"
            for d in missing
        }

        all_human_confirmed = True
        for d in available:
            for t in self.get_templates(d, verified_only=True):
                if not (t.source.startswith("verified_preset:") or t.source.startswith("verified_real:")):
                    all_human_confirmed = False

        return {
            "available_digits": available,
            "missing_digits": missing,
            "template_counts": tpl_counts,
            "independent_sources": source_counts,
            "missing_reasons": missing_reasons,
            "all_exported_human_confirmed": all_human_confirmed,
            "deduplication_active": True
        }

    def load_verified_file_templates(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads verified digit templates from a curated JSON or NPZ file.
        Strictly excludes unverified crops or rolling/reflective/unreadable crops.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return {"loaded": 0, "status": "file_not_found"}

        registered = []
        try:
            if filepath.suffix.lower() == ".json":
                with open(filepath, "r", encoding="utf-8") as f:
                    items = json.load(f)
                for item in items:
                    d = item.get("digit")
                    if d is None or not (0 <= int(d) <= 9):
                        continue
                    if not item.get("verified", False):
                        continue
                    if item.get("condition", "stationary") != "stationary":
                        continue
                    mask_list = item.get("mask")
                    if mask_list is None:
                        continue
                    mask = np.array(mask_list, dtype=np.uint8)
                    src = item.get("source", f"file:{filepath.name}:digit_{d}")
                    img_hash = item.get("image_hash")
                    self.register_template(
                        digit=int(d),
                        mask=mask,
                        source=src,
                        features=item.get("features"),
                        verified=True,
                        condition="stationary",
                        image_hash=img_hash,
                        deduplicate_identical=True
                    )
                    registered.append((int(d), src))
            elif filepath.suffix.lower() == ".npz":
                self.load_npz(filepath)
                registered = [(d, t.source) for d, tpls in self.templates.items() for t in tpls if t.verified]
        except Exception as e:
            print(f"[DigitTemplateLibrary] Could not load templates from {filepath}: {e}")

        return {
            "registered": registered,
            "available_digits": self.available_digits(),
            "missing_digits": self.missing_digits()
        }

    def load_verified_preset_templates(self, project_root: Optional[Path] = None) -> Dict[str, Any]:
        """
        Extracts verified templates from existing preset images with known ground truth readings:
          - data/presets/dial_0078.png -> Reading 0078 (slot 0=0, slot 1=0, slot 2=7, slot 3=8)
          - data/presets/dial_0080.png -> Reading 0080 (slot 0=0, slot 1=0, slot 2=8, slot 3=0)

        Also loads from data/labeled/verified_templates.json if present.
        Does NOT guess unlabeled images. Clearly documents sources.
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent

        registered = []
        preset_sources = [
            ("dial_0078.png", project_root / "data" / "presets" / "dial_0078.png", [0, 0, 7, 8]),
            ("dial_0080.png", project_root / "data" / "presets" / "dial_0080.png", [0, 0, 8, 0]),
        ]

        # Use WindowNormalizer to extract slots from presets
        from src.crop import WindowNormalizer
        normalizer = WindowNormalizer()

        import hashlib

        for filename, path, expected_digits in preset_sources:
            if not path.exists():
                continue
            try:
                # Compute image hash for provenance tracking
                with open(path, "rb") as f:
                    img_hash = hashlib.sha256(f.read()).hexdigest()[:16]

                img = Image.open(path).convert("RGB")
                proc = normalizer.process(img)
                slots = proc.get("digit_slots", [])
                if len(slots) != 4:
                    continue

                for slot_idx in range(4):
                    digit_label = expected_digits[slot_idx]
                    slot_crop = slots[slot_idx]
                    mask, feats = PixelDigitRecognizer.extract_normalized_digit_mask(
                        slot_crop,
                        target_size=self.canonical_size
                    )
                    if mask is not None and np.sum(mask > 0) > 20:
                        src_desc = f"verified_preset:{filename}:slot_{slot_idx}:digit_{digit_label}"
                        self.register_template(
                            digit=digit_label,
                            mask=mask,
                            source=src_desc,
                            features=feats,
                            verified=True,
                            condition="stationary",
                            image_hash=img_hash,
                            deduplicate_identical=True
                        )
                        registered.append((digit_label, src_desc))
            except Exception as e:
                print(f"[DigitTemplateLibrary] Could not load templates from {filename}: {e}")

        # Check for curated verified templates file
        verified_file = project_root / "data" / "labeled" / "verified_templates.json"
        if verified_file.exists():
            file_res = self.load_verified_file_templates(verified_file)
            registered.extend(file_res.get("registered", []))

        return {
            "registered": registered,
            "available_digits": self.available_digits(),
            "missing_digits": self.missing_digits()
        }

    def load_synthetic_test_templates(self) -> List[int]:
        """
        Renders clean synthetic binary masks for digits 0-9 using standard system fonts.
        Clearly marked verified=False. Used for unit tests and fallback testing.
        """
        font = None
        font_names = ["arial.ttf", "calibri.ttf", "DejaVuSans.ttf", "tahoma.ttf"]
        th, tw = self.canonical_size
        font_size = int(th * 0.75)

        for fn in font_names:
            try:
                font = ImageFont.truetype(fn, font_size)
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()

        registered = []
        for d in range(10):
            # Render numeral on canvas
            img = Image.new("L", (tw * 2, th * 2), color=0)
            draw = ImageDraw.Draw(img)
            draw.text((tw, th), str(d), fill=255, font=font, anchor="mm")
            arr = np.array(img)

            # Crop tightly to numeral
            white_pts = np.argwhere(arr > 100)
            if len(white_pts) == 0:
                continue
            y_min, x_min = white_pts.min(axis=0)
            y_max, x_max = white_pts.max(axis=0)
            crop = arr[y_min : y_max + 1, x_min : x_max + 1]

            # Aspect-preserving scale into canonical size
            bh, bw = crop.shape
            pad = 2
            scale = min((th - 2 * pad) / float(bh), (tw - 2 * pad) / float(bw))
            nw = max(1, int(round(bw * scale)))
            nh = max(1, int(round(bh * scale)))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)

            canvas = np.zeros((th, tw), dtype=np.uint8)
            yo = (th - nh) // 2
            xo = (tw - nw) // 2
            canvas[yo : yo + nh, xo : xo + nw] = (resized > 100).astype(np.uint8) * 255

            feats = PixelDigitRecognizer.extract_geometric_features(canvas)
            self.register_template(
                digit=d,
                mask=canvas,
                source=f"synthetic_font:digit_{d}",
                features=feats,
                verified=False
            )
            registered.append(d)

        return registered

    def save_npz(self, filepath: Union[str, Path]):
        """Saves template bank to a compressed NPZ file with metadata."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data_dict = {}
        meta = []

        idx = 0
        for d, tpls in self.templates.items():
            for tpl in tpls:
                key = f"tpl_{idx:03d}_d{d}"
                data_dict[key] = tpl.mask
                meta.append({
                    "key": key,
                    "digit": d,
                    "source": tpl.source,
                    "verified": tpl.verified,
                    "features": tpl.to_dict()["features"]
                })
                idx += 1

        data_dict["metadata_json"] = np.array(json.dumps(meta))
        np.savez_compressed(filepath, **data_dict)

    def load_npz(self, filepath: Union[str, Path]):
        """Loads template bank from NPZ file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Template file not found: {filepath}")

        loaded = np.load(filepath, allow_pickle=True)
        if "metadata_json" in loaded:
            meta = json.loads(str(loaded["metadata_json"]))
            for item in meta:
                key = item["key"]
                if key in loaded:
                    mask = loaded[key]
                    self.register_template(
                        digit=int(item["digit"]),
                        mask=mask,
                        source=item["source"],
                        features=item.get("features"),
                        verified=bool(item.get("verified", True)),
                        condition=str(item.get("condition", "stationary"))
                    )


class PixelDigitRecognizer:
    """
    Classical Computer Vision Digit Recognizer.
    Replaces DigitANNPredictor without any neural network dependencies.

    Combines:
    1. Binarization and connected component segmentation
    2. Aspect-preserving canonical normalization (default 48x32)
    3. Geometric feature extraction (holes, aspect ratio, density, projections, crossbar)
    4. Template matching with binary IoU and pixel agreement
    5. Scoring, ambiguity margin, and missing template rejection
    """

    def __init__(
        self,
        template_library: Optional[DigitTemplateLibrary] = None,
        min_confidence: float = 0.35,
        ambiguity_margin: float = 0.08,
        canonical_size: Tuple[int, int] = (48, 32),
        auto_load_verified: bool = True,
        project_root: Optional[Path] = None
    ):
        self.canonical_size = canonical_size
        self.min_confidence = min_confidence
        self.ambiguity_margin = ambiguity_margin

        if template_library is not None:
            self.templates = template_library
        else:
            self.templates = DigitTemplateLibrary(canonical_size=self.canonical_size)
            if auto_load_verified:
                self.templates.load_verified_preset_templates(project_root=project_root)

    # -------------------------------------------------------------------------
    # 1. Binarization and Normalization
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_normalized_digit_mask(
        slot_img: Union[Image.Image, np.ndarray],
        target_size: Tuple[int, int] = (48, 32)
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Converts a digit slot crop into a normalized binary mask preserving aspect ratio.
        Works for both black-drum (dark background) and red-drum (red background).

        Returns:
            (normalized_binary_mask, geometric_features)
            If no digit is found, returns (None, empty_features).
        """
        if isinstance(slot_img, Image.Image):
            rgb = np.array(slot_img.convert("RGB"))
        else:
            rgb = slot_img if slot_img.ndim == 3 else cv2.cvtColor(slot_img, cv2.COLOR_GRAY2RGB)

        h, w = rgb.shape[:2]
        if h < 10 or w < 5:
            return None, {"status": "too_small"}

        # White numeral has high R, G, B on both black drums and red drums.
        # Red drum background has high R, but low G and low B.
        # Black drum background has low R, G, B.
        # Therefore, min(R, G, B) isolates white numerals cleanly in both drum types.
        min_rgb = np.minimum(np.minimum(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2])

        # Otsu thresholding with bounds to handle contrast variations
        otsu_val, _ = cv2.threshold(min_rgb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_val = int(np.clip(otsu_val, 75, 155))
        _, binary = cv2.threshold(min_rgb, thresh_val, 255, cv2.THRESH_BINARY)

        # Morphological noise removal
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # Connected component analysis to find primary centered digit
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        candidates = []

        for j in range(1, num_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            bx, by, bw, bh = (
                int(stats[j, cv2.CC_STAT_LEFT]),
                int(stats[j, cv2.CC_STAT_TOP]),
                int(stats[j, cv2.CC_STAT_WIDTH]),
                int(stats[j, cv2.CC_STAT_HEIGHT]),
            )
            cx, cy = centroids[j]

            # Filter out slot divider vertical lines at boundary
            if bw < 10 and (bx <= 2 or bx + bw >= w - 2):
                continue

            # Candidate must have reasonable height and area
            if bh >= max(10, int(0.22 * h)) and bw >= 4 and area >= 40:
                dist_center_y = abs(cy - (h / 2.0))
                # Primary numeral score: large area close to vertical center
                score = area / (1.0 + 2.5 * (dist_center_y / float(h)))
                candidates.append((j, score, area, (bx, by, bw, bh), (cx, cy)))

        if not candidates:
            # Fallback: largest component
            best_area = 0
            best_j = 0
            for j in range(1, num_labels):
                if stats[j, cv2.CC_STAT_AREA] > best_area:
                    best_area = stats[j, cv2.CC_STAT_AREA]
                    best_j = j
            if best_j > 0 and best_area > 30:
                bx, by, bw, bh = (
                    int(stats[best_j, cv2.CC_STAT_LEFT]),
                    int(stats[best_j, cv2.CC_STAT_TOP]),
                    int(stats[best_j, cv2.CC_STAT_WIDTH]),
                    int(stats[best_j, cv2.CC_STAT_HEIGHT]),
                )
                raw_crop = (labels[by : by + bh, bx : bx + bw] == best_j).astype(np.uint8) * 255
            else:
                return None, {"status": "no_digit_candidate"}
        else:
            best_cand = max(candidates, key=lambda c: c[1])
            best_j = best_cand[0]
            bx, by, bw, bh = best_cand[3]
            raw_crop = (labels[by : by + bh, bx : bx + bw] == best_j).astype(np.uint8) * 255

        # Morphological closing (3x3) to bridge minor breaks in mechanical engraving strokes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_crop = cv2.morphologyEx(raw_crop, cv2.MORPH_CLOSE, kernel_close)

        # Aspect-preserving normalization to target canvas size
        th, tw = target_size
        pad = 2
        scale = min((th - 2 * pad) / float(bh), (tw - 2 * pad) / float(bw))
        nw = max(1, int(round(bw * scale)))
        nh = max(1, int(round(bh * scale)))
        resized = cv2.resize(closed_crop, (nw, nh), interpolation=cv2.INTER_NEAREST)

        canvas = np.zeros((th, tw), dtype=np.uint8)
        yo = (th - nh) // 2
        xo = (tw - nw) // 2
        canvas[yo : yo + nh, xo : xo + nw] = (resized > 0).astype(np.uint8) * 255

        # Compute geometric features
        features = PixelDigitRecognizer.extract_geometric_features(canvas, raw_bbox=(bx, by, bw, bh))

        return canvas, features

    # -------------------------------------------------------------------------
    # 2. Geometric Feature Extraction
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_geometric_features(
        mask: np.ndarray,
        raw_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Calculates geometric and pixel properties from a normalized digit mask:
          - Bounding box dimensions & aspect ratio
          - White-pixel density (fill factor)
          - Horizontal & vertical pixel projections
          - Enclosed loops/holes (Euler properties)
          - Middle crossbar presence
          - Top-half vs bottom-half balance
        """
        th, tw = mask.shape[:2]
        bin_mask = (mask > 0).astype(np.uint8)

        # Find tight bounding box inside normalized canvas
        white_pts = np.argwhere(bin_mask > 0)
        if len(white_pts) == 0:
            return {
                "bbox": raw_bbox or (0, 0, 0, 0),
                "aspect_ratio": 0.0,
                "density": 0.0,
                "holes": 0,
                "has_crossbar": False,
                "white_pixels": 0,
                "top_half_density": 0.0,
                "bottom_half_density": 0.0,
                "h_proj": [],
                "v_proj": []
            }

        y_min, x_min = white_pts.min(axis=0)
        y_max, x_max = white_pts.max(axis=0)
        norm_bw = int(max(1, x_max - x_min + 1))
        norm_bh = int(max(1, y_max - y_min + 1))
        aspect_ratio = float(round(norm_bw / float(norm_bh), 3))

        total_white = int(np.sum(bin_mask))
        density = float(round(total_white / float(norm_bw * norm_bh), 3))

        # Enclosed holes using RETR_CCOMP
        contours, hierarchy = cv2.findContours(bin_mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        num_holes = 0
        if hierarchy is not None:
            num_holes = int(sum(1 for h_info in hierarchy[0] if h_info[3] != -1))

        # Horizontal and vertical projections
        h_proj = [int(v) for v in np.sum(bin_mask, axis=1)]
        v_proj = [int(v) for v in np.sum(bin_mask, axis=0)]

        # Top vs bottom density balance
        mid_y = th // 2
        top_white = int(np.sum(bin_mask[:mid_y, :]))
        bottom_white = int(np.sum(bin_mask[mid_y:, :]))
        top_density = float(round(top_white / float(max(1, total_white)), 3))
        bottom_density = float(round(bottom_white / float(max(1, total_white)), 3))

        # Middle horizontal crossbar (characteristic of digit 8, absent in 0)
        crossbar_slice = bin_mask[max(0, mid_y - 2) : min(th, mid_y + 3), int(tw * 0.3) : int(tw * 0.7)]
        has_crossbar = bool(np.mean(crossbar_slice) > 0.45) if crossbar_slice.size > 0 else False

        raw_bx, raw_by, raw_bw, raw_bh = raw_bbox if raw_bbox else (int(x_min), int(y_min), int(norm_bw), int(norm_bh))
        clean_bbox = (int(raw_bx), int(raw_by), int(raw_bw), int(raw_bh))

        return {
            "bbox": clean_bbox,
            "norm_bbox": (int(x_min), int(y_min), int(norm_bw), int(norm_bh)),
            "aspect_ratio": aspect_ratio,
            "density": density,
            "holes": num_holes,
            "has_crossbar": has_crossbar,
            "white_pixels": total_white,
            "top_half_density": top_density,
            "bottom_half_density": bottom_density,
            "h_proj": h_proj,
            "v_proj": v_proj,
            "num_contours": int(len(contours))
        }

    # -------------------------------------------------------------------------
    # 3. Template Comparison & Geometric Consistency
    # -------------------------------------------------------------------------
    def compare_mask_to_template(
        self,
        candidate_mask: np.ndarray,
        candidate_feats: Dict[str, Any],
        template: DigitTemplate
    ) -> Tuple[float, float, float]:
        """
        Calculates similarity between candidate mask and a template.
        Returns:
            (combined_score, raw_iou, geometric_penalty_multiplier)
        """
        t_mask = template.mask
        cand_mask = candidate_mask

        # 1. Binary Intersection-over-Union (IoU)
        intersection = int(np.sum((cand_mask > 0) & (t_mask > 0)))
        union = int(np.sum((cand_mask > 0) | (t_mask > 0)))
        iou = intersection / float(union) if union > 0 else 0.0

        # 2. Pixel Agreement / Correlation
        match = cv2.matchTemplate(cand_mask.astype(np.float32), t_mask.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        corr = max(0.0, float(match[0, 0])) if match.size > 0 else 0.0

        raw_score = 0.6 * iou + 0.4 * corr

        # 3. Geometric consistency rules (penalize structural contradictions)
        geo_factor = 1.0
        digit = template.digit
        cand_holes = candidate_feats.get("holes", 0)
        cand_crossbar = candidate_feats.get("has_crossbar", False)
        cand_asp = candidate_feats.get("aspect_ratio", 0.7)
        top_density = candidate_feats.get("top_half_density", 0.5)
        bottom_density = candidate_feats.get("bottom_half_density", 0.5)

        # Digit 8: expects 2 holes and a central crossbar
        if digit == 8:
            if cand_holes != 2:
                geo_factor *= 0.4
            if not cand_crossbar:
                geo_factor *= 0.5

        # Digit 0: expects 1 hole and NO central crossbar
        elif digit == 0:
            if cand_holes != 1:
                geo_factor *= 0.5
            if cand_crossbar:
                geo_factor *= 0.4

        # Digit 7: expects 0 holes, top-heavy, and not extremely narrow
        elif digit == 7:
            if cand_holes != 0:
                geo_factor *= 0.3
            if top_density < 0.45:
                geo_factor *= 0.6
            if cand_asp < 0.48:
                geo_factor *= 0.5

        # Digit 1: expects narrow aspect ratio and 0 holes
        elif digit == 1:
            if cand_asp > 0.52:
                geo_factor *= 0.4
            if cand_holes > 0:
                geo_factor *= 0.2
            if cand_crossbar:
                geo_factor *= 0.5

        # Digits 2, 3, 5: expect 0 holes
        elif digit in (2, 3, 5):
            if cand_holes > 0:
                geo_factor *= 0.4
            if cand_asp < 0.45:
                geo_factor *= 0.5

        # Digit 4: expects 1 hole and crossbar
        elif digit == 4:
            if cand_holes != 1:
                geo_factor *= 0.5
            if not cand_crossbar:
                geo_factor *= 0.5

        # Digit 6: expects 1 hole located in bottom half
        elif digit == 6:
            if cand_holes != 1:
                geo_factor *= 0.5
            if bottom_density < 0.48:
                geo_factor *= 0.5

        # Digit 9: expects 1 hole located in top half
        elif digit == 9:
            if cand_holes != 1:
                geo_factor *= 0.5
            if top_density < 0.48:
                geo_factor *= 0.5

        combined_score = round(float(raw_score * geo_factor), 4)
        return combined_score, round(float(iou), 4), round(float(geo_factor), 4)

    # -------------------------------------------------------------------------
    # 4. Slot & Dial Inference
    # -------------------------------------------------------------------------
    def predict_slot(
        self,
        slot_crop: Union[Image.Image, np.ndarray],
        slot_index: int = 0
    ) -> Dict[str, Any]:
        """
        Classifies ONE digit slot crop (0-9).
        Returns candidate rankings, scores, geometric features, and diagnostic status.
        """
        mask, feats = self.extract_normalized_digit_mask(slot_crop, target_size=self.canonical_size)

        if mask is None:
            return {
                "digit": 0,
                "confidence": 0.0,
                "best_score": 0.0,
                "second_score": 0.0,
                "ambiguity_margin": 0.0,
                "status": "no_digit",
                "reason": "No valid white numeral detected in slot",
                "geometric_features": feats,
                "candidate_ranking": [],
                "all_probs": [0.0] * 10,
                "slot_index": slot_index
            }

        # Compare against all registered templates
        scores_by_digit: Dict[int, float] = {}
        ious_by_digit: Dict[int, float] = {}
        template_provenance: Dict[int, str] = {}

        for d in range(10):
            tpl_list = self.templates.get_templates(d)
            if not tpl_list:
                continue

            best_d_score = -1.0
            best_d_iou = 0.0
            best_src = ""

            for tpl in tpl_list:
                s, iou, _ = self.compare_mask_to_template(mask, feats, tpl)
                if s > best_d_score:
                    best_d_score = s
                    best_d_iou = iou
                    best_src = tpl.source

            scores_by_digit[d] = best_d_score
            ious_by_digit[d] = best_d_iou
            template_provenance[d] = best_src

        # Construct all_probs list (10 classes for backward compatibility with ANN)
        all_probs = [round(scores_by_digit.get(d, 0.0), 4) for d in range(10)]

        if not scores_by_digit:
            return {
                "digit": 0,
                "confidence": 0.0,
                "best_score": 0.0,
                "second_score": 0.0,
                "ambiguity_margin": 0.0,
                "status": "missing_template",
                "reason": "No templates registered in library",
                "geometric_features": feats,
                "candidate_ranking": [],
                "all_probs": all_probs,
                "slot_index": slot_index
            }

        # Rank candidates
        ranked = sorted(scores_by_digit.items(), key=lambda x: x[1], reverse=True)
        best_digit, best_score = ranked[0]
        second_digit, second_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
        margin = round(best_score - second_score, 4)

        # Base status logic
        reason = ""
        if best_score < self.min_confidence:
            status = "uncertain"
            reason = f"Low score {best_score:.4f} < {self.min_confidence}"
        elif len(ranked) > 1 and margin < self.ambiguity_margin:
            status = "ambiguous"
            reason = f"Ambiguity margin {margin:.4f} < {self.ambiguity_margin}"
        else:
            status = "ok"

        # Missing-digit handling and rolling stroke rejection (Task 4)
        cand_asp = feats.get("aspect_ratio", 0.0)
        cand_holes = feats.get("holes", 0)
        cand_crossbar = feats.get("has_crossbar", False)
        top_density = feats.get("top_half_density", 0.5)
        bottom_density = feats.get("bottom_half_density", 0.5)
        missing_digits = self.templates.missing_digits()

        # 1. Partial or rolling stroke cut-off
        if cand_asp >= 1.40:
            status = "uncertain"
            reason = f"Partial or rolling stroke detected (aspect ratio {cand_asp:.2f} >= 1.40)"
        # 2. Candidate matches a missing digit: reject forcing to available digits 0, 7, 8
        elif missing_digits:
            if 1 in missing_digits and cand_asp <= 0.48 and cand_holes == 0 and best_digit == 7:
                status = "uncertain"
                reason = "Candidate matches missing digit 1; forced match to 7 rejected"
            elif 4 in missing_digits and cand_holes == 1 and cand_crossbar and best_digit in (0, 8):
                status = "uncertain"
                reason = "Candidate matches missing digit 4; forced match rejected"
            elif 6 in missing_digits and cand_holes == 1 and bottom_density >= 0.58 and best_digit == 0:
                status = "uncertain"
                reason = "Candidate matches missing digit 6; forced match to 0 rejected"
            elif 9 in missing_digits and cand_holes == 1 and top_density >= 0.58 and best_digit == 0:
                status = "uncertain"
                reason = "Candidate matches missing digit 9; forced match to 0 rejected"
            elif (2 in missing_digits or 3 in missing_digits or 5 in missing_digits) and cand_holes == 0 and cand_asp >= 0.55 and top_density < 0.42 and best_digit == 7:
                status = "uncertain"
                reason = "Candidate matches missing digit (2/3/5); forced match to 7 rejected"

        candidate_ranking = [
            {
                "digit": d,
                "score": round(sc, 4),
                "iou": round(ious_by_digit.get(d, 0.0), 4),
                "source": template_provenance.get(d, "")
            }
            for d, sc in ranked
        ]

        return {
            "digit": best_digit,
            "confidence": round(float(best_score), 4),
            "best_score": round(float(best_score), 4),
            "second_digit": second_digit,
            "second_score": round(float(second_score), 4),
            "ambiguity_margin": margin,
            "status": status,
            "reason": reason if reason else ("OK" if status == "ok" else f"Score {best_score:.4f}"),
            "geometric_features": feats,
            "candidate_ranking": candidate_ranking,
            "all_probs": all_probs,
            "slot_index": slot_index
        }

    def predict_dial(self, slot_crops: List[Union[Image.Image, np.ndarray]]) -> Dict[str, Any]:
        """
        Runs the classical digit recognizer across all slots (e.g. 4 discrete slots).
        Combines results into dial reading string while preserving leading zeros.
        """
        if len(slot_crops) != 4:
            raise ValueError(f"Expected 4 slot crops for dial reading, received {len(slot_crops)}.")

        per_slot_results = []
        digits_str = ""
        confs = []

        for i, crop in enumerate(slot_crops):
            res = self.predict_slot(crop, slot_index=i)
            per_slot_results.append(res)
            digits_str += str(res["digit"])
            confs.append(res["confidence"])

        # Detect mid-roll drum condition on slot 4 (red drum)
        # Ambiguous top-2 score margin or low score on slot 4 indicates mid-roll
        last_slot = per_slot_results[3]
        last_margin = last_slot.get("ambiguity_margin", 1.0)
        is_mid_roll = (last_margin < 0.12) or (last_slot.get("status") in ("ambiguous", "uncertain"))

        avg_conf = sum(confs) / 4.0 if confs else 0.0

        # Diagnostics for first 3 slots (as well as 4th)
        diagnostics = [
            {
                "slot_index": s["slot_index"],
                "best_candidate_digit": s["digit"],
                "best_template_score": s["best_score"],
                "second_best_score": s["second_score"],
                "ambiguity_margin": s["ambiguity_margin"],
                "status": s["status"],
                "reason": s.get("reason", ""),
                "geometric_features": s.get("geometric_features", {})
            }
            for s in per_slot_results
        ]

        return {
            "reading": digits_str,
            "confidence": round(avg_conf, 4),
            "per_digit_confidence": confs,
            "is_mid_roll": is_mid_roll,
            "slots": per_slot_results,
            "diagnostics": diagnostics
        }


# -----------------------------------------------------------------------------
# Legacy compatibility classes (retained so obsolete imports do not crash)
# -----------------------------------------------------------------------------
class DigitClassifier:
    """Legacy stub for obsolete CNN architecture."""
    pass


class DigitPredictor:
    """Legacy stub for obsolete CNN predictor."""
    pass
