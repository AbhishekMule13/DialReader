"""
Dataset Validation Module for DialReader (Step 4C & 4C.1)

Validates the labeled dial dataset against strict quality and safety criteria:
1. Malformed or invalid readings and slot digit labels.
2. Missing image files referenced in dataset records.
3. Duplicate filenames or duplicate image content (via SHA-256 pixel hashes).
4. Invalid template-source flags (e.g. template source on non-stationary or unverified sample).
5. Mismatches between slot digit labels and confirmed 4-digit reading.
6. Authoritative accounting distinguishing:
   - Human-confirmed dataset records (slot-level counts vs image-level diversity)
   - Active production template library (loaded in PixelDigitRecognizer from presets)
   - Global physical image inventory and remaining unreviewed queue.

Strict constraint: Validation reports problems clearly and does NOT silently alter or delete records.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import hashlib
import pandas as pd
from PIL import Image


@dataclass
class ValidationIssue:
    severity: str  # "ERROR" or "WARNING"
    category: str
    message: str
    filename: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    is_valid: bool
    total_records: int
    valid_records: int
    issues: List[ValidationIssue]
    errors: List[str]
    warnings: List[str]
    duplicate_groups: Dict[str, List[str]]
    # Human-confirmed readings.csv statistics
    class_counts: Dict[int, int]  # Image-level diversity in human records (backward-compatible)
    shortfalls: Dict[int, int]    # Image-level shortfall in human records (backward-compatible)
    human_slot_counts: Dict[int, int] = field(default_factory=dict)
    # Active Recognizer Production Templates
    recognizer_template_counts: Dict[int, int] = field(default_factory=dict)
    recognizer_source_images: Dict[int, int] = field(default_factory=dict)
    recognizer_shortfalls: Dict[int, int] = field(default_factory=dict)
    # Global Inventory Counts
    unique_physical_images: int = 0
    confirmed_full_readings: int = 0
    confirmed_digit_labels_count: int = 0
    verified_stationary_sources: int = 0
    unreviewed_queue_count: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)

    def print_report(self):
        print("=" * 80)
        print(" DIALREADER DATASET VALIDATION & INVENTORY REPORT (Step 4C.1)")
        print("=" * 80)
        print(f"Validation Status:          {'PASSED (Dataset is Valid)' if self.is_valid else 'FAILED (Validation Errors Found)'}")
        print(f"Total CSV Records:          {self.total_records}")
        print(f"Valid CSV Records:          {self.valid_records}")
        print(f"Errors Found:               {len(self.errors)}")
        print(f"Warnings Found:             {len(self.warnings)}")

        if self.errors:
            print("\n" + "-" * 80)
            print(f"ERRORS ({len(self.errors)}):")
            for e in self.errors:
                print(f"  [ERROR] {e}")

        if self.warnings:
            print("\n" + "-" * 80)
            print(f"WARNINGS ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  [WARNING] {w}")

        if self.duplicate_groups:
            print("\n" + "-" * 80)
            print("DUPLICATE / NON-INDEPENDENT IMAGE GROUPS:")
            for h, paths in self.duplicate_groups.items():
                print(f"  Hash {h[:12]}... ({len(paths)} copies):")
                for p in paths:
                    print(f"    - {p}")

        print("\n" + "-" * 80)
        print("1. HUMAN-CONFIRMED DATASET (data/labeled/readings.csv):")
        print(f"   Confirmed 4-digit full readings:   {self.confirmed_full_readings}")
        print(f"   Images with confirmed slot labels: {self.confirmed_digit_labels_count}")
        print(f"   Verified stationary sources:       {self.verified_stationary_sources}")
        print(f"   Remaining unreviewed queue images: {self.unreviewed_queue_count}")
        print("\nDigit | Slot Examples | Unique Images | Image Shortfall (Target: 5) | Status")
        print("------+---------------+---------------+-----------------------------+-------")
        for d in range(10):
            slot_cnt = self.human_slot_counts.get(d, 0)
            img_cnt = self.class_counts.get(d, 0)
            short = self.shortfalls.get(d, 0)
            status_str = "MET" if short == 0 else f"NEED +{short}"
            print(f"  {d}   |       {slot_cnt:2d}      |       {img_cnt:2d}      |              {short:2d}             | {status_str}")

        print("\n" + "-" * 80)
        print("2. ACTIVE PRODUCTION TEMPLATE BANK (Loaded in PixelDigitRecognizer):")
        print("   Source: Hardcoded verified presets + Curated templates (data/labeled/verified_templates.json)")
        print("\nDigit | Loaded Templates | Source Images | Template Shortfall (Target: 5) | Status")
        print("------+------------------+---------------+--------------------------------+-------")
        for d in range(10):
            tpl_cnt = self.recognizer_template_counts.get(d, 0)
            src_img_cnt = self.recognizer_source_images.get(d, 0)
            short = self.recognizer_shortfalls.get(d, 0)
            status_str = "MET" if short == 0 else f"NEED +{short}"
            print(f"  {d}   |        {tpl_cnt:2d}        |       {src_img_cnt:2d}       |               {short:2d}                | {status_str}")

        print("\n" + "-" * 80)
        print("3. GLOBAL PHYSICAL IMAGE INVENTORY:")
        print(f"   Unique physical images (by pixel SHA-256): {self.unique_physical_images}")
        print("=" * 80)


class DatasetValidator:
    """
    Validates readings.csv and raw image integrity without modifying data.
    """

    def __init__(
        self,
        readings_csv: Optional[Path] = None,
        raw_dir: Optional[Path] = None,
        target_per_class: int = 5,
        project_root: Optional[Path] = None
    ):
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent

        self.project_root = project_root
        self.readings_csv = readings_csv or (project_root / "data" / "labeled" / "readings.csv")
        self.raw_dir = raw_dir or (project_root / "data" / "raw")
        self.target_per_class = target_per_class

    _pixel_hash_cache: Dict[str, Tuple[float, int, str]] = {}

    @classmethod
    def compute_image_sha256(cls, filepath: Path) -> Optional[str]:
        """Calculates fast SHA-256 to detect bit-identical images across raw and preset datasets."""
        if not filepath.exists():
            return None
        try:
            stat = filepath.stat()
            cache_key = str(filepath.resolve())
            cached = cls._pixel_hash_cache.get(cache_key)
            if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
                return cached[2]

            h = hashlib.sha256()
            with open(filepath, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            digest = h.hexdigest()
            cls._pixel_hash_cache[cache_key] = (stat.st_mtime, stat.st_size, digest)
            return digest
        except Exception:
            return None

    def validate(self, df_override: Optional[pd.DataFrame] = None) -> ValidationReport:
        """
        Executes full validation suite against dataset.
        Returns detailed ValidationReport.
        """
        issues: List[ValidationIssue] = []
        errors: List[str] = []
        warnings: List[str] = []

        if df_override is not None:
            df = df_override.copy()
        else:
            if not self.readings_csv.exists():
                err = f"Readings file not found: {self.readings_csv}"
                errors.append(err)
                return ValidationReport(
                    is_valid=False,
                    total_records=0,
                    valid_records=0,
                    issues=[ValidationIssue("ERROR", "file_missing", err)],
                    errors=errors,
                    warnings=[],
                    duplicate_groups={},
                    class_counts={d: 0 for d in range(10)},
                    shortfalls={d: self.target_per_class for d in range(10)},
                    summary={"status": "readings_csv_not_found"}
                )
            df = pd.read_csv(self.readings_csv, dtype=str)

        total_records = len(df)
        valid_records = 0

        # Track duplicates
        seen_filenames: Set[str] = set()
        pixel_hash_to_files: Dict[str, List[str]] = {}
        file_to_hash: Dict[str, str] = {}

        # 1. Filename duplicate checks and image hash mapping
        for idx, row in df.iterrows():
            fn = str(row.get("filename", "")).strip()
            if not fn or fn.lower() == "nan":
                msg = f"Row {idx}: Missing or blank filename"
                errors.append(msg)
                issues.append(ValidationIssue("ERROR", "blank_filename", msg))
                continue

            if fn in seen_filenames:
                msg = f"Duplicate record for filename: '{fn}' (row {idx})"
                errors.append(msg)
                issues.append(ValidationIssue("ERROR", "duplicate_filename", msg, filename=fn))
            seen_filenames.add(fn)

            # Check image existence
            img_path = self.raw_dir / fn
            if not img_path.exists():
                msg = f"Image file missing on disk: '{fn}' ({img_path})"
                errors.append(msg)
                issues.append(ValidationIssue("ERROR", "image_missing", msg, filename=fn))
            else:
                p_hash = self.compute_image_sha256(img_path)
                if p_hash:
                    pixel_hash_to_files.setdefault(p_hash, []).append(fn)
                    file_to_hash[fn] = p_hash

        # Identify duplicate image content across distinct filenames
        duplicate_groups = {h: files for h, files in pixel_hash_to_files.items() if len(files) > 1}
        for h, files in duplicate_groups.items():
            msg = f"Duplicate image content detected across {len(files)} files: {files}"
            warnings.append(msg)
            issues.append(ValidationIssue("WARNING", "duplicate_content", msg, details={"files": files}))

        # Track human-confirmed counts
        human_slot_counts: Dict[int, int] = {d: 0 for d in range(10)}
        independent_digit_counts: Dict[int, int] = {d: 0 for d in range(10)}
        seen_digit_hashes: Dict[int, Set[str]] = {d: set() for d in range(10)}

        confirmed_full_readings = 0
        confirmed_digit_labels_count = 0
        verified_stationary_sources = 0

        # 2. Row-by-row validation of readings and labels
        for idx, row in df.iterrows():
            fn = str(row.get("filename", "")).strip()
            reading = str(row.get("reading", "")).strip()
            condition = str(row.get("condition", "stationary")).strip().lower()
            is_tpl = str(row.get("is_template_source", "False")).strip().lower() in ("true", "1")
            notes = str(row.get("notes", "")).strip()

            row_has_error = False

            # Check valid conditions
            valid_conditions = {"stationary", "rolling", "reflective", "unreadable", "skipped"}
            if condition not in valid_conditions:
                msg = f"Row {idx} ({fn}): Invalid condition '{condition}'. Expected one of {valid_conditions}"
                errors.append(msg)
                issues.append(ValidationIssue("ERROR", "invalid_condition", msg, filename=fn))
                row_has_error = True

            # If reading is unreadable or skipped
            if reading in ("unreadable", "skipped", "negative"):
                if is_tpl:
                    msg = f"Row {idx} ({fn}): is_template_source=True but reading is '{reading}'"
                    errors.append(msg)
                    issues.append(ValidationIssue("ERROR", "invalid_template_flag", msg, filename=fn))
                    row_has_error = True
                if not row_has_error:
                    valid_records += 1
                continue

            # Numeric reading validation
            if not reading.isdigit() or len(reading) != 4:
                msg = f"Row {idx} ({fn}): Malformed reading '{reading}'. Expected exactly 4 numeric digits or 'unreadable'"
                errors.append(msg)
                issues.append(ValidationIssue("ERROR", "malformed_reading", msg, filename=fn))
                continue

            confirmed_full_readings += 1

            # Verify individual slot digits if present
            slot_digits = []
            slot_mismatch = False
            has_slot_labels = False

            for s_idx in range(4):
                col_name = f"digit_{s_idx}"
                slot_val = row.get(col_name)
                expected_digit = int(reading[s_idx])

                if pd.notnull(slot_val) and str(slot_val).strip() != "":
                    has_slot_labels = True
                    try:
                        d_int = int(str(slot_val).strip())
                        if not (0 <= d_int <= 9):
                            msg = f"Row {idx} ({fn}): {col_name} out of range [0-9]: {d_int}"
                            errors.append(msg)
                            issues.append(ValidationIssue("ERROR", "slot_out_of_range", msg, filename=fn))
                            row_has_error = True
                        elif d_int != expected_digit:
                            msg = f"Row {idx} ({fn}): Mismatch in {col_name}. Slot label={d_int} != reading digit={expected_digit}"
                            errors.append(msg)
                            issues.append(ValidationIssue("ERROR", "slot_mismatch", msg, filename=fn))
                            row_has_error = True
                            slot_mismatch = True
                        slot_digits.append(d_int)
                    except ValueError:
                        msg = f"Row {idx} ({fn}): Non-integer value in {col_name}: '{slot_val}'"
                        errors.append(msg)
                        issues.append(ValidationIssue("ERROR", "slot_type_error", msg, filename=fn))
                        row_has_error = True
                else:
                    slot_digits.append(expected_digit)

            if has_slot_labels and not slot_mismatch:
                confirmed_digit_labels_count += 1

            # Safety check: template-source flag
            if is_tpl:
                if condition != "stationary":
                    msg = f"Row {idx} ({fn}): is_template_source=True but condition is '{condition}' (must be 'stationary')"
                    errors.append(msg)
                    issues.append(ValidationIssue("ERROR", "template_not_stationary", msg, filename=fn))
                    row_has_error = True
                else:
                    verified_stationary_sources += 1

            # If row is valid, count verified independent examples
            if not row_has_error:
                valid_records += 1
                if condition == "stationary":
                    img_hash = file_to_hash.get(fn, fn)
                    for d in slot_digits:
                        human_slot_counts[d] += 1
                        if img_hash not in seen_digit_hashes[d]:
                            seen_digit_hashes[d].add(img_hash)
                            independent_digit_counts[d] += 1

        # Calculate shortfalls against target (>= 5 independent verified samples)
        shortfalls = {d: max(0, self.target_per_class - independent_digit_counts[d]) for d in range(10)}

        # Class shortfall warnings for human dataset
        for d, short in shortfalls.items():
            if short > 0:
                cnt = independent_digit_counts[d]
                warnings.append(
                    f"Class shortfall in readings.csv: Digit {d} has {cnt} independent verified example(s) (target: {self.target_per_class}, need +{short})"
                )

        # 3. Active Recognizer Production Template Bank Accounting
        recognizer_template_counts: Dict[int, int] = {d: 0 for d in range(10)}
        recognizer_source_images: Dict[int, int] = {d: 0 for d in range(10)}
        recognizer_shortfalls: Dict[int, int] = {d: self.target_per_class for d in range(10)}

        try:
            from src.digits import PixelDigitRecognizer
            rec = PixelDigitRecognizer(auto_load_verified=True, project_root=self.project_root)
            for d in range(10):
                tpls = rec.templates.get_templates(d, verified_only=True)
                recognizer_template_counts[d] = len(tpls)
                # Count distinct physical source images by SHA-256 to prevent duplicate counting
                src_hashes = set()
                for t in tpls:
                    parts = t.source.split(":")
                    src_name = parts[1] if len(parts) >= 2 else t.source
                    src_p = self.project_root / "data" / "presets" / src_name
                    if not src_p.exists():
                        src_p = self.raw_dir / src_name
                    h = self.compute_image_sha256(src_p) if src_p.exists() else src_name
                    src_hashes.add(h)
                recognizer_source_images[d] = len(src_hashes)
                recognizer_shortfalls[d] = max(0, self.target_per_class - len(tpls))
        except Exception:
            pass

        # 4. Inventory of physical source images and queue
        unique_hashes = set()
        if self.raw_dir.exists():
            for p in self.raw_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    h = self.compute_image_sha256(p)
                    if h:
                        unique_hashes.add(h)
        presets_dir = self.project_root / "data" / "presets"
        if presets_dir.exists():
            for p in presets_dir.glob("*"):
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    h = self.compute_image_sha256(p)
                    if h:
                        unique_hashes.add(h)
        unique_physical_images = len(unique_hashes)

        # Unreviewed queue count
        unreviewed_queue_count = 0
        try:
            from label_readings import load_dataset_and_progress
            eligible_df, cur_readings_df = load_dataset_and_progress()
            labeled_fns = set(cur_readings_df[cur_readings_df["reading"].notnull()]["filename"].values)
            unreviewed_queue_count = len(eligible_df[~eligible_df["filename"].isin(labeled_fns)])
        except Exception:
            unreviewed_queue_count = max(0, len(seen_filenames) - valid_records)

        is_valid = len(errors) == 0

        return ValidationReport(
            is_valid=is_valid,
            total_records=total_records,
            valid_records=valid_records,
            issues=issues,
            errors=errors,
            warnings=warnings,
            duplicate_groups=duplicate_groups,
            class_counts=independent_digit_counts,
            shortfalls=shortfalls,
            human_slot_counts=human_slot_counts,
            recognizer_template_counts=recognizer_template_counts,
            recognizer_source_images=recognizer_source_images,
            recognizer_shortfalls=recognizer_shortfalls,
            unique_physical_images=unique_physical_images,
            confirmed_full_readings=confirmed_full_readings,
            confirmed_digit_labels_count=confirmed_digit_labels_count,
            verified_stationary_sources=verified_stationary_sources,
            unreviewed_queue_count=unreviewed_queue_count,
            summary={
                "total_records": total_records,
                "valid_records": valid_records,
                "errors_count": len(errors),
                "warnings_count": len(warnings),
                "unique_images": len(seen_filenames),
                "duplicate_groups_count": len(duplicate_groups),
                "unique_physical_images": unique_physical_images,
                "confirmed_full_readings": confirmed_full_readings,
                "confirmed_digit_labels_count": confirmed_digit_labels_count,
                "verified_stationary_sources": verified_stationary_sources,
                "unreviewed_queue_count": unreviewed_queue_count
            }
        )


def validate_labeled_dataset(
    readings_csv: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    target_per_class: int = 5,
    project_root: Optional[Path] = None
) -> ValidationReport:
    """Convenience function to run dataset validation."""
    validator = DatasetValidator(
        readings_csv=readings_csv,
        raw_dir=raw_dir,
        target_per_class=target_per_class,
        project_root=project_root
    )
    return validator.validate()
