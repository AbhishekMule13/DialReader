"""
Verification and Test Script for Stage 1 Classical Window Detector.
Tests on:
- 80.2.png (close-up)
- 80.jpeg (full machine photo, small dial)
- Preset 1: dial_0078.png
- Preset 2: dial_0080.png
- 20 random images from data/raw
Saves overlays, warped windows, failures, and a contact sheet of warped windows.
"""

from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from src.window_detect import WindowDetector

def main():
    detector = WindowDetector()
    results_dir = Path("results/window_tests")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary test images
    primary_tests = [
        ("80.2.png", Path("data/raw/80.2.png")),
        ("80.jpeg", Path("data/raw/80.jpeg")),
        ("dial_0078.png", Path("data/presets/dial_0078.png")),
        ("dial_0080.png", Path("data/presets/dial_0080.png")),
    ]

    # 2. 20 random images from data/raw
    all_raw = list(Path("data/raw").rglob("*.jpg")) + list(Path("data/raw").rglob("*.png")) + list(Path("data/raw").rglob("*.jpeg"))
    all_raw = [p for p in all_raw if "scratch" not in str(p) and "debug" not in str(p) and "results" not in str(p)]
    
    # Exclude the 4 primary test images from random pool if present
    primary_paths = {p.resolve() for _, p in primary_tests if p.exists()}
    pool = [p for p in all_raw if p.resolve() not in primary_paths]
    
    random.seed(42)
    random_20 = random.sample(pool, min(20, len(pool)))

    test_queue = primary_tests + [(p.name, p) for p in random_20]

    print("=" * 80)
    print("STAGE 1 WINDOW DETECTOR EVALUATION REPORT")
    print("=" * 80)
    print(f"Total test images: {len(test_queue)} (4 primary + {len(random_20)} random)")
    print("-" * 80)

    stats = {
        "ok": 0,
        "no_window": 0,
        "reasons": {}
    }

    warped_windows_for_sheet = []
    failures_list = []

    for idx, (name, path) in enumerate(test_queue, 1):
        if not path.exists():
            print(f"[{idx:02d}/{len(test_queue)}] {name:30s} -> FILE NOT FOUND: {path}")
            continue

        res = detector.detect(path)
        status = res.get("status", "no_window")
        reason = res.get("reason", "unknown")
        overlay = res.get("detected_overlay")
        warped = res.get("warped_window")
        asp = res.get("aspect_ratio", 0.0)
        conf = res.get("confidence", 0.0)
        digits = res.get("white_digit_count", 0)
        bbox = res.get("pixel_bbox")

        # Save debug overlay
        clean_name = Path(name).stem
        if overlay is not None:
            overlay.save(results_dir / f"overlay_{clean_name}.png")

        if status == "ok" and warped is not None:
            stats["ok"] += 1
            warped.save(results_dir / f"warped_{clean_name}.png")
            warped_windows_for_sheet.append((name, warped, asp, conf, digits))
            print(f"[{idx:02d}/{len(test_queue)}] [OK]        {name:32s} | Asp: {asp:4.2f}:1 | Conf: {conf:4.0%} | Digits: {digits} | BBox: {bbox}")
        else:
            stats["no_window"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            failures_list.append((name, reason))
            # Save failure overlay
            if overlay is not None:
                overlay.save(results_dir / f"fail_{clean_name}.png")
            print(f"[{idx:02d}/{len(test_queue)}] [NO_WINDOW] {name:32s} | Reason: {reason}")

    print("=" * 80)
    print("DETECTION SUMMARY:")
    print(f"  Total Tested:      {len(test_queue)}")
    print(f"  Windows Found:     {stats['ok']} ({stats['ok'] / len(test_queue):.1%})")
    print(f"  No Window (Valid): {stats['no_window']} ({stats['no_window'] / len(test_queue):.1%})")
    print("Failures breakdown:")
    for r, cnt in stats["reasons"].items():
        print(f"    - {r:45s}: {cnt}")

    # Build Contact Sheet of Warped Windows
    if warped_windows_for_sheet:
        print("\nBuilding contact sheet of detected warped windows...")
        cell_w, cell_h = 320, 100
        cols = 2
        rows = int(np.ceil(len(warped_windows_for_sheet) / float(cols)))
        sheet_w = cols * (cell_w + 20) + 20
        sheet_h = rows * (cell_h + 50) + 60

        sheet = Image.new("RGB", (sheet_w, sheet_h), color="#0d1117")
        draw = ImageDraw.Draw(sheet)
        
        # Header title
        draw.text((20, 15), f"DialReader Stage 1: Warped Window Contact Sheet ({len(warped_windows_for_sheet)} Detected)", fill="#ff8c33")

        for i, (w_name, w_img, w_asp, w_conf, w_dig) in enumerate(warped_windows_for_sheet):
            col = i % cols
            row = i // cols
            x = 20 + col * (cell_w + 20)
            y = 50 + row * (cell_h + 50)

            # Resize warped window maintaining aspect ratio inside cell
            thumb = w_img.copy()
            thumb.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            
            # Draw card background
            draw.rectangle([x - 4, y - 4, x + cell_w + 4, y + cell_h + 40], fill="#161b22", outline="#30363d")
            sheet.paste(thumb, (x + (cell_w - thumb.size[0]) // 2, y + (cell_h - thumb.size[1]) // 2))

            # Caption
            caption = f"{w_name[:24]} | Asp: {w_asp:.2f}:1 | {w_conf:.0%}"
            draw.text((x, y + cell_h + 12), caption, fill="#f0f6fc")

        sheet_path = results_dir / "contact_sheet_warped_windows.png"
        sheet.save(sheet_path)
        print(f"Contact sheet saved to: {sheet_path}")

    print("=" * 80)

if __name__ == "__main__":
    main()
