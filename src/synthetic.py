"""
Synthetic Digit Generator & Augmentation Pipeline for DialReader
Used strictly for pretraining / offline dataset expansion (never used in val/test).
Simulates rolling drums, black/red backgrounds, mid-roll vertical transitions,
synthetic glare, blur, and perspective distortion. Strictly NO horizontal flips.
"""

from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import random
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2


class SyntheticDigitGenerator:
    """
    Renders realistic industrial rolling-drum counter digit slots.
    """

    def __init__(
        self,
        slot_height: int = 128,
        slot_width: int = 64
    ):
        self.slot_height = slot_height
        self.slot_width = slot_width

        # Attempt to load a clean sans-serif font, fallback to default
        self.font = self._load_font()

    def _load_font(self):
        font_names = ["arial.ttf", "calibri.ttf", "seguisb.ttf", "tahoma.ttf", "DejaVuSans-Bold.ttf"]
        for fn in font_names:
            try:
                # Target height for single numeral is ~55-65% of slot height
                return ImageFont.truetype(fn, int(self.slot_height * 0.62))
            except IOError:
                continue
        return ImageFont.load_default()

    def render_digit_slot(
        self,
        digit: int,
        is_red_drum: bool = False,
        offset_y: float = 0.0,  # in [-0.5, 0.5] vertical scroll offset for mid-roll
        add_glare: bool = True,
        add_blur: bool = True
    ) -> Image.Image:
        """
        Renders a single digit slot crop with continuous vertical rolling simulation.

        Args:
            digit: Integer 0-9
            is_red_drum: If True, red drum background; if False, black drum
            offset_y: Fraction of slot height shift (simulates mid-roll)
            add_glare: Whether to add specular glare highlight
            add_blur: Whether to add slight optical blur
        """
        # Base background color
        if is_red_drum:
            # Orange-red hue with drum curvature shading
            bg_color = (random.randint(180, 220), random.randint(25, 55), random.randint(25, 45))
        else:
            # Matte dark charcoal / black
            val = random.randint(15, 30)
            bg_color = (val, val, val)

        img = Image.new("RGB", (self.slot_width, self.slot_height), color=bg_color)
        draw = ImageDraw.Draw(img)

        # Drum cylindrical shading (darker at top and bottom)
        grad = np.linspace(0.65, 1.0, self.slot_height // 2)
        grad = np.concatenate([grad, grad[::-1]])
        arr = np.array(img, dtype=np.float32)
        arr = arr * grad[:, None, None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # Primary digit text
        text_primary = str(digit % 10)
        # Next digit if rolling upwards (offset > 0) or downwards (offset < 0)
        next_digit = (digit + 1) % 10 if offset_y > 0 else (digit - 1) % 10
        text_next = str(next_digit)

        # Font positioning
        # Center of slot
        cx = self.slot_width / 2.0
        cy = self.slot_height / 2.0 + (offset_y * self.slot_height)

        # Draw primary numeral
        draw.text(
            (cx, cy),
            text_primary,
            fill=(245, 245, 245),
            font=self.font,
            anchor="mm"
        )

        # If significant offset, draw incoming next digit
        if abs(offset_y) > 0.05:
            spacing = self.slot_height * 0.85
            next_cy = cy + (-spacing if offset_y > 0 else spacing)
            draw.text(
                (cx, next_cy),
                text_next,
                fill=(240, 240, 240),
                font=self.font,
                anchor="mm"
            )

        # Optional red vernier scale tick marks on right edge if is_red_drum
        if is_red_drum and random.random() < 0.5:
            for y_tick in range(10, self.slot_height - 10, 12):
                draw.line([(self.slot_width - 6, y_tick), (self.slot_width - 1, y_tick)], fill=(250, 250, 250), width=1)

        # Synthetic glare highlight (diagonal glossy streak)
        if add_glare and random.random() < 0.6:
            glare_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
            g_draw = ImageDraw.Draw(glare_layer)
            gy = random.randint(int(self.slot_height * 0.2), int(self.slot_height * 0.7))
            gh = random.randint(8, 24)
            opacity = random.randint(30, 90)
            g_draw.ellipse(
                [-self.slot_width * 0.2, gy, self.slot_width * 1.2, gy + gh],
                fill=(255, 255, 255, opacity)
            )
            glare_layer = glare_layer.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 5.0)))
            img = Image.alpha_composite(img.convert("RGBA"), glare_layer).convert("RGB")

        # Optical blur
        if add_blur and random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

        return img

    def apply_augmentations(self, img: Image.Image) -> Image.Image:
        """
        Applies photometric and geometric augmentations:
        - brightness, contrast jitter
        - small rotation (-10 to +10 degrees)
        - perspective warp
        - JPEG compression artifacts
        - STRICTLY NO HORIZONTAL FLIPS
        """
        # Brightness
        if random.random() < 0.5:
            factor = random.uniform(0.75, 1.25)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Contrast
        if random.random() < 0.5:
            factor = random.uniform(0.8, 1.3)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # Rotation: +/- 10 degrees only
        if random.random() < 0.5:
            angle = random.uniform(-10.0, 10.0)
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR)

        # JPEG Compression noise
        if random.random() < 0.4:
            quality = random.randint(40, 85)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        return img

    def generate_batch(
        self,
        num_samples_per_digit: int = 50,
        output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates a balanced synthetic dataset of black and red drums,
        both standard and mid-roll instances.
        """
        records = []
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for digit in range(10):
            for i in range(num_samples_per_digit):
                # 70% black drum, 30% red drum
                is_red = (random.random() < 0.3)
                # 35% mid-roll instances
                is_mid = (random.random() < 0.35)
                offset = random.uniform(0.18, 0.45) * (1 if random.random() < 0.5 else -1) if is_mid else random.uniform(-0.06, 0.06)

                raw_slot = self.render_digit_slot(digit, is_red_drum=is_red, offset_y=offset)
                aug_slot = self.apply_augmentations(raw_slot)

                record = {
                    "digit": digit,
                    "is_red_drum": is_red,
                    "is_mid_roll": is_mid,
                    "offset_y": offset,
                    "image": aug_slot
                }

                if output_dir:
                    fname = f"synth_d{digit}_{'red' if is_red else 'blk'}_{'mid' if is_mid else 'std'}_{i:04d}.jpg"
                    fpath = output_dir / fname
                    aug_slot.save(fpath, "JPEG", quality=95)
                    record["filepath"] = str(fpath)

                records.append(record)

        return records
