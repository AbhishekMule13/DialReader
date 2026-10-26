"""
Stage 1: Display Window Detector for DialReader
Uses Ultralytics YOLO (yolov8n or yolo11n) trained for class 'display_window'.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import numpy as np
from PIL import Image, ImageOps
import cv2


class DialDetector:
    """
    Detects the rectangular digit display window of orange mechanical dial counters.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None, device: str = "cpu"):
        self.device = device
        self.model_path = Path(model_path) if model_path else None
        self.model = None

        if self.model_path and self.model_path.exists():
            self._load_model()

    def _load_model(self):
        from ultralytics import YOLO
        print(f"Loading YOLO detector from {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def is_loaded(self) -> bool:
        return self.model is not None

    def detect(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect display windows in an image.

        Args:
            image_input: Image path, numpy array (BGR or RGB), or PIL Image.
            conf_threshold: Minimum detection confidence score.
            iou_threshold: NMS IoU threshold.

        Returns:
            List of detected windows, each a dict with:
                - 'bbox': [xmin, ymin, xmax, ymax] normalized to [0.0, 1.0]
                - 'pixel_bbox': [xmin, ymin, xmax, ymax] in integer image pixels
                - 'confidence': float
                - 'crop': PIL Image of the detected display window
        """
        if self.model is None:
            raise RuntimeError(
                f"Detector model not loaded. Please train or provide weights. (Path: {self.model_path})"
            )

        # Convert input to PIL RGB
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Assume BGR if from cv2
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = ImageOps.exif_transpose(image_input).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        w_img, h_img = pil_img.size
        results = self.model.predict(
            pil_img,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                x1, y1, x2, y2 = map(int, xyxy)
                x1 = max(0, min(w_img - 1, x1))
                y1 = max(0, min(h_img - 1, y1))
                x2 = max(x1 + 1, min(w_img, x2))
                y2 = max(y1 + 1, min(h_img, y2))

                crop = pil_img.crop((x1, y1, x2, y2))
                norm_bbox = [
                    round(x1 / w_img, 4),
                    round(y1 / h_img, 4),
                    round(x2 / w_img, 4),
                    round(y2 / h_img, 4)
                ]

                detections.append({
                    "bbox": norm_bbox,
                    "pixel_bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "crop": crop
                })

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections
