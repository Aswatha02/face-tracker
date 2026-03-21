"""
detector.py
YOLOv8-based face detector with multi-layer filtering:
  1. Confidence threshold
  2. Minimum face size
  3. Aspect ratio (faces are roughly square)
  4. Edge margin filter (removes partial heads at frame borders)
  5. Variance filter (removes blank/uniform regions)
"""

import cv2
import numpy as np
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, config: dict):
        det_cfg              = config["detection"]
        self.model_path      = det_cfg["model_path"]
        self.frame_skip      = det_cfg["frame_skip"]
        self.conf_threshold  = det_cfg["confidence_threshold"]
        self.input_size      = det_cfg["input_size"]
        self.min_face_size   = det_cfg.get("min_face_size", 30)
        self._frame_count    = 0
        self._last_detections = []

        logger.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info(
            f"YOLO ready | frame_skip={self.frame_skip} "
            f"| conf={self.conf_threshold} "
            f"| min_face_size={self.min_face_size}px"
        )

    def _is_valid_detection(self, x1, y1, x2, y2,
                             frame_h, frame_w, frame) -> tuple[bool, str]:
        """
        Multi-layer validation. Returns (valid, reason_if_rejected).
        """
        w = x2 - x1
        h = y2 - y1

        # ── 1. Minimum size ───────────────────────────────────────────────
        if w < self.min_face_size or h < self.min_face_size:
            return False, f"too small ({w:.0f}x{h:.0f})"

        # ── 2. Aspect ratio — faces are roughly square (0.5 to 1.8) ───────
        aspect = w / (h + 1e-6)
        if aspect < 0.5 or aspect > 1.8:
            return False, f"bad aspect ratio ({aspect:.2f})"

        # ── 3. Edge margin — reject detections too close to frame border ──
        # Partial heads at the top/sides of frame cause false matches
        margin_x = frame_w * 0.02   # 2% of width
        margin_y = frame_h * 0.02   # 2% of height
        if x1 < margin_x or y1 < margin_y:
            return False, "too close to top/left edge"
        if x2 > frame_w - margin_x or y2 > frame_h - margin_y:
            return False, "too close to bottom/right edge"

        # ── 4. Pixel variance — blank/uniform regions are not faces ───────
        # A real face crop has texture (eyes, nose, skin variation)
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            return False, "empty crop"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variance = gray.var()
        if variance < 100:
            return False, f"too uniform (variance={variance:.1f})"

        # ── 5. Brightness — reject very dark crops ────────────────────────
        brightness = gray.mean()
        if brightness < 20:
            return False, f"too dark (brightness={brightness:.1f})"

        return True, "ok"

    def detect(self, frame: np.ndarray) -> list:
        """
        Run detection on frame (respecting frame_skip).
        Returns list of (x1, y1, x2, y2, confidence) tuples.
        """
        self._frame_count += 1

        if self._frame_count % (self.frame_skip + 1) != 0:
            return self._last_detections

        frame_h, frame_w = frame.shape[:2]

        results = self.model(
            frame,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            verbose=False
        )

        detections = []
        rejected   = 0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                valid, reason = self._is_valid_detection(
                    x1, y1, x2, y2, frame_h, frame_w, frame
                )
                if valid:
                    detections.append((x1, y1, x2, y2, conf))
                else:
                    rejected += 1
                    logger.debug(f"Rejected detection: {reason}")

        logger.debug(
            f"Frame {self._frame_count}: "
            f"{len(detections)} accepted, {rejected} rejected"
        )
        self._last_detections = detections
        return detections

    @property
    def frame_count(self) -> int:
        return self._frame_count