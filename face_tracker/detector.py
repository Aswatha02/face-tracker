"""
detector.py
YOLOv8-based face detector.
Reads frames, skips N frames as configured, returns bounding boxes.
"""

import cv2
import numpy as np
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, config: dict):
        det_cfg = config["detection"]
        self.model_path = det_cfg["model_path"]
        self.frame_skip = det_cfg["frame_skip"]          # detect every N frames
        self.conf_threshold = det_cfg["confidence_threshold"]
        self.input_size = det_cfg["input_size"]
        self._frame_count = 0
        self._last_detections = []

        logger.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info("YOLO model loaded successfully")

    def detect(self, frame: np.ndarray) -> list:
        """
        Run detection on frame (respecting frame_skip).
        Returns list of (x1, y1, x2, y2, confidence) tuples.
        """
        self._frame_count += 1

        # Only run YOLO every frame_skip frames; reuse last result otherwise
        if self._frame_count % (self.frame_skip + 1) != 0:
            return self._last_detections

        results = self.model(
            frame,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))

        self._last_detections = detections
        logger.debug(f"Frame {self._frame_count}: detected {len(detections)} face(s)")
        return detections

    @property
    def frame_count(self) -> int:
        return self._frame_count