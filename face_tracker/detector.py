"""
detector.py
YOLOv8-based face detector with 8-layer filtering pipeline.
Includes spatial cooldown — blocks detections near recently exited positions.
Designed for CCTV footage with overhead camera angles.
"""

import cv2
import time
import numpy as np
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, config: dict):
        det_cfg             = config["detection"]
        self.model_path     = det_cfg["model_path"]
        self.frame_skip     = det_cfg["frame_skip"]
        self.conf_threshold = det_cfg["confidence_threshold"]
        self.input_size     = det_cfg["input_size"]
        self.min_face_size  = det_cfg.get("min_face_size", 40)
        self._frame_count   = 0
        self._last_detections = []

        # Spatial cooldown: { (cx,cy): timestamp }
        # Blocks detections near recently exited face positions for 2 seconds
        self._exit_zones: list[tuple] = []  # [(cx, cy, timestamp)]
        self._exit_zone_radius = 80         # pixels
        self._exit_zone_duration = 2.0      # seconds

        logger.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info(
            f"YOLO ready | frame_skip={self.frame_skip} "
            f"| conf={self.conf_threshold} "
            f"| min_face_size={self.min_face_size}px"
        )

    def register_exit_zone(self, bbox: tuple):
        """
        Call when a face exits. Marks that position as a cooldown zone.
        Any detection within exit_zone_radius pixels for 2 seconds is rejected.
        This prevents floor tiles from being detected where a person just stood.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        self._exit_zones.append((cx, cy, time.time()))
        logger.debug(f"Exit zone registered at ({cx:.0f},{cy:.0f})")

    def _clean_exit_zones(self):
        """Remove expired exit zones."""
        now = time.time()
        self._exit_zones = [
            z for z in self._exit_zones
            if now - z[2] < self._exit_zone_duration
        ]

    def _in_exit_zone(self, x1, y1, x2, y2) -> bool:
        """Check if detection centre is inside any active exit zone."""
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        for zx, zy, _ in self._exit_zones:
            dist = ((cx - zx) ** 2 + (cy - zy) ** 2) ** 0.5
            if dist < self._exit_zone_radius:
                return True
        return False

    def _is_valid_detection(self, x1, y1, x2, y2,
                             frame_h, frame_w, frame) -> tuple[bool, str]:
        w = x2 - x1
        h = y2 - y1

        # 1. Minimum size
        if w < self.min_face_size or h < self.min_face_size:
            return False, f"too small ({w:.0f}x{h:.0f})"

        # 2. Aspect ratio — faces are roughly square
        aspect = w / (h + 1e-6)
        if aspect < 0.6 or aspect > 1.5:
            return False, f"bad aspect ratio ({aspect:.2f})"

        # 3. Edge margin
        margin_x = frame_w * 0.02
        margin_y = frame_h * 0.02
        if x1 < margin_x or y1 < margin_y:
            return False, "too close to top/left edge"
        if x2 > frame_w - margin_x or y2 > frame_h - margin_y:
            return False, "too close to bottom/right edge"

        # 4. Spatial cooldown — reject detections near recently exited positions
        if self._in_exit_zone(x1, y1, x2, y2):
            return False, "in exit zone (spatial cooldown)"

        # 5. Pixel-level checks
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            return False, "empty crop"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 6. Variance — tiles are uniform, faces have texture
        variance = gray.var()
        if variance < 150:
            return False, f"too uniform (variance={variance:.1f})"

        # 7. Brightness
        brightness = gray.mean()
        if brightness < 25:
            return False, f"too dark (brightness={brightness:.1f})"

        # 8. Skin tone — tiles/metal have near-zero saturation
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()
        if saturation < 25:
            return False, f"no skin tone (saturation={saturation:.1f})"

        # 9. Position heuristic
        face_center_y = (y1 + y2) / 2
        if face_center_y > frame_h * 0.92:
            return False, "below face zone"

        return True, "ok"

    def detect(self, frame: np.ndarray) -> list:
        self._frame_count += 1
        self._clean_exit_zones()

        if self._frame_count % (self.frame_skip + 1) != 0:
            return self._last_detections

        frame_h, frame_w = frame.shape[:2]
        results = self.model(
            frame, imgsz=self.input_size,
            conf=self.conf_threshold, verbose=False
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
                    logger.debug(f"Rejected: {reason}")

        detections = self._apply_nms(detections, iou_threshold=0.4)
        logger.debug(
            f"Frame {self._frame_count}: "
            f"{len(detections)} accepted, {rejected} rejected"
        )
        self._last_detections = detections
        return detections

    def _apply_nms(self, detections: list, iou_threshold: float = 0.4) -> list:
        if len(detections) <= 1:
            return detections
        detections = sorted(detections, key=lambda d: d[4], reverse=True)
        kept = []
        for det in detections:
            x1, y1, x2, y2, _ = det
            overlap = False
            for kd in kept:
                kx1, ky1, kx2, ky2, _ = kd
                ix1 = max(x1, kx1); iy1 = max(y1, ky1)
                ix2 = min(x2, kx2); iy2 = min(y2, ky2)
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
                if inter / (union + 1e-6) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                kept.append(det)
        if len(kept) < len(detections):
            logger.debug(f"NMS: {len(detections)} → {len(kept)}")
        return kept

    @property
    def frame_count(self) -> int:
        return self._frame_count