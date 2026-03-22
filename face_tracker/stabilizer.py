"""
stabilizer.py
Temporal detection stabilizer.
Requires a detection to appear in the same position for min_frames
consecutive frames before passing it downstream.

Key insight: Tile/floor false positives appear for 1-2 frames only.
Real faces persist for 10+ frames. This filter kills tiles completely.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class DetectionStabilizer:
    def __init__(self, min_frames: int = 3, position_tolerance: int = 20):
        """
        min_frames: how many consecutive frames a detection must appear
        position_tolerance: pixels — two bboxes within this distance
                           are considered the same detection
        """
        self.min_frames      = min_frames
        self.tol             = position_tolerance
        # { bbox_key: count }
        self._buffer: dict   = {}

    def _bbox_key(self, x1, y1, x2, y2) -> tuple:
        """Snap coordinates to grid to handle small jitter between frames."""
        t = self.tol
        return (
            int(x1 // t) * t,
            int(y1 // t) * t,
            int(x2 // t) * t,
            int(y2 // t) * t,
        )

    def update(self, detections: list) -> list:
        """
        Pass detections through the stabilizer.
        Only returns detections that have been seen for >= min_frames frames.
        """
        new_buffer = {}
        stable     = []

        for det in detections:
            x1, y1, x2, y2, conf = det
            key   = self._bbox_key(x1, y1, x2, y2)
            count = self._buffer.get(key, 0) + 1
            new_buffer[key] = count

            if count >= self.min_frames:
                stable.append(det)
            else:
                logger.debug(
                    f"Stabilizer: detection at ({x1:.0f},{y1:.0f}) "
                    f"seen {count}/{self.min_frames} frames"
                )

        rejected = len(detections) - len(stable)
        if rejected:
            logger.debug(
                f"Stabilizer: {len(stable)} stable, {rejected} filtered"
            )

        self._buffer = new_buffer
        return stable