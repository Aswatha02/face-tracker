"""
tracker.py
DeepSort-based multi-object tracker.
Takes YOLO detections and returns stable track IDs across frames.
"""

import numpy as np
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)


class FaceTracker:
    def __init__(self, config: dict):
        trk_cfg = config["tracking"]
        self.max_age = trk_cfg["max_age"]
        self.min_hits = trk_cfg["min_hits"]
        self.iou_threshold = trk_cfg["iou_threshold"]

        logger.info("Initialising DeepSort tracker")
        self._tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.min_hits,
            max_iou_distance=self.iou_threshold,
            embedder=None       # we use InsightFace, not DeepSort's own embedder
        )
        logger.info("DeepSort tracker ready")

    def update(self, detections: list, frame: np.ndarray) -> list:
        """
        Update tracker with new detections.

        detections: list of (x1, y1, x2, y2, confidence)
        Returns: list of dicts:
            { track_id, bbox: (x1,y1,x2,y2), confidence }
        """
        if not detections:
            # Still need to call update to age out missing tracks
            tracks = self._tracker.update_tracks([], frame=frame)
            return self._format_tracks(tracks)

        # DeepSort expects detections as [([x1,y1,w,h], conf, class), ...]
        ds_detections = []
        for (x1, y1, x2, y2, conf) in detections:
            w = x2 - x1
            h = y2 - y1
            ds_detections.append(([x1, y1, w, h], conf, "face"))

        tracks = self._tracker.update_tracks(ds_detections, frame=frame)
        return self._format_tracks(tracks)

    def _format_tracks(self, raw_tracks: list) -> list:
        """Convert DeepSort track objects to plain dicts."""
        result = []
        for track in raw_tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()   # (left, top, right, bottom)
            result.append({
                "track_id": track.track_id,
                "bbox": (ltrb[0], ltrb[1], ltrb[2], ltrb[3]),
                "confidence": track.det_conf if track.det_conf else 0.0
            })
        return result