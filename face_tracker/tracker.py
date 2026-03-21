"""
tracker.py
DeepSort multi-face tracker.
Since we use our own ArcFace embedder, we pass embeddings directly to DeepSort.
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
            embedder=None  # we supply our own ArcFace embeddings
        )
        logger.info("DeepSort tracker ready")

    def update(self, detections: list, frame: np.ndarray,
               embeddings: list = None) -> list:
        """
        Update tracker with new detections.

        detections : list of (x1, y1, x2, y2, confidence)
        embeddings : list of np.ndarray (one per detection) — our ArcFace vectors
        Returns    : list of dicts { track_id, bbox, confidence }
        """
        if not detections:
            # Pass empty lists — DeepSort ages out missing tracks
            tracks = self._tracker.update_tracks(
                [], embeds=[], frame=frame
            )
            return self._format_tracks(tracks)

        # DeepSort expects [([x, y, w, h], conf, class_label), ...]
        ds_detections = []
        for (x1, y1, x2, y2, conf) in detections:
            w = x2 - x1
            h = y2 - y1
            ds_detections.append(([x1, y1, w, h], conf, "face"))

        # If no embeddings provided, use zero vectors as placeholders
        if embeddings is None or len(embeddings) != len(detections):
            embeddings = [np.zeros(512, dtype=np.float32)] * len(detections)

        tracks = self._tracker.update_tracks(
            ds_detections,
            embeds=embeddings,
            frame=frame
        )
        return self._format_tracks(tracks)

    def _format_tracks(self, raw_tracks: list) -> list:
        result = []
        for track in raw_tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            result.append({
                "track_id": track.track_id,
                "bbox": (ltrb[0], ltrb[1], ltrb[2], ltrb[3]),
                "confidence": track.det_conf if track.det_conf else 0.0
            })
        return result