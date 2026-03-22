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
        self.max_age = trk_cfg.get("max_age", 120)
        self.min_hits = trk_cfg["min_hits"]
        self.iou_threshold = trk_cfg.get("iou_threshold", 0.3)

        logger.info("Initialising DeepSort tracker")
        self._tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.min_hits,
            max_iou_distance=self.iou_threshold,
            embedder=None,  # we supply our own ArcFace embeddings
            half=True,                # if supported
            bgr=True,
            embedder_gpu=False
        )
        logger.info("DeepSort tracker ready")

    def update(self, detections: list, frame: np.ndarray,
           embeddings: list = None) -> list:

        if not detections:
            tracks = self._tracker.update_tracks([], embeds=[], frame=frame)
            return self._format_tracks(tracks)

        ds_detections = []
        for i, (x1, y1, x2, y2, conf) in enumerate(detections):
            w = x2 - x1
            h = y2 - y1

            # Attach index so we can recover embedding later
            ds_detections.append((
                [x1, y1, w, h],
                conf,
                i   # store detection index as "class"
            ))

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

            # Recover detection index
            det_idx = None
            if hasattr(track, "det_class"):
                det_idx = track.det_class

            result.append({
                "track_id": track.track_id,
                "bbox": (ltrb[0], ltrb[1], ltrb[2], ltrb[3]),
                "confidence": track.det_conf if track.det_conf else 0.0,
                "det_idx": det_idx   # 🔥 IMPORTANT
            })

        return result