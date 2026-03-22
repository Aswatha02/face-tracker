"""
event_logger.py
Manages entry and exit events.
Fires exactly one entry and one exit event per face visit.
On exit: notifies registry (cooldown) AND detector (spatial cooldown).
"""

import cv2
import logging
from utils import crop_face, save_face_image
from database import Database

logger = logging.getLogger(__name__)


class EventLogger:
    def __init__(self, config: dict, db: Database,
                 registry=None, detector=None):
        self.db           = db
        self.registry     = registry
        self.detector     = detector   # for spatial exit zone
        log_cfg           = config["logging"]
        self.entries_dir  = log_cfg["entries_dir"]
        self.exits_dir    = log_cfg["exits_dir"]
        self.img_fmt      = log_cfg["image_format"]
        self.exit_timeout = config["tracking"]["exit_timeout_frames"]
        self._active: dict[str, dict] = {}

    def _is_valid_crop(self, face_img) -> bool:
        if face_img is None or face_img.size == 0:
            return False
        if face_img.shape[0] < 20 or face_img.shape[1] < 20:
            return False
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 15:
            return False
        if gray.var() < 100:
            return False
        return True

    def _fire_exit(self, fid: str, state: dict, frame):
        """Fire exit event, notify registry and detector."""
        face_crop = crop_face(frame, state["bbox"])
        img_path  = save_face_image(
            face_crop, self.exits_dir, fid, "exit", self.img_fmt
        ) if self._is_valid_crop(face_crop) else None

        self.db.log_event(fid, "exit", img_path, state["track_id"])
        logger.info(
            f"EXIT  | face={fid} | track={state['track_id']} | img={img_path}"
        )

        # Tell registry to block this face for cooldown period
        if self.registry:
            self.registry.mark_exited(fid, bbox=state["bbox"])

        # Tell detector to block this spatial area for 2 seconds
        # This prevents floor tiles from being detected where person stood
        if self.detector:
            self.detector.register_exit_zone(state["bbox"])

    def update(self, frame, active_tracks: list):
        current_face_ids = set()

        for track in active_tracks:
            fid  = track["face_id"]
            tid  = track["track_id"]
            bbox = track["bbox"]
            current_face_ids.add(fid)

            if fid not in self._active:
                # ── ENTRY ─────────────────────────────────────────────────
                face_crop = crop_face(frame, bbox)
                img_path  = save_face_image(
                    face_crop, self.entries_dir, fid, "entry", self.img_fmt
                ) if self._is_valid_crop(face_crop) else None

                self.db.log_event(fid, "entry", img_path, tid)
                logger.info(f"ENTRY | face={fid} | track={tid} | img={img_path}")
                self._active[fid] = {
                    "track_id": tid, "missed_frames": 0, "bbox": bbox
                }
            else:
                self._active[fid]["missed_frames"] = 0
                self._active[fid]["bbox"]          = bbox
                self._active[fid]["track_id"]      = tid

        # Exit timeout
        to_exit = []
        for fid, state in self._active.items():
            if fid not in current_face_ids:
                state["missed_frames"] += 1
                if state["missed_frames"] >= self.exit_timeout:
                    to_exit.append(fid)

        for fid in to_exit:
            state = self._active.pop(fid)
            self._fire_exit(fid, state, frame)

    def flush_all_exits(self, frame):
        for fid, state in list(self._active.items()):
            self._fire_exit(fid, state, frame)
        self._active.clear()

    @property
    def active_count(self) -> int:
        return len(self._active)