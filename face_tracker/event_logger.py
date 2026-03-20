"""
event_logger.py
Manages entry and exit events.
Tracks which face IDs are currently "inside" the frame and fires
exactly one entry and one exit event per visit.
"""

import cv2
import logging
from datetime import datetime
from utils import crop_face, save_face_image
from database import Database

logger = logging.getLogger(__name__)


class EventLogger:
    def __init__(self, config: dict, db: Database):
        self.db = db
        log_cfg = config["logging"]
        self.entries_dir = log_cfg["entries_dir"]
        self.exits_dir = log_cfg["exits_dir"]
        self.img_fmt = log_cfg["image_format"]

        # exit_timeout: how many consecutive missed frames before we call it an exit
        self.exit_timeout = config["tracking"]["exit_timeout_frames"]

        # State: { face_id: { track_id, missed_frames, bbox } }
        self._active: dict[str, dict] = {}

    def update(self, frame, active_tracks: list):
        """
        Call every frame with the current list of confirmed tracks.
        Each track must have: face_id, track_id, bbox
        Fires entry/exit events as faces appear/disappear.
        """
        current_face_ids = set()

        for track in active_tracks:
            fid = track["face_id"]
            tid = track["track_id"]
            bbox = track["bbox"]
            current_face_ids.add(fid)

            if fid not in self._active:
                # ── ENTRY EVENT ──────────────────────────────────────────
                face_crop = crop_face(frame, bbox)
                img_path = save_face_image(
                    face_crop, self.entries_dir, fid, "entry", self.img_fmt
                )
                self.db.log_event(fid, "entry", img_path, tid)
                logger.info(f"ENTRY | face={fid} | track={tid} | img={img_path}")
                self._active[fid] = {
                    "track_id": tid,
                    "missed_frames": 0,
                    "bbox": bbox
                }
            else:
                # Face still visible — reset missed counter
                self._active[fid]["missed_frames"] = 0
                self._active[fid]["bbox"] = bbox
                self._active[fid]["track_id"] = tid

        # Increment missed counter for faces not seen this frame
        to_exit = []
        for fid, state in self._active.items():
            if fid not in current_face_ids:
                state["missed_frames"] += 1
                if state["missed_frames"] >= self.exit_timeout:
                    to_exit.append(fid)

        for fid in to_exit:
            state = self._active.pop(fid)
            # ── EXIT EVENT ────────────────────────────────────────────────
            # Use the last known bbox for the exit crop
            # (frame here is the current frame — best we can do after exit)
            face_crop = crop_face(frame, state["bbox"])
            img_path = save_face_image(
                face_crop, self.exits_dir, fid, "exit", self.img_fmt
            )
            self.db.log_event(fid, "exit", img_path, state["track_id"])
            logger.info(f"EXIT  | face={fid} | track={state['track_id']} | img={img_path}")

    def flush_all_exits(self, frame):
        """Call at end of stream to fire exit events for everyone still active."""
        for fid, state in list(self._active.items()):
            face_crop = crop_face(frame, state["bbox"])
            img_path = save_face_image(
                face_crop, self.exits_dir, fid, "exit", self.img_fmt
            )
            self.db.log_event(fid, "exit", img_path, state["track_id"])
            logger.info(f"EXIT(flush) | face={fid} | track={state['track_id']}")
        self._active.clear()

    @property
    def active_count(self) -> int:
        return len(self._active)