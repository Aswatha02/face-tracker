"""
registry.py
In-memory face registry with:
- Track continuity: DeepSort track ID → face ID mapping persists
  so the same person never gets two IDs while tracked
- Embedding averaging (rolling window of 15 frames)
- Exit cooldown (30s blocks re-matching after exit)
- Per-frame active set (same person can't appear twice in one frame)
"""

import uuid
import time
import numpy as np
import logging
from collections import defaultdict
from utils import cosine_similarity
from database import Database
from embedder import FaceEmbedder

logger = logging.getLogger(__name__)

ROLLING_WINDOW  = 15
EXIT_COOLDOWN_S = 30


class FaceRegistry:
    def __init__(self, config: dict, db: Database, embedder: FaceEmbedder):
        self.threshold = config["recognition"]["similarity_threshold"]
        self.db        = db
        self.embedder  = embedder

        # face_id → averaged embedding
        self._known: dict[str, np.ndarray] = {}

        # face_id → list of recent embeddings
        self._buffers: dict[str, list] = defaultdict(list)

        # face_id → exit timestamp
        self._exit_cooldown: dict[str, float] = {}

        # track_id → face_id  (continuity: same DeepSort track = same person)
        self._track_to_face: dict[int, str] = {}

        # Per-frame active set (same person can't appear twice)
        self._active_in_frame: set = set()

        self._load_from_db()

    def _load_from_db(self):
        faces  = self.db.get_all_faces()
        loaded = 0
        for face in faces:
            fid = face["id"]
            with self.db._get_conn() as conn:
                row = conn.execute(
                    "SELECT embedding FROM faces WHERE id = ?", (fid,)
                ).fetchone()
            if row and row["embedding"]:
                emb = self.embedder.bytes_to_embedding(row["embedding"])
                self._known[fid] = emb
                loaded += 1
        logger.info(f"Loaded {loaded} known face(s) from DB into memory")

    def mark_exited(self, face_id: str):
        self._exit_cooldown[face_id] = time.time()
        # Clear track mapping so re-entry gets fresh matching
        self._track_to_face = {
            tid: fid for tid, fid in self._track_to_face.items()
            if fid != face_id
        }
        logger.info(
            f"Face {face_id} entered cooldown ({EXIT_COOLDOWN_S}s)"
        )

    def _is_in_cooldown(self, face_id: str) -> bool:
        if face_id not in self._exit_cooldown:
            return False
        if time.time() - self._exit_cooldown[face_id] < EXIT_COOLDOWN_S:
            return True
        del self._exit_cooldown[face_id]
        logger.info(f"Face {face_id} cooldown expired")
        return False

    def _update_average(self, face_id: str, embedding: np.ndarray):
        buf = self._buffers[face_id]
        buf.append(embedding)
        if len(buf) > ROLLING_WINDOW:
            buf.pop(0)
        avg = np.mean(buf, axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        self._known[face_id] = avg
        if len(buf) % 5 == 0:
            with self.db._get_conn() as conn:
                conn.execute(
                    "UPDATE faces SET embedding = ? WHERE id = ?",
                    (self.embedder.embedding_to_bytes(avg), face_id)
                )

    def identify(self, embedding: np.ndarray,
                 track_id: int = None) -> tuple[str, bool]:
        """
        Match embedding to a face ID.

        Track continuity: if we've seen this DeepSort track_id before,
        return the same face_id immediately — no embedding comparison needed.
        This prevents the same person getting two IDs while still tracked.
        """
        # ── Track continuity check ────────────────────────────────────────
        if track_id is not None and track_id in self._track_to_face:
            face_id = self._track_to_face[track_id]
            # Make sure this face isn't in cooldown (would mean it was wrong)
            if not self._is_in_cooldown(face_id):
                self._update_average(face_id, embedding)
                self._active_in_frame.add(face_id)
                logger.debug(
                    f"Track {track_id} → face {face_id} (continuity)"
                )
                return face_id, False

        # ── Embedding matching ────────────────────────────────────────────
        best_id    = None
        best_score = -1.0

        for fid, known_emb in self._known.items():
            if self._is_in_cooldown(fid):
                continue
            if fid in self._active_in_frame:
                continue
            score = cosine_similarity(embedding, known_emb)
            if score > best_score:
                best_score = score
                best_id    = fid

        if best_score >= self.threshold and best_id is not None:
            logger.info(
                f"Recognised {best_id} "
                f"(sim={best_score:.3f}, buf={len(self._buffers[best_id])})"
            )
            self.db.update_face_last_seen(best_id)
            self._update_average(best_id, embedding)
            self._active_in_frame.add(best_id)
            if track_id is not None:
                self._track_to_face[track_id] = best_id
            return best_id, False

        # ── New face ──────────────────────────────────────────────────────
        new_id = str(uuid.uuid4())[:8].upper()
        self._buffers[new_id].append(embedding)
        self._known[new_id] = embedding
        self.db.register_face(new_id, self.embedder.embedding_to_bytes(embedding))
        self._active_in_frame.add(new_id)
        if track_id is not None:
            self._track_to_face[new_id] = new_id
            self._track_to_face[track_id] = new_id
        logger.info(f"New face: {new_id} (best_score={best_score:.3f})")
        return new_id, True

    def update_track_embedding(self, face_id: str, embedding: np.ndarray):
        if face_id in self._known:
            self._update_average(face_id, embedding)

    def reset_frame(self):
        """Call at start of each frame."""
        self._active_in_frame.clear()

    @property
    def known_count(self) -> int:
        return len(self._known)