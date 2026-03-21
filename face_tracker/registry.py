"""
registry.py
In-memory face registry with embedding averaging and exit cooldown.

Key fix: when a face exits, it enters a COOLDOWN period during which
it cannot be matched. This prevents new people from being incorrectly
identified as someone who just left the frame.

After cooldown expires, the face CAN be re-matched (genuine re-entry).
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

ROLLING_WINDOW   = 10   # frames to average per face
EXIT_COOLDOWN_S  = 30   # seconds to block re-matching after exit


class FaceRegistry:
    def __init__(self, config: dict, db: Database, embedder: FaceEmbedder):
        self.threshold = config["recognition"]["similarity_threshold"]
        self.db        = db
        self.embedder  = embedder

        # Stable averaged embeddings used for matching
        self._known: dict[str, np.ndarray] = {}

        # Rolling buffer of recent embeddings per face
        self._buffers: dict[str, list] = defaultdict(list)

        # Cooldown: { face_id: timestamp_of_exit }
        # Face cannot be matched until EXIT_COOLDOWN_S seconds have passed
        self._exit_cooldown: dict[str, float] = {}

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
        """
        Call this when a face exits the frame.
        Puts the face into cooldown — prevents false matches for EXIT_COOLDOWN_S seconds.
        """
        self._exit_cooldown[face_id] = time.time()
        logger.info(
            f"Face {face_id} entered cooldown for {EXIT_COOLDOWN_S}s "
            f"— will not be matched until cooldown expires"
        )

    def _is_in_cooldown(self, face_id: str) -> bool:
        if face_id not in self._exit_cooldown:
            return False
        elapsed = time.time() - self._exit_cooldown[face_id]
        if elapsed < EXIT_COOLDOWN_S:
            return True
        # Cooldown expired — allow re-matching
        del self._exit_cooldown[face_id]
        logger.info(f"Face {face_id} cooldown expired — eligible for re-entry")
        return False

    def _update_average(self, face_id: str, new_embedding: np.ndarray):
        buf = self._buffers[face_id]
        buf.append(new_embedding)
        if len(buf) > ROLLING_WINDOW:
            buf.pop(0)
        averaged = np.mean(buf, axis=0)
        averaged = averaged / (np.linalg.norm(averaged) + 1e-8)
        self._known[face_id] = averaged
        if len(buf) % 5 == 0:
            emb_bytes = self.embedder.embedding_to_bytes(averaged)
            with self.db._get_conn() as conn:
                conn.execute(
                    "UPDATE faces SET embedding = ? WHERE id = ?",
                    (emb_bytes, face_id)
                )

    def identify(self, embedding: np.ndarray) -> tuple[str, bool]:
        """
        Match embedding against known faces (excluding those in cooldown).
        Returns (face_id, is_new).
        """
        best_id    = None
        best_score = -1.0

        for fid, known_emb in self._known.items():
            # Skip faces in exit cooldown
            if self._is_in_cooldown(fid):
                logger.debug(f"Skipping {fid} — in exit cooldown")
                continue

            score = cosine_similarity(embedding, known_emb)
            if score > best_score:
                best_score = score
                best_id    = fid

        if best_score >= self.threshold and best_id is not None:
            logger.info(
                f"Recognised face {best_id} "
                f"(similarity={best_score:.3f}, buffer={len(self._buffers[best_id])})"
            )
            self.db.update_face_last_seen(best_id)
            self._update_average(best_id, embedding)
            return best_id, False

        # New face
        new_id    = str(uuid.uuid4())[:8].upper()
        self._buffers[new_id].append(embedding)
        self._known[new_id] = embedding
        emb_bytes = self.embedder.embedding_to_bytes(embedding)
        self.db.register_face(new_id, emb_bytes)
        logger.info(f"New face registered: {new_id} (best_score={best_score:.3f})")
        return new_id, True

    def update_track_embedding(self, face_id: str, embedding: np.ndarray):
        if face_id in self._known:
            self._update_average(face_id, embedding)

    @property
    def known_count(self) -> int:
        return len(self._known)