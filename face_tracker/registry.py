"""
registry.py
In-memory face registry with:
- Track continuity: same DeepSort track → same face ID always
- Dual threshold: 0.78 for active tracks, 0.90 for historical re-ID
- Identity aging: similarity penalised for long-absent faces
- Rolling embedding average (15 frames)
- Exit cooldown (configurable, default 60s)
- Per-frame active set (same person can't appear twice)
- Freeze embeddings after exit (prevent drift)
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

ROLLING_WINDOW = 15


class FaceRegistry:
    def __init__(self, config: dict, db: Database, embedder: FaceEmbedder):
        rec_cfg         = config["recognition"]
        self.threshold  = rec_cfg["similarity_threshold"]      # base (active)
        self.reid_threshold = rec_cfg.get("reid_threshold", 0.90)  # strict re-ID
        self.cooldown_s = rec_cfg.get("exit_cooldown_s", 60)
        self.db         = db
        self.embedder   = embedder

        self._known: dict[str, np.ndarray]  = {}
        self._buffers: dict[str, list]      = defaultdict(list)
        self._exit_cooldown: dict[str, float] = {}
        self._last_seen: dict[str, float]   = {}   # face_id → last seen time
        self._exit_pos: dict[str, tuple]    = {}   # face_id → (cx,cy) at exit
        self._frozen: set                   = set()  # face IDs frozen after exit
        self._track_to_face: dict[int, str] = {}
        self._active_in_frame: set          = set()

        self._load_from_db()

    def _load_from_db(self):
        faces = self.db.get_all_faces()
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
        logger.info(f"Loaded {loaded} known face(s) from DB")

    def mark_exited(self, face_id: str, bbox: tuple = None):
        """
        Call when a face exits.
        - Starts cooldown
        - Freezes embedding (prevents drift)
        - Records exit position for spatial gating
        """
        self._exit_cooldown[face_id] = time.time()
        self._frozen.add(face_id)

        if bbox:
            x1, y1, x2, y2 = bbox
            self._exit_pos[face_id] = ((x1+x2)/2, (y1+y2)/2)

        # Clear track mapping
        self._track_to_face = {
            tid: fid for tid, fid in self._track_to_face.items()
            if fid != face_id
        }
        logger.info(f"Face {face_id} exited — frozen, cooldown {self.cooldown_s}s")

    def _is_in_cooldown(self, face_id: str) -> bool:
        if face_id not in self._exit_cooldown:
            return False
        if time.time() - self._exit_cooldown[face_id] < self.cooldown_s:
            return True
        del self._exit_cooldown[face_id]
        logger.info(f"Face {face_id} cooldown expired")
        return False

    def _aged_similarity(self, face_id: str, raw_sim: float) -> float:
        """
        Apply time-based penalty to similarity score.
        Faces not seen recently are harder to match — prevents
        long-absent faces from being reused for new visitors.
        """
        if face_id not in self._last_seen:
            return raw_sim
        age     = time.time() - self._last_seen[face_id]
        penalty = min(age * 0.001, 0.12)  # max 0.12 penalty
        return raw_sim - penalty

    def _update_average(self, face_id: str, embedding: np.ndarray):
        """Update rolling average — skipped if face is frozen."""
        if face_id in self._frozen:
            return  # don't update frozen embeddings
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
                 track_id: int = None,
                 bbox: tuple = None) -> tuple[str, bool]:
        """
        Match embedding to a face ID using dual-threshold strategy.

        Active tracks   → threshold 0.78 (lenient, same person moving)
        Historical re-ID → threshold 0.90 (strict, prevent wrong reuse)
        Ambiguous match → register new ID (false split > false merge)
        """
        # ── Track continuity ──────────────────────────────────────────────
        if track_id is not None and track_id in self._track_to_face:
            face_id = self._track_to_face[track_id]
            if not self._is_in_cooldown(face_id):
                self._update_average(face_id, embedding)
                self._active_in_frame.add(face_id)
                self._last_seen[face_id] = time.time()
                logger.debug(f"Track {track_id} → {face_id} (continuity)")
                return face_id, False

        # ── Embedding matching ─────────────────────────────────────────────
        candidates = []

        for fid, known_emb in self._known.items():
            if self._is_in_cooldown(fid):
                continue
            if fid in self._active_in_frame:
                continue

            raw_sim = cosine_similarity(embedding, known_emb)
            adj_sim = self._aged_similarity(fid, raw_sim)

            # Dual threshold: stricter for historical (frozen) faces
            threshold = self.reid_threshold if fid in self._frozen else self.threshold

            if adj_sim >= threshold:
                candidates.append((fid, adj_sim))

        # ── Ambiguity rejection ────────────────────────────────────────────
        if len(candidates) >= 2:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_sim   = candidates[0][1]
            second_sim = candidates[1][1]
            if best_sim - second_sim < 0.04:
                # Too close to call → register as new person
                logger.info(
                    f"Ambiguous match ({best_sim:.3f} vs {second_sim:.3f}) "
                    f"→ new ID"
                )
                candidates = []

        if candidates:
            best_id, best_score = max(candidates, key=lambda x: x[1])

            # Unfreeze if confidently re-identified
            if best_id in self._frozen:
                self._frozen.discard(best_id)
                logger.info(f"Face {best_id} re-identified after exit (sim={best_score:.3f})")

            self.db.update_face_last_seen(best_id)
            self._update_average(best_id, embedding)
            self._active_in_frame.add(best_id)
            self._last_seen[best_id] = time.time()
            if track_id is not None:
                self._track_to_face[track_id] = best_id
            logger.info(f"Recognised {best_id} (sim={best_score:.3f})")
            return best_id, False

        # ── New face ──────────────────────────────────────────────────────
        new_id = str(uuid.uuid4())[:8].upper()
        self._buffers[new_id].append(embedding)
        self._known[new_id] = embedding
        self._last_seen[new_id] = time.time()
        self.db.register_face(new_id, self.embedder.embedding_to_bytes(embedding))
        self._active_in_frame.add(new_id)
        if track_id is not None:
            self._track_to_face[track_id] = new_id
        logger.info(f"New face: {new_id}")
        return new_id, True

    def update_track_embedding(self, face_id: str, embedding: np.ndarray):
        if face_id in self._known and face_id not in self._frozen:
            self._update_average(face_id, embedding)

    def reset_frame(self):
        self._active_in_frame.clear()

    @property
    def known_count(self) -> int:
        return len(self._known)