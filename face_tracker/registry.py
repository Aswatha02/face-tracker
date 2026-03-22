"""
registry.py
In-memory face registry with:
- Track continuity: same DeepSort track → same face ID always
- Dual threshold: 0.80 for active tracks, 0.87 for historical re-ID
- Identity aging: similarity penalised for long-absent faces
- Rolling embedding average (15 frames)
- Exit cooldown (configurable, default 60s)
- Per-frame active set (same person can't appear twice)
- Freeze embeddings after exit (prevent drift)
- ReIDConfirm: 2-frame voting for historical re-identification
- New track delay: 1 frame before registering new face
"""

import uuid
import time
import numpy as np
import logging
from collections import defaultdict, Counter
from utils import cosine_similarity
from database import Database
from embedder import FaceEmbedder

logger = logging.getLogger(__name__)

ROLLING_WINDOW = 15


class ReIDConfirm:
    def __init__(self, required_frames: int = 2):
        self.required = required_frames
        self._buffer: dict[int, list] = defaultdict(list)

    def vote(self, track_id: int, candidate_id: str):
        buf = self._buffer[track_id]
        buf.append(candidate_id)
        if len(buf) > self.required * 2:
            buf.pop(0)
        if len(buf) < self.required:
            return None
        recent = buf[-self.required:]
        winner, count = Counter(recent).most_common(1)[0]
        if count >= self.required:
            self._buffer[track_id] = []
            return winner
        return None

    def clear(self, track_id: int):
        self._buffer.pop(track_id, None)


class FaceRegistry:
    def __init__(self, config: dict, db: Database, embedder: FaceEmbedder):
        rec_cfg             = config["recognition"]
        self.threshold      = rec_cfg["similarity_threshold"]
        self.reid_threshold = rec_cfg.get("reid_threshold", 0.87)
        self.cooldown_s     = rec_cfg.get("exit_cooldown_s", 60)
        self.db             = db
        self.embedder       = embedder

        self._known: dict[str, np.ndarray]    = {}
        self._buffers: dict[str, list]        = defaultdict(list)
        self._exit_cooldown: dict[str, float] = {}
        self._last_seen: dict[str, float]     = {}
        self._exit_pos: dict[str, tuple]      = {}
        self._frozen: set                     = set()
        self._track_to_face: dict[int, str]   = {}
        self._active_in_frame: set            = set()
        self._reid_confirm                    = ReIDConfirm(required_frames=2)
        # New track delay: track must appear N frames before registering
        self._new_track_buffer: dict[int, int] = defaultdict(int)

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
        logger.info(f"Loaded {loaded} known face(s) from DB")

    def mark_exited(self, face_id: str, bbox: tuple = None):
        self._exit_cooldown[face_id] = time.time()
        self._frozen.add(face_id)
        if bbox:
            x1, y1, x2, y2 = bbox
            self._exit_pos[face_id] = ((x1+x2)/2, (y1+y2)/2)
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
        if face_id not in self._last_seen:
            return raw_sim
        age     = time.time() - self._last_seen[face_id]
        penalty = min(age * 0.001, 0.12)
        return raw_sim - penalty

    def _update_average(self, face_id: str, embedding: np.ndarray):
        if face_id in self._frozen:
            return
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
        # ── Track continuity ──────────────────────────────────────────────
        if track_id is not None and track_id in self._track_to_face:
            face_id = self._track_to_face[track_id]
            if not self._is_in_cooldown(face_id):
                self._update_average(face_id, embedding)
                self._active_in_frame.add(face_id)
                self._last_seen[face_id] = time.time()
                self._new_track_buffer.pop(track_id, None)
                logger.debug(f"Track {track_id} → {face_id} (continuity)")
                return face_id, False

        # ── Embedding matching ────────────────────────────────────────────
        candidates = []
        for fid, known_emb in self._known.items():
            if self._is_in_cooldown(fid):
                continue
            if fid in self._active_in_frame:
                continue
            raw_sim = cosine_similarity(embedding, known_emb)
            adj_sim = self._aged_similarity(fid, raw_sim)
            threshold = self.reid_threshold if fid in self._frozen else self.threshold
            if adj_sim >= threshold:
                candidates.append((fid, adj_sim))

        # ── Ambiguity rejection ───────────────────────────────────────────
        if len(candidates) >= 2:
            candidates.sort(key=lambda x: x[1], reverse=True)
            if candidates[0][1] - candidates[1][1] < 0.04:
                logger.info(
                    f"Ambiguous ({candidates[0][1]:.3f} vs "
                    f"{candidates[1][1]:.3f}) → new ID"
                )
                candidates = []

        if candidates:
            best_id, best_score = max(candidates, key=lambda x: x[1])

            # Historical face needs 2-frame confirmation
            if best_id in self._frozen and track_id is not None:
                confirmed = self._reid_confirm.vote(track_id, best_id)
                if confirmed is None:
                    logger.debug(f"Re-ID pending: {best_id} ({best_score:.3f})")
                    return None, False  # wait for confirmation
                best_id = confirmed

            if best_id in self._frozen:
                self._frozen.discard(best_id)
                logger.info(f"Face {best_id} re-confirmed (sim={best_score:.3f})")

            self.db.update_face_last_seen(best_id)
            self._update_average(best_id, embedding)
            self._active_in_frame.add(best_id)
            self._last_seen[best_id] = time.time()
            if track_id is not None:
                self._track_to_face[track_id] = best_id
                self._reid_confirm.clear(track_id)
                self._new_track_buffer.pop(track_id, None)
            logger.info(f"Recognised {best_id} (sim={best_score:.3f})")
            return best_id, False

        # ── New face — 1-frame delay before registering ───────────────────
        if track_id is not None:
            self._new_track_buffer[track_id] += 1
            if self._new_track_buffer[track_id] < 1:
                return None, False
            self._new_track_buffer.pop(track_id, None)

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