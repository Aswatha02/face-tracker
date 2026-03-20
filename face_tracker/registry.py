"""
registry.py
In-memory face registry backed by SQLite.
Matches new embeddings to known faces using cosine similarity.
Assigns new UUID when no match exceeds the threshold.
"""

import uuid
import numpy as np
import logging
from utils import cosine_similarity
from database import Database
from embedder import FaceEmbedder

logger = logging.getLogger(__name__)


class FaceRegistry:
    def __init__(self, config: dict, db: Database, embedder: FaceEmbedder):
        self.threshold = config["recognition"]["similarity_threshold"]
        self.db = db
        self.embedder = embedder

        # In-memory cache: { face_id: embedding_vector }
        self._known: dict[str, np.ndarray] = {}
        self._load_from_db()

    def _load_from_db(self):
        """Restore known embeddings into memory on startup."""
        faces = self.db.get_all_faces()
        loaded = 0
        for face in faces:
            fid = face["id"]
            # Re-fetch embedding blob
            with self.db._get_conn() as conn:
                row = conn.execute(
                    "SELECT embedding FROM faces WHERE id = ?", (fid,)
                ).fetchone()
            if row and row["embedding"]:
                emb = self.embedder.bytes_to_embedding(row["embedding"])
                self._known[fid] = emb
                loaded += 1
        logger.info(f"Loaded {loaded} known face(s) from DB into memory")

    def identify(self, embedding: np.ndarray) -> tuple[str, bool]:
        """
        Match embedding against known faces.
        Returns (face_id, is_new).
        - is_new=True  → brand-new person, registered now
        - is_new=False → recognised returning face
        """
        best_id = None
        best_score = -1.0

        for fid, known_emb in self._known.items():
            score = cosine_similarity(embedding, known_emb)
            if score > best_score:
                best_score = score
                best_id = fid

        if best_score >= self.threshold and best_id is not None:
            logger.debug(f"Recognised face {best_id} (similarity={best_score:.3f})")
            self.db.update_face_last_seen(best_id)
            return best_id, False

        # New face — register it
        new_id = str(uuid.uuid4())[:8].upper()
        emb_bytes = self.embedder.embedding_to_bytes(embedding)
        self.db.register_face(new_id, emb_bytes)
        self._known[new_id] = embedding
        logger.info(f"New face registered: {new_id} (best_score={best_score:.3f})")
        return new_id, True

    @property
    def known_count(self) -> int:
        return len(self._known)