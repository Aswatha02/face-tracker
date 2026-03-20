"""
embedder.py
InsightFace-based face embedding generator.
Takes a cropped face image and returns a 512-d embedding vector.
"""

import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)


class FaceEmbedder:
    def __init__(self, config: dict):
        rec_cfg = config["recognition"]
        self.model_name = rec_cfg["model_name"]
        self.embedding_size = rec_cfg["embedding_size"]
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load InsightFace model (downloads on first run)."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            logger.info(f"Loading InsightFace model: {self.model_name}")
            self._app = FaceAnalysis(
                name=self.model_name,
                allowed_modules=["recognition", "detection"]
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            raise

    def get_embedding(self, face_img: np.ndarray) -> np.ndarray | None:
        """
        Generate a 512-d embedding for a face crop.
        Returns None if no face is found in the crop.
        """
        if face_img is None or face_img.size == 0:
            return None

        # Ensure minimum size for InsightFace
        h, w = face_img.shape[:2]
        if h < 32 or w < 32:
            logger.debug("Face crop too small for embedding, skipping")
            return None

        try:
            faces = self._app.get(face_img)
            if not faces:
                logger.debug("InsightFace found no face in crop")
                return None

            # Use the most prominent (highest det_score) face
            best = max(faces, key=lambda f: f.det_score)
            embedding = best.normed_embedding  # already L2-normalised
            return embedding.astype(np.float32)

        except Exception as e:
            logger.warning(f"Embedding error: {e}")
            return None

    def embedding_to_bytes(self, embedding: np.ndarray) -> bytes:
        """Serialise embedding for DB storage."""
        return embedding.tobytes()

    def bytes_to_embedding(self, data: bytes) -> np.ndarray:
        """Deserialise embedding from DB."""
        return np.frombuffer(data, dtype=np.float32)