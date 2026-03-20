"""
embedder.py
Loads ArcFace (w600k_r50.onnx) directly via onnxruntime.
Bypasses insightface version incompatibility entirely.
Since YOLOv8 handles detection, we only need the recognition model.
"""

import numpy as np
import logging
import cv2
import pathlib

logger = logging.getLogger(__name__)


class FaceEmbedder:
    def __init__(self, config: dict):
        rec_cfg = config["recognition"]
        self.model_name = rec_cfg["model_name"]
        self.embedding_size = rec_cfg["embedding_size"]
        self._session = None
        self._load_model()

    def _load_model(self):
        """Load ArcFace onnx model directly via onnxruntime."""
        import onnxruntime as ort

        # Look for w600k_r50.onnx in the buffalo_l model directory
        model_path = (
            pathlib.Path.home()
            / ".insightface" / "models" / self.model_name / "w600k_r50.onnx"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at {model_path}\n"
                f"Make sure buffalo_l models are in: {model_path.parent}"
            )

        logger.info(f"Loading ArcFace model from: {model_path}")
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("ArcFace model loaded successfully")

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """
        Resize + normalise face crop to 112x112 as required by ArcFace.
        """
        img = cv2.resize(face_img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5          # normalise to [-1, 1]
        img = img.transpose(2, 0, 1)          # HWC → CHW
        img = np.expand_dims(img, axis=0)     # add batch dim → (1,3,112,112)
        return img

    def get_embedding(self, face_img: np.ndarray):
        """
        Generate a 512-d embedding for a face crop.
        Returns None if image is invalid.
        """
        if face_img is None or face_img.size == 0:
            return None

        h, w = face_img.shape[:2]
        if h < 32 or w < 32:
            logger.debug("Face crop too small, skipping")
            return None

        try:
            blob = self._preprocess(face_img)
            outputs = self._session.run(None, {self._input_name: blob})
            embedding = outputs[0][0]                     # shape: (512,)
            # L2 normalise
            norm = np.linalg.norm(embedding)
            embedding = embedding / (norm + 1e-8)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Embedding error: {e}")
            return None

    def embedding_to_bytes(self, embedding: np.ndarray) -> bytes:
        return embedding.tobytes()

    def bytes_to_embedding(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)