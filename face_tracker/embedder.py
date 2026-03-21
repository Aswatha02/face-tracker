"""
embedder.py
ArcFace embedder with proper face alignment.
Uses insightface's face_align utility to align faces to 112x112
before passing to w600k_r50.onnx — this is critical for accuracy.
"""

import numpy as np
import logging
import cv2
import pathlib

logger = logging.getLogger(__name__)

# Standard 5-point facial landmark template for ArcFace alignment
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def _align_face(img: np.ndarray, landmarks_5pt) -> np.ndarray:
    """
    Align face crop to 112x112 using 5 facial landmarks.
    landmarks_5pt: array of shape (5, 2) — left eye, right eye, nose, left mouth, right mouth
    """
    from skimage import transform as sktr
    tform = sktr.SimilarityTransform()
    tform.estimate(landmarks_5pt, ARCFACE_TEMPLATE)
    M = tform.params[0:2, :]
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    return aligned


class FaceEmbedder:
    def __init__(self, config: dict):
        rec_cfg = config["recognition"]
        self.model_name = rec_cfg["model_name"]
        self.embedding_size = rec_cfg["embedding_size"]
        self._session = None
        self._det_session = None
        self._load_model()

    def _load_model(self):
        """Load ArcFace + 2D landmark detector from buffalo_l."""
        import onnxruntime as ort
        model_dir = (
            pathlib.Path.home() / ".insightface" / "models" / self.model_name
        )

        # ArcFace recognition model
        rec_path = model_dir / "w600k_r50.onnx"
        if not rec_path.exists():
            raise FileNotFoundError(f"ArcFace model not found: {rec_path}")
        self._session = ort.InferenceSession(
            str(rec_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info(f"ArcFace model loaded: {rec_path.name}")

        # 2D landmark detector for alignment
        det_path = model_dir / "2d106det.onnx"
        if det_path.exists():
            self._det_session = ort.InferenceSession(
                str(det_path), providers=["CPUExecutionProvider"]
            )
            self._det_input = self._det_session.get_inputs()[0].name
            logger.info(f"Landmark model loaded: {det_path.name}")
        else:
            self._det_session = None
            logger.warning("2d106det.onnx not found — running without alignment")

    def _get_landmarks(self, face_img: np.ndarray):
        """
        Run 2D landmark detection and return 5-point landmarks for alignment.
        Returns None if detection fails.
        """
        if self._det_session is None:
            return None
        try:
            h, w = face_img.shape[:2]
            inp = cv2.resize(face_img, (192, 192))
            inp = inp.astype(np.float32) / 255.0
            inp = inp.transpose(2, 0, 1)[np.newaxis]  # (1,3,192,192)
            preds = self._det_session.run(None, {self._det_input: inp})[0][0]
            # preds shape: (106*2,) — 106 landmark (x,y) pairs
            lmks = preds.reshape(106, 2)
            # Scale back to original image coords
            lmks[:, 0] *= w / 192.0
            lmks[:, 1] *= h / 192.0
            # Extract 5 key points from 106 landmarks:
            # left eye center, right eye center, nose tip, left mouth, right mouth
            five = np.array([
                lmks[38],   # left eye
                lmks[88],   # right eye
                lmks[86],   # nose tip
                lmks[52],   # left mouth corner
                lmks[61],   # right mouth corner
            ], dtype=np.float32)
            return five
        except Exception as e:
            logger.debug(f"Landmark detection failed: {e}")
            return None

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Resize + normalise to 112x112 for ArcFace."""
        img = cv2.resize(face_img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = img.transpose(2, 0, 1)[np.newaxis]  # (1,3,112,112)
        return img

    def get_embedding(self, face_img: np.ndarray):
        """
        Generate a 512-d L2-normalised embedding for a face crop.
        Attempts alignment first; falls back to simple resize if it fails.
        """
        
        if face_img is None or face_img.size == 0:
            return None
        h, w = face_img.shape[:2]
        if h < 32 or w < 32:
            return None

        try:
            # Try landmark alignment
            lmks = self._get_landmarks(face_img)
            if lmks is not None:
                aligned = _align_face(face_img, lmks)
            else:
                aligned = face_img  # fallback: no alignment

            blob = self._preprocess(aligned)
            outputs = self._session.run(None, {self._input_name: blob})
            emb = outputs[0][0]
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            logger.debug(f"Embedding generated for face crop {h}x{w}")
            return emb.astype(np.float32)
        

        except Exception as e:
            logger.warning(f"Embedding error: {e}")
            return None

    def embedding_to_bytes(self, embedding: np.ndarray) -> bytes:
        return embedding.tobytes()

    def bytes_to_embedding(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)