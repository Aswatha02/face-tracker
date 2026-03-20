"""
utils.py
Shared utility functions used across all modules.
"""

import json
import os
import cv2
import numpy as np
import logging
from datetime import datetime
from pathlib import Path


def load_config(config_path: str = "config.json") -> dict:
    """Load and return the JSON config file."""
    with open(config_path, "r") as f:
        return json.load(f)


def setup_logging(log_file: str, level: str = "INFO") -> logging.Logger:
    """
    Set up root logger to write to both file and console.
    Called once at startup from main.py.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("face_tracker")


def crop_face(frame: np.ndarray, bbox: tuple, padding: float = 0.2) -> np.ndarray:
    """
    Crop a face region from a frame with optional padding.
    bbox: (x1, y1, x2, y2)
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Add padding
    pw = int((x2 - x1) * padding)
    ph = int((y2 - y1) * padding)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)

    return frame[y1:y2, x1:x2].copy()


def save_face_image(face_img: np.ndarray, directory: str,
                    face_id: str, event_type: str,
                    fmt: str = "jpg") -> str:
    """
    Save a cropped face image to disk.
    Returns the saved file path.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join(directory, date_str)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%H%M%S_%f")[:12]
    filename = f"{face_id}_{event_type}_{timestamp}.{fmt}"
    filepath = os.path.join(save_dir, filename)

    cv2.imwrite(filepath, face_img)
    return filepath


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def draw_overlay(frame: np.ndarray, tracks: list,
                 visitor_count: int, fps: float) -> np.ndarray:
    """
    Draw bounding boxes, labels, and stats on the frame.
    tracks: list of dicts with keys: bbox, face_id, track_id, is_new
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
        face_id = t.get("face_id", "unknown")
        track_id = t.get("track_id", -1)
        is_new = t.get("is_new", False)

        # Box colour: green for new face, blue for returning
        color = (0, 200, 0) if is_new else (255, 130, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{face_id[:6]}  T:{track_id}"
        label_y = max(y1 - 8, 14)
        cv2.putText(overlay, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Stats panel (top-left)
    cv2.rectangle(overlay, (0, 0), (260, 55), (20, 20, 20), -1)
    cv2.putText(overlay, f"Unique visitors: {visitor_count}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(overlay, f"Active tracks:   {len(tracks)}",
                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # FPS (top-right)
    cv2.putText(overlay, f"{fps:.1f} fps", (w - 90, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    return overlay


def ensure_dirs(config: dict):
    """Create all required log directories at startup."""
    for key in ["entries_dir", "exits_dir", "snapshots_dir"]:
        path = config["logging"][key]
        Path(path).mkdir(parents=True, exist_ok=True)
    log_dir = os.path.dirname(config["logging"]["log_file"])
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)