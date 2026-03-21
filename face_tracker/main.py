"""
main.py
Entry point for the Face Tracker system.
Wires all modules together and runs the main processing loop.
"""

import cv2
import time
import argparse
import logging
import numpy as np

from utils import load_config, setup_logging, ensure_dirs, draw_overlay, crop_face
from database import Database
from detector import FaceDetector
from embedder import FaceEmbedder
from tracker import FaceTracker
from registry import FaceRegistry
from event_logger import EventLogger

logger = logging.getLogger(__name__)


def get_video_source(config: dict):
    """Return cv2.VideoCapture for either file or RTSP."""
    if config.get("use_rtsp", False):
        src = config["rtsp_url"]
        logger.info(f"Connecting to RTSP stream: {src}")
    else:
        src = config["video_source"]
        logger.info(f"Opening video file: {src}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {src}")
    return cap


def run(config_path: str = "config.json"):
    config = load_config(config_path)
    ensure_dirs(config)

    # ── Logging ──────────────────────────────────────────────────────────────
    setup_logging(
        config["logging"]["log_file"],
        config["logging"].get("log_level", "INFO")
    )
    logger.info("=" * 60)
    logger.info("Face Tracker starting up")
    logger.info("=" * 60)

    # ── Init modules ─────────────────────────────────────────────────────────
    db       = Database(config["database"]["path"])
    detector = FaceDetector(config)
    embedder = FaceEmbedder(config)
    tracker  = FaceTracker(config)
    registry = FaceRegistry(config, db, embedder)
    ev_log   = EventLogger(config, db, registry=registry)

    # ── Video source ─────────────────────────────────────────────────────────
    cap  = get_video_source(config)
    show = config["display"]["show_video"]

    fps_counter = 0
    fps_timer   = time.time()
    display_fps = 0.0
    frame       = None

    logger.info("Processing started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream or read error. Stopping.")
                break

            # ── Detection ────────────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── Embed each detection for DeepSort ────────────────────────────
            det_embeddings = []
            for (x1, y1, x2, y2, conf) in detections:
                face_crop = crop_face(frame, (x1, y1, x2, y2))
                emb = embedder.get_embedding(face_crop)
                det_embeddings.append(
                    emb if emb is not None
                    else np.zeros(512, dtype=np.float32)
                )

            # ── Tracking ─────────────────────────────────────────────────────
            tracks = tracker.update(detections, frame, det_embeddings)

            # ── Registry: match each track to a face ID ───────────────────────
            enriched_tracks = []
            for t in tracks:
                bbox      = t["bbox"]
                face_crop = crop_face(frame, bbox)
                embedding = embedder.get_embedding(face_crop)

                if embedding is None:
                    continue

                face_id, is_new = registry.identify(embedding)

                # Keep rolling average fresh every frame face is visible
                if not is_new:
                    registry.update_track_embedding(face_id, embedding)

                enriched_tracks.append({
                    **t,
                    "face_id": face_id,
                    "is_new":  is_new
                })

            # ── Entry / Exit logging ─────────────────────────────────────────
            ev_log.update(frame, enriched_tracks)

            # ── FPS ───────────────────────────────────────────────────────────
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                display_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer   = time.time()

            # ── Display ──────────────────────────────────────────────────────
            if show:
                unique_count = db.get_unique_visitor_count()
                vis_frame    = draw_overlay(
                    frame, enriched_tracks, unique_count, display_fps
                )
                # Fit to screen — max 1280px wide
                h_f, w_f = vis_frame.shape[:2]
                if w_f > 1280:
                    scale     = 1280 / w_f
                    vis_frame = cv2.resize(
                        vis_frame, (1280, int(h_f * scale)),
                        interpolation=cv2.INTER_LINEAR
                    )
                cv2.imshow(config["display"]["window_name"], vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User pressed Q — stopping.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")

    finally:
        fallback = frame if (frame is not None) else np.zeros(
            (480, 640, 3), dtype=np.uint8
        )
        ev_log.flush_all_exits(fallback)
        cap.release()
        if show:
            cv2.destroyAllWindows()

        summary = db.get_summary()
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info(f"  Unique visitors : {summary['unique_visitors']}")
        logger.info(f"  Total entries   : {summary['total_entries']}")
        logger.info(f"  Total exits     : {summary['total_exits']}")
        logger.info(f"  Total events    : {summary['total_events']}")
        logger.info("=" * 60)
        print("\n✅ Done! Check logs/ and face_tracker.db for results.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker")
    parser.add_argument(
        "--config", default="config.json",
        help="Path to config.json (default: config.json)"
    )
    args = parser.parse_args()
    run(args.config)