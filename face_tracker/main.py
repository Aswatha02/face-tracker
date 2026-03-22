"""
main.py
Entry point for the Face Tracker system.
"""

import cv2
import os
import time
import argparse
import logging
import numpy as np
from datetime import datetime

from utils import load_config, setup_logging, ensure_dirs, draw_overlay, crop_face
from database import Database
from detector import FaceDetector
from embedder import FaceEmbedder
from tracker import FaceTracker
from registry import FaceRegistry
from event_logger import EventLogger
from stabilizer import DetectionStabilizer

logger = logging.getLogger(__name__)


def get_video_source(config: dict):
    if config.get("use_rtsp", False):
        src = config["rtsp_url"]
        logger.info(f"Connecting to RTSP: {src}")
    else:
        src = config["video_source"]
        logger.info(f"Opening video: {src}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {src}")
    return cap


def run(config_path: str = "config.json"):
    config = load_config(config_path)
    ensure_dirs(config)

    setup_logging(
        config["logging"]["log_file"],
        config["logging"].get("log_level", "INFO")
    )
    logger.info("=" * 60)
    logger.info("Face Tracker starting up")
    logger.info("=" * 60)

    db         = Database(config["database"]["path"])
    detector   = FaceDetector(config)
    embedder   = FaceEmbedder(config)
    tracker    = FaceTracker(config)
    registry   = FaceRegistry(config, db, embedder)
    ev_log     = EventLogger(config, db, registry=registry, detector=detector)
    stabilizer = DetectionStabilizer(min_frames=3, position_tolerance=20)

    cap   = get_video_source(config)
    show  = config["display"]["show_video"]
    delay = config["display"].get("frame_delay_ms", 40)

    snapshots_dir     = config["logging"].get("snapshots_dir", "logs/snapshots")
    snapshot_interval = config["logging"].get("snapshot_interval_s", 30)
    snapshot_timer    = time.time()

    fps_counter = 0
    fps_timer   = time.time()
    display_fps = 0.0
    frame       = None

    logger.info(f"Processing | delay={delay}ms | Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream. Stopping.")
                break

            # ── Reset per-frame state ─────────────────────────────────────
            registry.reset_frame()

            # ── Detection ─────────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── Temporal stabilizer — kills tile false positives ──────────
            detections = stabilizer.update(detections)

            # ── Embed detections for DeepSort ─────────────────────────────
            det_embeddings = []
            for (x1, y1, x2, y2, conf) in detections:
                face_crop = crop_face(frame, (x1, y1, x2, y2))
                emb = embedder.get_embedding(face_crop)
                det_embeddings.append(
                    emb if emb is not None
                    else np.zeros(512, dtype=np.float32)
                )

            # ── Tracking ──────────────────────────────────────────────────
            tracks = tracker.update(detections, frame, det_embeddings)

            # ── Registry — with track_id for continuity ───────────────────
            enriched_tracks = []
            for t in tracks:
                bbox      = t["bbox"]
                track_id  = t["track_id"]
                face_crop = crop_face(frame, bbox)
                embedding = embedder.get_embedding(face_crop)
                if embedding is None:
                    continue

                # Pass track_id so registry can use track continuity
                face_id, is_new = registry.identify(embedding, track_id=track_id)

                if not is_new:
                    registry.update_track_embedding(face_id, embedding)

                enriched_tracks.append({
                    **t,
                    "face_id": face_id,
                    "is_new":  is_new
                })

            # ── Entry / Exit logging ──────────────────────────────────────
            ev_log.update(frame, enriched_tracks)

            # ── FPS ───────────────────────────────────────────────────────
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                display_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer   = time.time()

            # ── Periodic snapshot ─────────────────────────────────────────
            if enriched_tracks and (time.time() - snapshot_timer >= snapshot_interval):
                date_str  = datetime.now().strftime("%Y-%m-%d")
                snap_dir  = os.path.join(snapshots_dir, date_str)
                os.makedirs(snap_dir, exist_ok=True)
                ts        = datetime.now().strftime("%H%M%S")
                snap_path = os.path.join(snap_dir, f"snapshot_{ts}.jpg")
                snap_frame = draw_overlay(
                    frame, enriched_tracks,
                    db.get_unique_visitor_count(), display_fps
                )
                cv2.imwrite(snap_path, snap_frame)
                logger.info(f"SNAPSHOT | {snap_path} | {len(enriched_tracks)} face(s)")
                snapshot_timer = time.time()

            # ── Display ───────────────────────────────────────────────────
            if show:
                unique_count = db.get_unique_visitor_count()
                vis_frame    = draw_overlay(
                    frame, enriched_tracks, unique_count, display_fps
                )
                h_f, w_f = vis_frame.shape[:2]
                if w_f > 1280:
                    scale     = 1280 / w_f
                    vis_frame = cv2.resize(
                        vis_frame, (1280, int(h_f * scale)),
                        interpolation=cv2.INTER_LINEAR
                    )
                cv2.imshow(config["display"]["window_name"], vis_frame)
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    logger.info("User pressed Q.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        fallback = frame if frame is not None else np.zeros(
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
    parser = argparse.ArgumentParser(description="Face Tracker")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    run(args.config)