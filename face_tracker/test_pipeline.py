"""
test_pipeline.py
Unit-tests each module independently WITHOUT needing a real video.
Generates a synthetic face frame and runs it through the full pipeline.
Usage: python test_pipeline.py
"""

import cv2
import numpy as np
import os
import json
import sys

PASS = "✅"
FAIL = "❌"

with open("config.json") as f:
    config = json.load(f)

def banner(msg):
    print(f"\n{'─'*50}\n  {msg}\n{'─'*50}")

def make_synthetic_frame():
    """Create a 640x480 BGR frame with a white oval (simulated face)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8) + 30
    cv2.ellipse(frame, (320, 240), (80, 100), 0, 0, 360, (200, 180, 160), -1)
    cv2.circle(frame, (295, 210), 10, (60, 40, 20), -1)  # eye
    cv2.circle(frame, (345, 210), 10, (60, 40, 20), -1)  # eye
    cv2.ellipse(frame, (320, 270), (30, 15), 0, 0, 180, (100, 60, 60), 2)  # mouth
    return frame

# ── Test 1: utils ─────────────────────────────────────────────────────────────
banner("Test 1: utils")
try:
    from utils import load_config, cosine_similarity, crop_face, ensure_dirs
    cfg = load_config("config.json")
    assert "detection" in cfg

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-5

    frame = make_synthetic_frame()
    crop = crop_face(frame, (240, 140, 400, 340))
    assert crop.shape[0] > 0 and crop.shape[1] > 0

    ensure_dirs(config)
    print(f"{PASS} utils: load_config, cosine_similarity, crop_face, ensure_dirs")
except Exception as e:
    print(f"{FAIL} utils: {e}")

# ── Test 2: database ──────────────────────────────────────────────────────────
banner("Test 2: database")
try:
    from database import Database
    db = Database(":memory:")  # use in-memory DB for tests
    db.register_face("TEST01", b"\x00" * 2048)
    assert db.face_exists("TEST01")
    db.log_event("TEST01", "entry", "/tmp/test.jpg", 1)
    db.log_event("TEST01", "exit",  "/tmp/test.jpg", 1)
    count = db.get_unique_visitor_count()
    assert count == 1, f"Expected 1, got {count}"
    summary = db.get_summary()
    assert summary["total_entries"] == 1
    assert summary["total_exits"]   == 1
    print(f"{PASS} database: register, log_event, visitor_count, summary")
except Exception as e:
    print(f"{FAIL} database: {e}")

# ── Test 3: embedder (needs insightface) ──────────────────────────────────────
banner("Test 3: embedder (InsightFace)")
try:
    from embedder import FaceEmbedder
    embedder = FaceEmbedder(config)

    # Test serialisation round-trip
    fake_emb = np.random.rand(512).astype(np.float32)
    byt = embedder.embedding_to_bytes(fake_emb)
    back = embedder.bytes_to_embedding(byt)
    assert np.allclose(fake_emb, back), "Serialisation mismatch"
    print(f"{PASS} embedder: model loaded, serialisation round-trip works")
except Exception as e:
    print(f"{FAIL} embedder: {e}")

# ── Test 4: registry ──────────────────────────────────────────────────────────
banner("Test 4: registry (face matching)")
try:
    from database import Database
    from embedder import FaceEmbedder
    from registry import FaceRegistry
    from utils import cosine_similarity

    db2 = Database(":memory:")
    embedder2 = FaceEmbedder(config)
    reg = FaceRegistry(config, db2, embedder2)

    emb1 = np.random.rand(512).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)

    fid, is_new = reg.identify(emb1)
    assert is_new, "First face should be new"

    # Same embedding → should be recognised
    fid2, is_new2 = reg.identify(emb1)
    assert not is_new2, "Same embedding should be recognised"
    assert fid == fid2, "Same face should get same ID"

    # Very different embedding → new face
    emb2 = -emb1  # opposite direction
    emb2 /= np.linalg.norm(emb2)
    fid3, is_new3 = reg.identify(emb2)
    assert is_new3, "Different embedding should be a new face"
    assert fid3 != fid, "Different face should get different ID"

    print(f"{PASS} registry: identify new, recognise returning, separate distinct faces")
except Exception as e:
    print(f"{FAIL} registry: {e}")

# ── Test 5: event_logger ──────────────────────────────────────────────────────
banner("Test 5: event_logger")
try:
    from database import Database
    from event_logger import EventLogger

    db3 = Database(":memory:")
    ev = EventLogger(config, db3)
    frame = make_synthetic_frame()

    # Simulate face appearing
    tracks = [{"face_id": "AABBCC", "track_id": 1, "bbox": (240, 140, 400, 340)}]
    ev.update(frame, tracks)
    assert ev.active_count == 1

    # Simulate face disappearing for exit_timeout frames
    for _ in range(config["tracking"]["exit_timeout_frames"] + 1):
        ev.update(frame, [])
    assert ev.active_count == 0

    events = db3.get_events("AABBCC")
    types = [e["event_type"] for e in events]
    assert "entry" in types, "Entry event missing"
    assert "exit"  in types, "Exit event missing"

    print(f"{PASS} event_logger: entry fired, exit fired after timeout")
except Exception as e:
    print(f"{FAIL} event_logger: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  All tests complete.")
print("  If all green above, run:  python main.py")
print("="*50 + "\n")