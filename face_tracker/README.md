# Intelligent Face Tracker with Auto-Registration & Visitor Counting

> This project is a part of a hackathon run by https://katomaran.com

---

## What It Does

Real-time AI pipeline that:
- Detects faces using **YOLOv8**
- Generates unique 512-d embeddings using **InsightFace (buffalo_l)**
- Tracks faces across frames with **DeepSort**
- Auto-registers new faces with a UUID, recognises returning ones
- Logs every **entry** and **exit** with a timestamped cropped image
- Counts **unique visitors** accurately (re-entry does not increment count)
- Stores everything in **SQLite** + structured local file system

---

## Architecture

```
Video / RTSP stream
        │
        ▼
 YOLOv8 detector  ◄── config.json (frame_skip, confidence)
        │
        ▼
  DeepSort tracker  (stable track IDs across frames)
        │
        ▼
 InsightFace embedder  (512-d face vectors)
        │
        ▼
 Face Registry (cosine similarity match / new UUID)
        │
   ┌────┴─────┐
   ▼          ▼
SQLite DB   Event Logger
(faces,     (entry/exit per face,
 events,     cropped images saved,
 count)      events.log)
```

---

## Setup Instructions

### 1. Clone & create environment
```bash
git clone <your-repo-url>
cd face_tracker
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download YOLO face model
```bash
# Option A: auto-download via ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n-face.pt')"

# Option B: download manually and place in project root
# https://github.com/derronqi/yolov8-face
```

### 3. InsightFace model
Downloaded automatically on first run into `~/.insightface/models/buffalo_l/`

### 4. Add your video
Place your video file in the project root and update `config.json`:
```json
"video_source": "your_video.mp4"
```

### 5. Run
```bash
python main.py
# or with custom config:
python main.py --config config.json
```

---

## Sample config.json

```json
{
  "video_source": "sample_video.mp4",
  "use_rtsp": false,
  "rtsp_url": "rtsp://192.168.1.100:554/stream",

  "detection": {
    "model_path": "yolov8n-face.pt",
    "frame_skip": 3,
    "confidence_threshold": 0.5,
    "input_size": 640
  },

  "recognition": {
    "model_name": "buffalo_l",
    "similarity_threshold": 0.55,
    "embedding_size": 512
  },

  "tracking": {
    "max_age": 30,
    "min_hits": 2,
    "iou_threshold": 0.3,
    "exit_timeout_frames": 45
  },

  "logging": {
    "log_file": "logs/events.log",
    "entries_dir": "logs/entries",
    "exits_dir": "logs/exits",
    "snapshots_dir": "logs/snapshots",
    "image_format": "jpg",
    "log_level": "INFO"
  },

  "database": { "path": "face_tracker.db" },

  "display": {
    "show_video": true,
    "draw_boxes": true,
    "draw_labels": true,
    "window_name": "Face Tracker"
  }
}
```

---

## Output Structure

```
logs/
├── events.log                    ← all system events timestamped
├── entries/
│   └── 2026-03-20/
│       ├── A1B2C3D4_entry_143201_123456.jpg
│       └── ...
└── exits/
    └── 2026-03-20/
        └── A1B2C3D4_exit_143215_654321.jpg

face_tracker.db                   ← SQLite (faces, events, visitor_summary)
```

---

## Assumptions Made

1. **One camera / one scene** — the system is designed for a single video stream.
2. **Front-facing faces** — InsightFace buffalo_l works best with forward-facing faces. Profile faces may not embed reliably.
3. **Lighting consistency** — extreme lighting changes may cause the same person to be registered twice. Adjust `similarity_threshold` down if this happens.
4. **Exit = N missed frames** — a face is considered "exited" after `exit_timeout_frames` consecutive frames without detection. Set higher for slow cameras, lower for fast streams.
5. **GPU optional** — runs on CPU with onnxruntime; GPU dramatically speeds up both YOLO and InsightFace.

---

## Compute Load Estimate

| Component | CPU load | GPU load (if available) |
|---|---|---|
| YOLOv8n detection | ~30–50% (1 core) | ~5–15% GPU |
| InsightFace embedding | ~20–40% (1 core) | ~5–10% GPU |
| DeepSort tracking | ~5% | — |
| SQLite + file I/O | ~2–5% | — |
| **Total** | **~60–80% CPU** | **~15–25% GPU** |

Expected throughput: ~8–15 FPS on CPU-only, ~25–30 FPS with CUDA GPU.

---

## AI Planning Document

### Planning Steps
1. Read problem statement → identified 3 core modules: detection, recognition, logging
2. Chose InsightFace over `face_recognition` lib for production-grade accuracy (ArcFace backbone)
3. Chose DeepSort for tracking because it handles occlusion and brief disappearances gracefully
4. Designed DB schema first (faces, events, visitor_summary) before writing any processing code
5. Made frame_skip configurable via config.json as required
6. Separated entry/exit logic into its own module with a timeout-based exit detector

### Key Design Decisions
- **Cosine similarity threshold 0.55** → tuned conservatively to avoid false matches; raise to 0.65 for stricter matching
- **Exit timeout 45 frames** → ~1.5s at 30fps; prevents false exits during brief occlusions
- **In-memory embedding cache** → avoids re-querying DB every frame; loaded from DB on startup
- **UUID face IDs** → globally unique, first 8 chars used for display

---

## Demo Video
[Add your Loom / YouTube link here]