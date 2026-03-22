# Intelligent Face Tracker with Auto-Registration & Visitor Counting

> This project is a part of a hackathon run by https://katomaran.com

---

## Demo Video
https://youtu.be/k2j2UcwrzzE



---

## What It Does

A real-time AI pipeline that processes a video stream or RTSP camera feed to:

- Detect faces using **YOLOv8s** with a 9-layer false-positive filter pipeline
- Generate **512-d ArcFace embeddings** with 106-point facial landmark alignment
- Track faces across frames using **DeepSort** with stable track IDs
- **Auto-register** every new face with a unique UUID
- **Re-identify** returning faces using cosine similarity + rolling embedding average
- Log every **entry and exit** with a timestamped cropped face image
- Save **periodic snapshots** of the annotated frame every 30 seconds
- Count **unique visitors** accurately — re-entries do not increment the count
- Display a **real-time WebSocket dashboard** at `http://localhost:5050`
- Store everything in **SQLite** + structured local file system

---

## Architecture

```
Video File / RTSP Stream
         │
         ▼
┌─────────────────────────────────────────┐
│   FaceDetector (detector.py)            │
│   YOLOv8s + 9-layer filter pipeline:   │
│   size, aspect ratio, edge margin,      │
│   spatial cooldown, variance,           │
│   brightness, skin tone, position       │
│   + NMS deduplication                   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│   DetectionStabilizer (stabilizer.py)   │
│   Requires 2 consecutive frames         │
│   Kills single-frame tile false pos.    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│   FaceEmbedder (embedder.py)            │
│   2d106det.onnx → 106-pt alignment      │
│   w600k_r50.onnx → 512-d ArcFace       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│   FaceTracker (tracker.py)              │
│   DeepSort — stable track IDs          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│   FaceRegistry (registry.py)            │
│   • Dual threshold (0.80 / 0.95)        │
│   • Identity aging penalty              │
│   • ReIDConfirm (2-frame voting)        │
│   • Exit cooldown (2  min)              │
│   • Frozen embeddings after exit        │
│   • Track continuity mapping            │
│   • Per-frame active set                │
│   • Rolling avg (15 frames)             │
└───────────┬──────────────────┬──────────┘
            │                  │
            ▼                  ▼
┌───────────────────┐  ┌──────────────────┐
│  EventLogger      │  │  Database        │
│  (event_logger.py)│  │  (database.py)   │
│  ENTRY/EXIT events│  │  faces           │
│  Cropped images   │  │  events          │
│  events.log       │  │  visitor_summary │
└───────────────────┘  └──────────────────┘
```

---

## Tech Stack

| Module | Technology |
|---|---|
| Face Detection | YOLOv8s (`yolov8s-face.pt`) |
| Face Recognition | ArcFace (`w600k_r50.onnx` from InsightFace buffalo_l) |
| Face Alignment | 106-point landmark detector (`2d106det.onnx`) |
| Tracking | DeepSort (`deep-sort-realtime`) |
| Backend | Python 3.10+ |
| Database | SQLite (faces, events, visitor_summary) |
| Configuration | `config.json` |
| Logging | Python logging + local image store + SQLite |
| Dashboard | Flask + Flask-SocketIO (WebSocket real-time) |
| Camera Input | Video file (dev) / RTSP stream (production) |

---

## Project Structure

```
face_tracker/
├── main.py               # Entry point
├── detector.py           # YOLOv8s + 9-layer filter + NMS + spatial cooldown
├── embedder.py           # ArcFace + landmark alignment via onnxruntime
├── tracker.py            # DeepSort tracker
├── registry.py           # Dual threshold + aging + ReIDConfirm + cooldown
├── stabilizer.py         # Temporal detection stabilizer
├── event_logger.py       # Entry/exit event firing + image saving
├── database.py           # SQLite handler
├── utils.py              # Shared utilities
├── dashboard.py          # Real-time Flask + SocketIO dashboard
├── run_all_videos.py     # Batch processor for multiple videos
├── setup_check.py        # Pre-flight environment checker (15 checks)
├── test_pipeline.py      # Unit tests for all modules
├── config.json           # All tunable parameters
├── requirements.txt      # Dependencies
├── videos/               # Input video files
├── logs/
│   ├── entries/YYYY-MM-DD/   # Cropped face images on entry
│   ├── exits/YYYY-MM-DD/     # Cropped face images on exit
│   ├── snapshots/YYYY-MM-DD/ # Periodic annotated frames
│   └── events.log            # All system events
└── face_tracker.db       # SQLite database
```

---

## Setup Instructions

### 1. Clone and create virtual environment
```bash
git clone https://github.com/Aswatha02/face-tracker
cd face_tracker
python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies
```bash
pip install --upgrade pip wheel setuptools cython
pip install ultralytics deep-sort-realtime opencv-python numpy onnxruntime scikit-image flask flask-socketio eventlet
pip install insightface==0.2.1 --only-binary :all:
```

> **Note on InsightFace:** We use insightface 0.2.1 binary on Windows and load ArcFace weights (`w600k_r50.onnx`) directly via onnxruntime — same weights, same accuracy, no C++ build required.

### 3. Download YOLO face model
```bash
# Download yolov8s-face-lindevs.pt from:
# https://github.com/lindevs/yolov8-face/releases/latest
# Rename to: yolov8s-face.pt and place in project root
```

### 4. Download InsightFace buffalo_l models
```bash
python -c "
import requests, zipfile, pathlib, shutil
url = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
dest = pathlib.Path.home() / '.insightface/models/buffalo_l.zip'
dest.parent.mkdir(parents=True, exist_ok=True)
r = requests.get(url, stream=True)
open(dest, 'wb').write(r.content)
with zipfile.ZipFile(dest) as z: z.extractall(dest.parent)
print('Done')
"
# If files extracted to wrong location:
# mv ~/.insightface/models/*.onnx ~/.insightface/models/buffalo_l/
```

### 5. Verify setup
```bash
python setup_check.py
# Expected: 15/15 checks passed
```

### 6. Run
```bash
# Single video
python main.py

# All videos in sequence (recommended)
python run_all_videos.py --fresh

# Live RTSP stream
# Set use_rtsp: true and rtsp_url in config.json, then:
python main.py

# Dashboard (second terminal)
python dashboard.py
# Open http://localhost:5050
```

---

## Sample config.json

```json
{
  "video_source": "videos/sample.mp4",
  "rtsp_url": "rtsp://camera_ip:554/stream",
  "use_rtsp": false,

  "detection": {
    "model_path": "yolov8s-face.pt",
    "frame_skip": 1,
    "confidence_threshold": 0.55,
    "input_size": 960,
    "min_face_size": 37
  },

  "recognition": {
    "model_name": "buffalo_l",
    "similarity_threshold": 0.80,
    "reid_threshold": 0.95,
    "exit_cooldown_s": 120,
    "embedding_size": 512
  },

  "tracking": {
    "max_age": 100,
    "min_hits": 1,
    "iou_threshold": 0.25,
    "exit_timeout_frames": 180
  },

  "logging": {
    "log_file": "logs/events.log",
    "entries_dir": "logs/entries",
    "exits_dir": "logs/exits",
    "snapshots_dir": "logs/snapshots",
    "snapshot_interval_s": 30,
    "image_format": "jpg",
    "log_level": "INFO"
  },

  "database": { "path": "face_tracker.db" },

  "display": {
    "show_video": true,
    "draw_boxes": true,
    "draw_labels": true,
    "window_name": "Face Tracker",
    "frame_delay_ms": 40
  }
}
```

---

## AI Planning Document

### Planning Steps

1. Read problem statement — identified 3 core modules: detection, recognition, logging
2. Chose ArcFace over `face_recognition` library — ArcFace uses angular margin loss and is significantly more discriminative
3. Loaded ArcFace via onnxruntime directly — bypasses Windows build issues, same accuracy
4. Added 106-point landmark alignment before embedding — aligns face to standard 112x112 template, critical for accuracy
5. Chose DeepSort for tracking — handles occlusion and brief disappearances
6. Designed dual threshold system — active tracks at 0.80, historical re-ID at 0.95
7. Added identity aging penalty — old faces harder to match, prevents long-gap false matches
8. Added ReIDConfirm 2-frame voting — historical identity only reused after consistent multi-frame match
9. Added temporal detection stabilizer — kills single-frame tile/floor false positives
10. Added spatial exit zones — blocks detections near recently exited positions for 2 seconds

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Dual threshold (0.80 / 0.95) | Active tracks vary in embedding; historical re-ID must be very confident |
| Identity aging (0.001/sec, cap 0.12) | Old identities naturally fade, reducing long-gap false matches |
| ReIDConfirm 2 frames | Single-frame similarity spike won't reuse wrong identity |
| Exit cooldown 120s | Covers entire session — exited faces effectively blocked from reuse |
| Frozen embeddings after exit | Prevents embedding drift from corrupting the identity |
| Track continuity mapping | Same DeepSort track → same face ID always, prevents mid-track splits |
| Temporal stabilizer (2 frames) | Tiles appear 1 frame, real faces persist — eliminates most false positives |
| Spatial exit zones (80px, 2s) | Reflective floor where person stood gets blocked briefly after exit |

### Compute Load Estimate

| Component | CPU Load | GPU (if available) |
|---|---|---|
| YOLOv8s detection | 35–50% | 10–20% |
| Landmark detection (2d106det) | 8–12% | 2–4% |
| ArcFace embedding (w600k_r50) | 20–30% | 5–8% |
| DeepSort tracking | 3–5% | — |
| SQLite + file I/O | 2–3% | — |
| **Total** | **~65–85% CPU** | **~18–30% GPU** |

Expected throughput: 6–10 FPS on CPU only, 20–30 FPS with CUDA GPU.

### Feature List (20 features)

1. YOLOv8s face detection
2. 9-layer false-positive filter pipeline
3. NMS deduplication
4. Spatial exit zone cooldown
5. Temporal detection stabilizer
6. 106-point facial landmark alignment
7. 512-d ArcFace embedding via onnxruntime
8. DeepSort multi-face tracking
9. Track continuity (same track → same ID)
10. Auto-registration with UUID
11. Rolling embedding average (15 frames)
12. Dual threshold (active vs historical)
13. Identity aging penalty
14. ReIDConfirm 2-frame voting
15. Exit cooldown (configurable)
16. Frozen embeddings after exit
17. Per-frame active set (no duplicate IDs)
18. Ambiguity rejection (< 0.04 margin → new ID)
19. Entry/exit event logging with cropped images
20. Periodic snapshot every 30 seconds
21. SQLite with faces, events, visitor_summary tables
22. Real-time WebSocket dashboard
23. RTSP stream support
24. Batch video processor
25. Pre-flight environment checker
26. Unit test suite

---

## Sample Output

### Session Summary (actual run output)
```
============================================================
SESSION SUMMARY
  Unique visitors : 11
  Total entries   : 11
  Total exits     : 11
  Total events    : 22
============================================================
```

### events.log (sample)
```
2026-03-22 18:29:43 | INFO | event_logger | ENTRY | face=9EC789C5 | track=3
2026-03-22 18:29:49 | INFO | event_logger | ENTRY | face=32BF977D | track=6
2026-03-22 18:30:17 | INFO | __main__     | SNAPSHOT | 3 face(s)
2026-03-22 18:30:28 | INFO | event_logger | EXIT  | face=9EC789C5 | track=3
2026-03-22 18:30:28 | INFO | registry     | Face 9EC789C5 exited — frozen
2026-03-22 18:31:40 | INFO | event_logger | ENTRY | face=87346776 | track=10
2026-03-22 18:21:16 | INFO | registry     | Recognised 584D6E8A (sim=0.990)
```

---

## Assumptions

1. System works best with semi-frontal face views. Overhead CCTV with side profiles reduces ArcFace accuracy — this is a model limitation.
2. `similarity_threshold` and `reid_threshold` should be tuned per camera setup.
3. For production with GPU, replace `yolov8s-face.pt` with SCRFD-10G from InsightFace for better surveillance-specific detection.
4. RTSP support is built in — set `use_rtsp: true` in config.json for live camera.

---

## Known Limitations

- ArcFace is trained on frontal faces — side/back profiles yield lower similarity scores
- CPU-only processing runs at 5–8 FPS — GPU recommended for real-time 25fps

---

*This project is a part of a hackathon run by https://katomaran.com*