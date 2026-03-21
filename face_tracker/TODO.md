# Face Tracker Improvement Plan - Accuracy Fixes

## Current Issues (from logs & runs)
- Too many unique visitors (~66 in short run) → false positives (legs/hands/necks detected as faces)
- Missed real faces
- New faces falsely matched as existing (despite threshold=0.92)
- Many new registrations with low best_score (~0.4-0.5)

## Diagnosis
- **YOLOv8n-face.pt**: nano model (fast but low accuracy) → upgrade to yolov8s-face.pt
- **Detection filters** (detector.py): min_size=30, aspect 0.5-1.8, variance>100 may let non-faces through
- **Embedding quality**: Small crops (e.g. 59x43 in logs) or partial faces → bad embeddings → poor matching
- **Similarity threshold 0.92**: Very strict → misses legitimate re-ids (logs show 0.96+ for good matches)
- **Config tweaks tried**: frame_skip=4-5, conf=0.4-0.6, min_size=25-60 → still fragmented tracking

## Detailed Code Update Plan

### Phase 1: Detection Improvements (detector.py)
1. **Use better YOLO model**: Change to yolov8s-face.pt (higher mAP)
2. **Tighten filters**:
   - min_face_size: 50px (reject small crops)
   - aspect ratio: 0.7-1.4 (more square)
   - variance >200
   - Add position heuristic: faces likely in upper 70% of frame (y2 < 0.8*height)
   - Add landmark check: reject if alignment fails in embedder
3. **Lower conf_threshold to 0.6** but compensate with filters

### Phase 2: Embedding Quality (embedder.py)
1. **Skip small crops**: Reject if <64px before embedding
2. **Require alignment success**: Skip if landmarks None
3. **Face quality score**: Use InsightFace's scrfd for quality

### Phase 3: Registry & Matching (registry.py)
1. **Dynamic threshold**: Start 0.9, tighten to 0.95 after 5 sightings
2. **Multi-frame confirmation**: Require 3 consecutive matches > threshold for ID
3. **Embedding averaging**: Increase rolling window to 20

### Phase 4: Tracking (tracker.py + config.json)
1. **Increase max_age to 60**, n_init=5 for stable tracks
2. **Lower iou_threshold to 0.45**

### Phase 5: Config Updates (config.json)
```
"detection": {
  "confidence_threshold": 0.6,
  "min_face_size": 50
},
"recognition": {
  "similarity_threshold": 0.90
},
"tracking": {
  "max_age": 60,
  "min_hits": 5,
  "iou_threshold": 0.45
}
```

## Dependent Files to Edit
- `detector.py`: Filters + model
- `config.json`: Params
- `embedder.py`: Crop validation + alignment check
- `registry.py`: Dynamic threshold + confirmation
- `tracker.py`: Params passed from config
- `main.py`: No change (uses config)

## Follow-up Steps
1. Download yolov8s-face.pt
2. Test on record_20250620_183903.mp4 (shorter, fewer people)
3. Run `python test_pipeline.py` (all pass)
4. Benchmark: unique visitors should drop 50%+, real faces not missed
5. `python main.py` → expect 4-6 uniques for full video

**Approve plan? Reply "yes" or suggest changes.**
