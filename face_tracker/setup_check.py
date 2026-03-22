"""
setup_check.py
Run this BEFORE main.py to verify all dependencies and models are ready.
Usage: python setup_check.py
"""

import sys
import os

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []

def check(label, fn):
    try:
        msg = fn()
        results.append((PASS, label, msg or ""))
        print(f"{PASS} {label}")
    except Exception as e:
        results.append((FAIL, label, str(e)))
        print(f"{FAIL} {label}: {e}")

# ── Python version ────────────────────────────────────────────────────────────
check("Python >= 3.10",
    lambda: (_ for _ in ()).throw(RuntimeError(f"Got {sys.version}"))
    if sys.version_info < (3, 10)
    else f"{sys.version.split()[0]}"
)

# ── Core imports ──────────────────────────────────────────────────────────────
check("opencv-python",        lambda: __import__("cv2") and __import__("cv2").__version__)
check("numpy",                lambda: __import__("numpy").__version__)
check("ultralytics (YOLO)",   lambda: __import__("ultralytics").__version__)
check("insightface",          lambda: __import__("insightface").__version__)
check("deep_sort_realtime",   lambda: __import__("deep_sort_realtime") and "ok")
check("onnxruntime",          lambda: __import__("onnxruntime").__version__)

# ── YOLO model file ───────────────────────────────────────────────────────────
import json

with open("config_fixed.json") as f:
    cfg = json.load(f)



model_path = cfg["detection"]["model_path"]
check(f"YOLO model file ({model_path})",
    lambda: "found" if os.path.exists(model_path)
    else (_ for _ in ()).throw(FileNotFoundError(
        f"Run: python -c \"from ultralytics import YOLO; YOLO('{model_path}')\""
    ))
)

# ── InsightFace model download ────────────────────────────────────────────────
def check_insightface_model():
    import onnxruntime as ort
    import pathlib
    model_path = pathlib.Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Not found: {model_path}")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return f"ArcFace ready ({model_path.name})"
check("InsightFace buffalo_l model", check_insightface_model)

# ── Log directories ───────────────────────────────────────────────────────────
check("logs/entries dir",   lambda: os.makedirs("logs/entries", exist_ok=True) or "ok")
check("logs/exits dir",     lambda: os.makedirs("logs/exits", exist_ok=True) or "ok")
check("logs/snapshots dir", lambda: os.makedirs("logs/snapshots", exist_ok=True) or "ok")

# ── Video source ──────────────────────────────────────────────────────────────
video_src = cfg["video_source"]
check(f"Video file ({video_src})",
    lambda: "found" if os.path.exists(video_src)
    else (_ for _ in ()).throw(FileNotFoundError(
        f"Place your video at: {video_src}"
    ))
)

# ── SQLite DB (create if needed) ──────────────────────────────────────────────
check("SQLite DB initialises",
    lambda: __import__("database").Database(cfg["database"]["path"]) and "ok"
)

# ── GPU detection (optional) ──────────────────────────────────────────────────
def check_gpu():
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return "CUDA GPU available 🚀"
    return "CPU only (runs fine, just slower)"
check("GPU / CUDA (optional)", check_gpu)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"  {passed} passed   {failed} failed  (out of {len(results)})")
print("="*50)

if failed == 0:
    print("\n🎉 All checks passed! You're ready to run:\n")
    print("   python main.py\n")
else:
    print(f"\n⚠️  Fix the {failed} failed item(s) above, then re-run this script.\n")