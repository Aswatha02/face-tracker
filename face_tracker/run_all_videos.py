"""
run_all_videos.py
Runs the face tracker on all videos in the videos/ folder sequentially.
DB and logs are NOT cleared between videos — unique visitors accumulate
across all clips (since they're from the same continuous scene).

Usage:
    python run_all_videos.py
    python run_all_videos.py --fresh   # clears DB and logs before starting
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
import json
from pathlib import Path


def clean_state():
    """Delete DB and logs for a fresh start."""
    if os.path.exists("face_tracker.db"):
        os.remove("face_tracker.db")
        print("  Deleted face_tracker.db")
    for d in ["logs/entries", "logs/exits", "logs/snapshots"]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        print(f"  Cleared {d}/")


def get_videos(video_dir: str = "videos") -> list:
    """Return sorted list of all .mp4 files in the videos/ folder."""
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    return videos


def run_video(video_path: str, config_path: str = "config.json"):
    """Update config to point to video_path and run main.py."""
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Update video source
    config["video_source"] = video_path
    config["display"]["show_video"] = True

    # Write temp config
    temp_config = "config_temp.json"
    with open(temp_config, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Processing: {video_path}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "main.py", "--config", temp_config],
        check=False
    )

    # Clean up temp config
    if os.path.exists(temp_config):
        os.remove(temp_config)

    return result.returncode


def print_final_summary():
    """Print DB summary after all videos processed."""
    try:
        from database import Database
        db = Database("face_tracker.db")
        s  = db.get_summary()
        faces = db.get_all_faces()

        print(f"\n{'='*60}")
        print("  FINAL SUMMARY — ALL VIDEOS")
        print(f"{'='*60}")
        print(f"  Unique visitors : {s['unique_visitors']}")
        print(f"  Total entries   : {s['total_entries']}")
        print(f"  Total exits     : {s['total_exits']}")
        print(f"  Total events    : {s['total_events']}")
        print(f"\n  Registered faces:")
        for face in faces:
            print(f"    {face['id']} | first={face['first_seen'][:19]} "
                  f"| last={face['last_seen'][:19]} "
                  f"| visits={face['visit_count']}")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"Could not load summary: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run face tracker on all videos sequentially"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Clear DB and logs before starting"
    )
    parser.add_argument(
        "--videos-dir", default="videos",
        help="Directory containing video files (default: videos/)"
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Config file to use (default: config.json)"
    )
    args = parser.parse_args()

    videos = get_videos(args.videos_dir)
    if not videos:
        print(f"No .mp4 files found in {args.videos_dir}/")
        sys.exit(1)

    print(f"\nFound {len(videos)} video(s):")
    for v in videos:
        print(f"  {v}")

    if args.fresh:
        print("\nFresh start — clearing DB and logs...")
        clean_state()

    print(f"\nProcessing {len(videos)} video(s) sequentially...")
    print("DB persists between videos — unique visitors accumulate.\n")

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {os.path.basename(video)}")
        returncode = run_video(video, args.config)
        if returncode != 0:
            print(f"  Warning: exited with code {returncode}")

    print_final_summary()
    print("✅ All videos processed. Check logs/ and face_tracker.db.\n")