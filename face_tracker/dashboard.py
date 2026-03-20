"""
dashboard.py
Optional web dashboard using Flask + SocketIO.
Shows live visitor count, recent entry/exit events, and face thumbnails.
Run alongside main.py:  python dashboard.py
"""

import os
import json
import base64
import sqlite3
from datetime import datetime
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "face_tracker_secret"
socketio = SocketIO(app, cors_allowed_origins="*")

with open("config.json") as f:
    CONFIG = json.load(f)

DB_PATH = CONFIG["database"]["path"]

# ── DB helpers ────────────────────────────────────────────────────────────────

def query_db(sql, args=()):
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_summary():
    rows = query_db("SELECT * FROM visitor_summary WHERE id=1")
    if not rows:
        return {"unique_visitors": 0, "last_updated": None}
    return rows[0]


def get_recent_events(limit=20):
    return query_db(
        "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,)
    )


def img_to_b64(path):
    """Convert a local image file to base64 string for embedding in HTML."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── HTML template ─────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Tracker Dashboard</title>
<script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e8e8e8; }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 32px; background: #161b22; border-bottom: 1px solid #30363d;
  }
  header h1 { font-size: 18px; font-weight: 600; color: #fff; }
  .live-dot {
    width: 10px; height: 10px; border-radius: 50%; background: #2ea043;
    display: inline-block; margin-right: 8px;
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  .stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; padding: 24px 32px;
  }
  .stat-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 20px; text-align: center;
  }
  .stat-card .num { font-size: 42px; font-weight: 700; color: #58a6ff; }
  .stat-card .label { font-size: 13px; color: #8b949e; margin-top: 6px; }

  .section { padding: 0 32px 32px; }
  .section h2 { font-size: 15px; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: .08em; margin-bottom: 14px; }

  .events-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .events-table th {
    text-align: left; padding: 8px 12px; border-bottom: 1px solid #30363d;
    color: #8b949e; font-weight: 500;
  }
  .events-table td { padding: 9px 12px; border-bottom: 1px solid #1c2128; }
  .events-table tr:hover td { background: #161b22; }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600;
  }
  .badge.entry { background: #0d4429; color: #2ea043; }
  .badge.exit  { background: #3d1a1a; color: #f85149; }

  .thumbs { display: flex; flex-wrap: wrap; gap: 12px; }
  .thumb {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    overflow: hidden; width: 110px; text-align: center;
  }
  .thumb img { width: 110px; height: 110px; object-fit: cover; }
  .thumb .caption {
    font-size: 11px; color: #8b949e; padding: 6px 4px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }
  .thumb .no-img {
    width: 110px; height: 110px; display: flex; align-items: center;
    justify-content: center; color: #30363d; font-size: 32px;
  }
  .updated { font-size: 12px; color: #444d56; text-align: right;
    padding: 0 32px 16px; }
</style>
</head>
<body>
<header>
  <h1><span class="live-dot"></span>Face Tracker Dashboard</h1>
  <span style="font-size:13px;color:#8b949e;" id="last-update">Loading...</span>
</header>

<div class="stats" id="stats">
  <div class="stat-card"><div class="num" id="s-unique">—</div><div class="label">Unique Visitors</div></div>
  <div class="stat-card"><div class="num" id="s-entries">—</div><div class="label">Total Entries</div></div>
  <div class="stat-card"><div class="num" id="s-exits">—</div><div class="label">Total Exits</div></div>
  <div class="stat-card"><div class="num" id="s-active">—</div><div class="label">Currently Active</div></div>
</div>

<div class="section">
  <h2>Recent Events</h2>
  <table class="events-table">
    <thead>
      <tr>
        <th>Time</th><th>Face ID</th><th>Event</th><th>Track ID</th>
      </tr>
    </thead>
    <tbody id="events-body">
      <tr><td colspan="4" style="color:#444;text-align:center;padding:24px">
        Waiting for data...
      </td></tr>
    </tbody>
  </table>
</div>

<div class="section">
  <h2>Face Thumbnails (latest entries)</h2>
  <div class="thumbs" id="thumbs">
    <div style="color:#444;font-size:13px">No faces logged yet.</div>
  </div>
</div>

<script>
const socket = io();

socket.on("update", (data) => {
  // Stats
  document.getElementById("s-unique").textContent  = data.unique_visitors ?? "—";
  document.getElementById("s-entries").textContent = data.total_entries  ?? "—";
  document.getElementById("s-exits").textContent   = data.total_exits    ?? "—";
  document.getElementById("s-active").textContent  = data.active_faces   ?? "—";
  document.getElementById("last-update").textContent =
    "Updated: " + new Date().toLocaleTimeString();

  // Events table
  const tbody = document.getElementById("events-body");
  if (data.events && data.events.length) {
    tbody.innerHTML = data.events.map(e => `
      <tr>
        <td>${e.timestamp.replace("T"," ").slice(0,19)}</td>
        <td style="font-family:monospace">${e.face_id}</td>
        <td><span class="badge ${e.event_type}">${e.event_type}</span></td>
        <td>${e.track_id ?? "—"}</td>
      </tr>`).join("");
  }

  // Thumbnails
  const thumbs = document.getElementById("thumbs");
  if (data.thumbnails && data.thumbnails.length) {
    thumbs.innerHTML = data.thumbnails.map(t => `
      <div class="thumb">
        ${t.b64
          ? `<img src="data:image/jpeg;base64,${t.b64}" alt="${t.face_id}">`
          : `<div class="no-img">👤</div>`}
        <div class="caption">${t.face_id}</div>
      </div>`).join("");
  }
});

// Poll every 2 seconds as fallback
setInterval(() => socket.emit("request_update"), 2000);
socket.emit("request_update");
</script>
</body>
</html>
"""

# ── Socket events ─────────────────────────────────────────────────────────────

@socketio.on("request_update")
def send_update():
    summary   = get_summary()
    events    = get_recent_events(20)
    entries   = [e for e in events if e["event_type"] == "entry"]
    exits_ev  = [e for e in events if e["event_type"] == "exit"]

    # Faces currently active = in entries but no matching exit after
    active_ids = set(e["face_id"] for e in entries) - set(e["face_id"] for e in exits_ev)

    # Thumbnails from recent entry images
    thumbnails = []
    for e in events[:12]:
        if e["event_type"] == "entry":
            b64 = img_to_b64(e.get("image_path"))
            thumbnails.append({"face_id": e["face_id"], "b64": b64})

    socketio.emit("update", {
        "unique_visitors": summary.get("unique_visitors", 0),
        "total_entries":   len(entries),
        "total_exits":     len(exits_ev),
        "active_faces":    len(active_ids),
        "events":          events,
        "thumbnails":      thumbnails,
        "last_updated":    summary.get("last_updated")
    })


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/summary")
def api_summary():
    summary = get_summary()
    events  = get_recent_events(100)
    return jsonify({
        **summary,
        "total_events": len(events),
        "total_entries": sum(1 for e in events if e["event_type"] == "entry"),
        "total_exits":   sum(1 for e in events if e["event_type"] == "exit"),
    })


@app.route("/api/events")
def api_events():
    return jsonify(get_recent_events(50))


if __name__ == "__main__":
    print("\n🌐 Dashboard running at: http://localhost:5050\n")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False)