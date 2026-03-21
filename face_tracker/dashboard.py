"""
dashboard.py
Real-time face tracker dashboard — Flask + SocketIO.
Run alongside main.py:  python dashboard.py
Open: http://localhost:5050
"""

import os
import json
import base64
import sqlite3
from datetime import datetime
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "face_tracker_2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

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
        return [dict(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()

def img_to_b64(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face Tracker — Live Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
<style>
  :root {
    --bg:      #080c10;
    --surface: #0d1117;
    --card:    #111820;
    --border:  #1e2d3d;
    --accent:  #00d4ff;
    --green:   #00ff88;
    --amber:   #ffb700;
    --red:     #ff4d6d;
    --text:    #cdd9e5;
    --muted:   #4a6070;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,212,255,.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .wrap { position: relative; z-index: 1; padding: 24px; max-width: 1400px; margin: 0 auto; }

  /* Header */
  header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 28px; padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }
  .logo {
    font-family: var(--font-display);
    font-size: 22px; font-weight: 800;
    letter-spacing: -.02em;
    color: #fff;
  }
  .logo span { color: var(--accent); }
  .live-pill {
    display: flex; align-items: center; gap: 8px;
    background: rgba(0,255,136,.08); border: 1px solid rgba(0,255,136,.2);
    border-radius: 100px; padding: 6px 14px;
    font-size: 12px; font-weight: 700; color: var(--green);
    letter-spacing: .1em;
  }
  .pulse {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.8)} }

  /* Stat cards */
  .stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 24px; }
  .stat {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px 24px;
    position: relative; overflow: hidden;
    transition: border-color .2s;
  }
  .stat:hover { border-color: var(--accent); }
  .stat::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
  }
  .stat.blue::after  { background: var(--accent); }
  .stat.green::after { background: var(--green); }
  .stat.amber::after { background: var(--amber); }
  .stat.red::after   { background: var(--red); }

  .stat-label {
    font-size: 10px; letter-spacing: .15em; font-weight: 700;
    color: var(--muted); text-transform: uppercase; margin-bottom: 10px;
  }
  .stat-value {
    font-family: var(--font-display);
    font-size: 44px; font-weight: 800; line-height: 1;
    color: #fff;
    transition: transform .15s;
  }
  .stat-value.bump { animation: bump .25s ease; }
  @keyframes bump { 0%{transform:scale(1.15)} 100%{transform:scale(1)} }
  .stat-sub { font-size: 11px; color: var(--muted); margin-top: 8px; }

  /* Main grid */
  .grid { display: grid; grid-template-columns: 1fr 380px; gap: 20px; }

  /* Panel */
  .panel {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
  }
  .panel-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; border-bottom: 1px solid var(--border);
  }
  .panel-title {
    font-size: 11px; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: var(--muted);
  }
  .badge {
    font-size: 11px; padding: 3px 10px;
    border-radius: 100px; font-weight: 700;
  }
  .badge-blue { background: rgba(0,212,255,.12); color: var(--accent); }
  .badge-green { background: rgba(0,255,136,.12); color: var(--green); }

  /* Events table */
  .events-wrap { overflow-y: auto; max-height: 420px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    padding: 10px 20px; text-align: left;
    font-size: 10px; letter-spacing: .1em; color: var(--muted);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; background: var(--card); z-index: 2;
  }
  td { padding: 10px 20px; border-bottom: 1px solid rgba(255,255,255,.03); }
  tr:hover td { background: rgba(255,255,255,.02); }
  tr.new-row { animation: slideIn .3s ease; }
  @keyframes slideIn { from{opacity:0;transform:translateX(-8px)} to{opacity:1;transform:none} }

  .tag {
    display: inline-block; padding: 2px 8px;
    border-radius: 4px; font-size: 10px; font-weight: 700;
    letter-spacing: .08em;
  }
  .tag-entry { background: rgba(0,255,136,.12); color: var(--green); }
  .tag-exit  { background: rgba(255,77,109,.12);  color: var(--red); }

  .face-id {
    font-family: var(--font-mono); font-size: 11px;
    color: var(--accent); letter-spacing: .05em;
  }

  /* Thumbnails */
  .thumbs-wrap { padding: 16px; display: flex; flex-direction: column; gap: 10px; overflow-y: auto; max-height: 500px; }
  .thumb-row {
    display: flex; align-items: center; gap: 12px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px;
    animation: slideIn .3s ease;
    transition: border-color .2s;
  }
  .thumb-row:hover { border-color: var(--accent); }
  .thumb-img {
    width: 56px; height: 56px; border-radius: 6px;
    object-fit: cover; background: var(--border); flex-shrink: 0;
    border: 1px solid var(--border);
  }
  .thumb-no-img {
    width: 56px; height: 56px; border-radius: 6px;
    background: var(--border); display: flex; align-items: center;
    justify-content: center; color: var(--muted); font-size: 22px; flex-shrink: 0;
  }
  .thumb-info { flex: 1; min-width: 0; }
  .thumb-id { font-size: 12px; color: var(--accent); font-weight: 700; }
  .thumb-time { font-size: 10px; color: var(--muted); margin-top: 3px; }
  .thumb-badge { margin-top: 5px; }

  /* Footer */
  footer {
    margin-top: 24px; padding-top: 16px;
    border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    font-size: 11px; color: var(--muted);
  }
  .footer-link { color: var(--accent); text-decoration: none; }
  #last-update { color: var(--muted); font-size: 11px; }

  @media (max-width: 900px) {
    .stats { grid-template-columns: repeat(2,1fr); }
    .grid  { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="logo">FACE<span>TRACKER</span> <span style="font-size:13px;font-weight:400;color:var(--muted)">/ live dashboard</span></div>
    <div style="display:flex;gap:12px;align-items:center">
      <span id="last-update">connecting...</span>
      <div class="live-pill"><div class="pulse"></div>LIVE</div>
    </div>
  </header>

  <div class="stats">
    <div class="stat blue">
      <div class="stat-label">Unique Visitors</div>
      <div class="stat-value" id="s-unique">—</div>
      <div class="stat-sub">distinct faces registered</div>
    </div>
    <div class="stat green">
      <div class="stat-label">Active Now</div>
      <div class="stat-value" id="s-active">—</div>
      <div class="stat-sub">faces in frame</div>
    </div>
    <div class="stat amber">
      <div class="stat-label">Total Entries</div>
      <div class="stat-value" id="s-entries">—</div>
      <div class="stat-sub">entry events logged</div>
    </div>
    <div class="stat red">
      <div class="stat-label">Total Exits</div>
      <div class="stat-value" id="s-exits">—</div>
      <div class="stat-sub">exit events logged</div>
    </div>
  </div>

  <div class="grid">
    <div class="panel">
      <div class="panel-head">
        <div class="panel-title">Event Log</div>
        <div class="badge badge-blue" id="event-count">0 events</div>
      </div>
      <div class="events-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Face ID</th>
              <th>Event</th>
              <th>Track</th>
            </tr>
          </thead>
          <tbody id="events-body">
            <tr><td colspan="4" style="text-align:center;padding:32px;color:var(--muted)">
              Waiting for events...
            </td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head">
        <div class="panel-title">Face Thumbnails</div>
        <div class="badge badge-green" id="thumb-count">0 faces</div>
      </div>
      <div class="thumbs-wrap" id="thumbs">
        <div style="text-align:center;padding:32px;color:var(--muted);font-size:12px">
          No faces logged yet.
        </div>
      </div>
    </div>
  </div>

  <footer>
    <div>Face Tracker — Katomaran Hackathon 2026</div>
    <div>Built with YOLOv8 + ArcFace + DeepSort + SQLite</div>
  </footer>

</div>

<script>
const socket = io();
let prevUnique = 0;

function bump(id) {
  const el = document.getElementById(id);
  el.classList.remove('bump');
  void el.offsetWidth;
  el.classList.add('bump');
}

function fmt(ts) {
  if (!ts) return '—';
  return ts.replace('T',' ').slice(0,19);
}

socket.on('update', data => {
  // Stats
  const unique = data.unique_visitors ?? 0;
  if (unique !== prevUnique) { bump('s-unique'); prevUnique = unique; }
  document.getElementById('s-unique').textContent  = unique;
  document.getElementById('s-active').textContent  = data.active_faces ?? 0;
  document.getElementById('s-entries').textContent = data.total_entries ?? 0;
  document.getElementById('s-exits').textContent   = data.total_exits ?? 0;
  document.getElementById('last-update').textContent = 'Updated ' + new Date().toLocaleTimeString();
  document.getElementById('event-count').textContent = (data.total_events ?? 0) + ' events';

  // Events
  if (data.events?.length) {
    document.getElementById('events-body').innerHTML = data.events.map((e,i) => `
      <tr class="${i===0?'new-row':''}">
        <td style="color:var(--muted)">${fmt(e.timestamp)}</td>
        <td><span class="face-id">${e.face_id}</span></td>
        <td><span class="tag tag-${e.event_type}">${e.event_type.toUpperCase()}</span></td>
        <td style="color:var(--muted)">${e.track_id ?? '—'}</td>
      </tr>`).join('');
  }

  // Thumbnails
  if (data.thumbnails?.length) {
    document.getElementById('thumb-count').textContent = data.thumbnails.length + ' faces';
    document.getElementById('thumbs').innerHTML = data.thumbnails.map(t => `
      <div class="thumb-row">
        ${t.b64
          ? `<img class="thumb-img" src="data:image/jpeg;base64,${t.b64}">`
          : `<div class="thumb-no-img">👤</div>`}
        <div class="thumb-info">
          <div class="thumb-id">${t.face_id}</div>
          <div class="thumb-time">${fmt(t.timestamp)}</div>
          <div class="thumb-badge">
            <span class="tag tag-${t.event_type}">${t.event_type.toUpperCase()}</span>
          </div>
        </div>
      </div>`).join('');
  }
});

socket.on('connect', () => {
  document.getElementById('last-update').textContent = 'Connected';
  socket.emit('request_update');
});

setInterval(() => socket.emit('request_update'), 2000);
</script>
</body>
</html>
"""

# ── SocketIO events ───────────────────────────────────────────────────────────

@socketio.on("request_update")
def send_update():
    rows    = query_db("SELECT * FROM visitor_summary WHERE id=1")
    summary = rows[0] if rows else {"unique_visitors": 0}

    events = query_db(
        "SELECT * FROM events ORDER BY timestamp DESC LIMIT 30"
    )
    entries = [e for e in events if e["event_type"] == "entry"]
    exits   = [e for e in events if e["event_type"] == "exit"]

    # Active = entry faces with no subsequent exit
    entered_ids = {e["face_id"] for e in entries}
    exited_ids  = {e["face_id"] for e in exits}
    active      = len(entered_ids - exited_ids)

    # Thumbnails — latest entry image per unique face
    seen = set()
    thumbnails = []
    for e in events:
        if e["face_id"] not in seen:
            seen.add(e["face_id"])
            thumbnails.append({
                "face_id":    e["face_id"],
                "b64":        img_to_b64(e.get("image_path")),
                "timestamp":  e["timestamp"],
                "event_type": e["event_type"]
            })
        if len(thumbnails) >= 12:
            break

    socketio.emit("update", {
        "unique_visitors": summary.get("unique_visitors", 0),
        "total_entries":   len(entries),
        "total_exits":     len(exits),
        "total_events":    len(events),
        "active_faces":    active,
        "events":          events[:20],
        "thumbnails":      thumbnails,
    })


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/summary")
def api_summary():
    from flask import jsonify
    rows    = query_db("SELECT * FROM visitor_summary WHERE id=1")
    summary = rows[0] if rows else {"unique_visitors": 0}
    events  = query_db("SELECT * FROM events")
    return jsonify({
        **summary,
        "total_events":  len(events),
        "total_entries": sum(1 for e in events if e["event_type"] == "entry"),
        "total_exits":   sum(1 for e in events if e["event_type"] == "exit"),
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Face Tracker Dashboard")
    print("  http://localhost:5050")
    print("="*50 + "\n")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False)