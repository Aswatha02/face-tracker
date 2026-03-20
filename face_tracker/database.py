"""
database.py
Handles all SQLite operations: faces, events, visitor count.
"""

import sqlite3
import os
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Context manager for safe DB connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"DB error: {e}")
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS faces (
                    id          TEXT PRIMARY KEY,
                    first_seen  TEXT NOT NULL,
                    last_seen   TEXT NOT NULL,
                    visit_count INTEGER DEFAULT 1,
                    embedding   BLOB
                );

                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id     TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    image_path  TEXT,
                    track_id    INTEGER,
                    FOREIGN KEY (face_id) REFERENCES faces(id)
                );

                CREATE TABLE IF NOT EXISTS visitor_summary (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    unique_visitors INTEGER DEFAULT 0,
                    last_updated    TEXT
                );

                INSERT OR IGNORE INTO visitor_summary (id, unique_visitors, last_updated)
                VALUES (1, 0, datetime('now'));
            """)
        logger.info(f"Database initialised at {self.db_path}")

    # ── Face operations ──────────────────────────────────────────────────────

    def register_face(self, face_id: str, embedding_bytes: bytes):
        """Insert a newly detected unique face."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO faces (id, first_seen, last_seen, visit_count, embedding)
                   VALUES (?, ?, ?, 1, ?)""",
                (face_id, now, now, embedding_bytes)
            )
            conn.execute(
                """UPDATE visitor_summary
                   SET unique_visitors = unique_visitors + 1,
                       last_updated = ?
                   WHERE id = 1""",
                (now,)
            )
        logger.info(f"Registered new face: {face_id}")

    def update_face_last_seen(self, face_id: str):
        """Bump last_seen for a returning face."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE faces
                   SET last_seen = ?, visit_count = visit_count + 1
                   WHERE id = ?""",
                (now, face_id)
            )

    def get_all_faces(self) -> list:
        """Return all stored faces (without embedding blob)."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, first_seen, last_seen, visit_count FROM faces"
            ).fetchall()
        return [dict(r) for r in rows]

    def face_exists(self, face_id: str) -> bool:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM faces WHERE id = ?", (face_id,)
            ).fetchone()
        return row is not None

    # ── Event operations ─────────────────────────────────────────────────────

    def log_event(self, face_id: str, event_type: str,
                  image_path: str = None, track_id: int = None):
        """Log an entry or exit event."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO events (face_id, event_type, timestamp, image_path, track_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (face_id, event_type, now, image_path, track_id)
            )
        logger.debug(f"Event logged: {event_type} for face {face_id}")

    def get_events(self, face_id: str = None) -> list:
        """Fetch events, optionally filtered by face_id."""
        with self._get_conn() as conn:
            if face_id:
                rows = conn.execute(
                    "SELECT * FROM events WHERE face_id = ? ORDER BY timestamp",
                    (face_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY timestamp"
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Visitor count ─────────────────────────────────────────────────────────

    def get_unique_visitor_count(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT unique_visitors FROM visitor_summary WHERE id = 1"
            ).fetchone()
        return row["unique_visitors"] if row else 0

    def get_summary(self) -> dict:
        """Return a full summary for reporting."""
        with self._get_conn() as conn:
            vs = conn.execute(
                "SELECT * FROM visitor_summary WHERE id = 1"
            ).fetchone()
            total_events = conn.execute(
                "SELECT COUNT(*) as cnt FROM events"
            ).fetchone()
            entries = conn.execute(
                "SELECT COUNT(*) as cnt FROM events WHERE event_type='entry'"
            ).fetchone()
            exits = conn.execute(
                "SELECT COUNT(*) as cnt FROM events WHERE event_type='exit'"
            ).fetchone()
        return {
            "unique_visitors": vs["unique_visitors"] if vs else 0,
            "total_events": total_events["cnt"],
            "total_entries": entries["cnt"],
            "total_exits": exits["cnt"],
            "last_updated": vs["last_updated"] if vs else None
        }