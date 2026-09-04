"""
manual_review_store.py — SQLite-backed metadata store for /predict-url
manual_review cases. Stores only advisor info, the original source URL,
and the escalation reason — never audio. The caller's own URL (e.g. an
S3 link) is the durable copy; this store exists so admins can still find
and open it.
"""
import os
import sqlite3
from datetime import datetime, timedelta

import config

DB_PATH = os.path.join(config.STORAGE_BASE, "manual_review_queue.db")


def init_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pending_reviews (
            id INTEGER PRIMARY KEY,
            advisor_id TEXT NOT NULL,
            advisor_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def add_pending_review(advisor_id: str, advisor_name: str, source_url: str, reason: str, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO pending_reviews (advisor_id, advisor_name, source_url, reason, created_at) VALUES (?, ?, ?, ?, ?)",
        (advisor_id, advisor_name, source_url, reason, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def list_pending_reviews(db_path: str = DB_PATH) -> list:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM pending_reviews ORDER BY created_at DESC, id DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def purge_expired_reviews(max_age_days: int, db_path: str = DB_PATH) -> int:
    """Deletes pending_reviews rows older than max_age_days. Returns the
    number of rows deleted."""
    cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("DELETE FROM pending_reviews WHERE created_at < ?", (cutoff,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted
