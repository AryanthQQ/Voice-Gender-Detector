"""
fingerprint_store.py — SQLite-backed storage for audio fingerprints, used to
detect replay-attack reuse of the same clip across different advisor_ids.

Stores only fingerprints (small, one-way) — never raw audio.
"""
import os
import sqlite3
from datetime import datetime

from fingerprint import is_match

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'fingerprints.db')


def init_db(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audio_fingerprints (
            id INTEGER PRIMARY KEY,
            fingerprint BLOB NOT NULL,
            advisor_id TEXT NOT NULL,
            advisor_name TEXT NOT NULL,
            decision TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def find_cross_advisor_match(fingerprint_bytes: bytes, advisor_id: str, db_path: str = DB_PATH):
    """Returns the first row (as a dict) whose fingerprint is within
    MATCH_THRESHOLD Hamming distance of fingerprint_bytes AND whose
    advisor_id differs from the given advisor_id. Returns None if no such
    row exists."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM audio_fingerprints WHERE advisor_id != ?", (advisor_id,)
    ).fetchall()
    conn.close()

    for row in rows:
        if is_match(fingerprint_bytes, row['fingerprint']):
            return dict(row)
    return None


def store_fingerprint(fingerprint_bytes: bytes, advisor_id: str, advisor_name: str, decision: str, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO audio_fingerprints (fingerprint, advisor_id, advisor_name, decision, created_at) VALUES (?, ?, ?, ?, ?)",
        (fingerprint_bytes, advisor_id, advisor_name, decision, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
