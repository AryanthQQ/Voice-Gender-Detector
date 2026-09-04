# /predict-url No Local Audio Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `/predict-url` from ever writing a persistent local copy of audio to disk, for any outcome. Replace local retention for manual_review cases with a lightweight SQLite metadata record (advisor info + the caller-provided source URL + the escalation reason), so a human reviewer can still find and open the original recording via its own durable URL.

**Architecture:** A new module (`manual_review_store.py`) mirroring the existing `fingerprint_store.py` SQLite pattern. `main.py`'s `_predict_url_sync` is rewired at its three `_keep_for_manual_review` call sites to record metadata + delete the audio immediately instead of moving it into local storage. `/predict`'s existing local-retention behavior is completely untouched.

**Tech Stack:** stdlib `sqlite3`, pytest, `unittest.mock`.

## Global Constraints

- Scope is `/predict-url` only. `/predict` (direct upload, no external URL to fall back on) keeps its existing `_keep_for_manual_review`/`MANUAL_REVIEW_DIR` behavior completely unchanged — do not touch it.
- No audio is ever written to a persistent local path by `/predict-url`, for any outcome (accept, reject, or manual_review) — accept/reject already delete immediately today; manual_review must now do the same.
- Store only metadata for manual_review cases: `advisor_id`, `advisor_name`, `source_url`, `reason`, `created_at` — never audio.
- No change to `/predict-url`'s request or response JSON shape, and no change to the email notification's content (it already includes `source_url`).
- `_keep_for_manual_review`, `MANUAL_REVIEW_DIR`, and the existing `/recordings` endpoint are NOT removed — `/predict` still needs them.
- Use the project's existing interpreter for every command: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe`.

---

### Task 1: Manual-review metadata store

**Files:**
- Create: `manual_review_store.py`
- Create: `tests/test_manual_review_store.py`

**Interfaces:**
- Produces: `init_db(db_path: str = DB_PATH) -> None`, `add_pending_review(advisor_id: str, advisor_name: str, source_url: str, reason: str, db_path: str = DB_PATH) -> None`, `list_pending_reviews(db_path: str = DB_PATH) -> list` (each item a dict with keys `id`, `advisor_id`, `advisor_name`, `source_url`, `reason`, `created_at`, most-recent-first). Task 2's `main.py` changes call `add_pending_review` at each of its three sites; the new admin endpoint calls `list_pending_reviews`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_manual_review_store.py`:

```python
import os
from manual_review_store import init_db, add_pending_review, list_pending_reviews


def test_list_pending_reviews_empty_on_fresh_db(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    assert list_pending_reviews(db_path) == []


def test_add_and_list_round_trip(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    add_pending_review('advisor-1', 'Alice', 'https://s3.example.com/audio1.wav', 'Ambiguous voice', db_path)

    reviews = list_pending_reviews(db_path)
    assert len(reviews) == 1
    assert reviews[0]['advisor_id'] == 'advisor-1'
    assert reviews[0]['advisor_name'] == 'Alice'
    assert reviews[0]['source_url'] == 'https://s3.example.com/audio1.wav'
    assert reviews[0]['reason'] == 'Ambiguous voice'
    assert 'created_at' in reviews[0]


def test_list_returns_most_recent_first(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    add_pending_review('advisor-1', 'Alice', 'url1', 'reason1', db_path)
    add_pending_review('advisor-2', 'Bob', 'url2', 'reason2', db_path)

    reviews = list_pending_reviews(db_path)
    assert len(reviews) == 2
    assert reviews[0]['advisor_id'] == 'advisor-2'
    assert reviews[1]['advisor_id'] == 'advisor-1'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_manual_review_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manual_review_store'`

- [ ] **Step 3: Write the implementation**

Create `manual_review_store.py`:

```python
"""
manual_review_store.py — SQLite-backed metadata store for /predict-url
manual_review cases. Stores only advisor info, the original source URL,
and the escalation reason — never audio. The caller's own URL (e.g. an
S3 link) is the durable copy; this store exists so admins can still find
and open it.
"""
import os
import sqlite3
from datetime import datetime

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_manual_review_store.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add manual_review_store.py tests/test_manual_review_store.py
git commit -m "Add SQLite-backed manual-review metadata store"
```

---

### Task 2: Wire into /predict-url and add the admin queue endpoint

**Files:**
- Modify: `main.py` (see exact line-anchored edits below; line numbers are from the pre-Task-2 file state)
- Modify: `tests/test_main_integration.py`

**Interfaces:**
- Consumes: `manual_review_store.init_db` / `add_pending_review` / `list_pending_reviews` (Task 1).
- Produces: nothing new consumed by a later task — this is the last task.

- [ ] **Step 1: Write the failing integration tests**

Read `tests/test_main_integration.py` first to see its existing imports/fixtures (`import main`, `import os`, `import pytest`, `from fastapi.testclient import TestClient`, a session-scoped `client` fixture, `FIXTURE`, `from unittest.mock import patch, MagicMock`) — reuse these, don't redefine them.

Add these tests to the file (keep everything already in it untouched):

```python
def test_predict_url_manual_review_stores_metadata_not_local_audio(client):
    audio_bytes = _read_fixture_bytes()
    mock_response = MagicMock()
    mock_response.read.return_value = audio_bytes
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False

    headers = {"X-API-Key": main.config.API_KEY}

    with patch('main._assert_public_url'), \
         patch('main.urllib.request.urlopen', return_value=mock_response), \
         patch('main.advanced_deepfake_detector.predict', return_value={'is_ai': True, 'confidence': 90.0, 'reason': 'AI/Synthetic voice detected (90.0%)', 'status': 'success'}), \
         patch('fingerprint.compute_fingerprint', return_value=b'\x22' * 16), \
         patch('fingerprint_store.find_cross_advisor_match', return_value=None), \
         patch('fingerprint_store.store_fingerprint'), \
         patch('main._keep_for_manual_review') as mock_keep_local, \
         patch('manual_review_store.add_pending_review') as mock_add_review:

        resp = client.post(
            "/predict-url",
            headers=headers,
            json={"url": "http://test.local/clip-deepfake.wav", "userId": "advisor-D", "fullname": "Advisor D"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data['decision'] == 'uncertain'
    mock_keep_local.assert_not_called()
    mock_add_review.assert_called_once()
    call_args = mock_add_review.call_args[0]
    assert call_args[0] == 'advisor-D'
    assert call_args[2] == 'http://test.local/clip-deepfake.wav'


def test_predict_url_replay_attack_stores_metadata_not_local_audio(client):
    audio_bytes = _read_fixture_bytes()
    mock_response = MagicMock()
    mock_response.read.return_value = audio_bytes
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False

    headers = {"X-API-Key": main.config.API_KEY}

    with patch('main._assert_public_url'), \
         patch('main.urllib.request.urlopen', return_value=mock_response), \
         patch('main.advanced_deepfake_detector.predict') as mock_deepfake, \
         patch('gender_verifier.classify_gender') as mock_classify, \
         patch('fingerprint.compute_fingerprint', return_value=b'\x33' * 16), \
         patch('fingerprint_store.find_cross_advisor_match', return_value={'advisor_id': 'advisor-E', 'advisor_name': 'Advisor E'}), \
         patch('main._keep_for_manual_review') as mock_keep_local, \
         patch('manual_review_store.add_pending_review') as mock_add_review:

        resp = client.post(
            "/predict-url",
            headers=headers,
            json={"url": "http://test.local/clip-replay.wav", "userId": "advisor-F", "fullname": "Advisor F"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data['decision'] == 'uncertain'
    mock_deepfake.assert_not_called()
    mock_classify.assert_not_called()
    mock_keep_local.assert_not_called()
    mock_add_review.assert_called_once()
    call_args = mock_add_review.call_args[0]
    assert call_args[0] == 'advisor-F'
    assert call_args[2] == 'http://test.local/clip-replay.wav'


def test_predict_url_ambiguous_gender_stores_metadata_not_local_audio(client):
    audio_bytes = _read_fixture_bytes()
    mock_response = MagicMock()
    mock_response.read.return_value = audio_bytes
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False

    headers = {"X-API-Key": main.config.API_KEY}

    with patch('main._assert_public_url'), \
         patch('main.urllib.request.urlopen', return_value=mock_response), \
         patch('main.advanced_deepfake_detector.predict', return_value={'is_ai': False, 'confidence': 0.0, 'reason': 'Real Human Voice (0.0%)', 'status': 'success'}), \
         patch('gender_verifier.classify_gender', return_value={'label': 'female', 'confidence': 60.0}), \
         patch('fingerprint.compute_fingerprint', return_value=b'\x44' * 16), \
         patch('fingerprint_store.find_cross_advisor_match', return_value=None), \
         patch('fingerprint_store.store_fingerprint'), \
         patch('main._keep_for_manual_review') as mock_keep_local, \
         patch('manual_review_store.add_pending_review') as mock_add_review:

        resp = client.post(
            "/predict-url",
            headers=headers,
            json={"url": "http://test.local/clip-ambiguous.wav", "userId": "advisor-G", "fullname": "Advisor G"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data['decision'] == 'uncertain'
    mock_keep_local.assert_not_called()
    mock_add_review.assert_called_once()
    call_args = mock_add_review.call_args[0]
    assert call_args[0] == 'advisor-G'
    assert call_args[2] == 'http://test.local/clip-ambiguous.wav'


def test_predict_manual_review_still_uses_local_retention_unaffected(client):
    """Hard boundary check: /predict (no external URL to fall back on)
    must keep using local retention for manual_review — this plan only
    changes /predict-url. Forces a manual_review via a low-confidence
    primary verdict (same mechanism as the existing
    test_predict_gender_manual_review_at_84_9_percent_confidence test,
    but exercised through the full /predict HTTP endpoint here instead of
    calling predict_gender() directly, since that's what actually proves
    _keep_for_manual_review gets called)."""
    with patch('gender_verifier.classify_gender', return_value={'label': 'female', 'confidence': 60.0}), \
         patch('main.advanced_deepfake_detector.predict', return_value={'is_ai': False, 'confidence': 0.0, 'reason': 'Real Human Voice (0.0%)', 'status': 'success'}), \
         patch('main._keep_for_manual_review', wraps=main._keep_for_manual_review) as mock_keep_local:
        with open(FIXTURE, "rb") as f:
            resp = client.post(
                "/predict",
                files={"file": ("female_test.wav", f, "audio/wav")},
                data={"advisor_name": "Test Advisor"},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert data['ensemble']['label'] == 'manual_review'
    mock_keep_local.assert_called_once()


def test_pending_url_reviews_endpoint_requires_auth(client):
    resp = client.get("/api/admin/pending-url-reviews")
    assert resp.status_code == 401


def test_pending_url_reviews_endpoint_returns_list(client):
    headers = {"X-API-Key": main.config.API_KEY}
    with patch('manual_review_store.list_pending_reviews', return_value=[
        {'id': 1, 'advisor_id': 'advisor-1', 'advisor_name': 'Alice', 'source_url': 'https://s3.example.com/a.wav', 'reason': 'test', 'created_at': '2026-09-04T00:00:00'}
    ]):
        resp = client.get("/api/admin/pending-url-reviews", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data['total'] == 1
    assert data['reviews'][0]['advisor_id'] == 'advisor-1'
```

**Note on the ambiguous-gender test's confidence value (60.0%):** this must land in the pitch-safety-filter's manual_review band for the fixture's real extracted pitch features, combined with the mocked 60% primary confidence (well below the 85% cutoff) — this is what actually drives it to `manual_review` regardless of the exact pitch values, since the confidence-based branch of `apply_pitch_safety_filter` triggers manual_review whenever confidence `< 0.85`, independent of pitch.

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_main_integration.py -v -k "manual_review_stores_metadata or pending_url_reviews or predict_manual_review_still_uses_local_retention"`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError` referencing `manual_review_store`, and/or `main._keep_for_manual_review` still being called (assertion failures) or not yet defined as importable in the test's `wraps=` usage, and/or `404 Not Found` for the not-yet-existing `/api/admin/pending-url-reviews` route. The `/predict`-unaffected test may already pass even before your edits, since it only asserts existing behavior — that's fine, it's a regression guard, not something this task needs to make pass.

- [ ] **Step 3: Add the import**

In `main.py`, find:

```python
import fingerprint
import fingerprint_store
```

Replace with:

```python
import fingerprint
import fingerprint_store
import manual_review_store
```

- [ ] **Step 4: Initialize the store at startup**

In `main.py`, find:

```python
fingerprint_store.init_db()
```

Replace with:

```python
fingerprint_store.init_db()
manual_review_store.init_db()
```

- [ ] **Step 5: Add the admin endpoint**

In `main.py`, find the existing `/recordings` endpoint:

```python
@app.get("/recordings", dependencies=[Depends(require_api_key)])
async def list_recordings():
    """Admin endpoint: lists pending manual_review audio (the only audio this app
    retains — everything else is deleted right after processing)."""
    files = []
    for fname in sorted(os.listdir(MANUAL_REVIEW_DIR), reverse=True):
        if fname.endswith(".npy"):
            continue
        fpath = os.path.join(MANUAL_REVIEW_DIR, fname)
        if os.path.isfile(fpath):
            stat = os.stat(fpath)
            files.append({
                'filename': fname,
                'size_kb': round(stat.st_size / 1024, 1),
                'saved_at': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            })
    return JSONResponse(content={'total': len(files), 'recordings': files})
```

Leave this exactly as-is (it's still needed for `/predict`), and add a new endpoint right after it:

```python
@app.get("/api/admin/pending-url-reviews", dependencies=[Depends(require_api_key)])
async def list_pending_url_reviews():
    """Admin endpoint: lists pending manual_review entries from /predict-url.
    Unlike /recordings (which lists locally-stored /predict audio), these
    entries carry only the caller's original source_url — /predict-url
    never retains a local copy of the audio."""
    reviews = manual_review_store.list_pending_reviews()
    return JSONResponse(content={'total': len(reviews), 'reviews': reviews})
```

- [ ] **Step 6: Replace the first `_keep_for_manual_review` call site (replay-attack branch)**

In `main.py`, find:

```python
        if duplicate is not None:
            reason_str = f"Replay Attack Detected: this audio was previously submitted under a different Advisor ID ({duplicate['advisor_id']})."
            logger.info(f"[MANUAL REVIEW] {reason_str} Current Advisor ID: {advisor_id}. Hamming distance: {duplicate.get('hamming_distance')}.")
            kept_path = _keep_for_manual_review(tmp_path)  # tmp_path itself no longer exists after this

            result = {
                'svm':      {'label': 'manual_review', 'confidence': 0.0},
                'gbm':      {'label': 'manual_review', 'confidence': 0.0},
                'rf':       {'label': 'manual_review', 'confidence': 0.0},
                'ensemble': {'label': 'manual_review', 'confidence': 0.0, 'male_votes': 0, 'total_votes': 3},
                'ai':       {'is_ai': False, 'confidence': 0.0, 'reason': reason_str, 'status': 'success'},
                'status': 'manual_review',
                'request_id': request_id,
                'advisor_id': advisor_id,
                'advisor_name': advisor_name,
                'reason': reason_str,
                'source_url': audio_url,
            }
            _dispatch_email_notification(result, f"{advisor_name} (ID:{advisor_id})", file_size_kb, kept_path)
```

Replace with:

```python
        if duplicate is not None:
            reason_str = f"Replay Attack Detected: this audio was previously submitted under a different Advisor ID ({duplicate['advisor_id']})."
            logger.info(f"[MANUAL REVIEW] {reason_str} Current Advisor ID: {advisor_id}. Hamming distance: {duplicate.get('hamming_distance')}.")
            manual_review_store.add_pending_review(advisor_id, advisor_name, audio_url, reason_str)
            _delete_audio(tmp_path)

            result = {
                'svm':      {'label': 'manual_review', 'confidence': 0.0},
                'gbm':      {'label': 'manual_review', 'confidence': 0.0},
                'rf':       {'label': 'manual_review', 'confidence': 0.0},
                'ensemble': {'label': 'manual_review', 'confidence': 0.0, 'male_votes': 0, 'total_votes': 3},
                'ai':       {'is_ai': False, 'confidence': 0.0, 'reason': reason_str, 'status': 'success'},
                'status': 'manual_review',
                'request_id': request_id,
                'advisor_id': advisor_id,
                'advisor_name': advisor_name,
                'reason': reason_str,
                'source_url': audio_url,
            }
            _dispatch_email_notification(result, f"{advisor_name} (ID:{advisor_id})", file_size_kb, None)
```

- [ ] **Step 7: Replace the second `_keep_for_manual_review` call site (deepfake-flagged branch)**

In `main.py`, find:

```python
            logger.info(f"[MANUAL REVIEW] Deepfake model flagged audio for Advisor ID: {advisor_id}, escalating instead of auto-rejecting. Reason: {reason_str}")
            kept_path = _keep_for_manual_review(tmp_path)  # tmp_path itself no longer exists after this

            result['status'] = 'manual_review'
            result['request_id'] = request_id
            result['advisor_id'] = advisor_id
            result['advisor_name'] = advisor_name
            result['reason'] = reason_str
            result['source_url'] = audio_url
            _dispatch_email_notification(result, f"{advisor_name} (ID:{advisor_id})", file_size_kb, kept_path)
```

Replace with:

```python
            logger.info(f"[MANUAL REVIEW] Deepfake model flagged audio for Advisor ID: {advisor_id}, escalating instead of auto-rejecting. Reason: {reason_str}")
            manual_review_store.add_pending_review(advisor_id, advisor_name, audio_url, reason_str)
            _delete_audio(tmp_path)

            result['status'] = 'manual_review'
            result['request_id'] = request_id
            result['advisor_id'] = advisor_id
            result['advisor_name'] = advisor_name
            result['reason'] = reason_str
            result['source_url'] = audio_url
            _dispatch_email_notification(result, f"{advisor_name} (ID:{advisor_id})", file_size_kb, None)
```

- [ ] **Step 8: Replace the third `_keep_for_manual_review` call site (final ambiguous-gender branch)**

In `main.py`, find:

```python
        # ── 7. Email notification (background) + audio retention ───────────
        if result['status'] == 'manual_review':
            kept_path = _keep_for_manual_review(tmp_path)  # tmp_path itself no longer exists after this
            _dispatch_email_notification(result, display_name, file_size_kb, kept_path)
        else:
            # Auto-decided (clean accept) — no audio is retained; finally: below deletes tmp_path.
            _dispatch_email_notification(result, display_name, file_size_kb, tmp_path)
```

Replace with:

```python
        # ── 7. Email notification (background) + audio retention ───────────
        if result['status'] == 'manual_review':
            manual_review_store.add_pending_review(advisor_id, advisor_name, audio_url, result.get('reason', 'Ambiguous voice'))
            _delete_audio(tmp_path)
            _dispatch_email_notification(result, display_name, file_size_kb, None)
        else:
            # Auto-decided (clean accept) — no audio is retained; finally: below deletes tmp_path.
            _dispatch_email_notification(result, display_name, file_size_kb, tmp_path)
```

Note: `result['reason']` may not always be set at this point for the ambiguous-gender-verdict path specifically (unlike the other two branches, which set it explicitly beforehand) — the `.get('reason', 'Ambiguous voice')` fallback handles that; if you find `result['reason']` IS already reliably set here when you read the surrounding code, using it directly instead of `.get(...)` is fine too, use your judgment based on what's actually in scope at this exact point.

- [ ] **Step 9: Run the new tests to verify they pass**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_main_integration.py -v -k "manual_review_stores_metadata or pending_url_reviews or predict_manual_review_still_uses_local_retention"`
Expected: 6 passed.

- [ ] **Step 10: Run the full test suite**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -v`
Expected: all pre-existing tests plus Task 1's 3 and this task's 5 all pass — confirm the actual total from the suite's own summary line, and specifically confirm no `/predict`-focused test (e.g. anything exercising `_keep_for_manual_review` or `MANUAL_REVIEW_DIR` for the `/predict` endpoint, if any such test exists) changed behavior.

- [ ] **Step 11: Manually verify against the live server**

If the local dev server is running on port 8003, restart it to pick up this code (it does not hot-reload), then:

```bash
curl -s -w "\nHTTP:%{http_code}\n" http://127.0.0.1:8003/api/admin/pending-url-reviews
```
Expected: `401` (no API key). Then repeat with the `X-API-Key` header from `.env` and confirm a `200` with `{"total": 0, "reviews": []}` (or existing entries, on a server that already has some) — not a `500` or `404`.

- [ ] **Step 12: Commit**

```bash
git add main.py tests/test_main_integration.py
git commit -m "Stop local audio retention on /predict-url manual_review, add admin queue endpoint"
```
