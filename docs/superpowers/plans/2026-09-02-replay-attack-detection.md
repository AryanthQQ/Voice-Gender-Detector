# Replay-Attack Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect when a `/predict-url` submission's audio clip (or a minor re-encoded/trimmed variant of it) was already submitted under a *different* `advisor_id`, and escalate that submission to `manual_review` instead of letting the normal pipeline decide it — skipping the expensive deepfake and gender models entirely when a duplicate is found.

**Architecture:** A pure perceptual-audio-fingerprinting module (`fingerprint.py`) produces a fixed-length binary fingerprint per audio clip, compared via Hamming distance. A SQLite-backed store (`fingerprint_store.py`) persists fingerprints (never raw audio) tagged by `advisor_id`. `main.py`'s `_predict_url_sync` is wired to check for a cross-advisor match right after the existing audio-quality/STT gates and before the deepfake/gender models run.

**Tech Stack:** `librosa`, `numpy` (already dependencies), stdlib `sqlite3`, pytest, `unittest.mock`.

## Global Constraints

- Only `/predict-url` is affected. `/predict` (manual upload, no `advisor_id`) is untouched.
- A match against the **same** `advisor_id` is never escalated — only a match against a **different** `advisor_id` counts as a replay attack.
- On a cross-advisor match: skip the deepfake-detector and gender-model calls entirely, escalate straight to `manual_review` (`decision: 'uncertain'`), keep the audio via `_keep_for_manual_review`, and send the existing email notification path.
- Only fingerprints are stored persistently — never raw audio.
- No new third-party dependency — use only `librosa`, `numpy`, and stdlib `sqlite3`.
- The exact fingerprint size (`NUM_SEGMENTS`, `NUM_MEL_BANDS`) and match threshold (`MATCH_THRESHOLD`) are starting points, not locked values — adjust `MATCH_THRESHOLD` if the true-positive/true-negative tests in Task 1 don't pass with the given defaults, and document why in a code comment if you change it.
- Use the project's existing interpreter for every command: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe`.

---

### Task 1: Perceptual audio fingerprinting module

**Files:**
- Create: `fingerprint.py`
- Create: `tests/test_fingerprint.py`

**Interfaces:**
- Produces: `compute_fingerprint(audio_path: str) -> bytes`, `hamming_distance(fp1: bytes, fp2: bytes) -> int` (raises `ValueError` on length mismatch), `is_match(fp1: bytes, fp2: bytes) -> bool`. Task 2 and Task 3 both import and call these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fingerprint.py`:

```python
import os
import pytest
import soundfile as sf
import librosa

from fingerprint import compute_fingerprint, hamming_distance, is_match

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..')
FEMALE_FIXTURE = os.path.join(FIXTURE_DIR, 'female_test.wav')
HUMAN_FIXTURE = os.path.join(FIXTURE_DIR, 'test_human.wav')


def test_identical_audio_produces_identical_fingerprint():
    fp1 = compute_fingerprint(FEMALE_FIXTURE)
    fp2 = compute_fingerprint(FEMALE_FIXTURE)
    assert fp1 == fp2
    assert hamming_distance(fp1, fp2) == 0


def test_trimmed_copy_still_matches(tmp_path):
    y, sr = librosa.load(FEMALE_FIXTURE, sr=16000, mono=True)
    trimmed = y[int(0.5 * sr):-int(0.5 * sr)]  # drop 0.5s off each end
    trimmed_path = os.path.join(str(tmp_path), 'trimmed.wav')
    sf.write(trimmed_path, trimmed, sr)

    original_fp = compute_fingerprint(FEMALE_FIXTURE)
    trimmed_fp = compute_fingerprint(trimmed_path)

    assert is_match(original_fp, trimmed_fp)


def test_resampled_copy_still_matches(tmp_path):
    y, sr = librosa.load(FEMALE_FIXTURE, sr=16000, mono=True)
    resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)
    resampled_path = os.path.join(str(tmp_path), 'resampled.wav')
    sf.write(resampled_path, resampled, 22050)

    original_fp = compute_fingerprint(FEMALE_FIXTURE)
    resampled_fp = compute_fingerprint(resampled_path)

    assert is_match(original_fp, resampled_fp)


def test_different_recordings_do_not_match():
    fp1 = compute_fingerprint(FEMALE_FIXTURE)
    fp2 = compute_fingerprint(HUMAN_FIXTURE)
    assert not is_match(fp1, fp2)


def test_hamming_distance_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        hamming_distance(b'\x00', b'\x00\x00')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_fingerprint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fingerprint'`

- [ ] **Step 3: Write the implementation**

Create `fingerprint.py`:

```python
"""
fingerprint.py — Perceptual audio fingerprinting for replay-attack detection.

Produces a fixed-length binary fingerprint from an audio file's mel-spectrogram,
robust to minor re-encoding, resampling, or trimming (unlike a byte-level hash
of the file, which any of those would defeat). Two fingerprints of the same
underlying audio should differ by only a few bits; two genuinely different
recordings should differ by many more.
"""
import librosa
import numpy as np

NUM_SEGMENTS = 8
NUM_MEL_BANDS = 32
MATCH_THRESHOLD = 20  # out of NUM_SEGMENTS * NUM_MEL_BANDS = 256 bits total


def compute_fingerprint(audio_path: str) -> bytes:
    """Computes a perceptual audio fingerprint for audio_path.
    Returns a fixed-length bytes object (NUM_SEGMENTS * NUM_MEL_BANDS bits, packed)."""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=NUM_MEL_BANDS)
    log_mel = librosa.power_to_db(mel)

    num_frames = log_mel.shape[1]
    bounds = np.linspace(0, num_frames, NUM_SEGMENTS + 1, dtype=int)
    segment_means = np.array([
        log_mel[:, bounds[i]:bounds[i + 1]].mean(axis=1) if bounds[i + 1] > bounds[i]
        else np.zeros(NUM_MEL_BANDS)
        for i in range(NUM_SEGMENTS)
    ])  # shape: (NUM_SEGMENTS, NUM_MEL_BANDS)

    band_medians = np.median(segment_means, axis=0)
    bits = (segment_means > band_medians).astype(np.uint8).flatten()

    return np.packbits(bits).tobytes()


def hamming_distance(fp1: bytes, fp2: bytes) -> int:
    """Number of differing bits between two same-length fingerprints."""
    if len(fp1) != len(fp2):
        raise ValueError("Fingerprints must be the same length to compare.")
    xor_bytes = bytes(a ^ b for a, b in zip(fp1, fp2))
    return sum(bin(byte).count('1') for byte in xor_bytes)


def is_match(fp1: bytes, fp2: bytes) -> bool:
    """True if two fingerprints are within MATCH_THRESHOLD Hamming distance."""
    return hamming_distance(fp1, fp2) <= MATCH_THRESHOLD
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_fingerprint.py -v`
Expected: 5 passed.

**If `test_trimmed_copy_still_matches` or `test_resampled_copy_still_matches` fail** (the Hamming distance for the modified copy exceeds `MATCH_THRESHOLD`): the fingerprint is behaving correctly (the fixture just needed a slightly higher tolerance) — raise `MATCH_THRESHOLD` in small increments (e.g. to 24, 28, ...) and re-run, until both pass, WITHOUT letting `test_different_recordings_do_not_match` start failing. If you can't satisfy both within `MATCH_THRESHOLD <= 40` (roughly 15% of 256 bits), that signals the segment/band counts need adjusting instead (try `NUM_SEGMENTS = 4`) — this is expected empirical tuning for a new signal-processing algorithm, not a sign you did something wrong. Document whatever final values you land on with a one-line comment explaining why (e.g. "raised to 24 — the 0.5s-trim test needed it, no false positive against the different-recording fixture at this level").

- [ ] **Step 5: Commit**

```bash
git add fingerprint.py tests/test_fingerprint.py
git commit -m "Add perceptual audio fingerprinting module for replay-attack detection"
```

---

### Task 2: SQLite-backed fingerprint store

**Files:**
- Create: `fingerprint_store.py`
- Create: `tests/test_fingerprint_store.py`

**Interfaces:**
- Consumes: `fingerprint.is_match` (Task 1).
- Produces: `init_db(db_path: str = DB_PATH) -> None`, `find_cross_advisor_match(fingerprint_bytes: bytes, advisor_id: str, db_path: str = DB_PATH) -> dict | None`, `store_fingerprint(fingerprint_bytes: bytes, advisor_id: str, advisor_name: str, decision: str, db_path: str = DB_PATH) -> None`. Task 3 calls all three by these exact names, using the default `db_path` (no `db_path` argument) in production code — the parameter exists only so tests can point at a temp file.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fingerprint_store.py`:

```python
import os
from fingerprint_store import init_db, find_cross_advisor_match, store_fingerprint


def test_no_match_on_empty_db(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    result = find_cross_advisor_match(b'\x00' * 32, 'advisor-1', db_path)
    assert result is None


def test_finds_match_from_different_advisor(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    fp = bytes([0b10101010] * 32)
    store_fingerprint(fp, 'advisor-1', 'Alice', 'female', db_path)

    match = find_cross_advisor_match(fp, 'advisor-2', db_path)
    assert match is not None
    assert match['advisor_id'] == 'advisor-1'
    assert match['advisor_name'] == 'Alice'


def test_ignores_match_from_same_advisor(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    fp = bytes([0b10101010] * 32)
    store_fingerprint(fp, 'advisor-1', 'Alice', 'female', db_path)

    match = find_cross_advisor_match(fp, 'advisor-1', db_path)
    assert match is None


def test_near_duplicate_within_threshold_still_matches(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    fp = bytes([0b10101010] * 32)
    near_duplicate = bytes([0b10101011] * 32)  # 1 bit different per byte, well within MATCH_THRESHOLD
    store_fingerprint(fp, 'advisor-1', 'Alice', 'female', db_path)

    match = find_cross_advisor_match(near_duplicate, 'advisor-2', db_path)
    assert match is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_fingerprint_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fingerprint_store'`

- [ ] **Step 3: Write the implementation**

Create `fingerprint_store.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_fingerprint_store.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add fingerprint_store.py tests/test_fingerprint_store.py
git commit -m "Add SQLite-backed fingerprint store for replay-attack detection"
```

---

### Task 3: Wire replay-attack detection into /predict-url

**Files:**
- Modify: `main.py` (see exact line-anchored edits below; line numbers are from the pre-Task-3 file state)
- Modify: `tests/test_main_integration.py`

**Interfaces:**
- Consumes: `fingerprint.compute_fingerprint` (Task 1), `fingerprint_store.init_db` / `find_cross_advisor_match` / `store_fingerprint` (Task 2).
- Produces: nothing new consumed by a later task — this is the last task.

- [ ] **Step 1: Write the failing integration tests**

Read `tests/test_main_integration.py` first to see its existing imports and fixtures (it already has `import main`, `import os`, `import pytest`, `from fastapi.testclient import TestClient`, a session-scoped `client` fixture, and `FIXTURE = os.path.join(os.path.dirname(__file__), '..', 'female_test.wav')` from earlier work — reuse `FIXTURE` and `client`, don't redefine them).

Add these to the file (keep everything already in it untouched):

```python
from unittest.mock import patch, MagicMock


def _read_fixture_bytes():
    with open(FIXTURE, 'rb') as f:
        return f.read()


def test_predict_url_escalates_on_cross_advisor_duplicate(client):
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
         patch('fingerprint.compute_fingerprint', return_value=b'\x00' * 32), \
         patch('fingerprint_store.find_cross_advisor_match', return_value={'advisor_id': 'advisor-A', 'advisor_name': 'Advisor A'}), \
         patch('fingerprint_store.store_fingerprint') as mock_store:

        resp = client.post(
            "/predict-url",
            headers=headers,
            json={"url": "http://test.local/clip-duplicate.wav", "userId": "advisor-B", "fullname": "Advisor B"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data['decision'] == 'uncertain'
    assert 'Replay Attack' in data['reason']
    mock_deepfake.assert_not_called()
    mock_classify.assert_not_called()
    mock_store.assert_not_called()


def test_predict_url_no_duplicate_proceeds_normally_and_stores_fingerprint(client):
    audio_bytes = _read_fixture_bytes()
    mock_response = MagicMock()
    mock_response.read.return_value = audio_bytes
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False

    headers = {"X-API-Key": main.config.API_KEY}

    with patch('main._assert_public_url'), \
         patch('main.urllib.request.urlopen', return_value=mock_response), \
         patch('main.advanced_deepfake_detector.predict', return_value={'is_ai': False, 'confidence': 0.0, 'reason': 'Real Human Voice', 'status': 'success'}), \
         patch('gender_verifier.classify_gender', return_value={'label': 'male', 'confidence': 99.0}), \
         patch('fingerprint.compute_fingerprint', return_value=b'\x11' * 32), \
         patch('fingerprint_store.find_cross_advisor_match', return_value=None), \
         patch('fingerprint_store.store_fingerprint') as mock_store:

        resp = client.post(
            "/predict-url",
            headers=headers,
            json={"url": "http://test.local/clip-new.wav", "userId": "advisor-C", "fullname": "Advisor C"},
        )

    assert resp.status_code == 200
    mock_store.assert_called_once()
    call_args = mock_store.call_args[0]
    assert call_args[1] == 'advisor-C'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_main_integration.py -v -k replay_or_duplicate_or_predict_url`

Actually run the two new tests by name:
Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest "tests/test_main_integration.py::test_predict_url_escalates_on_cross_advisor_duplicate" "tests/test_main_integration.py::test_predict_url_no_duplicate_proceeds_normally_and_stores_fingerprint" -v`
Expected: FAIL — `AttributeError` or similar, since `main.py` doesn't import `fingerprint`/`fingerprint_store` yet and never calls them, so `data['reason']` won't contain `'Replay Attack'` and `mock_deepfake`/`mock_classify` WILL have been called (the assertions that they weren't will fail).

- [ ] **Step 3: Add imports and startup init to main.py**

Find this line near the top of `main.py`:

```python
from deepfake_detector_v2 import AdvancedDeepfakeDetector
import gender_guesser.detector as gender
```

Add two new imports after it:

```python
from deepfake_detector_v2 import AdvancedDeepfakeDetector
import gender_guesser.detector as gender
import fingerprint
import fingerprint_store
```

Then find this line (near the other startup init calls, e.g. right after `advanced_deepfake_detector = AdvancedDeepfakeDetector()` or the primary gender model's `gender_verifier.load_model()` init block):

```python
advanced_deepfake_detector = AdvancedDeepfakeDetector()
```

Add the fingerprint DB init right after it:

```python
advanced_deepfake_detector = AdvancedDeepfakeDetector()

fingerprint_store.init_db()
```

- [ ] **Step 4: Insert the replay-attack check in `_predict_url_sync`**

In `main.py`, find this block (the STT check followed by the "Extract features + predict" comment):

```python
        except Exception as e:
            logger.exception(f"[WARN] STT Transcription failed: {e}")

        # ── 4. Extract features + predict ─────────────────────────────────────
        with GLOBAL_PROCESS_LOCK:
            t_df_start = time.time()
            ai_result = advanced_deepfake_detector.predict(tmp_path)
```

Replace it with (inserting the new "3b" block between the STT except and the "4. Extract features" comment):

```python
        except Exception as e:
            logger.exception(f"[WARN] STT Transcription failed: {e}")

        # ── 3b. Replay-attack check: same audio previously submitted under a
        # different advisor_id? Skip the expensive deepfake + gender model
        # calls entirely if so — the decision is already made.
        try:
            with GLOBAL_PROCESS_LOCK:
                current_fp = fingerprint.compute_fingerprint(tmp_path)
            duplicate = fingerprint_store.find_cross_advisor_match(current_fp, advisor_id)
        except Exception as e:
            logger.exception(f"[WARN] Fingerprint check failed for Advisor ID: {advisor_id}: {e}")
            current_fp = None
            duplicate = None

        if duplicate is not None:
            reason_str = f"Replay Attack Detected: this audio was previously submitted under a different Advisor ID ({duplicate['advisor_id']})."
            logger.info(f"[MANUAL REVIEW] {reason_str} Current Advisor ID: {advisor_id}.")
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

            n8n_result = {
                'decision': 'uncertain',
                'status': 1,
                'accepted': False,
                'advisor_id': advisor_id,
                'advisor_name': advisor_name,
                'source_url': audio_url,
                'is_female': False,
                'reason': reason_str
            }

            _add_to_cache(audio_url, n8n_result)
            return log_req(JSONResponse(content=n8n_result))

        # ── 4. Extract features + predict ─────────────────────────────────────
        with GLOBAL_PROCESS_LOCK:
            t_df_start = time.time()
            ai_result = advanced_deepfake_detector.predict(tmp_path)
```

- [ ] **Step 5: Store the fingerprint at each of the 3 later exit points**

There are exactly three places later in `_predict_url_sync` that call `_add_to_cache(audio_url, n8n_result)` after this point in the function (the deepfake-flagged manual_review branch, the male-reject branch, and the final female/manual_review branch). At each of these three, add a `fingerprint_store.store_fingerprint(...)` call immediately before the existing `_add_to_cache(audio_url, n8n_result)` line, guarded by `if current_fp is not None:` (it can be `None` if the fingerprint computation itself raised an exception in Step 4's try/except).

**3a. Deepfake-flagged branch.** Find:

```python
            _dispatch_email_notification(result, f"{advisor_name} (ID:{advisor_id})", file_size_kb, kept_path)

            n8n_result = {
                'decision': 'uncertain',
                'status': 1,
                'accepted': False,
                'advisor_id': advisor_id,
                'advisor_name': advisor_name,
                'source_url': audio_url,
                'is_female': False,
                'reason': reason_str
            }

            _add_to_cache(audio_url, n8n_result)
            return log_req(JSONResponse(content=n8n_result))

        if result['ensemble']['label'] == 'female':
```

(the `if result['ensemble']['label'] == 'female':` line at the end no longer exists post-primary-gender-model-upgrade — if you don't find it, just match on the `_dispatch_email_notification` / `n8n_result` / `_add_to_cache` block shown above, which is unique to the deepfake-flagged branch). Replace with:

```python
            _dispatch_email_notification(result, f"{advisor_name} (ID:{advisor_id})", file_size_kb, kept_path)

            n8n_result = {
                'decision': 'uncertain',
                'status': 1,
                'accepted': False,
                'advisor_id': advisor_id,
                'advisor_name': advisor_name,
                'source_url': audio_url,
                'is_female': False,
                'reason': reason_str
            }

            if current_fp is not None:
                fingerprint_store.store_fingerprint(current_fp, advisor_id, advisor_name, result.get('status', 'manual_review'))
            _add_to_cache(audio_url, n8n_result)
            return log_req(JSONResponse(content=n8n_result))
```

**3b. Male-reject branch.** Find:

```python
        # ── 5. REJECT male voice — no Email, no further action ─────────────
        if label == 'male':
            logger.info(f"[REJECT] Male voice detected for Advisor ID: {advisor_id} - rejected, no Email sent.")
            n8n_result = {
                'decision':     'reject',
                'status':       6,
                'accepted':     False,
                'advisor_id':   advisor_id,
                'advisor_name': advisor_name,
                'source_url':   audio_url,
                'is_female':    False,
                'reason':       'Male voice detected but name is female. Rejected for fake identity.' if gender_mismatch else 'Male voice detected. Only female voices are accepted.'
            }
            _add_to_cache(audio_url, n8n_result)
            return log_req(JSONResponse(content=n8n_result))
```

Replace with:

```python
        # ── 5. REJECT male voice — no Email, no further action ─────────────
        if label == 'male':
            logger.info(f"[REJECT] Male voice detected for Advisor ID: {advisor_id} - rejected, no Email sent.")
            n8n_result = {
                'decision':     'reject',
                'status':       6,
                'accepted':     False,
                'advisor_id':   advisor_id,
                'advisor_name': advisor_name,
                'source_url':   audio_url,
                'is_female':    False,
                'reason':       'Male voice detected but name is female. Rejected for fake identity.' if gender_mismatch else 'Male voice detected. Only female voices are accepted.'
            }
            if current_fp is not None:
                fingerprint_store.store_fingerprint(current_fp, advisor_id, advisor_name, 'reject')
            _add_to_cache(audio_url, n8n_result)
            return log_req(JSONResponse(content=n8n_result))
```

**3c. Final female/manual_review branch.** Find:

```python
        n8n_result = {
            'decision': result.get('decision', 'reject'),
            'status': 3 if result.get('decision') == 'accept' else (1 if result.get('decision') == 'uncertain' else 6),
            'accepted': result.get('accepted', False),
            'advisor_id': advisor_id,
            'advisor_name': advisor_name,
            'source_url': audio_url,
            'is_female': result.get('is_female', False),
            'reason': 'Voice processed successfully.'
        }

        _add_to_cache(audio_url, n8n_result)

        return log_req(JSONResponse(content=n8n_result))
```

Replace with:

```python
        n8n_result = {
            'decision': result.get('decision', 'reject'),
            'status': 3 if result.get('decision') == 'accept' else (1 if result.get('decision') == 'uncertain' else 6),
            'accepted': result.get('accepted', False),
            'advisor_id': advisor_id,
            'advisor_name': advisor_name,
            'source_url': audio_url,
            'is_female': result.get('is_female', False),
            'reason': 'Voice processed successfully.'
        }

        if current_fp is not None:
            fingerprint_store.store_fingerprint(current_fp, advisor_id, advisor_name, result.get('decision', 'unknown'))
        _add_to_cache(audio_url, n8n_result)

        return log_req(JSONResponse(content=n8n_result))
```

- [ ] **Step 6: Run the new integration tests to verify they pass**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest "tests/test_main_integration.py::test_predict_url_escalates_on_cross_advisor_duplicate" "tests/test_main_integration.py::test_predict_url_no_duplicate_proceeds_normally_and_stores_fingerprint" -v`
Expected: 2 passed.

- [ ] **Step 7: Run the full test suite**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -v`
Expected: 21 passed (the 14 pre-existing plus 5 from Task 1, 4 from Task 2 — wait, that's 14+5+4=23; recount: this run includes Task 1's 5 + Task 2's 4 + Task 3's 2 new ones = 11 new, plus the 14 already passing = 25 total). Confirm the actual total matches "14 (pre-existing) + 5 (Task 1) + 4 (Task 2) + 2 (Task 3) = 25 passed" and investigate any mismatch rather than assuming it's fine.

- [ ] **Step 8: Commit**

```bash
git add main.py tests/test_main_integration.py
git commit -m "Wire replay-attack detection into /predict-url"
```
