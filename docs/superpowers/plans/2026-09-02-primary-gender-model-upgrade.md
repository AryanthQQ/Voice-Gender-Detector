# Primary Gender Model Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing pretrained Wav2Vec2-XLSR gender model (`gender_verifier.py`) the primary decision-maker for `/predict` and `/predict-url`, retiring the classical SVM/GBM/RF ensemble entirely, while keeping the API request/response shape byte-for-byte backward compatible.

**Architecture:** A new pure function (`pitch_safety_filter.py`) encodes the existing pitch-based hard-safety override, tested in isolation with no model/audio dependency. `gender_verifier.py` is generalized from a female-only secondary check into an eager-loaded primary classifier. `main.py` is rewired to call the primary classifier + safety filter instead of the classical ensemble, and the now-redundant secondary corroboration step is deleted.

**Tech Stack:** FastAPI, PyTorch, transformers (Wav2Vec2), pytest, fastapi.testclient.TestClient.

## Global Constraints

- Request and response JSON shapes for `/predict` and `/predict-url` must not change: `ensemble`, `svm`, `gbm`, `rf`, `features`, and `decision` keys all still appear, with `svm`/`gbm`/`rf` mirroring `ensemble`'s `{label, confidence}`.
- The pitch safety filter must never turn a primary verdict into an auto-accept — it may only move a `female` verdict toward `male` (reject) or `manual_review`.
- The manual-review confidence cutoff stays at 85% (unchanged from the current ensemble) — no new calibration in this project.
- The primary gender model loads eagerly at server startup, not lazily on first request.
- No new training or data collection in this project — the classifier is used as-is, pretrained.
- Use the project's existing interpreter for every command below: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe`.

---

### Task 1: Extract the pitch safety filter into its own testable module

**Files:**
- Create: `pitch_safety_filter.py`
- Create: `tests/test_pitch_safety_filter.py`
- Modify: `requirements.txt` (add `pytest`)

**Interfaces:**
- Produces: `apply_pitch_safety_filter(label: str, confidence: float, meanfun_hz: float, meanfreq_hz: float) -> tuple[str, float]` — used by Task 3's rewritten `predict_gender` in `main.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pitch_safety_filter.py`:

```python
from pitch_safety_filter import apply_pitch_safety_filter


def test_male_verdict_passes_through_unchanged():
    label, conf = apply_pitch_safety_filter('male', 0.92, meanfun_hz=110.0, meanfreq_hz=115.0)
    assert label == 'male'
    assert conf == 0.92


def test_female_verdict_with_clear_female_pitch_and_high_confidence_passes():
    label, conf = apply_pitch_safety_filter('female', 0.95, meanfun_hz=210.0, meanfreq_hz=200.0)
    assert label == 'female'
    assert conf == 0.95


def test_female_verdict_with_male_range_pitch_is_overridden_to_male():
    label, conf = apply_pitch_safety_filter('female', 0.90, meanfun_hz=120.0, meanfreq_hz=125.0)
    assert label == 'male'
    assert conf == 0.999


def test_female_verdict_with_borderline_pitch_escalates_to_manual_review():
    label, conf = apply_pitch_safety_filter('female', 0.90, meanfun_hz=150.0, meanfreq_hz=200.0)
    assert label == 'manual_review'
    assert conf == 0.90


def test_female_verdict_with_low_confidence_escalates_to_manual_review():
    label, conf = apply_pitch_safety_filter('female', 0.70, meanfun_hz=210.0, meanfreq_hz=200.0)
    assert label == 'manual_review'
    assert conf == 0.70


def test_female_verdict_with_falsetto_pattern_escalates_to_manual_review():
    label, conf = apply_pitch_safety_filter('female', 0.95, meanfun_hz=280.0, meanfreq_hz=220.0)
    assert label == 'manual_review'
    assert conf == 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_pitch_safety_filter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pitch_safety_filter'`

- [ ] **Step 3: Write the implementation**

Create `pitch_safety_filter.py`:

```python
"""
pitch_safety_filter.py — Pure pitch-based safety override for the primary
gender model. Only ever pushes a 'female' verdict toward 'male' (reject) or
'manual_review' — never produces an accept on its own. No audio or model
dependencies, so it's cheap to run and easy to test in isolation.
"""


def apply_pitch_safety_filter(label: str, confidence: float, meanfun_hz: float, meanfreq_hz: float) -> tuple:
    """
    label: 'male' or 'female' (primary model's verdict).
    confidence: 0.0-1.0.
    meanfun_hz, meanfreq_hz: acoustic pitch features in Hz.

    Returns (final_label, final_confidence). final_label is 'male',
    'female', or 'manual_review'.
    """
    if label != 'female':
        return label, confidence

    if meanfun_hz < 130.0 or meanfreq_hz < 130.0:
        return 'male', 0.999

    if meanfun_hz < 170.0 or meanfreq_hz < 160.0 or confidence < 0.85:
        return 'manual_review', confidence

    if meanfun_hz > 270.0 and meanfreq_hz < 230.0:
        return 'manual_review', confidence

    return label, confidence
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_pitch_safety_filter.py -v`
Expected: 6 passed

- [ ] **Step 5: Add pytest to requirements.txt**

In `requirements.txt`, the last line is currently `psutil>=5.9.0`. Append a new line after it:

```
psutil>=5.9.0
pytest>=8.0.0
```

- [ ] **Step 6: Commit**

```bash
git add pitch_safety_filter.py tests/test_pitch_safety_filter.py requirements.txt
git commit -m "Extract pitch safety filter into its own testable module"
```

---

### Task 2: Generalize gender_verifier.py into an eager-loadable primary classifier

**Files:**
- Modify: `gender_verifier.py` (full rewrite of the existing file)
- Create: `tests/test_gender_verifier.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `gender_verifier.load_model() -> None` (idempotent), `gender_verifier.is_loaded() -> bool`, `gender_verifier.classify_gender(audio_path: str) -> dict` with keys `{'label': 'male'|'female', 'confidence': float 0-100}`. Task 3's `main.py` changes call all three of these by their exact names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gender_verifier.py`:

```python
import os
import gender_verifier

FIXTURE = os.path.join(os.path.dirname(__file__), '..', 'female_test.wav')


def test_load_model_is_idempotent_and_sets_loaded_flag():
    gender_verifier.load_model()
    assert gender_verifier.is_loaded() is True
    gender_verifier.load_model()  # calling again must not error or reload
    assert gender_verifier.is_loaded() is True


def test_classify_gender_returns_valid_schema():
    gender_verifier.load_model()
    result = gender_verifier.classify_gender(FIXTURE)
    assert set(result.keys()) == {'label', 'confidence'}
    assert result['label'] in ('male', 'female')
    assert 0.0 <= result['confidence'] <= 100.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_gender_verifier.py -v`
Expected: FAIL — `AttributeError: module 'gender_verifier' has no attribute 'load_model'`

- [ ] **Step 3: Rewrite gender_verifier.py**

Replace the entire contents of `gender_verifier.py` with:

```python
"""
gender_verifier.py — Primary Wav2Vec2-based gender classifier.

Loaded eagerly at server startup (see main.py) since it now runs on every
/predict and /predict-url request as the primary decision-maker, rather
than lazily on a subset of requests as it did when it was only a secondary
corroboration check.
"""
import threading

MODEL_NAME = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"

_load_lock = threading.Lock()
_processor = None
_model = None
_device = None


def load_model():
    """Loads the model if not already loaded. Idempotent — safe to call more than once."""
    global _processor, _model, _device
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(_device)
        model.eval()
        _processor, _model = processor, model


def is_loaded() -> bool:
    return _model is not None


def classify_gender(audio_path: str) -> dict:
    """Runs the primary gender model on audio_path.
    Returns {'label': 'male'|'female', 'confidence': 0-100}."""
    import torch
    import librosa

    load_model()
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt").to(_device)
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax())
    return {
        'label': _model.config.id2label[idx],
        'confidence': round(float(probs[idx]) * 100, 1),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_gender_verifier.py -v`
Expected: 2 passed (first run downloads the model from Hugging Face if not already cached — allow a minute or two)

- [ ] **Step 5: Commit**

```bash
git add gender_verifier.py tests/test_gender_verifier.py
git commit -m "Generalize gender_verifier into an eager-loadable primary classifier"
```

---

### Task 3: Wire main.py to the new primary model and retire the classical ensemble

**Files:**
- Modify: `main.py` (see exact line-anchored edits below; line numbers are from the pre-Task-3 file state)
- Create: `tests/test_main_integration.py`

**Interfaces:**
- Consumes: `pitch_safety_filter.apply_pitch_safety_filter` (Task 1), `gender_verifier.load_model` / `is_loaded` / `classify_gender` (Task 2).
- Produces: nothing new consumed by later tasks — this is the last code task.

- [ ] **Step 1: Write the failing integration tests**

Create `tests/test_main_integration.py`:

```python
import os
import pytest
from fastapi.testclient import TestClient

import main

FIXTURE = os.path.join(os.path.dirname(__file__), '..', 'female_test.wav')


@pytest.fixture(scope="session")
def client():
    return TestClient(main.app)


def test_health_reports_primary_gender_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["models_loaded"] is True
    assert data["detailed_model_status"]["primary_gender_model"] is True
    assert "svm_ensemble" not in data["detailed_model_status"]


def test_predict_returns_backward_compatible_response_shape(client):
    with open(FIXTURE, "rb") as f:
        resp = client.post(
            "/predict",
            files={"file": ("female_test.wav", f, "audio/wav")},
            data={"advisor_name": "Test Advisor"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "ensemble" in data
    for key in ("svm", "gbm", "rf"):
        assert data[key]["label"] == data["ensemble"]["label"]
        assert data[key]["confidence"] == data["ensemble"]["confidence"]
    assert data["decision"] in ("accept", "reject", "uncertain")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_main_integration.py -v`
Expected: FAIL — `assert False is True` on `detailed_model_status["primary_gender_model"]` (key doesn't exist yet), since `main.py` still reports the old `svm_ensemble` status.

- [ ] **Step 3: Update the module docstring**

In `main.py`, replace lines 1-8:

```python
"""
Voice Gender Detection - FastAPI Backend
- Accepts audio files (WAV/MP3/OGG etc.)
- Extracts 20 acoustic features via librosa/soundfile
- Runs SVM + GBM + Random Forest ensemble prediction
- Auto-saves every recording to recordings/ folder
- Sends Email notification to admin with verification result
"""
```

with:

```python
"""
Voice Gender Detection - FastAPI Backend
- Accepts audio files (WAV/MP3/OGG etc.)
- Extracts acoustic features via librosa/soundfile (used for the pitch
  safety filter and the UI's frequency display)
- Runs a pretrained Wav2Vec2-XLSR model as the primary gender classifier
- Auto-saves every recording to recordings/ folder
- Sends Email notification to admin with verification result
"""
```

- [ ] **Step 4: Replace the classical model-loading block**

In `main.py`, replace lines 197-209:

```python
# ── Models ───────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

try:
    svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl'))
    gbm_model = joblib.load(os.path.join(MODELS_DIR, 'gbm_model.pkl'))
    rf_model  = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
    scaler    = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    FEATURES  = joblib.load(os.path.join(MODELS_DIR, 'features.pkl'))
    logger.info("[OK] All models loaded successfully!")
except Exception as e:
    logger.exception(f"[ERR] Error loading models: {e}")
    svm_model = gbm_model = rf_model = scaler = None
```

with:

```python
# ── Primary Gender Model ─────────────────────────────────────────────────────
import gender_verifier

try:
    gender_verifier.load_model()
    logger.info("[OK] Primary gender model (Wav2Vec2-XLSR) loaded successfully!")
except Exception as e:
    logger.exception(f"[ERR] Error loading primary gender model: {e}")
```

- [ ] **Step 5: Remove the now-unused `joblib` import**

In `main.py`, find this block (around line 108-110):

```python
import numpy as np
import joblib
import librosa
```

Remove the `import joblib` line so it reads:

```python
import numpy as np
import librosa
```

- [ ] **Step 6: Delete `_corroborate_female`**

In `main.py`, delete this entire function (around lines 586-607, including the section-header comment above it):

```python
# ── Secondary Gender Verification (corroborates 'female' verdicts only) ───────
def _corroborate_female(audio_path: str, result: dict) -> dict:
    """Re-checks a 'female' ensemble verdict with the secondary Wav2Vec2 model before
    auto-accept. Any disagreement is escalated to manual_review instead of auto-rejecting
    or auto-accepting, since a false accept (male passing as female) is the costly failure
    mode we're guarding against. Skipped entirely for male/manual_review verdicts to avoid
    the extra model cost on requests that are already not going to auto-accept."""
    try:
        from gender_verifier import verify_female
        secondary = verify_female(audio_path)
    except Exception as e:
        logger.exception(f"[SECONDARY-GENDER] Verifier unavailable, keeping primary ensemble decision: {e}")
        result['secondary_check'] = {'status': 'unavailable'}
        return result

    result['secondary_check'] = secondary
    if secondary['label'] == 'male':
        logger.info(f"[SECONDARY-GENDER] Disagreement with ensemble (secondary said male, {secondary['confidence']}%). Escalating to manual_review.")
        result['ensemble']['label'] = 'manual_review'
        result['decision'] = 'uncertain'
    return result

```

(Leave the blank line that follows it in place, so the file still has exactly one blank line before the `# ── Prediction ──` comment.)

- [ ] **Step 7: Replace `predict_gender`**

In `main.py`, replace the entire existing function (starting at the `# ── Prediction ──` comment, through the end of `predict_gender`):

```python
# ── Prediction ────────────────────────────────────────────────────────────────
def predict_gender(features: dict) -> dict:
    """Run all 3 models and return ensemble prediction."""
    feat_vec    = np.array([[features[f] for f in FEATURES]])
    feat_scaled = scaler.transform(feat_vec)

    svm_prob = svm_model.predict_proba(feat_vec)[0]
    svm_pred = int(np.argmax(svm_prob))

    gbm_prob = gbm_model.predict_proba(feat_scaled)[0]
    gbm_pred = int(np.argmax(gbm_prob))

    rf_prob  = rf_model.predict_proba(feat_scaled)[0]
    rf_pred  = int(np.argmax(rf_prob))

    models = [
        ('male' if svm_pred == 1 else 'female', float(max(svm_prob))),
        ('male' if gbm_pred == 1 else 'female', float(max(gbm_prob))),
        ('male' if rf_pred  == 1 else 'female', float(max(rf_prob)))
    ]
    best_model = max(models, key=lambda x: x[1])
    
    final_label = best_model[0]
    final_conf = best_model[1]

    male_votes = sum([svm_pred, gbm_pred, rf_pred])

    # ── PITCH (FREQUENCY) HARD FILTER & MANUAL REVIEW ───────────────────
    meanfun_hz = features['meanfun'] * 1000
    meanfreq_hz = features['meanfreq'] * 1000
    if final_label == 'female':
        if meanfun_hz < 130.0 or meanfreq_hz < 130.0:
            # Definitely Male range (below 130 Hz is clearly male)
            final_label = 'male'
            final_conf = 0.999
            logger.info(f"[PITCH FILTER] Override applied. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz (Male range).")
        elif meanfun_hz < 170.0 or meanfreq_hz < 160.0 or (final_conf * 100) < 85.0:
            # Ambiguous pitch, frequency or low confidence -> send to manager
            final_label = 'manual_review'
            logger.info(f"[MANUAL REVIEW] Ambiguous voice. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz, Conf: {final_conf*100:.1f}%.")
        elif meanfun_hz > 270.0 and meanfreq_hz < 230.0:
            # High pitch but low formants (Child or Male Falsetto) -> send to manager
            final_label = 'manual_review'
            logger.info(f"[MANUAL REVIEW] Falsetto/Child detected. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz, Conf: {final_conf*100:.1f}%.")


    return {
        'svm': {'label': 'male' if svm_pred == 1 else 'female', 'confidence': float(max(svm_prob)) * 100},
        'gbm': {'label': 'male' if gbm_pred == 1 else 'female', 'confidence': float(max(gbm_prob)) * 100},
        'rf':  {'label': 'male' if rf_pred  == 1 else 'female', 'confidence': float(max(rf_prob))  * 100},
        'ensemble': {
            'label':      final_label,
            'confidence': float(final_conf) * 100,
            'male_votes': male_votes,
            'total_votes': 3,
        },
        'decision': 'accept' if final_label == 'female' else ('uncertain' if final_label == 'manual_review' else 'reject'),
        'features': {
            'meanfun_hz':  round(features['meanfun'] * 1000, 1),
            'meanfreq_hz': round(features['meanfreq'] * 1000, 1),
            'IQR':         round(features['IQR'], 4),
        },
    }
```

with:

```python
# ── Prediction ────────────────────────────────────────────────────────────────
from pitch_safety_filter import apply_pitch_safety_filter


def predict_gender(audio_path: str, features: dict) -> dict:
    """Runs the primary Wav2Vec2-XLSR gender model, then applies the pitch
    safety filter. Returns the same response shape the old SVM/GBM/RF
    ensemble used to (svm/gbm/rf/ensemble/decision/features) for API
    backward compatibility — svm/gbm/rf now all mirror the single primary
    model's result."""
    primary = gender_verifier.classify_gender(audio_path)
    primary_label = primary['label']
    primary_conf = primary['confidence'] / 100.0

    meanfun_hz = features['meanfun'] * 1000
    meanfreq_hz = features['meanfreq'] * 1000

    final_label, final_conf = apply_pitch_safety_filter(primary_label, primary_conf, meanfun_hz, meanfreq_hz)

    if final_label == 'manual_review':
        logger.info(f"[MANUAL REVIEW] Ambiguous voice. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz, Conf: {final_conf*100:.1f}%.")
    elif final_label != primary_label:
        logger.info(f"[PITCH FILTER] Override applied. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz (Male range).")

    model_output = {'label': final_label, 'confidence': float(final_conf) * 100}

    return {
        'svm': dict(model_output),
        'gbm': dict(model_output),
        'rf': dict(model_output),
        'ensemble': {
            'label':      final_label,
            'confidence': float(final_conf) * 100,
            'male_votes': 3 if final_label == 'male' else 0,
            'total_votes': 3,
        },
        'decision': 'accept' if final_label == 'female' else ('uncertain' if final_label == 'manual_review' else 'reject'),
        'features': {
            'meanfun_hz':  round(meanfun_hz, 1),
            'meanfreq_hz': round(meanfreq_hz, 1),
            'IQR':         round(features['IQR'], 4),
        },
    }
```

- [ ] **Step 8: Update the `/predict` model-loaded guard**

In `main.py`, find (around line 693):

```python
    if svm_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_model.py first.")
```

Replace with:

```python
    if not gender_verifier.is_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_model.py first.")
```

- [ ] **Step 9: Update the `_predict_sync` call site**

In `main.py`, find (around line 822):

```python
            t_gen_start = time.time()
            result   = predict_gender(features)
            t_gen = (time.time() - t_gen_start) * 1000
```

This is inside `_predict_sync`, where `saved_path` is already in scope. Replace with:

```python
            t_gen_start = time.time()
            result   = predict_gender(saved_path, features)
            t_gen = (time.time() - t_gen_start) * 1000
```

- [ ] **Step 10: Remove the corroboration call in `_predict_sync`**

In `main.py`, find (around lines 847-850):

```python
        else:
            if result['ensemble']['label'] == 'female':
                result = _corroborate_female(saved_path, result)

            result['saved_kb'] = round(file_size_kb, 1)
```

Replace with:

```python
        else:
            result['saved_kb'] = round(file_size_kb, 1)
```

- [ ] **Step 11: Update the `/health` endpoint**

In `main.py`, find (around lines 920-924):

```python
        "models_loaded": svm_model is not None,
        "detailed_model_status": {
            "svm_ensemble": svm_model is not None,
            "stt_whisper": stt_model is not None,
        },
```

Replace with:

```python
        "models_loaded": gender_verifier.is_loaded(),
        "detailed_model_status": {
            "primary_gender_model": gender_verifier.is_loaded(),
            "stt_whisper": stt_model is not None,
        },
```

- [ ] **Step 12: Update the `/predict-url` model-loaded guard**

In `main.py`, find (around line 977):

```python
    if svm_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_model.py first.")
```

Replace with:

```python
    if not gender_verifier.is_loaded():
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_model.py first.")
```

- [ ] **Step 13: Update the `_predict_url_sync` call site**

In `main.py`, find (around line 1159):

```python
            t_gen_start = time.time()
            result   = predict_gender(features)
            t_gen = (time.time() - t_gen_start) * 1000
```

This is inside `_predict_url_sync`, where `tmp_path` is already in scope. Replace with:

```python
            t_gen_start = time.time()
            result   = predict_gender(tmp_path, features)
            t_gen = (time.time() - t_gen_start) * 1000
```

- [ ] **Step 14: Remove the corroboration call in `_predict_url_sync`**

In `main.py`, find (around lines 1194-1196):

```python
        if result['ensemble']['label'] == 'female':
            result = _corroborate_female(tmp_path, result)

        label = result['ensemble']['label']
```

Replace with:

```python
        label = result['ensemble']['label']
```

- [ ] **Step 15: Run the integration tests to verify they pass**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_main_integration.py -v`
Expected: 2 passed (this run loads Whisper + the deepfake detector + the primary gender model, so allow 1-2 minutes)

- [ ] **Step 16: Run the full test suite**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -v`
Expected: 10 passed (6 from Task 1, 2 from Task 2, 2 from Task 3)

- [ ] **Step 17: Commit**

```bash
git add main.py tests/test_main_integration.py
git commit -m "Make Wav2Vec2-XLSR the primary gender model, retire the classical ensemble"
```

---

### Task 4: Restart the live server and manually verify end-to-end

**Files:** none (verification only, no code changes)

**Interfaces:** none.

- [ ] **Step 1: Stop the currently running server process**

The server is currently running via `start_server.bat`'s crash-loop wrapper on port 8003. Find and stop it:

Run: `netstat -ano | grep ":8003.*LISTENING"`

Note the PID in the last column, then stop it (this also lets the crash-loop `.bat` window's own `python.exe` process exit — if the crash-loop `.bat` is still looping, stop that shell too, or it will immediately relaunch the old code):

Run: `taskkill //F //PID <pid>`

- [ ] **Step 2: Restart via the crash-loop script**

```powershell
Set-Location "C:\Users\hp\Downloads\voice-gender-master\voice-gender-master"
Start-Process -FilePath "start_server.bat" -WindowStyle Hidden
```

Wait for port 8003 to come up (model loading takes 1-2 minutes):

```bash
until netstat -ano | grep -q ":8003.*LISTENING"; do sleep 2; done; echo "PORT_UP"
```

- [ ] **Step 3: Verify /health shows the new model status**

Run: `curl -s http://127.0.0.1:8003/health`
Expected: JSON contains `"models_loaded":true` and `"detailed_model_status":{"primary_gender_model":true,"stt_whisper":true}` — no `svm_ensemble` key.

- [ ] **Step 4: Verify /predict-url still returns the documented response shape**

Run:
```bash
curl -s -X POST http://127.0.0.1:8003/predict-url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/does-not-exist.wav"}'
```
Expected: `401` if no API key is sent (confirms auth still enforced) — then repeat with the `X-API-Key` header from `.env` and confirm a `400` with a download-failure detail (confirms the route still resolves and reaches the download step, matching the behavior verified during the earlier port-migration fix).

- [ ] **Step 5: Verify /predict end-to-end via the browser UI**

Open `http://127.0.0.1:8003/` in a browser, upload `female_test.wav` via the "Upload File" tab, click "Analyze Voice Gender", and confirm a result renders (Male or Female, with a confidence percentage) with no console errors — same manual check used to verify the CORS fix earlier in this project.

- [ ] **Step 6: Report results**

Note the actual label/confidence returned for `female_test.wav` under the new primary model — this is useful signal for whether the pretrained model agrees or disagrees with the old classical ensemble's read of this fixture (which called it "Male 96.3%").
