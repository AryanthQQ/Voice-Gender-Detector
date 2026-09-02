# Wav2Vec2 False-Accept Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the retired classical SVM/GBM/RF ensemble back as a secondary corroboration check on `female` gender verdicts, so a Wav2Vec2-XLSR primary-model false positive gets escalated to `manual_review` instead of auto-accepted, on both `/predict` and `/predict-url`.

**Architecture:** A new module (`classical_corroborator.py`) reloads the existing `models/*.pkl` files (unchanged, not retrained) and exposes a pure `should_escalate()` decision function alongside the model-dependent `corroborate()` call. `main.py`'s `predict_gender()` calls both, but only when the pitch-safety-filter's output is already `female` (about to auto-accept) — mirroring the codebase's existing corroboration-skip convention.

**Tech Stack:** `scikit-learn` (via existing `joblib`-loaded `.pkl` models), `numpy`, pytest.

## Global Constraints

- No retraining — reuse `models/svm_model.pkl`, `gbm_model.pkl`, `rf_model.pkl`, `scaler.pkl`, `features.pkl` exactly as they are.
- Corroboration runs only when the pitch-safety-filter's `final_label == 'female'` — never on `male`/`manual_review` verdicts (matches the codebase's existing corroboration-skip convention, avoids the extra model cost on requests that won't auto-accept anyway).
- Escalation rule: escalate `female` → `manual_review` if `male_votes >= 2` (majority of the 3 classical models disagree), OR any single classical model votes `male` at confidence `>= 90%`. Never force a `reject` or force an `accept` — corroboration only ever escalates toward a human decision.
- If corroboration itself fails (exception), treat that as grounds to escalate to `manual_review` too — never let a corroboration failure silently fall through to auto-accept.
- No API request/response shape changes on `/predict` or `/predict-url` — `svm`/`gbm`/`rf`/`ensemble` continue to all mirror the single final decision, exactly as established in the primary-gender-model-upgrade project. Corroboration is internal plumbing, not a new response field.
- Applies to both `/predict` and `/predict-url`.
- Use the project's existing interpreter for every command: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe`.

---

### Task 1: Classical ensemble corroboration module

**Files:**
- Create: `classical_corroborator.py`
- Create: `tests/test_classical_corroborator.py`

**Interfaces:**
- Produces: `load_models() -> None` (idempotent), `is_loaded() -> bool`, `corroborate(features: dict) -> dict` returning `{'svm': {'label': str, 'confidence': float}, 'gbm': {...}, 'rf': {...}, 'male_votes': int}`, and the pure function `should_escalate(corroboration: dict) -> bool`. Task 2's `main.py` changes call all four by these exact names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_classical_corroborator.py`:

```python
import os
import pytest

from classical_corroborator import load_models, is_loaded, corroborate, should_escalate

FIXTURE = os.path.join(os.path.dirname(__file__), '..', 'female_test.wav')

# Real feature vectors extracted from actual audio, used to exercise corroborate()
# against the real loaded models without needing extract_features() here.
MALE_LIKE_FEATURES = {
    'meanfreq': 0.173496531827012, 'sd': 0.06, 'median': 0.17, 'Q25': 0.13, 'Q75': 0.20,
    'IQR': 0.03125, 'skew': 0.5, 'kurt': 3.0, 'sp.ent': 0.9, 'sfm': 0.05, 'mode': 0.17,
    'centroid': 0.17, 'meanfun': 0.22541925340157143, 'minfun': 0.08, 'maxfun': 0.28,
    'meandom': 0.15, 'mindom': 0.05, 'maxdom': 0.25, 'dfrange': 0.2, 'modindx': 0.1,
}


def test_load_models_is_idempotent_and_sets_loaded_flag():
    load_models()
    assert is_loaded() is True
    load_models()  # calling again must not error
    assert is_loaded() is True


def test_corroborate_returns_valid_schema():
    load_models()
    result = corroborate(MALE_LIKE_FEATURES)
    assert set(result.keys()) == {'svm', 'gbm', 'rf', 'male_votes'}
    for key in ('svm', 'gbm', 'rf'):
        assert set(result[key].keys()) == {'label', 'confidence'}
        assert result[key]['label'] in ('male', 'female')
        assert 0.0 <= result[key]['confidence'] <= 100.0
    assert 0 <= result['male_votes'] <= 3


def test_should_escalate_true_when_majority_disagrees():
    corroboration = {
        'svm': {'label': 'male', 'confidence': 80.0},
        'gbm': {'label': 'male', 'confidence': 75.0},
        'rf':  {'label': 'female', 'confidence': 55.0},
        'male_votes': 2,
    }
    assert should_escalate(corroboration) is True


def test_should_escalate_true_on_single_high_confidence_disagreement():
    corroboration = {
        'svm': {'label': 'male', 'confidence': 95.0},
        'gbm': {'label': 'female', 'confidence': 60.0},
        'rf':  {'label': 'female', 'confidence': 55.0},
        'male_votes': 1,
    }
    assert should_escalate(corroboration) is True


def test_should_escalate_false_when_ensemble_agrees_female():
    corroboration = {
        'svm': {'label': 'female', 'confidence': 85.0},
        'gbm': {'label': 'female', 'confidence': 80.0},
        'rf':  {'label': 'male', 'confidence': 60.0},  # single low-confidence dissent, not enough
        'male_votes': 1,
    }
    assert should_escalate(corroboration) is False


def test_corroborate_gets_the_original_false_accept_case_right():
    """Regression test for the specific audio file that caused a real
    Wav2Vec2 false-accept in production testing. Skips gracefully if that
    personal debugging fixture isn't present on this machine — it's not a
    project asset."""
    audio_path = r"C:\Users\hp\Desktop\male voice\1779778933389-47662627.mp3"
    if not os.path.exists(audio_path):
        pytest.skip("Personal debugging fixture not present on this machine")

    import main  # heavy import, needed for extract_features + safe_load_audio
    import soundfile as sf
    y, sr = main.safe_load_audio(audio_path, sr=16000)
    tmp_wav = os.path.join(os.environ.get('TEMP', '.'), 'corroborator_regression_test.wav')
    sf.write(tmp_wav, y, 16000)
    features = main.extract_features(tmp_wav)

    load_models()
    result = corroborate(features)
    assert should_escalate(result) is True, (
        f"Classical ensemble should have flagged this known false-accept case, got: {result}"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_classical_corroborator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'classical_corroborator'`

- [ ] **Step 3: Write the implementation**

Create `classical_corroborator.py`:

```python
"""
classical_corroborator.py — Secondary classical-ML corroboration check for
'female' gender verdicts, using the SVM/GBM/RF ensemble that used to be the
primary decision-maker before the primary-gender-model-upgrade project
retired it in favor of Wav2Vec2-XLSR. Models are unchanged, still on disk.

Only invoked when the primary model + pitch-safety-filter have already
decided 'female' (i.e. about to auto-accept) — if this disagrees, the
caller escalates to manual_review instead of accepting. Never used to
force a reject or an accept on its own.
"""
import os

import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

_svm_model = None
_gbm_model = None
_rf_model = None
_scaler = None
_feature_order = None


def load_models():
    """Loads the classical models if not already loaded. Idempotent."""
    global _svm_model, _gbm_model, _rf_model, _scaler, _feature_order
    if _svm_model is not None:
        return
    _svm_model     = joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl'))
    _gbm_model     = joblib.load(os.path.join(MODELS_DIR, 'gbm_model.pkl'))
    _rf_model      = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
    _scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    _feature_order = joblib.load(os.path.join(MODELS_DIR, 'features.pkl'))


def is_loaded() -> bool:
    return _svm_model is not None


def corroborate(features: dict) -> dict:
    """Runs the classical SVM/GBM/RF ensemble on an already-extracted
    feature dict (the same dict extract_features() produces in main.py).
    Returns {'svm': {label, confidence}, 'gbm': {...}, 'rf': {...},
    'male_votes': int}."""
    load_models()

    feat_vec = np.array([[features[f] for f in _feature_order]])
    feat_scaled = _scaler.transform(feat_vec)

    # SVM was trained on unscaled features; GBM/RF on scaled — matches the
    # exact behavior of the original (pre-retirement) ensemble code.
    svm_prob = _svm_model.predict_proba(feat_vec)[0]
    gbm_prob = _gbm_model.predict_proba(feat_scaled)[0]
    rf_prob  = _rf_model.predict_proba(feat_scaled)[0]

    svm_pred = int(np.argmax(svm_prob))
    gbm_pred = int(np.argmax(gbm_prob))
    rf_pred  = int(np.argmax(rf_prob))

    def as_result(pred, prob):
        return {'label': 'male' if pred == 1 else 'female', 'confidence': float(max(prob)) * 100}

    male_votes = sum([svm_pred, gbm_pred, rf_pred])

    return {
        'svm': as_result(svm_pred, svm_prob),
        'gbm': as_result(gbm_pred, gbm_prob),
        'rf':  as_result(rf_pred, rf_prob),
        'male_votes': male_votes,
    }


def should_escalate(corroboration: dict) -> bool:
    """True if the classical ensemble disagrees strongly enough with a
    'female' primary verdict to escalate to manual_review instead of
    auto-accepting. Escalation-only signal — never forces a reject or an
    accept on its own.

    Rule: majority (2 or 3 of 3) vote 'male', OR any single model votes
    'male' at >=90% confidence. Starting-point thresholds, not validated
    at scale (derived from an 8-file batch) — revisit with real traffic."""
    if corroboration['male_votes'] >= 2:
        return True
    for key in ('svm', 'gbm', 'rf'):
        model = corroboration[key]
        if model['label'] == 'male' and model['confidence'] >= 90.0:
            return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_classical_corroborator.py -v`
Expected: 6 passed (5 always run; `test_corroborate_gets_the_original_false_accept_case_right` either passes or is skipped depending on whether the personal debugging fixture exists on this machine — both outcomes are acceptable, a FAIL is not).

- [ ] **Step 5: Commit**

```bash
git add classical_corroborator.py tests/test_classical_corroborator.py
git commit -m "Add classical-ensemble corroboration module for female verdicts"
```

---

### Task 2: Wire corroboration into predict_gender for both endpoints

**Files:**
- Modify: `main.py` (see exact line-anchored edits below; line numbers are from the pre-Task-2 file state)
- Modify: `tests/test_main_integration.py`

**Interfaces:**
- Consumes: `classical_corroborator.load_models` / `is_loaded` / `corroborate` / `should_escalate` (Task 1).
- Produces: nothing new consumed by a later task — this is the last task.

- [ ] **Step 1: Write the failing integration tests**

Read `tests/test_main_integration.py` first to see its existing imports/fixtures (`import main`, `import os`, `import pytest`, `from fastapi.testclient import TestClient`, a session-scoped `client` fixture, `FIXTURE` constant, and `from unittest.mock import patch, MagicMock` from earlier work) — reuse these, don't redefine them.

Add these two tests to the file (keep everything already in it untouched):

```python
def test_predict_gender_escalates_when_classical_ensemble_disagrees():
    fake_features = {'meanfun': 0.21, 'meanfreq': 0.20, 'IQR': 0.05}  # clearly-female pitch range
    with patch('gender_verifier.classify_gender', return_value={'label': 'female', 'confidence': 99.0}), \
         patch('classical_corroborator.corroborate', return_value={
             'svm': {'label': 'male', 'confidence': 90.0},
             'gbm': {'label': 'male', 'confidence': 85.0},
             'rf':  {'label': 'female', 'confidence': 55.0},
             'male_votes': 2,
         }):
        result = main.predict_gender('/fake/path/does-not-matter.wav', fake_features)
    assert result['ensemble']['label'] == 'manual_review'
    assert result['decision'] == 'uncertain'


def test_predict_gender_keeps_accept_when_classical_ensemble_agrees():
    fake_features = {'meanfun': 0.21, 'meanfreq': 0.20, 'IQR': 0.05}
    with patch('gender_verifier.classify_gender', return_value={'label': 'female', 'confidence': 99.0}), \
         patch('classical_corroborator.corroborate', return_value={
             'svm': {'label': 'female', 'confidence': 88.0},
             'gbm': {'label': 'female', 'confidence': 82.0},
             'rf':  {'label': 'male', 'confidence': 60.0},
             'male_votes': 1,
         }):
        result = main.predict_gender('/fake/path/does-not-matter.wav', fake_features)
    assert result['ensemble']['label'] == 'female'
    assert result['decision'] == 'accept'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest "tests/test_main_integration.py::test_predict_gender_escalates_when_classical_ensemble_disagrees" "tests/test_main_integration.py::test_predict_gender_keeps_accept_when_classical_ensemble_agrees" -v`
Expected: FAIL — `ModuleNotFoundError` or `AttributeError` referencing `classical_corroborator`, since `main.py` doesn't import or call it yet.

- [ ] **Step 3: Add the import**

In `main.py`, find:

```python
from deepfake_detector_v2 import AdvancedDeepfakeDetector
import gender_guesser.detector as gender
import fingerprint
import fingerprint_store
```

Replace with:

```python
from deepfake_detector_v2 import AdvancedDeepfakeDetector
import gender_guesser.detector as gender
import fingerprint
import fingerprint_store
import classical_corroborator
```

- [ ] **Step 4: Load the classical models at startup**

In `main.py`, find:

```python
# ── Primary Gender Model ─────────────────────────────────────────────────────
import gender_verifier

try:
    gender_verifier.load_model()
    logger.info("[OK] Primary gender model (Wav2Vec2-XLSR) loaded successfully!")
except Exception as e:
    logger.exception(f"[ERR] Error loading primary gender model: {e}")
```

Replace with:

```python
# ── Primary Gender Model ─────────────────────────────────────────────────────
import gender_verifier

try:
    gender_verifier.load_model()
    logger.info("[OK] Primary gender model (Wav2Vec2-XLSR) loaded successfully!")
except Exception as e:
    logger.exception(f"[ERR] Error loading primary gender model: {e}")

try:
    classical_corroborator.load_models()
    logger.info("[OK] Classical corroboration ensemble (SVM/GBM/RF) loaded successfully!")
except Exception as e:
    logger.exception(f"[ERR] Error loading classical corroboration ensemble: {e}")
```

- [ ] **Step 5: Update the two model-loaded guard checks**

In `main.py`, there are two occurrences of this exact block (one in the `/predict` route, one in the `/predict-url` route):

```python
    if not gender_verifier.is_loaded():
        raise HTTPException(status_code=503, detail="Primary gender model failed to load. Check server logs.")
```

Replace **both** occurrences with:

```python
    if not gender_verifier.is_loaded() or not classical_corroborator.is_loaded():
        raise HTTPException(status_code=503, detail="Gender models failed to load. Check server logs.")
```

- [ ] **Step 6: Update the /health endpoint**

In `main.py`, find:

```python
        "models_loaded": gender_verifier.is_loaded(),
        "detailed_model_status": {
            "primary_gender_model": gender_verifier.is_loaded(),
            "stt_whisper": stt_model is not None,
        },
```

Replace with:

```python
        "models_loaded": gender_verifier.is_loaded() and classical_corroborator.is_loaded(),
        "detailed_model_status": {
            "primary_gender_model": gender_verifier.is_loaded(),
            "classical_corroborator": classical_corroborator.is_loaded(),
            "stt_whisper": stt_model is not None,
        },
```

- [ ] **Step 7: Add the corroboration step to predict_gender**

In `main.py`, find:

```python
    if final_label == 'manual_review':
        logger.info(f"[MANUAL REVIEW] Ambiguous voice. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz, Conf: {final_conf*100:.1f}%.")
    elif final_label != primary_label:
        logger.info(f"[PITCH FILTER] Override applied. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz (Male range).")

    model_output = {'label': final_label, 'confidence': float(final_conf) * 100}
```

Replace with:

```python
    if final_label == 'manual_review':
        logger.info(f"[MANUAL REVIEW] Ambiguous voice. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz, Conf: {final_conf*100:.1f}%.")
    elif final_label != primary_label:
        logger.info(f"[PITCH FILTER] Override applied. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz (Male range).")

    if final_label == 'female':
        try:
            corroboration = classical_corroborator.corroborate(features)
            if classical_corroborator.should_escalate(corroboration):
                logger.info(f"[MANUAL REVIEW] Classical ensemble disagreed with primary 'female' verdict (male_votes={corroboration['male_votes']}/3, svm={corroboration['svm']}, gbm={corroboration['gbm']}, rf={corroboration['rf']}). Escalating.")
                final_label = 'manual_review'
        except Exception as e:
            logger.exception(f"[WARN] Classical corroboration failed, escalating to manual_review as a precaution: {e}")
            final_label = 'manual_review'

    model_output = {'label': final_label, 'confidence': float(final_conf) * 100}
```

- [ ] **Step 8: Run the new integration tests to verify they pass**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest "tests/test_main_integration.py::test_predict_gender_escalates_when_classical_ensemble_disagrees" "tests/test_main_integration.py::test_predict_gender_keeps_accept_when_classical_ensemble_agrees" -v`
Expected: 2 passed.

- [ ] **Step 9: Run the full test suite**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -v`
Expected: all pre-existing tests still pass, plus Task 1's 6 and this task's 2 (verify the actual total from the test run's own summary line — don't assume a specific number, count what the suite reports and confirm nothing was skipped that shouldn't be or failed).

- [ ] **Step 10: Manually verify against the live server**

The live dev server (if running) needs a restart to pick up this code change — it does not hot-reload. After restarting it on port 8003 (see `start_server.bat`), re-run the original false-accept case to confirm it's now escalated instead of accepted:

```bash
curl -s -X POST http://127.0.0.1:8003/predict -F "file=@C:\Users\hp\Desktop\male voice\1779778933389-47662627.mp3" -F "advisor_name=RegressionCheck"
```

Expected: `"decision":"uncertain"` (or `"ensemble":{"label":"manual_review",...}`), not `"decision":"accept"`.

- [ ] **Step 11: Commit**

```bash
git add main.py tests/test_main_integration.py
git commit -m "Wire classical-ensemble corroboration into predict_gender for /predict and /predict-url"
```
