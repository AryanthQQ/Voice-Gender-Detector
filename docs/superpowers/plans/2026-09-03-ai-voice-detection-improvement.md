# AI-Voice-Detection Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a real accuracy baseline for the current 3-second-truncation deepfake detector, build a windowed (whole-clip, length-confound-free) alternative, honestly compare both on a held-out test split, and ship whichever one actually wins.

**Architecture:** A pure, model-independent windowing function plus new windowed embedding/prediction methods added to `AdvancedDeepfakeDetector` (Task 1). A standalone evaluation script trains both approaches on the same 80/20 file-level train/test split and reports real metrics (Task 2). Based on those real numbers, the winning approach is retrained on the full dataset and shipped as the production model (Task 3).

**Tech Stack:** `torch`, `transformers` (Wav2Vec2), `xgboost`, `scikit-learn`, `librosa`, `numpy`, pytest.

## Global Constraints

- No new training data collection — use only the existing `data/real/` (629 files) and `data/fake/` (1193 files).
- No change to the embedding model (`facebook/wav2vec2-base-960h`) or classifier type (XGBoost) — only how audio is windowed/aggregated changes.
- Windowing: non-overlapping 3-second (48,000-sample) windows. A clip shorter than 3s produces exactly one window (its own full length). A trailing remainder shorter than 0.5s (8,000 samples) is dropped rather than kept as its own window, UNLESS it's the very first (and only) window for that clip, in which case it's always kept regardless of length.
- Train/test split: 80/20, stratified by class (real/fake), split at the **file level** (never split one file's windows across train and test), fixed `random_state=42` for reproducibility.
- Decision rule: ship the windowed approach only if its test-set accuracy is `>=` the baseline's test-set accuracy. Otherwise keep the current 3-second approach in production. Either way, the evaluation numbers get committed as a documented finding.
- Whichever approach wins gets retrained on the **full** dataset (100% of `data/real` + `data/fake`) before shipping — the 80/20 split is only for honest methodology validation, not for withholding data from the final production model.
- `main.py`'s call site (`advanced_deepfake_detector.predict(path)`) must not need to change — whichever approach wins stays behind that same method name/signature.
- Use the project's existing interpreter for every command: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe`.

---

### Task 1: Windowed embedding extraction and prediction

**Files:**
- Modify: `deepfake_detector_v2.py`
- Create: `tests/test_deepfake_windowing.py`

**Interfaces:**
- Produces: a module-level pure function `_split_into_windows(audio: np.ndarray, sr: int = 16000, window_seconds: float = 3.0, min_seconds: float = 0.5) -> list` (list of numpy arrays, one per window). Also two new methods on `AdvancedDeepfakeDetector`: `extract_windowed_embeddings(self, audio_path: str) -> list` (list of embedding arrays, one per window) and `predict_windowed(self, audio_path: str, threshold: float = 0.50) -> dict` (same response shape as the existing `predict()`). Task 2's evaluation script calls `_split_into_windows` and `extract_windowed_embeddings` directly; Task 3 may wire `predict_windowed`'s logic into `predict()` depending on the evaluation's outcome.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_deepfake_windowing.py`:

```python
import os
import numpy as np
import librosa

from deepfake_detector_v2 import _split_into_windows

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..')


def test_short_clip_produces_one_window():
    windows = _split_into_windows(np.zeros(16000))  # 1 second of silence
    assert len(windows) == 1
    assert len(windows[0]) == 16000


def test_exact_multiple_produces_correct_window_count():
    windows = _split_into_windows(np.zeros(16000 * 9))  # exactly 9 seconds = 3 windows
    assert len(windows) == 3
    assert all(len(w) == 16000 * 3 for w in windows)


def test_trailing_remainder_kept_if_substantial():
    windows = _split_into_windows(np.zeros(16000 * 4))  # 3s window + 1s remainder (>= 0.5s min)
    assert len(windows) == 2
    assert len(windows[0]) == 16000 * 3
    assert len(windows[1]) == 16000 * 1


def test_tiny_trailing_remainder_dropped():
    windows = _split_into_windows(np.zeros(16000 * 3 + 4000))  # 3s window + 0.25s remainder (< 0.5s min)
    assert len(windows) == 1
    assert len(windows[0]) == 16000 * 3


def test_real_fixture_female_test_wav_produces_two_windows():
    # female_test.wav is ~6.104s: two full 3s windows, then a ~0.104s
    # remainder that's below the 0.5s minimum and gets dropped.
    y, sr = librosa.load(os.path.join(FIXTURE_DIR, 'female_test.wav'), sr=16000)
    windows = _split_into_windows(y)
    assert len(windows) == 2
    assert all(len(w) == 16000 * 3 for w in windows)


def test_real_fixture_test_sine_wav_produces_one_window():
    # test_sine.wav is ~1s, well under the 3s window size.
    y, sr = librosa.load(os.path.join(FIXTURE_DIR, 'test_sine.wav'), sr=16000)
    windows = _split_into_windows(y)
    assert len(windows) == 1
    assert len(windows[0]) == len(y)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_deepfake_windowing.py -v`
Expected: FAIL with `ImportError: cannot import name '_split_into_windows'`

- [ ] **Step 3: Add the windowing function and windowed methods**

In `deepfake_detector_v2.py`, add this module-level function near the top of the file (after the imports, before the `_inference_lock` line):

```python
def _split_into_windows(audio: np.ndarray, sr: int = 16000, window_seconds: float = 3.0, min_seconds: float = 0.5) -> list:
    """Splits audio into non-overlapping windows of window_seconds each.
    A trailing remainder shorter than min_seconds is dropped, UNLESS it's
    the very first (and only) window for this clip, in which case it's
    always kept regardless of length (a clip shorter than window_seconds
    still produces exactly one window: itself)."""
    window_samples = int(sr * window_seconds)
    min_samples = int(sr * min_seconds)
    windows = []
    for start in range(0, len(audio), window_samples):
        chunk = audio[start:start + window_samples]
        if len(chunk) < min_samples and windows:
            continue
        windows.append(chunk)
    if not windows:
        windows = [audio]
    return windows
```

Then add these two methods to the `AdvancedDeepfakeDetector` class, right after the existing `extract_wav2vec_embedding` method:

```python
    def extract_windowed_embeddings(self, audio_path: str) -> list:
        """Splits audio_path into 3-second windows covering the whole clip
        and returns one Wav2Vec2 embedding per window. Each window's
        embedding is cached separately from the single-embedding cache
        (a different filename pattern) so the two caching schemes never
        collide or silently reuse stale data from one another."""
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return []

        windows = _split_into_windows(audio)
        embeddings = []
        for i, window_audio in enumerate(windows):
            cache_path = f"{audio_path}.win{i}.npy"
            if os.path.exists(cache_path):
                embeddings.append(np.load(cache_path))
                continue
            try:
                inputs = self.feature_extractor(window_audio, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with _inference_lock:
                    with torch.no_grad():
                        outputs = self.wav2vec_model(**inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
                np.save(cache_path, embedding)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing window {i} of {audio_path}: {e}")
        return embeddings

    def predict_windowed(self, audio_path: str, threshold: float = 0.50) -> dict:
        """Same response shape as predict(), but scores every 3-second
        window of the clip independently and takes the MAXIMUM fake
        probability across windows as the final score — any one
        fake-sounding segment is enough to flag the whole clip."""
        if not self.is_trained:
            if os.path.exists("models/wav2vec_xgb.pkl"):
                self.classifier = joblib.load("models/wav2vec_xgb.pkl")
                self.scaler = joblib.load("models/wav2vec_scaler.pkl")
                self.is_trained = True
            else:
                return {"is_ai": False, "confidence": 0.0, "reason": "Model not trained", "status": "error"}

        embeddings = self.extract_windowed_embeddings(audio_path)
        if not embeddings:
            return {"is_ai": False, "confidence": 0.0, "reason": "Failed to extract embeddings", "status": "error"}

        probs = []
        for embedding in embeddings:
            embedding_scaled = self.scaler.transform([embedding])
            probs.append(self.classifier.predict_proba(embedding_scaled)[0][1])

        prob = max(probs)
        is_ai = prob >= threshold
        confidence = float(round(prob * 100, 1))

        if is_ai:
            reason = f"AI Voice Detected (Confidence: {confidence}%)"
            print(f"[REJECT] FAKE! AI DETECTED -> {confidence}%")
        else:
            reason = f"Real Human Voice (AI Prob: {confidence}%)"
            print(f"[OK] REAL HUMAN -> {confidence}%")

        return {
            "is_ai": bool(is_ai),
            "confidence": confidence,
            "probability_ai": confidence,
            "reason": reason,
            "status": "success"
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/test_deepfake_windowing.py -v`
Expected: 6 passed.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -v`
Expected: all pre-existing tests still pass, plus these 6 new ones (this run takes 1-2 minutes due to real model loading in unrelated pre-existing tests).

- [ ] **Step 6: Commit**

```bash
git add deepfake_detector_v2.py tests/test_deepfake_windowing.py
git commit -m "Add windowed embedding extraction and prediction to AdvancedDeepfakeDetector"
```

---

### Task 2: Honest baseline-vs-windowed evaluation

**Files:**
- Create: `evaluate_deepfake_windowing.py`
- Create: `docs/superpowers/reports/2026-09-03-deepfake-detector-evaluation.md` (written by running the script)

**Interfaces:**
- Consumes: `AdvancedDeepfakeDetector.extract_wav2vec_embedding` (existing), `AdvancedDeepfakeDetector.extract_windowed_embeddings` (Task 1), `_split_into_windows` (Task 1, used indirectly).
- Produces: a committed evaluation report with real accuracy/precision/recall/F1 numbers for both approaches. Task 3 reads these numbers to decide which approach to ship — there is no code interface between this task and Task 3, only the report's contents.

- [ ] **Step 1: Write the evaluation script**

Create `evaluate_deepfake_windowing.py`:

```python
"""
evaluate_deepfake_windowing.py — Honest comparison of the current
(3-second-truncation) deepfake detector against a windowed (whole-clip,
max-aggregated) alternative, on a held-out test split neither model is
trained on.

Run: python evaluate_deepfake_windowing.py
Expected runtime: baseline embeddings mostly reuse existing .npy caches
(fast); windowed embeddings are computed fresh for every window of every
file (slow — expect roughly 10-45 minutes on CPU for ~1800 files).
"""
import os
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

from deepfake_detector_v2 import AdvancedDeepfakeDetector

REAL_DIR = "data/real"
FAKE_DIR = "data/fake"
AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.flac'}
RANDOM_STATE = 42


def list_audio_files(directory):
    return sorted(str(f) for f in Path(directory).glob("*.*") if f.suffix.lower() in AUDIO_EXTS)


def main():
    real_files = list_audio_files(REAL_DIR)
    fake_files = list_audio_files(FAKE_DIR)
    print(f"Found {len(real_files)} real files, {len(fake_files)} fake files.")

    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=0.2, stratify=all_labels, random_state=RANDOM_STATE
    )
    print(f"Train: {len(train_files)} files. Test: {len(test_files)} files.")

    detector = AdvancedDeepfakeDetector()

    # ── Baseline: single 3-second-truncation embedding per file ──────────
    print("\n=== Baseline (3-second truncation) ===")
    print("Extracting baseline train embeddings (reuses existing .npy caches where present)...")
    X_train_base, y_train_base = [], []
    for f, label in zip(train_files, train_labels):
        emb = detector.extract_wav2vec_embedding(f)
        if emb is not None:
            X_train_base.append(emb)
            y_train_base.append(label)

    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(np.array(X_train_base))
    clf_base = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    clf_base.fit(X_train_base_scaled, y_train_base)

    print("Evaluating baseline on held-out test files...")
    y_true_base, y_pred_base = [], []
    for f, label in zip(test_files, test_labels):
        emb = detector.extract_wav2vec_embedding(f)
        if emb is None:
            continue
        emb_scaled = scaler_base.transform([emb])
        pred = int(clf_base.predict(emb_scaled)[0])
        y_true_base.append(label)
        y_pred_base.append(pred)

    baseline_metrics = {
        'accuracy': accuracy_score(y_true_base, y_pred_base),
        'precision': precision_score(y_true_base, y_pred_base, zero_division=0),
        'recall': recall_score(y_true_base, y_pred_base, zero_division=0),
        'f1': f1_score(y_true_base, y_pred_base, zero_division=0),
    }
    print(f"Baseline metrics: {baseline_metrics}")

    # ── Windowed: every window of every train file is its own example ────
    print("\n=== Windowed (whole-clip, max-aggregated) ===")
    print("Extracting windowed train embeddings (fresh computation, this is the slow part)...")
    X_train_win, y_train_win = [], []
    for f, label in zip(train_files, train_labels):
        for emb in detector.extract_windowed_embeddings(f):
            X_train_win.append(emb)
            y_train_win.append(label)

    scaler_win = StandardScaler()
    X_train_win_scaled = scaler_win.fit_transform(np.array(X_train_win))
    clf_win = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    clf_win.fit(X_train_win_scaled, y_train_win)

    print("Evaluating windowed model on held-out test files (max-aggregation per file)...")
    y_true_win, y_pred_win = [], []
    for f, label in zip(test_files, test_labels):
        embeddings = detector.extract_windowed_embeddings(f)
        if not embeddings:
            continue
        probs = [clf_win.predict_proba(scaler_win.transform([emb]))[0][1] for emb in embeddings]
        pred = int(max(probs) >= 0.5)
        y_true_win.append(label)
        y_pred_win.append(pred)

    windowed_metrics = {
        'accuracy': accuracy_score(y_true_win, y_pred_win),
        'precision': precision_score(y_true_win, y_pred_win, zero_division=0),
        'recall': recall_score(y_true_win, y_pred_win, zero_division=0),
        'f1': f1_score(y_true_win, y_pred_win, zero_division=0),
    }
    print(f"Windowed metrics: {windowed_metrics}")

    # ── Report ─────────────────────────────────────────────────────────
    winner = "windowed" if windowed_metrics['accuracy'] >= baseline_metrics['accuracy'] else "baseline"
    report = f"""# Deepfake Detector: Baseline vs Windowed Evaluation

Date: 2026-09-03

Train/test split: 80/20, stratified by class, file-level, random_state={RANDOM_STATE}.
Train files: {len(train_files)}. Test files: {len(test_files)}.

## Baseline (current 3-second-truncation approach)

| Metric | Value |
|---|---|
| Accuracy | {baseline_metrics['accuracy']:.4f} |
| Precision | {baseline_metrics['precision']:.4f} |
| Recall | {baseline_metrics['recall']:.4f} |
| F1 | {baseline_metrics['f1']:.4f} |

## Windowed (whole-clip, max-aggregated approach)

| Metric | Value |
|---|---|
| Accuracy | {windowed_metrics['accuracy']:.4f} |
| Precision | {windowed_metrics['precision']:.4f} |
| Recall | {windowed_metrics['recall']:.4f} |
| F1 | {windowed_metrics['f1']:.4f} |

## Decision

Per the project's decision rule (ship windowed only if its test accuracy
is >= baseline's test accuracy): **{winner}** wins.
"""

    os.makedirs("docs/superpowers/reports", exist_ok=True)
    report_path = "docs/superpowers/reports/2026-09-03-deepfake-detector-evaluation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to {report_path}")
    print(f"WINNER: {winner}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the evaluation script**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe evaluate_deepfake_windowing.py`

Expected: this takes a while (the windowed embedding extraction step, in particular, may take 10-45 minutes on CPU since it computes a fresh embedding for every ~3-second window of every one of ~1800 files — there is no existing cache for these). Let it run to completion. It prints progress and ends with `Report written to ...` and `WINNER: baseline` or `WINNER: windowed`.

- [ ] **Step 3: Verify the report was written and looks sane**

Read `docs/superpowers/reports/2026-09-03-deepfake-detector-evaluation.md` and confirm it has real, non-placeholder numbers for both approaches (accuracy/precision/recall/F1 all present, not NaN or 0.0000 for both — a genuine 0.0000 across the board would indicate something went wrong in the script, e.g. no embeddings were extracted; investigate and fix rather than accepting nonsensical numbers). Note the winner in your task report (Step 5 below) — this is the single most important fact this task produces.

- [ ] **Step 4: Commit**

```bash
git add evaluate_deepfake_windowing.py docs/superpowers/reports/2026-09-03-deepfake-detector-evaluation.md
git commit -m "Add and run baseline-vs-windowed deepfake detector evaluation"
```

Do NOT commit the `.win*.npy` cache files this run generates inside `data/fake/` and `data/real/` — check `.gitignore` already excludes `data/` or these cache patterns before committing; if it doesn't, that's fine, just don't `git add` them explicitly (only add the two files named above).

- [ ] **Step 5: Report the winner clearly**

In your task report to the controller, state in the FIRST line: "WINNER: baseline" or "WINNER: windowed", followed by both sets of metrics. This determines how Task 3 is scoped.

---

### Task 3: Ship the winning approach

**This task's exact steps depend on Task 2's real result (baseline vs windowed winning) — the controller dispatching this task will fill in the applicable branch below based on the actual evaluation report. Both branches are fully specified here so nothing is invented at dispatch time.**

**Files:**
- Modify: `deepfake_detector_v2.py` (only if windowed wins)
- Modify: `models/wav2vec_xgb.pkl`, `models/wav2vec_scaler.pkl` (only if windowed wins — retrained on full data)
- Create: `retrain_deepfake_production_model.py` (only if windowed wins)

**Interfaces:** none new — `main.py`'s call site (`advanced_deepfake_detector.predict(path)`) is unchanged either way.

#### Branch A: baseline wins (windowed test accuracy < baseline test accuracy)

No production code or model changes needed — the current approach already IS what's in production. Nothing to ship. Skip directly to the final verification step below.

- [ ] **Step A1: No-op confirmation**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -c "from deepfake_detector_v2 import AdvancedDeepfakeDetector; d = AdvancedDeepfakeDetector(); print(d.predict('female_test.wav'))"`
Expected: prints a valid result dict (`is_ai`, `confidence`, `probability_ai`, `reason`, `status`) — confirms the existing, unchanged production path still works. No commit needed for this branch (Task 2's commit already captured the finding).

#### Branch B: windowed wins (windowed test accuracy >= baseline test accuracy)

- [ ] **Step B1: Back up the current production model files**

```bash
cp models/wav2vec_xgb.pkl models/wav2vec_xgb_pre_windowing_backup.pkl
cp models/wav2vec_scaler.pkl models/wav2vec_scaler_pre_windowing_backup.pkl
```

- [ ] **Step B2: Write and run a script to retrain on the full dataset using the windowed approach**

Create `retrain_deepfake_production_model.py`:

```python
"""
retrain_deepfake_production_model.py — Retrains the production deepfake
detector using the windowed approach (validated as the winner in
docs/superpowers/reports/2026-09-03-deepfake-detector-evaluation.md) on
the FULL dataset (data/real + data/fake, no train/test split held back —
the split was only for honest methodology validation).

Run: python retrain_deepfake_production_model.py
"""
from pathlib import Path

import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from deepfake_detector_v2 import AdvancedDeepfakeDetector

REAL_DIR = "data/real"
FAKE_DIR = "data/fake"
AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.flac'}


def list_audio_files(directory):
    return sorted(str(f) for f in Path(directory).glob("*.*") if f.suffix.lower() in AUDIO_EXTS)


def main():
    detector = AdvancedDeepfakeDetector()

    X, y = [], []
    for f in list_audio_files(REAL_DIR):
        for emb in detector.extract_windowed_embeddings(f):
            X.append(emb)
            y.append(0)
    for f in list_audio_files(FAKE_DIR):
        for emb in detector.extract_windowed_embeddings(f):
            X.append(emb)
            y.append(1)

    print(f"Training on {len(y)} windowed examples ({sum(1 for l in y if l == 0)} real, {sum(1 for l in y if l == 1)} fake).")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.array(X))
    clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(X_scaled, y)

    joblib.dump(clf, "models/wav2vec_xgb.pkl")
    joblib.dump(scaler, "models/wav2vec_scaler.pkl")
    print("Saved retrained production model to models/wav2vec_xgb.pkl and models/wav2vec_scaler.pkl")


if __name__ == "__main__":
    main()
```

Run it: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe retrain_deepfake_production_model.py`
Expected runtime: similar order of magnitude to Task 2's windowed extraction step (windowed embeddings for the full ~1800-file dataset), likely faster than Task 2 overall since only one approach is being trained, not two, and Task 2's run already populated the `.winN.npy` caches for every file (both train and test), so this step should mostly reuse those caches rather than recomputing them.

- [ ] **Step B3: Wire `predict()` to use the windowed logic**

In `deepfake_detector_v2.py`, replace the body of the existing `predict()` method:

```python
    def predict(self, audio_path: str, threshold=0.50) -> dict:
        """Final prediction function for your agent"""
        if not self.is_trained:
            if os.path.exists("models/wav2vec_xgb.pkl"):
                self.classifier = joblib.load("models/wav2vec_xgb.pkl")
                self.scaler = joblib.load("models/wav2vec_scaler.pkl")
                self.is_trained = True
            else:
                return {"is_ai": False, "confidence": 0.0, "reason": "Model not trained", "status": "error"}
        
        embedding = self.extract_wav2vec_embedding(audio_path)
        if embedding is None:
            return {"is_ai": False, "confidence": 0.0, "reason": "Failed to extract embedding", "status": "error"}

        embedding_scaled = self.scaler.transform([embedding])
        
        prob = self.classifier.predict_proba(embedding_scaled)[0][1]  # Probability of being FAKE
        is_ai = prob >= threshold
        
        confidence = float(round(prob * 100, 1))
        
        if is_ai:
            reason = f"AI Voice Detected (Confidence: {confidence}%)"
            print(f"[REJECT] FAKE! AI DETECTED -> {confidence}%")
        else:
            reason = f"Real Human Voice (AI Prob: {confidence}%)"
            print(f"[OK] REAL HUMAN -> {confidence}%")
        
        return {
            "is_ai": bool(is_ai),
            "confidence": confidence,
            "probability_ai": confidence,
            "reason": reason,
            "status": "success"
        }
```

with:

```python
    def predict(self, audio_path: str, threshold=0.50) -> dict:
        """Final prediction function for your agent. Uses the windowed
        (whole-clip, max-aggregated) approach — validated to outperform
        the old 3-second-truncation approach on a held-out test set, see
        docs/superpowers/reports/2026-09-03-deepfake-detector-evaluation.md."""
        return self.predict_windowed(audio_path, threshold=threshold)
```

(This keeps `extract_wav2vec_embedding` and the single-embedding path in the codebase — unused by `predict()` now, but still present since Task 2's evaluation script and any future re-evaluation depend on it existing.)

- [ ] **Step B4: Smoke-test the new production path**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -c "from deepfake_detector_v2 import AdvancedDeepfakeDetector; d = AdvancedDeepfakeDetector(); print(d.predict('female_test.wav'))"`
Expected: prints a valid result dict (`is_ai`, `confidence`, `probability_ai`, `reason`, `status`) — same shape as before, now backed by the retrained windowed model.

- [ ] **Step B5: Run the full test suite**

Run: `C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe -m pytest tests/ -v`
Expected: all tests still pass (this run takes 1-2 minutes due to real model loading).

- [ ] **Step B6: Commit**

```bash
git add deepfake_detector_v2.py retrain_deepfake_production_model.py models/wav2vec_xgb.pkl models/wav2vec_scaler.pkl models/wav2vec_xgb_pre_windowing_backup.pkl models/wav2vec_scaler_pre_windowing_backup.pkl
git commit -m "Ship windowed deepfake detector as production model (validated winner)"
```
