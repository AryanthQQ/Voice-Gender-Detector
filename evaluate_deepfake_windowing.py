"""
evaluate_deepfake_windowing.py — Honest comparison of the current
(3-second-truncation) deepfake detector against a windowed (whole-clip,
max-aggregated) alternative, on a held-out test split neither model is
trained on.

Before splitting, every candidate file is pre-scanned with a lightweight
librosa.load() check, independent of any embedding cache, so that files
which are simply unloadable/corrupted are excluded from BOTH approaches'
population up front. This matters because the baseline path
(extract_wav2vec_embedding) can return a cached embedding for a file
without ever re-reading the audio, while the windowed path
(extract_windowed_embeddings) had no pre-existing cache and always reads
the audio fresh — so without this pre-scan, a corrupted file could
silently "succeed" for baseline via a stale cache hit while genuinely
failing for windowed, making the two approaches' evaluated populations
different sizes and the accuracy comparison unfair (and unrepresentative
of production, where incoming audio is always uncached).

Run: python evaluate_deepfake_windowing.py
Expected runtime: the pre-scan reads every file once (moderate). Baseline
embeddings mostly reuse existing .npy caches (fast); windowed embeddings
also mostly reuse the .win{i}.npy caches computed by the previous run
(fast) — only files with no cache at all require fresh Wav2Vec2 inference.
"""
import os
from pathlib import Path

import librosa
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


def is_loadable(audio_path):
    """Lightweight pre-scan check, independent of any embedding cache:
    can this file actually be read as audio right now? Used to exclude
    corrupted/unreadable files from BOTH approaches' population up front,
    so neither approach gets an unfair pass via a stale cache hit."""
    try:
        librosa.load(audio_path, sr=16000)
        return True
    except Exception:
        return False


def main():
    real_files = list_audio_files(REAL_DIR)
    fake_files = list_audio_files(FAKE_DIR)
    print(f"Found {len(real_files)} real files, {len(fake_files)} fake files.")

    print("\nPre-scanning all files for loadability (independent of any embedding cache)...")
    candidate_files = real_files + fake_files
    candidate_labels = [0] * len(real_files) + [1] * len(fake_files)

    clean_files, clean_labels = [], []
    excluded_files = []
    for f, label in zip(candidate_files, candidate_labels):
        if is_loadable(f):
            clean_files.append(f)
            clean_labels.append(label)
        else:
            excluded_files.append(f)

    print(f"Excluded {len(excluded_files)} unloadable/corrupted files out of {len(candidate_files)} candidates:")
    for f in excluded_files:
        print(f"  excluded (unloadable): {f}")

    all_files = clean_files
    all_labels = clean_labels
    print(f"Clean population: {len(all_files)} files ({sum(1 for l in all_labels if l == 0)} real, {sum(1 for l in all_labels if l == 1)} fake).")

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

    print(f"Baseline evaluated on {len(y_true_base)} test files (out of {len(test_files)}).")

    baseline_metrics = {
        'accuracy': accuracy_score(y_true_base, y_pred_base),
        'precision': precision_score(y_true_base, y_pred_base, zero_division=0),
        'recall': recall_score(y_true_base, y_pred_base, zero_division=0),
        'f1': f1_score(y_true_base, y_pred_base, zero_division=0),
    }
    print(f"Baseline metrics: {baseline_metrics}")

    # ── Windowed: every window of every train file is its own example ────
    print("\n=== Windowed (whole-clip, max-aggregated) ===")
    print("Extracting windowed train embeddings (reuses existing .win{i}.npy caches where present)...")
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

    print(f"Windowed evaluated on {len(y_true_win)} test files (out of {len(test_files)}).")

    if len(y_true_base) != len(y_true_win):
        print(f"WARNING: evaluated population sizes differ! baseline={len(y_true_base)} vs windowed={len(y_true_win)}. "
              f"This indicates a bug — the pre-scan was supposed to make both populations identical.")

    windowed_metrics = {
        'accuracy': accuracy_score(y_true_win, y_pred_win),
        'precision': precision_score(y_true_win, y_pred_win, zero_division=0),
        'recall': recall_score(y_true_win, y_pred_win, zero_division=0),
        'f1': f1_score(y_true_win, y_pred_win, zero_division=0),
    }
    print(f"Windowed metrics: {windowed_metrics}")

    # ── Report ─────────────────────────────────────────────────────────
    winner = "windowed" if windowed_metrics['accuracy'] >= baseline_metrics['accuracy'] else "baseline"
    population_note = (
        f"Both approaches were evaluated on the same {len(y_true_base)} test files."
        if len(y_true_base) == len(y_true_win) else
        f"**WARNING: evaluated population sizes differ (baseline={len(y_true_base)}, windowed={len(y_true_win)}) — investigate before trusting this comparison.**"
    )
    report = f"""# Deepfake Detector: Baseline vs Windowed Evaluation

Date: 2026-09-03

Train/test split: 80/20, stratified by class, file-level, random_state={RANDOM_STATE}.

Pre-scan: {len(excluded_files)} of {len(candidate_files)} candidate files were excluded as
unloadable/corrupted (checked with a fresh `librosa.load()`, independent of any
embedding cache) BEFORE the train/test split, so both approaches are evaluated
on an identical population of loadable files by construction, not by accident
of which embedding cache happened to already exist.

Clean population after pre-scan: {len(all_files)} files ({sum(1 for l in all_labels if l == 0)} real, {sum(1 for l in all_labels if l == 1)} fake).
Train files: {len(train_files)}. Test files: {len(test_files)}.

{population_note}
Baseline evaluated on {len(y_true_base)} test files. Windowed evaluated on {len(y_true_win)} test files.

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
