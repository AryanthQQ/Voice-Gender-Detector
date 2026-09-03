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
