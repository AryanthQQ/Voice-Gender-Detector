# Deepfake Detector: Baseline vs Windowed Evaluation

Date: 2026-09-03

Train/test split: 80/20, stratified by class, file-level, random_state=42.

Pre-scan: 25 of 1822 candidate files were excluded as
unloadable/corrupted (checked with a fresh `librosa.load()`, independent of any
embedding cache) BEFORE the train/test split, so both approaches are evaluated
on an identical population of loadable files by construction, not by accident
of which embedding cache happened to already exist.

Clean population after pre-scan: 1797 files (628 real, 1169 fake).
Train files: 1437. Test files: 360.

Both approaches were evaluated on the same 360 test files.
Baseline evaluated on 360 test files. Windowed evaluated on 360 test files.

## Baseline (current 3-second-truncation approach)

| Metric | Value |
|---|---|
| Accuracy | 0.9583 |
| Precision | 0.9469 |
| Recall | 0.9915 |
| F1 | 0.9687 |

## Windowed (whole-clip, max-aggregated approach)

| Metric | Value |
|---|---|
| Accuracy | 0.9528 |
| Precision | 0.9540 |
| Recall | 0.9744 |
| F1 | 0.9641 |

## Decision

Per the project's decision rule (ship windowed only if its test accuracy
is >= baseline's test accuracy): **baseline** wins.
