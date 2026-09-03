# Deepfake Detector: Baseline vs Windowed Evaluation

Date: 2026-09-03

Train/test split: 80/20, stratified by class, file-level, random_state=42.
Train files: 1457. Test files: 365.

## Baseline (current 3-second-truncation approach)

| Metric | Value |
|---|---|
| Accuracy | 0.9449 |
| Precision | 0.9429 |
| Recall | 0.9747 |
| F1 | 0.9585 |

## Windowed (whole-clip, max-aggregated approach)

| Metric | Value |
|---|---|
| Accuracy | 0.9499 |
| Precision | 0.9498 |
| Recall | 0.9742 |
| F1 | 0.9619 |

## Decision

Per the project's decision rule (ship windowed only if its test accuracy
is >= baseline's test accuracy): **windowed** wins.
