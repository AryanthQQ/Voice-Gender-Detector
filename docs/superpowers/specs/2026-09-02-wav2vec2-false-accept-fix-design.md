# Wav2Vec2 False-Accept Fix (Classical Ensemble Corroboration) — Design

Date: 2026-09-02
Status: Approved

## Problem

Manual batch testing of 27 real male-voice audio files against the live
`/predict` endpoint (using the Wav2Vec2-XLSR primary model shipped in the
primary-gender-model-upgrade project) found:

- Of the 8 files that reached full gender classification (the other 19
  were filtered out earlier by audio-quality/STT gates), the raw
  Wav2Vec2-XLSR model misclassified **4 of 8 (50%)** as `female`.
- The pitch-safety-filter caught 3 of those 4 misclassifications (pushing
  them to `male`/`manual_review`), but **1 of 8 slipped through as a
  false `accept`** — a real male voice, classified `female` at 99.9%
  confidence, auto-accepted. The speaker's pitch (`meanfun_hz: 225.4`,
  `meanfreq_hz: 173.5`) is high enough that it doesn't trip any of the
  pitch-safety-filter's hard-override or manual-review thresholds.

To investigate, the same 8 files were run through the **retired classical
SVM/GBM/RF ensemble** (still present on disk in `models/*.pkl`, unused
since the primary-gender-model-upgrade project retired it). Results:

- SVM alone was correct (`male`) on all 8/8 files, including the one the
  primary model got wrong.
- GBM was inconsistent (flip-flopped `female`/`male` across the batch).
- RF was correct (`male`) on all 8/8, though often at lower confidence
  (53-61%).

This is a small sample, but it demonstrates the classical ensemble is not
uniformly worse than the new primary model on this specific advisor
population — it is a genuinely useful, currently-unused second signal
sitting right there on disk.

## Goal

Add the classical SVM/GBM/RF ensemble back as a **secondary corroboration
check**, structurally mirroring the corroboration pattern the codebase
already used before the primary-gender-model-upgrade project (where a
secondary model double-checked a `female` verdict before letting it
auto-accept) — except with the roles reversed: Wav2Vec2-XLSR stays
primary, the classical ensemble becomes the secondary check.

## Non-goals

- No retraining of the classical models — reuse the existing `models/*.pkl`
  files as-is.
- No changes to the pitch-safety-filter's own logic or thresholds — this
  is an additional, independent layer, not a replacement.
- No changes to `/predict`'s or `/predict-url`'s request/response JSON
  shape. `svm`/`gbm`/`rf`/`ensemble` continue to all mirror the same
  final decision, exactly as the primary-gender-model-upgrade project
  established — this corroboration check is internal plumbing, not
  something the API surfaces separately.
- No changes to the replay-attack-detection feature (a separate,
  already-complete project) — this is unrelated.

## Approach

**New module: `classical_corroborator.py`** (mirrors the structure of
`gender_verifier.py` / `pitch_safety_filter.py` from earlier projects):

- `load_models()` — eagerly loads `svm_model.pkl`, `gbm_model.pkl`,
  `rf_model.pkl`, `scaler.pkl`, `features.pkl` from `models/`, same
  loading code the app used before the classical ensemble was retired.
  Eager (not lazy) because these are lightweight sklearn models with no
  GPU/heavy-download cost — the original app loaded them eagerly too.
- `is_loaded() -> bool`.
- `corroborate(features: dict) -> dict` — runs all 3 classical models on
  the already-extracted feature vector (the same `features` dict
  `extract_features()` already produces for the pitch-safety-filter and
  the UI's frequency display — no new feature extraction needed) and
  returns each model's `{label, confidence}` plus a `male_votes` count.

**Escalation rule, wired into `predict_gender()` in `main.py`:**

Corroboration only runs when the pitch-safety-filter's output is
`female` — i.e., only on requests that are about to auto-accept. (Exactly
mirroring the old `_corroborate_female`'s skip condition: no extra model
cost on requests that were already going to reject or already going to
manual_review.)

If the classical ensemble's `male_votes >= 2` (a majority disagrees), OR
any single classical model votes `male` at confidence `>= 90%`, override
the final decision from `female`/`accept` to `manual_review`/`uncertain`.
Never let corroboration force a `reject` or force an `accept` — only ever
escalates toward a human decision, consistent with the project's
male-reject policy (always favor reject/manual_review over auto-accept on
ambiguity).

The 90%-single-model threshold and the "majority" (2-of-3) rule are
starting points, not empirically validated at scale (the evidence base is
an 8-file batch) — document them as tunable, to revisit once more
production traffic accumulates, the same way prior projects' thresholds
were documented.

## Scope

Applies to **both** `/predict` and `/predict-url` — this addresses
auto-accept correctness, not advisor-identity fraud (unlike the separate
replay-attack-detection feature), so both endpoints that can produce an
`accept` decision need the same protection.

## Testing

- Unit tests for `classical_corroborator.py`: verify `corroborate()`
  correctly runs all 3 models on a known feature vector and returns the
  right shape and vote count.
- A test using the real classical models against the actual audio file
  that produced the original false-accept
  (`C:\Users\hp\Desktop\male voice\1779778933389-47662627.mp3`) — this
  file is not part of the repo's test fixtures, so this test should be
  written to skip gracefully (not fail) if that path doesn't exist on
  the machine running the suite, since it's a personal debugging fixture
  outside version control, not a project asset.
- Integration test(s) in `tests/test_main_integration.py`: mock
  `gender_verifier.classify_gender` to return a confident `female` verdict
  with female-range pitch (so the pitch-filter would normally let it
  through), and mock `classical_corroborator.corroborate` to return a
  majority-disagrees result — assert the final decision is
  `manual_review`/`uncertain`, not `accept`. And the inverse: mock
  corroboration to agree (`male_votes` low, all confidences low) — assert
  the original `accept` decision still goes through unchanged.
