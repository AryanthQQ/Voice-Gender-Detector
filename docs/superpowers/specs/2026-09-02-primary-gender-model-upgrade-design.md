# Primary Gender Model Upgrade — Design

Date: 2026-09-02
Status: Approved

## Problem

The current primary gender decision is made by a classical ML ensemble (SVM +
Gradient Boosting + Random Forest) trained on 20 hand-crafted acoustic
features (`extract_features` in `main.py`), on a small (3,168-sample),
homogeneous training set (the 2016-era "Gender Recognition by Voice" Kaggle
dataset). This approach:

- Generalizes poorly to diverse accents, mic quality, and background noise.
- Uses brittle, dataset-tuned hardcoded thresholds (pitch ranges, spectral
  flatness cutoffs).
- Picks a final label via "whichever of the 3 models is most confident,"
  which is a statistically weak ensembling strategy.

The codebase already has a modern, pretrained deep model
(`wav2vec2-large-xlsr-53-gender-recognition-librispeech`, wired up in
`gender_verifier.py`) but it is only used as a secondary check to corroborate
"female" verdicts from the weak primary ensemble — never as the primary
decision-maker.

## Goal

Make the existing pretrained Wav2Vec2-XLSR model the **primary** gender
decision-maker. Retire the classical SVM/GBM/RF ensemble entirely from the
decision path. No new training or data collection in this project — that is
a separate, later initiative.

## Non-goals (deferred to later projects)

- Collecting new training data / training a custom model.
- Anti-spoofing hardening (pitch-shift/voice-changer detection, deepfake
  detection improvements, replay-attack detection). This was raised during
  design discussion but is explicitly out of scope here — it becomes its own
  spec/plan cycle immediately after this project ships.
- Any frontend redesign. The "Model Breakdown" UI will keep showing three
  values (now identical, backward-compat filled) — a known cosmetic gap, not
  fixed in this project.

## Architecture

Decision pipeline for both `/predict` and `/predict-url`, in order:

1. **Audio quality gate (unchanged)** — silence, too-short, and
   background-noise checks currently inside `extract_features` are pure
   signal-processing and don't depend on the classical models. They stay
   exactly as-is.
2. **Primary ML decision (new)** — the Wav2Vec2-XLSR model
   (`gender_verifier.py`) runs directly on the audio file and returns
   `{label, confidence}` (male or female).
3. **Pitch hard-safety filter (unchanged logic)** — activates only when the
   primary decision is "female":
   - `meanfun_hz < 130` or `meanfreq_hz < 130` → override to `male`
     (confidently male pitch range).
   - `meanfun_hz < 170` or `meanfreq_hz < 160` or confidence `< 85%` →
     escalate to `manual_review`.
   - `meanfun_hz > 270` and `meanfreq_hz < 230` → escalate to
     `manual_review` (falsetto/child heuristic).
   - This filter never forces an accept — it can only push toward
     `reject`/`manual_review`, consistent with the project's male-reject
     bias (favor reject/manual_review over auto-accept on ambiguity).

The existing `_corroborate_female` secondary-check step is **removed**. It
previously added value by running a *different* model (Wav2Vec2-XLSR) than
the primary ensemble as a second opinion. Once Wav2Vec2-XLSR *is* the
primary model, re-running it again on the same audio produces the same
result — no new signal, pure wasted compute.

## Components changed

- **`gender_verifier.py`**
  - Loading changes from lazy (on first "female" verdict) to eager: loaded
    once at server startup, like the Whisper/deepfake models.
  - Generalized from a female-corroboration-only helper to a general
    primary classifier (still returns `{label, confidence}` for either
    male or female — the underlying model already supports this; only the
    call site's intent changes).
- **`main.py`**
  - Remove `svm_model` / `gbm_model` / `rf_model` / `scaler` loading
    (`joblib.load` block).
  - Remove `_corroborate_female` and its call sites in `_predict_sync` /
    `_predict_url_sync`.
  - Replace `predict_gender(features)` with a new function that takes the
    audio path (for the primary model) plus the extracted features (for the
    pitch filter and UI frequency display), and returns the same response
    dict shape as before.
  - `/health` endpoint's `detailed_model_status.svm_ensemble` field is
    replaced with a status flag for the new primary model.

## API compatibility (hard requirement)

Both **request** and **response** shapes for `/predict` and `/predict-url`
stay exactly as they are today:

- Request bodies are unchanged (no fields added/removed).
- Response keeps `ensemble`, `svm`, `gbm`, `rf`, `features`, and `decision`.
  `ensemble` carries the new model's real result. `svm`/`gbm`/`rf` are all
  filled with the same `{label, confidence}` as `ensemble` (backward-compat
  values, not independent models anymore).
- This is required so the existing n8n/agent integration (calling
  `/predict-url` on port 8003) keeps working with zero changes on its side.

## Confidence threshold

The existing 85% manual-review cutoff is kept as the starting point (no new
calibration data available yet). To be revisited once real production
traffic / the planned data-collection project gives us a basis to tune it.

## Testing

- Unit-level: run the new primary-decision function against the existing
  local test fixtures (`female_test.wav`, `female_test.ogg`,
  `test_human.wav`, `test_human.mp3`, `test_sine.wav`) and confirm sane
  outputs (no exceptions, expected accept/reject/manual_review shape).
- Integration: `/predict` (file upload) and `/predict-url` (URL fetch) both
  exercised end-to-end via the running server on port 8003, same way the
  port-migration fix was verified earlier — confirm response shape matches
  the documented contract above.
- Manual browser check of the existing UI (`static/index.html`) to confirm
  the upload → analyze flow still renders a result (accepting the known
  cosmetic "Model Breakdown" gap).
