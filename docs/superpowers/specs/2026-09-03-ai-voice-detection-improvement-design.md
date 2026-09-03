# AI-Voice-Detection Improvement — Design

Date: 2026-09-03
Status: Approved

## Problem

The current AI/deepfake voice detector (`deepfake_detector_v2.py`,
`AdvancedDeepfakeDetector`) truncates every audio clip to its **first 3
seconds** before extracting a Wav2Vec2-base-960h embedding and classifying
it with XGBoost. This is the third and last of the three security-hardening
sub-projects identified after the primary-gender-model-upgrade project
(replay-attack detection and the Wav2Vec2 false-accept fix are already
complete).

Investigating the existing training data (`data/real/`: 629 audio files,
`data/fake/`: 1193 audio files — a real, substantial dataset, not something
that needs to be collected from scratch) revealed a second problem beyond
the 3-second truncation: **fake samples are almost all ~3 seconds long
(2.3-4.1s), while real samples vary widely (2-19.7s, averaging 6.5s)**. If
a "use the whole clip" fix were implemented naively (single embedding over
the entire clip regardless of length), the model could learn clip *length*
as a shortcut for the real/fake distinction instead of actual voice
authenticity signals — a spurious correlation that would look good on
paper but fail to generalize to real fraud attempts (which won't
conveniently be short).

There is also currently no honest accuracy measurement for the deployed
model: `deepfake_detector_v2.py`'s `train_on_your_data()` fits on 100% of
`data/real` + `data/fake` with no held-out test set, so no one knows the
model's real generalization performance versus its training-set fit.

## Goal

1. Establish a genuine accuracy baseline for the current (3-second
   truncation) approach on held-out data it was never trained on.
2. Build a "use the whole clip" alternative that avoids the length-shortcut
   risk, by extracting embeddings over fixed-length (3-second) windows
   covering the entire clip rather than only the first 3 seconds, with
   every training example — regardless of source clip length — being a
   uniform ~3-second window.
3. Compare both approaches honestly on the same held-out test split, and
   ship whichever one actually performs better — not assume the more
   sophisticated approach automatically wins.

## Non-goals

- No new training data collection — reuse the existing `data/real`/`data/fake`
  files as-is.
- No change to the underlying embedding model (`facebook/wav2vec2-base-960h`)
  or classifier type (XGBoost) — this project changes how audio is windowed
  and aggregated, not the model architecture itself. A full purpose-built
  anti-spoofing architecture (e.g. AASIST-style) was considered and
  explicitly deferred as a larger, separate future effort.
- No changes to `main.py`'s call site (`advanced_deepfake_detector.predict(path)`)
  — whichever approach wins stays behind that same method signature, so no
  other code needs to change.
- No changes to the gender-classification pipeline, replay-attack
  detection, or any other already-completed project.

## Approach

### Windowed embedding extraction

Add a windowed extraction path to `AdvancedDeepfakeDetector` alongside the
existing single-embedding path (kept, not removed, since it's needed for
the baseline comparison):

- Split the loaded audio into non-overlapping 3-second (48,000-sample)
  windows. A clip shorter than 3 seconds produces exactly one window (its
  full, shorter length) — this matches the current single-embedding
  behavior for such clips, no change there. A trailing remainder shorter
  than 3 seconds is kept as its own (shorter) window rather than being
  dropped or padded, as long as it's non-trivial in length.
- Extract a Wav2Vec2 embedding per window (same mean-pooling as today, just
  applied per-window instead of once per clip).
- Cache each window's embedding to its own file (a different filename
  pattern than today's single-embedding cache, so the two caching schemes
  never collide or silently reuse stale data from one another).

### Training with windowed data

Every window from every training file becomes its own training example,
labeled with its parent file's class (real/fake). This is what removes the
length-shortcut risk: whether a training example came from a 3-second fake
clip or a 19-second real clip, every example the classifier actually sees
is a uniform ~3-second window — there is no systematic length difference
between the real-labeled and fake-labeled training examples anymore.

### Inference with windowed data

At prediction time, extract all of a clip's window embeddings, score each
window independently with the classifier, and take the **maximum**
fake-probability across windows as the clip's final score (any one
fake-sounding segment is enough to flag the whole clip — consistent with
this project's established "favor catching over missing" posture in its
other safety features).

### Honest evaluation, not an assumed win

Split `data/real` + `data/fake` into an 80/20 train/test split **at the
file level** (never split a single file's windows across train and test —
that would leak information), stratified by class, with a fixed random
seed for reproducibility. Train two models on the *same* train split:

- **Baseline**: today's exact approach (3-second truncation, one embedding
  per file).
- **Windowed**: the new approach described above.

Evaluate both on the *same* held-out test split (aggregating window
predictions to a file-level score for the windowed model) and report
accuracy, precision, recall, and F1 for each.

**Decision rule:** ship the windowed approach only if its test-set accuracy
is greater than or equal to the baseline's test-set accuracy. Otherwise,
keep the current 3-second-truncation approach in production — the
evaluation and its numbers still get committed as a documented finding
either way, since establishing the real baseline has value regardless of
which approach wins.

Whichever approach wins, retrain its final production model on the **full**
dataset (100% of `data/real` + `data/fake`, not just the 80% train split)
before shipping — the train/test split exists to validate the *methodology*
honestly, not to withhold 20% of real data from the model that actually
goes into production.

## Testing

- Unit tests for the windowing arithmetic itself (how many windows a given
  audio length produces, using existing short repo fixtures like
  `test_sine.wav` and `female_test.wav` — no need for real/fake training
  data for this, it's pure audio-splitting logic).
- The evaluation step itself (train both models, measure both on the same
  held-out test set) is the substantive "test" of this project's central
  question — its successful completion with real, reported numbers is a
  deliverable in its own right, not something separately unit-tested.
- After the winning approach's final production model is retrained and
  shipped, a smoke test confirming `AdvancedDeepfakeDetector.predict()`
  still returns the same response shape (`is_ai`, `confidence`,
  `probability_ai`, `reason`, `status`) main.py already depends on.
