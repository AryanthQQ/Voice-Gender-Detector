# Replay-Attack Detection — Design

Date: 2026-09-02
Status: Approved

## Problem

`/predict-url` accepts an `advisor_id` alongside the audio URL. Nothing
today stops the same pre-recorded (genuine, human, female) audio clip from
being submitted under multiple different `advisor_id`s — one real female
voice recording could be shared and reused to pass the gender check for
several different (likely male) advisor identities. This is a fraud vector
distinct from the gender-classification accuracy work already shipped in
the primary-gender-model-upgrade project.

This is the first of three security-hardening sub-projects identified
during that project's review (the other two — pitch-shift/voice-changer
detection, and improving the deepfake/AI-cloned-voice detector — are
separate, later specs).

## Goal

Detect when a submitted audio clip (or a minor re-encoded/trimmed variant
of it) has already been submitted under a *different* `advisor_id`, and
escalate that submission to `manual_review` instead of letting the normal
gender-decision pipeline auto-accept or auto-reject it.

## Non-goals

- **Speaker verification** (detecting the same real person's voice across
  *different* recordings) is explicitly out of scope. This project only
  detects reuse of the same underlying audio clip (byte-identical or a
  re-encoded/trimmed copy of it) — not "is this the same person talking."
  Speaker verification is a materially different technique (voice
  embeddings / speaker diarization models) and, if wanted later, is its
  own separate project.
- **`/predict` (manual file upload) is not covered.** That endpoint has no
  `advisor_id` — "reuse across different advisor IDs" doesn't apply to it.
  Only `/predict-url` is in scope.
- Pitch-shift/voice-changer detection and deepfake-detector improvements
  are separate, later specs — not addressed here.
- No UI changes. This is a backend-only detection + escalation change.

## Approach: perceptual audio fingerprinting

A lightweight perceptual hash of each processed clip's audio content,
robust to minor re-encoding, resampling, or trimming (unlike a plain
SHA-256 of the file bytes, which a trivial re-encode would defeat):

1. Load the audio at 16kHz mono (already the standard sample rate used
   everywhere else in this codebase).
2. Compute a mel-spectrogram, split into a small number of fixed-length
   time segments.
3. For each segment, average energy per mel band, then binarize each band
   against its own median across segments (1 bit per band per segment).
4. Concatenate into a single fixed-length bit vector — the fingerprint.

Two fingerprints are considered a match if their Hamming distance (number
of differing bits) is at or below a threshold. Because the fingerprint is
built from coarse, time-averaged spectral energy rather than exact
waveform bytes, small transcoding artifacts or a few seconds trimmed off
either end change only a few bits, not the whole fingerprint.

The exact segment count, mel-band count, and Hamming-distance threshold
are implementation-time tuning parameters, validated against real audio
(see Testing below) — not locked by this spec. Document whatever values
are chosen and why, the same way the primary-gender-model project
documented its 85% cutoff as a starting point to revisit with real traffic.

No new third-party dependency is introduced — this uses `librosa` and
`numpy`, both already dependencies of this project.

## Storage

A new SQLite database, `data/fingerprints.db` (stdlib `sqlite3`, no new
dependency). One table:

```sql
CREATE TABLE audio_fingerprints (
    id INTEGER PRIMARY KEY,
    fingerprint BLOB NOT NULL,
    advisor_id TEXT NOT NULL,
    advisor_name TEXT NOT NULL,
    decision TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

Only the fingerprint is stored — never the raw audio — consistent with the
existing privacy design (accepted/rejected audio is deleted immediately;
only `manual_review` audio is retained, and only for
`MANUAL_REVIEW_RETENTION_DAYS`). Fingerprints are retained indefinitely:
they're small, one-way (can't reconstruct the audio from them), and the
whole point is to catch reuse that could happen long after the original
submission.

## Pipeline placement (in `_predict_url_sync`)

Insert the fingerprint check right after the existing STT/audio-quality
gates pass and before the expensive ML calls — i.e., between the STT
"unintelligible audio" check and the current

```python
# ── 4. Extract features + predict ─────────────────────────────────────
with GLOBAL_PROCESS_LOCK:
    ai_result = advanced_deepfake_detector.predict(tmp_path)
    ...
    features = extract_features(tmp_path)
    result = predict_gender(tmp_path, features)
```

block in `main.py`. Concretely:

1. Compute the fingerprint for `tmp_path`.
2. Query `audio_fingerprints` for any row with a matching fingerprint
   (within the Hamming-distance threshold) **and a different
   `advisor_id`** than the current request. Same-`advisor_id` matches
   (e.g. a legitimate retry) are ignored — not a fraud signal.
3. **If a cross-advisor match is found:** skip the deepfake-detector and
   gender-model calls entirely (both are expensive — this both saves
   compute and directly enforces the policy regardless of what those
   models would have said). Escalate straight to `manual_review`, with a
   reason string identifying which other `advisor_id` the clip was
   previously seen under (for the admin reviewing the case), keep the
   audio for manual review the same way a deepfake-flagged clip already
   is (`_keep_for_manual_review`), and send the existing email
   notification path.
4. **If no cross-advisor match is found:** proceed with the existing flow
   unchanged (deepfake detector, feature extraction, gender prediction).
   After a final decision is reached, insert the new fingerprint into
   `audio_fingerprints` with the resulting `advisor_id`, `advisor_name`,
   `decision`, and current timestamp — so future submissions (under any
   advisor_id, including this one) can be checked against it.

## API compatibility

`/predict-url`'s response shape for a duplicate-triggered `manual_review`
matches the existing `manual_review` response shape already used for a
deepfake-flagged clip (same `n8n_result` fields: `decision: 'uncertain'`,
`status: 1`, etc.) — no new top-level response fields required. The
`reason` string differs ("Duplicate audio detected — previously submitted
under a different advisor ID" vs the deepfake reason), which is already a
free-text field consumers don't parse structurally.

## Testing

- **True positive:** take a real audio fixture, re-encode it (e.g.
  wav → mp3 → wav round-trip) and/or trim a couple of seconds off one end,
  and confirm the fingerprint of the modified copy is within the match
  threshold of the original's fingerprint.
- **True negative:** confirm two genuinely different recordings (e.g. the
  existing `female_test.wav` and `test_human.wav` fixtures) do NOT match.
- **Cross-advisor escalation:** integration test simulating two
  `/predict-url` calls with the same audio under two different
  `advisor_id`s — first call proceeds normally and gets stored, second
  call is escalated to `manual_review` without invoking the deepfake
  detector or gender model (mock/patch those and assert they're not
  called).
- **Same-advisor no-op:** two calls with the same audio and the *same*
  `advisor_id` — second call should NOT be escalated on that basis alone.
