# /predict-url: No Local Audio Retention — Design

Date: 2026-09-04
Status: Approved

## Problem

`/predict-url` accepts audio via an S3 (or any HTTP) URL, downloads it to
process, then — for any `manual_review` outcome (deepfake-flagged, replay-
attack duplicate, or ambiguous gender verdict) — moves that downloaded copy
into `MANUAL_REVIEW_DIR` so a human can review it, retained for
`MANUAL_REVIEW_RETENTION_DAYS` before auto-purge.

Since today's security-hardening work (classical-ensemble corroboration,
replay-attack detection) increased how often results land in
`manual_review`, this local retention has become more frequent. The
original audio already lives at a durable URL the caller provided (already
included in the email notification and the API response as `source_url`) —
there is no need for this service to also keep its own local copy on disk.

## Goal

For `/predict-url` specifically: never write a persistent local copy of
audio to disk, for any outcome (accept, reject, or manual_review). Replace
local retention, for the manual_review case, with a lightweight metadata
record (advisor info + the original source URL + the reason) so a human
reviewer can still find and open the original recording via its S3 link.

## Non-goals

- `/predict` (direct browser/file upload) is **unaffected** — it has no
  external URL to fall back on, so its existing local
  retention-for-manual-review behavior via `_keep_for_manual_review` /
  `MANUAL_REVIEW_DIR` stays exactly as it is today.
- The existing `/recordings` endpoint (which lists `/predict`'s locally
  retained files) is not touched.
- No change to the email notification's content — it already includes the
  `source_url` (S3 link) for manual_review cases; that keeps working
  unchanged.
- No change to accept/reject behavior on `/predict-url` — audio for those
  outcomes is already deleted immediately today; that's unaffected.

## Approach

**New module `manual_review_store.py`** (mirrors the existing
`fingerprint_store.py` pattern — a small SQLite-backed store, no new
third-party dependency):

- `init_db()`
- `add_pending_review(advisor_id: str, advisor_name: str, source_url: str, reason: str) -> None`
- `list_pending_reviews() -> list[dict]`

Schema: `id, advisor_id, advisor_name, source_url, reason, created_at`. No
audio, ever — just enough metadata for a human to find and open the
original recording via its `source_url`.

**In `_predict_url_sync` (`main.py`)**, all three existing call sites of
`_keep_for_manual_review(tmp_path)` — the replay-attack-duplicate branch,
the deepfake-flagged branch, and the final ambiguous-gender-verdict branch
— are replaced with:

1. `manual_review_store.add_pending_review(advisor_id, advisor_name, audio_url, reason_str)`
2. Immediate deletion of the downloaded audio (`_delete_audio(tmp_path)`),
   the same way accept/reject outcomes already delete it — no local file
   is ever moved into `MANUAL_REVIEW_DIR` for this endpoint anymore.

`_keep_for_manual_review` itself, `MANUAL_REVIEW_DIR`, and the existing
`/recordings` endpoint are **not removed** — they're still needed by
`/predict`.

**New admin endpoint** `GET /api/admin/pending-url-reviews`
(API-key-protected, matching the existing `/api/admin/*` naming
convention) returns `manual_review_store.list_pending_reviews()` — the
admin-facing "queue" for `/predict-url`-originated manual reviews, showing
S3 links instead of locally hosted audio.

## API compatibility

No change to `/predict-url`'s request or response shape. The email
notification's content is unchanged (it already carries `source_url`).
Only the internal retention mechanism changes, plus one new admin-only
endpoint.

## Testing

- Unit tests for `manual_review_store.py`: `add_pending_review` +
  `list_pending_reviews` round-trip correctly, using a temp SQLite file
  per test (matching `fingerprint_store.py`'s existing test pattern).
- Integration tests in `tests/test_main_integration.py`: for each of the
  three manual_review trigger paths on `/predict-url` (mocking the
  relevant upstream signal — a cross-advisor fingerprint match, an
  `is_ai` deepfake flag, and a low-confidence/ambiguous gender verdict),
  assert `_keep_for_manual_review`/`MANUAL_REVIEW_DIR` is NOT touched
  (no file appears there) and that `manual_review_store.add_pending_review`
  IS called with the right arguments.
- A test confirming `/predict`'s existing manual_review behavior
  (`_keep_for_manual_review`, `MANUAL_REVIEW_DIR`) is completely unchanged
  by this plan — this is the hard boundary the whole design depends on.
