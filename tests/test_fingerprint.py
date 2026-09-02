import os
import pytest
import soundfile as sf
import librosa

from fingerprint import compute_fingerprint, hamming_distance, is_match

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..')
FEMALE_FIXTURE = os.path.join(FIXTURE_DIR, 'female_test.wav')
HUMAN_FIXTURE = os.path.join(FIXTURE_DIR, 'test_human.wav')


def test_identical_audio_produces_identical_fingerprint():
    fp1 = compute_fingerprint(FEMALE_FIXTURE)
    fp2 = compute_fingerprint(FEMALE_FIXTURE)
    assert fp1 == fp2
    assert hamming_distance(fp1, fp2) == 0


def test_trimmed_copy_still_matches(tmp_path):
    y, sr = librosa.load(FEMALE_FIXTURE, sr=16000, mono=True)
    trimmed = y[int(0.5 * sr):-int(0.5 * sr)]  # drop 0.5s off each end
    trimmed_path = os.path.join(str(tmp_path), 'trimmed.wav')
    sf.write(trimmed_path, trimmed, sr)

    original_fp = compute_fingerprint(FEMALE_FIXTURE)
    trimmed_fp = compute_fingerprint(trimmed_path)

    assert is_match(original_fp, trimmed_fp)


def test_resampled_copy_still_matches(tmp_path):
    y, sr = librosa.load(FEMALE_FIXTURE, sr=16000, mono=True)
    resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)
    resampled_path = os.path.join(str(tmp_path), 'resampled.wav')
    sf.write(resampled_path, resampled, 22050)

    original_fp = compute_fingerprint(FEMALE_FIXTURE)
    resampled_fp = compute_fingerprint(resampled_path)

    assert is_match(original_fp, resampled_fp)


def test_different_recordings_do_not_match():
    fp1 = compute_fingerprint(FEMALE_FIXTURE)
    fp2 = compute_fingerprint(HUMAN_FIXTURE)
    assert not is_match(fp1, fp2)


def test_hamming_distance_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        hamming_distance(b'\x00', b'\x00\x00')
