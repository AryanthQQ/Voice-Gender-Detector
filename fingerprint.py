"""
fingerprint.py — Perceptual audio fingerprinting for replay-attack detection.

Produces a fixed-length binary fingerprint from an audio file's mel-spectrogram,
robust to minor re-encoding, resampling, or trimming (unlike a byte-level hash
of the file, which any of those would defeat). Two fingerprints of the same
underlying audio should differ by only a few bits; two genuinely different
recordings should differ by many more.
"""
import librosa
import numpy as np

# NUM_SEGMENTS lowered from 8 to 4: at 8 segments a 0.5s trim off each end of the
# ~6s female_test.wav fixture shifted segment boundaries enough to flip 64/256 bits
# (already over the 40-bit tuning ceiling), while genuinely different audio only
# differed by 108/256 bits — too thin a margin to raise the threshold safely.
# At 4 segments the same trim only flips 10/128 bits (coarser segments are less
# sensitive to boundary shift) while different audio still differs by 28/128 bits,
# giving a clean separation to threshold in between.
NUM_SEGMENTS = 4
NUM_MEL_BANDS = 32
# MATCH_THRESHOLD changed from 20 to 16 (now out of NUM_SEGMENTS * NUM_MEL_BANDS =
# 128 bits total, since NUM_SEGMENTS dropped to 4): measured 10 bits for the
# trimmed-copy case and 28 bits for the genuinely-different-recording case, so 16
# sits with comfortable margin above the true-positive distance and below the
# true-negative distance.
MATCH_THRESHOLD = 16


def compute_fingerprint(audio_path: str) -> bytes:
    """Computes a perceptual audio fingerprint for audio_path.
    Returns a fixed-length bytes object (NUM_SEGMENTS * NUM_MEL_BANDS bits, packed)."""
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=NUM_MEL_BANDS)
    log_mel = librosa.power_to_db(mel)

    num_frames = log_mel.shape[1]
    bounds = np.linspace(0, num_frames, NUM_SEGMENTS + 1, dtype=int)
    segment_means = np.array([
        log_mel[:, bounds[i]:bounds[i + 1]].mean(axis=1) if bounds[i + 1] > bounds[i]
        else np.zeros(NUM_MEL_BANDS)
        for i in range(NUM_SEGMENTS)
    ])  # shape: (NUM_SEGMENTS, NUM_MEL_BANDS)

    band_medians = np.median(segment_means, axis=0)
    bits = (segment_means > band_medians).astype(np.uint8).flatten()

    return np.packbits(bits).tobytes()


def hamming_distance(fp1: bytes, fp2: bytes) -> int:
    """Number of differing bits between two same-length fingerprints."""
    if len(fp1) != len(fp2):
        raise ValueError("Fingerprints must be the same length to compare.")
    xor_bytes = bytes(a ^ b for a, b in zip(fp1, fp2))
    return sum(bin(byte).count('1') for byte in xor_bytes)


def is_match(fp1: bytes, fp2: bytes) -> bool:
    """True if two fingerprints are within MATCH_THRESHOLD Hamming distance."""
    return hamming_distance(fp1, fp2) <= MATCH_THRESHOLD
