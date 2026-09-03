import os
import numpy as np
import librosa

from deepfake_detector_v2 import _split_into_windows

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..')


def test_short_clip_produces_one_window():
    windows = _split_into_windows(np.zeros(16000))  # 1 second of silence
    assert len(windows) == 1
    assert len(windows[0]) == 16000


def test_exact_multiple_produces_correct_window_count():
    windows = _split_into_windows(np.zeros(16000 * 9))  # exactly 9 seconds = 3 windows
    assert len(windows) == 3
    assert all(len(w) == 16000 * 3 for w in windows)


def test_trailing_remainder_kept_if_substantial():
    windows = _split_into_windows(np.zeros(16000 * 4))  # 3s window + 1s remainder (>= 0.5s min)
    assert len(windows) == 2
    assert len(windows[0]) == 16000 * 3
    assert len(windows[1]) == 16000 * 1


def test_tiny_trailing_remainder_dropped():
    windows = _split_into_windows(np.zeros(16000 * 3 + 4000))  # 3s window + 0.25s remainder (< 0.5s min)
    assert len(windows) == 1
    assert len(windows[0]) == 16000 * 3


def test_real_fixture_female_test_wav_produces_two_windows():
    # female_test.wav is ~6.104s: two full 3s windows, then a ~0.104s
    # remainder that's below the 0.5s minimum and gets dropped.
    y, sr = librosa.load(os.path.join(FIXTURE_DIR, 'female_test.wav'), sr=16000)
    windows = _split_into_windows(y)
    assert len(windows) == 2
    assert all(len(w) == 16000 * 3 for w in windows)


def test_real_fixture_test_sine_wav_produces_one_window():
    # test_sine.wav is ~1s, well under the 3s window size.
    y, sr = librosa.load(os.path.join(FIXTURE_DIR, 'test_sine.wav'), sr=16000)
    windows = _split_into_windows(y)
    assert len(windows) == 1
    assert len(windows[0]) == len(y)
