"""
conftest.py — Runs before any test module is collected. Some modules
(deepfake_detector_v2.py) import torch at module level with no
protection against the Windows numba/torch OpenMP crash that main.py's
own import sequence carefully avoids (see the comment at main.py's
librosa.pyin warmup call). Performing that same warmup here, before
pytest collects any test file, guarantees correct ordering no matter
which test module happens to get imported first alphabetically.
"""
import numpy as np
import librosa

librosa.pyin(np.zeros(16000 * 2, dtype=np.float32), fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=16000)
