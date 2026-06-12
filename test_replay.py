import librosa
import numpy as np
import glob
import sys

def analyze(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=8000)
    
    # Phone speakers have very weak bass (< 300 Hz)
    bass_band = (freqs >= 0) & (freqs < 300)
    mid_band = (freqs >= 300) & (freqs < 3000)
    high_band = (freqs >= 3000) & (freqs <= 8000)
    
    bass_energy = np.mean(S[bass_band, :])
    mid_energy = np.mean(S[mid_band, :])
    high_energy = np.mean(S[high_band, :])
    
    print(f"Path: {path}")
    print(f"Bass/Mid ratio: {bass_energy / (mid_energy + 1e-10):.4f}")
    print(f"High/Mid ratio: {high_energy / (mid_energy + 1e-10):.4f}")

for f in glob.glob("recordings/*.wav"):
    analyze(f)
