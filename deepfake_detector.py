import os
import torch
import librosa
import numpy as np
from transformers import pipeline

# We will use a pre-trained model for deepfake audio detection.
# You can change the MODEL_NAME to a different Hugging Face model if needed.
MODEL_NAME = "Mihaiii/wav2vec2-base-deepfake-detection"

# Initialize the pipeline globally so it only loads once
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        print(f"[INFO] Loading Deepfake Detection Model ({MODEL_NAME}). This might take a moment...")
        try:
            device = 0 if torch.cuda.is_available() else -1
            _detector = pipeline(
                "audio-classification", 
                model=MODEL_NAME, 
                device=device
            )
            print("[INFO] Deepfake Detection Model loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load Deepfake Detection Model: {e}")
            _detector = False # Mark as failed so we don't keep trying
    return _detector

def detect_replay_attack(y, sr) -> dict:
    """
    Heuristic detection for Replay Attack (audio played from a speaker).
    Calculates the ratio of high frequency (4-8kHz) energy to low frequency (0-4kHz) energy.
    """
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        freqs = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=8000)
        
        low_band = (freqs >= 0) & (freqs < 4000)
        high_band = (freqs >= 4000) & (freqs <= 8000)
        
        low_energy = np.mean(S[low_band, :])
        high_energy = np.mean(S[high_band, :])
        
        if low_energy < 1e-10:
            return {"is_replay": False, "ratio": 0.0}
            
        ratio = high_energy / low_energy
        
        # Strict threshold: if high-frequency energy is less than 5% of low-freq, 
        # it's likely a speaker playback. (Increased from 1% for better detection)
        is_replay = ratio < 0.05
        
        return {"is_replay": is_replay, "ratio": ratio}
    except Exception as e:
        print(f"[ERROR] Replay detection failed: {e}")
        return {"is_replay": False, "ratio": 0.0}

def predict_is_ai(audio_path: str) -> dict:
    """
    Predicts whether the given audio file is AI-generated, Replay Attack, or real human voice.
    Returns:
        dict: {
            "is_ai": bool,
            "confidence": float,
            "reason": str,
            "status": str
        }
    """
    detector = get_detector()
    if not detector:
        return {
            "is_ai": False, 
            "confidence": 0.0, 
            "reason": "",
            "status": "model_error"
        }

    try:
        # Load audio at 16kHz as expected by wav2vec2
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Pipeline expects numpy array or path
        predictions = detector(y)
        
        # The model usually outputs 'fake' or 'real' / 'spoof' or 'bonafide'
        # Let's map it safely and aggressively.
        is_ai = False
        ai_conf = 0.0
        
        for pred in predictions:
            label = pred['label'].lower()
            score = pred['score']
            
            # Common labels for fake audio: 'fake', 'spoof', 'ai', 'synthetic'
            if label in ['fake', 'spoof', 'ai', 'synthetic']:
                ai_conf = score
                # STRICT THRESHOLD: If the AI model thinks there's >= 35% chance 
                # of it being fake, we immediately flag it as AI.
                if score >= 0.35:
                    is_ai = True
                break

        # --- REPLAY ATTACK CHECK ---
        replay_result = detect_replay_attack(y, sr)
        
        reason = ""
        if is_ai:
            reason = f"AI/Synthetic voice detected ({round(ai_conf * 100, 1)}%)"
        elif replay_result["is_replay"]:
            # Generic rejection message for replay attack to hide defense mechanism
            is_ai = True  # Flag as invalid/spoof
            ai_conf = 0.99
            reason = "Invalid audio quality (Playback/Replay detected)"
            print(f"[REPLAY] Replay attack detected! HF ratio: {replay_result['ratio']:.5f}")

        return {
            "is_ai": is_ai,
            "confidence": round(float(ai_conf) * 100, 2),
            "reason": reason,
            "status": "success"
        }
        
    except Exception as e:
        print(f"[ERROR] Deepfake detection failed for {audio_path}: {e}")
        return {
            "is_ai": False,
            "confidence": 0.0,
            "reason": "",
            "status": "processing_error"
        }
