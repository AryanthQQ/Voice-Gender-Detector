import os
import torch
import librosa
import numpy as np
from transformers import pipeline

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Set DEEPFAKE_CHECK_ENABLED=false in .env to completely disable AI detection
DEEPFAKE_CHECK_ENABLED = os.environ.get("DEEPFAKE_CHECK_ENABLED", "true").lower() != "false"

MODEL_NAME = "MelodyMachine/Deepfake-audio-detection-V2"

# Minimum confidence required to flag audio as AI-generated.
# Range: 0.0 - 1.0. Higher = less aggressive (fewer false positives).
# 0.95 = only flag if model is 95%+ sure it's fake.
AI_CONFIDENCE_THRESHOLD = float(os.environ.get("AI_CONFIDENCE_THRESHOLD", "0.95"))

# Initialize the pipeline globally so it only loads once
_detector = None


def get_detector():
    global _detector
    if not DEEPFAKE_CHECK_ENABLED:
        return None
    if _detector is None:
        print(f"[INFO] Loading Deepfake Detection Model ({MODEL_NAME})...")
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
            _detector = False  # Mark as failed so we don't keep trying
    return _detector


def predict_is_ai(audio_path: str) -> dict:
    """
    Predicts whether the given audio file is AI-generated.

    Returns:
        dict: {
            "is_ai": bool,
            "confidence": float,
            "reason": str,
            "status": str
        }
    """
    # ── SKIP if disabled ──────────────────────────────────────────────────────
    if not DEEPFAKE_CHECK_ENABLED:
        print("[DEEPFAKE] Check disabled via config. Passing all audio.")
        return {"is_ai": False, "confidence": 0.0, "reason": "", "status": "disabled"}

    detector = get_detector()
    if not detector:
        # If model failed to load, don't block users — pass them through
        print("[DEEPFAKE] Model not available. Passing audio through.")
        return {"is_ai": False, "confidence": 0.0, "reason": "", "status": "model_error"}

    try:
        # Load audio at 16kHz as expected by wav2vec2
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Run model
        predictions = detector(y)

        # ── DEBUG: print all raw predictions so we can tune labels/thresholds ─
        print(f"[DEEPFAKE] Raw model output for '{os.path.basename(audio_path)}':")
        for p in predictions:
            print(f"           label='{p['label']}'  score={p['score']:.4f}")

        # ── Map labels to fake/real ──────────────────────────────────────────
        # Known fake labels from various models:
        FAKE_LABELS = {'fake', 'spoof', 'ai', 'synthetic', 'deepfake', 'generated'}
        # Known real labels:
        REAL_LABELS = {'real', 'genuine', 'human', 'bonafide', 'natural', 'authentic'}

        is_ai = False
        ai_conf = 0.0
        top_label = predictions[0]['label'].lower() if predictions else ""
        top_score = predictions[0]['score'] if predictions else 0.0

        for pred in predictions:
            label = pred['label'].lower()
            score = pred['score']

            if label in FAKE_LABELS:
                ai_conf = score
                if score >= AI_CONFIDENCE_THRESHOLD:
                    is_ai = True
                    print(f"[DEEPFAKE] ⚠️  FAKE detected! label='{label}' conf={score:.3f} (threshold={AI_CONFIDENCE_THRESHOLD})")
                else:
                    print(f"[DEEPFAKE] ✅ Below threshold. label='{label}' conf={score:.3f} < {AI_CONFIDENCE_THRESHOLD} → PASS")
                break
            elif label in REAL_LABELS:
                # Model is confident it's real
                print(f"[DEEPFAKE] ✅ Real voice. label='{label}' conf={score:.3f} → PASS")
                break

        # If model returned unknown labels, assume real (pass-through)
        if top_label not in FAKE_LABELS and top_label not in REAL_LABELS:
            print(f"[DEEPFAKE] ⚠️  Unknown label '{top_label}' — treating as REAL to avoid false positives.")

        reason = f"AI/Synthetic voice detected ({round(ai_conf * 100, 1)}%)" if is_ai else ""

        return {
            "is_ai": is_ai,
            "confidence": round(float(ai_conf) * 100, 2),
            "reason": reason,
            "status": "success"
        }

    except Exception as e:
        print(f"[ERROR] Deepfake detection failed for {audio_path}: {e}")
        # On error, pass through — don't block users due to our bug
        return {
            "is_ai": False,
            "confidence": 0.0,
            "reason": "",
            "status": "processing_error"
        }
