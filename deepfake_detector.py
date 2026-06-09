import os
import torch
import librosa
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

def predict_is_ai(audio_path: str) -> dict:
    """
    Predicts whether the given audio file is AI-generated or real human voice.
    Returns:
        dict: {
            "is_ai": bool,
            "confidence": float,
            "status": str
        }
    """
    detector = get_detector()
    if not detector:
        return {
            "is_ai": False, 
            "confidence": 0.0, 
            "status": "model_error"
        }

    try:
        # Load audio at 16kHz as expected by wav2vec2
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Pipeline expects numpy array or path
        predictions = detector(y)
        
        # The model usually outputs 'fake' or 'real' / 'spoof' or 'bonafide'
        # Let's map it safely.
        is_ai = False
        ai_conf = 0.0
        
        for pred in predictions:
            label = pred['label'].lower()
            score = pred['score']
            
            # Common labels for fake audio: 'fake', 'spoof', 'ai'
            if label in ['fake', 'spoof', 'ai', 'synthetic']:
                if score > ai_conf:
                    ai_conf = score
                    is_ai = True
            elif label in ['real', 'bonafide', 'human']:
                if score > ai_conf:
                    # If highest confidence is 'real', then is_ai remains False
                    pass
        
        # If the model has different label names, we just pick the top one
        top_pred = predictions[0]
        if top_pred['label'].lower() in ['fake', 'spoof', 'ai', 'synthetic']:
            is_ai = True
            ai_conf = top_pred['score']
        elif top_pred['label'].lower() in ['real', 'bonafide', 'human']:
            is_ai = False
            ai_conf = top_pred['score']
        
        return {
            "is_ai": is_ai,
            "confidence": round(float(ai_conf) * 100, 2),
            "status": "success"
        }
        
    except Exception as e:
        print(f"[ERROR] Deepfake detection failed for {audio_path}: {e}")
        return {
            "is_ai": False,
            "confidence": 0.0,
            "status": "processing_error"
        }
