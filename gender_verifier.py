"""
gender_verifier.py — Secondary Wav2Vec2-based gender check.

Used only to corroborate a 'female' verdict from the primary SVM/GBM/RF
ensemble before auto-accepting. Deliberately NOT loaded at startup — it is
the heaviest model in the pipeline and is only needed for a subset of
requests (those the primary ensemble already called female), so it loads
lazily on first use and stays cached in the process afterward.
"""
import threading

MODEL_NAME = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"

_load_lock = threading.Lock()
_processor = None
_model = None


_device = None


def _ensure_loaded():
    global _processor, _model, _device
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(_device)
        model.eval()
        _processor, _model = processor, model


def verify_female(audio_path: str) -> dict:
    """Runs the secondary model on audio_path. Returns {'label': 'male'|'female', 'confidence': 0-100}."""
    import torch
    import librosa

    _ensure_loaded()
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt").to(_device)
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax())
    return {
        'label': _model.config.id2label[idx],
        'confidence': round(float(probs[idx]) * 100, 1),
    }
