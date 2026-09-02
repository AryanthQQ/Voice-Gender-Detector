"""
gender_verifier.py — Primary Wav2Vec2-based gender classifier.

Loaded eagerly at server startup (see main.py) since it now runs on every
/predict and /predict-url request as the primary decision-maker, rather
than lazily on a subset of requests as it did when it was only a secondary
corroboration check.
"""
import threading

MODEL_NAME = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"

_load_lock = threading.Lock()
_processor = None
_model = None
_device = None


def load_model():
    """Loads the model if not already loaded. Idempotent — safe to call more than once."""
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


def is_loaded() -> bool:
    return _model is not None


def classify_gender(audio_path: str) -> dict:
    """Runs the primary gender model on audio_path.
    Returns {'label': 'male'|'female', 'confidence': 0-100}."""
    import torch
    import librosa

    load_model()
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
