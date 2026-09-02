import os
import pytest
from unittest.mock import patch

from classical_corroborator import load_models, is_loaded, corroborate, should_escalate

# Real feature vectors extracted from actual audio, used to exercise corroborate()
# against the real loaded models without needing extract_features() here.
MALE_LIKE_FEATURES = {
    'meanfreq': 0.173496531827012, 'sd': 0.06, 'median': 0.17, 'Q25': 0.13, 'Q75': 0.20,
    'IQR': 0.03125, 'skew': 0.5, 'kurt': 3.0, 'sp.ent': 0.9, 'sfm': 0.05, 'mode': 0.17,
    'centroid': 0.17, 'meanfun': 0.22541925340157143, 'minfun': 0.08, 'maxfun': 0.28,
    'meandom': 0.15, 'mindom': 0.05, 'maxdom': 0.25, 'dfrange': 0.2, 'modindx': 0.1,
}


def test_load_models_is_idempotent_and_sets_loaded_flag():
    load_models()
    assert is_loaded() is True
    load_models()  # calling again must not error
    assert is_loaded() is True


def test_corroborate_returns_valid_schema():
    load_models()
    result = corroborate(MALE_LIKE_FEATURES)
    assert set(result.keys()) == {'svm', 'gbm', 'rf', 'male_votes'}
    for key in ('svm', 'gbm', 'rf'):
        assert set(result[key].keys()) == {'label', 'confidence'}
        assert result[key]['label'] in ('male', 'female')
        assert 0.0 <= result[key]['confidence'] <= 100.0
    assert 0 <= result['male_votes'] <= 3


def test_should_escalate_true_when_majority_disagrees():
    corroboration = {
        'svm': {'label': 'male', 'confidence': 80.0},
        'gbm': {'label': 'male', 'confidence': 75.0},
        'rf':  {'label': 'female', 'confidence': 55.0},
        'male_votes': 2,
    }
    assert should_escalate(corroboration) is True


def test_should_escalate_true_on_single_high_confidence_disagreement():
    corroboration = {
        'svm': {'label': 'male', 'confidence': 95.0},
        'gbm': {'label': 'female', 'confidence': 60.0},
        'rf':  {'label': 'female', 'confidence': 55.0},
        'male_votes': 1,
    }
    assert should_escalate(corroboration) is True


def test_should_escalate_false_when_ensemble_agrees_female():
    corroboration = {
        'svm': {'label': 'female', 'confidence': 85.0},
        'gbm': {'label': 'female', 'confidence': 80.0},
        'rf':  {'label': 'male', 'confidence': 60.0},  # single low-confidence dissent, not enough
        'male_votes': 1,
    }
    assert should_escalate(corroboration) is False


def test_corroborate_gets_the_original_false_accept_case_right():
    """Regression test for the specific audio file that caused a real
    Wav2Vec2 false-accept in production testing. Skips gracefully if that
    personal debugging fixture isn't present on this machine — it's not a
    project asset."""
    audio_path = r"C:\Users\hp\Desktop\male voice\1779778933389-47662627.mp3"
    if not os.path.exists(audio_path):
        pytest.skip("Personal debugging fixture not present on this machine")

    import main  # heavy import, needed for extract_features + safe_load_audio
    import soundfile as sf
    y, sr = main.safe_load_audio(audio_path, sr=16000)
    tmp_wav = os.path.join(os.environ.get('TEMP', '.'), 'corroborator_regression_test.wav')
    sf.write(tmp_wav, y, 16000)
    features = main.extract_features(tmp_wav)

    load_models()
    result = corroborate(features)
    assert should_escalate(result) is True, (
        f"Classical ensemble should have flagged this known false-accept case, got: {result}"
    )


def test_should_escalate_true_at_exactly_90_percent_confidence():
    corroboration = {
        'svm': {'label': 'male', 'confidence': 90.0},
        'gbm': {'label': 'female', 'confidence': 60.0},
        'rf':  {'label': 'female', 'confidence': 55.0},
        'male_votes': 1,
    }
    assert should_escalate(corroboration) is True


def test_should_escalate_false_just_below_90_percent_confidence():
    corroboration = {
        'svm': {'label': 'male', 'confidence': 89.9},
        'gbm': {'label': 'female', 'confidence': 60.0},
        'rf':  {'label': 'female', 'confidence': 55.0},
        'male_votes': 1,
    }
    assert should_escalate(corroboration) is False


def test_partial_load_failure_resets_cleanly_and_can_be_retried():
    import classical_corroborator as cc
    # Save real state to restore after this test
    saved = (cc._svm_model, cc._gbm_model, cc._rf_model, cc._scaler, cc._feature_order)
    cc._svm_model = cc._gbm_model = cc._rf_model = cc._scaler = cc._feature_order = None

    real_load = cc.joblib.load
    call_count = [0]
    def flaky_load(path):
        call_count[0] += 1
        if call_count[0] == 3:  # fail on rf_model.pkl (3rd call)
            raise IOError("simulated corrupt file")
        return real_load(path)

    try:
        with patch('classical_corroborator.joblib.load', side_effect=flaky_load):
            with pytest.raises(IOError):
                cc.load_models()
        assert cc.is_loaded() is False  # must not report loaded after a partial failure

        # Retry should work now that the mock is gone
        cc.load_models()
        assert cc.is_loaded() is True
    finally:
        cc._svm_model, cc._gbm_model, cc._rf_model, cc._scaler, cc._feature_order = saved
