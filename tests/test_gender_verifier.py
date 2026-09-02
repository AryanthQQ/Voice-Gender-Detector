import os
import gender_verifier

FIXTURE = os.path.join(os.path.dirname(__file__), '..', 'female_test.wav')


def test_load_model_is_idempotent_and_sets_loaded_flag():
    gender_verifier.load_model()
    assert gender_verifier.is_loaded() is True
    gender_verifier.load_model()  # calling again must not error or reload
    assert gender_verifier.is_loaded() is True


def test_classify_gender_returns_valid_schema():
    gender_verifier.load_model()
    result = gender_verifier.classify_gender(FIXTURE)
    assert set(result.keys()) == {'label', 'confidence'}
    assert result['label'] in ('male', 'female')
    assert 0.0 <= result['confidence'] <= 100.0
