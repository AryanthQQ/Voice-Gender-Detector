import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import main

FIXTURE = os.path.join(os.path.dirname(__file__), '..', 'female_test.wav')


@pytest.fixture(scope="session")
def client():
    return TestClient(main.app)


def test_health_reports_primary_gender_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["models_loaded"] is True
    assert data["detailed_model_status"]["primary_gender_model"] is True
    assert "svm_ensemble" not in data["detailed_model_status"]


def test_predict_returns_backward_compatible_response_shape(client):
    with open(FIXTURE, "rb") as f:
        resp = client.post(
            "/predict",
            files={"file": ("female_test.wav", f, "audio/wav")},
            data={"advisor_name": "Test Advisor"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "ensemble" in data
    for key in ("svm", "gbm", "rf"):
        assert data[key]["label"] == data["ensemble"]["label"]
        assert data[key]["confidence"] == data["ensemble"]["confidence"]
    assert data["decision"] in ("accept", "reject", "uncertain")


def test_predict_gender_manual_review_at_84_9_percent_confidence():
    fake_features = {'meanfun': 0.21, 'meanfreq': 0.20, 'IQR': 0.05}  # 210Hz/200Hz - clearly female pitch range
    with patch('gender_verifier.classify_gender', return_value={'label': 'female', 'confidence': 84.9}):
        result = main.predict_gender('/fake/path/does-not-matter.wav', fake_features)
    assert result['ensemble']['label'] == 'manual_review'
    assert result['decision'] == 'uncertain'


def test_predict_gender_accepts_at_85_1_percent_confidence():
    fake_features = {'meanfun': 0.21, 'meanfreq': 0.20, 'IQR': 0.05}  # 210Hz/200Hz - clearly female pitch range
    with patch('gender_verifier.classify_gender', return_value={'label': 'female', 'confidence': 85.1}), \
         patch('classical_corroborator.corroborate', return_value={
             'svm': {'label': 'female', 'confidence': 91.0},
             'gbm': {'label': 'female', 'confidence': 88.0},
             'rf':  {'label': 'female', 'confidence': 93.0},
             'male_votes': 0,
         }):
        result = main.predict_gender('/fake/path/does-not-matter.wav', fake_features)
    assert result['ensemble']['label'] == 'female'
    assert result['decision'] == 'accept'
    assert result['svm']['label'] == 'female'
    assert result['gbm']['label'] == 'female'
    assert result['rf']['label'] == 'female'


def test_predict_gender_escalates_when_classical_ensemble_disagrees():
    fake_features = {'meanfun': 0.21, 'meanfreq': 0.20, 'IQR': 0.05}  # clearly-female pitch range
    with patch('gender_verifier.classify_gender', return_value={'label': 'female', 'confidence': 99.0}), \
         patch('classical_corroborator.corroborate', return_value={
             'svm': {'label': 'male', 'confidence': 90.0},
             'gbm': {'label': 'male', 'confidence': 85.0},
             'rf':  {'label': 'female', 'confidence': 55.0},
             'male_votes': 2,
         }):
        result = main.predict_gender('/fake/path/does-not-matter.wav', fake_features)
    assert result['ensemble']['label'] == 'manual_review'
    assert result['decision'] == 'uncertain'


def test_predict_gender_keeps_accept_when_classical_ensemble_agrees():
    fake_features = {'meanfun': 0.21, 'meanfreq': 0.20, 'IQR': 0.05}
    with patch('gender_verifier.classify_gender', return_value={'label': 'female', 'confidence': 99.0}), \
         patch('classical_corroborator.corroborate', return_value={
             'svm': {'label': 'female', 'confidence': 88.0},
             'gbm': {'label': 'female', 'confidence': 82.0},
             'rf':  {'label': 'male', 'confidence': 60.0},
             'male_votes': 1,
         }):
        result = main.predict_gender('/fake/path/does-not-matter.wav', fake_features)
    assert result['ensemble']['label'] == 'female'
    assert result['decision'] == 'accept'


def _read_fixture_bytes():
    # NOTE: main.py's /predict-url handler has a pre-existing, unrelated
    # "IMMEDIATE VOLUME CHECK" gate (from the primary-gender-model-upgrade
    # plan) that rejects audio with max amplitude < 0.20 before the request
    # ever reaches the STT step / replay-attack check added by this task.
    # FIXTURE (female_test.wav) peaks at ~0.143, which trips that gate and
    # short-circuits the request before any of the mocks below are relevant.
    # We boost the gain (content/speech unchanged) so these tests actually
    # exercise the replay-attack code path under test, rather than muting
    # the gate itself or mocking away real STT processing.
    import io
    import soundfile as sf
    import numpy as np

    y, sr = sf.read(FIXTURE, dtype='float32')
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y = y * (0.6 / peak)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def test_predict_url_escalates_on_cross_advisor_duplicate(client):
    audio_bytes = _read_fixture_bytes()
    mock_response = MagicMock()
    mock_response.read.return_value = audio_bytes
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False

    headers = {"X-API-Key": main.config.API_KEY}

    with patch('main._assert_public_url'), \
         patch('main.urllib.request.urlopen', return_value=mock_response), \
         patch('main.advanced_deepfake_detector.predict') as mock_deepfake, \
         patch('gender_verifier.classify_gender') as mock_classify, \
         patch('fingerprint.compute_fingerprint', return_value=b'\x00' * 32), \
         patch('fingerprint_store.find_cross_advisor_match', return_value={'advisor_id': 'advisor-A', 'advisor_name': 'Advisor A'}), \
         patch('fingerprint_store.store_fingerprint') as mock_store:

        resp = client.post(
            "/predict-url",
            headers=headers,
            json={"url": "http://test.local/clip-duplicate.wav", "userId": "advisor-B", "fullname": "Advisor B"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data['decision'] == 'uncertain'
    assert data['reason'] == 'This audio matches a previously submitted recording under a different advisor. Sent for manual review.'
    mock_deepfake.assert_not_called()
    mock_classify.assert_not_called()
    mock_store.assert_not_called()


def test_predict_url_no_duplicate_proceeds_normally_and_stores_fingerprint(client):
    audio_bytes = _read_fixture_bytes()
    mock_response = MagicMock()
    mock_response.read.return_value = audio_bytes
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False

    headers = {"X-API-Key": main.config.API_KEY}

    with patch('main._assert_public_url'), \
         patch('main.urllib.request.urlopen', return_value=mock_response), \
         patch('main.advanced_deepfake_detector.predict', return_value={'is_ai': False, 'confidence': 0.0, 'reason': 'Real Human Voice', 'status': 'success'}), \
         patch('gender_verifier.classify_gender', return_value={'label': 'male', 'confidence': 99.0}), \
         patch('fingerprint.compute_fingerprint', return_value=b'\x11' * 32), \
         patch('fingerprint_store.find_cross_advisor_match', return_value=None), \
         patch('fingerprint_store.store_fingerprint') as mock_store:

        resp = client.post(
            "/predict-url",
            headers=headers,
            json={"url": "http://test.local/clip-new.wav", "userId": "advisor-C", "fullname": "Advisor C"},
        )

    assert resp.status_code == 200
    mock_store.assert_called_once()
    call_args = mock_store.call_args[0]
    assert call_args[1] == 'advisor-C'


def test_predict_gender_real_corroboration_on_female_range_audio_stays_accepted():
    """Runs the REAL classical ensemble (not mocked) against real features
    extracted from a genuine female-pitch-range audio file, to confirm
    corroboration doesn't itself cause an unwanted escalation on
    legitimate content. Only the primary model is mocked (to force a
    'female' verdict deterministically) — everything else is real."""
    fixture_path = os.path.join(os.path.dirname(__file__), '..', 'test_human.mp3')
    y, sr = main.safe_load_audio(fixture_path, sr=16000)
    tmp_wav = os.path.join(os.environ.get('TEMP', '.'), 'female_range_corroboration_test.wav')
    import soundfile as sf
    sf.write(tmp_wav, y, 16000)
    real_features = main.extract_features(tmp_wav)

    with patch('gender_verifier.classify_gender', return_value={'label': 'female', 'confidence': 99.0}):
        result = main.predict_gender(tmp_wav, real_features)

    assert result['decision'] == 'accept', (
        f"Real classical-ensemble corroboration unexpectedly escalated a "
        f"female-range voice; result: {result}"
    )
