import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

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
    with patch('gender_verifier.classify_gender', return_value={'label': 'female', 'confidence': 85.1}):
        result = main.predict_gender('/fake/path/does-not-matter.wav', fake_features)
    assert result['ensemble']['label'] == 'female'
    assert result['decision'] == 'accept'
    assert result['svm']['label'] == 'female'
    assert result['gbm']['label'] == 'female'
    assert result['rf']['label'] == 'female'
