"""
classical_corroborator.py — Secondary classical-ML corroboration check for
'female' gender verdicts, using the SVM/GBM/RF ensemble that used to be the
primary decision-maker before the primary-gender-model-upgrade project
retired it in favor of Wav2Vec2-XLSR. Models are unchanged, still on disk.

Only invoked when the primary model + pitch-safety-filter have already
decided 'female' (i.e. about to auto-accept) — if this disagrees, the
caller escalates to manual_review instead of accepting. Never used to
force a reject or an accept on its own.
"""
import os

import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

_svm_model = None
_gbm_model = None
_rf_model = None
_scaler = None
_feature_order = None


def load_models():
    """Loads the classical models if not already loaded. Idempotent.
    If loading fails partway through, resets all five globals back to
    None rather than leaving a partial load in place — otherwise
    is_loaded() would report True (since it only checked _svm_model)
    while corroborate() silently crashes on a missing model, and the
    fail-closed exception handler in main.py would turn every female
    verdict into manual_review indefinitely with no health signal
    indicating anything was wrong."""
    global _svm_model, _gbm_model, _rf_model, _scaler, _feature_order
    if is_loaded():
        return
    try:
        _svm_model     = joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl'))
        _gbm_model     = joblib.load(os.path.join(MODELS_DIR, 'gbm_model.pkl'))
        _rf_model      = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
        _rf_model.n_jobs = 1
        _scaler        = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        _feature_order = joblib.load(os.path.join(MODELS_DIR, 'features.pkl'))
    except Exception:
        _svm_model = _gbm_model = _rf_model = _scaler = _feature_order = None
        raise


def is_loaded() -> bool:
    return all(x is not None for x in (_svm_model, _gbm_model, _rf_model, _scaler, _feature_order))


def corroborate(features: dict) -> dict:
    """Runs the classical SVM/GBM/RF ensemble on an already-extracted
    feature dict (the same dict extract_features() produces in main.py).
    Returns {'svm': {label, confidence}, 'gbm': {...}, 'rf': {...},
    'male_votes': int}."""
    load_models()

    feat_vec = np.array([[features[f] for f in _feature_order]])
    feat_scaled = _scaler.transform(feat_vec)

    # svm_model.pkl is a full sklearn Pipeline (StandardScaler + SVC) that
    # scales internally, so it correctly takes the raw feat_vec — passing
    # it feat_scaled would double-scale. GBM and RF were actually trained
    # on UNSCALED features (scaler.pkl was fit afterward and never used to
    # train them), so feeding them feat_scaled here is out-of-distribution
    # relative to their training data — this is preserved, not fixed, to
    # match the pre-retirement ensemble's exact behavior; switching GBM/RF
    # to unscaled input is a follow-up that needs its own validation on a
    # broader dataset (including female audio), not a change to make here.
    svm_prob = _svm_model.predict_proba(feat_vec)[0]
    gbm_prob = _gbm_model.predict_proba(feat_scaled)[0]
    rf_prob  = _rf_model.predict_proba(feat_scaled)[0]

    svm_pred = int(np.argmax(svm_prob))
    gbm_pred = int(np.argmax(gbm_prob))
    rf_pred  = int(np.argmax(rf_prob))

    def as_result(pred, prob):
        return {'label': 'male' if pred == 1 else 'female', 'confidence': float(max(prob)) * 100}

    male_votes = sum([svm_pred, gbm_pred, rf_pred])

    return {
        'svm': as_result(svm_pred, svm_prob),
        'gbm': as_result(gbm_pred, gbm_prob),
        'rf':  as_result(rf_pred, rf_prob),
        'male_votes': male_votes,
    }


def should_escalate(corroboration: dict) -> bool:
    """True if the classical ensemble disagrees strongly enough with a
    'female' primary verdict to escalate to manual_review instead of
    auto-accepting. Escalation-only signal — never forces a reject or an
    accept on its own.

    Rule: majority (2 or 3 of 3) vote 'male', OR any single model votes
    'male' at >=90% confidence. Starting-point thresholds, not validated
    at scale (derived from an 8-file batch) — revisit with real traffic."""
    if corroboration['male_votes'] >= 2:
        return True
    for key in ('svm', 'gbm', 'rf'):
        model = corroboration[key]
        if model['label'] == 'male' and model['confidence'] >= 90.0:
            return True
    return False
