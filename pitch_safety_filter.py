"""
pitch_safety_filter.py — Pure pitch-based safety override for the primary
gender model. Only ever pushes a 'female' verdict toward 'male' (reject) or
'manual_review' — never produces an accept on its own. No audio or model
dependencies, so it's cheap to run and easy to test in isolation.
"""


def apply_pitch_safety_filter(label: str, confidence: float, meanfun_hz: float, meanfreq_hz: float) -> tuple:
    """
    label: 'male' or 'female' (primary model's verdict).
    confidence: 0.0-1.0.
    meanfun_hz, meanfreq_hz: acoustic pitch features in Hz.

    Returns (final_label, final_confidence). final_label is 'male',
    'female', or 'manual_review'.
    """
    if label != 'female':
        return label, confidence

    if meanfun_hz < 130.0 or meanfreq_hz < 130.0:
        return 'male', 0.999

    if meanfun_hz < 170.0 or meanfreq_hz < 160.0 or confidence < 0.85:
        return 'manual_review', confidence

    if meanfun_hz > 270.0 and meanfreq_hz < 230.0:
        return 'manual_review', confidence

    return label, confidence
