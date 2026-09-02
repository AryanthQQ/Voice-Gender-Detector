from pitch_safety_filter import apply_pitch_safety_filter


def test_male_verdict_passes_through_unchanged():
    label, conf = apply_pitch_safety_filter('male', 0.92, meanfun_hz=110.0, meanfreq_hz=115.0)
    assert label == 'male'
    assert conf == 0.92


def test_female_verdict_with_clear_female_pitch_and_high_confidence_passes():
    label, conf = apply_pitch_safety_filter('female', 0.95, meanfun_hz=210.0, meanfreq_hz=200.0)
    assert label == 'female'
    assert conf == 0.95


def test_female_verdict_with_male_range_pitch_is_overridden_to_male():
    label, conf = apply_pitch_safety_filter('female', 0.90, meanfun_hz=120.0, meanfreq_hz=125.0)
    assert label == 'male'
    assert conf == 0.999


def test_female_verdict_with_borderline_pitch_escalates_to_manual_review():
    label, conf = apply_pitch_safety_filter('female', 0.90, meanfun_hz=150.0, meanfreq_hz=200.0)
    assert label == 'manual_review'
    assert conf == 0.90


def test_female_verdict_with_low_confidence_escalates_to_manual_review():
    label, conf = apply_pitch_safety_filter('female', 0.70, meanfun_hz=210.0, meanfreq_hz=200.0)
    assert label == 'manual_review'
    assert conf == 0.70


def test_female_verdict_with_falsetto_pattern_escalates_to_manual_review():
    label, conf = apply_pitch_safety_filter('female', 0.95, meanfun_hz=280.0, meanfreq_hz=220.0)
    assert label == 'manual_review'
    assert conf == 0.95


def test_confidence_exactly_at_085_cutoff_passes():
    label, conf = apply_pitch_safety_filter('female', 0.85, meanfun_hz=210.0, meanfreq_hz=200.0)
    assert label == 'female'
    assert conf == 0.85


def test_confidence_just_below_085_cutoff_escalates():
    label, conf = apply_pitch_safety_filter('female', 0.8499, meanfun_hz=210.0, meanfreq_hz=200.0)
    assert label == 'manual_review'
    assert conf == 0.8499
