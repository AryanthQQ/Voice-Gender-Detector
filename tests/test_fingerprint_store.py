import os
from fingerprint_store import init_db, find_cross_advisor_match, store_fingerprint


def test_no_match_on_empty_db(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    result = find_cross_advisor_match(b'\x00' * 32, 'advisor-1', db_path)
    assert result is None


def test_finds_match_from_different_advisor(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    fp = bytes([0b10101010] * 32)
    store_fingerprint(fp, 'advisor-1', 'Alice', 'female', db_path)

    match = find_cross_advisor_match(fp, 'advisor-2', db_path)
    assert match is not None
    assert match['advisor_id'] == 'advisor-1'
    assert match['advisor_name'] == 'Alice'


def test_ignores_match_from_same_advisor(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    fp = bytes([0b10101010] * 32)
    store_fingerprint(fp, 'advisor-1', 'Alice', 'female', db_path)

    match = find_cross_advisor_match(fp, 'advisor-1', db_path)
    assert match is None


def test_near_duplicate_within_threshold_still_matches(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    fp = bytes([0b10101010] * 32)
    near_duplicate = bytes([0b10101010] * 31 + [0b10101011])  # only the last byte differs, by 1 bit -> Hamming distance 1
    store_fingerprint(fp, 'advisor-1', 'Alice', 'female', db_path)

    match = find_cross_advisor_match(near_duplicate, 'advisor-2', db_path)
    assert match is not None
