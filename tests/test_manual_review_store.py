import os
from manual_review_store import init_db, add_pending_review, list_pending_reviews


def test_list_pending_reviews_empty_on_fresh_db(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    assert list_pending_reviews(db_path) == []


def test_add_and_list_round_trip(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    add_pending_review('advisor-1', 'Alice', 'https://s3.example.com/audio1.wav', 'Ambiguous voice', db_path)

    reviews = list_pending_reviews(db_path)
    assert len(reviews) == 1
    assert reviews[0]['advisor_id'] == 'advisor-1'
    assert reviews[0]['advisor_name'] == 'Alice'
    assert reviews[0]['source_url'] == 'https://s3.example.com/audio1.wav'
    assert reviews[0]['reason'] == 'Ambiguous voice'
    assert 'created_at' in reviews[0]


def test_list_returns_most_recent_first(tmp_path):
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    add_pending_review('advisor-1', 'Alice', 'url1', 'reason1', db_path)
    add_pending_review('advisor-2', 'Bob', 'url2', 'reason2', db_path)

    reviews = list_pending_reviews(db_path)
    assert len(reviews) == 2
    assert reviews[0]['advisor_id'] == 'advisor-2'
    assert reviews[1]['advisor_id'] == 'advisor-1'
