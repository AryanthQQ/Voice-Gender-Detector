import os
from manual_review_store import init_db, add_pending_review, list_pending_reviews, purge_expired_reviews


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


def test_purge_expired_reviews_deletes_old_rows_only(tmp_path):
    import sqlite3
    db_path = os.path.join(str(tmp_path), 'test.db')
    init_db(db_path)
    add_pending_review('advisor-old', 'Old', 'url-old', 'reason-old', db_path)
    add_pending_review('advisor-new', 'New', 'url-new', 'reason-new', db_path)

    # Backdate the first row's created_at to 10 days ago, directly via SQL
    # (bypassing add_pending_review, which always stamps "now").
    from datetime import datetime, timedelta
    old_timestamp = (datetime.now() - timedelta(days=10)).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE pending_reviews SET created_at = ? WHERE advisor_id = 'advisor-old'", (old_timestamp,))
    conn.commit()
    conn.close()

    deleted = purge_expired_reviews(max_age_days=7, db_path=db_path)
    assert deleted == 1

    remaining = list_pending_reviews(db_path)
    assert len(remaining) == 1
    assert remaining[0]['advisor_id'] == 'advisor-new'
