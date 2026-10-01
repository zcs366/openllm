"""Nail test — session_causal_extractor.extract_recent time filter (P1-4).

Verified live defect: the old code bound `str(cutoff)` (an epoch *string*,
e.g. "1790700000.5") into `WHERE timestamp > ?`, while the real
~/.hermes/jiak/jcot.db stores reasoning_chains.timestamp as ISO *text*
("2026-07-30T01:56:01"). In SQLite text-vs-text comparison "2026-..." >
"1790..." is lexicographically ALWAYS TRUE — the time filter was dead code,
and "recent N hours" returned the OLDEST sessions (insertion order).

Trap shape: a temp sqlite with sessions whose timestamps mix ISO text and
epoch numerics, old and new. extract_recent(hours) must return the genuinely
most-recent sessions. With the fix reverted (str(cutoff) binding) this test
must blow up on the assertion — not in setup — because under the old写法 ALL
sessions pass the filter and the order is insertion-based, so the newest
sessions are missing / in wrong order.
"""
import sqlite3
import time
from datetime import datetime, timedelta

import pytest

from openllm.isa.session_causal_extractor import SessionCausalExtractor


SCHEMA = """
CREATE TABLE reasoning_chains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT, timestamp TEXT, topic TEXT,
    position TEXT, confidence REAL,
    reasoning_path TEXT, conclusion TEXT
)
"""


def _make_db(tmp_path, rows):
    """rows: list of (session, timestamp_value) inserted in given order."""
    db = tmp_path / "jcot.db"
    conn = sqlite3.connect(str(db))
    conn.execute(SCHEMA)
    for session, ts in rows:
        conn.execute(
            "INSERT INTO reasoning_chains (session, timestamp,"
            " confidence, reasoning_path, conclusion)"
            " VALUES (?, ?, 0.5, 'r', 'c')",
            (session, ts),
        )
    conn.commit()
    conn.close()
    return db


def _extractor(db):
    ex = SessionCausalExtractor()
    ex.jcot_db = db

    # stub the LLM path: we only test session *selection* here
    def _stub(session_id, session_context=""):
        return {"causal_chains": 0, "written": 0, "details": []}

    ex.extract_from_session = _stub
    return ex


NOW = time.time()


def _iso(days_ago, hours_ago=0):
    return (datetime.now() - timedelta(days=days_ago, hours=hours_ago)
            ).isoformat(timespec="seconds")


@pytest.fixture()
def mixed_db(tmp_path):
    """6 sessions, inserted OLDEST-FIRST (worst case for the old写法):
    - old_iso     ISO text, 10 days ago
    - old_epoch   epoch numeric (stored as TEXT by sqlite affinity), 8 days ago
    - old_iso2    ISO text, 5 days ago
    - new_iso     ISO text, 1 hour ago
    - new_epoch   epoch numeric, 2 hours ago
    - mid_iso     ISO text, 3 days ago
    """
    rows = [
        ("old_iso", _iso(10)),
        ("old_epoch", str(NOW - 8 * 86400)),
        ("old_iso2", _iso(5)),
        ("new_iso", _iso(0, hours_ago=1)),
        ("new_epoch", NOW - 2 * 3600),
        ("mid_iso", _iso(3)),
    ]
    return _make_db(tmp_path, rows)


def test_recent_returns_genuinely_newest(mixed_db):
    """Both storage formats (ISO text + epoch numeric) must classify correctly;
    window 24h → exactly the two ~1-2h-old sessions; order newest-first."""
    ex = _extractor(mixed_db)
    results = ex.extract_recent(hours=24, limit=5)
    ids = [r["session_id"] for r in results]
    assert ids == ["new_iso", "new_epoch"], \
        f"must get the genuinely newest (both formats!), got {ids}"


def test_recent_limit_takes_newest_not_oldest(mixed_db):
    """Window wide enough for everything, limit=3 → 3 NEWEST sessions.
    Under the old str(cutoff) binding the filter is always true and SQLite
    returns DISTINCT-session in insertion order → this returns the 3 OLDEST."""
    ex = _extractor(mixed_db)
    results = ex.extract_recent(hours=365 * 24, limit=3)
    ids = [r["session_id"] for r in results]
    assert set(ids) == {"new_iso", "new_epoch", "mid_iso"}, \
        f"limit must cut by recency, got {ids}"
    assert ids[0] == "new_iso", f"newest first, got {ids}"


def test_unparseable_ts_excluded_from_recent(mixed_db_bad_ts):
    """A session whose only timestamp is junk ('not-a-date') resolves to 0
    via coerce_ts → not 'recent', must not be picked up."""
    ex = _extractor(mixed_db_bad_ts)
    results = ex.extract_recent(hours=24, limit=5)
    ids = [r["session_id"] for r in results]
    assert "junk_session" not in ids
    assert "good_recent" in ids


@pytest.fixture()
def mixed_db_bad_ts(tmp_path):
    rows = [
        ("junk_session", "not-a-date"),
        ("good_recent", _iso(0, hours_ago=3)),
        ("old_iso", _iso(400)),
    ]
    return _make_db(tmp_path, rows)


def test_empty_db_and_missing(tmp_path):
    ex = _extractor(tmp_path / "nope.db")
    assert ex.extract_recent(hours=24) == []
