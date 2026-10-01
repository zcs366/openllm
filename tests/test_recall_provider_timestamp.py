"""Regression nail - recall provider: one dirty line must not kill the channel (P0).

Two verified defects:
1. Six lines in ~/.hermes/jiak/RECALL.jsonl carry ISO8601 string timestamps.
   `ts >= cutoff` raises TypeError comparing str with float; the except only
   caught JSONDecodeError and the outer one only OSError, so the exception
   escaped to memory_bus and the whole recall provider was skipped every run.
2. Field-name mismatch: the real canonical fields are ts (139 rows) and
   _timestamp (5 rows); the old code only read timestamp/_written_at, so
   only 4 of 177 rows were ever readable. And the old "missing -> 0 ->
   dropped by the 30-day window" semantics made 96% of records invisible.

Ruling (2026-10-01): filter ONLY when the timestamp is known AND older than
the cutoff; records with missing/unparseable timestamps are kept and counted
in stats.missing_ts.
"""
import calendar
import json
import logging
import time
from datetime import datetime, timedelta

from openllm.isa.memory_bus import Query
from openllm.isa.providers.recall_provider import RecallProvider

ISO_MARKER = "iso-row: P0 recall timestamp regression"
FLOAT_MARKER = "float-row: in window"
GARBAGE_MARKER = "never-parseable"
NO_TS_MARKER = "row with no time field at all"
EXPIRED_MARKER = "row from 90 days ago"
TS_STR_MARKER = "ts-field row: string epoch"

# malformed JSON: the value itself is not a valid JSON token
GARBAGE_LINE = '{"content": "%s", "stamp": <<<broken>>>' % GARBAGE_MARKER


def _make_recall(tmp_path):
    """Six mixed lines: (1) ISO-string timestamp in window, (2) float
    timestamp in window, (3) malformed JSON, (4) no time field at all,
    (5) float timestamp 90 days ago (must be filtered by cutoff), (6) only
    a ts field whose value is a string epoch in window.
    Line (1) is first: with the fix reverted the trap test must blow up on
    TypeError, not on setup."""
    now = time.time()
    iso_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    lines = [
        json.dumps({"content": ISO_MARKER, "type": "event", "timestamp": iso_str}),
        json.dumps({"content": FLOAT_MARKER, "type": "event", "timestamp": now - 3600}),
        GARBAGE_LINE,
        json.dumps({"content": NO_TS_MARKER, "type": "event"}),
        json.dumps({"content": EXPIRED_MARKER, "type": "event",
                    "timestamp": now - 90 * 86400}),
        json.dumps({"content": TS_STR_MARKER, "type": "event",
                    "ts": str(now - 7200)}),
    ]
    path = tmp_path / "RECALL.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_dirty_lines_do_not_kill_channel(tmp_path):
    """Trap-shape test: _load_records must not raise on mixed dirty data;
    the returned set is exactly {(1),(2),(4),(6)} ((3) skipped, (5) stale),
    and (1) is genuinely coerced to an epoch float (not 'luckily never
    compared')."""
    path = _make_recall(tmp_path)
    provider = RecallProvider(recall_path=path)

    records = provider._load_records()

    assert records is not None
    contents = {r.get("content") for r in records}
    assert contents == {ISO_MARKER, FLOAT_MARKER, NO_TS_MARKER, TS_STR_MARKER}

    iso_rec = next(r for r in records if r["content"] == ISO_MARKER)
    assert isinstance(iso_rec["timestamp"], float), \
        "ISO string must be normalized in place to an epoch float, or the" \
        " downstream search() blows up again"
    assert iso_rec["timestamp"] >= time.time() - 3 * 86400

    # end-to-end: the retrieval channel is alive
    hits = provider.search(Query(text=ISO_MARKER, top_k=5))
    assert any(ISO_MARKER in h.content for h in hits)


def test_ts_field_chain(tmp_path):
    """Field-chain trap: a row whose ONLY time field is ts (string epoch,
    in window) must be read and normalized. If the chain omits 'ts', this
    test is red (old code only read timestamp/_written_at)."""
    now = time.time()
    path = tmp_path / "RECALL.jsonl"
    path.write_text(json.dumps({"content": TS_STR_MARKER, "type": "event",
                                "ts": str(now - 7200)}) + "\n",
                    encoding="utf-8")
    provider = RecallProvider(recall_path=path)

    records = provider._load_records()

    assert [r.get("content") for r in records] == [TS_STR_MARKER], \
        "ts must be in the value chain: 139/177 real rows carry it"
    assert isinstance(records[0]["ts"], float)
    assert records[0]["ts"] >= now - 86400

    # _timestamp (ISO string) goes through the same chain
    path2 = tmp_path / "RECALL2.jsonl"
    path2.write_text(json.dumps({"content": "_timestamp row", "type": "event",
                                 "_timestamp": (datetime.now()
                                               - timedelta(hours=2)).isoformat()})
                     + "\n", encoding="utf-8")
    p2 = RecallProvider(recall_path=path2)
    recs2 = p2._load_records()
    assert [r.get("content") for r in recs2] == ["_timestamp row"]
    assert isinstance(recs2[0]["_timestamp"], float)


def test_missing_ts_record_survives(tmp_path):
    """Semantics trap: a row with NO time field at all must stay in the
    returned set and be counted in stats.missing_ts (ruling: unknown time
    is kept, never silently dropped)."""
    path = tmp_path / "RECALL.jsonl"
    path.write_text(json.dumps({"content": NO_TS_MARKER, "type": "event"})
                    + "\n", encoding="utf-8")
    provider = RecallProvider(recall_path=path)

    records = provider._load_records()

    assert [r.get("content") for r in records] == [NO_TS_MARKER]
    assert provider.last_load_stats["missing_ts"] == 1

    # an unparseable junk value is also kept, normalized to 0.0
    path2 = tmp_path / "RECALL2.jsonl"
    path2.write_text(json.dumps({"content": "junk stamp", "type": "event",
                                 "timestamp": "not-a-date"}) + "\n",
                     encoding="utf-8")
    p2 = RecallProvider(recall_path=path2)
    recs2 = p2._load_records()
    assert [r.get("content") for r in recs2] == ["junk stamp"]
    assert recs2[0]["timestamp"] == 0.0
    assert p2.last_load_stats["missing_ts"] == 1


def test_load_stats_counts(tmp_path):
    """Three counters: ok / skipped_bad / missing_ts. Row (5) has a valid
    ts and is filtered by the cutoff: counted in ok, not in kept."""
    path = _make_recall(tmp_path)
    provider = RecallProvider(recall_path=path)
    provider._load_records()

    stats = provider.last_load_stats
    assert stats["ok"] == 4            # (1)(2)(5)(6)
    assert stats["skipped_bad"] == 1   # (3)
    assert stats["missing_ts"] == 1    # (4)
    assert stats["kept"] == 4          # (1)(2)(6) + (4) kept by ruling
    assert stats["kept"] == len(provider._records)


def test_health_keys_survive(tmp_path):
    """health() keeps its original keys (status/path/exists/record_count)."""
    path = _make_recall(tmp_path)
    provider = RecallProvider(recall_path=path)

    h = provider.health()

    for key in ("status", "path", "exists", "record_count"):
        assert key in h
    assert h["exists"] is True
    assert h["status"] == "ok"
    assert h["record_count"] == 4


def test_coerce_ts_rules():
    """Unit rules for _coerce_ts: numbers pass through, numeric strings
    float(), ISO8601 with Z suffix and date-only parse, junk maps to 0.
    Expectations computed independently (timegm/mktime), not via the SUT."""
    from openllm.isa.providers.recall_provider import _coerce_ts

    assert _coerce_ts(1759000000) == 1759000000.0
    assert _coerce_ts(1759000000.5) == 1759000000.5
    assert _coerce_ts("1759000000.5") == 1759000000.5
    assert _coerce_ts("2026-09-29T12:00:00Z") == \
        calendar.timegm((2026, 9, 29, 12, 0, 0))
    assert _coerce_ts("2026-09-29") == time.mktime(
        (2026, 9, 29, 0, 0, 0, 0, 0, -1))
    assert _coerce_ts("not-a-date") == 0
    assert _coerce_ts("") == 0
    assert _coerce_ts(None) == 0
    assert _coerce_ts(True) == 0
    assert _coerce_ts([1, 2]) == 0


def _recall_warnings(caplog):
    return [r for r in caplog.records
            if r.name.startswith("openllm.providers.recall")
            and r.levelno >= logging.WARNING]


def test_warning_is_logged_once_per_instance(tmp_path, caplog):
    """Noise nail (军师 2026-10-01): _load_records re-reads whenever the 10s
    cache expires, so a per-load warning turns into fresh log spam. Only the
    first load of an instance may warn; later loads drop to debug."""
    path = _make_recall(tmp_path)
    provider = RecallProvider(recall_path=path)

    provider._load_records()
    first = len(_recall_warnings(caplog))
    for _ in range(3):                      # force three more re-reads
        provider._loaded_at = 0
        provider._load_records()
    later = len(_recall_warnings(caplog))

    assert first == 1, f"first load must warn exactly once (got {first})"
    assert later == 1, f"re-reads must not re-warn (got {later})"
