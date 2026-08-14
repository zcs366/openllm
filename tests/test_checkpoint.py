"""test_checkpoint.py — TaskCheckpoint save/load 往返测试。"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openllm.core.checkpoint import TaskCheckpoint, _default_state


# ── Fixtures ──

@pytest.fixture
def ckpt():
    """TaskCheckpoint with backup disabled for fast tests."""
    return TaskCheckpoint(backup=False)


@pytest.fixture
def ckpt_with_backup():
    """TaskCheckpoint with backup enabled."""
    return TaskCheckpoint(backup=True)


@pytest.fixture
def sample_state():
    """A realistic task checkpoint state."""
    return {
        "history": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "tool_calls": [
            {"tool": "calculator", "args": {"expr": "2+2"}, "result": "4"},
        ],
        "results": [
            {"type": "answer", "value": "4", "confidence": 0.99},
        ],
        "timestamp": datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc),
    }


# ── Basic round-trip ──

class TestRoundTrip:
    def test_save_load_roundtrip(self, ckpt, sample_state, tmp_path):
        path = str(tmp_path / "test_ckpt.json")
        ckpt.save(sample_state, path)
        loaded = ckpt.load(path)

        assert loaded["history"] == sample_state["history"]
        assert loaded["tool_calls"] == sample_state["tool_calls"]
        assert loaded["results"] == sample_state["results"]
        # timestamp serialized as ISO string
        assert isinstance(loaded["timestamp"], str)

    def test_json_format_valid(self, ckpt, sample_state, tmp_path):
        path = str(tmp_path / "test_ckpt.json")
        ckpt.save(sample_state, path)

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        assert "history" in raw
        assert "tool_calls" in raw
        assert "results" in raw
        assert "timestamp" in raw


# ── Missing fields ──

class TestMissingFields:
    def test_partial_state_fills_defaults(self, ckpt, tmp_path):
        path = str(tmp_path / "partial.json")
        ckpt.save({"history": [{"role": "user", "content": "hi"}]}, path)
        loaded = ckpt.load(path)

        assert len(loaded["history"]) == 1
        assert loaded["tool_calls"] == []
        assert loaded["results"] == []
        assert "timestamp" in loaded

    def test_empty_state(self, ckpt, tmp_path):
        path = str(tmp_path / "empty.json")
        ckpt.save({}, path)
        loaded = ckpt.load(path)

        assert loaded["history"] == []
        assert loaded["tool_calls"] == []
        assert loaded["results"] == []


# ── Backup ──

class TestBackup:
    def test_backup_created(self, ckpt_with_backup, sample_state, tmp_path):
        path = str(tmp_path / "with_bak.json")
        ckpt_with_backup.save(sample_state, path)
        ckpt_with_backup.save(sample_state, path)  # second save triggers .bak

        bak = Path(path + ".bak")
        assert bak.exists()

    def test_no_backup_when_disabled(self, ckpt, sample_state, tmp_path):
        path = str(tmp_path / "no_bak.json")
        ckpt.save(sample_state, path)
        ckpt.save(sample_state, path)

        bak = Path(path + ".bak")
        assert not bak.exists()


# ── Error handling ──

class TestErrors:
    def test_load_missing_file(self, ckpt):
        with pytest.raises(FileNotFoundError):
            ckpt.load("/nonexistent/path/ckpt.json")

    def test_load_corrupt_json(self, ckpt, tmp_path):
        path = str(tmp_path / "corrupt.json")
        Path(path).write_text("NOT VALID JSON {{{")

        with pytest.raises(json.JSONDecodeError):
            ckpt.load(path)

    def test_save_creates_dirs(self, ckpt, tmp_path):
        path = str(tmp_path / "a" / "b" / "c" / "deep.json")
        ckpt.save({"history": []}, path)
        assert Path(path).exists()


# ── Edge cases ──

class TestEdgeCases:
    def test_unicode_content(self, ckpt, tmp_path):
        state = {
            "history": [
                {"role": "user", "content": "你好世界 🌍"},
                {"role": "assistant", "content": "こんにちは世界"},
            ],
        }
        path = str(tmp_path / "unicode.json")
        ckpt.save(state, path)
        loaded = ckpt.load(path)

        assert loaded["history"][0]["content"] == "你好世界 🌍"
        assert loaded["history"][1]["content"] == "こんにちは世界"

    def test_nested_results(self, ckpt, tmp_path):
        state = {
            "results": [
                {"type": "analysis", "data": {"scores": [0.8, 0.9], "label": "ok"}},
            ],
        }
        path = str(tmp_path / "nested.json")
        ckpt.save(state, path)
        loaded = ckpt.load(path)

        assert loaded["results"][0]["data"]["scores"] == [0.8, 0.9]

    def test_large_history(self, ckpt, tmp_path):
        state = {
            "history": [
                {"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"message_{i}" * 100}
                for i in range(200)
            ],
        }
        path = str(tmp_path / "large.json")
        ckpt.save(state, path)
        loaded = ckpt.load(path)

        assert len(loaded["history"]) == 200


# ── default_state helper ──

class TestDefaultState:
    def test_has_all_keys(self):
        d = _default_state()
        assert "history" in d
        assert "tool_calls" in d
        assert "results" in d
        assert "timestamp" in d

    def test_timestamp_is_iso(self):
        d = _default_state()
        # Should be parseable as datetime
        datetime.fromisoformat(d["timestamp"])
