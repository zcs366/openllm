"""mirror_time_v0 测试。禁止触碰宿主真实 ~/.openllm——Path.home 全量 monkeypatch 到 tmp_path。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.mirror_time_v0 import (
    append_log,
    build_metrics,
    collect_causal_scars,
    collect_isl_chain,
    collect_preferences,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """把 Path.home 指到 tmp_path，并搭出 ~/.openllm 骨架。"""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    (tmp_path / ".openllm" / "iai").mkdir(parents=True)
    (tmp_path / ".openllm" / "memory" / "causal").mkdir(parents=True)
    return tmp_path


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scar(path: Path, success: float, importance: float, ts: float) -> None:
    path.write_text(json.dumps({
        "memory_id": path.stem, "created_at": ts, "action_signature": "sig",
        "prediction": "p", "actual_success": success, "delta_magnitude": 1,
        "lesson": "l", "source": "test", "importance": importance,
    }), encoding="utf-8")


def _epoch(n: int, h: str, prev: str) -> str:
    return json.dumps({"epoch": n, "session_id": f"s{n}", "hash": h, "prev_hash": prev,
                       "scars": [], "decisions": [], "wall_time": 1.0})


# ---------- ① 三源正常路径 ----------

def test_full_run_disclaimer_and_metrics(home):
    pref = home / ".openllm" / "iai" / "preference_pairs.jsonl"
    _write(pref, [
        "# generated=2026-09-02 header line",
        json.dumps({"prompt": "a", "chosen": "b", "rejected": "c", "meta": "m"}),
        json.dumps({"prompt": "d", "chosen": "e", "rejected": "f", "meta": "m"}),
        json.dumps({"prompt": "g", "chosen": "h", "rejected": "i", "meta": "m"}),
    ])
    chain = home / ".openllm" / "isl_chain.jsonl"
    _write(chain, [_epoch(1, "aa", ""), _epoch(2, "bb", "aa")])
    causal = home / ".openllm" / "memory" / "causal"
    _scar(causal / "s1.json", 1.0, 0.7, 1786712677.0)
    _scar(causal / "s2.json", 0.0, 0.5, 1787548278.0)

    metrics = build_metrics()
    assert metrics["preference_pairs"]["total"] == 3
    assert metrics["isl_chain"]["epochs"] == 2
    assert metrics["isl_chain"]["linkage_ok"] is True
    assert metrics["isl_chain"]["last_epoch"] == 2
    assert metrics["causal_scars"]["count"] == 2
    assert metrics["causal_scars"]["success_rate"] == 0.5
    assert metrics["causal_scars"]["importance_mean"] == 0.6

    log = home / ".openllm" / "mirror_time_log.jsonl"
    append_log(log, metrics)
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["type"] == "disclaimer"
    assert "判读权在造物主" in lines[0]["text"]
    assert lines[1]["type"] == "mirror_run"
    assert lines[1]["metrics"]["preference_pairs"]["total"] == 3


# ---------- ② 链破坏检测 ----------

def test_broken_chain_linkage(home):
    chain = home / ".openllm" / "isl_chain.jsonl"
    _write(chain, [_epoch(1, "aa", ""), _epoch(2, "bb", "XX")])  # prev_hash 错
    out = collect_isl_chain(chain)
    assert out["linkage_ok"] is False
    assert out["epochs"] == 2


def test_non_monotonic_epoch(home):
    chain = home / ".openllm" / "isl_chain.jsonl"
    _write(chain, [_epoch(1, "aa", ""), _epoch(3, "bb", "aa")])  # epoch 跳号
    out = collect_isl_chain(chain)
    assert out["linkage_ok"] is False


# ---------- ③ # 注释行跳过 ----------

def test_comment_lines_skipped(home):
    pref = home / ".openllm" / "iai" / "preference_pairs.jsonl"
    _write(pref, [
        "# header 1",
        "# header 2",
        json.dumps({"prompt": "x", "chosen": "y", "rejected": "z", "meta": "m"}),
        "",
    ])
    out = collect_preferences(pref)
    assert out["total"] == 1


# ---------- ④ 单源缺失不炸 ----------

def test_missing_sources_still_log(home):
    # 什么都不建：偏好对缺失、ISL 缺失；伤疤目录空
    metrics = build_metrics()
    assert metrics["preference_pairs"]["total"] is None
    assert "errors" in metrics["preference_pairs"]
    assert metrics["isl_chain"]["epochs"] is None
    assert metrics["causal_scars"]["count"] == 0

    log = home / ".openllm" / "mirror_time_log.jsonl"
    rec = append_log(log, metrics)  # 不应抛异常
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["event_id"] == rec["event_id"]


def test_corrupt_scar_file_isolated(home):
    causal = home / ".openllm" / "memory" / "causal"
    _scar(causal / "good.json", 1.0, 0.8, 100.0)
    (causal / "bad.json").write_text("{not json", encoding="utf-8")
    out = collect_causal_scars(causal)
    assert out["count"] == 2
    assert out["success_rate"] == 1.0  # 只统计成功的那个
    assert "errors" in out
