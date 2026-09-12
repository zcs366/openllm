"""context_audit_logger 测试 —— 3 条以上，pytest 跑绿。"""
import json
import time
from pathlib import Path

import pytest

from openllm.isa.context_audit_logger import ContextAuditLogger


@pytest.fixture
def tmp_log(tmp_path):
    """返回一个用临时文件隔离的日志器实例。"""
    return ContextAuditLogger(log_path=tmp_path / "audit.jsonl")


# ── C2: 接口签名存在 ──────────────────────────────────────────


def test_interface_exists():
    """C2: log_delegation / load / stats 接口存在且签名正确。"""
    import inspect

    logger = ContextAuditLogger()
    assert callable(logger.log_delegation)
    assert callable(logger.load)
    assert callable(logger.stats)

    sig = inspect.signature(logger.log_delegation)
    params = list(sig.parameters.keys())
    assert "target" in params
    assert "objective" in params
    assert "context_summary" in params
    assert "injected" in params
    assert "result_summary" in params
    assert "started_at" in params
    assert "finished_at" in params
    assert "tokens" in params
    assert "tags" in params


# ── C3: log + load 一条 ───────────────────────────────────────


def test_log_and_load(tmp_log):
    """C3a: 写入一条记录，读回完整。"""
    now = time.time()
    record = tmp_log.log_delegation(
        target="subagent-alpha",
        objective="搜索论文",
        context_summary="用户要求搜索attention mechanics",
        injected="system_prompt + 指令",
        result_summary="找到3篇论文",
        started_at=now,
        finished_at=now + 2.5,
        tokens=800,
        tags=["research"],
    )
    assert record["target"] == "subagent-alpha"
    assert record["duration_s"] == 2.5
    assert record["tokens"] == 800
    assert record["tags"] == ["research"]

    loaded = tmp_log.load(recent_n=10)
    assert len(loaded) == 1
    assert loaded[0]["target"] == "subagent-alpha"
    assert loaded[0]["objective"] == "搜索论文"


# ── C3: append-only 追加不丢 ──────────────────────────────────


def test_append_only(tmp_log):
    """C3b: 追写两条，读回两条，数量正确不丢不重。"""
    t1 = time.time()
    tmp_log.log_delegation(
        target="A", objective="o1", context_summary="c1",
        injected="i1", result_summary="r1",
        started_at=t1, finished_at=t1 + 1.0, tokens=100,
    )
    t2 = time.time()
    tmp_log.log_delegation(
        target="B", objective="o2", context_summary="c2",
        injected="i2", result_summary="r2",
        started_at=t2, finished_at=t2 + 2.0, tokens=200,
    )
    loaded = tmp_log.load(recent_n=100)
    assert len(loaded) == 2
    assert loaded[0]["target"] == "A"
    assert loaded[1]["target"] == "B"


# ── C5: stats 正确 ────────────────────────────────────────────


def test_stats(tmp_log):
    """C5: stats 统计总数、分组、平均耗时、token 总和正确。"""
    t1 = 1000.0
    tmp_log.log_delegation(
        target="A", objective="o1", context_summary="c1",
        injected="i1", result_summary="r1",
        started_at=t1, finished_at=t1 + 1.0, tokens=100,
    )
    tmp_log.log_delegation(
        target="A", objective="o2", context_summary="c2",
        injected="i2", result_summary="r2",
        started_at=t1, finished_at=t1 + 3.0, tokens=200,
    )
    tmp_log.log_delegation(
        target="B", objective="o3", context_summary="c3",
        injected="i3", result_summary="r3",
        started_at=t1, finished_at=t1 + 2.0, tokens=50,
    )

    s = tmp_log.stats()
    assert s["total"] == 3
    assert s["total_tokens"] == 350
    assert s["avg_duration_s"] == round((1.0 + 3.0 + 2.0) / 3, 3)
    assert s["targets"]["A"]["count"] == 2
    assert s["targets"]["A"]["avg_duration_s"] == 2.0
    assert s["targets"]["B"]["count"] == 1
    assert s["targets"]["B"]["avg_duration_s"] == 2.0


# ── load 读空文件 ─────────────────────────────────────────────


def test_load_empty(tmp_log):
    """文件不存在时 load 返回空列表。"""
    assert tmp_log.load() == []


# ── stats 空文件 ──────────────────────────────────────────────


def test_stats_empty(tmp_log):
    """空文件 stats 返回零值。"""
    s = tmp_log.stats()
    assert s["total"] == 0
    assert s["targets"] == {}
    assert s["avg_duration_s"] == 0.0
    assert s["total_tokens"] == 0
