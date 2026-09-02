"""tests/test_preference_collector.py — preference_collector 单元测试。

验证：proposed+critiqued(reject)→rejected 样本写入 JSONL；
approve→chosen 候选；stats 正确；append-only；start/stop 生命周期。
"""
import json
import os
import tempfile
from pathlib import Path

import pytest

from openllm.iai.event_bus import Event, EventBus
from openllm.iai.preference_collector import PreferenceCollector


@pytest.fixture
def bus_and_collector(tmp_path):
    """创建独立 bus + collector，使用临时 JSONL 文件。"""
    b = EventBus(log_dir=tmp_path)
    jsonl_path = tmp_path / "preference_pairs.jsonl"
    collector = PreferenceCollector(bus=b, path=jsonl_path)
    yield b, collector, jsonl_path
    collector.stop()


def _make_event(event_type: str, payload: dict) -> Event:
    return Event(
        source="iai", type=event_type,
        timestamp=1000.0, entropy_score=0.0, payload=payload,
    )


class TestRejectPair:
    """reject verdict → 写入 rejected 样本。"""

    def test_reject_writes_rejected(self, bus_and_collector):
        bus, collector, jsonl = bus_and_collector
        collector.start()

        # 发布 proposed → critiqued(reject)
        bus.publish(_make_event("reasoning.proposed", {
            "content": "用递归求解",
            "confidence": 0.7,
        }))
        bus.publish(_make_event("reasoning.critiqued", {
            "verdict": "reject",
            "concerns": 1,  # emit 用 len(concerns)
        }))

        assert jsonl.exists()
        lines = jsonl.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["type"] == "rejected"
        assert record["proposal_content"] == "用递归求解"
        assert record["proposal_confidence"] == 0.7
        assert record["critique_verdict"] == "reject"
        assert record["concerns_count"] == 1
        assert record["source_ref"].startswith("tick:evt-")


class TestApprovePair:
    """approve verdict → 写入 chosen 样本。"""

    def test_approve_writes_chosen(self, bus_and_collector):
        bus, collector, jsonl = bus_and_collector
        collector.start()

        bus.publish(_make_event("reasoning.proposed", {
            "content": "迭代求解",
            "confidence": 0.9,
        }))
        bus.publish(_make_event("reasoning.critiqued", {
            "verdict": "approve",
            "concerns": [],
        }))

        lines = jsonl.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["type"] == "chosen"
        assert record["proposal_content"] == "迭代求解"


class TestStats:
    """preference_stats() 返回正确计数。"""

    def test_stats_after_mixed(self, bus_and_collector):
        bus, collector, jsonl = bus_and_collector
        collector.start()

        # 第1对：reject
        bus.publish(_make_event("reasoning.proposed", {"content": "A", "confidence": 0.5}))
        bus.publish(_make_event("reasoning.critiqued", {"verdict": "reject", "concerns": []}))
        # 第2对：approve
        bus.publish(_make_event("reasoning.proposed", {"content": "B", "confidence": 0.8}))
        bus.publish(_make_event("reasoning.critiqued", {"verdict": "approve", "concerns": []}))
        # 第3对：reject
        bus.publish(_make_event("reasoning.proposed", {"content": "C", "confidence": 0.3}))
        bus.publish(_make_event("reasoning.critiqued", {"verdict": "reject", "concerns": ["不好"]}))

        stats = collector.preference_stats()
        assert stats["total"] == 3
        assert stats["rejected"] == 2
        assert stats["approved"] == 1
        assert stats["file_exists"] is True


class TestAppendOnly:
    """多轮发布只追加，不覆盖。"""

    def test_append_only(self, bus_and_collector):
        bus, collector, jsonl = bus_and_collector
        collector.start()

        for i in range(5):
            bus.publish(_make_event("reasoning.proposed", {
                "content": f"proposal_{i}",
                "confidence": 0.5 + i * 0.1,
            }))
            bus.publish(_make_event("reasoning.critiqued", {
                "verdict": "reject" if i % 2 == 0 else "approve",
                "concerns": [],
            }))

        lines = [l for l in jsonl.read_text().strip().split("\n") if l.strip()]
        assert len(lines) == 5

        # 每行都是独立 JSON
        for line in lines:
            record = json.loads(line)
            assert "type" in record
            assert record["type"] in ("rejected", "chosen")


class TestNoProposal:
    """critiqued 到达但没有配对 proposal → 不写入。"""

    def test_orphan_critique_ignored(self, bus_and_collector):
        bus, collector, jsonl = bus_and_collector
        collector.start()

        # 直接发 critiqued，无前序 proposed
        bus.publish(_make_event("reasoning.critiqued", {
            "verdict": "reject",
            "concerns": [],
        }))

        assert not jsonl.exists() or jsonl.read_text().strip() == ""


class TestStartStopLifecycle:
    """start/stop 生命周期：stop 后不再收集。"""

    def test_stop_prevents_collection(self, bus_and_collector):
        bus, collector, jsonl = bus_and_collector
        collector.start()

        # 收集一次
        bus.publish(_make_event("reasoning.proposed", {"content": "A", "confidence": 0.5}))
        bus.publish(_make_event("reasoning.critiqued", {"verdict": "reject", "concerns": []}))
        assert jsonl.exists()
        lines_before = jsonl.read_text().strip().split("\n")

        collector.stop()

        # 再发事件 → 不应写入
        bus.publish(_make_event("reasoning.proposed", {"content": "B", "confidence": 0.6}))
        bus.publish(_make_event("reasoning.critiqued", {"verdict": "reject", "concerns": []}))
        lines_after = jsonl.read_text().strip().split("\n")
        assert len(lines_after) == len(lines_before)

    def test_restart_resumes(self, bus_and_collector):
        bus, collector, jsonl = bus_and_collector
        collector.start()

        bus.publish(_make_event("reasoning.proposed", {"content": "A", "confidence": 0.5}))
        bus.publish(_make_event("reasoning.critiqued", {"verdict": "reject", "concerns": []}))

        collector.stop()
        collector.start()

        bus.publish(_make_event("reasoning.proposed", {"content": "B", "confidence": 0.7}))
        bus.publish(_make_event("reasoning.critiqued", {"verdict": "approve", "concerns": []}))

        stats = collector.preference_stats()
        assert stats["total"] == 2


class TestNotStarted:
    """未 start 的 collector 不收集任何事件。"""

    def test_idle_collector_ignores_events(self, bus_and_collector):
        bus, collector, jsonl = bus_and_collector
        # 故意不调 start()

        bus.publish(_make_event("reasoning.proposed", {"content": "X", "confidence": 0.5}))
        bus.publish(_make_event("reasoning.critiqued", {"verdict": "reject", "concerns": []}))

        stats = collector.preference_stats()
        assert stats["total"] == 0
        assert not jsonl.exists() or jsonl.read_text().strip() == ""
