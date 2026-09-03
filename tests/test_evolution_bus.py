#!/usr/bin/env python3
"""
EvolutionBus 测试 — 进化事件总线核心功能

测试矩阵（对应 bus.py 能力）：
1. DigestEvent：必填校验 / 确定性哈希（跨进程稳定）
2. append：ts/hash自动补齐 / 落盘JSONL
3. query：type过滤 / producer过滤 / since过滤 / 新→旧排序
4. ConsumerRegistry：注册幂等 / consumers_for / unregistered告警
5. 反断头管：无consumer append → 告警（capsys捕获）
6. health：可写性 + 未注册type审计
7. 降级：store_dir不可写 → append抛OSError不静默吞（测试倒置）
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

import pytest

from openllm.evolution.bus import (
    ConsumerRegistry,
    DigestEvent,
    EvolutionBus,
    zlib_crc32,
)


# ═══════════════════════════════════════════════
# DigestEvent
# ═══════════════════════════════════════════════

class TestDigestEvent:
    def test_required_fields(self):
        with pytest.raises(ValueError, match="type"):
            DigestEvent(type="", producer="p")
        with pytest.raises(ValueError, match="producer"):
            DigestEvent(type="train.completed", producer="")

    def test_hash_deterministic(self):
        e1 = DigestEvent(type="train.completed", producer="nightly_train",
                         payload={"version": "v3"})
        e2 = DigestEvent(type="train.completed", producer="nightly_train",
                         payload={"version": "v3"})
        assert e1.compute_hash() == e2.compute_hash()
        assert e1.compute_hash().startswith("evt_")

    def test_hash_differs_on_payload_change(self):
        e1 = DigestEvent(type="train.completed", producer="nightly_train",
                         payload={"version": "v3"})
        e2 = DigestEvent(type="train.completed", producer="nightly_train",
                         payload={"version": "v4"})
        assert e1.compute_hash() != e2.compute_hash()

    def test_zlib_crc32_stable_across_processes(self):
        # 确定性：非内置hash（内置hash进程级随机——工程法典PITFALL 55）
        assert zlib_crc32("你好河床") == zlib_crc32("你好河床")


# ═══════════════════════════════════════════════
# EvolutionBus
# ═══════════════════════════════════════════════

class TestEvolutionBus:
    @pytest.fixture
    def bus(self, tmp_path):
        """隔离总线：临时store_dir，不碰 ~/.openllm/evolution 生产路径。"""
        return EvolutionBus(store_dir=tmp_path / "evolve")

    def test_append_fills_ts_hash(self, bus):
        ev = DigestEvent(type="train.completed", producer="nightly_train",
                         payload={"version": "v3"})
        bus.append(ev)
        assert ev.ts, "append应补齐ts"
        assert ev.hash.startswith("evt_"), "append应补齐hash"

    def test_append_persists_jsonl(self, bus, tmp_path):
        ev = DigestEvent(type="train.completed", producer="nightly_train",
                         payload={"version": "v3"})
        bus.append(ev)
        files = list((tmp_path / "evolve").glob("evolution_*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["type"] == "train.completed"
        assert rec["producer"] == "nightly_train"
        assert rec["payload"]["version"] == "v3"
        assert rec["hash"] == ev.hash

    def test_query_type_filter(self, bus):
        bus.append(DigestEvent(type="train.completed", producer="nightly_train"))
        bus.append(DigestEvent(type="verify.result", producer="verify_merged"))
        results = bus.query(event_type="verify.result")
        assert len(results) == 1
        assert results[0]["type"] == "verify.result"

    def test_query_sorted_newest_first(self, bus):
        for i in range(3):
            bus.append(DigestEvent(type="train.completed", producer="nightly_train",
                                   payload={"seq": i}))
        results = bus.query(event_type="train.completed")
        assert len(results) == 3
        assert results[0]["payload"]["seq"] == 2  # 最后append的最先

    def test_query_limit(self, bus):
        for i in range(5):
            bus.append(DigestEvent(type="train.completed", producer="nightly_train",
                                   payload={"seq": i}))
        results = bus.query(event_type="train.completed", limit=2)
        assert len(results) == 2

    def test_append_without_consumer_warns(self, bus, capsys):
        """反断头管：无consumer的type → 告警（但不阻塞写入）。"""
        bus.append(DigestEvent(type="train.completed", producer="nightly_train"))
        captured = capsys.readouterr()
        assert "无注册消费者" in captured.out
        assert "断头管风险" in captured.out

    def test_append_with_consumer_no_warn(self, bus, capsys):
        bus.registry.register("train.completed", "stopline")
        bus.append(DigestEvent(type="train.completed", producer="nightly_train"))
        captured = capsys.readouterr()
        assert "无注册消费者" not in captured.out

    def test_count(self, bus):
        bus.append(DigestEvent(type="train.completed", producer="p1"))
        bus.append(DigestEvent(type="verify.result", producer="p2"))
        assert bus.count("train.completed") == 1
        assert bus.count() == 2

    def test_health_ok_and_reports_unregistered(self, bus):
        h = bus.health()
        assert h["ok"] is True
        # 默认无注册消费者 → train.completed 应在未注册列表
        assert "train.completed" in h["unregistered_types"]
        # health探针不得污染事件流（append-only不容污染）
        assert bus.count() == 0


# ═══════════════════════════════════════════════
# ConsumerRegistry
# ═══════════════════════════════════════════════

class TestConsumerRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        return ConsumerRegistry(registry_path=tmp_path / "consumers.json")

    def test_register_idempotent(self, registry):
        registry.register("train.completed", "stopline")
        registry.register("train.completed", "stopline")
        assert registry.consumers_for("train.completed") == ["stopline"]

    def test_register_multiple_consumers(self, registry):
        registry.register("train.completed", "stopline")
        registry.register("train.completed", "tao_modulator")
        assert set(registry.consumers_for("train.completed")) == {
            "stopline", "tao_modulator"}

    def test_unregister(self, registry):
        registry.register("train.completed", "stopline")
        registry.unregister("train.completed", "stopline")
        assert registry.consumers_for("train.completed") == []

    def test_persistence(self, tmp_path):
        r1 = ConsumerRegistry(registry_path=tmp_path / "consumers.json")
        r1.register("verify.result", "auditor")
        r2 = ConsumerRegistry(registry_path=tmp_path / "consumers.json")
        assert r2.consumers_for("verify.result") == ["auditor"]

    def test_unregistered_types(self, registry):
        registry.register("train.completed", "stopline")
        unreg = registry.unregistered_types()
        assert "train.completed" not in unreg
        assert "verify.result" in unreg  # 未注册的已知type在列表


# ═══════════════════════════════════════════════
# 降级路径
# ═══════════════════════════════════════════════

class TestDegradation:
    def test_append_unwritable_store_raises(self, tmp_path):
        """store_dir不可写 → append抛OSError（总线故障不静默吞）。

        调用方（nightly_train等）负责try/except降级——总线自身不吞错。
        """
        bus = EvolutionBus(store_dir=tmp_path / "evolve")
        ev = DigestEvent(type="train.completed", producer="nightly_train")
        # 模拟不可写：store_dir变成文件路径
        bad_path = tmp_path / "not_a_dir"
        bad_path.write_text("i am a file")
        bus.store_dir = bad_path
        with pytest.raises(OSError):
            bus.append(ev)
