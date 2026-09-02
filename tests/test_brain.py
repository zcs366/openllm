"""
tests/test_brain.py — SA/ZA 头脑系统测试。

覆盖：BRAIN_VOCAB 词表 → BrainRegistry 注册/查询/归档
      → BrainActivator 激活/替换/deactivate → 文件落盘验证
"""
import json
import os
from pathlib import Path

import pytest

from openllm.iai.brain import (
    BRAIN_VOCAB,
    BrainActivator,
    BrainRegistry,
    BrainRecord,
)


# ── fixtures：用 tmp_path 隔离磁盘，测试不污染真实文件 ──

@pytest.fixture
def registry(tmp_path: Path) -> BrainRegistry:
    """每个测试用独立的注册表文件。"""
    return BrainRegistry(path=tmp_path / "brain_registry.json")


@pytest.fixture
def activator(tmp_path: Path) -> BrainActivator:
    """每个测试用独立的激活状态文件。"""
    return BrainActivator(path=tmp_path / "active_brain.json")


# ── BRAIN_VOCAB 词表 ──

class TestBrainVocab:
    def test_has_five_brains(self):
        assert len(BRAIN_VOCAB) == 5

    def test_expected_names(self):
        expected = {"军师", "子贡", "包拯", "韩信", "探照灯"}
        assert set(BRAIN_VOCAB.keys()) == expected

    def test_each_has_required_keys(self):
        for name, entry in BRAIN_VOCAB.items():
            assert "persona" in entry, f"{name} 缺 persona"
            assert "mode" in entry, f"{name} 缺 mode"
            assert entry["mode"] in ("SA", "ZA"), f"{name} mode 非法: {entry['mode']}"
            assert "memory_domain" in entry, f"{name} 缺 memory_domain"

    def test_mode_distribution(self):
        sa = [n for n, e in BRAIN_VOCAB.items() if e["mode"] == "SA"]
        za = [n for n, e in BRAIN_VOCAB.items() if e["mode"] == "ZA"]
        assert len(sa) >= 2, f"SA 头脑不足: {sa}"
        assert len(za) >= 1, f"ZA 头脑不足: {za}"


# ── BrainRecord 数据类 ──

class TestBrainRecord:
    def test_roundtrip(self):
        rec = BrainRecord(
            brain_id="brain-test001",
            persona_name="军师",
            kind="SA",
            persona="战略分析",
            memory_domain="strategy",
            created_at="2026-09-02T10:00:00",
        )
        d = rec.to_dict()
        rec2 = BrainRecord.from_dict(d)
        assert rec2.brain_id == rec.brain_id
        assert rec2.persona_name == rec.persona_name

    def test_archived_at_default_none(self):
        rec = BrainRecord(
            brain_id="brain-x",
            persona_name="子贡",
            kind="SA",
            persona="执行调度",
            memory_domain="execution",
            created_at="2026-09-02T10:00:00",
        )
        assert rec.archived_at is None
        assert rec.adapter_path is None


# ── BrainRegistry 注册表 ──

class TestBrainRegistry:
    def test_create_returns_id(self, registry: BrainRegistry):
        brain_id = registry.create("军师", kind="SA")
        assert brain_id.startswith("brain-")
        assert len(brain_id) > 10

    def test_create_za(self, registry: BrainRegistry):
        brain_id = registry.create("韩信", kind="ZA")
        record = registry.get(brain_id)
        assert record is not None
        assert record["kind"] == "ZA"
        assert record["persona_name"] == "韩信"

    def test_create_unknown_persona(self, registry: BrainRegistry):
        """不在词表中的 persona 也能创建，用默认值。"""
        brain_id = registry.create("路人甲", kind="ZA")
        record = registry.get(brain_id)
        assert record is not None
        assert record["persona_name"] == "路人甲"
        assert record["memory_domain"] == "general"

    def test_list_excludes_archived(self, registry: BrainRegistry):
        id1 = registry.create("军师")
        id2 = registry.create("子贡")
        registry.archive(id1)
        visible = registry.list()
        assert len(visible) == 1
        assert visible[0]["brain_id"] == id2

    def test_list_includes_archived_when_flagged(self, registry: BrainRegistry):
        id1 = registry.create("包拯")
        registry.archive(id1)
        all_brains = registry.list(include_archived=True)
        assert len(all_brains) == 1
        assert all_brains[0]["archived_at"] is not None

    def test_get_nonexistent_returns_none(self, registry: BrainRegistry):
        assert registry.get("brain-nope") is None

    def test_archive_nonexistent_returns_false(self, registry: BrainRegistry):
        assert registry.archive("brain-nope") is False

    def test_archive_existing_returns_true(self, registry: BrainRegistry):
        brain_id = registry.create("军师")
        assert registry.archive(brain_id) is True
        record = registry.get(brain_id)
        assert record["archived_at"] is not None

    def test_file_persists(self, registry: BrainRegistry, tmp_path: Path):
        """写入后文件确实存在且可解析。"""
        registry.create("军师")
        reg_file = tmp_path / "brain_registry.json"
        assert reg_file.exists()
        data = json.loads(reg_file.read_text(encoding="utf-8"))
        assert len(data) == 1


# ── BrainActivator 单意识激活器 ──

class TestBrainActivator:
    def test_initially_none(self, activator: BrainActivator):
        assert activator.current() is None

    def test_activate(self, activator: BrainActivator):
        activator.activate("brain-001")
        assert activator.current() == "brain-001"

    def test_activate_replaces_previous(self, activator: BrainActivator):
        """一次只能一个活跃头脑——激活新的自动替换旧的。"""
        activator.activate("brain-001")
        assert activator.current() == "brain-001"
        activator.activate("brain-002")
        assert activator.current() == "brain-002"

    def test_deactivate(self, activator: BrainActivator):
        activator.activate("brain-001")
        activator.deactivate()
        assert activator.current() is None

    def test_file_persists(self, activator: BrainActivator, tmp_path: Path):
        activator.activate("brain-001")
        state_file = tmp_path / "active_brain.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["active_brain_id"] == "brain-001"
        assert "activated_at" in data

    def test_new_activator_sees_existing(self, activator: BrainActivator, tmp_path: Path):
        """新 Activator 实例能读到旧实例写入的激活状态。"""
        activator.activate("brain-001")
        activator2 = BrainActivator(path=tmp_path / "active_brain.json")
        assert activator2.current() == "brain-001"


# ── 集成：Registry + Activator 配合 ──

class TestRegistryActivatorIntegration:
    def test_full_lifecycle(self, registry: BrainRegistry, activator: BrainActivator):
        """完整生命周期：创建→激活→激活另一个→归档→确认。"""
        id1 = registry.create("军师", kind="SA")
        id2 = registry.create("韩信", kind="ZA")

        activator.activate(id1)
        assert activator.current() == id1

        activator.activate(id2)
        assert activator.current() == id2

        registry.archive(id1)
        visible = registry.list()
        assert len(visible) == 1
        assert visible[0]["brain_id"] == id2

        activator.deactivate()
        assert activator.current() is None

    def test_five_brains_from_vocab(self, registry: BrainRegistry):
        """从词表创建全部 5 个头脑。"""
        ids = {}
        for name, entry in BRAIN_VOCAB.items():
            brain_id = registry.create(name, kind=entry["mode"])
            ids[name] = brain_id
        all_brains = registry.list()
        assert len(all_brains) == 5
        for name, brain_id in ids.items():
            rec = registry.get(brain_id)
            assert rec["persona_name"] == name
