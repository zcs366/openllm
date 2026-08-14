"""test_memory_bus_update.py — MemoryBus.update() Update操作测试（T-ISA-7 Phase2）

覆盖：
  T1: jiak record 更新成功（版本化+历史存档）
  T2: jiak immutable 卡片拒绝修改
  T3: 非jiak record 返回明确不支持
  T4: 空record_id 报错

所有测试用临时卡片，测试后物理清理，不污染真实jiak。
"""
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openllm.memory.memory_bus import MemoryBus
from openllm.memory import jiak_lifecycle as jl

CARDS_DIR = Path.home() / ".hermes" / "jiak" / "cards"


def _make_temp_card(card_id: str, immutable: bool = False) -> dict:
    card = {
        "card_id": card_id,
        "title": f"test {card_id}",
        "keywords": ["kw1"],
        "summary": "temp card for update test",
        "notes": [],
        "decisions": [],
        "status": "active",
        "importance": 0.3,
        "memory_type": "test",
        "confidence": 0.5,
        "confidence_reason": "test",
        "immutable": immutable,
        "last_accessed": 0,
        "access_count": 0,
    }
    jl.write(card_id, card, written_by="test_memory_bus_update")
    return card


def _cleanup(card_ids: list):
    """清理测试临时卡。glob兜底防残留污染（与test_jiak_lifecycle一致）。"""
    for cid in card_ids:
        p = CARDS_DIR / f"{cid}.json"
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    for prefix in ("test-mbu-", "test-lifecycle-"):
        for p in CARDS_DIR.glob(f"{prefix}*.json"):
            try:
                p.unlink()
            except OSError:
                pass


def test_update_jiak_record_success():
    """jiak record 更新成功（版本化+历史存档）。"""
    bus = MemoryBus()
    cid = f"test-mbu-{uuid.uuid4().hex[:6]}"
    _make_temp_card(cid)
    try:
        record_id = f"jiak:{cid}:opinion1"
        result = bus.update(record_id, "summary", "updated summary", reason="test update")
        assert result.get("ok") is True, f"更新失败: {result}"
        assert result.get("record_id") == record_id
        # 验证卡片已更新+历史存档
        card = jl._read(cid)
        assert card["summary"] == "updated summary"
        hist = card["field_history"]["summary"]
        assert len(hist) == 1
        assert hist[0]["new"] == "updated summary"
        assert "revise" in [e["action"] for e in card["lifecycle_events"]]
    finally:
        _cleanup([cid])


def test_update_jiak_immutable_blocked():
    """jiak immutable 卡片拒绝修改。"""
    bus = MemoryBus()
    cid = f"test-mbu-imm-{uuid.uuid4().hex[:6]}"
    _make_temp_card(cid, immutable=True)
    try:
        record_id = f"jiak:{cid}:opinion1"
        result = bus.update(record_id, "title", "hacked", reason="should block")
        assert result.get("ok") is False
        assert "immutable" in result.get("error", "")
    finally:
        _cleanup([cid])


def test_update_non_jiak_not_supported():
    """非jiak record 返回明确不支持。"""
    bus = MemoryBus()
    result = bus.update("capsule:session1:abc123", "content", "new", reason="test")
    assert result.get("ok") is False
    assert "暂不支持" in result.get("error", "")
    assert result.get("supported") == ["jiak"]

    result2 = bus.update("causal:abcdef123456", "importance", 0.9, reason="test")
    assert result2.get("ok") is False
    assert "暂不支持" in result2.get("error", "")


def test_update_empty_record_id():
    """空record_id 报错。"""
    bus = MemoryBus()
    result = bus.update("", "summary", "x", reason="test")
    assert result.get("ok") is False
    assert "不能为空" in result.get("error", "")
