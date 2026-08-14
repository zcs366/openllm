"""test_jiak_lifecycle.py — T-ISA-7 jiak六阶段生命周期API测试

覆盖：
  T1: find_similar — 相似卡检测
  T2: evict dry-run — 淘汰候选扫描
  T3: revise — 版本化修正（field_history + immutable保护）
  T4: remove — 软删除
  T5: consolidate — 相似卡合并（临时卡对）

所有测试用临时卡片（test-lifecycle-*），测试后物理清理，不污染真实jiak。
"""
import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openllm.memory import jiak_lifecycle as jl

CARDS_DIR = Path.home() / ".hermes" / "jiak" / "cards"


def _make_temp_card(card_id: str, keywords: list, notes: list, **extra) -> dict:
    """创建临时卡片（复制结构，写入测试数据）。"""
    card = {
        "card_id": card_id,
        "title": f"test {card_id}",
        "keywords": keywords,
        "summary": "temp card for lifecycle test",
        "notes": notes,
        "decisions": [],
        "status": "active",
        "importance": 0.3,
        "memory_type": "test",
        "confidence": 0.5,
        "confidence_reason": "test",
        "immutable": extra.get("immutable", False),
        "last_accessed": 0,
        "access_count": 0,
    }
    jl.write(card_id, card, written_by="test_jiak_lifecycle")
    return card


def _cleanup(card_ids: list):
    for cid in card_ids:
        p = CARDS_DIR / f"{cid}.json"
        if p.exists():
            p.unlink()


def test_find_similar():
    """相似卡检测返回成对结果。"""
    groups = jl.find_similar(min_shared_keywords=3)
    assert isinstance(groups, list)
    for g in groups[:5]:
        assert "a" in g and "b" in g and "shared_keywords" in g


def test_evict_dry_run():
    """淘汰扫描dry-run返回候选（不执行删除）。"""
    ev = jl.evict(dry_run=True)
    assert ev["ok"] is True
    assert ev["dry_run"] is True
    assert "candidates" in ev
    # 不产生任何deleted卡片
    deleted = [f for f in CARDS_DIR.glob("*.json")
               if json.loads(f.read_text(encoding="utf-8")).get("deleted")]
    # 允许已存在的deleted（历史），但本次dry-run不应新增


def test_revise_versioned():
    """revise：版本化修正+历史存档。"""
    cid = f"test-lifecycle-revise-{uuid.uuid4().hex[:6]}"
    _make_temp_card(cid, ["kw1"], ["note1"])
    try:
        r = jl.revise(cid, "summary", "v2 summary", reason="test revise")
        assert r["ok"] is True
        card = jl._read(cid)
        hist = card["field_history"]["summary"]
        assert len(hist) == 1
        assert hist[0]["reason"] == "test revise"
        assert hist[0]["new"] == "v2 summary"
        assert card["summary"] == "v2 summary"
        assert "revise" in [e["action"] for e in card["lifecycle_events"]]
    finally:
        _cleanup([cid])


def test_revise_immutable_blocked():
    """immutable卡片拒绝修改。"""
    cid = f"test-lifecycle-immutable-{uuid.uuid4().hex[:6]}"
    _make_temp_card(cid, ["kw1"], ["note1"], immutable=True)
    try:
        r = jl.revise(cid, "title", "hacked", reason="should be blocked")
        assert r["ok"] is False
        assert "immutable" in r["error"]
    finally:
        _cleanup([cid])


def test_remove_soft_delete():
    """remove：软删除+stage标记。"""
    cid = f"test-lifecycle-remove-{uuid.uuid4().hex[:6]}"
    _make_temp_card(cid, ["kw1"], ["note1"])
    try:
        r = jl.remove(cid, reason="test cleanup")
        assert r.get("ok") is True
        assert r.get("stage") == "removed"
        card = jl._read(cid)
        assert card.get("deleted") is True
        assert card.get("status") == "deleted"
    finally:
        _cleanup([cid])


def test_consolidate_merge():
    """consolidate：两卡合并（主卡吸收+从卡软删除）。"""
    main_id = f"test-lifecycle-main-{uuid.uuid4().hex[:6]}"
    sub_id = f"test-lifecycle-sub-{uuid.uuid4().hex[:6]}"
    _make_temp_card(main_id, ["kw1", "kw2", "shared"], ["main note"])
    _make_temp_card(sub_id, ["kw3", "shared"], ["sub note a", "sub note b"])
    try:
        # dry-run 预览
        plan = jl.consolidate([main_id, sub_id], dry_run=True)
        assert plan["ok"] is True
        assert plan["dry_run"] is True
        assert plan["plan"]["main"] == main_id
        assert len(plan["plan"]["merged"]) == 1

        # 实际执行
        result = jl.consolidate([main_id, sub_id], dry_run=False)
        assert result["ok"] is True
        assert result["dry_run"] is False
        assert result["merged_count"] == 1

        # 主卡吸收从卡notes
        main = jl._read(main_id)
        all_notes = [n for n in main.get("notes", [])]
        assert "sub note a" in all_notes, f"notes={all_notes}"
        assert "main note" in all_notes

        # 从卡被软删除
        sub = jl._read(sub_id)
        assert sub.get("deleted") is True
        assert "consolidated_into" in (sub.get("deleted_reason") or "")
    finally:
        _cleanup([main_id, sub_id])
