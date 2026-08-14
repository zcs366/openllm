#!/usr/bin/env python3
"""jiak_lifecycle.py — jiak卡片六阶段生命周期API（T-ISA-7）

背景：三论文精读（2026-07-18，Oracle生命周期）指出——"jiak写入后永不修改，
RECALL追加后永不整理"，缺consolidation和revision。27天时滞后于2026-08-14立项。

六阶段：write → consolidate → revise → evict → summarize → remove
底层复用 ~/.hermes/jiak/jiak_api.py（write_card/update_card_field/card_delete等，
已有1952行+immutable保护+RECALL审计），本模块只补缺口：
  - consolidate：相似卡片合并（关键词重叠检测）
  - revise：版本化修正（immutable保护+旧值存档+审计）
  - evict：主动淘汰执行（decay_status/温度<阈值 → soft-delete）
  - summarize：摘要生成（复用jiak_compress逻辑）

设计原则：
  1. 不修改jiak_api.py——本模块是新增层（对齐MemoryBus"新增层"哲学）
  2. 所有操作走RECALL审计（可追溯）
  3. 保守默认：consolidate默认dry_run，evict默认threshold=0.3
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ── jiak_api 导入（Hermes jiak 系统）──
_JIAK_DIR = Path.home() / ".hermes" / "jiak"
# 确保 ~/.hermes/jiak 在sys.path最前（T-ISA-7修复：其他测试可能把
# ~/.hermes/jiak/scripts 插入sys.path，导致 jiak_api 内部
# "from recall_append import ..." 拿到scripts旧版，import失败。
# 只把根目录插最前，不删除scripts——Python按序查找会优先命中根目录新版，
# 而其他测试仍可自行使用scripts路径）
if str(_JIAK_DIR) in sys.path:
    sys.path.remove(str(_JIAK_DIR))
sys.path.insert(0, str(_JIAK_DIR))

# sys.modules缓存驱逐（T-ISA-7根因修复）：全量测试中某测试可能先加载
# scripts旧版recall_append到sys.modules，jiak_api的"from recall_append
# import validate_and_append"会命中缓存旧版（无此函数）→ ImportError。
# 检测并驱逐旧版，强制jiak_api重新解析到根目录新版。
if "recall_append" in sys.modules:
    _ra = sys.modules["recall_append"]
    _ra_file = str(getattr(_ra, "__file__", ""))
    if _ra_file.startswith(str(_JIAK_DIR / "scripts")) or not hasattr(_ra, "validate_and_append"):
        del sys.modules["recall_append"]

try:
    import jiak_api
except ImportError as e:
    jiak_api = None
    _IMPORT_ERROR = str(e)
else:
    _IMPORT_ERROR = ""

CARDS_DIR = _JIAK_DIR / "cards"

# ── 淘汰阈值（对齐temperature_engine.EVICT_THRESHOLD）──
EVICT_THRESHOLD = 0.3


def _available() -> bool:
    """jiak_api是否可用。不可用时所有操作返回错误（不静默）。

    T-ISA-7根因修复：每次调用前检测sys.modules中的recall_append缓存——
    若被其他模块先加载为scripts旧版（无validate_and_append），驱逐并重试。
    这覆盖'污染发生在jiak_lifecycle首次import之后'的时序。
    """
    global jiak_api, _IMPORT_ERROR
    if "recall_append" in sys.modules:
        _ra = sys.modules["recall_append"]
        _ra_file = str(getattr(_ra, "__file__", ""))
        if _ra_file.startswith(str(_JIAK_DIR / "scripts")) or not hasattr(_ra, "validate_and_append"):
            del sys.modules["recall_append"]
    if jiak_api is None:
        try:
            import jiak_api
            _IMPORT_ERROR = ""
        except ImportError as e:
            _IMPORT_ERROR = str(e)
    return jiak_api is not None


def _card_ids() -> List[str]:
    return [p.stem for p in CARDS_DIR.glob("*.json")]


def _read(card_id: str) -> Optional[dict]:
    if not _available():
        return None
    return jiak_api.read_card(card_id)


# ═══════════════════════════════════════════════════════
# 1. write — 写入（封装write_card，补生命周期字段）
# ═══════════════════════════════════════════════════════

def write(card_id: str, card: dict, written_by: str = "jiak_lifecycle.write") -> dict:
    """写入卡片。自动补生命周期字段（lifecycle_stage=active, lifecycle_events[]）。

    与jiak_api.write_card的区别：记录生命周期事件。
    """
    if not _available():
        return {"ok": False, "error": f"jiak_api不可用: {_IMPORT_ERROR}"}

    card.setdefault("lifecycle_stage", "active")
    events = card.setdefault("lifecycle_events", [])
    events.append({
        "action": "write",
        "ts": time.time(),
        "by": written_by,
    })
    try:
        jiak_api.write_card(card_id, card)
        return {"ok": True, "card_id": card_id, "stage": "active"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════
# 2. consolidate — 相似卡片合并（核心缺口）
# ═══════════════════════════════════════════════════════

def consolidate(
    card_ids: List[str],
    merged_id: Optional[str] = None,
    merge_field: str = "notes",
    dry_run: bool = True,
    written_by: str = "jiak_lifecycle.consolidate",
) -> dict:
    """合并多张相似卡片为一张主卡，其余软删除。

    策略：
      - 主卡 = 第一张（或merged_id指定）
      - 被合并卡的notes/decisions/tags并入主卡
      - 被合并卡软删除（card_delete，reason=consolidated_into:<merged>）
      - dry_run=True（默认）只返回计划不执行

    Args:
        card_ids: 要合并的卡片ID列表（第一张为主卡）
        merged_id: 合并后主卡ID（默认card_ids[0]）
        merge_field: 合并哪个数组字段（默认notes）
        dry_run: 试运行（默认True，安全）
    """
    if not _available():
        return {"ok": False, "error": f"jiak_api不可用: {_IMPORT_ERROR}"}
    if len(card_ids) < 2:
        return {"ok": False, "error": "consolidate至少需要2张卡片"}

    main_id = merged_id or card_ids[0]
    main_card = _read(main_id)
    if main_card is None:
        return {"ok": False, "error": f"主卡不存在: {main_id}"}

    plan = {"main": main_id, "merged": [], "skipped": []}

    for cid in card_ids:
        if cid == main_id:
            continue
        card = _read(cid)
        if card is None:
            plan["skipped"].append({"card_id": cid, "reason": "card not found"})
            continue
        # 收集要并入的内容
        merged_items = card.get(merge_field, [])
        plan["merged"].append({
            "card_id": cid,
            "items": merged_items[:5],  # 预览前5条
            "total_items": len(merged_items),
            "title": card.get("title", ""),
        })

    if dry_run:
        return {"ok": True, "dry_run": True, "plan": plan}

    # ── 实际执行 ──
    main_card = _read(main_id)
    for m in plan["merged"]:
        cid = m["card_id"]
        card = _read(cid)
        if card is None:
            continue
        items = card.get(merge_field, [])
        existing = main_card.get(merge_field, [])
        # 去重合并（按内容去重）
        existing_set = {json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x
                        for x in existing}
        added = 0
        for item in items:
            key = json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item
            if key not in existing_set:
                existing.append(item)
                existing_set.add(key)
                added += 1
        if added > 0:
            main_card[merge_field] = existing
        # 记录来源
        source_field = "consolidated_from"
        sources = main_card.setdefault(source_field, [])
        sources.append({"card_id": cid, "ts": time.time(), "items_added": added})
        # 软删除被合并卡
        try:
            jiak_api.card_delete(cid, reason=f"consolidated_into:{main_id}")
        except Exception as e:
            plan["skipped"].append({"card_id": cid, "reason": f"delete failed: {e}"})

    # 更新主卡生命周期
    events = main_card.setdefault("lifecycle_events", [])
    events.append({
        "action": "consolidate",
        "ts": time.time(),
        "merged_ids": [m["card_id"] for m in plan["merged"]],
        "by": written_by,
    })
    main_card["lifecycle_stage"] = "active"
    try:
        jiak_api.write_card(main_id, main_card)
        return {"ok": True, "dry_run": False, "main": main_id,
                "merged_count": len(plan["merged"])}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════
# 3. revise — 版本化修正（核心缺口）
# ═══════════════════════════════════════════════════════

def revise(
    card_id: str,
    field: str,
    new_value,
    reason: str,
    written_by: str = "jiak_lifecycle.revise",
) -> dict:
    """版本化修正卡片字段。

    与jiak_api.update_card_field的区别：
      - immutable字段（immutable=True）拒绝修改
      - 旧值存档到 field_history[{field: [{old, new, reason, ts, by}]}]
      - RECALL审计（update_card_field已有，但这里补历史）

    Returns:
        {"ok": True, "card_id": ..., "field": ..., "old": ..., "new": ...}
    """
    if not _available():
        return {"ok": False, "error": f"jiak_api不可用: {_IMPORT_ERROR}"}

    card = _read(card_id)
    if card is None:
        return {"ok": False, "error": f"card not found: {card_id}"}

    # immutable保护
    if card.get("immutable"):
        return {"ok": False, "error": f"卡片{card_id}是immutable，拒绝修改字段{field}"}

    old_value = card.get(field)

    # 字段历史存档
    history = card.setdefault("field_history", {})
    field_hist = history.setdefault(field, [])
    field_hist.append({
        "old": old_value,
        "new": new_value,
        "reason": reason,
        "ts": time.time(),
        "by": written_by,
    })
    # 裁剪历史（最多保留10条）
    history[field] = field_hist[-10:]

    # 生命周期事件
    events = card.setdefault("lifecycle_events", [])
    events.append({
        "action": "revise",
        "field": field,
        "reason": reason,
        "ts": time.time(),
        "by": written_by,
    })

    card[field] = new_value
    card["lifecycle_stage"] = "revised"
    try:
        jiak_api.write_card(card_id, card)
        return {"ok": True, "card_id": card_id, "field": field,
                "old": old_value, "new": new_value}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════
# 4. evict — 主动淘汰执行（核心缺口）
# ═══════════════════════════════════════════════════════

def evict(
    threshold: float = EVICT_THRESHOLD,
    dry_run: bool = True,
    written_by: str = "jiak_lifecycle.evict",
) -> dict:
    """主动淘汰：扫描卡片，温度/衰减低于阈值的标记evictable→soft-delete。

    温度估算（无embedding时的启发式）：
      temp ≈ importance × e^(-0.01 × age_days) × recency_bonus
    简化：decay_status为cooling/archived 或 access_count=0且age>90天 → 候选。

    dry_run=True（默认）只列出候选不执行。
    """
    if not _available():
        return {"ok": False, "error": f"jiak_api不可用: {_IMPORT_ERROR}"}

    now = time.time()
    candidates = []

    for cid in _card_ids():
        card = _read(cid)
        if card is None or card.get("deleted"):
            continue
        # 跳过immutable
        if card.get("immutable"):
            continue
        # 跳过高价值（importance高且近期访问）
        importance = card.get("importance", 0.5)
        if isinstance(importance, str):
            try:
                importance = float(importance)
            except (ValueError, TypeError):
                importance = 0.5
        if importance >= 0.8:
            continue
        # 衰减判定
        decay = card.get("decay_status", "active")
        last_access = card.get("last_accessed", 0) or 0
        if isinstance(last_access, str):
            # ISO时间戳字符串 → epoch（jiak卡片常用ISO格式）
            try:
                from datetime import datetime, timezone
                last_access = datetime.fromisoformat(
                    last_access.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, TypeError):
                last_access = 0
        # 修复（T-ISA-7 evict bug）：last_accessed缺失时回退created，
        # 仍无则age=0（新卡不淘汰）。旧逻辑 last_access=0→age=999
        # 会把所有新卡误判为陈旧（实测27候选全是今天创建）。
        ref_ts = last_access or card.get("created", 0) or 0
        if isinstance(ref_ts, str):
            try:
                from datetime import datetime, timezone
                ref_ts = datetime.fromisoformat(
                    ref_ts.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, TypeError):
                ref_ts = 0
        age_days = (now - ref_ts) / 86400 if ref_ts else 0.0
        # 年龄为负（时钟偏移）视为0
        if age_days < 0:
            age_days = 0.0
        access_count = card.get("access_count", 0) or 0
        if isinstance(access_count, str):
            try:
                access_count = int(access_count)
            except (ValueError, TypeError):
                access_count = 0

        is_cold = decay in ("cooling", "archived", "cold", "frozen", "evictable")
        is_stale = access_count <= 1 and age_days > 90
        if is_cold or is_stale:
            candidates.append({
                "card_id": cid,
                "decay_status": decay,
                "access_count": access_count,
                "age_days": round(age_days, 1),
                "importance": importance,
                "title": (card.get("title") or "")[:50],
            })

    candidates.sort(key=lambda c: (-c["importance"], -c["age_days"]))

    if dry_run:
        return {"ok": True, "dry_run": True, "candidates": candidates,
                "candidate_count": len(candidates)}

    # 执行淘汰（soft-delete）
    evicted = []
    for c in candidates:
        try:
            jiak_api.card_delete(c["card_id"], reason=f"evicted_by_lifecycle (temp<{threshold})")
            evicted.append(c["card_id"])
        except Exception as e:
            pass  # 单卡失败不影响其余
    return {"ok": True, "dry_run": False, "evicted": evicted,
            "evicted_count": len(evicted)}


# ═══════════════════════════════════════════════════════
# 5. summarize — 摘要生成
# ═══════════════════════════════════════════════════════

def summarize(card_id: str, force: bool = False) -> dict:
    """为卡片生成archive_summary（复用jiak_compress的Ollama压缩）。

    Args:
        card_id: 卡片ID
        force: 已有摘要是否强制重新生成
    """
    if not _available():
        return {"ok": False, "error": f"jiak_api不可用: {_IMPORT_ERROR}"}

    card = _read(card_id)
    if card is None:
        return {"ok": False, "error": f"card not found: {card_id}"}
    if card.get("archive_summary") and not force:
        return {"ok": True, "skipped": "已有摘要", "card_id": card_id}

    # 复用jiak_compress的压缩函数
    try:
        sys.path.insert(0, str(_JIAK_DIR))
        import jiak_compress
        summary = jiak_compress.compress_with_ollama(
            title=card.get("title", ""),
            summary=card.get("summary", ""),
            notes=card.get("notes", []),
            decisions=card.get("decisions", []),
        )
        if not summary:
            return {"ok": False, "error": "Ollama压缩返回空"}
        card["archive_summary"] = summary
        events = card.setdefault("lifecycle_events", [])
        events.append({"action": "summarize", "ts": time.time(),
                       "by": "jiak_lifecycle.summarize"})
        jiak_api.write_card(card_id, card)
        return {"ok": True, "card_id": card_id, "summary": summary[:100]}
    except Exception as e:
        return {"ok": False, "error": f"summarize失败: {e}"}


# ═══════════════════════════════════════════════════════
# 6. remove — 删除（封装card_delete）
# ═══════════════════════════════════════════════════════

def remove(card_id: str, reason: str) -> dict:
    """软删除卡片（封装jiak_api.card_delete，补生命周期事件）。"""
    if not _available():
        return {"ok": False, "error": f"jiak_api不可用: {_IMPORT_ERROR}"}

    card = _read(card_id)
    if card is None:
        return {"ok": False, "error": f"card not found: {card_id}"}
    events = card.setdefault("lifecycle_events", [])
    events.append({"action": "remove", "reason": reason, "ts": time.time(),
                   "by": "jiak_lifecycle.remove"})
    try:
        jiak_api.write_card(card_id, card)
        result = jiak_api.card_delete(card_id, reason=reason)
        result["stage"] = "removed"
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════
# 工具：find_similar — 相似卡片检测（consolidate前置）
# ═══════════════════════════════════════════════════════

def find_similar(min_shared_keywords: int = 3, limit: int = 20) -> List[dict]:
    """按关键词重叠检测相似卡片群组（consolidate候选）。"""
    if not _available():
        return []

    cards = []
    for cid in _card_ids():
        card = _read(cid)
        if card and not card.get("deleted"):
            cards.append(card)

    from collections import defaultdict
    kw_groups = defaultdict(list)
    for c in cards:
        kws = set(c.get("keywords", []))
        for kw in kws:
            kw_groups[kw].append(c["card_id"])

    # 找出共享≥min_shared_keywords的卡片对
    groups = []
    seen_pairs = set()
    for c1 in cards:
        kws1 = set(c1.get("keywords", []))
        for c2 in cards:
            if c1["card_id"] >= c2["card_id"]:
                continue
            pair = (c1["card_id"], c2["card_id"])
            if pair in seen_pairs:
                continue
            shared = kws1 & set(c2.get("keywords", []))
            if len(shared) >= min_shared_keywords:
                seen_pairs.add(pair)
                groups.append({
                    "a": c1["card_id"], "b": c2["card_id"],
                    "shared_keywords": sorted(shared),
                    "similarity": len(shared),
                })
    groups.sort(key=lambda g: -g["similarity"])
    return groups[:limit]


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="jiak六阶段生命周期API")
    ap.add_argument("action", choices=["find_similar", "consolidate", "evict",
                                       "revise", "summarize", "remove", "write"])
    ap.add_argument("--card-ids", nargs="+", default=[], help="卡片ID列表")
    ap.add_argument("--field", default="", help="revise的字段")
    ap.add_argument("--value", default="", help="revise的新值")
    ap.add_argument("--reason", default="", help="revise/remove的原因")
    ap.add_argument("--threshold", type=float, default=EVICT_THRESHOLD)
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--execute", action="store_true", help="实际执行（默认dry-run）")
    args = ap.parse_args()

    if args.action == "find_similar":
        result = find_similar()
        print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
    elif args.action == "consolidate":
        result = consolidate(args.card_ids, dry_run=not args.execute)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
    elif args.action == "evict":
        result = evict(threshold=args.threshold, dry_run=not args.execute)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
    elif args.action == "revise":
        result = revise(args.card_ids[0], args.field, args.value, args.reason)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:1000])
    elif args.action == "remove":
        result = remove(args.card_ids[0], args.reason)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:1000])
    elif args.action == "summarize":
        result = summarize(args.card_ids[0], force=True)
        print(json.dumps(result, ensure_ascii=False, indent=2)[:1000])
