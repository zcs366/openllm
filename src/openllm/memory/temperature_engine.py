"""
TemperatureEngine — 牛顿冷却定律记忆温度计算
=============================================

移植自 jiak/temperature.py (63行) + working_memory.py (456行) + card_temperature.py (311行)。
为 openLLM 的 ICE 管线提供统一的温度计算能力。

核心公式:
  temperature = importance × e^(-λt × emotion_factor) + heat_from_access

λ值按记忆类型分:
  insight: 0.01 (慢衰减) / preference: 0.001 (极慢) / event: 0.1 (快) / noise: 0.5 (极速)

新增(来自card_temperature.py):
  - emotion_factor: 深度探索(高importance低频) vs 高频浅访问(低importance高频)
  - topic_relevance: 卡片与近期会话的话题相关性(可选，需提供recent_keywords)

用途:
  1. UShapedContextEngine: 用温度替代天数判断entry衰减状态
  2. ContextPressureMonitor: 用温度决定淘汰优先级
  3. MemoryBus: 写入时自动计算temperature字段
"""
import math
import re
import time
from datetime import datetime, timezone
from typing import Optional, Set

# ── 衰减系数 ──
LAMBDA_MAP = {
    "insight": 0.01,       # 知识洞察：慢衰减
    "preference": 0.001,   # 用户偏好：极慢衰减
    "event": 0.1,          # 事件记忆：快衰减
    "noise": 0.5,          # 噪声：极速衰减
}

BASE_HEAT = 3.0
MAX_TEMP = 10.0

# 因果效应参与遗忘决策：因果记忆的温度加成（跟BASE_HEAT同级）
CAUSAL_BONUS = 3.0

# ── 情感权重参数(来自card_temperature.py) ──
EMOTION_ENABLED = True

# ── 话题相关性参数 ──
TOPIC_RELEVANCE_BETA = 0.5    # 话题相关性对温度的增益权重

# ── 温度→状态阈值 ──
HOT_THRESHOLD = 7.0
WARM_THRESHOLD = 3.0
COLD_THRESHOLD = 1.0
EVICT_THRESHOLD = 0.3   # 低于此温度→淘汰候选


def _extract_keywords(text: str) -> Set[str]:
    """从文本中提取关键词集合(中英文混合)。中文2-gram，英文3+字符。"""
    if not text:
        return set()
    kws = set()
    for m in re.finditer(r'[\u4e00-\u9fff]{2,}', text):
        word = m.group()
        for i in range(len(word) - 1):
            kws.add(word[i:i+2])
    for m in re.finditer(r'[a-zA-Z]{3,}', text):
        kws.add(m.group().lower())
    return kws


def _compute_emotion_factor(importance: float, access_count: int) -> float:
    """情感权重：深度探索(高importance低频)衰减更慢，高频浅访问衰减更快。

    来自card_temperature.py的酒神启示(2026-06-30):
    单次深层探讨(imp=10) ≠ 百次日常提醒(acc=100, imp=1)
    """
    amplitude = math.sqrt(max(importance, 1.0) / 5.0)
    return 1.0 / (1.0 + math.log(1.0 + max(access_count, 0) * amplitude))


def _compute_topic_relevance(
    entry_keywords: Set[str],
    recent_keywords: Set[str],
) -> float:
    """计算entry与近期会话的话题相关性(0.0-1.0)。

    如果entry的关键词与最近会话高频词重叠多，说明该记忆
    与当前用户关注的主题相关，衰减应更慢。

    Args:
        entry_keywords: entry的关键词集合
        recent_keywords: 近期会话的关键词集合
    """
    if not entry_keywords or not recent_keywords:
        return 0.0
    overlap = entry_keywords & recent_keywords
    return min(len(overlap) / max(len(entry_keywords), 1), 1.0)


def calculate_temperature(
    importance: float = 5.0,
    days_since_access: int = 0,
    memory_type: str = "insight",
    access_count: int = 0,
    topic_relevance: float = 0.0,
    causal_weight: float = 0.0,
) -> float:
    """牛顿冷却定律计算记忆温度(增强版+因果效应)。

    Args:
        importance: 重要性 (0-10)
        days_since_access: 距上次访问的天数
        memory_type: 记忆类型 (insight/preference/event/noise)
        access_count: 累计访问次数（访问越多越热）
        topic_relevance: 话题相关性(0.0-1.0)，与近期会话相关时衰减更慢
        causal_weight: 因果效应权重(0-1)，0=无因果数据，1=完全偏差。因果效应参与遗忘决策。

    Returns:
        温度值 (0.0 - MAX_TEMP)
    """
    lambda_val = LAMBDA_MAP.get(memory_type, 0.1)

    # 情感权重：深度探索衰减慢，高频浅访问衰减快
    if EMOTION_ENABLED:
        emotion_factor = _compute_emotion_factor(importance, access_count)
    else:
        emotion_factor = 1.0

    # 话题相关性：与近期会话相关时衰减更慢
    relevance_boost = 1.0 + TOPIC_RELEVANCE_BETA * max(0.0, min(1.0, topic_relevance))

    # 最终温度
    decay = importance * math.exp(-lambda_val * emotion_factor * max(days_since_access, 0)) * relevance_boost
    heat = BASE_HEAT / (1 + access_count * 0.1)
    # 因果效应参与遗忘决策：causal_weight越大=教训越深=越不该忘
    causal_boost = causal_weight * CAUSAL_BONUS
    return round(min(decay + heat + causal_boost, MAX_TEMP), 2)


def temperature_state(temp: float) -> str:
    """温度→状态标记。"""
    if temp > HOT_THRESHOLD:
        return "hot"
    elif temp > WARM_THRESHOLD:
        return "warm"
    elif temp > COLD_THRESHOLD:
        return "cold"
    elif temp > EVICT_THRESHOLD:
        return "frozen"
    else:
        return "evictable"


def temperature_state_emoji(temp: float) -> str:
    """温度→emoji标记。"""
    state = temperature_state(temp)
    return {"hot": "🔥", "warm": "♨️", "cold": "❄️", "frozen": "💀", "evictable": "🕳️"}.get(state, "?")


def days_since(iso_timestamp: str) -> int:
    """从ISO时间戳计算距今天数。"""
    if not iso_timestamp:
        return 0
    try:
        ts = iso_timestamp.replace("Z", "+00:00")
        if "+" not in ts and "-" not in ts[10:]:
            ts += "+00:00"
        dt = datetime.fromisoformat(ts)
        now = datetime.now(timezone.utc) if dt.tzinfo else datetime.now()
        return max(0, (now.replace(tzinfo=None) - dt.replace(tzinfo=None)).days)
    except Exception:
        return 0


def compute_entry_temperature(
    importance: float = 5.0,
    last_accessed: str = "",
    memory_type: str = "insight",
    access_count: int = 0,
    causal_weight: float = 0.0,
) -> float:
    """从entry元数据计算当前温度（便捷接口）。

    兼容 openLLM MemoryRecord 的 metadata 字段。
    causal_weight: 因果效应权重(0-1)，从因果记忆的delta_magnitude获取。
    """
    days = days_since(last_accessed)
    return calculate_temperature(importance, days, memory_type, access_count, causal_weight=causal_weight)


def should_evict(temp: float) -> bool:
    """判断是否应淘汰。"""
    return temp < EVICT_THRESHOLD


def eviction_priority(temp: float) -> float:
    """淘汰优先级（越低越先淘汰）。返回 inf 的 entry 永不淘汰。"""
    return temp


# ════════════════════════════════════════════════════════════════
# 通电Phase 1（2026-08-14）：扫描淘汰
# ════════════════════════════════════════════════════════════════

def scan_for_eviction(
    capsule_dir: str = "~/.openllm/capsules",
    threshold: float = EVICT_THRESHOLD,
    max_scan: int = 2000,
) -> list:
    """扫描所有v06胶囊，找出温度低于阈值的淘汰候选。

    Returns:
        [{"session_id": str, "file": str, "temperature": float,
          "state": str, "importance": float, "insight": str}]
    """
    from pathlib import Path
    import json as _json

    cap_dir = Path(capsule_dir).expanduser()
    if not cap_dir.exists():
        return []

    evictable = []
    scanned = 0
    for f in sorted(cap_dir.glob("v06_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if scanned >= max_scan:
            break
        scanned += 1
        try:
            data = _json.loads(f.read_text(encoding="utf-8"))
            ts = data.get("timestamp", 0)
            importance = 5.0
            access_count = 0
            # 从decisions中提取importance
            for d in data.get("decisions", []):
                meta = d.get("metadata", {})
                if "importance" in meta and meta["importance"] is not None:
                    importance = float(meta["importance"])
                if "access_count" in meta:
                    access_count = int(meta["access_count"])
            days = max(0, (time.time() - ts) / 86400) if ts else 999
            temp = calculate_temperature(importance, int(days), "insight", access_count)
            state = temperature_state(temp)
            if should_evict(temp):
                insight = (data.get("insights", [""])[0] if data.get("insights") else "")[:100]
                evictable.append({
                    "session_id": data.get("session_id", "?"),
                    "file": f.name,
                    "temperature": temp,
                    "state": state,
                    "importance": importance,
                    "insight": insight,
                })
        except Exception:
            continue

    # 按温度升序排列（最冷的先淘汰）
    evictable.sort(key=lambda x: x["temperature"])
    return evictable
