"""
MistakeLedger桥接模块 — 连接错误记录与因果记忆/免疫系统

核心逻辑：
  MistakeLedger记录 → 自动转换为CausalMemory教训
  MistakeLedger模式 → 触发ImmuneSystem异常检测

设计原则：
  - 单向桥接（MistakeLedger→CausalMemory，不反向）
  - 不修改现有模块接口，只在上层做胶水
  - 延迟执行（不阻塞主流程）
"""

import time
from typing import Optional

from .mistake_ledger import MistakeLedger
from .causal_memory import CausalMemory, TrustLevel


def mistake_to_causal(
    mistake: dict,
    trust_level: TrustLevel = TrustLevel.INTERNAL,
    action_signature: str = "",
) -> CausalMemory:
    """
    将一条MistakeLedger记录转换为CausalMemory条目。

    映射规则：
      what → delta（差距描述）
      why → lesson（学到的教训）
      agent → action_signature（谁做的操作）
      severity → delta_magnitude（严重程度映射为差距大小）

    Args:
        mistake: MistakeLedger的一条记录
        trust_level: 信任级别（默认INTERNAL=Agent自身推理）
        action_signature: 操作签名（为空则用what代替）

    Returns:
        CausalMemory条目
    """
    severity_map = {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8,
        "critical": 1.0,
    }

    return CausalMemory(
        memory_id=f"causal-from-{mistake.get('id', 'unknown')}",
        created_at=time.time(),
        action_signature=action_signature or mistake.get("what", "")[:100],
        context_features=mistake.get("tags", []),
        prediction="",  # MistakeLedger不记录预测
        prediction_confidence=0.0,
        actual_result=mistake.get("what", ""),
        actual_success=False,  # 错误记录=操作不成功
        delta=mistake.get("what", ""),
        delta_magnitude=severity_map.get(mistake.get("severity", "medium"), 0.5),
        lesson=mistake.get("why", ""),
        source=f"mistake_ledger:{mistake.get('agent', 'unknown')}",
        trust_level=trust_level,
    )


def find_recurring_patterns(
    ledger: MistakeLedger,
    min_count: int = 3,
    window_hours: int = 24,
) -> list[dict]:
    """
    检测重复出现的错误模式。

    当同一agent在window小时内犯min_count次类似的错误时，
    触发ImmuneSystem的异常模式检测。

    Args:
        ledger: MistakeLedger实例
        min_count: 最少出现次数
        window_hours: 时间窗口（小时）

    Returns:
        重复模式列表，每项包含{agent, pattern, count, first_seen, last_seen}
    """
    since = time.strftime(
        "%Y-%m-%dT%H:%M:%S",
        time.gmtime(time.time() - window_hours * 3600),
    )
    records = ledger.query(since=since, limit=1000)

    # 按agent分组
    by_agent: dict[str, list[dict]] = {}
    for r in records:
        ag = r.get("agent", "unknown") or "unknown"
        by_agent.setdefault(ag, []).append(r)

    patterns = []
    for agent, agent_records in by_agent.items():
        if len(agent_records) >= min_count:
            # 简单模式检测：what的前20字相同的归为一类
            what_clusters: dict[str, list[dict]] = {}
            for r in agent_records:
                key = r.get("what", "")[:20]
                what_clusters.setdefault(key, []).append(r)

            for pattern_key, cluster in what_clusters.items():
                if len(cluster) >= min_count:
                    timestamps = [r.get("timestamp", "") for r in cluster]
                    patterns.append({
                        "agent": agent,
                        "pattern": pattern_key,
                        "count": len(cluster),
                        "first_seen": min(timestamps),
                        "last_seen": max(timestamps),
                        "severity": max(
                            (r.get("severity", "low") for r in cluster),
                            key=lambda s: {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(s, 0),
                        ),
                    })

    return patterns


def auto_causal_bridge(
    ledger: MistakeLedger,
    causal_store,  # CausalMemoryStore instance
    min_count: int = 1,
) -> int:
    """
    自动桥接：将MistakeLedger中的新记录批量转换为CausalMemory。

    Args:
        ledger: MistakeLedger实例
        causal_store: CausalMemoryStore（需有add方法）
        min_count: 至少几条错误才触发桥接（默认1=每条都转）

    Returns:
        成功转换的条数
    """
    # 读取最近的错误记录
    records = ledger.query(limit=100)
    converted = 0

    for r in records:
        try:
            causal = mistake_to_causal(r)
            # 写入因果记忆存储（如果支持）
            if hasattr(causal_store, "add"):
                causal_store.add(causal)
                converted += 1
            elif hasattr(causal_store, "store"):
                causal_store.store(causal)
                converted += 1
        except Exception:
            # 桥接失败不阻塞主流程
            continue

    return converted
