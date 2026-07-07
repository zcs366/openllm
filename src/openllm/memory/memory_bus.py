"""
MemoryBus — 统一记忆总线

ISA和ICE的唯一记忆接口。替代五套存储的直连模式。

核心原则：
1. 单一写入口 — 所有记忆写入经过MemoryBus → 免疫检查 → 路由到provider
2. 统一检索接口 — ICE不需要知道记忆在哪个provider
3. Provider注册制 — 新增记忆源只需实现MemoryProvider接口
4. 向后兼容 — 现有API不删，MemoryBus是新增层

数据流：
  写入: WriteRequest → MemoryBus.write() → provider.store() → WriteResult
  检索: Query → MemoryBus.query() → provider.search() → [MemoryRecord]
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger("openllm.memory_bus")


# ═══════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════

@dataclass
class MemoryRecord:
    """统一记忆记录——所有provider返回此格式"""
    record_id: str
    content: str
    source: str             # "isa/causal" | "jiak" | "recall" | "delta" | "capsule"
    record_type: str        # "causal" | "opinion" | "lesson" | "event" | "insight"
    importance: float       # 0.0-1.0
    temperature: float      # 0.0-1.0（衰减后）
    trust_level: str        # "trusted" | "internal" | "untrusted" | "unknown"
    tags: List[str]
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0      # 检索相关性分数（检索时填充）
    provider: str = ""      # 实际provider名称（检索时填充）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "content": self.content,
            "source": self.source,
            "record_type": self.record_type,
            "importance": self.importance,
            "temperature": self.temperature,
            "trust_level": self.trust_level,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "context": self.context,
            "score": self.score,
            "provider": self.provider,
        }


@dataclass
class WriteRequest:
    """统一写入请求"""
    content: str
    source: str
    trust_level: str = "internal"
    record_type: str = "insight"
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WriteResult:
    """写入结果"""
    success: bool
    record_id: str = ""
    provider: str = ""
    blocked: bool = False
    reason: str = ""


@dataclass
class Query:
    """统一检索查询"""
    text: str
    top_k: int = 5
    token_budget: int = 1000
    record_types: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    min_importance: float = 0.0
    min_temperature: float = 0.0


# ═══════════════════════════════════════════════
# Provider协议
# ═══════════════════════════════════════════════

@runtime_checkable
class MemoryProvider(Protocol):
    """记忆Provider协议"""

    @property
    def name(self) -> str: ...

    @property
    def priority(self) -> int: ...

    def search(self, query: Query) -> List[MemoryRecord]: ...

    def store(self, request: WriteRequest) -> WriteResult: ...

    def count(self) -> int: ...

    def health(self) -> Dict[str, Any]: ...


# ═══════════════════════════════════════════════
# Token估算（与ICE共享）
# ═══════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """估算文本token数。中文1字≈1.5token，英文1词≈1token。"""
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_words = len([w for w in text.split() if w.isascii() and w.isalpha()])
    return int(chinese_chars * 1.5 + english_words)


# ═══════════════════════════════════════════════
# MemoryBus
# ═══════════════════════════════════════════════

class MemoryBus:
    """
    统一记忆总线。

    ISA和ICE通过MemoryBus读写记忆，不需要知道底层provider的实现细节。

    用法：
        bus = MemoryBus()
        bus.register(JiakProvider())
        bus.register(RecallProvider())

        # 写入
        result = bus.write(WriteRequest(content="端口配错", source="isa"))

        # 检索
        records = bus.query(Query(text="端口配置", top_k=5))
    """

    def __init__(self):
        self._providers: Dict[str, MemoryProvider] = {}
        self._write_log: List[WriteResult] = []
        self._max_log = 100

    # ── Provider管理 ──

    def register(self, provider: MemoryProvider) -> None:
        """注册记忆provider"""
        if provider.name in self._providers:
            logger.warning(f"Provider '{provider.name}' 已存在，覆盖注册")
        self._providers[provider.name] = provider
        logger.info(f"✅ 注册provider: {provider.name} (priority={provider.priority})")

    def unregister(self, name: str) -> bool:
        """注销provider"""
        if name in self._providers:
            del self._providers[name]
            logger.info(f"❌ 注销provider: {name}")
            return True
        return False

    def get_providers(self) -> List[str]:
        """列出所有已注册provider名称"""
        return sorted(self._providers.keys(),
                      key=lambda n: self._providers[n].priority)

    def _sorted_providers(self) -> List[MemoryProvider]:
        """按priority排序的provider列表"""
        return sorted(self._providers.values(), key=lambda p: p.priority)

    # ── 写入路径 ──

    def write(self, request: WriteRequest) -> WriteResult:
        """
        统一写入——路由到合适的provider。

        路由规则：
        1. 按priority顺序，询问每个provider
        2. 第一个返回success=True的provider接受写入
        3. 全部不接受 → 返回失败
        4. 结果记入审计日志
        """
        for provider in self._sorted_providers():
            try:
                result = provider.store(request)
                if result.success:
                    result.provider = provider.name
                    self._log_write(result)
                    logger.debug(f"写入成功: {provider.name} → {result.record_id}")
                    return result
            except Exception as e:
                logger.error(f"Provider '{provider.name}' 写入异常: {e}")
                continue

        # 所有provider都不接受
        result = WriteResult(
            success=False,
            blocked=True,
            reason=f"无provider接受写入 (source={request.source}, type={request.record_type})",
        )
        self._log_write(result)
        return result

    # ── 检索路径 ──

    def query(self, query: Query) -> List[MemoryRecord]:
        """
        统一检索——多源查询 + 融合排序。

        流程：
        1. 按priority顺序查询各provider
        2. 合并所有结果
        3. 按score降序排序
        4. 去重（同一record_id只保留最高分）
        5. 贪心填充token_budget
        6. 返回top_k条
        """
        all_records: List[MemoryRecord] = []

        for provider in self._sorted_providers():
            try:
                records = provider.search(query)
                all_records.extend(records)
            except Exception as e:
                logger.error(f"Provider '{provider.name}' 检索异常: {e}")
                continue

        if not all_records:
            return []

        # 按score降序排序
        all_records.sort(key=lambda r: r.score, reverse=True)

        # 去重（同一record_id只保留最高分）
        seen_ids = set()
        deduped = []
        for r in all_records:
            if r.record_id not in seen_ids:
                seen_ids.add(r.record_id)
                deduped.append(r)

        # 贪心填充token_budget
        selected = []
        used_tokens = 0

        for r in deduped:
            if len(selected) >= query.top_k:
                break
            est_tokens = estimate_tokens(r.content)
            if used_tokens + est_tokens > query.token_budget:
                continue
            selected.append(r)
            used_tokens += est_tokens

        return selected

    # ── 管理 ──

    def stats(self) -> Dict[str, Any]:
        """全局统计"""
        provider_stats = {}
        for name, provider in self._providers.items():
            try:
                provider_stats[name] = {
                    "count": provider.count(),
                    "priority": provider.priority,
                    "health": provider.health(),
                }
            except Exception as e:
                provider_stats[name] = {"error": str(e)}

        return {
            "providers": len(self._providers),
            "provider_list": self.get_providers(),
            "write_log_size": len(self._write_log),
            "total_records": sum(
                s.get("count", 0) for s in provider_stats.values()
                if isinstance(s, dict) and "count" in s
            ),
            "provider_stats": provider_stats,
        }

    def health_check(self) -> Dict[str, Any]:
        """所有provider健康检查"""
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = provider.health()
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        return results

    def get_write_log(self, last_n: int = 20) -> List[WriteResult]:
        """最近N条写入记录（审计用）"""
        return self._write_log[-last_n:]

    def _log_write(self, result: WriteResult) -> None:
        """记录写入到审计日志"""
        self._write_log.append(result)
        if len(self._write_log) > self._max_log:
            self._write_log = self._write_log[-self._max_log:]
