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
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger("openllm.memory_bus")

# ── Schema兼容性检查 ──
# 下游消费者（ICE/ISA）检索时期望的最小MemoryRecord字段集
DEFAULT_OUTPUT_SCHEMA: set = {
    "record_id", "content", "source", "record_type",
    "importance", "temperature", "trust_level",
    "tags", "timestamp", "score",
}

# ── 中文分词（jieba懒加载） ──
_jieba = None

def _get_jieba():
    global _jieba
    if _jieba is None:
        try:
            import jieba as _jb
            _jieba = _jb
        except ImportError:
            _jieba = False  # 标记不可用，fallback到split
    return _jieba

def tokenize(text: str) -> set:
    """
    中英文混合分词。中文走jieba，英文走split。
    返回去重关键词集合（小写，长度>1）。
    """
    if not text:
        return set()
    jb = _get_jieba()
    if jb is not None and jb is not False:
        words = list(jb.cut(text.lower()))
    else:
        words = text.lower().split()
    return {w for w in words if len(w) > 1 and w.strip()}


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


def calc_temperature(
    importance: float,
    last_accessed: float,
    heat: float = 0.0,
    access_count: int = 0,
    tags: Optional[List[str]] = None,
    decay_lambda: Optional[float] = None,
) -> float:
    """
    增强温度计算（集成到MemoryBus）。

    委托给temperature_engine.calculate_temperature（T-ISA-6统一双实现）。
    保留签名兼容：tags→memory_type映射，heat参数兼容（access热度）。
    """
    # tags → memory_type 映射（保持向后兼容的λ语义）
    tag_set = set(tg.lower() for tg in (tags or []))
    if "preference" in tag_set or "偏好" in tag_set:
        memory_type = "preference"
    elif "event" in tag_set or "事件" in tag_set:
        memory_type = "event"
    elif "noise" in tag_set or "噪声" in tag_set:
        memory_type = "noise"
    else:
        memory_type = "insight"

    from .temperature_engine import calculate_temperature as _calc, BASE_HEAT
    days = max(0, int((time.time() - last_accessed) / 86400)) if last_accessed else 0
    # 外部heat参数（历史调用）转为等效access_count：heat=BASE_HEAT/(1+access*0.1)
    if heat > 0 and access_count == 0:
        access_count = int(max(0, (BASE_HEAT / heat - 1) * 10))
    return _calc(
        importance=importance,
        days_since_access=days,
        memory_type=memory_type,
        access_count=access_count,
    )


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
    use_hybrid: bool = False  # 启用BM25+Embedding混合检索
    hybrid_weights: tuple = (0.6, 0.4)  # (BM25权重, Embedding权重)


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

    @property
    def input_schema(self) -> set:
        """该Provider在search()中使用的Query字段集合"""
        ...

    @property
    def output_schema(self) -> set:
        """该Provider在search()中产出的MemoryRecord字段集合"""
        ...

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
        self._immune = None  # 惰性初始化（T-ISA-4免疫接线）
        self._immune_failed = False  # 免疫初始化失败标记（避免重复尝试）
        self._register_builtin_providers()

    def _get_immune(self):
        """惰性获取记忆免疫系统（失败静默降级，不阻断写入）。"""
        if self._immune_failed:
            return None
        if self._immune is None:
            try:
                from .immune import MemoryImmuneSystem
                self._immune = MemoryImmuneSystem()
            except Exception as e:
                logger.debug(f"免疫系统初始化失败(降级): {e}")
                self._immune_failed = True
        return self._immune

    def _immune_check(self, request: WriteRequest) -> Optional[WriteResult]:
        """L1免疫检查（T-ISA-4）：untrusted来源/速率异常/异常模式 → 拦截。

        免疫系统不可用时静默放行（免疫是增强层，不是依赖层）。
        Returns:
            WriteResult(blocked) 或 None（放行）
        """
        immune = self._get_immune()
        if immune is None:
            return None
        try:
            from .immune import WriteRequest as ImmuneWriteRequest
            from .causal_memory import TrustLevel
            # trust_level字符串 → TrustLevel枚举
            try:
                trust_enum = TrustLevel(request.trust_level)
            except (ValueError, KeyError):
                trust_enum = TrustLevel.INTERNAL
            immune_req = ImmuneWriteRequest(
                content={"content": request.content, "source": request.source},
                source=request.source,
                trust_level=trust_enum,
                session_id=request.session_id,
            )
            allowed, threat, reason = immune.check_write(immune_req)
            if not allowed:
                logger.warning(f"🚫 MemoryBus写入被免疫拦截: {reason}")
                result = WriteResult(success=False, blocked=True, reason=reason)
                self._log_write(result)
                return result
        except Exception as e:
            logger.debug(f"免疫检查异常(降级放行): {e}")
        return None

    def _register_builtin_providers(self) -> None:
        """注册全部内置provider——通电Phase 1（2026-08-14）。

        六个provider全部注册，失败静默降级。
        每个provider支持无参构造（延迟初始化），无循环依赖风险。
        """
        _providers = [
            ("DeltaCapsuleProvider", ".providers.delta_capsule_provider", "DeltaCapsuleProvider"),
            ("JiakProvider", ".providers.jiak_provider", "JiakProvider"),
            ("RecallProvider", ".providers.recall_provider", "RecallProvider"),
            ("CausalProvider", ".providers.causal_provider", "CausalProvider"),
            ("UnifiedProvider", ".providers.unified_provider", "UnifiedProvider"),
            ("SourceIndexProvider", ".providers.source_index_provider", "SourceIndexProvider"),
        ]
        for name, mod_path, cls_name in _providers:
            try:
                import importlib
                mod = importlib.import_module(mod_path, package="openllm.memory")
                cls = getattr(mod, cls_name)
                self.register(cls())
            except Exception as e:
                logger.warning(f"{name}注册失败(降级): {e}")

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

    def record_causal(
        self,
        action: str,
        prediction: str,
        actual: str,
        success: bool,
        context: str = "",
    ) -> dict:
        """
        记录因果记忆——通过AutoCausalWriter写入。

        便捷方法：任何有MemoryBus引用的代码都可以调用此方法记录因果数据。
        不侵入write()主路径，按需调用。
        """
        try:
            from .auto_causal_writer import AutoCausalWriter
            from pathlib import Path
            writer = AutoCausalWriter()
            return writer.record(
                action=action,
                prediction=prediction,
                actual=actual,
                success=success,
                context=context,
            )
        except Exception as e:
            logger.warning(f"record_causal failed: {e}")
            return {}

    def write(self, request: WriteRequest) -> WriteResult:
        """
        统一写入——免疫检查 → 路由到合适的provider。

        路由规则：
        1. L1免疫检查（untrusted/速率异常/异常模式 → 拦截）
        2. 按priority顺序，询问每个provider
        3. 第一个返回success=True的provider接受写入
        4. 全部不接受 → 返回失败
        5. 结果记入审计日志
        """
        # ── L1免疫检查（T-ISA-4接线，免疫不可用时静默放行）──
        immune_blocked = self._immune_check(request)
        if immune_blocked is not None:
            return immune_blocked

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

    # ── Schema兼容性检查 ──

    @staticmethod
    def _query_used_fields(query: Query) -> set:
        """提取Query中实际设置了非默认值的字段名集合"""
        used = {"text"}  # text是必填字段
        if query.top_k != 5:
            used.add("top_k")
        if query.token_budget != 1000:
            used.add("token_budget")
        if query.record_types is not None:
            used.add("record_types")
        if query.sources is not None:
            used.add("sources")
        if query.tags is not None:
            used.add("tags")
        if query.min_importance != 0.0:
            used.add("min_importance")
        if query.min_temperature != 0.0:
            used.add("min_temperature")
        if query.use_hybrid:
            used.add("use_hybrid")
        if query.hybrid_weights != (0.6, 0.4):
            used.add("hybrid_weights")
        return used

    def _check_schema_compatibility(self, query: Query) -> None:
        """
        Schema兼容性检查（warning级别，不阻断查询）。

        检查两件事：
        1. Provider的input_schema是否覆盖Query实际使用的字段
        2. Provider的output_schema是否覆盖下游消费者的最小期望字段
        """
        query_fields = self._query_used_fields(query)

        for provider in self._sorted_providers():
            try:
                # 检查1: input_schema — Provider是否支持Query使用的字段
                provider_input = getattr(provider, "input_schema", None)
                if isinstance(provider_input, set) and provider_input:
                    missing_input = query_fields - provider_input
                    if missing_input:
                        logger.warning(
                            f"Schema不兼容[input]: Provider '{provider.name}' "
                            f"input_schema缺少Query字段 {missing_input}。"
                            f"查询仍会执行，但缺失字段可能被忽略。"
                        )

                # 检查2: output_schema — Provider是否产出消费者期望的字段
                provider_output = getattr(provider, "output_schema", None)
                if isinstance(provider_output, set) and provider_output:
                    missing_output = DEFAULT_OUTPUT_SCHEMA - provider_output
                    if missing_output:
                        logger.warning(
                            f"Schema不兼容[output]: Provider '{provider.name}' "
                            f"output_schema缺少下游期望字段 {missing_output}。"
                            f"查询仍会执行，返回的MemoryRecord可能缺少这些字段。"
                        )
            except Exception as e:
                logger.debug(f"Schema检查异常(非阻断): {provider.name}: {e}")

    # ── 检索路径 ──

    def query(self, query: Query) -> List[MemoryRecord]:
        """
        统一检索——多源查询 + 融合排序。

        流程：
        1. Schema兼容性检查（warning级别）
        2. 按priority顺序查询各provider
        3. 合并所有结果
        4. 按score降序排序
        5. 去重（同一record_id只保留最高分）
        6. 贪心填充token_budget
        7. 返回top_k条
        """
        # ── Schema兼容性检查 ──
        self._check_schema_compatibility(query)

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

        # ── score归一化（T-ISA-5）：按provider分桶，桶内score/max归一化 ──
        # 背景：不同provider的score尺度不可比（jiak 0-1 / capsule 0-2+ /
        # causal温度0-3+ / recall 0.7-1.0），直接sort是"苹果+橘子"排序。
        # 方案：每provider桶内除以该桶最大score → 各桶top≈1.0，公平竞争。
        buckets: Dict[str, List[MemoryRecord]] = {}
        for r in all_records:
            buckets.setdefault(r.provider, []).append(r)

        normalized: List[MemoryRecord] = []
        for prov, records in buckets.items():
            max_score = max((r.score for r in records), default=0.0)
            if max_score <= 0:
                normalized.extend(records)  # 全0桶保持原样
                continue
            for r in records:
                r.score = round(r.score / max_score, 4)
                normalized.append(r)

        # 按score降序排序
        normalized.sort(key=lambda r: r.score, reverse=True)

        # 去重（同一record_id只保留最高分）
        seen_ids = set()
        seen_content_prefix = set()  # 内容前缀去重（T-ISA-5）：防同内容刷屏
        deduped = []
        for r in normalized:
            if r.record_id not in seen_ids:
                # 内容前缀去重：前60字符相同视为重复（跨provider同样适用）
                content_prefix = r.content[:60].strip()
                if content_prefix and content_prefix in seen_content_prefix:
                    continue
                seen_ids.add(r.record_id)
                seen_content_prefix.add(content_prefix)
                deduped.append(r)

        # 贪心填充token_budget
        selected = []
        used_tokens = 0

        for r in deduped:
            if len(selected) >= query.top_k:
                break
            est_tokens = estimate_tokens(r.content)
            if used_tokens + est_tokens > query.token_budget:
                break  # 后面的更大，直接停止
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

    def hybrid_rerank(
        self,
        records: List[MemoryRecord],
        query_text: str,
        bm25_weight: float = 0.6,
        embed_weight: float = 0.4,
    ) -> List[MemoryRecord]:
        """
        混合重排：BM25关键词匹配 + Embedding语义匹配。
        
        Args:
            records: 初始检索结果（不修改原列表）
            query_text: 查询文本
            bm25_weight: BM25权重
            embed_weight: Embedding权重
            
        Returns:
            新的重排后记录列表
        """
        if not records:
            return records
        
        try:
            from ..retrieval.hybrid import hybrid_score
            import copy
            # 创建副本避免修改原记录
            scored = []
            for r in records:
                r_copy = copy.copy(r)
                r_copy.score = hybrid_score(
                    query_text, r_copy.content,
                    bm25_weight=bm25_weight,
                    embed_weight=embed_weight
                )
                scored.append(r_copy)
            scored.sort(key=lambda r: r.score, reverse=True)
            return scored
        except Exception as e:
            # 任何异常都降级为原始排序
            logger.debug(f"hybrid_rerank降级: {e}")
            return records

    def get_write_log(self, last_n: int = 20) -> List[WriteResult]:
        """最近N条写入记录（审计用）"""
        return self._write_log[-last_n:]

    def _log_write(self, result: WriteResult) -> None:
        """记录写入到审计日志"""
        self._write_log.append(result)
        if len(self._write_log) > self._max_log:
            self._write_log = self._write_log[-self._max_log:]
