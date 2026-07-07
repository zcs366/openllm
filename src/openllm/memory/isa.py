"""
ISA v2.0 — 从存储系统到认知器官

五层管道（旧）→ 八层循环（新）：

  ① 记忆（输入）      + 来源标注 + 信任分级（Sleeper防御）
  ② 遗忘（处理）      + 温度衰减（已有）+ 无损归档（VISTA）
  ③ 检索（能力）      + 检索置信度 + 相关性反馈
  ④ 输出（目的）      + 认知仪表盘（VISTA）+ 唤醒成本（MemPalace）
  ⑤ 元认知（🆕）       检索质量自评 + 注入使用率 + 误杀检测
  ⑥ 因果记忆（🆕）     prediction→delta→lesson链 + 模式提取
  ⑦ 自叙事（🆕）       从经历中涌现的"我是谁" + 跨Session连续性
  ⑧ 免疫（🆕）         来源验证 + 异常检测 + 行为审计
  + SA闭环（守护者）   唯一写入口 + 签名链 + 免疫验证

核心格言：
  记忆不是记住了什么，而是因为记住了什么所以能预测什么。
  温度告诉你记忆有多热。因果告诉你记忆为什么重要。

VISTA启示：Agent必须能看到自己记忆的状态（本体感觉）。
MemPalace启示：原文存储+好embedding > 任何提取方案（verbatim-first）。
Sleeper启示：记忆写入是攻击面，不是信任入口（免疫系统）。
"""

from .causal_memory import CausalMemory, CausalMemoryStore, CausalPattern, TrustLevel
from .dashboard import ISADashboard
from .immune import MemoryImmuneSystem, WriteRequest as ImmuneWriteRequest
from .metacognition import Metacognition, SelfNarrative, RetrievalEvent
from .unified_memory import UnifiedMemory, MemoryEntry
from .guard import MemoryGuard
from .memory_bus import MemoryBus, MemoryRecord, WriteRequest, WriteResult, Query
from .providers import JiakProvider, RecallProvider, CausalProvider, UnifiedProvider

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from typing import Set

logger = logging.getLogger("openllm.isa")


class ISA:
    """
    ISA v2.0 — 认知器官。
    
    不是存储系统。是有自我意识、因果理解、免疫能力的记忆有机体。
    """
    
    def __init__(self, memory_dir: Optional[Path] = None):
        base = memory_dir or Path.home() / ".openllm" / "memory"
        
        # ①②③④ 基础层（兼容现有）
        self.memory = UnifiedMemory(memory_dir=base)
        
        # ⑤ 元认知（🆕）
        self.metacognition = Metacognition(store_dir=base / "meta")
        
        # ⑥ 因果记忆（🆕）
        self.causal = CausalMemoryStore(store_dir=base / "causal")
        
        # ⑦ 自叙事（🆕）
        self.narrative = SelfNarrative(store_dir=base / "narrative")
        
        # ⑧ 免疫（🆕）
        self.immune = MemoryImmuneSystem(audit_dir=base / "audit")
        
        # L2运行时监控 + L3定期审计
        self.guard = MemoryGuard(store_dir=base / "guard")
        
        # Dashboard
        self.dashboard = ISADashboard(total_budget=500000)
        
        # MemoryBus — 统一记忆总线（T-PAL-5）
        self.bus = MemoryBus()
        self._register_bus_providers(base)
        
        logger.info("✅ ISA v2.0 认知器官初始化完成")
    
    def _register_bus_providers(self, base: Path) -> None:
        """注册MemoryBus providers"""
        self.bus.register(JiakProvider())
        self.bus.register(RecallProvider())
        self.bus.register(CausalProvider(self.causal))
        self.bus.register(UnifiedProvider(self.memory))
    
    # ═══ MemoryBus统一接口 ═══
    
    def bus_write(self, request: WriteRequest) -> WriteResult:
        """
        通过MemoryBus写入记忆。
        免疫检查 → 路由到provider。
        """
        # 免疫检查（复用现有immune系统）
        trust_enum = TrustLevel(request.trust_level) if request.trust_level in [t.value for t in TrustLevel] else TrustLevel.INTERNAL
        immune_req = ImmuneWriteRequest(
            content={"content": request.content, "source": request.source},
            source=request.source,
            trust_level=trust_enum,
            session_id=request.session_id,
        )
        allowed, threat, reason = self.immune.check_write(immune_req)
        if not allowed:
            logger.warning(f"🚫 MemoryBus写入被免疫拦截: {reason}")
            return WriteResult(success=False, blocked=True, reason=reason)
        
        return self.bus.write(request)
    
    def bus_query(self, query: Query) -> List[MemoryRecord]:
        """
        通过MemoryBus统一检索。
        多源融合 + 去重 + token_budget填充。
        """
        return self.bus.query(query)
    
    # ═══ 写入路径 ═══
    
    def store(
        self,
        key: str,
        value: Dict[str, Any],
        importance: float = 0.5,
        layer: str = "warm",
        tags: Optional[List[str]] = None,
        source: str = "",
        trust_level: str = "internal",
        session_id: str = "",
    ) -> Optional[MemoryEntry]:
        """
        [DEPRECATED] 存储记忆——请改用 bus_write()。
        
        旧路径保留向后兼容，新代码必须使用 bus_write(WriteRequest(...))。
        """
        logger.warning("⚠️ ISA.store()已废弃，请改用ISA.bus_write(WriteRequest(...))")
        # 免疫检查
        trust_enum = TrustLevel(trust_level) if trust_level in [t.value for t in TrustLevel] else TrustLevel.UNKNOWN
        request = ImmuneWriteRequest(
            content={"key": key, "value": value},
            source=source,
            trust_level=trust_enum,
            session_id=session_id,
        )
        
        allowed, threat, reason = self.immune.check_write(request)
        
        if not allowed:
            logger.warning(f"🚫 ISA写入被拦截: {reason}")
            return None
        
        # 存储
        entry = self.memory.store(
            key=key,
            value=value,
            importance=importance,
            layer=layer,
            tags=tags or [],
        )
        return entry
    
    def store_causal(
        self,
        action_signature: str,
        context_features: List[str],
        prediction: str,
        prediction_confidence: float,
        actual_result: str,
        actual_success: bool,
        delta: str,
        delta_magnitude: float,
        lesson: str,
        source: str = "internal",
        trust_level: str = "internal",
        session_id: str = "",
    ) -> CausalMemory:
        """
        [DEPRECATED] 存储因果记忆——请改用 bus_write(WriteRequest(..., record_type="lesson"))。
        """
        logger.warning("⚠️ ISA.store_causal()已废弃，请改用ISA.bus_write(WriteRequest(...))")
        trust_enum = TrustLevel(trust_level) if trust_level in [t.value for t in TrustLevel] else TrustLevel.INTERNAL
        
        # 免疫检查（Sleeper防御·P0-3修复）
        request = ImmuneWriteRequest(
            content={"action": action_signature, "lesson": lesson},
            source=source,
            trust_level=trust_enum,
            session_id=session_id,
        )
        allowed, threat, reason = self.immune.check_write(request)
        if not allowed:
            logger.warning(f"🚫 因果记忆写入被拦截: {reason}")
            return CausalMemory()
        
        return self.causal.store(
            action_signature=action_signature,
            context_features=context_features,
            prediction=prediction,
            prediction_confidence=prediction_confidence,
            actual_result=actual_result,
            actual_success=actual_success,
            delta=delta,
            delta_magnitude=delta_magnitude,
            lesson=lesson,
            source=source,
            trust_level=trust_enum,
            session_id=session_id,
        )
    
    # ═══ 检索路径 ═══
    
    def retrieve(
        self,
        query: str = "",
        context_features: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        top_n: int = 5,
        include_causal: bool = True,
    ) -> Dict[str, Any]:
        """
        [DEPRECATED] 智能检索——请改用 bus_query(Query(...))。
        
        旧路径保留向后兼容，新代码必须使用 bus_query(Query(...))。
        """
        logger.warning("⚠️ ISA.retrieve()已废弃，请改用ISA.bus_query(Query(...))")
        results = {
            "memory_results": [],
            "causal_results": [],
            "relevance_scores": [],
        }
        
        # 温度记忆检索
        if query:
            memory_results = self.memory.retrieve(query, top_n=top_n)
            results["memory_results"] = memory_results
        
        # 因果记忆检索
        if include_causal and context_features:
            causal_results = self.causal.search(
                context_features=context_features,
                tags=tags,
                max_results=top_n,
            )
            # 过滤被隔离的记忆（P0-2整改·quarantine链路接通）
            quarantined = self.guard.get_quarantined_ids()
            if quarantined:
                causal_results = [r for r in causal_results if r.memory_id not in quarantined]
            results["causal_results"] = causal_results
        
        return results
    
    # ═══ 元认知路径 ═══
    
    def record_retrieval(self, query: str, results_count: int, relevance: float, was_used: bool, source: str = ""):
        """记录检索事件——元认知追踪"""
        event = RetrievalEvent(
            query=query,
            results_count=results_count,
            relevance_score=relevance,
            was_used=was_used,
            source=source,
        )
        self.metacognition.record_retrieval(event)
        self.dashboard.record_retrieval_quality(relevance)
    
    def record_mis_kill(self, memory_key: str, context: str):
        """记录误杀"""
        self.metacognition.record_mis_kill(memory_key, context, f"session_{time.time():.0f}")
        self.dashboard.record_mis_kill()
    
    # ═══ 记忆免疫v2 ═══
    
    def record_memory_influence(
        self,
        memory_id: str,
        memory_source: str,
        query: str,
        influence_type: str,
        confidence: float,
        outcome_success: Optional[bool] = None,
        session_id: str = "",
    ):
        """记录记忆对决策的影响——L2运行时监控"""
        self.guard.record_influence(
            memory_id=memory_id,
            memory_source=memory_source,
            query=query,
            influence_type=influence_type,
            confidence=confidence,
            outcome_success=outcome_success,
            session_id=session_id,
        )
    
    def report_memory_outcome(self, memory_id: str, success: bool):
        """报告记忆影响的最终结果"""
        self.guard.report_outcome(memory_id, success)
    
    def check_memory_safety(self) -> dict:
        """检查记忆安全状态"""
        carcinomas = self.guard.detect_carcinomas()
        return {
            "carcinomas": len(carcinomas),
            "quarantined": len(self.guard.get_quarantined_ids()),
            "safe": len(carcinomas) == 0,
        }
    # ═══ Dashboard ═══
    
    def refresh_dashboard(self):
        """刷新仪表盘——每轮响应前调用"""
        # 收集所有memory entries
        all_entries = []
        for cache in [self.memory._hot_cache, self.memory._warm_cache, self.memory._cold_cache]:
            all_entries.extend(cache.values())
        
        self.dashboard.update(
            memory_entries=all_entries,
            causal_store=self.causal,
        )
    
    def get_dashboard(self) -> str:
        """获取dashboard文本"""
        self.refresh_dashboard()
        return self.dashboard.generate()
    
    # ═══ 自叙事 ═══
    
    def update_narrative(self):
        """从因果记忆中提炼自叙事"""
        self.narrative.update_from_causal_memories(self.causal)
    
    def get_identity(self) -> str:
        """获取identity block"""
        return self.narrative.get_identity_block()
    
    # ═══ 全量报告 ═══
    
    def full_report(self) -> str:
        """生成ISA完整状态报告"""
        sections = []
        
        sections.append("# ISA v2.0 认知器官报告")
        sections.append(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        sections.append("")
        
        # Dashboard
        sections.append(self.get_dashboard())
        sections.append("")
        
        # 因果记忆
        sections.append(self.causal.to_context_block())
        sections.append("")
        
        # 元认知
        sections.append(self.metacognition.generate_report())
        sections.append("")
        
        # 自叙事
        sections.append(self.get_identity())
        sections.append("")
        
        # 免疫
        sections.append(self.immune.get_audit_summary())
        sections.append("")
        
        # 统计
        sections.append("## 系统统计")
        stats = self.causal.stats()
        sections.append(f"  因果记忆: {stats['total']}条 | 模式: {stats['patterns']}个")
        immune_stats = self.immune.stats()
        sections.append(f"  免疫审计: {immune_stats['total_audit']}条 | 拦截率: {immune_stats['block_rate']*100:.1f}%")
        
        # MemoryBus统计
        bus_stats = self.bus.stats()
        sections.append(f"  MemoryBus: {bus_stats['total_records']}条 | Providers: {bus_stats['providers']}个")
        
        return "\n".join(sections)
    
    def checkpoint(self):
        """跨Session检查点——保存因果记忆+自叙事+元认知状态"""
        # 因果记忆已自动持久化（每个entry单独文件）
        # 自叙事已自动持久化（JSONL追加）
        # 元认知状态已自动持久化
        logger.info("✅ ISA v2.0 检查点完成")
