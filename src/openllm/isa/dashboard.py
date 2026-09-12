"""
ISA 认知仪表盘 — VISTA启示：Agent必须能看到自己记忆的状态

VISTA核心发现：LLM对自身上下文状态是"本体感觉盲"。
诊断实验：剥离dashboard后，误差0.43-0.84。加dashboard后误差归零。

ISA Dashboard暴露：
  - 每个记忆块的元数据（温度、年龄、访问次数、来源信任）
  - 总预算条（用了多少tokens、还剩多少）
  - 因果记忆状态（多少条、成功率、最高温度）
  - 检索质量自评（上次检索相关吗）
  - 记忆免疫状态（多少untrusted、多少被拦截）

设计哲学：
  - 每轮响应前生成dashboard
  - Agent看到dashboard后自主决定：保留/归档/恢复/删除
  - 不是告诉Agent怎么做，是告诉Agent"你现在是什么状态"
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger("openllm.dashboard")


@dataclass
class MemoryBlockStatus:
    """单个记忆块的状态"""
    block_id: str
    temperature: float
    age_seconds: float        # 距创建的时间
    access_count: int
    source_trust: str         # trusted/internal/untrusted/unknown
    layer: str                # hot/warm/cold
    token_estimate: int = 0   # 估算token量
    last_accessed_ago: float = 0.0  # 距上次访问的时间
    status: str = "visible"   # visible/pinned/archived/dead


@dataclass
class DashboardState:
    """完整的仪表盘状态"""
    # 预算
    total_budget: int = 500000     # 总token预算
    used_tokens: int = 0           # 已用tokens
    causal_memory_count: int = 0
    causal_memory_success_rate: float = 0.0
    
    # 记忆分布
    hot_count: int = 0
    warm_count: int = 0
    cold_count: int = 0
    dead_count: int = 0
    
    # 信任分布
    trusted_count: int = 0
    untrusted_count: int = 0
    unknown_count: int = 0
    
    # 最近检索
    last_retrieval_relevance: float = 0.0
    last_retrieval_timestamp: float = 0.0
    
    # 元认知
    retrieval_quality_history: List[float] = field(default_factory=list)
    mis_kill_count: int = 0  # 被衰减到dead但后来被需要的次数
    
    # 时间
    session_start: float = field(default_factory=time.time)
    last_dashboard_time: float = field(default_factory=time.time)


class ISADashboard:
    """
    ISA认知仪表盘。
    
    每轮响应前生成，暴露给Agent。
    Agent据此自主决策记忆管理。
    
    VISTA验证：dashboard matters beyond archive and recovery tools。
    没有dashboard，即使有archive/recovery工具，Agent也做不好context管理。
    """
    
    def __init__(self, total_budget: int = 500000):
        self.state = DashboardState(total_budget=total_budget)
        self._blocks: List[MemoryBlockStatus] = []
    
    def update(
        self,
        memory_entries: List[Any] = None,
        causal_store: Any = None,
        session_id: str = "",
    ):
        """
        刷新仪表盘状态。每轮响应前调用。
        
        Args:
            memory_entries: UnifiedMemory的MemoryEntry列表
            causal_store: CausalMemoryStore实例
        """
        now = time.time()
        self.state.last_dashboard_time = now
        self._blocks = []
        
        if memory_entries:
            hot = warm = cold = dead = 0
            trusted = untrusted = unknown = 0
            total_tokens = 0
            
            for entry in memory_entries:
                temp = entry.temperature() if hasattr(entry, 'temperature') else 0.0
                age = now - entry.created_at if hasattr(entry, 'created_at') else 0
                layer = getattr(entry, 'layer', 'warm')
                access_count = getattr(entry, 'access_count', 0)
                
                # 估算token量
                value_str = json.dumps(getattr(entry, 'value', {}), ensure_ascii=False)
                token_est = len(value_str) // 4  # 粗略估算
                total_tokens += token_est
                
                # 状态判定
                if temp > 7:
                    status = "visible"
                    hot += 1
                elif temp > 3:
                    status = "visible"
                    warm += 1
                elif temp > 1:
                    status = "visible"
                    cold += 1
                else:
                    status = "dead"
                    dead += 1
                
                # 信任分布
                trust = getattr(entry, 'trust_level', 'internal')
                if hasattr(trust, 'value'):
                    trust = trust.value
                if trust == 'trusted':
                    trusted += 1
                elif trust == 'untrusted':
                    untrusted += 1
                else:
                    unknown += 1
                
                block = MemoryBlockStatus(
                    block_id=getattr(entry, 'key', 'unknown'),
                    temperature=temp,
                    age_seconds=age,
                    access_count=access_count,
                    source_trust=trust,
                    layer=layer,
                    token_estimate=token_est,
                    status=status,
                )
                self._blocks.append(block)
            
            self.state.used_tokens = total_tokens
            self.state.hot_count = hot
            self.state.warm_count = warm
            self.state.cold_count = cold
            self.state.dead_count = dead
            self.state.trusted_count = trusted
            self.state.untrusted_count = untrusted
            self.state.unknown_count = unknown
        
        # 因果记忆统计
        if causal_store:
            stats = causal_store.stats()
            self.state.causal_memory_count = stats.get("total", 0)
            success = stats.get("by_success", {})
            total = success.get("success", 0) + success.get("failure", 0)
            self.state.causal_memory_success_rate = (
                success.get("success", 0) / total if total > 0 else 0
            )
    
    def record_retrieval_quality(self, relevance: float):
        """记录检索质量——元认知反馈"""
        self.state.last_retrieval_relevance = relevance
        self.state.last_retrieval_timestamp = time.time()
        self.state.retrieval_quality_history.append(relevance)
        # 只保留最近20次
        if len(self.state.retrieval_quality_history) > 20:
            self.state.retrieval_quality_history = self.state.retrieval_quality_history[-20:]
    
    def record_mis_kill(self):
        """记录误杀——被衰减到dead但后来被需要"""
        self.state.mis_kill_count += 1
    
    def generate(self) -> str:
        """
        生成dashboard文本——注入给Agent。
        
        格式借鉴VISTA的context workspace status。
        """
        now = time.time()
        session_duration = now - self.state.session_start
        budget_used_pct = (self.state.used_tokens / self.state.total_budget * 100) if self.state.total_budget > 0 else 0
        
        lines = []
        lines.append("## ISA 认知仪表盘")
        lines.append(f"### 预算")
        bar_len = 20
        filled = int(budget_used_pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        lines.append(f"[{bar}] {budget_used_pct:.1f}% ({self.state.used_tokens:,}/{self.state.total_budget:,} tokens)")
        lines.append("")
        
        # 记忆分布
        lines.append("### 记忆分布")
        total = self.state.hot_count + self.state.warm_count + self.state.cold_count + self.state.dead_count
        if total > 0:
            lines.append(f"  🔥 热: {self.state.hot_count} ({self.state.hot_count/total*100:.0f}%)")
            lines.append(f"  ♨️  温: {self.state.warm_count} ({self.state.warm_count/total*100:.0f}%)")
            lines.append(f"  ❄️  冷: {self.state.cold_count} ({self.state.cold_count/total*100:.0f}%)")
            lines.append(f"  💀 死: {self.state.dead_count} ({self.state.dead_count/total*100:.0f}%)")
        else:
            lines.append("  (空)")
        lines.append("")
        
        # 信任分布
        lines.append("### 信任分布")
        lines.append(f"  ✅ 可信: {self.state.trusted_count} | ⚠️  未验证: {self.state.unknown_count} | 🚫 不可信: {self.state.untrusted_count}")
        lines.append("")
        
        # 因果记忆
        lines.append("### 因果记忆")
        lines.append(f"  条数: {self.state.causal_memory_count} | 成功率: {self.state.causal_memory_success_rate*100:.0f}%")
        lines.append("")
        
        # 元认知
        lines.append("### 元认知")
        if self.state.retrieval_quality_history:
            avg_relevance = sum(self.state.retrieval_quality_history) / len(self.state.retrieval_quality_history)
            lines.append(f"  最近检索相关性: {self.state.last_retrieval_relevance:.2f} (平均: {avg_relevance:.2f})")
        else:
            lines.append(f"  最近检索相关性: 无记录")
        if self.state.mis_kill_count > 0:
            lines.append(f"  ⚠️ 误杀记录: {self.state.mis_kill_count}次")
        lines.append("")
        
        # 高温度块top5
        hot_blocks = sorted(self._blocks, key=lambda b: b.temperature, reverse=True)[:5]
        if hot_blocks:
            lines.append("### 高温度记忆块:")
            for b in hot_blocks:
                emoji = "✅" if b.source_trust == "trusted" else "⚠️" if b.source_trust == "unknown" else "🚫" if b.source_trust == "untrusted" else "ℹ️"
                lines.append(f"  {emoji} [{b.block_id[:20]}] T={b.temperature:.2f} | age={b.age_seconds/3600:.1f}h | {b.token_estimate}tok")
        
        # Session信息
        lines.append(f"\n### Session: {session_duration/60:.0f}min")
        
        return "\n".join(lines)
    
    def get_blocks_for_archival(self, budget_pressure: float = 0.8) -> List[str]:
        """
        推荐归档的记忆块ID。
        
        当budget使用率 > budget_pressure时，推荐归档低温度块。
        返回block_id列表，Agent决定是否真的归档。
        """
        budget_used_pct = self.state.used_tokens / self.state.total_budget if self.state.total_budget > 0 else 0
        
        if budget_used_pct < budget_pressure:
            return []
        
        # 按温度排序，低温优先归档
        candidates = sorted(self._blocks, key=lambda b: b.temperature)
        candidates = [b for b in candidates if b.status != "pinned"]
        
        # 推荐归档直到预算降到70%
        target_tokens = int(self.state.total_budget * 0.7)
        need_to_free = self.state.used_tokens - target_tokens
        
        freed = 0
        to_archive = []
        for b in candidates:
            if freed >= need_to_free:
                break
            to_archive.append(b.block_id)
            freed += b.token_estimate
        
        return to_archive
