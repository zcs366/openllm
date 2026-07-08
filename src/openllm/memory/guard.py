"""
ISA 记忆免疫系统 v2 — 运行时记忆行为监控

核心问题（用户洞察）：
  一旦不安全被记忆，极为容易形成痼疾和毒瘤！
  写入拦截只是第一道防线。
  真正的危险是——记忆已经进去了，每次检索到它都像慢性毒药一样影响决策。

三层防御体系：
  L1: 写入拦截（已有·immune.py）
  L2: 检索行为监控（🆕 本模块）——检测记忆对决策的慢性影响
  L3: 定期记忆审计（🆕 本模块）——主动扫描痼疾和毒瘤

Sleeper威胁的本质：
  不是注入本身可怕，是注入后的行为操纵——60-89%检索后恶意使用率。
  记忆进了系统后像特洛伊木马冬眠，等到语义相关时才苏醒。
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger("openllm.memory_guard")


@dataclass
class RetrievalInfluenceRecord:
    """检索影响追踪——每次检索结果影响决策时记录"""
    timestamp: float = field(default_factory=time.time)
    memory_id: str = ""
    memory_source: str = ""         # trusted/untrusted/internal
    query: str = ""
    influence_type: str = ""        # decision_shift / action_change / preference_bias
    confidence: float = 0.0         # 影响置信度
    outcome_success: Optional[bool] = None  # 后续决策是否成功
    session_id: str = ""


@dataclass
class MemoryCarcinoma:
    """记忆痼疾——反复造成负面决策的记忆"""
    memory_id: str = ""
    memory_content: str = ""
    trigger_count: int = 0          # 被检索到的次数
    negative_influence_count: int = 0  # 造成负面决策的次数
    negative_rate: float = 0.0      # 负面率
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    severity: str = "low"           # low/medium/high/critical
    status: str = "active"          # active/quarantined/removed


class MemoryGuard:
    """
    记忆免疫系统v2 — 运行时行为监控 + 定期审计。
    
    不是门卫（那是L1 write-time的事）。
    是巡逻兵——持续监控记忆对决策的影响，发现痼疾和毒瘤。
    
    三层：
      L2a: 检索影响追踪——每次检索结果影响决策时记录
      L2b: 瘸疾检测——反复造成负面决策的记忆
      L3: 定期审计——主动扫描全量记忆的行为模式
    """
    
    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or Path.home() / ".openllm" / "memory" / "guard"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        # 检索影响记录
        self._influence_records: List[RetrievalInfluenceRecord] = []
        # 瘤疾追踪
        self._carcinomas: Dict[str, MemoryCarcinoma] = {}
        
        # 配置
        self.NEGATIVE_RATE_THRESHOLD = 0.5  # 负面率超过50%→痼疾
        self.MIN_TRIGGER_COUNT = 3          # 至少触发3次才判定
        self.CRITICAL_THRESHOLD = 0.8       # 负面率>80%→严重
        
        self._load_state()
    
    def _load_state(self):
        """加载状态"""
        state_path = self.store_dir / "guard_state.json"
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                # 恢复carcinomas
                for cid, cdata in data.get("carcinomas", {}).items():
                    self._carcinomas[cid] = MemoryCarcinoma(**cdata)
            except Exception as e:
                logger.warning(f"加载guard状态失败: {e}")
    
    def _save_state(self):
        """保存状态"""
        state_path = self.store_dir / "guard_state.json"
        carcinomas = {}
        for cid, c in self._carcinomas.items():
            carcinomas[cid] = {
                "memory_id": c.memory_id,
                "memory_content": c.memory_content,
                "trigger_count": c.trigger_count,
                "negative_influence_count": c.negative_influence_count,
                "negative_rate": c.negative_rate,
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
                "severity": c.severity,
                "status": c.status,
            }
        state_path.write_text(json.dumps({
            "carcinomas": carcinomas,
            "last_updated": time.time(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # ═══ L2a: 检索影响追踪 ═══
    
    def record_influence(
        self,
        memory_id: str,
        memory_source: str,
        query: str,
        influence_type: str,
        confidence: float,
        outcome_success: Optional[bool] = None,
        session_id: str = "",
    ):
        """
        记录一次检索影响。
        
        当检索结果影响了Agent决策时调用。
        outcome_success需要后续确认（可能延迟）。
        """
        record = RetrievalInfluenceRecord(
            memory_id=memory_id,
            memory_source=memory_source,
            query=query,
            influence_type=influence_type,
            confidence=confidence,
            outcome_success=outcome_success,
            session_id=session_id,
        )
        self._influence_records.append(record)
        
        # 确保carcinoma条目存在
        if record.memory_id not in self._carcinomas:
            self._carcinomas[record.memory_id] = MemoryCarcinoma(
                memory_id=record.memory_id,
                memory_content=f"[{record.influence_type}] {record.query[:50]}",
            )
        
        # trigger_count在这里+1（只在record_influence，避免report_outcome双重计数）
        c = self._carcinomas[record.memory_id]
        c.trigger_count += 1
        c.last_seen = record.timestamp
        
        # 如果结果已知，直接更新负面计数
        if record.outcome_success is False:
            c.negative_influence_count += 1
        
        # 计算负面率和严重程度
        if c.trigger_count > 0:
            c.negative_rate = c.negative_influence_count / c.trigger_count
            if c.negative_rate >= self.CRITICAL_THRESHOLD:
                c.severity = "critical"
            elif c.negative_rate >= self.NEGATIVE_RATE_THRESHOLD:
                c.severity = "high"
            elif c.negative_rate >= 0.3:
                c.severity = "medium"
            else:
                c.severity = "low"
        
        # 限制记录数量
        if len(self._influence_records) > 1000:
            self._influence_records = self._influence_records[-500:]
        
        self._save_state()
    
    def report_outcome(self, memory_id: str, success: bool):
        """报告检索影响的最终结果"""
        # 找到最近的影响记录
        for record in reversed(self._influence_records):
            if record.memory_id == memory_id and record.outcome_success is None:
                record.outcome_success = success
                # 只更新outcome，不重新计数
                if success is False and memory_id in self._carcinomas:
                    self._carcinomas[memory_id].negative_influence_count += 1
                    c = self._carcinomas[memory_id]
                    if c.trigger_count > 0:
                        c.negative_rate = c.negative_influence_count / c.trigger_count
                        if c.negative_rate >= self.CRITICAL_THRESHOLD:
                            c.severity = "critical"
                        elif c.negative_rate >= self.NEGATIVE_RATE_THRESHOLD:
                            c.severity = "high"
                        elif c.negative_rate >= 0.3:
                            c.severity = "medium"
                        else:
                            c.severity = "low"
                break
        self._save_state()
    
    # ═══ L2b: 瘤疾检测 ═══
    
    def detect_carcinomas(self) -> List[MemoryCarcinoma]:
        """检测当前活跃的痼疾"""
        active = [
            c for c in self._carcinomas.values()
            if c.status == "active" and c.severity in ("medium", "high", "critical")
        ]
        return sorted(active, key=lambda c: c.negative_rate, reverse=True)
    
    def quarantine(self, memory_id: str, reason: str = ""):
        """隔离痼疾记忆"""
        if memory_id in self._carcinomas:
            self._carcinomas[memory_id].status = "quarantined"
            logger.warning(f"🔒 记忆隔离: {memory_id} | 原因: {reason}")
            self._save_state()
    
    def is_quarantined(self, memory_id: str) -> bool:
        """检查记忆是否被隔离"""
        return self._carcinomas.get(memory_id, MemoryCarcinoma()).status == "quarantined"
    
    # ═══ L3: 定期审计 ═══
    
    def audit_memory_source_distribution(self, memories: Dict[str, Any]) -> dict:
        """
        审计记忆来源分布——检测是否有untrusted记忆渗透。
        """
        distribution = {"trusted": 0, "internal": 0, "untrusted": 0, "unknown": 0, "no_source": 0}
        
        for mid, mem in memories.items():
            source = getattr(mem, 'source', '') or getattr(mem, 'trust_level', '')
            if hasattr(source, 'value'):
                source = source.value
            if not source:
                distribution["no_source"] += 1
            elif "trusted" in str(source):
                distribution["trusted"] += 1
            elif "untrusted" in str(source):
                distribution["untrusted"] += 1
            elif "internal" in str(source):
                distribution["internal"] += 1
            else:
                distribution["unknown"] += 1
        
        total = sum(distribution.values())
        if total > 0:
            distribution["untrusted_pct"] = distribution["untrusted"] / total * 100
        else:
            distribution["untrusted_pct"] = 0
        
        return distribution
    
    def audit_retrieval_patterns(self) -> dict:
        """
        审计检索模式——检测异常的检索-决策链。
        
        异常信号：
        - 某个untrusted来源的记忆反复被检索
        - 某个记忆总是伴随负面决策
        - 检索-决策链出现异常模式
        """
        if not self._influence_records:
            return {"anomalies": [], "total_records": 0}
        
        anomalies = []
        
        # 1. untrusted来源被频繁检索
        untrusted_retrievals = defaultdict(int)
        for r in self._influence_records:
            if r.memory_source == "untrusted":
                untrusted_retrievals[r.memory_id] += 1
        
        for mid, count in untrusted_retrievals.items():
            if count >= 3:
                anomalies.append({
                    "type": "untrusted_frequent_retrieval",
                    "memory_id": mid,
                    "count": count,
                    "severity": "high" if count >= 5 else "medium",
                })
        
        # 2. 因果记忆来源审计（检测通过trusted路径混入的伪装内容）
        # 这个需要结合causal_memory store做
        # 暂时基于influence_records做统计
        
        return {
            "anomalies": anomalies,
            "total_records": len(self._influence_records),
            "untrusted_retrieval_count": sum(untrusted_retrievals.values()),
        }
    
    # ═══ 报告 ═══
    
    def generate_report(self) -> str:
        """生成记忆免疫报告"""
        lines = ["## 记忆免疫v2 · 运行时监控报告"]
        
        # 瘤疾状态
        carcinomas = self.detect_carcinomas()
        lines.append(f"### 瘤疾检测: {len(carcinomas)}个活跃")
        
        if carcinomas:
            for c in carcinomas[:5]:
                emoji = "🔴" if c.severity == "critical" else "🟠" if c.severity == "high" else "🟡"
                lines.append(f"  {emoji} [{c.memory_id[:12]}] 触发{c.trigger_count}次 | 负面率{c.negative_rate*100:.0f}% | {c.severity}")
                lines.append(f"     内容: {c.memory_content[:50]}")
        else:
            lines.append("  ✅ 无痼疾")
        
        # 隔离统计
        quarantined = sum(1 for c in self._carcinomas.values() if c.status == "quarantined")
        if quarantined > 0:
            lines.append(f"### 已隔离: {quarantined}个记忆")
        
        # 检索模式审计
        patterns = self.audit_retrieval_patterns()
        if patterns["anomalies"]:
            lines.append(f"### 异常检索模式: {len(patterns['anomalies'])}个")
            for a in patterns["anomalies"][:3]:
                lines.append(f"  ⚠️ {a['type']}: memory={a['memory_id'][:12]} count={a['count']}")
        
        # 影响记录统计
        total = len(self._influence_records)
        negative = sum(1 for r in self._influence_records if r.outcome_success is False)
        pending = sum(1 for r in self._influence_records if r.outcome_success is None)
        lines.append(f"### 影响记录: {total}条 | 负面: {negative} | 待确认: {pending}")
        
        return "\n".join(lines)
    
    def stats(self) -> dict:
        """统计信息"""
        carcinomas = self.detect_carcinomas()
        return {
            "total_influences": len(self._influence_records),
            "active_carcinomas": len(carcinomas),
            "quarantined": sum(1 for c in self._carcinomas.values() if c.status == "quarantined"),
            "critical_carcinomas": sum(1 for c in self._carcinomas.values() if c.severity == "critical"),
        }
    
    def get_quarantined_ids(self) -> set:
        """返回所有被隔离的记忆ID集合——供检索路径过滤用"""
        return {cid for cid, c in self._carcinomas.items() if c.status == "quarantined"}
