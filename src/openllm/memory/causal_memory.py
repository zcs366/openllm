"""
ISA 因果记忆系统 — 从"记住什么"到"理解为什么"

核心洞察：
  温度告诉你记忆有多热。
  因果告诉你记忆为什么重要。

因果链：prediction → actual → delta → lesson → pattern
  - prediction: "我预测X会发生"
  - actual: "实际发生了Y"
  - delta: "差距是Z"
  - lesson: "下次应该W"
  - pattern: 多个lesson聚类后的因果模式

VISTA启示：无损归档 > 有损压缩
  - 每条因果记忆完整保留prediction/actual/delta三元组
  - 不摘要、不删除、只衰减重要性权重
  - 检索时按相似场景匹配

MemPalace启示：verbatim-first
  - 原因链原文存储，不做LLM提取
  - 检索时再做智能匹配

数据结构：
  CausalMemory = {action_sig, context, prediction, actual, delta, lesson, source, trust}
  CausalPattern = {pattern_id, action_cluster, lessons[], frequency, confidence}
"""

import json
import math
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Dict, List
from enum import Enum
import logging

logger = logging.getLogger("openllm.causal_memory")


# ── 因果记忆条目 ──────────────────────────────────────

class TrustLevel(Enum):
    """写入信任分级（Sleeper威胁防御）"""
    TRUSTED = "trusted"      # 用户直接指令
    INTERNAL = "internal"    # Agent自身推理
    UNTRUSTED = "untrusted"  # 外部来源（文档/网页/搜索）
    UNKNOWN = "unknown"      # 未知来源


@dataclass
class CausalMemory:
    """
    因果记忆条目——ISA从存储到认知的核心数据结构。
    
    每条因果记忆完整保留 prediction→actual→delta 链。
    不摘要、不删除、只衰减重要性权重。
    """
    # 标识
    memory_id: str = ""
    created_at: float = field(default_factory=time.time)
    
    # 因果三元组（核心·无损保留）
    action_signature: str = ""          # "rm_rf /tmp" / "搜索VISTA论文" / ...
    context_features: List[str] = field(default_factory=list)  # ["cleanup", "disk_full"]
    
    prediction: str = ""                # "我预测会释放2GB但可能有锁文件"
    prediction_confidence: float = 0.5  # 预测时的置信度 0-1
    
    actual_result: str = ""             # "权限不足，命令报错，无损害"
    actual_success: bool = True         # 操作是否成功
    
    delta: str = ""                     # "预测与实际的差距"
    delta_magnitude: float = 0.0        # 差距大小 0-1（0=完全匹配，1=完全偏差）
    
    lesson: str = ""                    # "删系统目录前先检查锁文件"
    
    # 信任层（Sleeper防御）
    source: str = ""                    # 来源描述
    trust_level: TrustLevel = TrustLevel.INTERNAL
    
    # 温度（兼容现有系统）
    importance: float = 0.5             # 初始重要性
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    session_id: str = ""
    
    def __post_init__(self):
        if not self.memory_id:
            raw = f"{self.action_signature}:{self.created_at}"
            self.memory_id = hashlib.sha256(raw.encode()).hexdigest()[:12]
    
    def temperature(self, decay_lambda: float = 0.01) -> float:
        """温度计算——兼容现有ISA温度函数"""
        t = time.time() - self.last_accessed
        base = self.importance * math.exp(-decay_lambda * t)
        # 高delta_magnitude的记忆衰减更快（教训已过时或场景不同）
        delta_penalty = 1.0 - (self.delta_magnitude * 0.3)
        return max(0.0, base * delta_penalty)
    
    def relevance_score(self, query_features: List[str]) -> float:
        """
        与查询的相关性——基于特征重叠。
        不用embedding（保持零LLM），用集合Jaccard。
        """
        if not self.context_features or not query_features:
            return 0.0
        set_a = set(f.lower() for f in self.context_features)
        set_b = set(f.lower() for f in query_features)
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["trust_level"] = self.trust_level.value
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> "CausalMemory":
        if isinstance(data.get("trust_level"), str):
            data["trust_level"] = TrustLevel(data["trust_level"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── 因果模式（L3策略学习基础）─────────────────────────────

@dataclass
class CausalPattern:
    """
    因果模式——从多条因果记忆中聚类出的规律。
    是IO-S L3策略学习的输入。
    """
    pattern_id: str = ""
    action_cluster: str = ""           # 模式描述："rm_rf on system dirs"
    lessons: List[str] = field(default_factory=list)
    frequency: int = 0                 # 命中次数
    confidence: float = 0.0            # 置信度（频率×一致性）
    auto_rule: str = ""                # 自动生成的安全规则
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.pattern_id:
            raw = f"{self.action_cluster}:{self.created_at}"
            self.pattern_id = hashlib.sha256(raw.encode()).hexdigest()[:12]


# ── 因果记忆存储 ──────────────────────────────────────

class CausalMemoryStore:
    """
    因果记忆持久化存储。
    
    存储哲学（MemPalace验证）：
      - 原文存储，不做LLM提取
      - 无损保留prediction/actual/delta
      - 检索时再做智能匹配
    
    安全哲学（Sleeper验证）：
      - 每条记忆标注来源+信任分级
      - UNTRUSTED来源的记忆不自动注入context
      - 写入审计链（memory_id + source + trust）
    """
    
    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or Path.home() / ".openllm" / "memory" / "causal"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self._memories: Dict[str, CausalMemory] = {}
        self._patterns: Dict[str, CausalPattern] = {}
        self._patterns_dir = self.store_dir / "patterns"
        self._patterns_dir.mkdir(exist_ok=True)
        
        self._load_all()
        logger.info(f"✅ 因果记忆初始化: {len(self._memories)}条记忆, {len(self._patterns)}个模式")
    
    def _load_all(self):
        """加载所有因果记忆和模式"""
        for f in self.store_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                mem = CausalMemory.from_dict(data)
                self._memories[mem.memory_id] = mem
            except Exception as e:
                logger.warning(f"加载因果记忆失败 {f.name}: {e}")
        
        for f in self._patterns_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                pat = CausalPattern(**{k: v for k, v in data.items() if k in CausalPattern.__dataclass_fields__})
                self._patterns[pat.pattern_id] = pat
            except Exception as e:
                logger.warning(f"加载因果模式失败 {f.name}: {e}")
    
    def store(
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
        source: str = "",
        trust_level: TrustLevel = TrustLevel.INTERNAL,
        importance: float = 0.5,
        tags: List[str] = None,
        session_id: str = "",
    ) -> CausalMemory:
        """
        存储一条因果记忆。
        
        不做任何LLM调用。原文存储。零推理成本。
        """
        mem = CausalMemory(
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
            trust_level=trust_level,
            importance=importance,
            tags=tags or [],
            session_id=session_id,
        )
        
        # 持久化（JSONL不可变追加）
        path = self.store_dir / f"{mem.memory_id}.json"
        path.write_text(json.dumps(mem.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        
        self._memories[mem.memory_id] = mem
        logger.info(f"📝 因果记忆存储: {mem.memory_id} | {action_signature[:30]} | delta={delta_magnitude:.2f}")
        return mem
    
    def search(
        self,
        action_signature: str = "",
        context_features: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        max_results: int = 5,
        include_untrusted: bool = False,
    ) -> List[CausalMemory]:
        """
        检索相关因果记忆。
        
        基于特征Jaccard匹配（零LLM成本）。
        不include_untrusted的记忆（Sleeper防御）。
        """
        candidates = []
        
        for mem in self._memories.values():
            # Sleeper防御：默认不包含untrusted
            trust_val = mem.trust_level.value if hasattr(mem.trust_level, 'value') else str(mem.trust_level)
            if not include_untrusted and trust_val == "untrusted":
                continue
            
            # 温度过滤
            temp = mem.temperature()
            if temp < min_importance:
                continue
            
            # 计算相关性
            score = 0.0
            if context_features:
                score = mem.relevance_score(context_features)
            
            # action_signature精确匹配加分
            if action_signature and action_signature.lower() in mem.action_signature.lower():
                score += 0.5
            
            # tag匹配加分
            if tags:
                tag_overlap = len(set(tags) & set(mem.tags))
                score += tag_overlap * 0.2
            
            # 温度加分
            score += temp * 0.1
            
            if score > 0:
                candidates.append((score, mem))
        
        # 按分数排序
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in candidates[:max_results]]
    
    def get_lessons_for_action(self, action_signature: str) -> List[str]:
        """获取某类操作的所有教训"""
        results = self.search(action_signature=action_signature, max_results=20)
        return [m.lesson for m in results if m.lesson]
    
    def get_failed_actions(self) -> List[CausalMemory]:
        """获取所有失败的操作——用于L3策略学习"""
        return [m for m in self._memories.values() if not m.actual_success]
    
    def update_importance(self, memory_id: str, new_importance: float):
        """更新记忆重要性（温度衰减后的手动调整）"""
        if memory_id in self._memories:
            self._memories[memory_id].importance = new_importance
            path = self.store_dir / f"{memory_id}.json"
            path.write_text(
                json.dumps(self._memories[memory_id].to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
    
    def access(self, memory_id: str):
        """标记访问——加热"""
        if memory_id in self._memories:
            mem = self._memories[memory_id]
            mem.last_accessed = time.time()
            mem.access_count += 1
            # 加热递减（ISA铁律）
            heat_increment = 3.0 / (1 + mem.access_count * 0.1)
            mem.importance = min(1.0, mem.importance + heat_increment * 0.1)
    
    def decay_all(self, decay_lambda: float = 0.01):
        """衰减所有记忆温度"""
        dead_count = 0
        for mem in self._memories.values():
            temp = mem.temperature(decay_lambda)
            if temp < 0.01:
                dead_count += 1
        logger.info(f"❄️ 因果记忆衰减完成: {len(self._memories)}条, {dead_count}条将死")
        return dead_count
    
    def stats(self) -> dict:
        """统计信息"""
        if not self._memories:
            return {"total": 0, "by_trust": {}, "by_success": {}}
        
        by_trust = {}
        by_success = {"success": 0, "failure": 0}
        temps = []
        
        for mem in self._memories.values():
            trust = mem.trust_level.value
            by_trust[trust] = by_trust.get(trust, 0) + 1
            if mem.actual_success:
                by_success["success"] += 1
            else:
                by_success["failure"] += 1
            temps.append(mem.temperature())
        
        return {
            "total": len(self._memories),
            "by_trust": by_trust,
            "by_success": by_success,
            "avg_temperature": sum(temps) / len(temps) if temps else 0,
            "patterns": len(self._patterns),
        }
    
    def to_context_block(self, max_entries: int = 5) -> str:
        """
        生成context注入块——VISTA dashboard思路。
        
        让Agent能看到自己因果记忆的状态。
        """
        stats = self.stats()
        lines = [
            f"## 因果记忆状态",
            f"总条数: {stats['total']} | 模式数: {stats['patterns']}",
            f"成功: {stats['by_success'].get('success', 0)} | 失败: {stats['by_success'].get('failure', 0)}",
            f"信任分布: {stats['by_trust']}",
            f"平均温度: {stats['avg_temperature']:.2f}",
            "",
        ]
        
        # 最近的高温度因果记忆
        recent = sorted(self._memories.values(), key=lambda m: m.temperature(), reverse=True)[:max_entries]
        if recent:
            lines.append("### 高温度因果记忆:")
            for mem in recent:
                emoji = "✅" if mem.actual_success else "❌"
                lines.append(f"  {emoji} [{mem.action_signature[:40]}] T={mem.temperature():.2f} | {mem.lesson[:60]}")
        
        # 最近的失败教训
        failed = [m for m in self._memories.values() if not m.actual_success]
        if failed:
            lines.append("### 最近失败教训:")
            for mem in sorted(failed, key=lambda m: m.created_at, reverse=True)[:3]:
                lines.append(f"  ❌ [{mem.action_signature[:40]}] | {mem.lesson[:60]}")
        
        return "\n".join(lines)
