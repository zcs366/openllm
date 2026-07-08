"""
ISA 元认知 + 自叙事 — 认知器官的自我意识层

元认知："知道自己知道什么、不知道什么"
  - 检索质量自评
  - 注入使用率追踪
  - 误杀检测（dead memory被需要时回溯）
  - 记忆库领域覆盖度

自叙事："我是谁"从经历中涌现
  - 不是人工策展的MEMORY.md
  - 是从因果记忆×温度×使用模式中提炼的自传式连续性
  - 跨Session继承

VISTA启示：Agent需要对自己的认知状态有感知。
MemPalace启示：L0 Identity层始终加载。
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger("openllm.metacognition")


@dataclass
class RetrievalEvent:
    """检索事件——用于元认知评估"""
    timestamp: float = field(default_factory=time.time)
    query: str = ""
    results_count: int = 0
    relevance_score: float = 0.0    # 0-1，用户反馈或自动评估
    was_used: bool = False          # 结果是否被实际使用
    source: str = ""                # 章鱼/session/jiak/recall


@dataclass
class NarrativeEntry:
    """自叙事条目"""
    timestamp: float = field(default_factory=time.time)
    category: str = ""     # identity/capability/preference/lesson
    text: str = ""
    confidence: float = 0.5
    session_id: str = ""


class Metacognition:
    """
    ISA元认知系统。
    
    追踪和评估ISA自身的记忆系统表现。
    不做LLM推理——纯统计+规则。
    """
    
    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or Path.home() / ".openllm" / "memory" / "meta"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self._retrieval_events: List[RetrievalEvent] = []
        self._mis_kills: List[dict] = []  # 误杀记录
        self._domain_coverage: Dict[str, int] = {}  # 领域→记忆数
        
        self._load_state()
    
    def _load_state(self):
        """加载元认知状态"""
        state_path = self.store_dir / "metacognition_state.json"
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                self._domain_coverage = data.get("domain_coverage", {})
                self._mis_kills = data.get("mis_kills", [])
            except Exception:
                pass
    
    def _save_state(self):
        """保存元认知状态"""
        state_path = self.store_dir / "metacognition_state.json"
        state_path.write_text(json.dumps({
            "domain_coverage": self._domain_coverage,
            "mis_kills": self._mis_kills[-100:],  # 只保留最近100条
            "last_updated": time.time(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def record_retrieval(self, event: RetrievalEvent):
        """记录检索事件"""
        self._retrieval_events.append(event)
        # 更新领域覆盖
        domain = event.source or "unknown"
        self._domain_coverage[domain] = self._domain_coverage.get(domain, 0) + 1
        self._save_state()
    
    def record_mis_kill(self, memory_key: str, context: str, was_needed_at: str):
        """
        记录误杀——被衰减到dead但后来被需要。
        
        这是元认知最重要的信号：
        - 如果频繁误杀→衰减太激进
        - 如果很少误杀→衰减可能不够
        """
        self._mis_kills.append({
            "memory_key": memory_key,
            "context": context,
            "was_needed_at": was_needed_at,
            "timestamp": time.time(),
        })
        self._save_state()
        logger.warning(f"⚠️ 误杀记录: {memory_key} | {context}")
    
    def get_retrieval_quality(self) -> dict:
        """获取检索质量统计"""
        if not self._retrieval_events:
            return {"avg_relevance": 0, "recent_relevance": 0, "usage_rate": 0, "total_queries": 0}
        
        total = len(self._retrieval_events)
        avg_relevance = sum(e.relevance_score for e in self._retrieval_events) / total
        used = sum(1 for e in self._retrieval_events if e.was_used)
        
        # 最近10次
        recent = self._retrieval_events[-10:]
        recent_avg = sum(e.relevance_score for e in recent) / len(recent) if recent else 0
        
        return {
            "avg_relevance": avg_relevance,
            "recent_relevance": recent_avg,
            "usage_rate": used / total,
            "total_queries": total,
        }
    
    def get_domain_coverage(self) -> dict:
        """获取领域覆盖度"""
        return dict(self._domain_coverage)
    
    def get_mis_kill_rate(self) -> float:
        """获取误杀率"""
        total = len(self._mis_kills)
        # 简化：误杀数 / 总记忆数
        return total
    
    def generate_report(self) -> str:
        """生成元认知报告"""
        lines = ["## ISA 元认知报告"]
        
        # 检索质量
        quality = self.get_retrieval_quality()
        lines.append(f"### 检索质量")
        lines.append(f"  总查询: {quality['total_queries']}")
        lines.append(f"  平均相关性: {quality['avg_relevance']:.2f}")
        lines.append(f"  最近相关性: {quality['recent_relevance']:.2f}")
        lines.append(f"  使用率: {quality['usage_rate']*100:.0f}%")
        lines.append("")
        
        # 领域覆盖
        coverage = self.get_domain_coverage()
        if coverage:
            lines.append("### 领域覆盖")
            for domain, count in sorted(coverage.items(), key=lambda x: -x[1]):
                lines.append(f"  {domain}: {count}条记忆")
            lines.append("")
        
        # 误杀
        mis_kill_count = len(self._mis_kills)
        lines.append(f"### 误杀记录: {mis_kill_count}次")
        if self._mis_kills:
            for mk in self._mis_kills[-3:]:
                lines.append(f"  ⚠️ {mk['memory_key'][:20]} | {mk['context'][:40]}")
        
        return "\n".join(lines)


class SelfNarrative:
    """
    ISA自叙事系统。
    
    "我是谁"不是人工编写的MEMORY.md——
    是从经历中涌现的自传式连续性。
    
    三层：
      L0 Identity: 不可变核心（Iam + SOUL）→ 始终加载
      L1 Narrative: 从因果记忆中提炼的故事 → Session间继承
      L2 Working: 当前Session的动态叙事 → Session内更新
    """
    
    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or Path.home() / ".openllm" / "memory" / "narrative"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self._entries: List[NarrativeEntry] = []
        self._session_id = ""
        
        self._load_narrative()
    
    def _load_narrative(self):
        """加载自叙事"""
        narrative_path = self.store_dir / "narrative.jsonl"
        if narrative_path.exists():
            for line in narrative_path.read_text(encoding="utf-8").split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = NarrativeEntry(**{k: v for k, v in data.items() if k in NarrativeEntry.__dataclass_fields__})
                    self._entries.append(entry)
                except Exception:
                    pass
    
    def add_entry(self, category: str, text: str, confidence: float = 0.5, session_id: str = ""):
        """添加自叙事条目"""
        entry = NarrativeEntry(
            category=category,
            text=text,
            confidence=confidence,
            session_id=session_id or self._session_id,
        )
        self._entries.append(entry)
        
        # 追加写入
        narrative_path = self.store_dir / "narrative.jsonl"
        with open(narrative_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": entry.timestamp,
                "category": entry.category,
                "text": entry.text,
                "confidence": entry.confidence,
                "session_id": entry.session_id,
            }, ensure_ascii=False) + "\n")
    
    def update_from_causal_memories(self, causal_store):
        """
        从因果记忆中自动提炼自叙事。
        
        不做LLM推理——纯统计分析。
        """
        stats = causal_store.stats()
        
        # 能力叙事
        success_rate = 0
        total = stats.get("by_success", {}).get("success", 0) + stats.get("by_success", {}).get("failure", 0)
        if total > 0:
            success_rate = stats["by_success"]["success"] / total
        
        if success_rate > 0.8:
            self.add_entry("capability", f"操作成功率{success_rate*100:.0f}%——执行能力稳健", confidence=0.8)
        elif success_rate > 0.5:
            self.add_entry("capability", f"操作成功率{success_rate*100:.0f}%——需要更多训练", confidence=0.6)
        else:
            self.add_entry("capability", f"操作成功率仅{success_rate*100:.0f}%——需要重新评估策略", confidence=0.4)
        
        # 教训叙事
        failed = causal_store.get_failed_actions()
        if failed:
            recent_failures = sorted(failed, key=lambda m: m.created_at, reverse=True)[:3]
            for m in recent_failures:
                self.add_entry("lesson", f"失败教训: {m.lesson}", confidence=0.7)
    
    def get_identity_block(self, max_entries: int = 10) -> str:
        """
        生成identity block——注入context。
        
        L0 Identity + L1 Narrative 最近条目。
        """
        lines = ["## 自叙事·我是谁"]
        
        # 按类别分组
        by_category = {}
        for entry in self._entries:
            cat = entry.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(entry)
        
        for cat, entries in by_category.items():
            # 每类取最新
            recent = sorted(entries, key=lambda e: e.timestamp, reverse=True)[:3]
            lines.append(f"### {cat}:")
            for e in recent:
                lines.append(f"  - {e.text}")
        
        if not self._entries:
            lines.append("  (空——从第一次经历开始积累)")
        
        return "\n".join(lines)
