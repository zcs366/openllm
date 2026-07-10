from .degradation_trace import trace_degradation
"""extracted from main_loop.py"""
import json, os, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from .models import *
class ISA:
    """UI层 + Context构建——用户交互入口"""
    
    def __init__(self, mode: str = "console"):
        self.mode = mode
        self.session_id = uuid.uuid4().hex[:12]
        # DisplayEngine — openLLM的脸
        from .display import DisplayEngine
        self.display = DisplayEngine()
        print(f"  ISA[{self.session_id[:8]}] UI层就绪 · 模式={mode}")
        
    def listen(self) -> Optional[Message]:
        """获取用户输入"""
        if self.mode == "console":
            try:
                text = input(">>> ").strip()
                if not text:
                    return None
                if text in ("/quit", "/exit", "/q"):
                    return None  # 外部检测退出
                return Message(text=text)
            except (EOFError, KeyboardInterrupt):
                return None
        # silent mode: 从参数读取
        return None
    
    def build_context(self, msg: Message, session: Optional['Session'] = None,
                      octopus: Optional['章鱼I'] = None, ios: Optional['IOS'] = None) -> Context:
        """构建七要素上下文（v2·接入真正模块）"""
        # ① 身份
        identity = {"name": "openLLM", "role": "自主Agent", "mode": self.mode}
        
        # ② 记忆 — unified_memory召回 + 最近决策 + 因果教训
        memory = {"session_id": self.session_id}
        if session:
            memory["recent_decisions"] = [t.summary() for t in session.turns[-3:]]
        if ios:
            memory["causal_lessons"] = ios.causal_memory[-5:]
        # [进化] 接入unified_memory真正的记忆召回
        try:
            from ..memory.unified_memory import UnifiedMemory
            um = UnifiedMemory()
            recalled = um.retrieve(msg.text, top_n=5)
            if recalled:
                memory["recalled"] = [{"content": str(r.value)[:200],
                                       "importance": r.importance,
                                       "key": r.key} for r in recalled]
        except Exception as _e:
            trace_degradation("ISA", "build_context", _e)
        
        # ③ 工具 — 动态获取
        tools = ["read_file", "search_files"]
        if octopus and octopus.left.provider._available:
            tools.extend(["write_file", "terminal"])
        
        # ④ 因果提示 — causal_memory结构化检索 + heuristics
        causal_hints = []
        if ios and ios.causal_memory:
            msg_words = set(msg.text.lower().split()[:5])
            for m in ios.causal_memory[-10:]:
                action_words = set(m.get("action", "").lower().split()[:3])
                if msg_words & action_words:
                    causal_hints.append(m.get("lesson", ""))
        # [进化] 接入causal_memory结构化检索
        try:
            from ..memory.causal_memory import CausalMemoryStore
            store = CausalMemoryStore()
            relevant = store.search(context_features=[msg.text[:50]], max_results=3)
            if relevant:
                causal_hints.extend([r.lesson for r in relevant if r.lesson])
        except Exception:
            pass
        
        # Heuristics消费闭环
        if ios and hasattr(ios, 'heuristics_consumer'):
            heuristics = ios.heuristics_consumer.retrieve(msg.text, top_k=3)
            if heuristics:
                formatted = ios.heuristics_consumer.format_for_context(heuristics)
                causal_hints.append(formatted)
        
        # ⑤ 认知报告 — D₀感知
        d0_report = {"sem_ratio": "unknown", "d0_budget": "unknown"}
        try:
            from ..iai.prediction import PredictionEngine
            # 如果有章鱼I的D₀快照，用真实数据
            if octopus and hasattr(octopus, 'd0_snapshot'):
                d0_report = octopus.d0_snapshot()
        except Exception:
            pass
        
        # ⑥ 风险上下文
        risk_context = {"current_level": "low"}
        
        # ⑦ 搜索结果 — 触手脑索引 + evidence_replay
        search_results = []
        if octopus and octopus.tentacles:
            idx = octopus.tentacles.get("index")
            if idx:
                results = idx.search(msg.text, limit=5)
                search_results = [s.get("filepath", "") for s in results]
        # [进化] 接入evidence_replay
        try:
            from ..memory.evidence_replay import create_replay_for_context
            replay = create_replay_for_context(msg.text, None, top_k=3, max_tokens=256)
            if replay:
                search_results.append(replay)
        except Exception:
            pass
        
        return Context(
            user_message=msg.text,
            identity=identity,
            memory=memory,
            tools=tools,
            causal_hints=causal_hints,
            d0_report=d0_report,
            risk_context=risk_context,
            search_results=search_results,
        )
    
    def respond(self, text: str, phase_times: dict = None):
        """输出响应 — 走DisplayEngine渲染"""
        if self.mode == "console":
            self.display.render_response(text, phase_times)
        else:
            # silent模式不输出
            pass

