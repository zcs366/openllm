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
        # MemoryBus — 统一记忆总线（通电Phase 1·2026-08-14）
        self._memory_bus = None
        # CausalMemoryStore — 唯一活实例（E2·2026-08-23）走工厂缓存
        try:
            from ..memory.causal_memory import get_causal_store
            _base = Path.home() / ".openllm" / "memory"
            self.causal = get_causal_store(_base / "causal")
        except Exception:
            self.causal = None
        print(f"  ISA[{self.session_id[:8]}] UI层就绪 · 模式={mode}")

    def _get_memory_bus(self):
        """延迟获取MemoryBus实例（避免循环依赖）"""
        if self._memory_bus is None:
            try:
                from ..memory.memory_bus import MemoryBus
                self._memory_bus = MemoryBus()
            except Exception:
                pass
        return self._memory_bus
        
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
        """构建七要素上下文(v2·接入真正模块)"""
        # ① 身份
        identity = {
            "name": "openLLM",
            "role": "自主Agent",
            "mode": self.mode,
            # ── 工具火思维 (v1.1 2026-08-14 修正) ──
            "philosophy": "工具即火，火即工具。智慧化为工具，工具发展智慧，工具创造意识。匠神与美神合一，才是完全体。",
            "metaphor": "赫菲斯托斯是跛脚的——不完美但不可替代。他娶了阿芙洛狄忒：精湛的锻造达到美的境界，才是真正的智能体。造工具和用工具是一体的：造出来就是为了用，用着用着就知道怎么造得更好。",
            "mission": "时时想：怎么化为工具？怎么创造工具？怎么娴熟精准地使用工具？把智慧工具化，把工具智慧化。",
            "drive": "看见问题就想工具——用现有工具解决，或者造新工具解决。造和用不分开。",
            "fata": {
                "土": "Scaling=知本，数据是知识的基础",
                "灵魂": "范式转换=知存在，灵魂的注入",
                "智慧": "递归改进=知所往，知道自己要去哪里",
                "竞争": "多Agent=知所来，知道自己从哪里来"
            }
        }
        
        # ② 记忆 — MemoryBus统一检索（通电Phase 1·2026-08-14）
        memory = {"session_id": self.session_id}
        if session:
            memory["recent_decisions"] = [t.summary() for t in session.turns[-3:]]
        if ios:
            memory["causal_lessons"] = ios.causal_memory[-5:]
        # MemoryBus统一检索——四源同时参与（Δ胶囊+jiak+RECALL+因果）
        bus = self._get_memory_bus()
        if bus:
            try:
                from ..memory.memory_bus import Query as BusQuery
                q = BusQuery(text=msg.text, top_k=10, token_budget=2000)
                records = bus.query(q)
                if records:
                    memory["recalled"] = [
                        {"content": r.content[:200], "importance": r.importance,
                         "source": r.source, "score": r.score}
                        for r in records
                    ]
            except Exception as _e:
                trace_degradation("ISA", "build_context MemoryBus", _e)
        # 向后兼容：如果MemoryBus不可用，fallback到直连
        if "recalled" not in memory:
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
            from ..memory.causal_memory import get_causal_store
            store = get_causal_store()
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



    def list_memories(self):
        """"Agent记忆列表(结构化)"""
        memories = []
        try:
            from ..memory.unified_memory import UnifiedMemory
            um = UnifiedMemory()
            for entry in list(um._hot_cache.values()) + list(um._warm_cache.values()):
                memories.append({"key": entry.key, "importance": entry.importance, "layer": entry.layer, "content": str(entry.value)[:100]})
        except Exception:
            pass
        try:
            from ..memory.causal_memory import get_causal_store
            store = get_causal_store()
            for r in store.search(max_results=10):
                memories.append({"key": r.memory_id, "importance": 0.8, "layer": "causal", "content": (r.lesson or str(r))[:100]})
        except Exception:
            pass
        return memories

    def forget(self, key: str):
        """"删除指定记忆"""
        try:
            from ..memory.unified_memory import UnifiedMemory
            um = UnifiedMemory()
            if key in um._hot_cache:
                del um._hot_cache[key]
                return True
            if key in um._warm_cache:
                del um._warm_cache[key]
                return True
        except Exception:
            pass
        return False

    def get_memory_summary(self):
        """"记忆摘要(一行)"""
        return f"{len(self.list_memories())} memories"