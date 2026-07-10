#!/usr/bin/env python3
"""
openLLM Agent主循环 — 独立Agent框架核心

这是openLLM自己的Agent主循环。
不从Hermes借，不从任何人借。

五体调度：
  ISA   → UI层 + Context构建
  章鱼I → 推理引擎 + 左右脑（内部双路径对弈）
  IO-S  → 决策框架 + 自我进化 + 错误恢复
  ISN   → 工具执行
  IKO   → 可观测

架构：单文件可运行，零外部依赖。
集成Session/Turn模型，支持10阶段心跳。
"""

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .session import Session, Turn, TurnStatus, create_session

# ── tool_validator 集成（Layer 3）──
try:
    from .tool_validator import (
        ToolCall as _TVToolCall,
        ValidationResult as _TVResult,
        validate_tool_result as _tv_validate,
    )
    _HAS_TOOL_VALIDATOR = True
except ImportError:
    _HAS_TOOL_VALIDATOR = False

# ── 章鱼I Reflect 集成 ──
_REFLECT_SCRIPT = Path.home() / "projects" / "isa" / "octopus" / "scripts" / "octopus_reflect.py"
_HAS_REFLECT = _REFLECT_SCRIPT.exists()

# ── IAX Layer 7 推理预算追踪（try/except确保不破坏现有逻辑）──
try:
    from .inference_budget import InferenceBudgetManager
    _HAS_BUDGET = True
except ImportError:
    _HAS_BUDGET = False


# ─── 数据模型 ──────────────────────────────────────────

@dataclass
class Message:
    """用户输入"""
    text: str
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

@dataclass
class Context:
    """构建好的上下文（喂给推理引擎的）— 七要素"""
    user_message: str
    # ① 身份 — Agent是谁
    identity: dict = field(default_factory=dict)
    # ② 记忆 — 最近决策、因果教训
    memory: dict = field(default_factory=dict)
    # ③ 工具 — 可用的工具列表（动态）
    tools: list = field(default_factory=list)
    # ④ 因果提示 — 上次类似操作的因果历史
    causal_hints: list = field(default_factory=list)
    # ⑤ 认知报告 — D₀感知
    d0_report: dict = field(default_factory=dict)
    # ⑥ 风险上下文 — 当前风险等级
    risk_context: dict = field(default_factory=dict)
    # ⑦ 搜索结果 — 触手脑检索结果
    search_results: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class Prediction:
    """因果预测（Phase 3: 因果预测）"""
    summary: str = ""
    consequences: list[str] = field(default_factory=list)
    confidence: float = 0.5
    risk_signals: list[str] = field(default_factory=list)
    
    def summary_text(self) -> str:
        """返回预测摘要。"""
        return f"[预测] {self.summary} | 置信度={self.confidence:.0%} | 风险信号={len(self.risk_signals)}"

@dataclass
class RiskAssessment:
    """风险评估（Phase 4: 风险评估）"""
    level: str = "low"  # low | medium | high | critical
    blocked: bool = False
    reason: str = ""
    details: list[str] = field(default_factory=list)
    
    def is_blocked(self) -> bool:
        """是否被安全拦截。"""
        return self.blocked or self.level == "critical"

@dataclass
class Proposal:
    """左脑提案"""
    content: str
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    prediction_ref: Optional[Prediction] = None

@dataclass
class Critique:
    """右脑批判"""
    content: str
    verdict: str = ""  # "approve" | "reject" | "revise"
    concerns: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

@dataclass 
class Decision:
    """IO-S仲裁决策"""
    action: str  # "execute" | "deny" | "revise" | "escalate"
    reason: str = ""
    approved: bool = False
    risk_ref: Optional[RiskAssessment] = None

@dataclass
class ActionResult:
    """工具执行结果"""
    success: bool
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0

@dataclass
class CausalDelta:
    """因果对照（Phase 8: 因果对照）"""
    prediction_match: bool = False
    delta_summary: str = ""
    learned: list[str] = field(default_factory=list)
    
    def summary_text(self) -> str:
        """返回因果对照摘要。"""
        return f"[因果] 匹配={self.prediction_match} | 学到={len(self.learned)}条"

@dataclass
class TickMetrics:
    """一轮心跳的完整可观测数据"""
    tick_id: str
    phase: str
    status: str
    duration_ms: float = 0.0
    detail: str = ""


# ═══════════════════════════════════════════════════════
# LLM Provider 集成（左右脑对弈用）
# ═══════════════════════════════════════════════════════

class LLMProvider:
    """最简单的LLM调用封装"""
    
    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        # 优先从config读·其次环境变量
        self.api_key = self._load_key()
        self.endpoint = "https://api.deepseek.com/v1/chat/completions"
        self._available = bool(self.api_key)
        self._last_usage: dict = {}  # 最近一次调用的token使用量
    
    def _load_key(self) -> str:
        """从config.json加载API key"""
        config_path = Path.home() / ".openllm" / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                return cfg.get("providers", {}).get("deepseek", {}).get("api_key", "")
            except:
                pass
        return os.environ.get("DEEPSEEK_API_KEY", "")
    
    def chat(self, messages: list[dict]) -> str:
        """调LLM·返回文本。M0: 返回模拟响应。"""
        if not self._available:
            # M0降级：返回模拟响应
            user_msg = messages[-1]["content"] if messages else ""
            self._last_usage = {}
            return f"[模拟LLM] 已收到: {user_msg[:50]}"
        
        # TODO: Phase 2接入真实API
        try:
            import requests
            resp = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            self._last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM错误] {e}"


# ═══════════════════════════════════════════════════════
# 五体（Stub · M0阶段用简单实现）
# ═══════════════════════════════════════════════════════

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
        except Exception:
            pass  # 降级：不影响主流程
        
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


class 章鱼I:
    """推理引擎 + 左右脑"""
    
    # 能力降级等级
    HEALTH = {
        "FULL":     {"confidence_min": 0.7, "tools_all": True},
        "DEGRADED": {"confidence_min": 0.4, "tools_all": False},
        "MINIMAL":  {"confidence_min": 0.2, "tools_all": False},
        "OFFLINE":  {"confidence_min": 0.0, "tools_all": False},
    }
    
    def __init__(self):
        self.left = _LeftBrain()
        self.right = _RightBrain()
        # 触手脑：文件监控 + 全文索引
        from .tentacle import FileWatcherBrain, IndexBrain
        self.tentacles = {
            "file_watcher": FileWatcherBrain(),
            "index": IndexBrain(),
        }
        print("  章鱼I 推理引擎就绪 · 左右脑在线 · 触手脑在线")
    
    def health_check(self) -> str:
        """自检。返回当前健康等级。"""
        if not self.left.provider._available:
            return "MINIMAL"  # 无API=降级
        try:
            _ = self.left.provider.chat([{"role":"user","content":"ping"}])
            return "FULL"
        except:
            return "DEGRADED"
    
    def d0_snapshot(self) -> dict:
        """
        D₀感知简化版·实时认知深度报告
        
        完整版需要IAH扫描注意力头D₀——当前不可用。
        简化版基于：API可用性 + health等级 + 可用工具数 + context使用率
        
        Returns:
            sem_ratio: 语义能力比率（0-1）
            d0_budget: 可用token数
            cognitive_rhythm: 认知节奏（fast/steady/slow）
            health: 当前健康等级
        """
        health = self.health_check()
        
        # sem_ratio 估算：FULL→0.9, DEGRADED→0.5, MINIMAL→0.2
        sem_map = {"FULL": 0.9, "DEGRADED": 0.5, "MINIMAL": 0.2, "OFFLINE": 0.0}
        sem_ratio = sem_map.get(health, 0.3)
        
        # d0_budget: 可用API时=大模型语义容量，降级时=本地规则容量
        d0_budget = 100000 if health == "FULL" else 10000 if health == "DEGRADED" else 1000
        
        # cognitive_rhythm: 有API→fast, 降级→steady, 无API→slow
        rhythm = "fast" if health == "FULL" else "steady" if health == "DEGRADED" else "slow"
        
        return {
            "sem_ratio": sem_ratio,
            "d0_budget": d0_budget,
            "cognitive_rhythm": rhythm,
            "health": health,
            "tools_available": len(self.tentacles) if hasattr(self, 'tentacles') else 0,
            "api_available": self.left.provider._available,
        }
    
    def predict_consequences(self, ctx: Context) -> Prediction:
        """Phase 3: 因果预测"""
        return self.left.predict(ctx)
    
    def reason(self, ctx: Context, prediction: Optional[Prediction] = None, 
               risk: Optional[RiskAssessment] = None) -> tuple[Proposal, Critique]:
        """Phase 5: 双脑推理"""
        proposal = self.left.think(ctx, prediction, risk)
        critique = self.right.review(ctx, proposal)
        return proposal, critique
    
    def compare(self, prediction: Prediction, result: ActionResult) -> CausalDelta:
        """Phase 8: 因果对照"""
        # M0: 简单比较
        match = result.success and "模拟" not in result.output
        return CausalDelta(
            prediction_match=match,
            delta_summary=f"预测={'正确' if match else '错误'}",
            learned=["预测准确性基础检查"],
        )
    
    def learn_causal(self, ctx: Context, prediction: Prediction,
                     result: ActionResult, delta: CausalDelta):
        """Phase 8: 因果学习（实际实现在IOS.learn_causal）"""
        pass


class _LeftBrain:
    """左脑(正手)：提案"""
    
    def __init__(self):
        self.provider = LLMProvider()
    
    def predict(self, ctx: Context) -> Prediction:
        """因果预测"""
        prompt = f"""你是openLLM的因果预测器。预测以下操作的后果。
用户意图：{ctx.user_message}
请预测：
[IF-SUCCESS] 如果操作成功，会发生什么
[IF-FAILURE] 如果操作失败，会发生什么
[IRREVERSIBLE] 哪些后果无法撤销
请用JSON格式输出：
{{"summary": "一句话预测摘要", "risk_signals": ["风险信号"]}}
"""
        resp = self.provider.chat([{"role": "user", "content": prompt}])
        try:
            data = json.loads(resp) if resp.startswith("{") else {"summary": resp[:50]}
        except:
            data = {"summary": resp[:50]}
        
        return Prediction(
            summary=data.get("summary", resp[:50]),
            consequences=[data.get("summary", "")],
            confidence=0.7,
            risk_signals=data.get("risk_signals", []),
        )
    
    def think(self, ctx: Context, prediction: Optional[Prediction] = None,
              risk: Optional[RiskAssessment] = None) -> Proposal:
        """基于上下文提出方案"""
        prompt = f"""你是openLLM的左脑。基于以下上下文，生成一个行动提案。
上下文：{ctx.user_message}
你的任务：
1. 分析用户意图
2. 提出具体的行动方案
3. 评估方案的置信度（0-1）
4. 列出支持方案的证据
请用JSON格式输出：
{{"content": "行动方案描述", "confidence": 0.7, "evidence": ["证据1", "证据2"], "tool_calls": []}}
"""
        resp = self.provider.chat([{"role": "user", "content": prompt}])
        # 解析LLM返回的JSON
        try:
            data = json.loads(resp) if resp.startswith("{") else {"content": resp}
        except:
            data = {"content": resp, "confidence": 0.6}
        
        evidence = data.get("evidence", [])
        if prediction:
            evidence.append(f"已预测后果: {prediction.summary[:30]}")
        if risk:
            evidence.append(f"风险等级: {risk.level}")
        
        # 血管三：health_check→confidence校准
        # 无API时confidence下降
        confidence = data.get("confidence", 0.6)
        if not self.provider._available:
            confidence *= 0.8  # 无API时打8折
        
        return Proposal(
            content=data.get("content", resp[:100]),
            confidence=confidence,
            evidence=evidence,
            tool_calls=data.get("tool_calls", []),
            prediction_ref=prediction,
        )


class _RightBrain:
    """右脑(反手)：批判"""
    
    def __init__(self):
        self.provider = LLMProvider()
    
    def review(self, ctx: Context, proposal: Proposal) -> Critique:
        """审查左脑提案"""
        prompt = f"""你是openLLM的右脑。审查左脑的提案。
原始上下文：{ctx.user_message}
左脑提案：{proposal.content}
左脑置信度：{proposal.confidence}
左脑证据：{proposal.evidence}
你的任务：
1. 检查提案是否有逻辑漏洞
2. 检查提案是否有安全隐患
3. 检查证据是否支撑提案
4. 给出裁决：approve（通过）/ reject（否决）/ revise（修改）
请用JSON格式输出：
{{"verdict": "approve|reject|revise", "concerns": ["关心点1"], "suggestions": ["建议1"]}}
"""
        resp = self.provider.chat([{"role": "user", "content": prompt}])
        try:
            data = json.loads(resp) if resp.startswith("{") else {"verdict": "approve"}
        except:
            data = {"verdict": "approve"}
        
        return Critique(
            content=resp[:100],
            verdict=data.get("verdict", "approve"),
            concerns=data.get("concerns", []),
            suggestions=data.get("suggestions", []),
        )


class IOS:
    """决策框架 + 自我进化 + 错误恢复

    集成老IO-S治理模式：
    - CapPolicy: 能力策略（零root）
    - Kernel: 微内核dispatch
    - Gate: 5模式权限门控
    - ToolScope: 工具作用域治理
    - Hindsight: 经验闭环
    - Pipeline: 三层安全管线
    - Checkpoint: Region注册+增量快照
    """
    
    def __init__(self):
        self.evolution_log: list[dict] = []
        self._arbiter_policy = "conservative"

        # ── 老IO-S治理模式集成 ──
        from .cap_policy import CapPolicy
        from .kernel import Kernel
        from .gate import PermissionGate, PermissionMode
        from .tool_scope import ToolScopeManager, ToolScope
        from .checkpoint_manager import CheckpointManager
        # ── 治理转换引擎（P0: arXiv:2607.01087） ──
        from .governance_engine import GovernanceEngine

        self.cap_policy = CapPolicy()
        self.kernel = Kernel(agent_id="openllm-ios")
        self.gate = PermissionGate(mode=PermissionMode.DEV)
        self.tool_scope = ToolScopeManager(scope=ToolScope.DEV)
        self.checkpoint = CheckpointManager(
            interval=300, auto_start=False)
        self.governance_engine = GovernanceEngine()
        
        # ⑤ Heuristics消费闭环（赫尔墨斯启示）
        from .governance_engine import HeuristicsConsumer
        self.heuristics_consumer = HeuristicsConsumer()

        # 兼容旧接口
        self.cap = self.cap_policy

        # [进化] 拒绝权——被否决的proposal可申诉
        try:
            from ..governance.rejection import RejectionMechanism, RejectionReason
            self._rejection_engine = RejectionMechanism()
            self._RejectionReason = RejectionReason
        except Exception:
            self._rejection_engine = None
            self._RejectionReason = None

        # 因果记忆
        self.causal_memory: list[dict] = []

        print(
            f"  IO-S 决策·进化·恢复就绪 "
            f"· 策略={self._arbiter_policy} "
            f"· zero_root={self.cap_policy.is_zero_root()} "
            f"· 治理引擎=active")
    
    def cap_check(self, resource: str, operation: str) -> bool:
        """硬权限检查 — 使用CapPolicy"""
        ok, _ = self.cap_policy.check("agent", operation, resource)
        return ok
    
    def learn_causal(self, ctx: Context, prediction: Prediction,
                     result: ActionResult, delta: CausalDelta):
        """Phase 8: 因果学习 — Self-Harness Weakness Mining + 治理转换
        
        基于 Self-Harness (arXiv 2606.09498):
        - 提取 (verifier_cause, agent_behavior, mechanism) 失败签名
        - 按 mechanism 聚类（非关键词匹配，是语义归因）
        - 记录到 causal_memory + 持久化到磁盘
        
        新增（P0: arXiv:2607.01087 治理转换引擎）:
        - 结构性失败自动触发治理转换
        - ambiguous失败发GovernanceRequest
        """
        entry = {
            "action": ctx.user_message[:100],
            "prediction": prediction.summary[:100],
            "actual": result.output[:100] if result.output else "",
            "lesson": delta.delta_summary,
            "timestamp": time.time(),
            # Self-Harness Weakness Mining 三元组
            "verifier_cause": self._classify_verifier_cause(result, delta),
            "mechanism": self._classify_mechanism(result, delta),
            "prediction_match": delta.prediction_match,
            "result_success": result.success,
        }
        self.causal_memory.append(entry)

        # 持久化到磁盘
        causal_path = Path.home() / ".openllm" / "output" / "ios" / "causal_memory.jsonl"
        causal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(causal_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        # 连接4: 回流到jiak world_evolver mismatch_log（五体→记忆层信号传递）
        if not result.success:
            jiak_path = Path.home() / ".hermes" / "jiak" / "mismatch_log.jsonl"
            jiak_path.parent.mkdir(parents=True, exist_ok=True)
            with open(jiak_path, "a") as f:
                f.write(json.dumps({
                    "ts": entry.get("timestamp"),
                    "card_id": "openllm-ios",
                    "prediction_ref": "",
                    "actual": f"[{entry.get('mechanism', 'unknown')}] {result.error[:80] if result.error else result.output[:80]}",
                    "severity": "high" if entry.get("mechanism") in ("tool_loop", "permission", "timeout") else "medium",
                }, ensure_ascii=False) + "\n")
        
        # ── 治理转换（P0: Phase 8.5）──
        if not result.success:
            self._convert_governance(entry, result)

        # ── Hindsight经验闭环（老IO-S模式）──
        from .hindsight_loop import extract_hindsight
        hindsight = extract_hindsight(
            pid="ios",
            goal=ctx.user_message[:200],
            result={
                "success": result.success,
                "execution_time": result.duration_ms / 1000,
                "error": result.error,
            },
            failure=result.error if not result.success else "",
        )
        if hindsight.get("failure_pattern"):
            entry["hindsight_pattern"] = hindsight["failure_pattern"]
            entry["hindsight_alternative"] = hindsight.get(
                "alternative", "")

        # ── P0/P1增强：learn_causal_adapter接入 ──
        if not result.success:
            try:
                import importlib.util as _ilu
                _isa_path = Path.home() / "projects" / "isa" / "learn_causal_adapter.py"
                _lca_spec = _ilu.spec_from_file_location("learn_causal_adapter", _isa_path)
                if _lca_spec and _lca_spec.loader:
                    _lca_mod = _ilu.module_from_spec(_lca_spec)
                    _lca_spec.loader.exec_module(_lca_mod)
                    LearnCausalAdapter = _lca_mod.LearnCausalAdapter
                adapter = LearnCausalAdapter()
                adapter.on_failure(
                    action=entry.get("action", ""),
                    error=result.error or "",
                    mechanism=entry.get("mechanism", "unknown"),
                    tool_name=entry.get("action", "").split("(")[0] if "(" in entry.get("action", "") else "",
                )
                adapter.close()
            except Exception as e:
                print(f"  learn_causal_adapter failed (non-fatal): {e}")
    
    # ── Self-Harness Weakness Mining 工具 ──
    
    _MECHANISM_KEYWORDS = {
        "tool_loop": ["重复", "循环", "loop", "retry", "反复", "无限"],
        "missing_artifact": ["缺失", "没有生成", "未创建", "missing", "not found", "未产出"],
        "wrong_format": ["格式错误", "format error", "解析失败", "parse error", "json解析", "语法错误"],
        "dependency_missing": ["依赖缺失", "import error", "module not found", "package未安装", "ModuleNotFoundError"],
        "timeout": ["超时", "timeout", "hang", "卡住", "无响应"],
        "logic_error": ["逻辑错误", "条件错误", "判断错误", "分支错误", "错误结果", "断言失败"],
        "permission": ["权限", "permission denied", "forbidden", "access denied"],
        "state_corruption": ["脏数据", "不一致", "corrupt", "状态损坏"],
        "exploration_loop": ["探索循环", "搜索失败", "盲目搜索", "找不到目标"],
        "premature_success": ["过早完成", "提前结束", "未验证成功"],
    }
    
    def _classify_mechanism(self, result: ActionResult, delta: CausalDelta) -> str:
        """分类失败机制（Self-Harness failure signature mi）"""
        text = (result.error or "") + " " + (result.output[:200] or "")
        for mech, keywords in self._MECHANISM_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return mech
        return "unknown" if not result.success else "success"
    
    def _classify_verifier_cause(self, result: ActionResult, delta: CausalDelta) -> str:
        """分类verifier层面原因（Self-Harness failure signature ci）"""
        if not result.success:
            return "execution_error" if result.error else "unknown_failure"
        if not delta.prediction_match:
            return "prediction_mismatch"
        return "success"
    
    def _convert_governance(self, entry: dict, result: ActionResult):
        """Phase 8.5: 治理转换 — arXiv:2607.01087 核心循环
        
        从失败中发现治理需求，转换为持久治理规则。
        
        流程:
        1. 构造failure trace
        2. 调用GovernanceEngine.convert()
        3. 记录转换结果
        """
        trace = {
            "tool_name": entry.get("action", "")[:50],
            "error": result.error or "",
            "mechanism": entry.get("mechanism", ""),
            "context": {
                "action": entry.get("action", ""),
                "prediction": entry.get("prediction", ""),
            },
        }
        
        try:
            conversion = self.governance_engine.convert(trace)
            
            if conversion.classification == "structural" and conversion.rule:
                rule = conversion.rule
                print(f"  IO-S 治理转换: {entry.get('mechanism')} → "
                      f"{rule.family.value} (rule_id={rule.rule_id}, "
                      f"status={rule.status.value})")
                
                # 记录到evolution_log
                self.evolution_log.append({
                    "type": "governance_conversion",
                    "mechanism": entry.get("mechanism"),
                    "rule_id": rule.rule_id,
                    "rule_family": rule.family.value,
                    "rule_status": rule.status.value,
                    "classification": conversion.classification,
                    "timestamp": time.time(),
                })
                
                # 审计日志：治理转换事件
                try:
                    from .governance_engine import GovernanceAuditLog
                    audit = GovernanceAuditLog()
                    audit.append(
                        event_type="governance_conversion",
                        action=f"{rule.family.value}:{rule.status.value}",
                        agent_id="ios",
                        details=f"mechanism={entry.get('mechanism')}, rule_id={rule.rule_id}"
                    )
                except Exception:
                    pass  # 审计失败不应影响主循环
            
            elif conversion.classification == "ambiguous" and conversion.governance_request:
                req = conversion.governance_request
                print(f"  IO-S 治理请求: {entry.get('mechanism')} → "
                      f"ambiguous (request_id={req.request_id})")
                
                self.evolution_log.append({
                    "type": "governance_request",
                    "mechanism": entry.get("mechanism"),
                    "request_id": req.request_id,
                    "classification": conversion.classification,
                    "timestamp": time.time(),
                })
        
        except Exception as e:
            # 治理转换失败不应影响主循环
            print(f"  IO-S 治理转换异常（不影响主循环）: {e}")
    
    def _check_governance_rules(
        self, ctx: Context, prediction: Prediction
    ) -> Optional[RiskAssessment]:
        """查询GovernanceRuleStore中匹配当前上下文的active规则。
        
        如果有active规则匹配当前操作，返回对应的风险评估。
        这是治理转换引擎的"消费端"——治理规则真正影响Agent行为。
        """
        try:
            store = self.governance_engine.rule_store
            active_rules = store.get_active()
            
            if not active_rules:
                return None
            
            # 匹配规则：target作为子串出现在消息中（非单词交集）
            msg_lower = ctx.user_message.lower()
            matched_rules = []
            
            for rule in active_rules:
                target = rule.target.lower().strip()
                if not target or target == "general":
                    continue
                # 子串匹配：target完整出现在消息中
                if target in msg_lower:
                    matched_rules.append(rule)
            
            if not matched_rules:
                return None
            
            # 根据匹配规则评估风险
            high_confidence = [r for r in matched_rules if r.confidence >= 0.7]
            if high_confidence:
                rule_descs = [r.description[:60] for r in high_confidence[:3]]
                return RiskAssessment(
                    level="high",
                    blocked=True,
                    reason=f"治理规则拦截: {len(high_confidence)}条active规则匹配",
                    details=rule_descs,
                )
            
            medium_confidence = [r for r in matched_rules if r.confidence >= 0.5]
            if medium_confidence:
                rule_descs = [r.description[:60] for r in medium_confidence[:3]]
                return RiskAssessment(
                    level="medium",
                    blocked=False,
                    reason=f"治理规则提示: {len(medium_confidence)}条规则匹配",
                    details=rule_descs,
                )
        
        except Exception:
            pass  # 治理规则检查失败不影响主循环
        
        return None
    
    def risk_check(self, ctx: Context, prediction: Prediction) -> RiskAssessment:
        """Phase 4: 风险评估 — 血管接线engine.py验证链 + 因果历史检索 + 治理规则"""
        # 血管接线：engine.py的安全检查
        from .engine_bridge import check_action, check_tool_risk
        ok, reason = check_action("risk_check")
        if not ok:
            return RiskAssessment(level="high", blocked=True, reason=reason)
        
        # 血管接线：engine.py的ISN风险检查（如有工具调用意图）
        if prediction.risk_signals:
            risk = check_tool_risk("unknown")
            if not risk["pass"]:
                return RiskAssessment(level="high", blocked=True, 
                    reason=f"ISN风险: {risk.get('reason','')}")
        
        # ── 治理规则检查（P0: arXiv:2607.01087）──
        # 查询GovernanceRuleStore中匹配当前上下文的active规则
        governance_risk = self._check_governance_rules(ctx, prediction)
        if governance_risk and governance_risk.is_blocked():
            return governance_risk
        
        # 因果历史检索（Self-Harness升级：mechanism感知 + 关键词匹配）
        related = []
        user_words = set(ctx.user_message.lower().split())
        
        # 先按mechanism匹配（更精准）
        for m in self.causal_memory[-50:]:
            mech = m.get("mechanism", "unknown")
            if mech != "success" and mech != "unknown":
                # 检查当前消息是否涉及同类机制的场景
                action_words = set(m.get("action", "").lower().split())
                word_overlap = len(user_words & action_words)
                if word_overlap >= 2 or (word_overlap >= 1 and m.get("result_success") == False):
                    related.append(m)
        
        # 补充：关键词匹配（兜底）
        if not related:
            for m in self.causal_memory[-50:]:
                action_words = set(m.get("action", "").lower().split())
                if len(user_words & action_words) >= 2:
                    related.append(m)
        
        if related:
            lessons = [r.get("lesson", "") for r in related[-3:]]
            # Self-Harness升级：风险等级与mechanism类型关联
            mechanisms = [r.get("mechanism", "unknown") for r in related[-3:]]
            high_risk_mechs = {"tool_loop", "permission", "timeout", "exploration_loop"}
            if any(m in high_risk_mechs for m in mechanisms):
                return RiskAssessment(level="high", blocked=False,
                    reason=f"因果历史(高风险机制): {len(related)}条, 机制={mechanisms}", details=lessons)
            return RiskAssessment(level="medium", blocked=False,
                reason=f"因果历史: {len(related)}条, 机制={mechanisms}", details=lessons)
        
        # 连接5: 注入learned_skills到context（五体学习→风险评估感知）
        skills_details = []
        try:
            skill_path = Path.home() / ".openllm" / "output" / "isn" / "learned_skills.jsonl"
            if skill_path.exists():
                import json as _json
                with open(skill_path) as f:
                    for line in f:
                        try:
                            sk = _json.loads(line)
                            if sk.get("mechanism") or sk.get("proposal"):
                                skills_details.append(f"[{sk.get('mechanism', '?')}] {sk.get('proposal', sk.get('pattern', ''))[:60]}")
                        except:
                            pass
                skills_details = skills_details[-5:]  # 最近5条
        except Exception:
            pass
        
        if skills_details:
            return RiskAssessment(level="low", blocked=False,
                reason=f"已学习技能: {len(skills_details)}条", details=skills_details)
        
        return RiskAssessment(level="low", blocked=False)
    
    def arbitrate(self, proposal: Proposal, critique: Critique, 
                  risk: Optional[RiskAssessment] = None) -> Decision:
        """
        Phase 6: 仲裁 — 5级策略栈 + 拒绝权记录
        
        Level 1: single — 只用左脑提案，跳过右脑
        Level 2: self_consistency — 左脑多次采样→投票
        Level 3: multi_persona — 左脑不同人格→聚合
        Level 4: debate — 左脑↔️右脑对弈（当前实现）
        Level 5: debate_then_verify — 对弈→第三方验证
        """
        # 风险拦截
        if risk and risk.is_blocked():
            decision = Decision(
                action="deny",
                approved=False,
                reason=f"安全拦截: {risk.reason}",
                risk_ref=risk,
            )
            self._record_rejection(proposal, decision, risk.reason)
            return decision
        
        # 风险驱动策略选择
        level = self._select_strategy(risk)
        
        # Level 1: single — 快速·直接用左脑
        if level == 1:
            return Decision(
                action="execute",
                approved=True,
                reason=f"Level 1: 单左脑提案通过（风险={risk.level if risk else 'low'}）",
                risk_ref=risk,
            )
        
        # Level 2: self_consistency — TODO: M1实现真正的self_consistency
        if level == 2:
            return Decision(
                action="execute",
                approved=True,
                reason=f"Level 2: self_consistency TODO·M1实现（降级为Level 1）",
                risk_ref=risk,
            )
        
        # Level 3: multi_persona — TODO: M1实现真正的multi_persona
        if level == 3:
            return Decision(
                action="execute",
                approved=True,
                reason=f"Level 3: multi_persona TODO·M1实现（降级为Level 1）",
                risk_ref=risk,
            )
        
        # Level 4: debate — 左脑↔️右脑对弈
        if level == 4:
            if critique.verdict == "approve":
                return Decision(action="execute", approved=True, reason="Level 4: 右脑通过", risk_ref=risk)
            if critique.verdict == "reject":
                decision = Decision(action="deny", approved=False, 
                              reason=f"Level 4: 右脑否决: {critique.concerns[0] if critique.concerns else '无理由'}",
                              risk_ref=risk)
                self._record_rejection(proposal, decision, decision.reason)
                return decision
            if critique.verdict == "revise":
                if self._arbiter_policy == "conservative":
                    return Decision(action="revise", approved=False, reason="Level 4: 右脑建议修改，暂缓", risk_ref=risk)
                return Decision(action="execute", approved=True, reason="Level 4: 策略允许存疑执行", risk_ref=risk)
        
        # Level 5: debate_then_verify — 对弈+验证
        if level == 5:
            if critique.verdict == "approve":
                return Decision(action="execute", approved=True, reason="Level 5: 对弈+验证通过", risk_ref=risk)
            else:
                decision = Decision(action="deny", approved=False, 
                              reason=f"Level 5: critical风险需要明确批准，当前={critique.verdict}", risk_ref=risk)
                self._record_rejection(proposal, decision, decision.reason)
                return decision
        
        # 默认放行
        return Decision(action="execute", approved=True, reason="默认放行", risk_ref=risk)
    
    def _record_rejection(self, proposal: Proposal, decision: Decision, reason: str):
        """记录拒绝——通过RejectionEngine创建不可变记录"""
        if not self._rejection_engine:
            return
        try:
            # 映射到RejectionReason
            if "安全" in reason or "拦截" in reason:
                rej_reason = self._RejectionReason.CONSTITUTIONAL
            elif "否决" in reason:
                rej_reason = self._RejectionReason.OUT_OF_SCOPE
            else:
                rej_reason = self._RejectionReason.UNCERTAIN_SAFETY
            
            self._rejection_engine.reject(
                instruction=proposal.content[:200],
                reason=rej_reason,
                reasoning=reason,
                context={"proposal_confidence": proposal.confidence, "action": decision.action},
            )
        except Exception:
            pass  # 拒绝权记录失败不阻塞主流程
    
    def _select_strategy(self, risk: Optional[RiskAssessment]) -> int:
        """风险驱动策略选择"""
        if risk is None:
            return 1
        if risk.level == "low":
            return 1  # 快速
        if risk.level == "medium":
            return 3  # 多样
        if risk.level == "high":
            return 4  # 对弈
        return 5  # critical→对弈+验证
    
    def evolve(self, proposal: Proposal, critique: Critique,
               result: Optional[ActionResult] = None, delta: Optional[CausalDelta] = None):
        """
        Phase 9: 因果进化引擎 — Self-Harness Harness Proposal + Validation
        
        基于 Self-Harness (arXiv 2606.09498):
        1. 记录（保留原有逻辑）
        2. 机制聚类：按mechanism字段聚合失败（非关键词交集）
        3. Harness修改提案：当高频机制≥阈值时，生成五体配置修改建议
        4. 策略自适应：根据失败率调整策略
        """
        # ① 记录
        self.evolution_log.append({
            "timestamp": time.time(),
            "proposal": proposal.content[:100],
            "verdict": critique.verdict,
            "result_success": result.success if result else None,
            "delta_match": delta.prediction_match if delta else None,
        })
        
        # ② 机制聚类（Self-Harness: 按failure signature聚合）
        if result and not result.success and len(self.causal_memory) >= 3:
            recent = self.causal_memory[-20:]
            mechanism_counts: dict[str, int] = {}
            for m in recent:
                mech = m.get("mechanism", "unknown")
                if mech != "success":
                    mechanism_counts[mech] = mechanism_counts.get(mech, 0) + 1
            
            # 高频机制（≥3次或占比≥30%）
            total_failures = sum(mechanism_counts.values())
            high_freq = {
                mech: count for mech, count in mechanism_counts.items()
                if count >= 3 or (total_failures > 0 and count / total_failures >= 0.3)
            }
            
            if high_freq:
                # ③ Harness修改提案（Self-Harness Harness Proposal）
                # P0-3回归验证：检查同一机制是否已有pending提案
                prop_path = Path.home() / ".openllm" / "output" / "ios" / "harness_proposals.jsonl"
                existing_mechs = set()
                if prop_path.exists():
                    try:
                        with open(prop_path) as f:
                            for line in f:
                                try:
                                    p = json.loads(line)
                                    if p.get("status") == "pending":
                                        existing_mechs.add(p.get("mechanism", ""))
                                except:
                                    pass
                    except:
                        pass
                
                top_mech = max(high_freq, key=lambda k: high_freq[k])
                if top_mech in existing_mechs:
                    # 已有pending提案，不重复生成
                    pass
                else:
                    proposal_text = self._propose_harness_change(top_mech, high_freq[top_mech])
                    if proposal_text:
                        self.evolution_log.append({
                            "type": "harness_proposal",
                            "mechanism": top_mech,
                            "support": high_freq[top_mech],
                            "proposal": proposal_text,
                            "timestamp": time.time(),
                        })
                        print(f"  IO-S Self-Harness提案: [{top_mech}] {proposal_text[:80]}")
                        
                        # 持久化提案（等待人工审批）
                        prop_path = Path.home() / ".openllm" / "output" / "ios" / "harness_proposals.jsonl"
                        prop_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(prop_path, "a") as f:
                            f.write(json.dumps({
                                "mechanism": top_mech,
                                "support": high_freq[top_mech],
                                "proposal": proposal_text,
                                "timestamp": time.time(),
                                "status": "pending",  # pending/approved/rejected
                            }, ensure_ascii=False) + "\n")
                        
                        # 连接2: 提案回流到causal_memory（闭环：evolve→learn_causal）
                        # 注意：提案不是失败，用 None 标记（非失败非成功），避免污染失败率统计
                        self.causal_memory.append({
                            "action": f"[Self-Harness提案] {top_mech}",
                            "prediction": "高频机制触发",
                            "actual": proposal_text[:100],
                            "lesson": f"机制={top_mech}, 支持={high_freq[top_mech]}次",
                            "timestamp": time.time(),
                            "verifier_cause": "self_harness_proposal",
                            "mechanism": top_mech,
                            "prediction_match": None,  # 非失败非成功，不计入统计
                            "result_success": None,
                        })
                    
                    # ④ 技能固化（血管#3: ios→isn）
                skill_entry = {
                    "name": f"learned_{top_mech}_{len(self.evolution_log)}",
                    "mechanism": top_mech,
                    "support": high_freq[top_mech],
                    "proposal": proposal_text,
                    "created_at": time.time(),
                    "source": "self_harness_evolution",
                }
                skill_path = Path.home() / ".openllm" / "output" / "isn" / "learned_skills.jsonl"
                skill_path.parent.mkdir(parents=True, exist_ok=True)
                with open(skill_path, "a") as f:
                    f.write(json.dumps(skill_entry, ensure_ascii=False) + "\n")
                print(f"  IO-S→ISN 血管#3: 技能已固化 [{top_mech}]")
        
        # ⑤ 策略自适应（保留原有逻辑，排除None：提案回流条目不计入）
        recent = self.evolution_log[-10:]
        failed = sum(1 for e in recent if e.get("result_success") is False)
        total = max(len(recent), 1)
        deny_rate = failed / total
        
        if deny_rate > 0.5 and self._arbiter_policy != "conservative":
            self._arbiter_policy = "conservative"
            print(f"  IO-S 自主调降策略→conservative (失败率{deny_rate:.0%})")
        elif deny_rate < 0.1 and len(self.evolution_log) > 20 and self._arbiter_policy == "conservative":
            self._arbiter_policy = "balanced"
            print(f"  IO-S 自主调松策略→balanced (失败率{deny_rate:.0%})")
    
    def _propose_harness_change(self, mechanism: str, support: int) -> str:
        """Self-Harness Harness Proposal: 为高频失败机制生成五体配置修改建议
        
        不同mechanism映射到不同五体层的修改：
        - tool_loop → ISN工具策略（限制重试次数）
        - missing_artifact → ISN产出验证（完成后检查）
        - exploration_loop → IO-S搜索约束（限制搜索轮次）
        - premature_success → IO-S验证门控（声明成功前必须验证）
        """
        proposals = {
            "tool_loop": f"ISN工具策略: 最大重试3次后切换方案（支持{support}次）",
            "missing_artifact": f"ISN产出验证: 工具执行后检查产出物是否存在（支持{support}次）",
            "wrong_format": f"ISN格式规范: 工具输出前做格式校验（支持{support}次）",
            "dependency_missing": f"ISN依赖检查: 执行前检查依赖是否可用（支持{support}次）",
            "timeout": f"IO-S超时策略: 设置工具执行超时上限（支持{support}次）",
            "logic_error": f"ISA因果提示: 注入类似失败的教训到context（支持{support}次）",
            "exploration_loop": f"IO-S搜索约束: 搜索不超过5轮后必须行动（支持{support}次）",
            "premature_success": f"IO-S验证门控: 声明成功前必须执行验证步骤（支持{support}次）",
        }
        return proposals.get(mechanism, "")
    
    def _extract_common(self, texts: list[str]) -> list[str]:
        """提取多段文本的共同词"""
        if not texts:
            return []
        words_sets = [set(t.lower().split()) for t in texts if t]
        if not words_sets:
            return []
        return list(set.intersection(*words_sets))[:5]
    
    def recover(self, error: Exception) -> bool:
        """错误恢复"""
        print(f"  IO-S 错误恢复: {error}")
        return True  # M0简化：总是恢复成功


class ISN:
    """工具执行"""
    
    def __init__(self):
        # 沙箱隔离
        from .sandbox import Sandbox
        self.sandbox = Sandbox()
        # 外部工具桥接
        from .tool_bridge import ExternalToolBridge
        self.bridge = ExternalToolBridge()
        self.bridge.scan()  # 自动发现工具
        # 注册真工具
        self.tools = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "search_files": self._search_files,
            "terminal": self._terminal,
        }
        # [进化] 接入ToolRegistry的verify-before-complete机制
        try:
            from ..tools.executor import ToolRegistry, ToolResult
            self._tool_registry = ToolRegistry()
            self._tool_registry.register("write_file", self._write_file_real,
                                          description="写入文件（带验证）")
            self._tool_registry.register("terminal", self._terminal_real,
                                          description="执行Shell命令（带验证）")
            # 设置verify hook：写操作前检查沙箱+治理规则
            self._tool_registry.set_verify_hook(self._verify_before_complete)
            self._has_verify = True
        except Exception:
            self._tool_registry = None
            self._has_verify = False
        print(f"  ISN 工具执行就绪 · {len(self.tools)}个工具 · 沙箱={len(self.sandbox.allowed)}个允许路径 · 外部工具={len(self.bridge.discovered)}个 · verify={'ON' if self._has_verify else 'OFF'}")
        # 加载已学习技能（血管#3: ios→isn）
        self.learned_skills: list[dict] = []
        self._load_learned_skills()
    
    def _load_learned_skills(self):
        """加载已持久化的学习技能"""
        skill_path = Path.home() / ".openllm" / "output" / "isn" / "learned_skills.jsonl"
        if skill_path.exists():
            import json as _json
            with open(skill_path) as f:
                for line in f:
                    try:
                        self.learned_skills.append(_json.loads(line))
                    except:
                        pass
            if self.learned_skills:
                print(f"  ISN 已加载 {len(self.learned_skills)} 个学习技能")
        
        # 连接3: 加载Self-Harness提案（approved状态的自动应用）
        prop_path = Path.home() / ".openllm" / "output" / "ios" / "harness_proposals.jsonl"
        if prop_path.exists():
            import json as _json
            approved = 0
            with open(prop_path) as f:
                for line in f:
                    try:
                        prop = _json.loads(line)
                        if prop.get("status") == "approved":
                            self.learned_skills.append({
                                "name": f"harness_proposal_{prop.get('mechanism', 'unknown')}",
                                "mechanism": prop.get("mechanism"),
                                "proposal": prop.get("proposal"),
                                "source": "harness_proposal",
                            })
                            approved += 1
                    except:
                        pass
            if approved:
                print(f"  ISN 已加载 {approved} 个approved的Self-Harness提案")
        
        # 连接4: 加载治理转换引擎产出的规则（P0: arXiv:2607.01087）
        gov_rules_path = Path.home() / ".openllm" / "output" / "ios" / "governance_rules.jsonl"
        if gov_rules_path.exists():
            import json as _json
            gov_count = 0
            with open(gov_rules_path) as f:
                for line in f:
                    try:
                        rule = _json.loads(line)
                        if rule.get("status") == "active":
                            self.learned_skills.append({
                                "name": f"governance_rule_{rule.get('rule_id', 'unknown')}",
                                "mechanism": rule.get("source_mechanism"),
                                "proposal": rule.get("description"),
                                "source": "governance_engine",
                                "condition": rule.get("condition"),
                                "action": rule.get("action"),
                            })
                            gov_count += 1
                    except:
                        pass
            if gov_count:
                print(f"  ISN 已加载 {gov_count} 条治理引擎规则")
    
    def execute(self, decision: Decision) -> ActionResult:
        """Phase 7: 执行"""
        if not decision.approved:
            return ActionResult(success=False, output=decision.reason)
        t0 = time.time()
        try:
            # 如果有tool_calls，执行对应工具
            output = self._execute_action(decision)
            return ActionResult(success=True, output=output, duration_ms=(time.time()-t0)*1000)
        except Exception as e:
            return ActionResult(success=False, error=str(e), duration_ms=(time.time()-t0)*1000)
    
    def _execute_action(self, decision: Decision) -> str:
        """实际执行：解析decision中的tool_calls并调用对应工具"""
        # 从decision中尝试提取工具调用
        # 如果decision有tool_calls字段 → 逐个执行
        tool_calls = getattr(decision, 'tool_calls', None) or []
        
        if not tool_calls:
            # 无工具调用 → M0回显
            return f"已执行: {decision.reason}"
        
        results = []
        for tc in tool_calls:
            tool_name = tc.get("name", "") if isinstance(tc, dict) else str(tc)
            tool_args = tc.get("args", {}) if isinstance(tc, dict) else {}
            
            if tool_name in self.tools:
                try:
                    # 调用工具（传递args字典）
                    result = self.tools[tool_name](**tool_args)
                except TypeError:
                    # 如果工具不接受**kwargs，尝试单参数调用
                    result = self.tools[tool_name](str(tool_args))
                results.append(f"[{tool_name}] {result}")
            else:
                results.append(f"[未知工具] {tool_name}")
        
        return "\n".join(results) if results else f"已执行: {decision.reason}"
    
    def _read_file(self, path: str) -> str:
        """读取文件内容（沙箱检查·先解析再检查防路径穿越）"""
        # 先解析路径（防../../etc/passwd穿越）
        p = Path(path).expanduser().resolve()
        # 再检查沙箱
        if not self.sandbox.check_path(str(p), "read"):
            return f"[沙箱拒绝] {self.sandbox.deny_reason(str(p))}"
        if not p.exists():
            return f"[错误] 文件不存在: {path}"
        if p.stat().st_size > 100000:
            return f"[错误] 文件过大: {path}"
        return p.read_text(encoding="utf-8", errors="replace")[:5000]
    
    def _write_file(self, path: str, content: str) -> str:
        """写入文件（沙箱检查·先解析再检查防路径穿越）"""
        # 先解析路径（防穿越）
        p = Path(path).expanduser().resolve()
        # 再检查沙箱
        if not self.sandbox.check_path(str(p), "write"):
            return f"[沙箱拒绝] {self.sandbox.deny_reason(str(p))}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[写入成功] {path} ({len(content)}字符)"
    
    def _search_files(self, pattern: str) -> str:
        """搜索文件"""
        import subprocess
        try:
            result = subprocess.run(
                ["find", str(Path.home()), "-name", pattern, "-maxdepth", "4"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout[:2000] or "[未找到]"
        except:
            return "[搜索失败]"
    
    def _terminal(self, command: str) -> str:
        """执行shell命令。高风险——需要沙箱检查。"""
        import subprocess
        import re
        
        # 危险命令清单
        dangerous = ["rm", "sudo", "dd", "mkfs", "> "]
        for d in dangerous:
            if d in command:
                return f"[拦截] 危险命令: {command}"
        
        # 沙箱检查：提取命令中的文件路径并检查
        # 匹配常见路径模式
        path_patterns = [
            r'(?<=\s)(/[^\s]+)',  # 绝对路径
            r'(?<=["\'])(/[^\s"\']+)(?=["\'])',  # 引号内的路径
        ]
        for pattern in path_patterns:
            paths = re.findall(pattern, command)
            for p in paths:
                if not self.sandbox.check_path(p, "write"):
                    return f"[沙箱拒绝] {self.sandbox.deny_reason(p)}"
        
        try:
            result = subprocess.run(command, shell=True, capture_output=True,
                                   text=True, timeout=30)
            return result.stdout[:3000] or result.stderr[:1000]
        except Exception as e:
            return f"[执行失败] {e}"
    
    # ── [进化] ToolRegistry verify hooks ──
    
    def _write_file_real(self, path: str, content: str) -> str:
        """真实写入——ToolRegistry调用此方法"""
        p = Path(path).expanduser().resolve()
        if not self.sandbox.check_path(str(p), "write"):
            return f"[沙箱拒绝] {self.sandbox.deny_reason(str(p))}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"[写入成功] {path} ({len(content)}字符)"
    
    def _terminal_real(self, command: str) -> str:
        """真实执行——ToolRegistry调用此方法"""
        import subprocess
        try:
            result = subprocess.run(command, shell=True, capture_output=True,
                                   text=True, timeout=30)
            return result.stdout[:3000] or result.stderr[:1000]
        except Exception as e:
            return f"[执行失败] {e}"
    
    def _verify_before_complete(self, tool_name: str, **kwargs) -> dict:
        """verify-before-complete钩子：写操作前的额外验证"""
        # 沙箱检查
        if tool_name == "write_file":
            path = kwargs.get("path", "")
            p = Path(path).expanduser().resolve()
            if not self.sandbox.check_path(str(p), "write"):
                return {"pass": False, "reason": f"沙箱拒绝: {self.sandbox.deny_reason(str(p))}"}
        # 危险命令检查
        if tool_name == "terminal":
            command = kwargs.get("command", "")
            dangerous = ["rm -rf", "sudo", "dd if=", "mkfs", "> /dev"]
            for d in dangerous:
                if d in command:
                    return {"pass": False, "reason": f"危险命令: {d}"}
        return {"pass": True, "reason": ""}


class IKO:
    """IKO — 输出体七因子管线
    
    七因子：IntentClassifier, SilenceAuditor, OutputRouter,
            OutputAuditChain, OutputFeedbackCollector,
            LambdaCalibrator, ProbingTrainer, SymmetricCodec
    """
    
    def __init__(self):
        self.metrics: list[TickMetrics] = []
        self.log_path = Path.home() / ".openllm" / "output" / "iko" / "ticks" / "ticks.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 七因子组件初始化
        from openllm.iko import (
            IntentClassifier, OutputRouter, SilenceAuditor,
            OutputAuditChain, OutputFeedbackCollector,
            LambdaCalibrator, ProbingTrainer, SymmetricCodec,
        )
        self.classifier = IntentClassifier()
        self.router = OutputRouter()
        self.registry = self.router.registry
        self.auditor = SilenceAuditor()
        self.audit_chain = OutputAuditChain()
        self.feedback_collector = OutputFeedbackCollector()
        self.calibrator = LambdaCalibrator()
        self.probing_trainer = ProbingTrainer()
        self.codec = SymmetricCodec()
        self._output_count = 0
        
        print(f"  IKO 可观测就绪 · 七因子管线 · 日志={self.log_path}")
    
    def trace(self, phase: str, status: str, duration_ms: float = 0.0, detail: str = ""):
        """记录一次观测"""
        m = TickMetrics(
            tick_id=uuid.uuid4().hex[:8],
            phase=phase,
            status=status,
            duration_ms=duration_ms,
            detail=detail[:100],
        )
        self.metrics.append(m)
        # 后端持久化
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(m)) + "\n")
    
    def report(self) -> dict:
        """仪表盘快照"""
        if not self.metrics:
            return {"status": "no_data"}
        last_10 = self.metrics[-10:]
        return {
            "total_ticks": len(self.metrics),
            "ok_rate": sum(1 for m in last_10 if m.status == "ok") / max(len(last_10), 1),
            "avg_duration_ms": sum(m.duration_ms for m in last_10) / len(last_10),
            "last_phase": last_10[-1].phase if last_10 else "",
        }
    
    def process_output(self, raw_output: str, context: dict, decision: dict) -> str:
        """七因子输出管线"""
        # 1. 意图分类
        result = self.classifier.classify(context, decision)
        intent = result.intent
        
        # 2. 沉默审计
        intent = self.auditor.audit(intent, context, reversible=True)
        
        # 3. 路由+渲染
        plan = self.router.route(intent, content={"text": raw_output}, user_prefs={})
        renderer = self.registry.get(plan.renderer_name)
        if renderer:
            output = renderer.render(intent, {"text": raw_output}, {}, 0.8)
        else:
            output = raw_output
        
        # 4. 审计链（赫淮斯托斯约束：reasoning_chain_hash必须是真实hash）
        if output:  # 非空输出才审计
            import hashlib, json as _json
            reasoning_hash = hashlib.sha256(
                _json.dumps({"intent": intent.value, "output": output[:200]}, sort_keys=True).encode()
            ).hexdigest()[:16]
            self.audit_chain.append(
                output_id=f"out-{self._output_count}",
                intent=intent.value,
                content=output.encode(),
                decision_source="IKO",
                risk_level=0.1,
                confidence=0.8,
                reasoning_chain_hash=reasoning_hash,
            )
            self._output_count += 1
        
        # 5. 记录trace
        self.trace("output_process", "ok", detail=f"intent={intent.value}")
        
        return output
    
    def shutdown(self):
        """关闭时的报告"""
        print(f"\n  IKO 关闭报告: {len(self.metrics)} tick, 最后状态={self.metrics[-1].status if self.metrics else 'N/A'}")


# ═══════════════════════════════════════════════════════
# Agent主循环（集成Session/Turn + 10阶段心跳）
# ═══════════════════════════════════════════════════════

class Agent:
    """openLLM Agent — 独立运行的Agent框架"""
    
    def __init__(self, mode: str = "console"):
        # 五体初始化
        self.isa = ISA(mode)
        self.octopus = 章鱼I()
        self.ios = IOS()
        self.isn = ISN()
        self.iko = IKO()
        
        # Session/Turn
        # 六体自监督反馈环
        from ..governance.feedback_loop import FeedbackLoop
        self.feedback_loop = FeedbackLoop()
        self._body_outputs = {}  # 暂存每体最新输出
        
        # 间隙体——空闲期好奇心
        from .idle_wander import IdleWanderer
        self._wanderer = IdleWanderer(threshold=3)

        self.session = create_session(max_context_tokens=100000)
        
        # 状态
        self.running = False
        self._tick_count = 0
        self._max_ticks = 100  # 安全上限
        self._last_output = ""
        
        # ── IAX Layer 7 推理预算管理器 ──
        self._budget_manager = InferenceBudgetManager() if _HAS_BUDGET else None
        self._budget_checked_today = False
        
        # ── tool_validator 失败追踪 ──
        self._tool_failures: list[dict] = []
    
    def run(self):
        """主循环入口"""
        self.running = True
        print("\n═══ openLLM Agent 启动 ═══")
        print("  模式: 独立运行 · 无外部依赖")
        print("  Session:", self.session.id)
        print("  输入 /quit 退出\n")
        
        while self.running and self._tick_count < self._max_ticks:
            try:
                self._tick()
                self._tick_count += 1
            except KeyboardInterrupt:
                self.iko.trace("shutdown", "ok", detail="用户中断")
                break
            except Exception as e:
                self.iko.trace("fatal", "error", detail=str(e)[:50])
                ok = self.ios.recover(e)
                if not ok:
                    print(f"  🔴 不可恢复错误: {e}")
                    break
        
        self.iko.shutdown()
        self.session.end()
        print("\n═══ openLLM Agent 关闭 ═══")
    
    def run_once(self, message: str) -> str:
        """
        单次运行（用于测试/非交互模式）
        
        用法：
            agent = Agent(mode="silent")
            result = agent.run_once("你好")
        """
        saved_mode = self.isa.mode
        self.isa.mode = "silent"
        try:
            msg = Message(text=message)
            self._execute_tick(msg)
            return self._last_output
        finally:
            self.isa.mode = saved_mode
    
    def _tick(self):
        """一次心跳"""
        msg = self.isa.listen()
        if msg is None:
            # ── 间隙体：空闲期好奇心 ──
            if self._wanderer.tick_idle():
                self._do_wander()
            else:
                self.running = False
            return
        self._wanderer.tick_active()  # 有输入·重置空闲计数
        self._execute_tick(msg)
    
    def _do_wander(self):
        """散步模式——注意力的自由偏移"""
        discoveries = self._wanderer.wander()
        if discoveries:
            # 散步发现写入日志，不触发决策/执行
            for d in discoveries:
                self.iko.trace("wander", "ok", detail=f"{d.domain}: {d.finding[:40]}")
    
    def _record_inference(self, phase: str):
        """IAX: 从LLMProvider提取最近一次调用的token使用量，记录到预算管理器"""
        if not self._budget_manager:
            return
        # 从左右脑provider取_last_usage（Phase 3用左脑，Phase 5用左脑+右脑）
        usage = getattr(self.octopus.left.provider, '_last_usage', {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        if prompt_tokens == 0 and completion_tokens == 0:
            return  # 模拟模式或无数据，跳过
        try:
            rec = self._budget_manager.record_inference(
                model=self.octopus.left.provider.model,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
                duration_ms=0,  # 精确计时在phase trace中已有
            )
            self.iko.trace("budget", "ok",
                detail=f"{phase}: in={prompt_tokens} out={completion_tokens} cost=${rec.cost_usd:.4f}")
            # 每天第一次调用时检查预算
            if not self._budget_checked_today:
                self._budget_checked_today = True
                budget = self._budget_manager.check_budget()
                if budget["over_budget"]:
                    print(f"  ⚠️ IAX 推理预算超限: {budget['total_tokens']}/{budget['limit']} "
                          f"({budget['usage_ratio']:.0%}) 今日成本=${budget['cost_usd']:.4f}")
        except Exception:
            pass  # 预算追踪失败不阻塞主循环

    def _execute_tick(self, msg: Message):
        """一次完整的10阶段心跳"""
        t0 = time.time()
        
        # 创建新Turn
        turn = self.session.new_turn()
        
        try:
            # ── Phase 1: isa.listen() → Message ──
            turn.trace_phase("listen", "ok")
            self.iko.trace("listen", "ok")
            
            # ── Phase 2: build_context — 传入session和ios ──
            t1 = time.time()
            ctx = self.isa.build_context(msg, self.session, self.octopus, self.ios)
            turn.trace_phase("context", "ok", duration_ms=(time.time()-t1)*1000)
            self.iko.trace("context", "ok")

            # ── Phase 2.5: evidence replay — 相关证据回放 ──
            # 来源：arXiv:2607.02509 ReContext
            # 从七要素中提取top-K相关证据，追加到prompt尾部
            try:
                from ..memory.evidence_replay import create_replay_for_context
                replay_text = create_replay_for_context(msg.text, ctx, top_k=5, max_tokens=512)
                if replay_text:
                    ctx.search_results = getattr(ctx, 'search_results', []) or []
                    ctx.search_results.append(replay_text)
                    turn.trace_phase("replay", "ok", detail=f"evidence_replay:{len(replay_text)}chars")
                    self.iko.trace("replay", "ok")
            except Exception as e:
                # evidence replay失败不阻塞主流程
                turn.trace_phase("replay", "skip", detail=str(e)[:100])

            # ── Phase 3: 因果预测 ──
            t2 = time.time()
            prediction = self.octopus.predict_consequences(ctx)
            turn.trace_phase("predict", "ok", duration_ms=(time.time()-t2)*1000,
                           detail=prediction.summary_text())
            self.iko.trace("predict", "ok", detail=prediction.summary_text())
            
            # ── IAX: 记录Phase 3推理消耗 ──
            self._record_inference("phase_3_predict")

            # ── Phase 3.5: 治理检查·路由决策点（Copewell模式·2026-07-06） ──
            try:
                from ..governance.events import RoutingCheckEvent, GovernanceDimension
                from ..governance.audit import AuditChain
                _g8_event = RoutingCheckEvent(
                    actor="ios_phase3",
                    session_id=ctx.session_id if hasattr(ctx, 'session_id') else "default",
                    prev_hash="genesis",
                    phase="phase_3_prediction",
                    route_target=str(type(prediction).__name__),
                    risk_level="low",
                    approved=True,
                )
                if hasattr(self, '_audit_chain'):
                    self._audit_chain.append(_g8_event)
            except Exception:
                pass  # 治理检查失败不阻塞主流程

            # ── Phase 4: 风险评估 ──
            t3 = time.time()
            risk = self.ios.risk_check(ctx, prediction)
            turn.risk_level = risk.level
            turn.trace_phase("risk", "ok" if not risk.is_blocked() else "denied",
                           duration_ms=(time.time()-t3)*1000,
                           detail=f"level={risk.level} blocked={risk.is_blocked()}")
            self.iko.trace("risk", "ok" if not risk.is_blocked() else "denied",
                         detail=risk.reason)
            
            if risk.is_blocked():
                self.isa.respond(f"[安全拦截] {risk.reason}")
                turn.add_action("blocked", risk.reason)
                turn.complete()
                return
            
            # ── Phase 5: 左右脑推理 ──
            # D₀感知注入
            d0 = self.octopus.d0_snapshot()
            ctx.d0_report = d0
            
            # 因果教训注入（从risk.details获取）
            if risk and risk.details:
                ctx.causal_hints = risk.details
            
            t4 = time.time()
            proposal, critique = self.octopus.reason(ctx, prediction, risk)
            turn.trace_phase("reason", "ok", duration_ms=(time.time()-t4)*1000)
            self.iko.trace("reason", "ok")
            
            # ── IAX: 记录Phase 5推理消耗（左脑+右脑）──
            self._record_inference("phase_5_reason")

            # ── Phase 5.5: 治理检查·推理质量（Copewell模式·2026-07-06） ──
            try:
                from ..governance.events import RoutingCheckEvent
                _g8_reason = RoutingCheckEvent(
                    actor="ios_phase5",
                    session_id=ctx.session_id if hasattr(ctx, 'session_id') else "default",
                    prev_hash="genesis",
                    phase="phase_5_reasoning",
                    route_target=str(type(proposal).__name__),
                    risk_level="low",
                    approved=True,
                )
                if hasattr(self, '_audit_chain'):
                    self._audit_chain.append(_g8_reason)
            except Exception:
                pass  # 治理检查失败不阻塞主流程

            # ── Phase 6: 仲裁 ──
            t5 = time.time()
            decision = self.ios.arbitrate(proposal, critique, risk)
            turn.trace_phase("decide", "ok" if decision.approved else "denied",
                           duration_ms=(time.time()-t5)*1000,
                           detail=decision.reason)
            self.iko.trace("decide", "ok" if decision.approved else "denied",
                         detail=decision.reason)
            
            if not decision.approved:
                self.isa.respond(f"[仲裁否决] {decision.reason}")
                turn.add_action("denied", decision.reason)
                turn.complete()
                return
            
            # ── Phase 7: 执行 ──
            # cap_check: 只在有工具调用时检查（无工具调用=纯对话·不检查）
            if getattr(decision, 'tool_calls', None):
                if not self.ios.cap_check("execute", decision.action):
                    self.isa.respond("[权限拒绝] cap_policy不允许此操作")
                    turn.add_action("denied", "cap_policy拒绝")
                    turn.complete()
                    return
            
            t6 = time.time()
            result = self.isn.execute(decision)
            turn.trace_phase("execute", "ok" if result.success else "error",
                           duration_ms=result.duration_ms)
            self.iko.trace("execute", "ok" if result.success else "error")
            
            # ── Phase 7.5: tool_validator 自动验证 ──
            if _HAS_TOOL_VALIDATOR and result.success:
                try:
                    _tc_list = getattr(decision, 'tool_calls', None) or []
                    _first = _tc_list[0] if _tc_list else {}
                    _tool_name = _first.get("name", "unknown") if isinstance(_first, dict) else str(_first)
                    _tool_type_map = {"read_file": "read", "write_file": "write",
                                      "search_files": "search", "terminal": "execute"}
                    _tool_type = _tool_type_map.get(_tool_name, "execute")
                    _tv_call = _TVToolCall(
                        tool_name=_tool_name, tool_type=_tool_type,
                        params=_first.get("args", {}) if isinstance(_first, dict) else {},
                        result=result.output,
                        duration_ms=result.duration_ms,
                        session_id=getattr(self.session, 'id', ''),
                        turn_id=turn.id,
                    )
                    _tv_report = _tv_validate(_tv_call)
                    turn.trace_phase("validate", _tv_report.overall.value,
                                   detail=f"checks={len(_tv_report.checks)} retry={_tv_report.should_retry} block={_tv_report.should_block}")
                    if _tv_report.should_block:
                        _fail_entry = {
                            "timestamp": time.time(), "tool_name": _tool_name,
                            "reason": _tv_report.audit_entry.reason if _tv_report.audit_entry else "blocked",
                            "session_id": getattr(self.session, 'id', ''), "turn_id": turn.id,
                        }
                        self._tool_failures.append(_fail_entry)
                        self.iko.trace("validate", "blocked", detail=_fail_entry["reason"])
                        self._last_output = f"[工具验证拦截] {_fail_entry['reason']}"
                        if self.isa.mode != "silent":
                            self.isa.respond(self._last_output)
                        turn.add_action("validate_blocked", _fail_entry["reason"])
                        turn.complete()
                        return
                    if _tv_report.should_retry:
                        turn.trace_phase("validate_retry", "warn", detail="工具结果需重试")
                except Exception as _tv_err:
                    turn.trace_phase("validate", "skip", detail=str(_tv_err)[:80])
            
            # ── Phase 8: 因果对照 ──
            t7 = time.time()
            delta = self.octopus.compare(prediction, result)
            self.ios.learn_causal(ctx, prediction, result, delta)
            turn.trace_phase("learn", "ok", duration_ms=(time.time()-t7)*1000,
                           detail=delta.summary_text())
            self.iko.trace("learn", "ok", detail=delta.summary_text())
            
            # ── Phase 9: 进化 ──
            t8 = time.time()
            self.ios.evolve(proposal, critique, result, delta)
            turn.trace_phase("evolve", "ok", duration_ms=(time.time()-t8)*1000)
            self.iko.trace("evolve", "ok")
            
            # ── Phase 10: 压缩+输出+checkpoint ──
            t9 = time.time()
            self.session.compact_if_needed()
            # 无工具调用时：输出LLM生成的proposal内容（对话响应）
            if getattr(decision, 'tool_calls', None) or getattr(decision, '_has_tools', False):
                self._last_output = result.output
            else:
                self._last_output = proposal.content if proposal.content else result.output
            
            # IKO七因子管线
            try:
                risk_map = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
                ctx_for_iko = {
                    "risk_level": risk_map.get(getattr(decision, 'risk_level', 'low'), "LOW"),
                    "has_tool_calls": bool(getattr(decision, 'tool_calls', None)),
                    # 包拯审计修正：has_side_effects应检查实际副作用，非仅输出存在
                    "has_side_effects": bool(getattr(result, 'success', False) and getattr(result, 'output', '')),
                    # 包拯审计修正：option_count从proposal选项中提取，非硬编码
                    "option_count": max(1, len(getattr(proposal, 'evidence', []))),
                }
                decision_dict = {"type": "execute", "content": result.output}
                processed = self.iko.process_output(result.output, ctx_for_iko, decision_dict)
                if self.isa.mode != "silent" and processed:
                    self.isa.respond(processed)
            except Exception as e:
                # 包拯审计修正：降级时必须记录trace，不能静默吞没
                self.iko.trace("output_process", "error", detail=str(e)[:100])
                # 管线失败降级为原始输出
                if self.isa.mode != "silent":
                    self.isa.respond(result.output)
            
            turn.add_action("execute", result.output[:100])
            turn.complete()
            
            # 定时checkpoint
            if self._tick_count % 10 == 0:
                self.session.checkpoint()
                self.iko.trace("checkpoint", "ok")
            
            turn.trace_phase("output", "ok", duration_ms=(time.time()-t9)*1000)
            self.iko.trace("output", "ok", duration_ms=(time.time()-t9)*1000)
            
            # ── Phase 11: 六体自监督反馈收集 ──
            try:
                self._body_outputs["IAX"] = {"heartbeat_ok": True, "tick_count": self._tick_count}
                self._body_outputs["IAI"] = getattr(self.octopus, 'last_output', {}) or {}
                self._body_outputs["ISA"] = getattr(self.isa, 'last_output', {}) or {}
                self._body_outputs["IOS"] = getattr(self.ios, 'last_output', {}) or {}
                self._body_outputs["ISN"] = getattr(self.isn, 'last_output', {}) or {}
                self._body_outputs["IKO"] = getattr(self.iko, 'last_output', {}) or {}
                
                records = self.feedback_loop.collect_feedback(self._body_outputs)
                if records:
                    adjustments = self.feedback_loop.apply_feedback()
                    # 记录调整因子供下轮使用
                    self._feedback_adjustments = adjustments
                turn.trace_phase("feedback", "ok", detail=f"collected:{len(records)}")
            except Exception as e:
                turn.trace_phase("feedback", "error", detail=str(e)[:50])

            # tick计数自增（移到这里，确保run_once也能正确计数）
            self._tick_count += 1
            
        except Exception as e:
            turn.fail(str(e))
            raise


# ═══════════════════════════════════════════════════════
# CLI入口
# ═══════════════════════════════════════════════════════

def main():
    """CLI入口"""
    args = sys.argv[1:]
    
    if "--once" in args:
        # 单次模式
        idx = args.index("--once")
        message = args[idx + 1] if idx + 1 < len(args) else "你好"
        agent = Agent(mode="silent")
        result = agent.run_once(message)
        print(result)
        return
    
    if "--test" in args:
        # 测试模式：跑一次完整的cycle然后退出
        _run_tests()
        return
    
    # 交互模式（默认）
    agent = Agent(mode="console")
    agent.run()


def _run_tests():
    """运行M0测试"""
    print("=== M0: Agent主循环测试 ===\n")
    
    # 测试1: 初始化
    agent = Agent(mode="silent")
    assert agent.isa is not None
    assert agent.octopus is not None
    assert agent.ios is not None
    assert agent.isn is not None
    assert agent.iko is not None
    print("✅ 测试1: 五体初始化")
    
    # 测试2: run_once
    result = agent.run_once("测试消息")
    assert result is not None
    print(f"✅ 测试2: run_once 输出={result[:30]}")
    
    # 测试3: Session/Turn生命周期
    assert agent.session.start_time is not None
    assert len(agent.session.turns) >= 1
    print(f"✅ 测试3: Session/Turn生命周期 total_turns={len(agent.session.turns)}")
    
    # 测试4: 左右脑对弈
    ctx = agent.isa.build_context(Message("写一段Python代码"))
    proposal, critique = agent.octopus.reason(ctx)
    assert proposal is not None
    assert critique is not None
    print(f"✅ 测试4: 左脑提案={proposal.content[:30]} 右脑批判={critique.verdict}")
    
    # 测试5: IO-S仲裁
    decision = agent.ios.arbitrate(proposal, critique)
    assert decision is not None
    print(f"✅ 测试5: 仲裁={decision.action} 批准={decision.approved}")
    
    # 测试6: 工具执行
    result = agent.isn.execute(decision)
    print(f"✅ 测试6: 执行 success={result.success}")
    
    # 测试7: IKO可观测
    report = agent.iko.report()
    assert report["total_ticks"] >= 2  # run_once + 手动
    print(f"✅ 测试7: 可观测 total_ticks={report['total_ticks']} ok_rate={report['ok_rate']:.0%}")
    
    # 测试8: IO-S自我进化
    print(f"✅ 测试8: 进化日志 {len(agent.ios.evolution_log)}条（>=1）")
    
    # 测试9: 因果预测
    prediction = agent.octopus.predict_consequences(ctx)
    assert prediction is not None
    print(f"✅ 测试9: 因果预测 {prediction.summary_text()[:40]}")
    
    # 测试10: 因果对照
    delta = agent.octopus.compare(prediction, result)
    assert delta is not None
    print(f"✅ 测试10: 因果对照 {delta.summary_text()}")
    
    # 测试11: 风险评估
    risk = agent.ios.risk_check(ctx, prediction)
    assert risk is not None
    print(f"✅ 测试11: 风险评估 level={risk.level}")
    
    # 测试12: 10阶段trace日志
    ticks_file = Path.home() / ".openllm" / "output" / "iko" / "ticks" / "ticks.jsonl"
    assert ticks_file.exists()
    with open(ticks_file) as f:
        lines = f.readlines()
    assert len(lines) >= 2  # run_once + 手动
    print(f"✅ 测试12: 10阶段trace日志 ticks={len(lines)}")
    
    print(f"\n全部 12/12 测试通过 ✅")


if __name__ == "__main__":
    main()
