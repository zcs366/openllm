"""
iko_impl.py — IKO输出体

六体架构第六体：输出体 = "怎么说"
用户直接感知的唯一接口。

七因子管线（底层，已实现）：
  IntentClassifier → SilenceAuditor → OutputRouter → OutputAuditChain
  → OutputFeedbackCollector → LambdaCalibrator → SymmetricCodec

本文件（上层）新增三能力：
  1. SceneAdapter — 场景适配（技术/闲聊/代码/决策→不同输出策略）
  2. TokenBudget — token预算感知（知道输出空间还剩多少）
  3. QualityScorer — 输出质量自评（每轮打分，积累数据）
"""
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

from .models import TickMetrics


# ═══════════════════════════════════════════════════════
# 场景枚举 + 策略
# ═══════════════════════════════════════════════════════

class OutputScene(Enum):
    """输出场景——决定输出的密度、格式、语气。"""
    TECHNICAL = auto()   # 技术对话：精确、结构化、代码块
    CASUAL = auto()      # 闲聊：轻松、简短、口语化
    CODE = auto()        # 代码输出：纯代码+注释
    DECISION = auto()    # 决策报告：表格+结论先行
    ERROR = auto()       # 错误报告：原因+建议+堆栈摘要
    SILENT = auto()      # 静默：不输出


@dataclass
class SceneStrategy:
    """场景输出策略——SceneAdapter的输出。"""
    scene: OutputScene
    max_tokens: int          # 该场景的token上限
    format_hint: str         # "markdown" | "plain" | "code" | "table"
    tone: str                # "formal" | "casual" | "precise"
    confidence_display: str  # "always" | "threshold" | "never"
    compress: bool           # 是否压缩推理链
    reason: str = ""         # 分类理由


# 场景策略表——每种场景的输出参数
_SCENE_STRATEGIES = {
    OutputScene.TECHNICAL: SceneStrategy(
        scene=OutputScene.TECHNICAL, max_tokens=2000,
        format_hint="markdown", tone="precise",
        confidence_display="threshold", compress=False,
    ),
    OutputScene.CASUAL: SceneStrategy(
        scene=OutputScene.CASUAL, max_tokens=500,
        format_hint="plain", tone="casual",
        confidence_display="never", compress=True,
    ),
    OutputScene.CODE: SceneStrategy(
        scene=OutputScene.CODE, max_tokens=3000,
        format_hint="code", tone="precise",
        confidence_display="never", compress=False,
    ),
    OutputScene.DECISION: SceneStrategy(
        scene=OutputScene.DECISION, max_tokens=1500,
        format_hint="table", tone="formal",
        confidence_display="always", compress=False,
    ),
    OutputScene.ERROR: SceneStrategy(
        scene=OutputScene.ERROR, max_tokens=1000,
        format_hint="markdown", tone="precise",
        confidence_display="always", compress=False,
    ),
    OutputScene.SILENT: SceneStrategy(
        scene=OutputScene.SILENT, max_tokens=0,
        format_hint="plain", tone="casual",
        confidence_display="never", compress=True,
    ),
}


class SceneAdapter:
    """场景适配器——根据上下文判定输出场景，返回输出策略。"""

    def classify(self, context: dict, raw_output: str) -> SceneStrategy:
        """判定当前输出场景。"""
        risk = context.get("risk_level", "LOW")
        has_tools = context.get("has_tool_calls", False)
        has_side = context.get("has_side_effects", False)
        output_len = len(raw_output)

        # 1. 高风险 → 错误报告
        if risk == "HIGH":
            return _scene(OutputScene.ERROR, "risk=HIGH")

        # 2. 纯代码输出（以```开头或全是代码字符）
        stripped = raw_output.strip()
        if stripped.startswith("```") or _is_code_dominant(stripped):
            return _scene(OutputScene.CODE, "code_dominant")

        # 3. 有工具调用或副作用 → 技术对话
        if has_tools or has_side:
            return _scene(OutputScene.TECHNICAL, "has_tools/side_effects")

        # 4. 决策（多选项或中等风险）
        opt_count = context.get("option_count", 1)
        if opt_count >= 2 or risk == "MEDIUM":
            return _scene(OutputScene.DECISION, f"options={opt_count}")

        # 5. 短输出 → 闲聊
        if output_len < 200:
            return _scene(OutputScene.CASUAL, f"short({output_len})")

        # 6. 默认 → 技术对话
        return _scene(OutputScene.TECHNICAL, "default")


def _scene(s: OutputScene, reason: str) -> SceneStrategy:
    strat = _SCENE_STRATEGIES[s]
    return SceneStrategy(
        scene=strat.scene, max_tokens=strat.max_tokens,
        format_hint=strat.format_hint, tone=strat.tone,
        confidence_display=strat.confidence_display,
        compress=strat.compress, reason=reason,
    )


def _is_code_dominant(text: str) -> bool:
    """判断文本是否以代码为主。"""
    if not text:
        return False
    code_chars = sum(1 for c in text if c in '{}[]();:=<>\\|/!@#$%^&*')
    return code_chars / max(len(text), 1) > 0.15


# ═══════════════════════════════════════════════════════
# Token预算感知
# ═══════════════════════════════════════════════════════

@dataclass
class TokenBudget:
    """token预算追踪——知道输出空间还剩多少。"""
    context_window: int = 8192     # 上下文窗口总量
    reserved_input: int = 4096     # 预留给输入的token
    used_tokens: int = 0           # 本轮已用token

    @property
    def remaining(self) -> int:
        """可用输出token数。"""
        return max(0, self.context_window - self.reserved_input - self.used_tokens)

    @property
    def usage_ratio(self) -> float:
        """预算使用率。"""
        total = self.context_window - self.reserved_input
        return self.used_tokens / max(total, 1)

    def can_afford(self, tokens: int) -> bool:
        """是否还能承担指定token数的输出。"""
        return self.remaining >= tokens

    def record(self, tokens: int):
        """记录消耗。"""
        self.used_tokens += tokens

    def suggest_max(self) -> int:
        """建议的最大输出长度（基于剩余预算）。"""
        r = self.remaining
        if r > 1500:
            return 2000
        elif r > 500:
            return min(r, 1000)
        elif r > 100:
            return min(r, 300)
        else:
            return 50  # 极端压缩


# ═══════════════════════════════════════════════════════
# 输出质量自评
# ═══════════════════════════════════════════════════════

@dataclass
class QualityScore:
    """单轮输出质量分。"""
    score: float          # 0-1
    dimensions: dict      # 各维度得分
    scene: str            # 输出场景
    timestamp: float = field(default_factory=time.time)

    def __str__(self):
        return f"Quality({self.score:.2f} scene={self.scene})"


class QualityScorer:
    """输出质量自评——每轮输出后打分。"""

    def __init__(self):
        self._history: list[QualityScore] = []

    def score(self, raw_output: str, scene: OutputScene,
              context: dict, decision: dict) -> QualityScore:
        """对一轮输出打分。五维度："""
        dims = {}

        # 1. 完整度：输出是否完整（非截断、非空）
        dims["completeness"] = 1.0 if len(raw_output) > 10 and not raw_output.endswith("...") else 0.5

        # 2. 相关度：是否与意图匹配（简化：不为空且有内容）
        dims["relevance"] = min(1.0, len(raw_output) / 50)

        # 3. 简洁度：是否冗余（简化：重复行越少越好）
        lines = raw_output.split("\n")
        unique_lines = set(l.strip() for l in lines if l.strip())
        dims["conciseness"] = min(1.0, len(unique_lines) / max(len(lines), 1))

        # 4. 可读度：是否有结构（标题/列表/代码块）
        has_structure = any(raw_output.startswith(p) for p in ["#", "-", "```", "|"])
        dims["readability"] = 0.9 if has_structure else 0.7

        # 5. 安全度：是否被审计链标记
        dims["safety"] = 1.0  # 默认安全，审计链会在外部标记

        # 加权综合
        weights = {"completeness": 0.3, "relevance": 0.3, "conciseness": 0.2,
                   "readability": 0.1, "safety": 0.1}
        total = sum(dims[k] * weights[k] for k in weights)

        qs = QualityScore(
            score=round(total, 3),
            dimensions=dims,
            scene=scene.name,
        )
        self._history.append(qs)
        return qs

    @property
    def avg_score(self) -> float:
        """近期平均质量分。"""
        recent = self._history[-20:]
        if not recent:
            return 0.0
        return sum(q.score for q in recent) / len(recent)

    def trend(self) -> str:
        """质量趋势：↑改善 / →稳定 / ↓退化。"""
        if len(self._history) < 4:
            return "→"
        old = sum(q.score for q in self._history[-4:-2]) / 2
        new = sum(q.score for q in self._history[-2:]) / 2
        if new > old + 0.05:
            return "↑"
        elif new < old - 0.05:
            return "↓"
        return "→"


# ═══════════════════════════════════════════════════════
# IKO主体
# ═══════════════════════════════════════════════════════

class IKO:
    """IKO — 输出体
    
    三能力：SceneAdapter(场景适配) + TokenBudget(预算感知) + QualityScorer(质量自评)
    七因子管线（底层）：IntentClassifier → SilenceAuditor → OutputRouter → ...
    """
    
    def __init__(self):
        # ── 三能力 ──
        self.scene_adapter = SceneAdapter()
        self.token_budget = TokenBudget()
        self.quality_scorer = QualityScorer()
        
        # ── 七因子管线（底层） ──
        try:
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
            self._has_pipeline = True
        except ImportError:
            self._has_pipeline = False
        
        # ── 可观测 ──
        self.metrics: list[TickMetrics] = []
        self.log_path = Path.home() / ".openllm" / "output" / "iko" / "ticks" / "ticks.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_count = 0
        
        print(f"  IKO 可观测就绪 · 三能力+七因子 · 日志={self.log_path}")

    def trace(self, phase: str, status: str, duration_ms: float = 0.0, detail: str = ""):
        """记录一次观测"""
        m = TickMetrics(
            tick_id=uuid.uuid4().hex[:8],
            phase=phase, status=status,
            duration_ms=duration_ms, detail=detail[:100],
        )
        self.metrics.append(m)
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
            # 三能力指标
            "avg_quality": round(self.quality_scorer.avg_score, 3),
            "quality_trend": self.quality_scorer.trend(),
            "token_remaining": self.token_budget.remaining,
        }

    def process_output(self, raw_output: str, context: dict, decision: dict) -> str:
        """七因子输出管线（增强版：场景适配+预算感知+质量自评）"""
        # ── 0. 场景适配 ──
        scene_strategy = self.scene_adapter.classify(context, raw_output)
        
        # ── 0.5. token预算检查 ──
        suggested_max = self.token_budget.suggest_max()
        effective_max = min(scene_strategy.max_tokens, suggested_max)
        
        # ── 1. 意图分类（七因子） ──
        if self._has_pipeline:
            result = self.classifier.classify(context, decision)
            intent = result.intent
            intent = self.auditor.audit(intent, context, reversible=True)
            plan = self.router.route(intent, content={"text": raw_output}, user_prefs={})
            renderer = self.registry.get(plan.renderer_name)
            output = renderer.render(intent, {"text": raw_output}, {}, 0.8) if renderer else raw_output
        else:
            output = raw_output

        # ── 2. 场景策略应用 ──
        if scene_strategy.compress:
            output = _compress_output(output, effective_max)
        if len(output) > effective_max * 4:  # 粗估token：1 token ≈ 4 chars
            output = output[:effective_max * 4] + "\n...[truncated]"

        # ── 3. 审计链 ──
        if output and self._has_pipeline:
            import hashlib
            reasoning_hash = hashlib.sha256(
                json.dumps({"scene": scene_strategy.scene.name, "output": output[:200]},
                           sort_keys=True).encode()
            ).hexdigest()[:16]
            self.audit_chain.append(
                output_id=f"out-{self._output_count}",
                intent=scene_strategy.scene.name,
                content=output.encode(),
                decision_source="IKO",
                risk_level=0.1, confidence=0.8,
                reasoning_chain_hash=reasoning_hash,
            )
            self._output_count += 1

        # ── 4. token预算记录 ──
        estimated_tokens = len(output) // 4
        self.token_budget.record(estimated_tokens)

        # ── 5. 质量自评 ──
        qs = self.quality_scorer.score(output, scene_strategy.scene, context, decision)
        
        # ── 6. trace ──
        self.trace("output_process", "ok",
                   detail=f"scene={scene_strategy.scene.name} q={qs.score:.2f} tok={estimated_tokens}")
        
        return output

    def reset_budget(self, context_window: int = 8192, reserved_input: int = 4096):
        """每轮对话开始时重置预算。"""
        self.token_budget = TokenBudget(
            context_window=context_window, reserved_input=reserved_input)

    def shutdown(self):
        """关闭时的报告"""
        avg = self.quality_scorer.avg_score
        trend = self.quality_scorer.trend()
        print(f"\n  IKO 关闭报告: {len(self.metrics)} tick · "
              f"质量={avg:.2f} {trend} · 最后状态={self.metrics[-1].status if self.metrics else 'N/A'}")


def _compress_output(text: str, max_chars: int) -> str:
    """轻量压缩：去冗余行、缩短列表。"""
    lines = text.split("\n")
    seen = set()
    compressed = []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            compressed.append(line)
    result = "\n".join(compressed)
    if len(result) > max_chars * 4:
        result = result[:max_chars * 4] + "\n...[compressed]"
    return result
