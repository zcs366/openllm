"""
IKO — 输出体 (Output Organ)
============================

openLLM 六体架构（IAX-心跳、IAI-感知、ISA-记忆、IOS-决策、ISN-执行、IKO-输出）
中唯一面向用户的器官。

IKO 的职责是将系统内部的决策、推理和执行结果转化为用户可理解的输出。
它是六体与人类之间的最后一道桥梁，负责：

1. **意图分类** — 判断输出应该以何种形式呈现（信息/确认/执行/决策/错误/静默）
2. **路由渲染** — 根据意图选择合适的渲染器，生成结构化输出
3. **静默审计** — 防止不当沉默，确保关键信息不被吞没
4. **推理压缩** — 对称编解码，压缩推理链为轻量摘要
5. **输出审计** — 为每次输出建立不可变的链式哈希审计记录
6. **反馈收集** — 从用户行为中推断偏好，动态调整输出密度
7. **追问训练** — 根据用户成熟度生成追问建议，提升交互质量
8. **λ 自校准** — 根据反馈信号动态校准输出质量置信度

七因子公式::

    OutputQuality = Intent × Adaptive × Trust × SharedState
                    × SelfCalibrate × Muscle × Refuse

七神约束（贯穿所有模块）：
- 赫淮斯托斯：对称性——compress/decompress 互为逆操作
- 赫尔墨斯：开口性——不可逆时必须主动开口，不能静默返回残缺数据
- 雅典娜：可解释性——每次分类/路由附带 reason 字段
- 阿瑞斯：防御性——高风险决策必须确认，高置信度错误必须惩罚
- 德墨忒尔：持久化——λ 状态和反馈历史可持久化到 JSON
- 赫拉：链式审计——每条输出和消息包含前一条的 hash
- 阿波罗：透明性——输出审计链可被外部工具独立验证

模块组成::

    ┌─────────────┐     ┌──────────────┐
    │IntentClassif│────▶│ SilenceAudito│
    └──────┬──────┘     └──────────────┘
           │
    ┌──────▼──────┐     ┌──────────────┐
    │ OutputRouter│────▶│ SymmetricCede│
    └──────┬──────┘     └──────────────┘
           │
    ┌──────▼──────┐     ┌──────────────┐
    │ OutputAudit │     │ LambdaCalibra│
    └──────┬──────┘     └──────┬───────┘
           │                    │
    ┌──────▼──────┐     ┌──────▼───────┐
    │FeedbackColle│     │ProbingTrainer│
    └─────────────┘     └──────────────┘

用法::

    from openllm.iko import (
        IntentClassifier, OutputIntent, ClassificationResult,
        SilenceAuditor,
        OutputRouter, RenderPlan,
        SymmetricCodec, CompressedReasoning, FullReasoningChain,
        OutputAuditChain, OutputAuditEntry,
        FeedbackSignal, OutputFeedbackCollector,
        ProbingTrainer,
        LambdaState, LambdaCalibrator,
    )

    # 1. 意图分类
    classifier = IntentClassifier()
    result = classifier.classify(
        context={"risk_level": "LOW", "has_tool_calls": False,
                 "has_side_effects": False, "option_count": 0},
        decision={"content": "Hello"},
    )
    assert result.intent == OutputIntent.INFORM

    # 2. 路由渲染
    router = OutputRouter()
    plan = router.route(result.intent, {"text": "Hello"}, {})
    assert plan.renderer_name == "text"

    # 3. 审计链
    chain = OutputAuditChain()
    entry = chain.append(
        output_id="out-001", intent="INFORM",
        content=b"Hello", decision_source="IKO",
        risk_level=0.1, confidence=0.95,
        reasoning_chain_hash="abc123",
    )
    assert chain.verify()

    # 4. 推理压缩
    codec = SymmetricCodec()
    compressed = codec.compress(FullReasoningChain(
        phases=[{"description": "test", "confidence": 0.8}],
        risk_assessments=[{"decision": "proceed"}],
    ))
    assert codec.verify_reversibility(compressed)

    # 5. λ 自校准
    calibrator = LambdaCalibrator()
    calibrator.update(FeedbackSignal.ACCEPTED, 0.8)
    assert calibrator.get_current_lambda() > 0.5

版本历史:
    - 0.1.0: 初始版本，8 个核心模块
"""

from openllm.iko.intent_classifier import (
    ClassificationResult,
    IntentClassifier,
    OutputIntent,
)
from openllm.iko.feedback_collector import (
    FeedbackSignal,
    OutputFeedbackCollector,
)
from openllm.iko.output_audit import (
    OutputAuditChain,
    OutputAuditEntry,
)
from openllm.iko.probing_trainer import ProbingTrainer
from openllm.iko.output_router import (
    BaseRenderer,
    OutputRouter,
    RendererRegistry,
    RenderPlan,
)
from openllm.iko.silence_auditor import SilenceAuditor
from openllm.iko.symmetric_codec import (
    CompressedReasoning,
    FullReasoningChain,
    SymmetricCodec,
)
from openllm.iko.lambda_calibrator import (
    LambdaState,
    LambdaCalibrator,
)
from openllm.iko.pulse import (
    Pulse,
    PulseEngine,
    PulseTrigger,
    IntervalTrigger,
    TimeWindow,
)

__version__ = "0.1.0"

__all__ = [
    # intent_classifier
    "OutputIntent",
    "IntentClassifier",
    "ClassificationResult",
    # feedback_collector
    "FeedbackSignal",
    "OutputFeedbackCollector",
    # output_audit
    "OutputAuditEntry",
    "OutputAuditChain",
    # probing_trainer
    "ProbingTrainer",
    # output_router
    "BaseRenderer",
    "OutputRouter",
    "RendererRegistry",
    "RenderPlan",
    # silence_auditor
    "SilenceAuditor",
    # symmetric_codec
    "CompressedReasoning",
    "FullReasoningChain",
    "SymmetricCodec",
    # lambda_calibrator
    "LambdaState",
    "LambdaCalibrator",
    # pulse（P1-4 温度脉冲引擎）
    "Pulse",
    "PulseEngine",
    "PulseTrigger",
    "IntervalTrigger",
    "TimeWindow",
]
