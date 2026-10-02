"""
openLLM 三体通信协议 — 感知→决策→执行的数据契约

三层不直接调用彼此。通过HeartbeatContext通信。
每层只写自己负责的字段，读上一层的输出。
"""
from dataclasses import dataclass, field
from typing import Any, Optional
from .models import (Message, Context, Prediction, RiskAssessment,
                     Proposal, Critique, Decision, ActionResult,
                     CausalDelta, TickMetrics)


@dataclass
class HeartbeatContext:
    """
    三体通信的合同——每层写自己负责的字段。
    
    数据流：感知层→决策层→执行层
    
    感知层写：identity, memory, search_results, prediction, left_proposal, right_critique
    决策层写：risk, decision
    执行层写：result, output, metrics
    """
    # ── 输入 ──
    user_message: str = ""
    tick_id: str = ""
    
    # ── 感知层输出 ──
    identity: dict = field(default_factory=dict)      # ISA: 身份
    memory: dict = field(default_factory=dict)        # ISA: 记忆召回
    search_results: list = field(default_factory=list) # 章鱼I: 搜索结果
    prediction: Optional[Prediction] = None            # 章鱼I: 因果预测
    left_proposal: Optional[Proposal] = None           # 左脑: 行动提案
    right_critique: Optional[Critique] = None          # 右脑: 批判审查
    
    # ── 决策层输出 ──
    risk: Optional[RiskAssessment] = None              # IOS: 风险评估
    decision: Optional[Decision] = None                # IOS: 仲裁决策
    rejection_record: Optional[Any] = None             # IOS: 拒绝记录（如有）
    
    # ── 执行层输出 ──
    result: Optional[ActionResult] = None              # ISN: 执行结果
    output: str = ""                                   # IKO: 最终输出
    # DR-20260828-01：tool_calls升为协议一等公民（左脑产生→Decision携带→ISN消费→此处全程可见）
    tool_calls: list = field(default_factory=list)     # 本轮实际执行的工具调用清单
    # P0.5(20260925)：综合消化回路产物（_synthesize 写→_learn 4.3 读，空=回退裸输出）。
    # 协议字段显式声明——不许隐式 setattr（防未来加 __slots__ 时静默炸）。
    synth_output: str = ""                             # IAX: 模型消化后的最终回答
    # 加固④(20261001)：工具任务「空输出」告警（送达闸写→监控/终端读，None=未触发）。
    tool_output_alarm: Optional[dict] = None           # IAX: 工具任务无产出告警（如有）
    # 加固⑤(20261001)：工具失败信号（退出码非0/超时/错误/拦截，及"不像命令的 shell 输入"）。
    # 第一步只观测（不阻断、不改 success），第二步再定语义。
    tool_failure_signal: Optional[dict] = None         # IAX: 工具失败/异常输入信号（如有）
    # 加固⑤·第二步 2-pre(20261002)：执行前输入侧拦截（shell 类工具收到"显然不像命令"
    # 的文本→不执行本 turn，直接退回）。协议字段显式声明——不许隐式 setattr。
    tool_input_reject: Optional[dict] = None           # IAX: 工具输入拦截记录（如有）
    
    # ── 研究层输出 ──
    research: Optional[dict] = None                    # 研究循环: 假说/实验/结论状态
    metrics: Optional[TickMetrics] = None              # IKO: 可观测数据
    
    # ── 元数据 ──
    phase_log: list = field(default_factory=list)      # 各阶段日志
    errors: list = field(default_factory=list)         # 错误记录
    
    def log_phase(self, phase: str, status: str, detail: str = ""):
        """记录一个阶段的执行"""
        self.phase_log.append({
            "phase": phase,
            "status": status,
            "detail": detail[:100],
        })
    
    def add_error(self, phase: str, error: str):
        """记录一个错误"""
        self.errors.append({"phase": phase, "error": error[:200]})
    
    def get_perception_output(self) -> dict:
        """感知层输出摘要"""
        return {
            "identity": bool(self.identity),
            "memory_keys": list(self.memory.keys()),
            "search_count": len(self.search_results),
            "has_prediction": self.prediction is not None,
            "has_proposal": self.left_proposal is not None,
            "has_critique": self.right_critique is not None,
        }
    
    def get_decision_output(self) -> dict:
        """决策层输出摘要"""
        return {
            "risk_level": self.risk.level if self.risk else None,
            "risk_blocked": self.risk.is_blocked() if self.risk else False,
            "decision_action": self.decision.action if self.decision else None,
            "decision_approved": self.decision.approved if self.decision else False,
            "has_rejection": self.rejection_record is not None,
        }
    
    def get_execution_output(self) -> dict:
        """执行层输出摘要"""
        return {
            "result_success": self.result.success if self.result else None,
            "output_length": len(self.output),
            "has_metrics": self.metrics is not None,
        }
    
    def is_complete(self) -> bool:
        """检查三层是否都已写入"""
        return (
            bool(self.identity) and
            self.decision is not None and
            self.result is not None
        )
