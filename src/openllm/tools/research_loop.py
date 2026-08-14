"""
研究循环 — 研究agent的主循环扩展。

在openLLM通用12阶段心跳之上，增加研究专属阶段:
  observe → hypothesize → design → run → analyze → conclude → iterate

与通用心跳的关系:
  通用心跳处理"用户说了什么→系统做什么"
  研究循环处理"观察到什么→假说什么→验证什么→学到什么"

两层循环嵌套: 研究循环的每一步可以触发通用心跳来执行具体操作。
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .experiment_engine import ExperimentEngine, Hypothesis, HypothesisStatus
from .paper_knowledge import PaperKnowledge


class ResearchPhase(Enum):
    """研究循环阶段。"""
    OBSERVE = "observe"        # 观察现象、收集数据
    HYPOTHESIZE = "hypothesize"  # 提出假说
    DESIGN = "design"          # 设计实验
    RUN = "run"                # 执行实验
    ANALYZE = "analyze"        # 分析结果
    CONCLUDE = "conclude"      # 得出结论
    ITERATE = "iterate"        # 基于结论迭代


@dataclass
class ResearchState:
    """研究循环的当前状态。"""
    phase: ResearchPhase = ResearchPhase.OBSERVE
    current_hypothesis_id: Optional[str] = None
    current_experiment_id: Optional[str] = None
    observations: list[dict] = field(default_factory=list)
    conclusions: list[dict] = field(default_factory=list)
    iteration_count: int = 0
    started_at: float = field(default_factory=time.time)

    def dict(self) -> dict:
        return {
            "phase": self.phase.value,
            "hypothesis": self.current_hypothesis_id,
            "experiment": self.current_experiment_id,
            "observations": len(self.observations),
            "conclusions": len(self.conclusions),
            "iterations": self.iteration_count,
        }


class ResearchLoop:
    """
    研究agent的循环引擎。

    核心区别于通用agent循环:
    - 通用循环: 输入→处理→输出 (单次)
    - 研究循环: 观察→假说→实验→结论→迭代 (递归)

    每次迭代必须有可证伪预测，否则不是研究——是浏览。
    """

    def __init__(self):
        self.engine = ExperimentEngine()
        self.knowledge = PaperKnowledge()
        self.state = ResearchState()
        self._research_log: list[dict] = []

    # ── 阶段推进 ──

    def observe(self, observation: str, source: str = "", metrics: Optional[dict] = None) -> dict:
        """记录一个观察。"""
        obs = {
            "text": observation,
            "source": source,
            "metrics": metrics or {},
            "timestamp": time.time(),
        }
        self.state.observations.append(obs)
        self.state.phase = ResearchPhase.OBSERVE
        self._log("observe", f"New observation: {observation[:80]}")
        return obs

    def hypothesize(self, claim: str, prediction: str, id: Optional[str] = None) -> Hypothesis:
        """从观察中提出假说。"""
        hid = id or f"h_{len(self.engine.hypotheses)+1:03d}"
        hyp = self.engine.add_hypothesis(hid, claim, prediction)
        self.state.current_hypothesis_id = hid
        self.state.phase = ResearchPhase.HYPOTHESIZE
        self._log("hypothesize", f"[{hid}] {claim}")
        return hyp

    def design_experiment(
        self, name: str, method: str, inputs: Optional[dict] = None
    ) -> str:
        """为当前假说设计实验。"""
        if not self.state.current_hypothesis_id:
            raise ValueError("No hypothesis selected. Call hypothesize() first.")

        exp = self.engine.design_experiment(
            name, self.state.current_hypothesis_id, method, inputs
        )
        self.state.current_experiment_id = exp.id
        self.state.phase = ResearchPhase.DESIGN
        self._log("design", f"Experiment [{exp.id}]: {name}")
        return exp.id

    def run_experiment(self, success: bool, metrics: Optional[dict] = None,
                       conclusion: str = "", duration_s: float = 0.0) -> dict:
        """记录实验运行结果。"""
        if not self.state.current_experiment_id:
            raise ValueError("No experiment to run. Call design_experiment() first.")

        result = self.engine.record_result(
            self.state.current_experiment_id, success,
            metrics, conclusion, duration_s=duration_s
        )
        self.state.phase = ResearchPhase.RUN
        self._log("run", f"[{self.state.current_experiment_id}] {'✅' if success else '❌'} {conclusion[:60]}")
        return result.dict()

    def conclude(self, finding: str, confidence: str = "medium") -> dict:
        """得出研究结论。"""
        conclusion = {
            "finding": finding,
            "confidence": confidence,
            "hypothesis": self.state.current_hypothesis_id,
            "experiment": self.state.current_experiment_id,
            "iteration": self.state.iteration_count,
            "timestamp": time.time(),
        }
        self.state.conclusions.append(conclusion)
        self.state.phase = ResearchPhase.CONCLUDE
        self._log("conclude", f"[{confidence}] {finding[:80]}")
        return conclusion

    def iterate(self) -> dict:
        """基于结论开始新一轮迭代。"""
        self.state.iteration_count += 1
        self.state.current_experiment_id = None
        self.state.phase = ResearchPhase.ITERATE
        self._log("iterate", f"Iteration #{self.state.iteration_count}")
        return self.state.dict()

    # ── 查询 ──

    def status(self) -> dict:
        """当前研究状态。"""
        return {
            "state": self.state.dict(),
            "engine_summary": self.engine.summary(),
            "knowledge_summary": self.knowledge.summary(),
            "recent_log": self._research_log[-10:],
        }

    def _log(self, phase: str, message: str):
        self._research_log.append({
            "phase": phase,
            "message": message,
            "time": time.time(),
        })

    # ── 与通用心跳的桥接 ──

    def to_heartbeat_context(self) -> dict:
        """
        将研究状态转换为通用心跳可消费的context。
        研究循环的每一步可以注入到通用心跳的Phase 1(perceive)中。
        """
        return {
            "research_mode": True,
            "current_phase": self.state.phase.value,
            "active_hypothesis": self.state.current_hypothesis_id,
            "hypotheses_summary": self.engine.summary(),
            "last_conclusion": (
                self.state.conclusions[-1]["finding"]
                if self.state.conclusions else None
            ),
        }
