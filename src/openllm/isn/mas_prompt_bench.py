"""
MASPromptBench — ISN Layer 11.b MAS Prompt评测
=============================================

多智能体系统（MAS）Prompt质量评测模块。对prompt模板在多agent协作场景
中的表现进行规则化评分，无需LLM调用。

评分维度：
- 角色一致性（role_consistency）：prompt是否为每个agent定义了清晰角色  × 0.3
- 任务覆盖（task_coverage）：prompt是否完整覆盖场景任务要求           × 0.4
- 协作引导（collaboration_guide）：prompt是否包含协作机制（交接/共识） × 0.3

纯规则引擎，零LLM调用。JSON持久化到 ~/.hermes/isn/mas_prompt_bench.json
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("openllm.isn.mas_prompt_bench")

_STATE_DIR = Path.home() / ".hermes" / "isn"
_STATE_PATH = _STATE_DIR / "mas_prompt_bench.json"

# 评分权重
W_ROLE = 0.3
W_TASK = 0.4
W_COLLAB = 0.3

# 协作信号关键词
_COLLAB_KEYWORDS = [
    "协作", "合作", "交接", "共识", "讨论", "反馈", "协调",
    "分工", "汇总", "集成", "review", "handoff", "consensus",
]


@dataclass
class Scenario:
    """评测场景定义。"""
    name: str
    agents: list[str]        # 参与agent角色列表
    task: str                # 任务描述
    expected_outcome: str    # 期望产出


@dataclass
class EvalResult:
    """单条prompt的评测结果。"""
    template_id: int
    scenario: str
    role_consistency: float
    task_coverage: float
    collaboration: float
    total: float


@dataclass
class BenchState:
    """持久化状态。"""
    scenarios: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)


class MASPromptBench:
    """MAS Prompt评测器。

    评测prompt模板在多agent协作场景中的质量。

    用法::

        bench = MASPromptBench()
        bench.add_scenario("代码审查", ["审查员", "作者"], "review PR", "高质量review报告")
        report = bench.evaluate_prompt("你是{role}，请{task}...", agent_profiles)
        best = bench.get_best_prompt()
    """

    def __init__(self, state_path: Path | None = None):
        self._state_path = state_path or _STATE_PATH
        self._scenarios: list[Scenario] = []
        self._results: list[EvalResult] = []
        self._load()

    # ── 公共 API ──

    def add_scenario(
        self, name: str, agents: list[str], task: str, expected_outcome: str
    ) -> None:
        """注册评测场景。

        Args:
            name: 场景唯一名称
            agents: 参与agent角色列表
            task: 任务描述
            expected_outcome: 期望产出描述
        """
        self._scenarios.append(Scenario(name, agents, task, expected_outcome))
        self._persist()

    def evaluate_prompt(
        self, prompt_template: str, agent_profiles: dict[str, str]
    ) -> dict[str, Any]:
        """评估单条prompt模板在所有已注册场景中的表现。

        Args:
            prompt_template: prompt模板文本
            agent_profiles: {角色名: 角色描述} 映射

        Returns:
            {scenario_name: {scores, total}} 以及 overall 均值
        """
        if not self._scenarios:
            return {"error": "无注册场景", "overall": 0.0}

        reports: dict[str, dict] = {}
        totals: list[float] = []

        for sc in self._scenarios:
            rc = self._score_role_consistency(prompt_template, sc, agent_profiles)
            tc = self._score_task_coverage(prompt_template, sc)
            co = self._score_collaboration(prompt_template, sc)
            total = round(rc * W_ROLE + tc * W_TASK + co * W_COLLAB, 4)
            reports[sc.name] = {
                "role_consistency": round(rc, 4),
                "task_coverage": round(tc, 4),
                "collaboration": round(co, 4),
                "total": total,
            }
            totals.append(total)
            self._results.append(EvalResult(
                template_id=hash(prompt_template) & 0xFFFFFFFF,
                scenario=sc.name,
                role_consistency=rc,
                task_coverage=tc,
                collaboration=co,
                total=total,
            ))

        overall = round(sum(totals) / len(totals), 4) if totals else 0.0
        self._persist()
        return {"scores": reports, "overall": overall}

    def compare_prompts(
        self, templates: list[str], agent_profiles: dict[str, str]
    ) -> dict[str, Any]:
        """对比多条prompt模板，返回排名。

        Args:
            templates: prompt模板列表
            agent_profiles: {角色名: 角色描述}

        Returns:
            {rankings: [...], winner: best_template_id}
        """
        rankings: list[dict] = []
        for i, tpl in enumerate(templates):
            report = self.evaluate_prompt(tpl, agent_profiles)
            rankings.append({
                "template_index": i,
                "template_preview": tpl[:80],
                "overall": report["overall"],
            })
        rankings.sort(key=lambda r: r["overall"], reverse=True)
        winner = rankings[0]["template_index"] if rankings else None
        return {"rankings": rankings, "winner": winner}

    def get_best_prompt(self) -> str | None:
        """返回历史评测中得分最高的prompt模板ID。

        Returns:
            最高分模板的 hash ID，无结果时返回 None
        """
        if not self._results:
            return None
        best = max(self._results, key=lambda r: r.total)
        return str(best.template_id)

    def get_scenarios(self) -> list[dict]:
        """返回所有已注册场景。"""
        return [asdict(s) for s in self._scenarios]

    # ── 评分方法（纯规则） ──

    def _score_role_consistency(
        self, prompt: str, sc: Scenario, profiles: dict[str, str]
    ) -> float:
        """角色一致性：prompt是否为每个agent定义了清晰角色。"""
        hits = 0
        for agent in sc.agents:
            # 角色名在prompt中出现 或 对应profile关键词出现
            if agent in prompt:
                hits += 1
            elif agent in profiles:
                # profile描述的关键词是否出现在prompt中
                keywords = set(profiles[agent].split())
                if any(kw in prompt for kw in keywords):
                    hits += 1
        return hits / len(sc.agents) if sc.agents else 0.0

    def _score_task_coverage(self, prompt: str, sc: Scenario) -> float:
        """任务覆盖：prompt是否完整覆盖场景任务要求。"""
        # 任务关键词命中率
        task_words = [w for w in re.split(r"[，。、；\s]+", sc.task) if len(w) >= 2]
        if not task_words:
            return 1.0
        hits = sum(1 for w in task_words if w in prompt)
        return hits / len(task_words)

    def _score_collaboration(self, prompt: str, sc: Scenario) -> float:
        """协作引导：prompt是否包含协作机制。"""
        if len(sc.agents) <= 1:
            return 1.0  # 单agent无协作需求
        hits = sum(1 for kw in _COLLAB_KEYWORDS if kw in prompt)
        return min(hits / 3, 1.0)  # 命中3个关键词即满分

    # ── 持久化 ──

    def _load(self) -> None:
        if self._state_path.exists():
            try:
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                self._scenarios = [Scenario(**s) for s in raw.get("scenarios", [])]
                self._results = [EvalResult(**r) for r in raw.get("results", [])]
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("MAS Prompt Bench 状态文件损坏，使用空状态")

    def _persist(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        state = BenchState(
            scenarios=[asdict(s) for s in self._scenarios],
            results=[asdict(r) for r in self._results],
        )
        self._state_path.write_text(
            json.dumps(asdict(state), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
