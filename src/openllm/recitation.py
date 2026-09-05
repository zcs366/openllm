"""
RecitationEngine — 诵经引擎（T1排版艺的动态臂）
================================================

来源：Manus六课之四"Manipulate Attention Through Recitation"
（Peak Ji 2025-07, manus.im/blog，原文已亲读精读）——
  "By constantly rewriting the todo list, Manus is reciting its
   objectives into the end of the context. This pushes the global
   plan into the model's recent attention span, avoiding
   'lost-in-the-middle' issues and reducing goal misalignment."

宪章位：T1地形定理（U形位置曲线，尾端=注意力高区）的生产级实证移植；
与ushaped_context_engine互补——ushaped管静态摆放（注入时按U形排），
recitation管动态尾端强化（长循环中周期性把目标诵进上下文末端）。

问题：长任务（Manus实测平均~50次工具调用）中LLM决策必然漂移/遗忘
早期目标。诵经=零架构改动的注意力偏置：纯自然语言把全局计划推进
模型近期注意力窗口。

设计纪律（三条全部来自Manus原文，均有出处）：
  1. **append-only**（课1 KV-cache纪律）：诵经消息只追加在上下文
     末端，永不修改前缀——单token差异即毁缓存，诵经若改前缀就是
     自杀。recite_into()保证输入前缀逐条不动。
  2. **controlled variation**（课6 Don't Get Few-Shotted）：诵经
     文本完全逐字重复会形成模式锁死（"上下文越均质agent越脆"）——
     模板轮转注入受控变化，确定性（seeded）可复现。
  3. **确定性零LLM**（家规，同compaction_control/context_pressure/
     consolidation_score族）：诵经是规则引擎，不调模型。

用法：
    engine = RecitationEngine(RecitationConfig(every_n_steps=10))
    engine.set_goals(["下载论文", "精读", "写宪章v0.3"])
    ...
    msg = engine.on_step()          # 每步调用；到点返回诵经Message否则None
    if msg: messages.append(msg)    # 或 messages = engine.recite_into(messages)
    engine.mark_done(0)             # 完成一项
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .message import Message

logger = logging.getLogger("openllm.recitation")


# ── 诵经模板（controlled variation：轮转防模式锁死）──────────
# 模板差异=措辞与结构，信息内容恒等（目标+进度）——变化是排版级
# 不是语义级，防止诵经本身引入歧义。
DEFAULT_TEMPLATES: List[str] = [
    (
        "【目标诵经 #{n}】当前任务目标（共{total}项，已完成{done}项）：\n"
        "{goals}\n请对照以上目标继续下一步，勿偏离。"
    ),
    (
        "【进度检查点 #{n}】全局计划回顾——{done}/{total} 已完成。\n"
        "待办：\n{goals}\n下一步行动必须服务于上述待办之一。"
    ),
    (
        "【诵经 #{n}】防止目标漂移，重申任务全貌（{done}/{total}完成）：\n"
        "{goals}\n若当前动作与任何待办无关，先停下来重新对齐。"
    ),
]


@dataclass
class RecitationConfig:
    """诵经配置——声明式。

    Attributes:
        every_n_steps: 每N步诵经一次（Manus任务均长~50步，默认10=
            每任务约5次诵经；过密毁缓存经济性，过疏防不住漂移）
        max_recitations: 单任务诵经上限（防上下文洪泛——诵经本身
            也占token，无限诵经=自己制造context rot）
        seed: 模板轮转种子（确定性可复现；None=按时间）
        role: 诵经消息的role（system=最高注意力权重；user=部分
            harness会重排。默认system）
        include_done: 是否列出已完成项（False=只诵待办，省token）
        templates: 自定义模板列表（覆盖DEFAULT_TEMPLATES）
    """
    every_n_steps: int = 10
    max_recitations: int = 50
    seed: Optional[int] = 42
    role: str = "system"
    include_done: bool = False
    templates: List[str] = field(default_factory=lambda: list(DEFAULT_TEMPLATES))

    def __post_init__(self):
        if self.every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1")
        if self.role not in {"system", "user"}:
            raise ValueError(f"role must be system|user, got {self.role!r}")
        if not self.templates:
            raise ValueError("templates must not be empty")


@dataclass
class RecitationStats:
    """诵经统计——P-Manus-1 A/B验证的度量位。"""
    steps: int = 0                  # 总步数
    recitations: int = 0            # 实际诵经次数
    skipped_at_cap: int = 0         # 因上限跳过的次数
    goals_set: int = 0              # 目标设定/更新次数
    goals_done: int = 0             # 完成项数
    last_recited_step: int = -1     # 最近一次诵经的步号

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "recitations": self.recitations,
            "skipped_at_cap": self.skipped_at_cap,
            "goals_set": self.goals_set,
            "goals_done": self.goals_done,
            "last_recited_step": self.last_recited_step,
        }


class RecitationEngine:
    """诵经引擎——纯规则，零LLM，append-only。"""

    def __init__(self, config: Optional[RecitationConfig] = None):
        self.config = config or RecitationConfig()
        self._goals: List[str] = []
        self._done: List[bool] = []
        self._rng = random.Random(self.config.seed)
        self._template_order: List[int] = list(range(len(self.config.templates)))
        self._rng.shuffle(self._template_order)  # 确定性洗牌（seed固定）
        self._template_cursor = 0
        self.stats = RecitationStats()

    # ── 目标管理（todo.md的等价物）────────────────────────

    def set_goals(self, goals: List[str]) -> None:
        """设定/重置目标列表。空串剔除。"""
        self._goals = [g for g in goals if g and g.strip()]
        self._done = [False] * len(self._goals)
        self.stats.goals_set += 1

    def add_goal(self, goal: str) -> int:
        """追加单个目标，返回索引。"""
        if not goal or not goal.strip():
            raise ValueError("goal must be non-empty")
        self._goals.append(goal)
        self._done.append(False)
        return len(self._goals) - 1

    def mark_done(self, index: int) -> bool:
        """标记第index项完成。越界返回False。"""
        if 0 <= index < len(self._done):
            if not self._done[index]:
                self._done[index] = True
                self.stats.goals_done += 1
            return True
        return False

    @property
    def pending_goals(self) -> List[str]:
        return [g for g, d in zip(self._goals, self._done) if not d]

    @property
    def done_goals(self) -> List[str]:
        return [g for g, d in zip(self._goals, self._done) if d]

    @property
    def all_done(self) -> bool:
        return bool(self._goals) and all(self._done)

    # ── 诵经核心 ─────────────────────────────────────────

    def on_step(self) -> Optional[Message]:
        """每步调用一次。到诵经点且未达上限→返回诵经Message；否则None。

        无目标时不诵经（没有经文可诵）。全部完成时不诵经（任务已毕，
        再诵=噪音）。
        """
        self.stats.steps += 1
        if not self._goals or self.all_done:
            return None
        if self.stats.steps % self.config.every_n_steps != 0:
            return None
        if self.stats.recitations >= self.config.max_recitations:
            self.stats.skipped_at_cap += 1
            return None
        msg = self._render()
        self.stats.recitations += 1
        self.stats.last_recited_step = self.stats.steps
        return msg

    def _render(self) -> Message:
        """渲染诵经消息——模板轮转（controlled variation）。"""
        tpl_idx = self._template_order[
            self._template_cursor % len(self._template_order)
        ]
        self._template_cursor += 1
        tpl = self.config.templates[tpl_idx]

        lines = []
        for i, (g, d) in enumerate(zip(self._goals, self._done)):
            if d:
                if self.config.include_done:
                    lines.append(f"  [x] {g}")
            else:
                lines.append(f"  [ ] {g}")
        goals_text = "\n".join(lines) if lines else "（无待办）"

        content = tpl.format(
            n=self.stats.recitations + 1,
            total=len(self._goals),
            done=sum(self._done),
            goals=goals_text,
        )
        return Message(
            role=self.config.role,
            content=content,
            metadata={
                "recitation": True,
                "recitation_no": self.stats.recitations + 1,
                "step": self.stats.steps,
                "template_idx": tpl_idx,
                "pending": len(self.pending_goals),
            },
        )

    def recite_into(self, messages: List[Message]) -> List[Message]:
        """到点则把诵经消息追加到消息列表末端，返回新列表。

        **append-only纪律**：输入列表的前缀逐条不动（同对象引用），
        只在末端追加——KV-cache友好（Manus课1）。调用方每步调一次
        本方法即可，无需自己管on_step。
        """
        msg = self.on_step()
        if msg is None:
            return messages
        return list(messages) + [msg]

    # ── 观测 ─────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        d = self.stats.to_dict()
        d["pending_goals"] = len(self.pending_goals)
        d["every_n_steps"] = self.config.every_n_steps
        return d
