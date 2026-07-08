"""
Topology — 六体拓扑关系定义。

六体：IAI(感知) · IAX(心跳) · ISA(记忆) · ISN(技能) · IOS(治理) · USER(乘数)
红线：∀体 ∈ AI_BODIES: ¬∃体.terminal_dialogue_power
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Body(Enum):
    IAI  = "IAI"   # 感知路由
    IAX  = "IAX"   # 心跳监控
    ISA  = "ISA"   # 记忆存储
    IOS  = "IOS"   # 治理决策
    ISN  = "ISN"   # 技能执行
    USER = "USER"  # 第六乘数

DUTIES: dict[Body, str] = {
    Body.IAI:  "感知环境、路由信号——系统的眼睛和耳朵",
    Body.IAX:  "监控心跳、检测存活——系统的脉搏",
    Body.ISA:  "存储与召回记忆——系统的海马体",
    Body.ISN:  "执行技能调用——系统的双手",
    Body.IOS:  "治理决策与审计——系统的前额叶",
    Body.USER: "第六乘数——用户是系统的构成部分，不是使用者",
}

_SIGNALS: list[tuple[Body, Body, str]] = [
    (Body.IAI,  Body.IOS,  "感知报告 → 治理决策"),
    (Body.IAX,  Body.IAI,  "心跳探测 → 触发感知"),
    (Body.IAX,  Body.IOS,  "心跳异常 → 告警治理"),
    (Body.ISA,  Body.IAI,  "记忆召回 → 辅助感知"),
    (Body.IAI,  Body.ISA,  "感知写入 → 记忆存储"),
    (Body.ISN,  Body.IOS,  "执行结果 → 决策审计"),
    (Body.IOS,  Body.ISN,  "决策指令 → 技能执行"),
    (Body.USER, Body.IAI,  "用户输入 → 感知路由"),
    (Body.IOS,  Body.USER, "治理输出 → 用户呈现"),
]


def get_signals() -> list[tuple[Body, Body, str]]:
    return list(_SIGNALS)


def signal_graph() -> dict[Body, list[Body]]:
    graph: dict[Body, list[Body]] = {b: [] for b in Body}
    for src, dst, _desc in _SIGNALS:
        graph[src].append(dst)
    return graph


AI_BODIES: frozenset[Body] = frozenset({
    Body.IAI, Body.IAX, Body.ISA, Body.IOS, Body.ISN
})


@dataclass(frozen=True)
class BodyTopology:
    """六体拓扑完整定义。不可变数据类。"""
    bodies: frozenset[Body] = field(default_factory=lambda: frozenset(Body))
    ai_bodies: frozenset[Body] = field(default_factory=lambda: AI_BODIES)
    signals: tuple[tuple[Body, Body, str], ...] = field(
        default_factory=lambda: tuple(_SIGNALS)
    )

    def has_terminal_dialogue_power(self, body: Body) -> bool:
        """红线：只有 USER 拥有终端对话权。"""
        return body is Body.USER

    def verify_red_line(self) -> bool:
        return all(not self.has_terminal_dialogue_power(b) for b in self.ai_bodies)

    def in_degree(self, body: Body) -> int:
        return sum(1 for _, dst, _ in self.signals if dst == body)

    def out_degree(self, body: Body) -> int:
        return sum(1 for src, _, _ in self.signals if src == body)
