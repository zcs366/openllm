"""六体统一接口协议。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import time as _time


class BodyInterface(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def status(self) -> dict: ...
    def execute(self, hc, **kwargs) -> Optional[Any]:
        return None


class BodyRegistry:
    def __init__(self):
        self._bodies: dict[str, BodyInterface] = {}
    def register(self, body):
        self._bodies[body.name] = body
    def get(self, name):
        return self._bodies.get(name)
    def status(self):
        return {n: b.status() for n, b in self._bodies.items()}
    def names(self):
        return list(self._bodies.keys())


@dataclass
class UnifiedHeartbeat:
    user_message: str = ""
    tick_id: str = ""
    registry: Optional[BodyRegistry] = None
    identity: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    search_results: list = field(default_factory=list)
    prediction: Any = None
    left_proposal: Any = None
    right_critique: Any = None
    risk: Any = None
    decision: Any = None
    result: Any = None
    output: str = ""
    research: Optional[dict] = None
    phase_log: list = field(default_factory=list)
    def log_phase(self, phase, status, detail=""):
        self.phase_log.append({"phase": phase, "status": status, "detail": detail, "time": _time.time()})


class IAIBody(BodyInterface):
    name = "IAI"
    def __init__(self, brain=None):
        self._brain = brain
    def status(self):
        return {"health": self._brain.health_check() if self._brain else "offline"}
    def execute(self, hc, **kwargs):
        if not self._brain:
            return
        ctx = kwargs.get("ctx")
        if ctx and hc.prediction is None:
            hc.prediction = self._brain.left.predict(ctx)
        if ctx and hc.left_proposal is None:
            hc.left_proposal, hc.right_critique = self._brain.reason(ctx, hc.prediction, kwargs.get("risk") or hc.risk)


class IOSBody(BodyInterface):
    name = "IOS"
    def __init__(self, ios=None):
        self._ios = ios
    def status(self):
        return {"health": "online" if self._ios else "offline"}
    def execute(self, hc, **kwargs):
        if not self._ios:
            return
        ctx = kwargs.get("ctx")
        if ctx and hc.prediction and hc.risk is None:
            hc.risk = self._ios.risk_check(ctx, hc.prediction)
        if hc.left_proposal and hc.right_critique and hc.decision is None:
            hc.decision = self._ios.arbitrate(hc.left_proposal, hc.right_critique, hc.risk)


class ISNBody(BodyInterface):
    name = "ISN"
    def __init__(self, isn=None):
        self._isn = isn
    def status(self):
        return {"health": "online" if self._isn else "offline", "tools": len(getattr(self._isn, 'tools', {})) if self._isn else 0}
    def execute(self, hc, **kwargs):
        if self._isn and hc.decision and hc.result is None:
            hc.result = self._isn.execute(hc.decision)


class IKOBody(BodyInterface):
    name = "IKO"
    def __init__(self, iko=None):
        self._iko = iko
    def status(self):
        return {"health": "online" if self._iko else "offline"}
    def execute(self, hc, **kwargs):
        if self._iko and hc.result:
            hc.output = hc.result.output if hasattr(hc.result, 'output') else str(hc.result)


# ═══════════════════════════════════════════════
# 六体显式接线（2026-10-01 配平 · 治「隐式接线、断了不响」）
# ═══════════════════════════════════════════════
# 病（实测）：ISN 以前是 `getattr(getattr(agent,"isn",None),"execute",None)` 取的——
#   改名不报错、装不上就返回 None、然后静默跳过整个工具循环。
#   这就是「工具异动，下面没了」能在系统里存活的机制。
# 药（两条，缺一不可）：
#   ① 启动即验 validate_bodies()：缺件**当场抛**，不许带病上工；
#   ② 运行时取件 resolve_body_method()：缺件**必吼一声**（ERROR），不许静默。
# 诚实标注：IAX 是**机制体**（本身就是心跳，无实例），故其「接线」= 本模块被加载。
SIX_BODIES = {
    "IAI": ("iai", ("gate", "emit")),
    "ISA": ("isa", ("build_context", "respond")),
    "IOS": ("ios", ("risk_check", "arbitrate")),
    "ISN": ("isn", ("execute",)),
    "IKO": ("iko", ("trace",)),
    "IAX": (None, ()),   # 机制体：无实例
}


def validate_bodies(agent, strict: bool = True) -> dict:
    """启动即验六体是否真的接上了。返回体检表；strict 时缺件抛 RuntimeError。"""
    report: dict = {}
    missing: list = []
    for body, (attr, methods) in SIX_BODIES.items():
        if attr is None:
            report[body] = {"kind": "机制体", "ok": True, "missing": []}
            continue
        obj = getattr(agent, attr, None)
        lack = []
        if obj is None:
            lack.append(f"agent.{attr}（实例缺失）")
        else:
            for m in methods:
                if getattr(obj, m, None) is None:
                    lack.append(f"agent.{attr}.{m}()")
        report[body] = {"kind": "实例体", "ok": not lack, "missing": lack}
        if lack:
            missing.append(body)
    if missing and strict:
        raise RuntimeError(
            "六体接线不全，拒绝带病启动：" + "；".join(f"{b} → {report[b]['missing']}" for b in missing))
    return report


def resolve_body_method(agent, body: str, method: str, *, logger=None):
    """运行时显式取入口方法。缺件**不静默**：打 ERROR 后返回 None。"""
    attr = SIX_BODIES.get(body.upper(), (body.lower(), ()))[0] or body.lower()
    fn = getattr(getattr(agent, attr, None), method, None)
    if fn is None and logger is not None:
        logger.error("六体接线缺口：agent.%s.%s 不存在——本阶段该项能力缺失（非静默告警）",
                     attr, method)
    return fn


# ═══════════════════════════════════════════════
# 加固一批（2026-10-01 夜 · 承重账反事实实验的三条建议）
# ═══════════════════════════════════════════════
# 实验实锤：把 IAI 的 predict_consequences 换成返回 list、把 IOS 的 arbitrate 换成返回
# Proposal，系统**当场崩**——崩点不是功能缺失，而是下游直接取属性：
#   `.summary_text()` / `.approved`。即：**六体边界不是被强制的，是靠"大家都记得
#   该返回什么"维系的**（「边界假」的机器证据）。
# 三道加固（都不改行为，只把「约定」变成「可机判」）：
#   ① 命名协议：每个体一份 runtime_checkable Protocol（有形可查的接口）
#   ② 书面契约：入口必须有返回类型注解（没写＝口头约定，启动即拒）
#   ③ 形状冒烟：声明的返回类型须含**下游真正会读的成员**（零副作用静态检查，
#      不调用真实入口，因此不会写记忆、不会执行工具）
from typing import Protocol, runtime_checkable  # noqa: E402


@runtime_checkable
class ISAEntry(Protocol):
    def build_context(self, msg, session=None, octopus=None, ios=None): ...
    def respond(self, text, phase_times=None): ...


@runtime_checkable
class IAIEntry(Protocol):
    def gate(self, *a, **k): ...
    def emit(self, *a, **k): ...


@runtime_checkable
class IOSEntry(Protocol):
    def risk_check(self, ctx, prediction): ...
    def arbitrate(self, proposal, critique, risk): ...


@runtime_checkable
class ISNEntry(Protocol):
    def execute(self, decision): ...


@runtime_checkable
class IKOEntry(Protocol):
    def trace(self, phase, status, duration_ms=0.0, detail=""): ...
    def process_output(self, *a, **k): ...


BODY_ENTRY_PROTOCOLS = {
    "ISA": ISAEntry, "IAI": IAIEntry, "IOS": IOSEntry,
    "ISN": ISNEntry, "IKO": IKOEntry,
}

# (实例属性, 方法) → 契约：returns=注解里声明的类型名；attrs=下游真正会读的成员
ENTRY_CONTRACTS = {
    ("isa", "build_context"): {"returns": "Context", "attrs": ("user_message", "memory", "identity")},
    ("isa", "respond"): {"returns": "None", "attrs": ()},
    ("octopus", "predict_consequences"): {"returns": "Prediction", "attrs": ("summary", "summary_text")},
    ("octopus", "reason"): {"returns": "tuple", "attrs": ()},
    ("octopus", "d0_snapshot"): {"returns": "dict", "attrs": ()},
    ("ios", "risk_check"): {"returns": "RiskAssessment", "attrs": ("level", "blocked")},
    ("ios", "arbitrate"): {"returns": "Decision", "attrs": ("approved",)},
    ("ios", "cap_check"): {"returns": "bool", "attrs": ()},
    ("isn", "execute"): {"returns": "ActionResult", "attrs": ("success", "output")},
    ("iko", "trace"): {"returns": "None", "attrs": ()},
    ("iko", "process_output"): {"returns": "str", "attrs": ()},
}


def _resolve_return_name(annotation) -> str:
    """把返回注解归一成名字（'Context' / 'tuple[Proposal, Critique]' → 'tuple'）。"""
    if annotation is None:
        return "None"
    if isinstance(annotation, type):
        return annotation.__name__
    text = str(annotation)
    text = text.replace("typing.", "")
    for name in ("None", "Protocol"):
        if text == name:
            return name
    return text.split("[")[0].split(".")[-1]


def validate_entry_contracts(agent, strict: bool = True) -> dict:
    """②③：入口必须有**书面返回契约**，且声明的返回类型含**下游真正会读的成员**。

    零副作用：只读注解与类型对象，**不调用任何真实入口**（不写记忆、不执行工具）。
    """
    import dataclasses
    import inspect

    report: dict = {}
    bad: list = []
    for (attr, method), spec in ENTRY_CONTRACTS.items():
        key = f"agent.{attr}.{method}()"
        fn = getattr(getattr(agent, attr, None), method, None)
        if fn is None:
            report[key] = {"ok": False, "why": "入口缺失"}
            bad.append(key)
            continue
        try:
            annotation = inspect.signature(fn).return_annotation
        except (TypeError, ValueError):
            annotation = inspect.Signature.empty
        if annotation is inspect.Signature.empty:
            report[key] = {"ok": False, "why": "无书面返回契约（缺返回注解）"}
            bad.append(key)
            continue

        declared = _resolve_return_name(annotation)
        if spec["returns"] not in (declared, "None", "Protocol") and declared != spec["returns"]:
            report[key] = {"ok": False, "why": f"返回契约不符：声明 {declared}，期望 {spec['returns']}"}
            bad.append(key)
            continue

        # ③ 形状：若声明的类型是仓内 dataclass，检查下游会读的成员是否存在
        #    注意：**必填字段在类上不是属性**（只在 __init__ 里赋值），
        #    故必须查 dataclasses.fields()，不能用 hasattr（否则误报，实测踩过）。
        lack = []
        if spec["attrs"]:
            from . import models as _models
            cls = getattr(_models, declared, None)
            if cls is not None and dataclasses.is_dataclass(cls):
                names = {f.name for f in dataclasses.fields(cls)}
                for a in spec["attrs"]:
                    if a not in names and not hasattr(cls, a):
                        lack.append(f"{declared}.{a}")
        if lack:
            report[key] = {"ok": False, "why": "返回形状缺下游要读的成员：" + "、".join(lack)}
            bad.append(key)
            continue

        report[key] = {"ok": True, "returns": declared, "checked_attrs": list(spec["attrs"])}

    if bad and strict:
        raise RuntimeError("六体接口契约不成立，拒绝带病启动：" + "；".join(
            f"{k} → {report[k]['why']}" for k in bad))
    return report


def validate_protocols(agent) -> dict:
    """① 命名协议：实例是否满足该体的入口协议（runtime_checkable，按成员存在判定）。"""
    out = {}
    for body, proto in BODY_ENTRY_PROTOCOLS.items():
        attr = SIX_BODIES[body][0]
        obj = getattr(agent, attr, None)
        out[body] = bool(obj is not None and isinstance(obj, proto))
    return out
