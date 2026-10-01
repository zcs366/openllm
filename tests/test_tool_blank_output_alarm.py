"""加固④ 钉子：工具任务「空输出」必须告警（不许安静地交白卷）。

依据（承重账·带工具反事实，2026-10-01 实测）：
    把 ISN 桩化后，问「读一下 pyproject.toml 的前两行」——**答长 0、一个字没出、
    且没有任何报错**。静默白卷比报错更坏：报错能修，白卷只能猜。

判据（本文件逐条验）：
  A. 工具任务 + 最终输出为空            ⇒ 必告警（写显式文本、ERROR 日志、IKO 留痕）
  B. 工具成功但输出没送达（合成断链）    ⇒ 必告警，且子型标注为「送达断链」
  C. 非工具任务 + 输出为空              ⇒ **不告警**（不许误报、不许往对话里塞文本）
  D. 工具任务 + 正常输出                ⇒ **不告警**（零行为变更）
  E. 兼容旧字段 `_has_tools`
  F. 端到端：真 Agent + ISN 桩化 + 工具问句 ⇒ 返回值不得为空
"""
from types import SimpleNamespace

import pytest

from openllm.core.models import ActionResult, Decision
from openllm.iax.agent_heartbeat import _guard_blank_tool_output

MARK = "【工具任务无产出"


class _TurnStub:
    """最小 Turn 替身：只记 trace_phase / add_action（签名与真 Turn 一致）。"""

    def __init__(self):
        self.phase_metrics = []
        self.actions = []

    def trace_phase(self, phase, status, duration_ms=0.0, detail=""):
        self.phase_metrics.append({"phase": phase, "status": status,
                                   "duration_ms": duration_ms, "detail": detail})

    def add_action(self, action_type, detail="", **kw):
        self.actions.append({"type": action_type, "detail": detail, **kw})


class _IKOStub:
    def __init__(self):
        self.traces = []

    def trace(self, phase, status, duration_ms=0.0, detail=""):
        self.traces.append({"phase": phase, "status": status, "detail": detail})


class _ISAStub:
    mode = "silent"

    def respond(self, text):  # pragma: no cover - 静默模式不打印
        pass


def _agent(output=""):
    return SimpleNamespace(_last_output=output, iko=_IKOStub(), isa=_ISAStub())


def _hc(tool_calls=None, result=None, has_tools=False):
    decision = Decision(approved=True, action="read_file", reason="",
                        tool_calls=tool_calls or [])
    if has_tools:
        decision._has_tools = True
    return SimpleNamespace(decision=decision, result=result, output="")


# ── A. 工具任务 + 空输出 ⇒ 必告警 ────────────────────────────

def test_tool_task_blank_output_alarms(caplog):
    agent = _agent("")
    turn = _TurnStub()
    hc = _hc(tool_calls=[{"name": "read_file", "args": {"path": "x"}}],
             result=ActionResult(success=False, output=""))

    with caplog.at_level("ERROR", logger="openllm.iax.agent_heartbeat"):
        alarm = _guard_blank_tool_output(agent, hc, turn)

    assert alarm and MARK in alarm, "工具任务空输出必须产出显式告警文本"
    assert "read_file" in alarm, "告警必须点名是哪个工具"
    assert agent._last_output == alarm, "必须写回 _last_output，不许静默白卷"
    assert hc.output == alarm, "hc.output 同步，交互模式屏幕才不会空白"
    assert hc.tool_output_alarm["kind"] == "工具未产出"
    assert any("empty_output" == t["status"] for t in agent.iko.traces), "IKO 必须留痕"
    assert any(r.levelname == "ERROR" for r in caplog.records), "必须打 ERROR 日志"
    assert any(p["status"] == "empty_output" for p in turn.phase_metrics), "Turn 必须记痕"


# ── B. 工具成功但没送达 ⇒ 同样告警，子型为「送达断链」 ────────

def test_tool_success_but_blank_final_output_alarms():
    agent = _agent("")                       # ← 最终输出为空
    turn = _TurnStub()
    hc = _hc(tool_calls=[{"name": "read_file", "args": {}}],
             result=ActionResult(success=True, output="文件内容在此"))

    alarm = _guard_blank_tool_output(agent, hc, turn)

    assert alarm and MARK in alarm
    assert hc.tool_output_alarm["kind"] == "送达断链", "工具成功却没送达，子型必须区分"
    assert agent._last_output == alarm


# ── C. 非工具任务 ⇒ 不告警，也不许改输出 ──────────────────────

def test_non_tool_blank_output_does_not_alarm():
    agent = _agent("")
    turn = _TurnStub()
    hc = _hc(tool_calls=[], result=ActionResult(success=False, output=""))

    assert _guard_blank_tool_output(agent, hc, turn) is None
    assert agent._last_output == "", "非工具任务不得被塞入工具告警文本"
    assert not hasattr(hc, "tool_output_alarm")


# ── D. 工具任务 + 正常输出 ⇒ 不告警（零行为变更） ─────────────

def test_tool_task_with_output_no_alarm():
    agent = _agent("读到了：[build-system]")
    turn = _TurnStub()
    hc = _hc(tool_calls=[{"name": "read_file", "args": {}}],
             result=ActionResult(success=True, output="[build-system]"))

    assert _guard_blank_tool_output(agent, hc, turn) is None
    assert agent._last_output == "读到了：[build-system]"


# ── E. 兼容旧字段 `_has_tools` ───────────────────────────────

def test_has_tools_flag_without_tool_calls_still_guarded():
    agent = _agent("")
    hc = _hc(tool_calls=[], result=ActionResult(success=False, output=""), has_tools=True)

    alarm = _guard_blank_tool_output(agent, hc, _TurnStub())
    assert alarm and MARK in alarm, "旧字段路径不得漏闸"


# ── E2. 已被明确拦截的 ⇒ 不重复告警（拦截处已给过文案） ────────

def test_rejected_tool_task_not_double_reported():
    agent = _agent("")
    hc = _hc(tool_calls=[{"name": "read_file", "args": {}}],
             result=ActionResult(success=False, output=""))
    hc.rejection_record = {"reason": "cap_policy拒绝", "phase": "execute"}

    assert _guard_blank_tool_output(agent, hc, _TurnStub()) is None
    assert agent._last_output == "", "已拦截的任务不得被重复改写成告警"


# ── E3. 用**真** HeartbeatContext 跑：只许写声明过的字段（不许隐式 setattr） ──

def test_guard_writes_only_declared_protocol_fields():
    """core/protocol.py 明令「协议字段显式声明——不许隐式 setattr」。

    本钉用真 HeartbeatContext（不是替身）+ 真 Turn，检查告警只落在声明字段上。
    """
    from dataclasses import fields as dc_fields

    from openllm.core.protocol import HeartbeatContext
    from openllm.core.session import Turn

    declared = {f.name for f in dc_fields(HeartbeatContext)}
    assert "tool_output_alarm" in declared, "加固④ 的告警字段必须在协议里显式声明"
    assert HeartbeatContext().tool_output_alarm is None, "默认值必须是 None（未触发）"

    hc = HeartbeatContext(user_message="读文件")
    hc.decision = Decision(approved=True, action="read_file", reason="",
                           tool_calls=[{"name": "read_file", "args": {}}])
    hc.result = ActionResult(success=False, output="")
    turn = Turn(id="t-alarm", session=None, token_budget=100)
    turn_before = set(vars(turn))

    alarm = _guard_blank_tool_output(_agent(""), hc, turn)

    assert alarm and MARK in alarm
    assert hc.tool_output_alarm["tools"] == "read_file"
    hidden = set(vars(hc)) - declared - {"__dict__"}
    assert not hidden, f"隐式 setattr 了未声明字段：{hidden}"
    assert set(vars(turn)) == turn_before, "Turn 上不得凭空空降新属性"
    assert any(a["type"] == "tool_empty_output" for a in turn.actions), "留痕走 add_action"


# ── F. 整条心跳链路的端到端：工具任务 + ISN 空产出 ⇒ 必返回告警 ──
#
# 说明：这里**不依赖真实模型**（沙盒内 provider 不可用时走模拟 LLM）。
# 做法是把「仲裁」这一个缝隙换成一份带 tool_calls 的 Decision，
# 其余 PERCEIVE→DECIDE→EXECUTE→LEARN→FEEDBACK 全是真链路——
# 这样复现的正是实测缺陷（工具任务 + ISN 无产出 → 空串）。

def test_end_to_end_blank_tool_output_alarms():
    """摘掉 ISN ⇒ 工具问句必须有**可读告警**，而不是 0 字白卷（实测复现点）。"""
    from openllm.core.main_loop import Agent
    from openllm.core.models import Decision

    a = Agent(mode="silent")
    a.ios.cap_check = lambda *ar, **kw: True                    # 权限放行，聚焦被测路径
    a.isn.execute = lambda *ar, **kw: ActionResult(success=False, output="")
    a.ios.arbitrate = lambda proposal, critique, risk=None: Decision(
        approved=True, action="read_file", reason="",
        tool_calls=[{"name": "read_file", "args": {"path": "pyproject.toml"}}],
    )
    try:
        out = a.run_once("读一下 pyproject.toml 的前两行")
    finally:
        a.shutdown()

    assert out and out.strip(), "工具任务不许交白卷（实测就是这里返回了空串）"
    assert MARK in out, "空输出必须变成显式告警文本，而非空串"
