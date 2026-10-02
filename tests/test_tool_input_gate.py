"""加固⑤·第二步 2-pre 钉子：工具输入侧闸——非命令文本**执行前拦截**。

病（真库实证，同一形态犯两次，见 test_tool_failure_signal.py 头注）：展示块/中文标签
被喂进 shell，产生 `/bin/sh: N: …: not found` + `[exit_code=127]`，却被记成成功。
第一步只观测；本步在执行前拦下：`looks_like_non_command`（判据一字未改）给出原因
→ 本 turn 不执行，退回显式文案。

逐条验（本文件）：
  A. 两种真库形态（框线多行块 / 首词非 ASCII 含换行）→ 拦，且处置齐全
     （_last_output/hc.output 文案、hc.tool_input_reject、失败 ActionResult、
      turn 留痕并 complete、ledger 记 source=input_reject）
  B. 正常命令（ls -la / grep 中文 file）→ 不拦（返回 None，一切照旧）
  C. 非 shell 工具（read_file）哪怕内容像展示块 → 不拦
  D. 协议字段已显式声明，不隐式 setattr
  E. 端到端：真 Agent + ISN 桩 → 不执行、交付物含退回文案
"""
import json
from dataclasses import fields as dc_fields
from types import SimpleNamespace

import pytest

from openllm.core.models import ActionResult, Decision
from openllm.core.tool_failure_ledger import LEDGER_PATH
from openllm.iax.agent_heartbeat import _reject_non_command_tool_inputs

# ── 真库两条真命令（与 test_tool_failure_signal.py 同源，不另编） ──
REAL_170_COMMAND = "线路：宜宾→成都\n车型：蓝色东风天龙\n车牌：川Q12345"
REAL_188_COMMAND = ("│ 线路：宜宾→成都\n│ 车辆：蓝色东风天龙，车牌川Q12345\n"
                    "│ 车况：正常\n│ 状态：自检完成\n└─")


class _TurnStub:
    def __init__(self, id="t-input-gate"):
        self.id = id
        self.phase_metrics = []
        self.actions = []
        self.completed = False

    def trace_phase(self, phase, status, duration_ms=0.0, detail=""):
        self.phase_metrics.append({"phase": phase, "status": status,
                                   "duration_ms": duration_ms, "detail": detail})

    def add_action(self, action_type, detail="", **kw):
        self.actions.append({"type": action_type, "detail": detail, **kw})

    def complete(self):
        self.completed = True


class _IKOStub:
    def __init__(self):
        self.traces = []

    def trace(self, phase, status, duration_ms=0.0, detail=""):
        self.traces.append({"phase": phase, "status": status, "detail": detail})


def _agent(session_id="s-gate"):
    return SimpleNamespace(_last_output="", iko=_IKOStub(),
                           session=SimpleNamespace(id=session_id))


def _hc(command, tool="shell"):
    hc = SimpleNamespace()
    hc.output = ""
    hc.result = None
    hc.tool_input_reject = None
    hc.rejection_record = None
    hc.decision = Decision(approved=True, action=tool, reason="",
                           tool_calls=[{"name": tool, "args": {"command": command}}])
    return hc


def _read_last_ledger_line():
    if not LEDGER_PATH.exists():
        return None
    lines = [l for l in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


# ── A. 两种真库形态：拦，且处置齐全 ──────────────────────────

@pytest.mark.parametrize("command,expect", [
    (REAL_188_COMMAND, "框线"),
    (REAL_170_COMMAND, "非 ASCII"),
])
def test_display_block_rejected_before_execute(command, expect):
    agent, turn = _agent(), _TurnStub()
    hc = _hc(command)

    sig = _reject_non_command_tool_inputs(agent, hc, turn)

    assert sig is not None and expect in sig["reason"]
    assert sig["tool"] == "shell" and sig["action"] == "rejected_before_execute"
    # 处置：显式退回文案（不许静默）
    assert "[工具输入拦截]" in agent._last_output and "请改写为真正的命令" in agent._last_output
    assert hc.output == agent._last_output
    # 处置：协议字段 + 失败结果（不是 None——None 会让 _learn 崩，见函数 docstring）
    assert hc.tool_input_reject is sig
    assert isinstance(hc.result, ActionResult) and hc.result.success is False
    # 处置：留痕 + turn 终结
    assert any(a["type"] == "tool_input_reject" for a in turn.actions)
    assert any(p["status"] == "input_rejected" for p in turn.phase_metrics)
    assert any(t["status"] == "input_rejected" for t in agent.iko.traces)
    assert turn.completed is True
    # 处置：ledger 记一行 source=input_reject
    row = _read_last_ledger_line()
    assert row and row["source"] == "input_reject" and row["tool"] == "shell"
    assert row["kind"] == "input_rejected" and row["turn_id"] == turn.id


# ── B. 正常命令：不拦 ────────────────────────────────────────

@pytest.mark.parametrize("command", ["ls -la", "grep -rn '中文' f.py"])
def test_legit_commands_pass_through(command):
    agent, turn = _agent(), _TurnStub()
    hc = _hc(command)

    assert _reject_non_command_tool_inputs(agent, hc, turn) is None
    assert hc.tool_input_reject is None and hc.result is None
    assert turn.completed is False and turn.actions == []
    assert agent._last_output == "" and hc.output == ""


def test_non_shell_tool_never_gated():
    """read_file 拿到展示块样内容也不归本闸管（判据只针对 shell 类工具名）。"""
    agent, turn = _agent(), _TurnStub()
    hc = _hc(REAL_188_COMMAND, tool="read_file")
    assert _reject_non_command_tool_inputs(agent, hc, turn) is None


# ── D. 协议字段显式声明 + 不隐式 setattr ─────────────────────

def test_protocol_field_declared():
    from openllm.core.protocol import HeartbeatContext

    declared = {f.name for f in dc_fields(HeartbeatContext)}
    assert "tool_input_reject" in declared, "跨体字段必须先在协议里声明"
    assert HeartbeatContext().tool_input_reject is None

    hc = HeartbeatContext(user_message="x")
    hc.decision = Decision(approved=True, action="shell", reason="",
                           tool_calls=[{"name": "shell", "args": {"command": REAL_170_COMMAND}}])
    _reject_non_command_tool_inputs(_agent(), hc, _TurnStub())
    hidden = set(vars(hc)) - declared
    assert not hidden, f"隐式 setattr 了未声明字段：{hidden}"


# ── E. 端到端：真 Agent 全链路，ISN 不得被调用 ───────────────

def test_end_to_end_rejects_without_executing():
    from openllm.core.main_loop import Agent

    a = Agent(mode="silent")
    executed = {"called": False}

    def _fake_execute(*ar, **kw):
        executed["called"] = True
        return ActionResult(success=True, output="SHOULD NOT RUN")

    a.ios.cap_check = lambda *ar, **kw: True
    a.ios.arbitrate = lambda proposal, critique, risk=None: Decision(
        approved=True, action="shell", reason="",
        tool_calls=[{"name": "shell", "args": {"command": REAL_188_COMMAND}}],
    )
    a.isn.execute = _fake_execute
    try:
        out = a.run_once("把这段贴进终端")
    finally:
        a.shutdown()

    assert executed["called"] is False, "输入闸必须拦在执行之前——ISN 不得被调用"
    assert out and "[工具输入拦截]" in out, f"交付物必须带退回文案，实际: {out!r}"
    turn = a.session.turns[-1]
    assert any(x["type"] == "tool_input_reject" for x in turn.actions)
