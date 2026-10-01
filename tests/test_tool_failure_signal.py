"""加固⑤·第一步 钉子：工具失败信号——**只观测，零行为变更**。

依据（真库实证，同一形态犯两次）：`RECALL.jsonl` 第 170 行（10-01 03:28）与第 188 行
（10-01 21:10）都记 `status="ok"`，内容却是 `/bin/sh: 1: …: not found` + `[exit_code=127]`。
根因链：`tools/executor.py` 只在**文本**里附退出码（不抛异常）→ `isn_impl` 见无异常即
`success=True` → 唯一解析点 `validate_tool_result` 只认 `[错误]`/`[拦截]`
⇒ 命令失败被记成成功，且加固④（判据 `result.success`）对它不响。

本文件的钉子用**真库那两条真串**当 fixture（不是编的），并**专门验"零行为变更"**。
"""
from dataclasses import fields as dc_fields
from types import SimpleNamespace

from openllm.core.models import ActionResult, Decision
from openllm.core.tool_validator_types import (detect_tool_failure,
                                               looks_like_non_command)
from openllm.iax.agent_heartbeat import _guard_tool_failure_signal

# ── 真库第 170 行（10-01 03:28）的真实文本 ──
REAL_170_SUMMARY = ("\n[stderr]\n/bin/sh: 1: 线路：宜宾→成都: not found\n"
                    "/bin/sh: 2: 车型：蓝色东风天龙: not found\n"
                    "/bin/sh: 3: 车牌：川Q12345: not found\n\n[exit_code=127]")
REAL_170_COMMAND = "线路：宜宾→成都\n车型：蓝色东风天龙\n车牌：川Q12345"

# ── 真库第 188 行（10-01 21:10）的真实文本 ──
REAL_188_SUMMARY = ("\n[stderr]\n/bin/sh: 1: │: not found\n/bin/sh: 2: │: not found\n"
                    "/bin/sh: 3: │: not found\n/bin/sh: 4: │: not found\n"
                    "/bin/sh: 5: └─: not found\n\n[exit_code=127]")
REAL_188_COMMAND = "│ 线路：宜宾→成都\n│ 车辆：蓝色东风天龙，车牌川Q12345\n│ 车况：正常\n│ 状态：自检完成\n└─"


class _TurnStub:
    def __init__(self):
        self.phase_metrics = []
        self.actions = []

    def trace_phase(self, phase, status, duration_ms=0.0, detail=""):
        self.phase_metrics.append({"phase": phase, "status": status, "detail": detail})

    def add_action(self, action_type, detail="", **kw):
        self.actions.append({"type": action_type, "detail": detail, **kw})


class _IKOStub:
    def __init__(self):
        self.traces = []

    def trace(self, phase, status, duration_ms=0.0, detail=""):
        self.traces.append({"phase": phase, "status": status, "detail": detail})


# ── A. 失败标记识别（用真库真串） ─────────────────────────────

def test_real_line_170_is_recognised_as_failure():
    f = detect_tool_failure(REAL_170_SUMMARY)
    assert f and f["kind"] == "exit_code" and f["exit_code"] == 127


def test_real_line_188_is_recognised_as_failure():
    f = detect_tool_failure(REAL_188_SUMMARY)
    assert f and f["kind"] == "exit_code" and f["exit_code"] == 127


def test_timeout_error_blocked_and_clean():
    assert detect_tool_failure("[超时] 命令执行超过30秒")["kind"] == "timeout"
    assert detect_tool_failure("[错误] 写入失败: x")["kind"] == "error"
    assert detect_tool_failure("[拦截] 越权")["kind"] == "blocked"
    assert detect_tool_failure("[exit_code=0]") is None, "退出码 0 不是失败"
    assert detect_tool_failure("[build-system]\nrequires=[\"setuptools\"]") is None, "正常输出不得误报"
    assert detect_tool_failure("") is None


def test_exit_code_wins_over_error_marker():
    """同一条文本多种标记时，退出码优先（有退出码就有明确事实）。"""
    f = detect_tool_failure("[错误] 失败\n[exit_code=2]")
    assert f["kind"] == "exit_code" and f["exit_code"] == 2


# ── B. 输入形态检查（治理"把展示块当命令执行"） ────────────────

def test_real_188_command_is_flagged_as_display_block():
    why = looks_like_non_command(REAL_188_COMMAND)
    assert why and "框线" in why


def test_real_170_command_is_flagged_as_display_text():
    why = looks_like_non_command(REAL_170_COMMAND)
    assert why and "非 ASCII" in why


def test_legit_commands_not_flagged():
    for cmd in ("ls -la", "git status --short", "grep -rn '中文' f.py",
                "printf 'a\\nb' > /tmp/f", "python -c \"print('中文')\""):
        assert looks_like_non_command(cmd) is None, f"误报: {cmd}"
    assert looks_like_non_command("") == "空命令"


# ── C. 观测闸：置位 + 留痕，且**零行为变更** ─────────────────

def _agent(output=""):
    return SimpleNamespace(_last_output=output, iko=_IKOStub(),
                           isa=SimpleNamespace(mode="silent"))


def _hc(output="", success=True, command=REAL_188_COMMAND, tool="shell"):
    hc = SimpleNamespace()
    hc.result = ActionResult(success=success, output=output)
    hc.output = output
    # 替身也要照抄真协议的默认值（真 HeartbeatContext 两个字段默认 None）
    hc.tool_failure_signal = None
    hc.tool_output_alarm = None
    hc.decision = Decision(approved=True, action="shell", reason="",
                           tool_calls=[{"name": tool, "args": {"command": command}}])
    return hc


def test_signal_set_and_zero_behaviour_change(caplog):
    """核心断言：置信号 + 打 ERROR + 留痕，**且一个字不改**（这是第一步的全部意义）。"""
    agent = _agent("原始工具输出")
    turn = _TurnStub()
    hc = _hc(output=REAL_188_SUMMARY)

    with caplog.at_level("ERROR", logger="openllm.iax.agent_heartbeat"):
        sig = _guard_tool_failure_signal(agent, hc, turn)

    assert sig and sig["kind"] == "exit_code" and sig["exit_code"] == 127
    assert "框线" in (sig.get("input_shape") or ""), "输入形态与失败标记应并存"
    assert sig["tools"] == ["shell"]

    # 信号置位 + 留痕
    assert hc.tool_failure_signal is sig
    assert any(a["type"] == "tool_failure_signal" for a in turn.actions)
    assert any(p["status"] == "failure_signal" for p in turn.phase_metrics)
    assert any(t["status"] == "failure_signal" for t in agent.iko.traces)
    assert any(r.levelname == "ERROR" for r in caplog.records)

    # ★ 零行为变更：输出、success 都不许动
    assert agent._last_output == "原始工具输出", "第一步不得改输出"
    assert hc.output == REAL_188_SUMMARY, "第一步不得改 hc.output"
    assert hc.result.success is True, "第一步不得改 success（第二步议题）"


def test_clean_result_produces_no_signal():
    agent, turn = _agent("正常输出"), _TurnStub()
    hc = _hc(output="[build-system]\nrequires=[\"setuptools\"]", command="cat pyproject.toml")

    assert _guard_tool_failure_signal(agent, hc, turn) is None
    assert hc.tool_failure_signal is None, "未触发时字段保持 None"
    assert turn.actions == []


def test_suspicious_command_alone_still_signals():
    """命令形态可疑，但工具本身"成功"——也要报（这是 21:10 事故的输入侧）。"""
    agent, turn = _agent(""), _TurnStub()
    hc = _hc(output="", success=True, command=REAL_188_COMMAND)

    sig = _guard_tool_failure_signal(agent, hc, turn)
    assert sig and sig["kind"] == "suspicious_command"
    assert sig["source"] == "input_shape"


# ── D. 真协议对象：只写声明过的字段（严禁隐式 setattr） ────────

def test_declared_field_and_no_hidden_setattr():
    from openllm.core.protocol import HeartbeatContext
    from openllm.core.session import Turn

    declared = {f.name for f in dc_fields(HeartbeatContext)}
    assert "tool_failure_signal" in declared, "跨体字段必须先在协议里声明"
    assert HeartbeatContext().tool_failure_signal is None

    hc = HeartbeatContext(user_message="跑个命令")
    hc.decision = Decision(approved=True, action="shell", reason="",
                           tool_calls=[{"name": "shell", "args": {"command": REAL_188_COMMAND}}])
    hc.result = ActionResult(success=True, output=REAL_188_SUMMARY)
    turn = Turn(id="t-fail-sig", session=None, token_budget=100)
    before = set(vars(turn))

    _guard_tool_failure_signal(_agent(""), hc, turn)

    hidden = set(vars(hc)) - declared
    assert not hidden, f"隐式 setattr 了未声明字段：{hidden}"
    assert set(vars(turn)) == before, "Turn 上不得凭空空降新属性"


# ── E. 整条心跳链路：只观测、不改交付 ────────────────────────

def test_end_to_end_observes_without_changing_answer():
    """真 Agent 全链路：ISN 报"成功"但退出码 127 ⇒ **不注入任何告警文本**，信号已留痕。

    注：交付物可能与裸工具输出不同——`_synthesize` 合法地会把工具输出交给模型再消化
    （实测批量跑时就是这样，答案甚至正确说出了"命令执行失败"）。故**不断言输出内容**，
    只断言"观测闸没往交付物里塞东西" + 信号留痕——这才是"零行为变更"的可判形式。
    """
    from openllm.core.main_loop import Agent

    a = Agent(mode="silent")
    a.ios.cap_check = lambda *ar, **kw: True
    a.ios.arbitrate = lambda proposal, critique, risk=None: Decision(
        approved=True, action="shell", reason="",
        tool_calls=[{"name": "shell", "args": {"command": REAL_170_COMMAND}}],
    )
    a.isn.execute = lambda *ar, **kw: ActionResult(success=True, output=REAL_170_SUMMARY)
    try:
        out = a.run_once("跑一下")
        turn = a.session.turns[-1]
    finally:
        a.shutdown()

    assert out and out.strip(), "失败任务也必须有可读输出"
    assert "工具失败信号" not in out, "观测闸不得把自己的告警注入交付物"
    assert "【工具任务无产出" not in out, "加固④ 在本路径不应触发（success=True 且有输出）"
    assert any(x["type"] == "tool_failure_signal" for x in turn.actions), "信号必须留痕"
