"""加固⑤·第二步 2a 钉子：工具失败仪表——拆早退 + validator 认标记(WARN) + 可计数账。

三件套逐条验（本文件）：
  B② validator 认标记：detect_tool_failure 命中（exit_code 非 0 / 超时）
      → overall=WARN、check 含 ("failure_marker","warn")、**should_block=False**。
      红线：[错误]/[拦截] → FAIL+block 一字未动；空结果 → WARN+retry 一字未动。
  B① 拆早退：result.success=False 也进验证（validator 才看得见失败标记），且不崩。
  B③ 账：record() 落一行 JSON，字段齐全；写失败**绝不抛异常**；
      `_guard_tool_failure_signal`（第一步观测闸）已接到 record(source=marker/input_shape)。

行为中性是 B 的立意：validator 只 WARN 不拦（能拦=2b，下一轮）；账只是多一个落点。
"""
import json
from types import SimpleNamespace

from openllm.core.models import ActionResult, Decision
from openllm.core.tool_failure_ledger import LEDGER_PATH, record
from openllm.core.tool_validator_types import (ValidationResult, ToolCall,
                                               detect_tool_failure,
                                               validate_tool_result)
from openllm.iax.agent_heartbeat import (_guard_tool_failure_signal,
                                         _handle_tool_validation)

# 真库串（与第一步同源）
REAL_170_SUMMARY = ("\n[stderr]\n/bin/sh: 1: 线路：宜宾→成都: not found\n"
                    "\n[exit_code=127]")
REAL_170_COMMAND = "线路：宜宾→成都\n车型：蓝色东风天龙\n车牌：川Q12345"


class _TurnStub:
    def __init__(self, id="t-2a"):
        self.id = id
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


def _agent():
    return SimpleNamespace(_last_output="", iko=_IKOStub(),
                           _tool_failures=[],
                           session=SimpleNamespace(id="s-2a"),
                           isa=SimpleNamespace(mode="silent", respond=lambda t: None))


def _call(result_text, tool="shell"):
    return ToolCall(tool_name=tool, tool_type="execute", params={},
                    result=result_text, duration_ms=1.0,
                    session_id="s", turn_id="t")


def _read_ledger_lines():
    if not LEDGER_PATH.exists():
        return []
    return [json.loads(l) for l in LEDGER_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]


# ── B② validator 认标记：WARN 且不拦 ─────────────────────────

def test_exit_code_marker_warns_without_blocking():
    rep = validate_tool_result(_call(REAL_170_SUMMARY))
    assert rep.overall is ValidationResult.WARN
    assert ("failure_marker", "warn") in rep.checks
    assert rep.should_block is False, "本次红线：能拦=行为变更，属下一轮(2b)"


def test_timeout_marker_warns_without_blocking():
    rep = validate_tool_result(_call("命令跑了很久\n[超时] 命令执行超过30秒"))
    assert rep.overall is ValidationResult.WARN
    assert rep.should_block is False


def test_error_blocked_branch_unchanged():
    """红线：[错误]/[拦截] → FAIL+block 的行为一字未动。"""
    for text in ("[错误] 写入失败: x", "[拦截] 越权"):
        rep = validate_tool_result(_call(text))
        assert rep.overall is ValidationResult.FAIL
        assert rep.should_block is True
        assert rep.audit_entry is not None


def test_clean_and_empty_branches_unchanged():
    rep = validate_tool_result(_call("正常输出\n[exit_code=0]"))
    assert rep.overall is ValidationResult.PASS and not rep.should_block
    rep = validate_tool_result(_call(""))
    assert rep.overall is ValidationResult.WARN and rep.should_retry


# ── B① 拆早退：失败结果进验证且不崩 ─────────────────────────

def test_failed_result_now_enters_validation():
    agent, turn = _agent(), _TurnStub()
    decision = Decision(approved=True, action="shell", reason="",
                        tool_calls=[{"name": "shell", "args": {"command": "false"}}])
    result = ActionResult(success=False, output=REAL_170_SUMMARY, duration_ms=2.0)

    _handle_tool_validation(agent, decision, result, turn)  # 不许抛

    validate_traces = [p for p in turn.phase_metrics if p["phase"] == "validate"]
    assert validate_traces, "失败结果必须进验证（拆早退的全部意义）"
    assert validate_traces[0]["status"] == "warn", \
        f"失败标记应判 WARN，实际: {validate_traces[0]}"
    assert not any(p["status"] == "skip" for p in validate_traces), "不得靠异常兜底跳过"


def test_validation_still_no_block_on_marker():
    """拆早退后，失败标记仍不触发拦截分支（_tool_failures 保持空、无 blocked trace）。"""
    agent, turn = _agent(), _TurnStub()
    decision = Decision(approved=True, action="shell", reason="",
                        tool_calls=[{"name": "shell", "args": {"command": "false"}}])
    result = ActionResult(success=False, output=REAL_170_SUMMARY, duration_ms=2.0)

    _handle_tool_validation(agent, decision, result, turn)

    assert agent._tool_failures == []
    assert agent._last_output == ""
    assert not any(t["status"] == "blocked" for t in agent.iko.traces)


# ── B③ 可计数账 ──────────────────────────────────────────────

def test_record_writes_full_row(tmp_path, monkeypatch):
    import openllm.core.tool_failure_ledger as ledger_mod
    target = tmp_path / "guard" / "tool_failure_ledger.jsonl"
    monkeypatch.setattr(ledger_mod, "LEDGER_PATH", target)

    record({"tool": ["shell"], "kind": "exit_code", "exit_code": 127,
            "source": "marker", "input_shape": "含框线字符"},
           session_id="sess-1", turn_id="turn-1")

    row = json.loads(target.read_text(encoding="utf-8").splitlines()[-1])
    assert isinstance(row["ts"], float)
    assert row["tool"] == ["shell"] and row["kind"] == "exit_code"
    assert row["exit_code"] == 127 and row["input_shape"] == "含框线字符"
    assert row["source"] == "marker"
    assert row["session_id"] == "sess-1" and row["turn_id"] == "turn-1"


def test_record_never_raises(tmp_path, monkeypatch):
    """仪表铁律：落盘失败必须静默——账记不上只是瞎眼，顶进心跳是新病。"""
    import openllm.core.tool_failure_ledger as ledger_mod
    # 指向一个不可写的路径（父节点是普通文件 → mkdir 必炸）
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setattr(ledger_mod, "LEDGER_PATH",
                        blocker / "sub" / "tool_failure_ledger.jsonl")

    record({"kind": "exit_code", "source": "marker"}, session_id="", turn_id="")


def test_guard_signal_appends_marker_row():
    """第一步观测闸接线到账：detect 命中 → 账上多一行 source=marker。"""
    before = len(_read_ledger_lines())
    agent, turn = _agent(), _TurnStub()
    hc = SimpleNamespace()
    hc.result = ActionResult(success=True, output=REAL_170_SUMMARY)
    hc.tool_failure_signal = None
    hc.decision = Decision(approved=True, action="shell", reason="",
                           tool_calls=[{"name": "shell",
                                        "args": {"command": REAL_170_COMMAND}}])

    sig = _guard_tool_failure_signal(agent, hc, turn)

    assert sig and sig["kind"] == "exit_code"
    rows = _read_ledger_lines()
    assert len(rows) == before + 1, "观测闸必须落账恰一行"
    row = rows[-1]
    assert row["source"] == "marker" and row["kind"] == "exit_code"
    assert row["exit_code"] == 127
    assert row["input_shape"], "输入形态与标记并存时应同记"
    assert row["tool"] == ["shell"] and row["turn_id"] == turn.id
