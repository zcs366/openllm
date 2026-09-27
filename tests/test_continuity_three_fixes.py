"""会话连续性三修验收测试（刀⑥空轮重试+墓碑 / 刀⑦窗口扩容 / 刀⑧轨迹入账）

图纸: 20260926 openLLM会话连续性三修 §五 T1-T5（T6=全量回归另跑）。
全部替身，不建真 Agent、不碰网络。CLI 断言复用 test_cli_p06 的 stub 模式。
"""
import re
import sys
import importlib
from types import SimpleNamespace

import pytest


# ── CLI stub（同 test_cli_p06 模式）──

def _stub_fast_path_imports(monkeypatch):
    import types as _t
    fake_isa = _t.ModuleType("openllm.core.isa_impl")
    fake_isa.FULL_TOOLS = []
    fake_isa.BASE_TOOLS = []
    fake_isa.TOOL_DESCRIPTIONS = {}
    monkeypatch.setitem(sys.modules, "openllm.core.isa_impl", fake_isa)

    fake_models = _t.ModuleType("openllm.core.models")
    fake_models.Decision = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "openllm.core.models", fake_models)

    fake_octopus = _t.ModuleType("openllm.iai.octopus")
    fake_octopus._LeftBrain = type("_LB", (), {
        "_TOOLCALL_RE": re.compile(
            r"^\s*\*{0,2}\s*TOOL_CALLS:\s*(\{.*\})\s*\*{0,2}\s*$", re.MULTILINE)
    })()
    monkeypatch.setitem(sys.modules, "openllm.iai.octopus", fake_octopus)

    fake_main_loop = _t.ModuleType("openllm.core.main_loop")
    fake_main_loop.Agent = type("Agent", (), {})
    monkeypatch.setitem(sys.modules, "openllm.core.main_loop", fake_main_loop)


def _stub_main(monkeypatch):
    _stub_fast_path_imports(monkeypatch)
    mod = sys.modules.get("openllm.cli.main")
    if mod is not None:
        importlib.reload(mod)


def _build_shell(provider, monkeypatch, isn_exec=None, extract_fn=None,
                 run_once=None):
    """构造带 stub agent 的 AgentShell（快路径条件: <50字+无触发词）。"""
    _stub_main(monkeypatch)
    from openllm.cli.main import AgentShell

    shell = AgentShell.__new__(AgentShell)
    _left = SimpleNamespace(provider=provider, _extract_tool_calls=extract_fn)
    _right = SimpleNamespace(provider=provider)
    _octopus = SimpleNamespace(left=_left, right=_right)
    _isn = SimpleNamespace(execute=isn_exec) if isn_exec else SimpleNamespace()
    _session = SimpleNamespace(active_turn=None, turns=[])
    if run_once is not None:
        _agent = SimpleNamespace(octopus=_octopus, isn=_isn, session=_session,
                                 run_once=run_once)
    else:
        _agent = SimpleNamespace(octopus=_octopus, isn=_isn, session=_session)
    shell.agent = _agent
    shell._history = []
    return shell


class _FakeTurn:
    """心跳 turn 替身——带 actions（刀⑧数据源）与 phase_metrics。"""
    def __init__(self, actions=None):
        self.actions = actions or []
        self.phase_metrics = []
        self.status = "ACTIVE"

    def trace_phase(self, phase, status, duration_ms=0.0, detail=""):
        self.phase_metrics.append({
            "phase": phase, "status": status,
            "duration_ms": duration_ms, "detail": detail[:100]})


# ════════════════════════════════════════════════════════════
# T1 空轮墓碑（刀⑥-2）
# ════════════════════════════════════════════════════════════

class TestT1EmptyTombstone:
    def test_two_blank_replies_book_tombstone_pair(self, capsys, monkeypatch):
        """mock provider 两连空 → history 尾部 = {user:问题}+{assistant:TOMBSTONE_EMPTY}。

        '两连空'覆盖快路径 chat 直接空（无 synthesize 环节）——墓碑在 CLI 层兜。
        """
        from openllm.cli.main import TOMBSTONE_EMPTY
        calls = []

        def fake_chat(messages):
            calls.append(messages)
            return ""

        shell = _build_shell(SimpleNamespace(chat=fake_chat), monkeypatch)
        shell.default("今天天气")
        hist = shell._hist()
        assert len(hist) == 2, f"空轮必须落墓碑对，got {hist}"
        assert hist[0] == {"role": "user", "content": "今天天气"}
        assert hist[1] == {"role": "assistant", "content": TOMBSTONE_EMPTY}
        # hist 里不存在字面 "(无输出)"（那是屏幕提示，不是账本内容）
        assert "(无输出)" not in str(hist)
        out = capsys.readouterr().out
        assert "(无输出)" in out  # 屏幕提示保留

    def test_heartbeat_empty_books_tombstone(self, capsys, monkeypatch):
        """心跳路径 run_once 返回空 → 同样落墓碑对（快/慢通吃）。"""
        from openllm.cli.main import TOMBSTONE_EMPTY
        shell = _build_shell(
            SimpleNamespace(chat=lambda m: "unused"),
            monkeypatch,
            run_once=lambda text, history=None: "")
        shell.default("请分析这个方案的可行性并写出完整的执行计划文档来供我参考看看效果")
        hist = shell._hist()
        assert len(hist) == 2
        assert hist[1]["content"] == TOMBSTONE_EMPTY


# ════════════════════════════════════════════════════════════
# T2 重试成功（刀⑥-1 synthesize 层 + CLI 层落账）
# ════════════════════════════════════════════════════════════

class TestT2RetrySuccess:
    def test_synthesize_first_empty_then_answer(self, monkeypatch):
        """_synthesize：首空调 provider.chat 第二次 → '答案X' 成为 synth_output。"""
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)
        from openllm.iax.agent_heartbeat import _synthesize
        from openllm.core.models import Decision, ActionResult

        _chat_calls = []
        _n = [0]

        def _chat(messages):
            _chat_calls.append(messages)
            _n[0] += 1
            return "" if _n[0] == 1 else "答案X"

        provider = SimpleNamespace(chat=_chat, _available=True)
        agent = SimpleNamespace(
            octopus=SimpleNamespace(left=SimpleNamespace(
                provider=provider, _extract_tool_calls=lambda t: [])),
            isn=SimpleNamespace(execute=lambda d: "x"),
            _record_inference=lambda p: None)
        hc = SimpleNamespace(
            user_message="读文件",
            decision=Decision(action="execute", approved=True, reason="t",
                              tool_calls=[{"name": "terminal", "args": {"command": "ls"}}]),
            result=ActionResult(success=True, output="工具结果", duration_ms=1.0),
            synth_output=None)
        turn = _FakeTurn()
        _synthesize(agent, SimpleNamespace(text="读文件"), hc, turn)

        assert len(_chat_calls) == 2, \
            f"首空必须触发一次重试，provider.chat 应恰2次，got {len(_chat_calls)}"
        assert hc.synth_output == "答案X"
        # 重试消息里附了强制表态提示
        retry_msgs = _chat_calls[1]
        assert any("上一次回复为空" in m.get("content", "")
                   for m in retry_msgs if m["role"] == "user")
        # 首空打点保留（事件留痕），且未改写为 retried 终态
        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        first_marks = [p for p in synth_phases if p["status"] == "skip"]
        assert len(first_marks) == 1
        assert first_marks[0]["detail"] == "empty_response"
        # 重试成功后走正常终态打点（ok/warn/error），本轮综合未静默失败
        assert any(p["status"] in ("ok", "warn") for p in synth_phases)

    def test_synthesize_retry_still_empty_marks_retried(self, monkeypatch):
        """重试仍空 → 打点1条 detail=empty_response_retried，hc.synth_output 不写。"""
        monkeypatch.delenv("OPENLLM_HEARTBEAT_SYNTH", raising=False)
        from openllm.iax.agent_heartbeat import _synthesize
        from openllm.core.models import Decision, ActionResult

        _n = [0]

        def _chat(messages):
            _n[0] += 1
            return "   "  # 空白也算空

        provider = SimpleNamespace(chat=_chat, _available=True)
        agent = SimpleNamespace(
            octopus=SimpleNamespace(left=SimpleNamespace(
                provider=provider, _extract_tool_calls=lambda t: [])),
            isn=SimpleNamespace(execute=lambda d: "x"),
            _record_inference=lambda p: None)
        hc = SimpleNamespace(
            user_message="读文件",
            decision=Decision(action="execute", approved=True, reason="t",
                              tool_calls=[{"name": "terminal", "args": {"command": "ls"}}]),
            result=ActionResult(success=True, output="工具结果", duration_ms=1.0),
            synth_output=None)
        turn = _FakeTurn()
        _synthesize(agent, SimpleNamespace(text="读文件"), hc, turn)

        assert _n[0] == 2
        synth_phases = [p for p in turn.phase_metrics if p["phase"] == "synthesize"]
        assert len(synth_phases) == 1, "打点保持1条（原地改写终态，不追加）"
        assert synth_phases[0]["status"] == "skip"
        assert synth_phases[0]["detail"] == "empty_response_retried"
        # 旧钉子兼容：'empty_response' 仍是 'empty_response_retried' 的子串
        assert "empty_response" in synth_phases[0]["detail"]
        assert hc.synth_output is None

    def test_cli_books_answer_not_tombstone(self, capsys, monkeypatch):
        """CLI 层：引擎产出'答案X' → 用户看到、history 记'答案X'非墓碑。"""
        shell = _build_shell(
            SimpleNamespace(chat=lambda m: "unused"),
            monkeypatch,
            run_once=lambda text, history=None: "答案X")
        shell.default("请分析这个方案的可行性并写出完整的执行计划文档来供我参考看看效果")
        out = capsys.readouterr().out
        assert "答案X" in out
        hist = shell._hist()
        assert hist[-1] == {"role": "assistant", "content": "答案X"}
        from openllm.cli.main import TOMBSTONE_EMPTY
        assert TOMBSTONE_EMPTY not in str(hist)


# ════════════════════════════════════════════════════════════
# T3 错误墓碑（刀⑥-2）
# ════════════════════════════════════════════════════════════

class TestT3ErrorTombstone:
    def test_llm_error_books_tombstone_error(self, capsys, monkeypatch):
        """mock 返回 '[LLM错误]timeout' → 用户看到原文；history 记 TOMBSTONE_ERROR 对。"""
        from openllm.cli.main import TOMBSTONE_ERROR
        error_text = "[LLM错误]timeout"
        shell = _build_shell(
            SimpleNamespace(chat=lambda m: error_text), monkeypatch)
        shell.default("hello")
        out = capsys.readouterr().out
        assert "[LLM错误]timeout" in out, f"错误原文照常显示: {out}"
        hist = shell._hist()
        assert len(hist) == 2
        assert hist[0] == {"role": "user", "content": "hello"}
        assert hist[1] == {"role": "assistant", "content": TOMBSTONE_ERROR}
        assert error_text not in hist[1]["content"], "错误原文不得进账本"


# ════════════════════════════════════════════════════════════
# T4 窗口扩容（刀⑦）
# ════════════════════════════════════════════════════════════

class TestT4Window:
    def _preset(self, shell, n=60):
        shell._history = [
            {"role": ("user" if i % 2 == 0 else "assistant"),
             "content": f"m{i:03d}"} for i in range(n)]

    def test_fast_path_window_le_40_and_freshest(self, capsys, monkeypatch):
        """快路径：预置60条 → provider 收到的历史切片 ≤40 且含最新一条。"""
        captured = []

        def fake_chat(messages):
            captured.append(messages)
            return "ok"

        shell = _build_shell(SimpleNamespace(chat=fake_chat), monkeypatch)
        self._preset(shell)
        shell.default("hello")
        msgs = captured[0]
        body = [m for m in msgs if m["role"] != "system"]
        # 最新一条 m059 必须在窗口内
        assert any(m["content"] == "m059" for m in body)
        assert not any(m["content"] == "m019" for m in body), \
            "窗口外的旧条目不得混入"
        hist_slice = body[:-1]  # 末条是本轮 user 输入
        assert len(hist_slice) <= 40, f"history 切片应≤40，got {len(hist_slice)}"
        assert len(hist_slice) == 40, "60条预置应截满40条"

    def test_heartbeat_path_window_le_40_and_freshest(self, capsys, monkeypatch):
        """慢路径：预置60条 → run_once(history=) 收到 ≤40 条且含最新一条。"""
        seen = {}

        def fake_run_once(text, history=None):
            seen["history"] = history
            return "ok"

        shell = _build_shell(
            SimpleNamespace(chat=lambda m: "unused"), monkeypatch,
            run_once=fake_run_once)
        self._preset(shell)
        shell.default("请分析这个方案的可行性并写出完整的执行计划文档来供我参考看看效果")
        h = seen["history"]
        assert len(h) <= 40, f"慢路径 history 应≤40，got {len(h)}"
        assert h[-1]["content"] == "m059", "必须含最新一条"

    def test_window_constant_is_40(self, monkeypatch):
        _stub_main(monkeypatch)
        from openllm.cli.main import HISTORY_WINDOW
        assert HISTORY_WINDOW == 40


# ════════════════════════════════════════════════════════════
# T5 轨迹前缀（刀⑧）
# ════════════════════════════════════════════════════════════

class TestT5ToolTrace:
    def test_fast_path_prefix(self, capsys, monkeypatch):
        """快路径工具回路 → booked 以 '[本轮工具:' 开头，前缀行≤220字符。"""
        _n = [0]

        def fake_chat(messages):
            _n[0] += 1
            if _n[0] == 1:
                return 'searching\nTOOL_CALLS: {"tool_calls": [{"name": "web_search", "args": {"query": "jev laya"}}]}'
            return "最终回答"

        shell = _build_shell(
            SimpleNamespace(chat=fake_chat), monkeypatch,
            isn_exec=lambda d: "搜索结果",
            extract_fn=lambda t: ([{"name": "web_search", "args": {"query": "jev laya"}}]
                                  if "TOOL_CALLS:" in t else []),
        )
        shell.default("hello")
        booked = shell._hist()[-1]["content"]
        assert booked.startswith("[本轮工具:"), booked
        prefix_line = booked.split("\n", 1)[0]
        assert 'web_search("jev laya")' in prefix_line
        assert len(prefix_line) <= 220
        assert booked.split("\n", 1)[1] == "最终回答"
        out = capsys.readouterr().out
        assert "[本轮工具:" in out  # 前缀同时进显示

    def _fake_watcher_cls(self, monkeypatch, turn):
        """替掉 CLI 的 PhaseWatcher——注意两点：
        ① openllm.cli.__init__ re-export 把包属性 main 遮蔽成函数，走 sys.modules 取真模块；
        ② _build_shell 内部会 reload main 模块冲掉补丁，本方法必须在建壳之后调用。"""
        main_mod = sys.modules["openllm.cli.main"]

        class _FakeWatcher:
            def __init__(self, **kw):
                self.last_turn = turn

            def start(self):
                pass

            def stop(self):
                pass

        monkeypatch.setattr(main_mod, "PhaseWatcher", _FakeWatcher)
        monkeypatch.setenv("OPENLLM_ACTIVITY", "1")

    def test_heartbeat_turn_prefix(self, capsys, monkeypatch):
        """心跳 turn 带 tool_calls action → booked 带前缀；turn 无记录 → 无前缀不报错。"""
        turn = _FakeTurn(actions=[
            {"type": "execute", "detail": "out", "timestamp": 0.0,
             "tool_calls": [{"name": "web_fetch", "arg": "https://github.com/jev/laya-repo-a-str"}]}])

        shell = _build_shell(
            SimpleNamespace(chat=lambda m: "unused"), monkeypatch,
            run_once=lambda text, history=None: "心跳回答")
        # watcher 补丁必须在 _build_shell 之后打（其内部 _stub_main 会 reload 冲掉）
        self._fake_watcher_cls(monkeypatch, turn)
        shell.default("请分析这个方案的可行性并写出完整的执行计划文档来供我参考看看效果")
        booked = shell._hist()[-1]["content"]
        assert booked.startswith("[本轮工具:"), booked
        assert len(booked.split("\n", 1)[0]) <= 220
        # arg 截断30字符（正则提取 web_fetch("…") 内的参数部分）
        _m = re.search(r'web_fetch\("([^"]*)"\)', booked.split("\n", 1)[0])
        assert _m, booked
        assert len(_m.group(1)) == 30
        assert booked.endswith("心跳回答")

    def test_no_records_graceful_degrade(self, capsys, monkeypatch):
        """turn 无工具记录 → 无前缀、不报错、正常入账。"""
        shell = _build_shell(
            SimpleNamespace(chat=lambda m: "unused"), monkeypatch,
            run_once=lambda text, history=None: "干净回答")
        self._fake_watcher_cls(monkeypatch, _FakeTurn(actions=[]))
        shell.default("请分析这个方案的可行性并写出完整的执行计划文档来供我参考看看效果")
        assert shell._hist()[-1]["content"] == "干净回答"

    def test_tombstone_pair_no_prefix(self, capsys, monkeypatch):
        """墓碑对不带前缀（图纸刀⑧：摆放规则）。"""
        shell = _build_shell(SimpleNamespace(chat=lambda m: ""), monkeypatch)
        shell.default("hello")
        assert not shell._hist()[-1]["content"].startswith("[本轮工具")


# ════════════════════════════════════════════════════════════
# 刀⑧ 心跳写入端：add_action 携带 tool_calls 摘要
# ════════════════════════════════════════════════════════════

class TestT5HeartbeatWriter:
    def test_add_action_carries_tool_calls(self):
        """session.Turn.add_action 的 kwargs 容器承接 tool_calls（Turn结构零改动）。"""
        from openllm.core.session import Session, TurnStatus
        s = Session()
        s.start()
        t = s.new_turn()
        t.add_action("execute", "out", tool_calls=[{"name": "web_search", "arg": "q"}])
        assert t.actions[-1]["tool_calls"] == [{"name": "web_search", "arg": "q"}]
        # JSON 可序列化（checkpoint 持久化路径不吃哑巴对象）
        import json
        json.dumps(t.actions)
