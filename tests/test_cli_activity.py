"""
Deliverable C — CLI 实时活动流观察器测试。

T1: format_phase 格式化
T2: PhaseWatcher 线程轮询
T3: 端到端 AgentShell 心跳路径
T4: OPENLLM_ACTIVITY=0 开关
T5: turn_account 小账
"""
import sys
import time
from types import SimpleNamespace

import pytest


# ══════════════════════════════════════════════════════════════════
# T1: format_phase 格式化
# ══════════════════════════════════════════════════════════════════

class TestFormatPhase:

    def test_ok_with_duration(self):
        from openllm.cli.activity import format_phase
        entry = {"phase": "context", "status": "ok", "duration_ms": 123.4,
                 "detail": "done"}
        result = format_phase(entry)
        assert "ISA" in result
        assert "建境" in result
        assert "✓" in result
        assert "123ms" in result
        assert "done" in result

    def test_ok_with_long_duration(self):
        from openllm.cli.activity import format_phase
        entry = {"phase": "search", "status": "ok", "duration_ms": 1500.0,
                 "detail": "ok"}
        result = format_phase(entry)
        assert "1.5s" in result

    def test_skip_no_duration(self):
        from openllm.cli.activity import format_phase
        entry = {"phase": "drift", "status": "skip", "duration_ms": 0,
                 "detail": ""}
        result = format_phase(entry)
        assert "·" in result
        assert "ms" not in result
        assert "s" not in result.split("·")[-1]  # no duration suffix

    def test_denied_with_detail(self):
        from openllm.cli.activity import format_phase
        entry = {"phase": "guard_forbidden", "status": "denied",
                 "duration_ms": 5.0,
                 "detail": "policy violation detected"}
        result = format_phase(entry)
        assert "✗" in result
        assert "policy violation detected" in result

    def test_unknown_phase_fallback(self):
        from openllm.cli.activity import format_phase
        entry = {"phase": "mystery_phase", "status": "ok",
                 "duration_ms": 0, "detail": ""}
        result = format_phase(entry)
        assert "?" in result
        assert "mystery_phase" in result

    def test_unknown_status_shows_raw(self):
        from openllm.cli.activity import format_phase
        entry = {"phase": "context", "status": "weird", "duration_ms": 0,
                 "detail": ""}
        result = format_phase(entry)
        assert "weird" in result

    def test_missing_keys_no_crash(self):
        from openllm.cli.activity import format_phase
        result = format_phase({})
        assert "?" in result

    def test_detail_truncated_40_chars(self):
        from openllm.cli.activity import format_phase
        long_detail = "x" * 100
        entry = {"phase": "context", "status": "ok", "duration_ms": 0,
                 "detail": long_detail}
        result = format_phase(entry)
        # detail should be at most 40 chars (after stripping newlines)
        assert "x" * 41 not in result

    def test_newlines_in_detail_stripped(self):
        from openllm.cli.activity import format_phase
        entry = {"phase": "context", "status": "ok", "duration_ms": 0,
                 "detail": "line1\nline2\nline3"}
        result = format_phase(entry)
        assert "\n" not in result
        assert "line1 line2 line3" in result


# ══════════════════════════════════════════════════════════════════
# T2: PhaseWatcher 线程轮询
# ══════════════════════════════════════════════════════════════════

class TestPhaseWatcher:

    def _make_fake_turn(self, turn_id="t1"):
        """创建假 turn 对象，phase_metrics 可动态 append。"""
        return SimpleNamespace(
            id=turn_id,
            phase_metrics=[],
            risk_level="low",
            start_time=time.time(),
            end_time=None,
        )

    def test_captures_appended_entries(self):
        from openllm.cli.activity import PhaseWatcher
        turn = self._make_fake_turn()
        captured = []
        watcher = PhaseWatcher(
            get_turn=lambda: turn,
            on_entry=lambda e: captured.append(e),
            interval=0.1,
        )
        watcher.start()
        time.sleep(0.15)  # let first poll see empty list

        turn.phase_metrics.append({"phase": "context", "status": "ok",
                                   "duration_ms": 10, "detail": "a"})
        time.sleep(0.2)
        turn.phase_metrics.append({"phase": "search", "status": "ok",
                                   "duration_ms": 20, "detail": "b"})
        time.sleep(0.2)
        turn.phase_metrics.append({"phase": "decide", "status": "ok",
                                   "duration_ms": 30, "detail": "c"})
        time.sleep(0.3)
        watcher.stop()

        assert len(captured) == 3
        assert captured[0]["phase"] == "context"
        assert captured[1]["phase"] == "search"
        assert captured[2]["phase"] == "decide"

    def test_drains_last_turn_on_none(self):
        """getter 返回 None 后，最后 turn 的残余条目必须送达。"""
        from openllm.cli.activity import PhaseWatcher
        turn = self._make_fake_turn()
        current = [turn]  # mutable reference
        captured = []
        watcher = PhaseWatcher(
            get_turn=lambda: current[0] if current else None,
            on_entry=lambda e: captured.append(e),
            interval=0.1,
        )
        watcher.start()
        time.sleep(0.15)

        # Append 2 entries while turn is active
        turn.phase_metrics.append({"phase": "context", "status": "ok",
                                   "duration_ms": 0, "detail": ""})
        turn.phase_metrics.append({"phase": "search", "status": "ok",
                                   "duration_ms": 0, "detail": ""})
        time.sleep(0.2)

        # Getter now returns None (turn completed)
        current.clear()
        # Append 1 more after turn becomes None (simulates late append)
        turn.phase_metrics.append({"phase": "decide", "status": "ok",
                                   "duration_ms": 0, "detail": "last"})
        time.sleep(0.3)
        watcher.stop()

        # All 3 must be captured
        assert len(captured) == 3
        assert captured[2]["detail"] == "last"

    def test_error_in_getter_exits_silently(self):
        """getter 抛错时线程静默退出，不崩。"""
        from openllm.cli.activity import PhaseWatcher
        call_count = [0]

        def boom():
            call_count[0] += 1
            if call_count[0] > 2:
                raise RuntimeError("boom")
            return None

        watcher = PhaseWatcher(
            get_turn=boom,
            on_entry=lambda e: None,
            interval=0.05,
        )
        watcher.start()
        time.sleep(0.5)  # let it fail a few times
        watcher.stop()
        # no crash = pass

    def test_different_turns_reset_seen_count(self):
        """Turn 切换后重新开始计数。"""
        from openllm.cli.activity import PhaseWatcher
        turn1 = self._make_fake_turn("t1")
        turn2 = self._make_fake_turn("t2")
        turns = [turn1]
        captured = []
        watcher = PhaseWatcher(
            get_turn=lambda: turns[0],
            on_entry=lambda e: captured.append(e),
            interval=0.1,
        )
        watcher.start()
        time.sleep(0.15)

        # Turn 1: add 2 entries
        turn1.phase_metrics.append({"phase": "context", "status": "ok",
                                    "duration_ms": 0, "detail": ""})
        time.sleep(0.2)
        turn1.phase_metrics.append({"phase": "search", "status": "ok",
                                    "duration_ms": 0, "detail": ""})
        time.sleep(0.2)
        watcher.stop()
        assert len(captured) == 2

        # Switch to turn2
        turns[0] = turn2
        captured.clear()
        watcher2 = PhaseWatcher(
            get_turn=lambda: turns[0],
            on_entry=lambda e: captured.append(e),
            interval=0.1,
        )
        watcher2.start()
        time.sleep(0.15)

        turn2.phase_metrics.append({"phase": "decide", "status": "ok",
                                    "duration_ms": 0, "detail": ""})
        time.sleep(0.2)
        watcher2.stop()
        assert len(captured) == 1
        assert captured[0]["phase"] == "decide"


# ══════════════════════════════════════════════════════════════════
# T3: 端到端 AgentShell 心跳路径
# ══════════════════════════════════════════════════════════════════

class TestAgentShellE2E:
    """E2E tests: 走心跳路径，mock 快路径 import 防崩。"""

    # 心跳路径触发条件：len >= 50 AND 无触发词（搜/写/读/执行/分析）
    _LONG_INPUT = "这是一条足够长的中文测试输入字符串" * 3  # 51 chars, no trigger words

    @staticmethod
    def _ensure_fresh_main():
        """Force reload openllm.cli.main to pick up latest code."""
        import importlib
        mod = sys.modules.get("openllm.cli.main")
        if mod is not None:
            importlib.reload(mod)

    @staticmethod
    def _stub_fast_path_imports(monkeypatch):
        """Mock 快路径所需 import，防止在心跳路径测试中意外崩。"""
        import types
        import re

        # Stub core.main_loop (the deepest dependency of main.py)
        fake_main_loop = types.ModuleType("openllm.core.main_loop")
        fake_main_loop.Agent = type("Agent", (), {})  # type: ignore[attr-defined]
        monkeypatch.setitem(
            __import__("sys").modules, "openllm.core.main_loop", fake_main_loop)

        # Stub isa_impl for fast-path imports
        fake_isa_impl = types.ModuleType("openllm.core.isa_impl")
        fake_isa_impl.FULL_TOOLS = []  # type: ignore[attr-defined]
        fake_isa_impl.BASE_TOOLS = []  # type: ignore[attr-defined]
        fake_isa_impl.TOOL_DESCRIPTIONS = {}  # type: ignore[attr-defined]
        monkeypatch.setitem(
            __import__("sys").modules, "openllm.core.isa_impl", fake_isa_impl)

        # Stub iai.octopus for fast-path tool extraction
        fake_octopus = types.ModuleType("openllm.iai.octopus")
        fake_octopus._LeftBrain = type("_LB", (), {  # type: ignore[attr-defined]
            "_TOOLCALL_RE": re.compile(r"TOOL_CALLS:.*", re.DOTALL)
        })()
        monkeypatch.setitem(
            __import__("sys").modules, "openllm.iai.octopus", fake_octopus)

        # Stub core.models (needed by fast-path Decision import)
        fake_models = types.ModuleType("openllm.core.models")
        for _name in ("Message", "Context", "Prediction", "RiskAssessment",
                       "Proposal", "Critique", "Decision", "ActionResult",
                       "CausalDelta", "TickMetrics"):
            setattr(fake_models, _name, type(_name, (), {}))  # type: ignore[attr-defined]
        monkeypatch.setitem(
            __import__("sys").modules, "openllm.core.models", fake_models)

    def test_heartbeat_shows_activity_flow(self, capsys, monkeypatch):
        """长输入走心跳路径：活动流行出现、最终答案出现、噪声不泄漏。"""
        self._stub_fast_path_imports(monkeypatch)
        self._ensure_fresh_main()
        from openllm.cli.main import AgentShell

        class FakeTurn:
            def __init__(self):
                self.id = "e2e-t1"
                self.phase_metrics = []
                self.risk_level = "low"
                self.start_time = time.time()
                self.end_time = None

            @property
            def duration_ms(self):
                return (time.time() - self.start_time) * 1000

        fake_turn = FakeTurn()

        class FakeSession:
            active_turn = None

        class FakeRunOnce:
            """假 run_once：塞 phase_metrics + 印噪声（必须被封印）。"""
            def __call__(self, line, history=None):
                fake_turn.phase_metrics.append(
                    {"phase": "context", "status": "ok", "duration_ms": 5.0,
                     "detail": "建境ok"})
                fake_turn.phase_metrics.append(
                    {"phase": "decide", "status": "ok", "duration_ms": 10.0,
                     "detail": "仲裁ok"})
                print("NOISE_LEAK_TEST")
                return "最终答案：活动流工作正常"

        class FakeAgent:
            def __init__(self):
                self.session = FakeSession()
                self.run_once = FakeRunOnce()
                self.octopus = SimpleNamespace(
                    left=SimpleNamespace(provider=SimpleNamespace(
                        _available=False, model="test")),
                    right=SimpleNamespace(provider=SimpleNamespace(
                        _available=False, model="test")))

        agent = FakeAgent()
        orig_run = agent.run_once

        def patched_run(*args, **kwargs):
            agent.session.active_turn = fake_turn  # set BEFORE run
            result = orig_run(*args, **kwargs)
            # Keep active_turn set long enough for PhaseWatcher (interval=0.5s)
            import time as _t; _t.sleep(1.0)
            agent.session.active_turn = None
            return result

        agent.run_once = patched_run

        shell = AgentShell.__new__(AgentShell)
        shell.agent = agent
        monkeypatch.delenv("OPENLLM_ACTIVITY", raising=False)

        shell.default(self._LONG_INPUT)
        out = capsys.readouterr().out

        # 活动流行必须出现
        assert "ISA" in out, f"活动流未出现，输出=\n{out}"
        assert "建境" in out
        assert "IOS" in out
        assert "仲裁" in out
        # 最终答案必须出现
        assert "最终答案：活动流工作正常" in out
        # 封印必须有效：噪声不泄漏
        assert "NOISE_LEAK_TEST" not in out, f"封印回归！噪声泄漏={out}"

    def test_heartbeat_turn_account_appears(self, capsys, monkeypatch):
        """心跳路径轮末：turn_account 而非 [{dt:.1f}s]。"""
        self._stub_fast_path_imports(monkeypatch)
        self._ensure_fresh_main()
        from openllm.cli.main import AgentShell

        class FakeTurn:
            def __init__(self):
                self.id = "acct-t1"
                self.phase_metrics = [
                    {"phase": "context", "status": "ok", "duration_ms": 5.0,
                     "detail": "ok"},
                ]
                self.risk_level = "medium"
                self.start_time = time.time() - 1.0
                self.end_time = None

            @property
            def duration_ms(self):
                return (time.time() - self.start_time) * 1000

        fake_turn = FakeTurn()

        class FakeSession:
            active_turn = None

        class FakeAgent:
            def __init__(self):
                self.session = FakeSession()
                self.octopus = SimpleNamespace(
                    left=SimpleNamespace(provider=SimpleNamespace(
                        _available=False, model="test")),
                    right=SimpleNamespace(provider=SimpleNamespace(
                        _available=False, model="test")))

            def run_once(self, line, history=None):
                self.session.active_turn = fake_turn
                result = "ok"
                # Keep active_turn long enough for PhaseWatcher to capture it
                import time as _t; _t.sleep(1.0)
                self.session.active_turn = None
                return result

        agent = FakeAgent()
        shell = AgentShell.__new__(AgentShell)
        shell.agent = agent
        monkeypatch.delenv("OPENLLM_ACTIVITY", raising=False)

        shell.default(self._LONG_INPUT)
        out = capsys.readouterr().out

        # turn_account 输出应包含阶段数和风险级别
        assert "1 阶段" in out, f"turn_account 未出现，输出=\n{out}"
        assert "风险medium" in out


# ══════════════════════════════════════════════════════════════════
# T4: OPENLLM_ACTIVITY=0 开关
# ══════════════════════════════════════════════════════════════════

class TestActivitySwitch:

    def test_activity_off_hides_flow(self, capsys, monkeypatch):
        """OPENLLM_ACTIVITY=0 时活动流不出现，答案照常。"""
        TestAgentShellE2E._stub_fast_path_imports(monkeypatch)
        TestAgentShellE2E._ensure_fresh_main()
        from openllm.cli.main import AgentShell

        class FakeTurn:
            def __init__(self):
                self.id = "sw-t1"
                self.phase_metrics = []
                self.risk_level = "low"
                self.start_time = time.time()
                self.end_time = None

            @property
            def duration_ms(self):
                return (time.time() - self.start_time) * 1000

        fake_turn = FakeTurn()

        class FakeSession:
            active_turn = None

        class FakeAgent:
            def __init__(self):
                self.session = FakeSession()
                self.octopus = SimpleNamespace(
                    left=SimpleNamespace(provider=SimpleNamespace(
                        _available=False, model="test")),
                    right=SimpleNamespace(provider=SimpleNamespace(
                        _available=False, model="test")))

            def run_once(self, line, history=None):
                self.session.active_turn = fake_turn
                fake_turn.phase_metrics.append(
                    {"phase": "context", "status": "ok", "duration_ms": 5.0,
                     "detail": "ok"})
                result = "开关测试答案"
                self.session.active_turn = None
                return result

        agent = FakeAgent()
        shell = AgentShell.__new__(AgentShell)
        shell.agent = agent
        monkeypatch.setenv("OPENLLM_ACTIVITY", "0")

        shell.default(TestAgentShellE2E._LONG_INPUT)
        out = capsys.readouterr().out

        # 活动流不出现
        assert "ISA" not in out, f"OPENLLM_ACTIVITY=0 下活动流仍出现=\n{out}"
        assert "建境" not in out
        # 答案照常
        assert "开关测试答案" in out
        # 回退到 [{dt:.1f}s]
        assert "[" in out  # timing line


# ══════════════════════════════════════════════════════════════════
# T5: turn_account 小账
# ══════════════════════════════════════════════════════════════════

class TestTurnAccount:

    def test_account_with_phases(self):
        from openllm.cli.activity import turn_account
        turn = SimpleNamespace(
            phase_metrics=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            risk_level="low",
            start_time=time.time() - 2.0,
        )
        # Provide duration_ms as property-like attribute
        turn.duration_ms = 23400.0  # 23.4s in ms
        result = turn_account(turn)
        assert "9 阶段" in result
        assert "风险low" in result
        assert "23.4s" in result

    def test_none_returns_empty(self):
        from openllm.cli.activity import turn_account
        assert turn_account(None) == ""

    def test_missing_risk_level(self):
        """turn 缺 risk_level 属性时容错。"""
        from openllm.cli.activity import turn_account
        turn = SimpleNamespace(
            phase_metrics=[1, 2],
            start_time=time.time() - 1.0,
        )
        turn.duration_ms = 1000.0
        result = turn_account(turn)
        assert "2 阶段" in result

    def test_no_duration_ms_attribute(self):
        """turn 没有 duration_ms 属性时不崩。"""
        from openllm.cli.activity import turn_account
        turn = SimpleNamespace(
            phase_metrics=[1],
            risk_level="high",
            start_time=time.time(),
        )
        result = turn_account(turn)
        assert "1 阶段" in result
        assert "风险high" in result
