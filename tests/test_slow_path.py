"""tests/test_slow_path.py — SlowAnalyzer 测试 (含mock LLM)"""
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openllm.iai.event_bus import Event, EventBus
from openllm.iai.slow_path import SlowAnalyzer, HIGH_ENTROPY_THRESHOLD


# ── fixtures ──

@pytest.fixture
def tmp_rules(tmp_path):
    return tmp_path / "iai_rules.json"


@pytest.fixture
def mock_llm():
    """返回一个 mock LLM 函数，后续可配置返回值。"""
    fn = MagicMock(return_value={
        "suggestion": "add retry rule for timeout",
        "new_rules": [
            {"condition": "timeout > 3", "action": "retry",
             "description": "超时重试规则"},
            {"condition": "error_rate > 0.5", "action": "degrade",
             "description": "高错误率降级规则"},
        ],
    })
    return fn


@pytest.fixture
def analyzer(mock_llm, tmp_rules):
    return SlowAnalyzer(llm_fn=mock_llm, rules_path=tmp_rules, threshold=0.6)


def _make_event(entropy: float = 0.8, etype: str = "code_exec") -> Event:
    return Event(source="IOS", type=etype, timestamp=time.time(),
                 entropy_score=entropy, payload={"detail": "test event"})


# ── tests ──

class TestAnalyze:
    def test_calls_injected_llm(self, analyzer, mock_llm):
        event = _make_event()
        result = analyzer.analyze(event)
        mock_llm.assert_called_once()
        call_arg = mock_llm.call_args[0][0]
        assert call_arg["source"] == "IOS"
        assert call_arg["type"] == "code_exec"
        assert call_arg["entropy_score"] == 0.8
        assert "existing_rules" in call_arg
        assert result == mock_llm.return_value

    def test_analyze_returns_dict(self, analyzer, mock_llm):
        result = analyzer.analyze(_make_event())
        assert isinstance(result, dict)
        assert "suggestion" in result
        assert "new_rules" in result

    def test_no_real_llm_called(self, analyzer):
        """验证不调用任何真实LLM——只通过注入函数。"""
        assert callable(analyzer._llm_fn)
        assert not hasattr(analyzer._llm_fn, '__wrapped__')  # 不是真实调用


class TestUpdateRules:
    def test_adds_valid_rules(self, analyzer, tmp_rules):
        analysis = {
            "new_rules": [
                {"condition": "x > 1", "action": "skip", "description": "规则A"},
                {"condition": "y < 0", "action": "run", "description": "规则B"},
            ]
        }
        added = analyzer.update_rules(analysis)
        assert len(added) == 2
        assert "规则A" in added
        assert "规则B" in added
        # 持久化验证
        saved = json.loads(tmp_rules.read_text(encoding="utf-8"))
        assert len(saved) == 2

    def test_ignores_rules_without_condition(self, analyzer, tmp_rules):
        analysis = {"new_rules": [{"action": "skip"}]}  # 无condition
        added = analyzer.update_rules(analysis)
        assert len(added) == 0

    def test_ignores_non_dict_rules(self, analyzer):
        analysis = {"new_rules": ["not a dict", 42, None]}
        added = analyzer.update_rules(analysis)
        assert len(added) == 0

    def test_empty_analysis(self, analyzer):
        added = analyzer.update_rules({})
        assert added == []

    def test_default_description(self, analyzer, tmp_rules):
        analysis = {"new_rules": [{"condition": "a"}]}
        added = analyzer.update_rules(analysis)
        assert added == ["unnamed"]

    def test_persistence_across_instances(self, mock_llm, tmp_rules):
        a1 = SlowAnalyzer(llm_fn=mock_llm, rules_path=tmp_rules)
        a1.update_rules({"new_rules": [{"condition": "c", "description": "持久规则"}]})
        a2 = SlowAnalyzer(llm_fn=mock_llm, rules_path=tmp_rules)
        assert len(a2.get_rules()) == 1
        assert a2.get_rules()[0]["description"] == "持久规则"


class TestRunLoop:
    def test_analyzes_high_entropy_events(self, analyzer, mock_llm):
        """发布高熵事件 → run_loop应分析并发布routing.suggestion。"""
        bus = EventBus()
        suggestions = []

        def capture(event: Event):
            if event.type == "routing.suggestion":
                suggestions.append(event)

        bus.subscribe(capture, type_filter="routing.suggestion")

        stop = threading.Event()

        def run():
            analyzer.run_loop(bus, interval=0.05, stop_event=stop)

        t = threading.Thread(target=run, daemon=True)
        t.start()

        time.sleep(0.1)
        bus.publish(_make_event(entropy=0.9))
        bus.publish(_make_event(entropy=0.1))  # 低熵，不应触发
        time.sleep(0.3)

        stop.set()
        t.join(timeout=2)

        assert len(suggestions) == 1  # 只有高熵事件被分析
        assert suggestions[0].payload["event_id"] != ""
        assert len(suggestions[0].payload["rules_added"]) == 2
        mock_llm.assert_called()

    def test_low_entropy_not_analyzed(self, analyzer, mock_llm):
        bus = EventBus()
        suggestions = []
        bus.subscribe(lambda e: suggestions.append(e), type_filter="routing.suggestion")

        stop = threading.Event()
        t = threading.Thread(target=lambda: analyzer.run_loop(bus, 0.05, stop), daemon=True)
        t.start()

        time.sleep(0.1)
        bus.publish(_make_event(entropy=0.3))
        time.sleep(0.2)

        stop.set()
        t.join(timeout=2)
        assert len(suggestions) == 0
        mock_llm.assert_not_called()

    def test_stop_loop(self, analyzer, mock_llm):
        bus = EventBus()
        stop = threading.Event()
        t = threading.Thread(target=lambda: analyzer.run_loop(bus, 0.05, stop), daemon=True)
        t.start()
        time.sleep(0.15)
        analyzer.stop_loop()
        stop.set()
        t.join(timeout=2)
        assert not analyzer._running

    def test_unsubscribes_on_exit(self, analyzer):
        bus = EventBus()
        count_before = bus.subscriber_count()
        stop = threading.Event()
        t = threading.Thread(target=lambda: analyzer.run_loop(bus, 0.05, stop), daemon=True)
        t.start()
        time.sleep(0.1)
        assert bus.subscriber_count() == count_before + 1  # run_loop added 1
        stop.set()
        t.join(timeout=2)
        # 退出后 run_loop 的订阅被清除
        assert bus.subscriber_count() == count_before


class TestLoadRules:
    def test_corrupt_file_returns_empty(self, mock_llm, tmp_rules):
        tmp_rules.write_text("NOT JSON!!!", encoding="utf-8")
        a = SlowAnalyzer(llm_fn=mock_llm, rules_path=tmp_rules)
        assert a.get_rules() == []

    def test_missing_file_returns_empty(self, mock_llm, tmp_path):
        a = SlowAnalyzer(llm_fn=mock_llm, rules_path=tmp_path / "nope.json")
        assert a.get_rules() == []
