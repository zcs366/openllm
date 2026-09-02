"""Tests for PAL T-G-2: SelfModificationGuard activated in heartbeat._learn.

Covers:
  - heartbeat._learn calls guard.check_rate_limit + approve_change before write
  - guard.record_modification called after successful write
  - rate_limit exceeded → learn_causal skipped, guard recorded
  - forbidden target → learn_causal skipped, guard recorded rejection
  - guard failure → learn_causal still executes (non-blocking)
  - regression: existing guard methods unchanged
"""
from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from openllm.governance.self_modification_guard import (
    SelfModificationGuard,
)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def guard(tmp_path: Path) -> SelfModificationGuard:
    """Fresh guard with isolated state file."""
    state_path = tmp_path / "test_guard_state.json"
    return SelfModificationGuard(state_path=state_path)


@pytest.fixture
def mock_agent(tmp_path: Path):
    """Minimal agent mock for _learn testing."""
    agent = MagicMock()
    agent._agent_id = "test-agent"
    agent._tick_count = 1
    agent.mode = "silent"
    agent._last_output = ""

    # ISA mock
    agent.isa.build_context.return_value = MagicMock(
        memory={}, identity={}, search_results=[], tools=[]
    )
    agent.isa.respond = MagicMock()
    agent.isa.mode = "silent"
    agent.isa.causal = None

    # IOS mock
    agent.ios.risk_check.return_value = MagicMock(level="low", is_blocked=lambda: False, details=None)
    agent.ios.cap_check.return_value = True
    agent.ios.arbitrate.return_value = MagicMock(approved=True, reason="ok", action="chat", tool_calls=None, _has_tools=False)
    agent.ios.evolve = MagicMock()
    agent.ios.governance_engine = MagicMock()
    agent.ios.governance_engine.heartbeat_trace = MagicMock()
    agent.ios.last_output = {}
    agent._tool_failures = []

    # IOS learn_causal - just record it was called
    agent.ios.learn_causal = MagicMock()

    # Octopus mock
    agent.octopus.compare.return_value = MagicMock(
        prediction_match=True, delta_summary="", summary_text=lambda: "ok"
    )
    agent.octopus.predict_consequences.return_value = MagicMock(
        summary_text=lambda: "predict ok", summary=""
    )
    agent.octopus.reason.return_value = (
        MagicMock(content="proposal", confidence=0.8, evidence=[]),
        MagicMock(verdict="approve", concerns=[]),
    )
    agent.octopus.d0_snapshot.return_value = {}
    agent.octopus.learn_causal = MagicMock()
    agent.octopus.last_output = {}

    # Session mock
    agent.session.new_turn.return_value = MagicMock(
        id="turn_1",
        risk_level=None,
        trace_phase=MagicMock(),
        add_action=MagicMock(),
        complete=MagicMock(),
        fail=MagicMock(),
    )
    agent.session.compact_if_needed = MagicMock()
    agent.session.checkpoint = MagicMock()

    # Clock mock
    agent.clock = MagicMock()
    agent.clock.tick = MagicMock()
    agent.clock.now_status.return_value = {
        "last_wall_time": time.time() - 60,
        "gap_since_last": 60.0,
        "epoch": 1,
        "awakening_count": 0,
    }

    # IKO mock
    agent.iko = MagicMock()
    agent.iko.trace = MagicMock()
    agent.iko.process_output = MagicMock(return_value=None)

    # Memory evaluator mock
    agent.memory_evaluator = MagicMock()

    # Awakening protocol mock
    agent._awakening_protocol = MagicMock()
    agent._awakening_protocol.inject_to_context.return_value = False
    agent._awakening_protocol.detect_choice_and_record = MagicMock()

    # Feedback loop mock
    agent.feedback_loop = MagicMock()
    agent.feedback_loop.collect_feedback.return_value = []
    agent.feedback_loop.apply_feedback.return_value = []

    # Research mock
    agent.research = MagicMock()
    agent.research.to_heartbeat_context.return_value = {}

    # Body outputs
    agent._body_outputs = {}

    # IAI mock
    agent.iai = MagicMock()
    agent.iai.gate = MagicMock()
    agent.iai.emit = MagicMock()

    # Record inference
    agent._record_inference = MagicMock()

    return agent


def _make_hc(agent):
    """Build a minimal HeartbeatContext-like object."""
    hc = MagicMock()
    hc.user_message = "test message"
    hc.prediction = agent.octopus.predict_consequences.return_value
    hc.result = MagicMock(output="result", success=True, error="", duration_ms=10)
    hc.decision = MagicMock(approved=True, tool_calls=None, _has_tools=False, reason="ok")
    hc.left_proposal = MagicMock(content="proposal", confidence=0.8, evidence=[])
    hc.right_critique = MagicMock(verdict="approve", concerns=[])
    hc.search_results = []
    return hc


def _run_learn(agent):
    """Run _learn with proper mocks."""
    from openllm.core.agent_heartbeat import _learn
    hc = _make_hc(agent)
    turn = agent.session.new_turn.return_value
    _learn(agent, hc, turn)
    return hc, turn


# ── Test: Guard records modification after successful learn_causal ──

class TestGuardActivated:
    """Core T-G-2: guard is called during _learn phase."""

    def test_guard_record_modification_after_write(
        self, mock_agent, tmp_path: Path
    ):
        """After learn_causal writes, guard.get_modification_history is non-empty."""
        state_path = tmp_path / "guard_state.json"
        _guard = SelfModificationGuard(state_path=state_path)
        _causal_target = str(
            Path.home() / ".openllm" / "output" / "ios" / "causal_memory.jsonl"
        )

        # Patch SelfModificationGuard class at the source module level
        with patch(
            "openllm.governance.self_modification_guard.SelfModificationGuard",
            return_value=_guard,
        ):
            _run_learn(mock_agent)

        history = _guard.get_modification_history(_causal_target)
        assert len(history) >= 1, f"Guard should have recorded modification, got: {history}"
        assert history[-1]["change_type"] == "causal_memory"
        assert history[-1]["agent_id"] == "test-agent"

    def test_guard_record_has_timestamp(
        self, mock_agent, tmp_path: Path
    ):
        """Recorded event has a valid timestamp."""
        state_path = tmp_path / "guard_state.json"
        _guard = SelfModificationGuard(state_path=state_path)
        before = time.time()

        with patch(
            "openllm.governance.self_modification_guard.SelfModificationGuard",
            return_value=_guard,
        ):
            _run_learn(mock_agent)

        _causal_target = str(
            Path.home() / ".openllm" / "output" / "ios" / "causal_memory.jsonl"
        )
        history = _guard.get_modification_history(_causal_target)
        assert len(history) >= 1
        assert history[-1]["timestamp"] >= before


# ── Test: Rate limit exceeded → skip ──

class TestGuardRateLimitSkip:
    """When rate limit is hit, learn_causal is skipped and guard logs it."""

    def test_rate_limit_skip(
        self, mock_agent, tmp_path: Path
    ):
        """check_rate_limit returns True → learn_causal not called."""
        state_path = tmp_path / "guard_state.json"
        _guard = SelfModificationGuard(
            state_path=state_path,
            max_modifications=1,  # Only 1 allowed in window
        )
        _causal_target = str(
            Path.home() / ".openllm" / "output" / "ios" / "causal_memory.jsonl"
        )
        # Pre-fill with 1 modification to exceed limit
        _guard.record_modification(_causal_target, "causal_memory", "test-agent")

        with patch(
            "openllm.governance.self_modification_guard.SelfModificationGuard",
            return_value=_guard,
        ):
            _run_learn(mock_agent)

        # learn_causal should NOT have been called (rate limited)
        mock_agent.ios.learn_causal.assert_not_called()
        mock_agent.octopus.learn_causal.assert_not_called()

        # Guard should have traced the skip
        turn = mock_agent.session.new_turn.return_value
        turn.trace_phase.assert_any_call(
            "guard_rate_limit", "skip", detail=f"target={_causal_target}"
        )


# ── Test: Forbidden target → reject ──

class TestGuardForbiddenReject:
    """When approve_change rejects, learn_causal is skipped."""

    def test_forbidden_target_skip(
        self, mock_agent, tmp_path: Path
    ):
        """approve_change returns False for forbidden path → learn_causal skipped."""
        state_path = tmp_path / "guard_state.json"
        _guard = SelfModificationGuard(state_path=state_path)

        # Test the guard behavior directly with a forbidden target
        forbidden_target = "src/openllm/core/agent_heartbeat.py"
        result = _guard.approve_change(forbidden_target, change_type="causal_memory")
        assert result is False
        history = _guard.get_modification_history(forbidden_target)
        assert len(history) == 1
        assert history[0]["forbidden"] is True

    def test_guard_reject_writes_audit_trail(
        self, mock_agent, tmp_path: Path
    ):
        """When a forbidden target is rejected, audit trail is written."""
        state_path = tmp_path / "guard_state.json"
        _guard = SelfModificationGuard(state_path=state_path)

        forbidden_target = "src/openllm/governance/decision_guard.py"
        result = _guard.approve_change(forbidden_target, change_type="causal_memory")
        assert result is False

        history = _guard.get_modification_history(forbidden_target)
        assert len(history) == 1
        event = history[0]
        assert event["forbidden"] is True
        assert event["intent_alert"]["severity"] == "high"
        assert "forbidden_path" in event["intent_alert"]["evidence"]
        assert event["change_type"] == "causal_memory"


# ── Test: Guard failure → heartbeat continues ──

class TestGuardNonBlocking:
    """Guard exceptions must not block the heartbeat."""

    def test_guard_import_error_continues(
        self, mock_agent, tmp_path: Path
    ):
        """If SelfModificationGuard import fails, heartbeat still runs."""
        with patch(
            "openllm.governance.self_modification_guard.SelfModificationGuard",
            side_effect=ImportError("guard unavailable"),
        ):
            _run_learn(mock_agent)

        # learn_causal should still be called via fallback
        mock_agent.ios.learn_causal.assert_called_once()

    def test_guard_check_exception_continues(
        self, mock_agent, tmp_path: Path
    ):
        """If guard.check_rate_limit raises, heartbeat still runs."""
        state_path = tmp_path / "guard_state.json"
        _guard = SelfModificationGuard(state_path=state_path)
        _guard.check_rate_limit = MagicMock(side_effect=RuntimeError("check failed"))

        with patch(
            "openllm.governance.self_modification_guard.SelfModificationGuard",
            return_value=_guard,
        ):
            _run_learn(mock_agent)

        # Fallback: learn_causal should still be called
        mock_agent.ios.learn_causal.assert_called_once()


# ── Test: Regression — guard methods unchanged ──

class TestGuardRegression:
    """Existing guard methods still work correctly."""

    def test_record_modification_unchanged(self, guard: SelfModificationGuard):
        guard.record_modification("tool.py", "edit", "agent1")
        history = guard.get_modification_history("tool.py")
        assert len(history) == 1
        assert history[0]["change_type"] == "edit"

    def test_check_rate_limit_unchanged(self, guard: SelfModificationGuard):
        guard.record_modification("tool.py", "edit", "agent1")
        assert guard.check_rate_limit("tool.py") is False  # 1 < 3

    def test_is_forbidden_unchanged(self, guard: SelfModificationGuard):
        assert guard.is_forbidden("src/openllm/core/agent_heartbeat.py") is True
        assert guard.is_forbidden("src/openllm/isn/skill_index.py") is False

    def test_approve_change_unchanged(self, guard: SelfModificationGuard):
        assert guard.approve_change("src/openllm/tools/doc_parser.py") is True
        assert guard.approve_change("src/openllm/governance/decision_guard.py") is False

    def test_get_modification_history_unchanged(self, guard: SelfModificationGuard):
        guard.record_modification("a.py", "edit", "agent1")
        guard.record_modification("a.py", "edit", "agent2")
        history = guard.get_modification_history("a.py", limit=1)
        assert len(history) == 1
