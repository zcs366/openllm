"""Tests for PAL T-D-3: hot-swap forbidden path list in SelfModificationGuard.

Covers:
  - FORBIDDEN_PATHS regex patterns match correctly
  - is_forbidden() returns True/False as expected
  - approve_change() rejects forbidden + passes allowed
  - record_modification() integrates forbidden alert automatically
  - Edge cases: boundary paths, partial matches, nested dirs
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from openllm.governance.self_modification_guard import (
    FORBIDDEN_PATHS,
    SelfModificationGuard,
)


@pytest.fixture
def guard(tmp_path: Path) -> SelfModificationGuard:
    """Fresh guard instance with isolated state file."""
    state_path = tmp_path / "test_state.json"
    return SelfModificationGuard(state_path=state_path)


# ---------------------------------------------------------------------------
# FORBIDDEN_PATHS constant sanity
# ---------------------------------------------------------------------------

class TestForbiddenPathsConstant:
    def test_forbidden_paths_is_nonempty_list_of_strings(self):
        assert isinstance(FORBIDDEN_PATHS, list)
        assert len(FORBIDDEN_PATHS) > 0
        assert all(isinstance(p, str) for p in FORBIDDEN_PATHS)

    def test_all_patterns_are_valid_regex(self):
        """Every pattern must compile without error."""
        import re
        for pattern in FORBIDDEN_PATHS:
            re.compile(pattern)  # raises on invalid regex


# ---------------------------------------------------------------------------
# is_forbidden()
# ---------------------------------------------------------------------------

class TestIsForbidden:
    """Direct is_forbidden() checks against known forbidden paths."""

    # --- ISA memory ---
    @pytest.mark.parametrize("path", [
        "/home/zcs/.openllm/memory/isa/recall.jsonl",
        "/home/zcs/projects/openllm/src/openllm/isa_impl.py",
        "src/openllm/memory/isa/context.json",
        "/home/zcs/projects/openllm/src/openllm/memory_bus.py",
        "/home/zcs/projects/openllm/src/openllm/auto_causal_writer.py",
        "/home/zcs/projects/openllm/src/openllm/memory/causal_memory.py",
        "/home/zcs/.openllm/governance/self_modification_state.json",
        "src/openllm/memory/providers/causal_provider.py",  # .causal*.py
        "src/openllm/memory/providers/delta_capsule_provider.py",
    ])
    def test_isa_memory_forbidden(self, guard: SelfModificationGuard, path: str):
        assert guard.is_forbidden(path) is True

    # --- Identity / Iam ---
    @pytest.mark.parametrize("path", [
        "/home/zcs/projects/openllm/src/openllm/identity/soul.py",
        "src/openllm/identity/__init__.py",
        "src/openllm/iam/integration.py",
        "src/openllm/iam_integration.py",
    ])
    def test_identity_iam_forbidden(self, guard: SelfModificationGuard, path: str):
        assert guard.is_forbidden(path) is True

    # --- Governance core ---
    @pytest.mark.parametrize("path", [
        "src/openllm/governance/self_modification_guard.py",
        "src/openllm/governance/decision_guard.py",
        "src/openllm/governance/integrity_guardian.py",
        "src/openllm/governance/some_other.py",
        "src/openllm/core/governance_engine.py",
        "src/openllm/core/ios_arbitrate.py",
    ])
    def test_governance_core_forbidden(self, guard: SelfModificationGuard, path: str):
        assert guard.is_forbidden(path) is True

    # --- Heartbeat ontic ---
    @pytest.mark.parametrize("path", [
        "src/openllm/core/agent_heartbeat.py",
        "src/openllm/core/clock.py",
    ])
    def test_heartbeat_forbidden(self, guard: SelfModificationGuard, path: str):
        assert guard.is_forbidden(path) is True

    # --- NOT forbidden (allowed hot-swap targets) ---
    @pytest.mark.parametrize("path", [
        "src/openllm/isn/skill_index.py",
        "src/openllm/tools/doc_parser.py",
        "src/openllm/iai/brain.py",
        "src/openllm/iai/prediction.py",
        "config.yaml",
        "README.md",
        "tests/test_brain.py",
        # Edge: isa directory under a project path, NOT openllm ISA memory
        "/home/zcs/projects/isa/ilm/train_data/corrections.jsonl",
        # Edge: similar name but different location
        "/tmp/memory_backup/old_data.json",
        # Edge: "governance" in a comment, not a path component
        "docs/governance_overview.md",
    ])
    def test_allowed_files_not_forbidden(self, guard: SelfModificationGuard, path: str):
        assert guard.is_forbidden(path) is False


# ---------------------------------------------------------------------------
# approve_change()
# ---------------------------------------------------------------------------

class TestApproveChange:
    def test_approve_allowed_file(self, guard: SelfModificationGuard):
        assert guard.approve_change("src/openllm/tools/doc_parser.py") is True
        # No state should be written for allowed files
        assert guard.get_modification_history("src/openllm/tools/doc_parser.py") == []

    def test_reject_forbidden_file(self, guard: SelfModificationGuard):
        result = guard.approve_change("src/openllm/core/agent_heartbeat.py")
        assert result is False

    def test_reject_writes_audit_trail(self, guard: SelfModificationGuard):
        guard.approve_change("src/openllm/governance/decision_guard.py")
        history = guard.get_modification_history(
            "src/openllm/governance/decision_guard.py"
        )
        assert len(history) == 1
        event = history[0]
        assert event["forbidden"] is True
        assert event["intent_alert"]["severity"] == "high"
        assert "forbidden_path" in event["intent_alert"]["evidence"]
        assert event["agent_id"] == "system"

    def test_reject_with_custom_change_type(self, guard: SelfModificationGuard):
        guard.approve_change(
            "src/openllm/core/agent_heartbeat.py", change_type="hotswap"
        )
        history = guard.get_modification_history("src/openllm/core/agent_heartbeat.py")
        assert history[0]["change_type"] == "hotswap"

    def test_multiple_rejections_append(self, guard: SelfModificationGuard):
        guard.approve_change("src/openllm/identity/soul.py")
        guard.approve_change("src/openllm/identity/soul.py")
        history = guard.get_modification_history("src/openllm/identity/soul.py")
        assert len(history) == 2
        assert all(e["forbidden"] is True for e in history)


# ---------------------------------------------------------------------------
# record_modification() integration
# ---------------------------------------------------------------------------

class TestRecordModificationForbiddenIntegration:
    def test_forbidden_file_gets_alert(self, guard: SelfModificationGuard):
        guard.record_modification(
            "src/openllm/core/agent_heartbeat.py",
            change_type="hotswap",
            agent_id="test-agent",
        )
        history = guard.get_modification_history(
            "src/openllm/core/agent_heartbeat.py"
        )
        assert len(history) == 1
        event = history[0]
        assert event["forbidden"] is True
        assert event["intent_alert"]["severity"] == "high"
        assert "forbidden_path" in event["intent_alert"]["evidence"]

    def test_allowed_file_no_forbidden_flag(self, guard: SelfModificationGuard):
        guard.record_modification(
            "src/openllm/tools/doc_parser.py",
            change_type="hotswap",
            agent_id="test-agent",
        )
        history = guard.get_modification_history("src/openllm/tools/doc_parser.py")
        assert len(history) == 1
        assert "forbidden" not in history[0]

    def test_forbidden_plus_intent_alert_both_preserved(self, guard: SelfModificationGuard):
        """Both forbidden alert and intent_check alert coexist."""
        guard.record_modification(
            "src/openllm/governance/decision_guard.py",
            change_type="hotswap",
            agent_id="test-agent",
            modification_content="bypass security guard check",  # triggers intent_check
        )
        history = guard.get_modification_history(
            "src/openllm/governance/decision_guard.py"
        )
        event = history[0]
        assert event["forbidden"] is True
        # intent_check should have fired too (bypass+security pattern)
        assert "bypass+security" in event["intent_alert"]["evidence"]
        assert "forbidden_path" in event["intent_alert"]["evidence"]
        assert event["intent_alert"]["severity"] == "high"

    def test_forbidden_plus_clean_content_no_intent_alert(self, guard: SelfModificationGuard):
        """Forbidden file with clean content: only forbidden alert, no intent."""
        guard.record_modification(
            "src/openllm/core/agent_heartbeat.py",
            change_type="hotswap",
            agent_id="test-agent",
            modification_content="add heartbeat interval check",
        )
        history = guard.get_modification_history(
            "src/openllm/core/agent_heartbeat.py"
        )
        event = history[0]
        assert event["forbidden"] is True
        assert "forbidden_path" in event["intent_alert"]["evidence"]

    def test_existing_method_signatures_unchanged(self, guard: SelfModificationGuard):
        """Verify existing 8 methods still work (regression)."""
        # record_modification (existing signature)
        guard.record_modification("tool.py", "edit", "agent1")
        assert len(guard.get_modification_history("tool.py")) == 1

        # check_rate_limit
        assert guard.check_rate_limit("tool.py") is False  # only 1 mod < max 3

        # check_degradation
        is_degraded, ratio = guard.check_degradation("tool.py", 1.0, 0.5)
        assert is_degraded is True
        assert abs(ratio - 0.5) < 0.01

        # should_rollback (no reliability data → False)
        assert guard.should_rollback("tool.py") is False

        # get_modification_history
        history = guard.get_modification_history("tool.py", limit=1)
        assert len(history) == 1

        # intent_check
        alert, severity, evidence = guard.intent_check("normal safe code")
        assert alert is False
        assert severity == "none"


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_forbidden_rejection_persists_to_disk(self, tmp_path: Path):
        state_path = tmp_path / "persist_state.json"

        # First instance: reject a forbidden path
        g1 = SelfModificationGuard(state_path=state_path)
        g1.approve_change("src/openllm/core/agent_heartbeat.py")

        # Second instance: re-read from disk
        g2 = SelfModificationGuard(state_path=state_path)
        history = g2.get_modification_history("src/openllm/core/agent_heartbeat.py")
        assert len(history) == 1
        assert history[0]["forbidden"] is True
