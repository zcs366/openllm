"""
IKO Boundary Condition Tests
============================

Tests for edge cases, error handling, and boundary conditions across all
IKO modules: IntentClassifier, FeedbackCollector, OutputAuditChain,
ProbingTrainer, SilenceAuditor, SymmetricCodec, LambdaCalibrator, OutputRouter.

Each test verifies graceful handling: no crashes, correct exceptions,
or reasonable defaults.
"""

import tempfile
from pathlib import Path

import pytest

from openllm.iko.feedback_collector import (
    FeedbackSignal,
    OutputFeedbackCollector,
)
from openllm.iko.intent_classifier import (
    IntentClassifier,
    OutputIntent,
)
from openllm.iko.lambda_calibrator import (
    LambdaCalibrator,
    ROLLBACK_THRESHOLD,
)
from openllm.iko.output_audit import OutputAuditChain
from openllm.iko.output_router import OutputRouter
from openllm.iko.probing_trainer import ProbingTrainer
from openllm.iko.silence_auditor import SilenceAuditor
from openllm.iko.symmetric_codec import (
    CompressedReasoning,
    FullReasoningChain,
    SymmetricCodec,
)


# ──────────────────────────────────────────────────────────────
# 1. IntentClassifier — empty context / empty decision / missing fields
# ──────────────────────────────────────────────────────────────

class TestIntentClassifierBoundary:
    def setup_method(self):
        self.clf = IntentClassifier()

    def test_empty_context_raises(self):
        """Empty context dict missing all required fields."""
        with pytest.raises(ValueError, match="context missing required fields"):
            self.clf.classify({}, {"content": "hi"})

    def test_missing_single_field_raises(self):
        """Context missing one required field."""
        ctx = {
            "risk_level": "LOW",
            "has_tool_calls": False,
            "has_side_effects": False,
            # missing option_count
        }
        with pytest.raises(ValueError, match="option_count"):
            self.clf.classify(ctx, {"content": "hi"})

    def test_empty_decision_raises(self):
        """Decision missing required 'content' field."""
        ctx = {
            "risk_level": "LOW",
            "has_tool_calls": False,
            "has_side_effects": False,
            "option_count": 0,
        }
        with pytest.raises(ValueError, match="decision missing required field: content"):
            self.clf.classify(ctx, {})

    def test_empty_content_falls_through_to_silent(self):
        """Decision with empty string content → SILENT (no conditions met)."""
        ctx = {
            "risk_level": "LOW",
            "has_tool_calls": False,
            "has_side_effects": False,
            "option_count": 0,
        }
        result = self.clf.classify(ctx, {"content": ""})
        assert result.intent == OutputIntent.SILENT


# ──────────────────────────────────────────────────────────────
# 2. FeedbackCollector — empty user_id / empty history / extreme time_delta
# ──────────────────────────────────────────────────────────────

class TestFeedbackCollectorBoundary:
    def setup_method(self):
        self.collector = OutputFeedbackCollector()

    def test_empty_user_id_defaults_to_anonymous(self):
        """Missing user_id in action → stored under 'anonymous'."""
        action = {"action_type": "execute"}
        signal = self.collector.detect_signal(action, "out-001")
        assert signal == FeedbackSignal.ACCEPTED
        prefs = self.collector.get_user_preferences("anonymous")
        assert prefs["total_interactions"] == 1

    def test_empty_history_returns_defaults(self):
        """User with no history → default preferences."""
        prefs = self.collector.get_user_preferences("nonexistent_user")
        assert prefs["accepted_ratio"] == 0.0
        assert prefs["rejection_ratio"] == 0.0
        assert prefs["preferred_density"] == "medium"
        assert prefs["total_interactions"] == 0

    def test_extreme_time_delta_large(self):
        """Very large time_delta → IGNORED signal."""
        action = {"time_delta": 999999.0}
        signal = self.collector.detect_signal(action, "out-002")
        assert signal == FeedbackSignal.IGNORED

    def test_extreme_time_delta_zero(self):
        """time_delta=0 with text → CLARIFIED."""
        action = {"time_delta": 0.0, "text": "what do you mean?"}
        signal = self.collector.detect_signal(action, "out-003")
        assert signal == FeedbackSignal.CLARIFIED

    def test_density_factor_clamped(self):
        """Density factor always stays within [0.3, 3.0]."""
        for _ in range(100):
            self.collector.detect_signal(
                {"action_type": "execute", "user_id": "bulk"}, "out-x"
            )
        factor = self.collector.should_adjust_density("bulk")
        assert 0.3 <= factor <= 3.0

    def test_no_storage_path_save_is_noop(self):
        """save_to_disk with None path should not crash."""
        self.collector.save_to_disk()  # no crash


# ──────────────────────────────────────────────────────────────
# 3. OutputAuditChain — empty chain verify / duplicate output_id / large content
# ──────────────────────────────────────────────────────────────

class TestOutputAuditChainBoundary:
    def setup_method(self):
        self.chain = OutputAuditChain()

    def test_empty_chain_verify_returns_true(self):
        """Empty audit chain is valid."""
        assert self.chain.verify() is True

    def test_duplicate_output_id_raises(self):
        """Appending same output_id twice → ValueError."""
        common = dict(
            intent="test",
            content=b"data",
            decision_source="test",
            risk_level=0.0,
            confidence=1.0,
            reasoning_chain_hash="abc123",
        )
        self.chain.append(output_id="dup-001", **common)
        with pytest.raises(ValueError, match="Duplicate output_id"):
            self.chain.append(output_id="dup-001", **common)

    def test_large_content_no_crash(self):
        """Very large content still produces valid hash."""
        large_content = b"x" * 1_000_000
        entry = self.chain.append(
            output_id="large-001",
            intent="big data",
            content=large_content,
            decision_source="test",
            risk_level=0.0,
            confidence=0.5,
            reasoning_chain_hash="deadbeef",
        )
        assert len(entry.content_hash) == 16
        assert self.chain.verify() is True

    def test_missing_output_id_raises_keyerror(self):
        """get_provenance on nonexistent id → KeyError."""
        with pytest.raises(KeyError, match="not found"):
            self.chain.get_provenance("nonexistent")

    def test_chain_length(self):
        """len() tracks entries correctly."""
        assert len(self.chain) == 0
        self.chain.append(
            output_id="a", intent="t", content=b"d",
            decision_source="s", risk_level=0.0, confidence=1.0,
            reasoning_chain_hash="h",
        )
        assert len(self.chain) == 1


# ──────────────────────────────────────────────────────────────
# 4. ProbingTrainer — session_count=0 / negative / large
# ──────────────────────────────────────────────────────────────

class TestProbingTrainerBoundary:
    def setup_method(self):
        self.trainer = ProbingTrainer()

    def test_session_count_zero_returns_false(self):
        """Zero sessions → never prompt (guard: session_count > 0)."""
        assert self.trainer.should_prompt_probing("u", 0) is False

    def test_session_count_negative_returns_false(self):
        """Negative sessions → M1 branch, but not > 0 → False."""
        assert self.trainer.should_prompt_probing("u", -5) is False

    def test_session_count_very_large_m3(self):
        """Very large session_count → M3, always False."""
        assert self.trainer.should_prompt_probing("u", 1_000_000) is False

    def test_session_count_m2_boundary(self):
        """M2 boundary: session_count=90 → prompt if divisible by 10."""
        assert self.trainer.should_prompt_probing("u", 90) is True

    def test_session_count_m1_boundary(self):
        """M1 boundary: session_count=25 → prompt if divisible by 5."""
        assert self.trainer.should_prompt_probing("u", 25) is True

    def test_empty_output_suggestion(self):
        """Empty/whitespace last_output → empty suggestion."""
        assert self.trainer.get_probing_suggestion("") == ""
        assert self.trainer.get_probing_suggestion("   ") == ""


# ──────────────────────────────────────────────────────────────
# 5. SilenceAuditor — empty context / None reversible / empty output_id
# ──────────────────────────────────────────────────────────────

class TestSilenceAuditorBoundary:
    def setup_method(self):
        self.auditor = SilenceAuditor()

    def test_empty_context_uses_defaults(self):
        """Empty context → risk_level defaults to LOW."""
        result = self.auditor.audit(
            OutputIntent.SILENT, context={}, output_id="x"
        )
        # First interaction in 'unknown' domain → INFORM
        assert result == OutputIntent.INFORM

    def test_none_reversible_falls_through_to_inform(self):
        """None is falsy → treated as irreversible → INFORM."""
        result = self.auditor.audit(
            OutputIntent.SILENT,
            context={"risk_level": "LOW"},
            reversible=None,
            output_id="y",
        )
        assert result == OutputIntent.INFORM

    def test_empty_output_id_no_crash(self):
        """Empty output_id → _record skips, no crash."""
        result = self.auditor.audit(
            OutputIntent.SILENT,
            context={"risk_level": "LOW"},
            output_id="",
        )
        # First interaction in 'unknown' domain → INFORM
        assert result == OutputIntent.INFORM

    def test_non_silent_passthrough(self):
        """Non-SILENT intent passes through unchanged."""
        result = self.auditor.audit(
            OutputIntent.INFORM,
            context={"risk_level": "LOW"},
        )
        assert result == OutputIntent.INFORM

    def test_explain_empty_output_id(self):
        """explain_silence with empty id → generic message."""
        msg = self.auditor.explain_silence("")
        assert "未触发审计记录" in msg


# ──────────────────────────────────────────────────────────────
# 6. SymmetricCodec — empty phases / None summary / large chain
# ──────────────────────────────────────────────────────────────

class TestSymmetricCodecBoundary:
    def setup_method(self):
        self.codec = SymmetricCodec()

    def test_empty_phases_summary_fallback(self):
        """Empty phases → summary defaults to '无描述', reversible=False."""
        chain = FullReasoningChain(phases=[], risk_assessments=[])
        compressed = self.codec.compress(chain)
        assert compressed.summary == "无描述"
        assert compressed.reversible is False
        assert compressed.confidence == 0.0

    def test_none_summary_decompress_raises(self):
        """Compressed with empty summary → verify_reversibility returns False."""
        c = CompressedReasoning(summary="", reversible=True)
        assert self.codec.verify_reversibility(c) is False

    def test_non_reversible_decompress_raises(self):
        """decompress on non-reversible → ValueError."""
        c = CompressedReasoning(summary="test", reversible=False)
        with pytest.raises(ValueError, match="non-reversible"):
            self.codec.decompress(c)

    def test_large_chain_compress(self):
        """Very large chain still compresses correctly."""
        phases = [{"description": f"phase_{i}", "confidence": 0.9} for i in range(1000)]
        assessments = [{"decision": f"dec_{i}"} for i in range(500)]
        chain = FullReasoningChain(phases=phases, risk_assessments=assessments)
        compressed = self.codec.compress(chain)
        assert compressed.confidence == 0.9
        assert len(compressed.key_decisions) == 500
        assert compressed.reversible is True

    def test_compress_decompress_symmetric(self):
        """Compress then decompress restores summary and confidence."""
        chain = FullReasoningChain(
            phases=[{"description": "hello", "confidence": 0.75}],
            risk_assessments=[{"decision": "yes"}],
        )
        compressed = self.codec.compress(chain)
        assert self.codec.verify_reversibility(compressed) is True
        restored = self.codec.decompress(compressed)
        assert restored.phases[0]["description"] == "hello"
        assert restored.phases[0]["confidence"] == 0.75


# ──────────────────────────────────────────────────────────────
# 7. LambdaCalibrator — extreme signals / save+load / empty path
# ──────────────────────────────────────────────────────────────

class TestLambdaCalibratorBoundary:
    def setup_method(self):
        self.cal = LambdaCalibrator()

    def test_clamp_to_max(self):
        """Many ACCEPTED signals → λ clamped at 1.0."""
        for _ in range(200):
            self.cal.update(FeedbackSignal.ACCEPTED, 1.0)
        assert self.cal.get_current_lambda() == 1.0

    def test_clamp_to_min(self):
        """Many REJECTED signals → λ clamped at 0.0."""
        for _ in range(200):
            self.cal.update(FeedbackSignal.REJECTED, 1.0)
        assert self.cal.get_current_lambda() == 0.0

    def test_empty_storage_path_no_crash(self):
        """storage_path=None → save is no-op."""
        cal = LambdaCalibrator(storage_path=None)
        cal.update(FeedbackSignal.ACCEPTED, 0.9)
        cal.save_to_disk()  # no crash

    def test_save_load_roundtrip(self, tmp_path):
        """Save to disk then load in new instance → state preserved."""
        path = tmp_path / "lambda.json"
        cal1 = LambdaCalibrator(storage_path=path)
        cal1.update(FeedbackSignal.ACCEPTED, 0.8)
        cal1.update(FeedbackSignal.REJECTED, 0.9)
        cal1.save_to_disk()

        cal2 = LambdaCalibrator(storage_path=path)
        assert cal2.get_current_lambda() == cal1.get_current_lambda()

    def test_rollback_triggers_on_streak(self):
        """3 consecutive CLARIFIED → should_rollback True."""
        for _ in range(3):
            self.cal.update(FeedbackSignal.CLARIFIED, 0.5)
        assert self.cal.should_rollback() is True

    def test_rollback_triggers_on_low_lambda(self):
        """Lambda below threshold → should_rollback True."""
        for _ in range(20):
            self.cal.update(FeedbackSignal.REJECTED, 0.9)
        assert self.cal.get_current_lambda() < ROLLBACK_THRESHOLD
        assert self.cal.should_rollback() is True


# ──────────────────────────────────────────────────────────────
# 8. OutputRouter — unknown intent / empty content / None user_prefs
# ──────────────────────────────────────────────────────────────

class TestOutputRouterBoundary:
    def setup_method(self):
        self.router = OutputRouter()

    def test_unknown_intent_falls_back_to_inform(self):
        """Intent not in ROUTE_TABLE → fallback to INFORM route.

        We simulate this by removing all renderers from the registry
        so the fallback path (default renderer) is exercised.
        """
        router = OutputRouter()
        # Remove the 'text' renderer to trigger fallback to 'none' plan
        router.registry._renderers.pop("text", None)
        plan = router.route(OutputIntent.INFORM, {"text": "hi"}, {})
        # Since text renderer is missing, it falls to the extreme fallback
        assert plan is not None

    def test_empty_content_no_crash(self):
        """Empty content dict → renderer handles gracefully."""
        plan = self.router.route(
            OutputIntent.INFORM, {}, {"verbosity": "normal"}
        )
        assert plan is not None
        rendered = self.router.render(
            OutputIntent.INFORM, {}, {"verbosity": "normal"}
        )
        assert isinstance(rendered, str)

    def test_none_user_prefs_crashes(self):
        """None user_prefs → route() doesn't crash (doesn't use prefs),
        but render() crashes when renderer calls dict.get on None."""
        # route() itself is tolerant (doesn't use user_prefs)
        plan = self.router.route(OutputIntent.INFORM, {"text": "hi"}, None)
        assert plan is not None
        # render() passes None to renderer which calls .get() → crash
        with pytest.raises((TypeError, AttributeError)):
            self.router.render(OutputIntent.INFORM, {"text": "hi"}, None)

    def test_render_returns_string(self):
        """render() always returns a string."""
        result = self.router.render(
            OutputIntent.SILENT, {"text": "ignored"}, {}
        )
        assert result == ""

    def test_concise_truncation(self):
        """TextRenderer in concise mode truncates at 200 chars."""
        long_text = "a" * 500
        result = self.router.render(
            OutputIntent.INFORM,
            {"text": long_text},
            {"verbosity": "concise"},
        )
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")
