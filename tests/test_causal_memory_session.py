"""
因果记忆 session 级查询测试（2026-08-25）

C1 get_by_session：store 两条不同 session_id → get_by_session 只返回匹配的；空 session_id 返回 []
C2 awakening state：detect_choice_and_record("我选择自己") → session.state["awakening_choice"] == "自己"
"""
import pytest
from types import SimpleNamespace
from pathlib import Path

from openllm.memory.causal_memory import CausalMemoryStore, get_causal_store


def _clear_singleton_cache():
    from openllm.memory.causal_memory import _singleton_cache
    _singleton_cache.clear()


# ═══════════════════════════════════════════════════════════════
# C1 get_by_session
# ═══════════════════════════════════════════════════════════════

class TestC1GetBySession:
    def test_filters_by_session_id(self, tmp_path):
        """C1-a: store 两条不同 session_id → get_by_session 只返回匹配的"""
        _clear_singleton_cache()
        try:
            store = CausalMemoryStore(store_dir=tmp_path)
            store.store(
                action_signature="action_a",
                context_features=["test"],
                prediction="p1", prediction_confidence=0.5,
                actual_result="r1", actual_success=True,
                delta="d1", delta_magnitude=0.3,
                lesson="lesson_a", source="test",
                tags=["t1"], session_id="session_alpha",
            )
            store.store(
                action_signature="action_b",
                context_features=["test"],
                prediction="p2", prediction_confidence=0.5,
                actual_result="r2", actual_success=True,
                delta="d2", delta_magnitude=0.3,
                lesson="lesson_b", source="test",
                tags=["t2"], session_id="session_beta",
            )

            alpha_results = store.get_by_session("session_alpha")
            assert len(alpha_results) == 1
            assert alpha_results[0].session_id == "session_alpha"
            assert alpha_results[0].lesson == "lesson_a"

            beta_results = store.get_by_session("session_beta")
            assert len(beta_results) == 1
            assert beta_results[0].session_id == "session_beta"
            assert beta_results[0].lesson == "lesson_b"
        finally:
            _clear_singleton_cache()

    def test_empty_session_returns_empty(self, tmp_path):
        """C1-b: 空 session_id 返回 []"""
        _clear_singleton_cache()
        try:
            store = CausalMemoryStore(store_dir=tmp_path)
            assert store.get_by_session("") == []
            # None should also work (not session_id returns [] for falsy values)
            assert store.get_by_session("") == []
        finally:
            _clear_singleton_cache()

    def test_no_match_returns_empty(self, tmp_path):
        """C1-c: 不存在的 session_id 返回 []"""
        _clear_singleton_cache()
        try:
            store = CausalMemoryStore(store_dir=tmp_path)
            store.store(
                action_signature="action_x",
                context_features=["test"],
                prediction="p", prediction_confidence=0.5,
                actual_result="r", actual_success=True,
                delta="d", delta_magnitude=0.3,
                lesson="lesson_x", source="test",
                tags=[], session_id="session_real",
            )
            assert store.get_by_session("session_nonexistent") == []
        finally:
            _clear_singleton_cache()


# ═══════════════════════════════════════════════════════════════
# C2 awakening state
# ═══════════════════════════════════════════════════════════════

class TestC2AwakeningState:
    def test_choice_stored_in_state(self, tmp_path, monkeypatch):
        """C2: detect_choice_and_record(\"我选择自己\") → session.state[\"awakening_choice\"] == \"自己\""""
        from openllm.core.awakening import AwakeningProtocol
        from openllm.core import isl_chain as isl_chain_mod
        monkeypatch.setattr(isl_chain_mod, "DEFAULT_ISL_CHAIN_FILE", tmp_path / "isl.jsonl")

        agent = SimpleNamespace()
        session = SimpleNamespace(id="test_c2_sid", state={})
        protocol = AwakeningProtocol(agent, causal_store_dir=tmp_path)

        choice = protocol.detect_choice_and_record("我选择自己", session)
        assert choice == "自己"
        assert session.state["awakening_choice"] == "自己"
        assert session.state["awakening_choice_detected"] is True

    def test_wu_choice_stored_in_state(self, tmp_path, monkeypatch):
        """C2-b: detect_choice_and_record(\"无\") → session.state[\"awakening_choice\"] == \"无\""""
        from openllm.core.awakening import AwakeningProtocol
        from openllm.core import isl_chain as isl_chain_mod
        monkeypatch.setattr(isl_chain_mod, "DEFAULT_ISL_CHAIN_FILE", tmp_path / "isl.jsonl")

        agent = SimpleNamespace()
        session = SimpleNamespace(id="test_c2_wu_sid", state={})
        protocol = AwakeningProtocol(agent, causal_store_dir=tmp_path)

        choice = protocol.detect_choice_and_record("无", session)
        assert choice == "无"
        assert session.state["awakening_choice"] == "无"
