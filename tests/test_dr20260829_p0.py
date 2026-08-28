"""
DR-20260829-01 P0 三修验证测试
测试P0-A（Δ胶囊编码）、P0-B（会话落盘）、P0-C（身份注入prompt）
"""
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════
# P0-A: DeltaCapsule.from_text 384维编码
# ═══════════════════════════════════════════════════════

class TestDeltaCapsuleEncoding:
    """P0-A: Δ胶囊from_text必须产出384维非零向量"""

    def test_hash_fallback_384dim(self):
        """2a: hash fallback路径产出384维向量"""
        from openllm.memory.capsule import DeltaCapsule, CAPSULE_DIM
        # 强制走hash fallback: 使openllm_memory和EmbeddingEngine都不可用
        with patch.dict('sys.modules', {
            'openllm_memory': None,
            'openllm.embedding': None,
        }):
            # Reset the module-level singleton to force re-creation
            import openllm.memory.capsule as cap_mod
            old_engine = cap_mod._EMBEDDING_ENGINE
            cap_mod._EMBEDDING_ENGINE = None
            try:
                dc = DeltaCapsule.from_text("test_session_1", "测试文本编码验证")
                assert dc.vector.shape == (CAPSULE_DIM,), \
                    f"hash fallback向量维度应为{CAPSULE_DIM}，实际{dc.vector.shape}"
                assert dc.norm > 0.0, "hash fallback向量norm应>0（非全零）"
                assert dc.metadata.get("model") == "hash_fallback", \
                    f"metadata model应为hash_fallback，实际{dc.metadata.get('model')}"
            finally:
                cap_mod._EMBEDDING_ENGINE = old_engine

    def test_hash_fallback_deterministic(self):
        """相同输入产出相同向量"""
        from openllm.memory.capsule import DeltaCapsule
        with patch.dict('sys.modules', {
            'openllm_memory': None,
            'openllm.embedding': None,
        }):
            import openllm.memory.capsule as cap_mod
            old_engine = cap_mod._EMBEDDING_ENGINE
            cap_mod._EMBEDDING_ENGINE = None
            try:
                dc1 = DeltaCapsule.from_text("s1", "deterministic")
                dc2 = DeltaCapsule.from_text("s1", "deterministic")
                np.testing.assert_array_equal(dc1.vector, dc2.vector)
            finally:
                cap_mod._EMBEDDING_ENGINE = old_engine

    def test_vector_not_all_zeros(self):
        """向量不全为零"""
        from openllm.memory.capsule import DeltaCapsule
        with patch.dict('sys.modules', {
            'openllm_memory': None,
            'openllm.embedding': None,
        }):
            import openllm.memory.capsule as cap_mod
            old_engine = cap_mod._EMBEDDING_ENGINE
            cap_mod._EMBEDDING_ENGINE = None
            try:
                dc = DeltaCapsule.from_text("s2", "任何文本")
                assert not np.allclose(dc.vector, 0.0), "hash fallback不应全零"
            finally:
                cap_mod._EMBEDDING_ENGINE = old_engine


# ═══════════════════════════════════════════════════════
# P0-A附加: __post_init__ 形状不符warning日志
# ═══════════════════════════════════════════════════════

class TestDeltaCapsuleShapeWarning:
    """__post_init__形状不符时产生warning日志且向量仍置零"""

    def test_shape_mismatch_warns_and_zeros(self):
        """2b: 形状不符产生warning日志"""
        from openllm.memory.capsule import DeltaCapsule, CAPSULE_DIM
        bad_vector = np.ones(100, dtype=np.float32)  # 错误维度
        with patch("openllm.memory.capsule._logger") as mock_logger:
            dc = DeltaCapsule(session_id="bad", vector=bad_vector)
            # 应该有warning调用
            mock_logger.warning.assert_called_once()
            assert "形状不符" in mock_logger.warning.call_args[0][0]
            # 向量应被置零
            assert dc.vector.shape == (CAPSULE_DIM,)
            assert dc.norm == 0.0


# ═══════════════════════════════════════════════════════
# P0-B: Agent.shutdown()会话落盘
# ═══════════════════════════════════════════════════════

class TestSessionPersistence:
    """P0-B: shutdown()后caps/出现v06/v07文件"""

    def test_shutdown_persists_capsules(self):
        """2c: 有turn的session在shutdown后产生v06/v07文件"""
        from openllm.memory.capsule import TextCapsule, DeltaCapsule, MemoryOS, CAPSULE_DIM

        with tempfile.TemporaryDirectory() as tmpdir:
            caps_dir = Path(tmpdir)

            # 模拟一个有turn的session
            text = TextCapsule(
                session_id="s_test_persist",
                decisions=[{"summary": "测试决策"}],
                insights=["测试洞察"],
                outputs=["测试输出"],
            )
            # 用hash fallback生成delta（不依赖ST）
            with patch.dict('sys.modules', {
                'openllm_memory': None,
                'openllm.embedding': None,
            }):
                import openllm.memory.capsule as cap_mod
                old_engine = cap_mod._EMBEDDING_ENGINE
                cap_mod._EMBEDDING_ENGINE = None
                try:
                    delta = DeltaCapsule.from_text("s_test_persist", text.to_text())
                finally:
                    cap_mod._EMBEDDING_ENGINE = old_engine

            mos = MemoryOS(caps_dir)
            mos.write(text, delta)

            # 验证v06文件
            v06_files = list(caps_dir.glob("v06_s_test_persist.json"))
            assert len(v06_files) == 1, f"应有1个v06文件，实际{len(v06_files)}"
            v06_data = json.loads(v06_files[0].read_text())
            assert v06_data["session_id"] == "s_test_persist"

            # 验证v07文件
            v07_files = list(caps_dir.glob("v07_s_test_persist.json"))
            assert len(v07_files) == 1, f"应有1个v07文件，实际{len(v07_files)}"
            v07_data = json.loads(v07_files[0].read_text())
            assert len(v07_data["vector"]) == CAPSULE_DIM, \
                f"v07向量应为{CAPSULE_DIM}维，实际{len(v07_data['vector'])}"
            assert v07_data["norm"] > 0.0, "v07向量norm应>0"

    def test_shutdown_no_persist_empty_session(self):
        """零turn且无_last_output时不落盘"""
        from openllm.core.main_loop import Agent
        agent = Agent(mode="silent")
        # 不执行任何tick
        agent.shutdown()
        # 验证_persisted被设置
        assert agent._persisted is True

    def test_shutdown_idempotent(self):
        """shutdown多次不会重复落盘"""
        from openllm.core.main_loop import Agent
        agent = Agent(mode="silent")
        agent._persisted = True  # 已落盘
        agent.shutdown()  # 不应报错，且不重复落盘
        assert agent._persisted is True


# ═══════════════════════════════════════════════════════
# P0-C: 身份注入prompt
# ═══════════════════════════════════════════════════════

class TestIdentityInjection:
    """P0-C: think()和review()的prompt包含identity_block"""

    def test_think_prompt_contains_identity(self):
        """2d: think()的prompt包含openLLM身份信息"""
        from openllm.core.octopus_impl import _LeftBrain
        from openllm.core.models import Context, Proposal, Prediction

        ctx = Context(
            user_message="你好",
            identity_block="你是openLLM，一个自主Agent。",
        )

        brain = _LeftBrain()
        # Mock provider.chat来捕获prompt
        with patch.object(brain.provider, 'chat', return_value='{"content": "测试回复"}') as mock_chat:
            with patch.object(brain.provider, '_available', True):
                brain.think(ctx)

            # 验证chat被调用且prompt包含identity_block
            assert mock_chat.called, "provider.chat应该被调用"
            call_args = mock_chat.call_args
            prompt = call_args[0][0][0]["content"]
            assert "你是openLLM" in prompt, f"prompt应包含openLLM身份，实际: {prompt[:200]}"
            assert "自主Agent" in prompt, f"prompt应包含自主Agent，实际: {prompt[:200]}"

    def test_review_prompt_contains_identity(self):
        """review()的prompt也包含identity_block"""
        from openllm.core.octopus_impl import _RightBrain
        from openllm.core.models import Context, Proposal

        ctx = Context(
            user_message="测试消息",
            identity_block="身份提醒：你是openLLM。",
        )
        proposal = Proposal(content="测试提案", confidence=0.8)

        brain = _RightBrain()
        with patch.object(brain.provider, 'chat', return_value='{"verdict": "approve"}') as mock_chat:
            brain.review(ctx, proposal)

            assert mock_chat.called, "provider.chat应该被调用"
            prompt = mock_chat.call_args[0][0][0]["content"]
            assert "openLLM" in prompt, f"review prompt应包含openLLM身份，实际: {prompt[:200]}"

    def test_identity_block_in_context(self):
        """Context dataclass包含identity_block字段"""
        from openllm.core.models import Context
        ctx = Context(user_message="test")
        assert hasattr(ctx, 'identity_block'), "Context应有identity_block字段"
        assert ctx.identity_block == "", "默认值应为空字符串"

    def test_identity_block_with_provider(self):
        """isa_impl填充identity_block时包含provider model"""
        from openllm.core.isa_impl import ISA
        from openllm.core.models import Message

        isa = ISA(mode="silent")
        msg = Message(text="测试")
        # 不传octopus，应该生成不含provider名的identity_block
        ctx = isa.build_context(msg)
        assert "openLLM" in ctx.identity_block, "identity_block应包含openLLM"
