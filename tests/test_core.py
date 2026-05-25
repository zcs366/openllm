"""
OpenLLM Phase 0 端到端测试。

验证：跨会话记忆回环——"老搭档接上"。
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from openllm.core.loop import AgentLoop, AgentState, LoopPhase
from openllm.memory.capsule import (
    MemoryOS, TextCapsule, DeltaCapsule,
    Arbitrator, should_checkpoint, CAPSULE_DIM,
)
from openllm.identity.soul import Soul, IamPrinciples, IdentityReconstructor, IdentityLevel
from openllm.security.gate import SecurityFoundation, PermissionGate, AuditLog, PermissionLevel


# ── Agent Loop 测试 ──────────────────────────────

class TestAgentLoop:
    def test_initial_state(self):
        loop = AgentLoop()
        assert loop.state == AgentState.WAKING
        assert loop.turn_count == 0

    def test_wake(self):
        loop = AgentLoop()
        loop.wake({"prompt": "test"}, {"status": "empty"})
        assert loop.state == AgentState.WAKING

    def test_turn_advances(self):
        loop = AgentLoop()
        loop.wake({"prompt": "test"}, {"status": "empty"})
        ctx = loop.turn("你好")
        assert loop.turn_count == 1
        assert ctx.user_input == "你好"
        assert ctx.phase == LoopPhase.REFLECT

    def test_overload_detection(self):
        loop = AgentLoop(max_context_tokens=100, context_used=90)
        v = loop.check_vitals()
        assert v["overloaded"] is True
        assert loop.state == AgentState.OVERLOADED

    def test_sleep(self):
        loop = AgentLoop()
        loop.wake({"prompt": "test"}, {"status": "empty"})
        loop.turn("测试")
        delta = loop.sleep()
        assert loop.state == AgentState.SLEEPING
        assert "turns" in delta
        assert delta["turns"] == 1


# ── Memory OS 测试 ──────────────────────────────

class TestMemoryOS:
    def test_write_and_read_text(self):
        """v0.6文本胶囊写入与读取。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mos = MemoryOS(Path(tmpdir))
            text = TextCapsule(
                session_id="test-001",
                decisions=[{"summary": "测试决策"}],
                insights=["关键洞察"],
            )
            path = mos.write(text)
            assert Path(path).exists()

            ctx = mos.read()
            assert ctx["status"] == "restored"
            assert "测试决策" in str(ctx)

    def test_empty_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mos = MemoryOS(Path(tmpdir))
            ctx = mos.read()
            assert ctx["status"] == "empty"

    def test_delta_accumulation(self):
        """Δ向量累加。"""
        d1 = DeltaCapsule(
            session_id="s1",
            vector=np.ones(CAPSULE_DIM, dtype=np.float32),
        )
        d2 = DeltaCapsule(
            session_id="s2",
            vector=np.ones(CAPSULE_DIM, dtype=np.float32) * 2,
        )
        d1.accumulate(d2.vector)
        expected = np.ones(CAPSULE_DIM) * 3
        assert np.allclose(d1.vector, expected, atol=0.01)

    def test_checkpoint_trigger(self):
        """检查点触发条件。"""
        assert should_checkpoint(5, 0.1) is True   # N≥5
        assert should_checkpoint(3, 0.5) is True   # 范数>0.3
        assert should_checkpoint(3, 0.1) is False  # 都不满足

    def test_write_and_read_with_checkpoint(self):
        """带检查点的完整读写流程。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mos = MemoryOS(Path(tmpdir))
            # 写入5次触发检查点
            for i in range(5):
                text = TextCapsule(session_id=f"s{i}", insights=[f"洞察{i}"])
                delta = DeltaCapsule(
                    session_id=f"s{i}",
                    vector=np.ones(CAPSULE_DIM) * 0.05,
                )
                mos.write(text, delta)

            # 检查点文件应存在
            cps = list(Path(tmpdir).glob("checkpoint_*.json"))
            assert len(cps) >= 1

            # 读取应恢复
            ctx = mos.read()
            assert ctx["status"] == "restored"


class TestArbitrator:
    def test_resolve(self):
        text = TextCapsule(session_id="s1", insights=["测试"])
        delta = DeltaCapsule(session_id="s1", vector=np.zeros(CAPSULE_DIM))
        ctx = Arbitrator.resolve(text, delta)
        assert ctx["source"] == "text_primary_delta_secondary"

    def test_conflict_detection(self):
        t1 = TextCapsule(session_id="s1")
        d1 = DeltaCapsule(session_id="s1", vector=np.zeros(CAPSULE_DIM))
        assert Arbitrator.conflict_check(t1, d1) is True

        t2 = TextCapsule(session_id="s2")
        assert Arbitrator.conflict_check(t2, d1) is False


# ── Identity 测试 ──────────────────────────────

class TestIdentity:
    def test_soul_creation(self):
        soul = Soul()
        assert soul.name == "OpenLLM"
        assert len(soul.identity_hash) == 12

    def test_soul_prompt_levels(self):
        soul = Soul()
        l1 = soul.to_prompt(IdentityLevel.L1_SKELETON)
        l4 = soul.to_prompt(IdentityLevel.L4_FULL)
        assert len(l1) > 0
        assert len(l4) > len(l1)
        assert "OpenLLM" in l1

    def test_iam_principles(self):
        iam = IamPrinciples()
        assert len(iam.rules) == 19
        assert iam.can_modify(4) is True    # 普通规则可修改
        assert iam.can_modify(1) is False   # #1 不可修改
        assert iam.can_modify(13) is False  # #13 不可修改

    def test_iam_prompt_marks_immutable(self):
        iam = IamPrinciples()
        prompt = iam.to_prompt()
        assert "🔒" in prompt   # 不可修改标记

    def test_reconstructor(self):
        soul = Soul()
        iam = IamPrinciples()
        recon = IdentityReconstructor(soul, iam)

        # 老搭档+战略 → L4
        level = recon.select_level({"relationship_depth": "老搭档", "topic": "战略"})
        assert level == IdentityLevel.L4_FULL

        # 新用户 → L1
        level = recon.select_level({"relationship_depth": "unknown", "topic": "general"})
        assert level == IdentityLevel.L1_SKELETON

        # 重建
        prompt = recon.reconstruct(
            {"relationship_depth": "老搭档", "topic": "一般"},
            {"status": "restored", "decisions": ["测试"], "insights": [], "unresolved": []}
        )
        assert "OpenLLM" in prompt
        assert "Iam" in prompt


# ── Security 测试 ──────────────────────────────

class TestSecurity:
    def test_audit_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog(Path(tmpdir) / "audit.jsonl")
            log.record("test", PermissionLevel.READ_ONLY, "tester", "allowed")
            entries = log.read()
            assert len(entries) == 1
            assert entries[0]["action"] == "test"

    def test_permission_gate_read(self):
        audit = AuditLog(Path("caps") / "test_audit.jsonl")
        gate = PermissionGate(audit, PermissionLevel.READ_ONLY)
        ok, _ = gate.check("read_file")
        assert ok is True

    def test_permission_gate_deny_write(self):
        audit = AuditLog(Path("caps") / "test_audit.jsonl")
        gate = PermissionGate(audit, PermissionLevel.READ_ONLY)
        ok, reason = gate.check("write_file")
        assert ok is False
        assert "权限不足" in reason

    def test_immutable_action_denied(self):
        audit = AuditLog(Path("caps") / "test_audit.jsonl")
        gate = PermissionGate(audit, PermissionLevel.NETWORK)
        ok, reason = gate.check("disable_security")
        assert ok is False
        assert "安全基座" in reason

    def test_security_foundation_integrity(self):
        sf = SecurityFoundation()
        assert sf.verify_integrity() is True


# ── 端到端：跨会话记忆回环 ──────────────────────

class TestEndToEnd:
    """核心验证：模拟两次会话，确认记忆恢复。"""

    def test_cross_session_remember(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # === 会话1 ===
            mos1 = MemoryOS(tmp)
            text1 = TextCapsule(
                session_id="s1",
                decisions=[{"summary": "确认了架构方向"}],
                insights=["Agent需要六维躯体：记忆+工具+安全+身份+元认知+自修"],
                unresolved=["OpenLLM和CC在工具层的差异需要进一步对齐"],
            )
            delta1 = DeltaCapsule(
                session_id="s1",
                vector=np.random.randn(CAPSULE_DIM).astype(np.float32) * 0.1,
            )
            mos1.write(text1, delta1)

            # === 会话2（新进程模拟）===
            mos2 = MemoryOS(tmp)
            ctx = mos2.read()

            # 验证：记忆恢复成功
            assert ctx["status"] == "restored"
            assert "确认了架构方向" in str(ctx)
            assert "六维躯体" in str(ctx)
            assert "工具层的差异" in str(ctx)

    def test_full_wake_sleep_cycle(self):
        """完整苏醒→工作→休眠循环。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Session 1: 工作→休眠
            mos = MemoryOS(tmp)
            loop = AgentLoop()
            soul = Soul()
            iam = IamPrinciples()
            recon = IdentityReconstructor(soul, iam)
            sf = SecurityFoundation(tmp)

            # 苏醒
            mem_ctx = mos.read()
            identity = recon.reconstruct(
                {"relationship_depth": "老搭档", "topic": "架构"},
                mem_ctx,
            )
            loop.wake({"prompt": identity, "hash": soul.identity_hash}, mem_ctx)
            assert loop.state == AgentState.WAKING

            # 工作
            ctx = loop.turn("我们来讨论架构。")
            assert loop.turn_count == 1
            assert len(ctx.plan) >= 0  # plan可能为空（无回调），但不报错

            # 安全操作
            ok, _ = sf.check_action("read_memory")
            assert ok is True

            # 休眠：保存记忆
            text = TextCapsule(
                session_id=f"s{int(time.time())}",
                decisions=[{"summary": "架构讨论"}],
                insights=["Agent需要六维躯体"],
            )
            delta_vec = np.random.randn(CAPSULE_DIM).astype(np.float32) * 0.05
            delta = DeltaCapsule(session_id=text.session_id, vector=delta_vec)
            mos.write(text, delta)
            loop.sleep()
            assert loop.state == AgentState.SLEEPING

            # Session 2: 苏醒→验证记忆
            mos2 = MemoryOS(tmp)
            ctx2 = mos2.read()
            assert ctx2["status"] == "restored"
            assert "六维躯体" in str(ctx2)
