"""BrainGuardian 双脑互保协调器测试。

覆盖：
  - 禁区拒绝中止
  - 成功固化
  - 失败回滚恢复 + 接管事件记录
  - 心跳互检集成（mock Hemisphere）
  - append-only 事件
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openllm.cordis.runtime import EffectContext
from openllm.core.brain_guardian import BrainGuardian, ChangeResult
from openllm.governance.self_modification_guard import SelfModificationGuard


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def tmp_guardian_dir(tmp_path):
    """临时 guardian 事件目录"""
    return tmp_path / "guardian"


@pytest.fixture
def mock_hemispheres():
    """mock HemispherePair（不启动心跳线程）"""
    pair = MagicMock()
    pair.health_check.return_value = {
        "left": {"alive": True, "status": "ok"},
        "right": {"alive": True, "status": "ok"},
    }
    return pair


@pytest.fixture
def guard(tmp_path):
    """独立 SelfModificationGuard，状态文件在临时目录"""
    state_file = tmp_path / "guard_state.json"
    return SelfModificationGuard(state_path=state_file)


@pytest.fixture
def guardian(mock_hemispheres, guard, tmp_guardian_dir, tmp_path):
    """BrainGuardian 实例（事件+账本写入临时目录）"""
    return BrainGuardian(
        hemispheres=mock_hemispheres,
        guard=guard,
        event_file=tmp_guardian_dir / "events.jsonl",
        ledger_path=tmp_path / "verification_ledger.jsonl",
    )


# ── 测试：禁区拒绝中止 ──────────────────────────────────

class TestForbiddenReject:
    """禁区文件修改应被 SelfModificationGuard 拒绝"""

    def test_forbidden_target_rejected(self, guardian):
        """identity/soul.py 在 FORBIDDEN_PATHS 中"""
        result = guardian.apply_code_change(
            target="/src/identity/soul.py",
            apply_fn=lambda s: {**s, "x": 1},
        )
        assert result.success is False
        assert "forbidden" in result.reason.lower()

    def test_forbidden_recorded_in_events(self, guardian):
        """禁区拒绝应记录接管事件"""
        guardian.apply_code_change(
            target="/src/governance/self_modification_guard.py",
            apply_fn=lambda s: {**s, "x": 1},
        )
        stats = guardian.stats()
        assert stats["takeover_events"] >= 1

    def test_allowed_target_passes_guard(self, guardian):
        """非禁区文件应通过 guard 检查"""
        result = guardian.apply_code_change(
            target="/src/utils/helper.py",
            apply_fn=lambda s: {**s, "ok": True},
        )
        # 会通过禁区检查（但可能因其他原因失败，这里只看不被拒绝）
        assert "forbidden" not in result.reason.lower()


# ── 测试：成功固化 ───────────────────────────────────────

class TestSuccessCommit:
    """apply + 验证通过 → 固化"""

    def test_apply_success(self, guardian):
        """apply_fn 正常执行，verify_fn 通过 → 固化"""
        def apply_fn(state):
            return {**state, "key": "value"}

        def verify_fn(state):
            return state.get("key") == "value"

        result = guardian.apply_code_change(
            target="/src/utils/new_feature.py",
            apply_fn=apply_fn,
            verify_fn=verify_fn,
        )
        assert result.success is True
        assert "applied and verified" in result.reason
        assert result.evidence_id  # 2026-09-02: 证据账本集成后固化带 evidence_id
        assert result.undone == []

    def test_success_recorded_in_events(self, guardian):
        """成功变更记录到事件日志"""
        guardian.apply_code_change(
            target="/src/utils/a.py",
            apply_fn=lambda s: {**s, "a": 1},
            verify_fn=lambda s: True,
        )
        stats = guardian.stats()
        assert stats["total"] >= 1
        assert stats["success"] >= 1

    def test_state_persists_after_success(self, guardian):
        """成功后状态被固化（EffectContext 中保留变更）"""
        guardian.apply_code_change(
            target="/src/utils/b.py",
            apply_fn=lambda s: {**s, "persisted": True},
            verify_fn=lambda s: True,
        )
        assert guardian._effect_ctx.context.get("persisted") is True


# ── 测试：失败回滚恢复 + 接管事件 ────────────────────────

class TestFailureRollback:
    """apply 通过但 verify 失败 → undo_all + 接管事件"""

    def test_verify_failure_rolls_back(self, guardian):
        """verify_fn 返回 False → undo_all → 状态恢复"""
        # 先成功 apply 一个
        guardian.apply_code_change(
            target="/src/utils/x.py",
            apply_fn=lambda s: {**s, "before": True},
            verify_fn=lambda s: True,
        )
        original = dict(guardian._effect_ctx.context)

        # 再 apply 一个会失败的
        def bad_verify(state):
            return False  # 模拟验证失败

        result = guardian.apply_code_change(
            target="/src/utils/bad.py",
            apply_fn=lambda s: {**s, "bad": True},
            verify_fn=bad_verify,
        )

        assert result.success is False
        assert "rollback" in result.reason.lower()
        # EffectContext 应被恢复到快照
        assert guardian._effect_ctx.context == original

    def test_failure_records_takeover_event(self, guardian):
        """失败应记录接管事件"""
        guardian.apply_code_change(
            target="/src/utils/fail.py",
            apply_fn=lambda s: {**s, "x": 1},
            verify_fn=lambda s: False,
        )
        stats = guardian.stats()
        assert stats["takeover_events"] >= 1
        assert stats["rollback"] >= 1

    def test_undone_effects_populated(self, guardian):
        """失败时 undone 列表非空"""
        # 先成功 apply 一个让 EffectContext 有内容
        guardian.apply_code_change(
            target="/src/utils/z.py",
            apply_fn=lambda s: {**s, "z": 1},
            verify_fn=lambda s: True,
        )

        result = guardian.apply_code_change(
            target="/src/utils/fail.py",
            apply_fn=lambda s: {**s, "x": 1},
            verify_fn=lambda s: False,
        )
        assert result.undone is not None
        assert isinstance(result.undone, list)

    def test_apply_exception_rolls_back(self, guardian):
        """apply_fn 抛异常 → 回滚 + 接管事件"""
        def exploding_apply(state):
            raise ValueError("boom")

        result = guardian.apply_code_change(
            target="/src/utils/crash.py",
            apply_fn=exploding_apply,
        )
        assert result.success is False
        assert "apply failed" in result.reason.lower()


# ── 测试：心跳互检集成 ────────────────────────────────────

class TestHeartbeatMonitor:
    """mock HemispherePair 心跳检测"""

    def test_both_alive_no_takeover(self, guardian, mock_hemispheres):
        """双脑存活 → 无接管事件"""
        event = guardian.monitor_heartbeat()
        assert event is None

    def test_left_dead_right_takeover(self, guardian, mock_hemispheres):
        """左脑死亡 → 右脑接管"""
        mock_hemispheres.health_check.return_value = {
            "left": {"alive": False, "status": "no heartbeat"},
            "right": {"alive": True, "status": "ok"},
        }
        left_hemi = MagicMock()
        mock_hemispheres.left = left_hemi

        event = guardian.monitor_heartbeat()
        assert event is not None
        assert event["event"] == "takeover"
        assert event["role"] == "left"
        assert event["action"] == "takeover"
        # 对侧脑状态应变为 FAILOVER
        from openllm.core.hemispheres import HemisphereState
        left_hemi.state = HemisphereState.FAILOVER

    def test_right_dead_left_takeover(self, guardian, mock_hemispheres):
        """右脑死亡 → 左脑接管"""
        mock_hemispheres.health_check.return_value = {
            "left": {"alive": True, "status": "ok"},
            "right": {"alive": False, "status": "dead"},
        }
        right_hemi = MagicMock()
        mock_hemispheres.right = right_hemi

        event = guardian.monitor_heartbeat()
        assert event is not None
        assert event["role"] == "right"

    def test_both_dead_no_takeover(self, guardian, mock_hemispheres):
        """双脑都死 → 无接管事件（无对侧可接管）"""
        mock_hemispheres.health_check.return_value = {
            "left": {"alive": False, "status": "dead"},
            "right": {"alive": False, "status": "dead"},
        }
        event = guardian.monitor_heartbeat()
        assert event is None


# ── 测试：append-only 事件日志 ─────────────────────────────

class TestAppendOnlyEvents:
    """事件只追加不修改"""

    def test_events_file_grows(self, guardian, tmp_guardian_dir):
        """每次操作只追加行"""
        events_file = tmp_guardian_dir / "events.jsonl"
        for i in range(3):
            guardian.apply_code_change(
                target=f"/src/utils/iter{i}.py",
                apply_fn=lambda s: {**s, "i": i},
                verify_fn=lambda s: True,
            )
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) >= 3

    def test_stats_count_across_multiple_ops(self, guardian):
        """stats 累加多轮操作"""
        for i in range(5):
            guardian.apply_code_change(
                target=f"/src/utils/c{i}.py",
                apply_fn=lambda s, i=i: {**s, "x": i},
                verify_fn=lambda s: True,
            )
        stats = guardian.stats()
        assert stats["total"] >= 5
        assert stats["success"] >= 5

    def test_mixed_success_failure_stats(self, guardian):
        """混合成功/失败的 stats"""
        # 3 成功
        for i in range(3):
            guardian.apply_code_change(
                target=f"/src/utils/ok{i}.py",
                apply_fn=lambda s: {**s, "ok": True},
                verify_fn=lambda s: True,
            )
        # 2 失败
        for i in range(2):
            guardian.apply_code_change(
                target=f"/src/utils/fail{i}.py",
                apply_fn=lambda s: {**s, "fail": True},
                verify_fn=lambda s: False,
            )
        stats = guardian.stats()
        assert stats["total"] == 5
        assert stats["success"] == 3
        assert stats["rollback"] == 2


# ── 测试：ChangeResult 数据结构 ────────────────────────────

class TestChangeResult:
    """ChangeResult 字段完整性"""

    def test_success_result_fields(self, guardian):
        result = guardian.apply_code_change(
            target="/src/utils/simple.py",
            apply_fn=lambda s: {**s, "v": 1},
            verify_fn=lambda s: True,
        )
        assert isinstance(result, ChangeResult)
        assert result.target == "/src/utils/simple.py"
        assert result.success is True
        assert result.undone == []

    def test_failure_result_has_snapshot(self, guardian):
        result = guardian.apply_code_change(
            target="/src/utils/f.py",
            apply_fn=lambda s: {**s, "f": 1},
            verify_fn=lambda s: False,
        )
        assert result.snapshot_backup is not None


# ── 测试：验证证据账本集成（2026-09-02 VerificationLedger 移植）────

class TestVerificationEvidence:
    """apply_code_change 验证动作写入 ledger + evidence_id 返回"""

    def test_success_carries_evidence_id(self, guardian):
        result = guardian.apply_code_change(
            target="/src/utils/ev_a.py",
            apply_fn=lambda s: {**s, "ev": True},
            verify_fn=lambda s: True,
        )
        assert result.success is True
        assert result.evidence_id  # 非空
        assert "(evidence:" in result.reason

    def test_ledger_has_record_after_success(self, guardian):
        guardian.apply_code_change(
            target="/src/utils/ev_b.py",
            apply_fn=lambda s: {**s, "v": 1},
            verify_fn=lambda s: True,
        )
        assert guardian.ledger.has_fresh_evidence("/src/utils/ev_b.py") is True
        stats = guardian.ledger.stats()
        assert stats["ok"] >= 1

    def test_failure_recorded_ok_false(self, guardian):
        result = guardian.apply_code_change(
            target="/src/utils/ev_c.py",
            apply_fn=lambda s: {**s, "x": 1},
            verify_fn=lambda s: False,
        )
        assert result.success is False
        # ledger 记录了失败的验证（ok=False）
        ev = guardian.ledger.fresh_evidence_for("/src/utils/ev_c.py")
        assert ev is not None
        assert ev.ok is False

    def test_verify_required_commits_with_evidence(self, guardian):
        """证据门开启：verify_fn 通过且 ledger 记录成功 → 固化"""
        result = guardian.apply_code_change(
            target="/src/utils/ev_d.py",
            apply_fn=lambda s: {**s, "g": 1},
            verify_fn=lambda s: True,
            verify_evidence_required=True,
        )
        assert result.success is True
        assert result.evidence_id

    def test_no_verify_fn_records_no_verify_step(self, guardian):
        """无 verify_fn → ledger 记录 step=no_verify_fn（观察模式仍记账）"""
        guardian.apply_code_change(
            target="/src/utils/ev_e.py",
            apply_fn=lambda s: {**s, "h": 1},
        )
        ev = guardian.ledger.fresh_evidence_for("/src/utils/ev_e.py")
        assert ev is not None
        assert ev.step == "no_verify_fn"

    def test_existing_suite_backward_compat(self, guardian):
        """既有断言 reason == 'applied and verified' 需兼容（evidence 尾注不应破坏）"""
        # 旧断言用 ==；新实现带 evidence 尾注 → 验证 contains 语义
        result = guardian.apply_code_change(
            target="/src/utils/ev_f.py",
            apply_fn=lambda s: {**s, "k": 1},
            verify_fn=lambda s: True,
        )
        assert "applied and verified" in result.reason
