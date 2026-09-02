"""test_curriculum.py — CurriculumManager 测试。

军规七：正常+边界+异常三类全覆盖。
"""
import json
import time
from pathlib import Path

import pytest

from openllm.iai.curriculum import (
    CONFIDENCE_THRESHOLD,
    CLOUD_EXPLORE_MIN_RATIO,
    TRAIN_THRESHOLD,
    CurriculumManager,
    _default_state,
)


@pytest.fixture
def tmp_state(tmp_path):
    """提供隔离的账本文件。"""
    return tmp_path / "curriculum_state.json"


@pytest.fixture
def mgr(tmp_state):
    """创建一个干净的 CurriculumManager。"""
    return CurriculumManager(state_path=tmp_state)


# ═══════════════════════════════════════════════════════════
# 1. 账本初始化 & 持久化
# ═══════════════════════════════════════════════════════════

class TestStateLifecycle:
    def test_file_not_exist_init_default(self, tmp_state):
        assert not tmp_state.exists()
        m = CurriculumManager(state_path=tmp_state)
        assert m.state["pending_pairs"] == 0
        assert m.state["last_trained"] == 0.0
        assert m.state["confidence_threshold"] == CONFIDENCE_THRESHOLD

    def test_persist_and_reload(self, tmp_state):
        m1 = CurriculumManager(state_path=tmp_state)
        m1._state["pending_pairs"] = 42
        m1._save_state()
        assert tmp_state.exists()

        m2 = CurriculumManager(state_path=tmp_state)
        assert m2.state["pending_pairs"] == 42

    def test_corrupt_file_fallback(self, tmp_state):
        tmp_state.parent.mkdir(parents=True, exist_ok=True)
        tmp_state.write_text("NOT JSON!!!", encoding="utf-8")
        m = CurriculumManager(state_path=tmp_state)
        assert m.state["pending_pairs"] == 0  # fallback to default


# ═══════════════════════════════════════════════════════════
# 2. 置信度过滤（<0.6 拒收）
# ═══════════════════════════════════════════════════════════

class TestConfidenceFilter:
    def test_high_confidence_accepted(self, mgr):
        pairs = [{"chosen": "a", "rejected": "b"}]
        result = mgr.add_pairs("corrections", pairs, confidence=0.8)
        assert result["accepted"] == 1
        assert result["rejected"] == 0
        assert mgr.state["pending_pairs"] == 1
        assert mgr.state["source_counts"]["corrections"] == 1

    def test_low_confidence_rejected(self, mgr):
        pairs = [{"chosen": "a", "rejected": "b"}]
        result = mgr.add_pairs("critique", pairs, confidence=0.4)
        assert result["accepted"] == 0
        assert result["rejected"] == 1
        assert mgr.state["pending_pairs"] == 0
        assert mgr.state["total_rejected"] == 1

    def test_boundary_exactly_threshold(self, mgr):
        """边界：恰好等于阈值应该通过。"""
        pairs = [{"chosen": "a", "rejected": "b"}]
        result = mgr.add_pairs("corrections", pairs, confidence=CONFIDENCE_THRESHOLD)
        assert result["accepted"] == 1
        assert result["rejected"] == 0

    def test_boundary_just_below_threshold(self, mgr):
        """边界：差0.01应该拒收。"""
        pairs = [{"chosen": "a", "rejected": "b"}]
        result = mgr.add_pairs("corrections", pairs, confidence=CONFIDENCE_THRESHOLD - 0.01)
        assert result["accepted"] == 0
        assert result["rejected"] == 1

    def test_multiple_pairs_batch(self, mgr):
        pairs = [{"chosen": f"c{i}", "rejected": f"r{i}"} for i in range(5)]
        result = mgr.add_pairs("cloud_explore", pairs, confidence=0.7)
        assert result["accepted"] == 5
        assert mgr.state["pending_pairs"] == 5
        assert mgr.state["source_counts"]["cloud_explore"] == 5

    def test_zero_confidence_rejected(self, mgr):
        pairs = [{"chosen": "x", "rejected": "y"}]
        result = mgr.add_pairs("corrections", pairs, confidence=0.0)
        assert result["rejected"] == 1


# ═══════════════════════════════════════════════════════════
# 3. should_train 阈值
# ═══════════════════════════════════════════════════════════

class TestShouldTrain:
    def test_below_threshold_no_train(self, mgr):
        mgr._state["pending_pairs"] = TRAIN_THRESHOLD - 1
        mgr._state["last_trained"] = time.time()  # 刚训练过
        assert mgr.should_train() is False

    def test_at_threshold_should_train(self, mgr):
        mgr._state["pending_pairs"] = TRAIN_THRESHOLD
        assert mgr.should_train() is True

    def test_above_threshold_should_train(self, mgr):
        mgr._state["pending_pairs"] = TRAIN_THRESHOLD + 50
        assert mgr.should_train() is True

    def test_timeout_forces_train(self, mgr):
        """超过24h即使样本不够也强制训练。"""
        mgr._state["pending_pairs"] = 10
        mgr._state["last_trained"] = time.time() - 25 * 3600  # 25h ago
        assert mgr.should_train() is True

    def test_zero_pending_no_timeout(self, mgr):
        """0条样本+刚训练过→不训练。"""
        mgr._state["pending_pairs"] = 0
        mgr._state["last_trained"] = time.time()
        assert mgr.should_train() is False

    def test_empty_initial_no_train(self, mgr):
        """初始状态不训练（0 pending, 0 last_trained → 未超时）。"""
        assert mgr.state["pending_pairs"] == 0
        assert mgr.should_train() is False


# ═══════════════════════════════════════════════════════════
# 4. build_training_set 混合比例（≥30% 云探索）
# ═══════════════════════════════════════════════════════════

class TestBuildTrainingSet:
    def test_sufficient_cloud_no_deficit(self, mgr):
        mgr._state["source_counts"] = {
            "corrections": 30,
            "critique": 30,
            "cloud_explore": 40,  # 40/100 = 40% ≥ 30%
        }
        result = mgr.build_training_set()
        assert result["stats"]["deficit"] is False
        assert result["stats"]["cloud_ratio"] >= CLOUD_EXPLORE_MIN_RATIO
        assert result["stats"]["total"] == 100

    def test_insufficient_cloud_deficit(self, mgr):
        mgr._state["source_counts"] = {
            "corrections": 60,
            "critique": 30,
            "cloud_explore": 10,  # 10/100 = 10% < 30%
        }
        result = mgr.build_training_set()
        assert result["stats"]["deficit"] is True
        assert "云端探索缺口" in result["stats"]["deficit_note"]
        assert result["stats"]["cloud_ratio"] == pytest.approx(0.1, abs=0.01)

    def test_no_data_no_deficit(self, mgr):
        """空数据不算 deficit。"""
        result = mgr.build_training_set()
        assert result["stats"]["deficit"] is False
        assert result["stats"]["total"] == 0

    def test_only_cloud_explore(self, mgr):
        """全是云探索→无缺口。"""
        mgr._state["source_counts"] = {"cloud_explore": 50}
        result = mgr.build_training_set()
        assert result["stats"]["cloud_ratio"] == 1.0
        assert result["stats"]["deficit"] is False

    def test_exact_boundary_30_percent(self, mgr):
        """恰好30%→无缺口。"""
        mgr._state["source_counts"] = {
            "corrections": 42,
            "critique": 28,
            "cloud_explore": 30,  # 30/100 = 30%
        }
        result = mgr.build_training_set()
        assert result["stats"]["deficit"] is False
        assert result["stats"]["cloud_ratio"] == pytest.approx(0.3, abs=0.01)

    def test_training_set_structure(self, mgr):
        """训练集结构验证。"""
        mgr._state["source_counts"] = {
            "corrections": 5,
            "cloud_explore": 5,
        }
        result = mgr.build_training_set()
        assert "chosen" in result
        assert "rejected" in result
        assert "stats" in result
        assert len(result["chosen"]) == 10
        assert len(result["rejected"]) == 10


# ═══════════════════════════════════════════════════════════
# 5. schedule_nightly 生成命令
# ═══════════════════════════════════════════════════════════

class TestScheduleNightly:
    def test_default_command(self, mgr):
        cmd = mgr.schedule_nightly()
        assert "python3" in cmd
        assert "train_qlora.py" in cmd
        assert "--resume" in cmd
        assert "--epochs 2" in cmd
        assert "--batch-size 2" in cmd
        assert "--lr 2e-4" in cmd

    def test_custom_params(self, mgr):
        cmd = mgr.schedule_nightly(epochs=3, resume=False)
        assert "--epochs 3" in cmd
        assert "--resume" not in cmd

    def test_recorded_in_history(self, mgr):
        mgr.schedule_nightly()
        assert len(mgr.state["train_history"]) == 1
        assert mgr.state["train_history"][0]["status"] == "scheduled"

    def test_no_resume_flag(self, mgr):
        cmd = mgr.schedule_nightly(resume=False)
        assert "--resume" not in cmd

    def test_custom_data_dir(self, mgr):
        cmd = mgr.schedule_nightly(data_dir="/tmp/mydata")
        assert "--data-dir /tmp/mydata" in cmd


# ═══════════════════════════════════════════════════════════
# 6. verify_and_rollback 回归门
# ═══════════════════════════════════════════════════════════

class TestVerifyAndRollback:
    def test_success_log_passes(self, mgr):
        mgr.schedule_nightly()
        log = "Epoch 1/2 ... Epoch 2/2 ... ✅ 训练完成 | save_pretrained done"
        assert mgr.verify_and_rollback(log) is True
        assert mgr.state["pending_pairs"] == 0
        assert mgr.state["train_history"][-1]["status"] == "success"

    def test_failure_log_triggers_rollback(self, mgr):
        mgr.schedule_nightly()
        log = "CUDA OOM error at step 100, killing process"
        assert mgr.verify_and_rollback(log) is False
        assert mgr.state["train_history"][-1]["status"] == "failed"
        assert mgr.state["train_history"][-1]["rollback_triggered"] is True

    def test_success_updates_last_trained(self, mgr):
        before = time.time()
        mgr.schedule_nightly()
        log = "✅ all done"
        mgr.verify_and_rollback(log)
        assert mgr.state["last_trained"] >= before

    def test_failure_preserves_last_trained(self, mgr):
        mgr._state["last_trained"] = 100.0
        mgr.schedule_nightly()
        log = "FATAL error"
        mgr.verify_and_rollback(log)
        assert mgr.state["last_trained"] == 100.0  # unchanged

    def test_empty_log_fails(self, mgr):
        mgr.schedule_nightly()
        assert mgr.verify_and_rollback("") is False

    def test_multiple_success_markers(self, mgr):
        """只要包含任一标记即为成功。"""
        mgr.schedule_nightly()
        for marker in ["训练完成", "✅", "save_pretrained"]:
            log = f"Some output ... {marker} ... end"
            assert mgr.verify_and_rollback(log) is True


# ═══════════════════════════════════════════════════════════
# 7. 端到端流程
# ═══════════════════════════════════════════════════════════

class TestEndToEnd:
    def test_full_cycle(self, mgr):
        """完整流程：添加→判断→构建→调度→验证。"""
        # 1. 添加样本（高置信度通过）
        for _ in range(50):
            mgr.add_pairs("corrections", [{"chosen": "c", "rejected": "r"}], 0.9)
        for _ in range(30):
            mgr.add_pairs("critique", [{"chosen": "c", "rejected": "r"}], 0.8)
        for _ in range(20):
            mgr.add_pairs("cloud_explore", [{"chosen": "c", "rejected": "r"}], 0.7)

        assert mgr.state["pending_pairs"] == 100

        # 2. 判断是否训练
        assert mgr.should_train() is True

        # 3. 构建训练集
        ts = mgr.build_training_set()
        assert ts["stats"]["total"] == 100
        assert ts["stats"]["cloud_ratio"] == pytest.approx(0.2, abs=0.01)
        assert ts["stats"]["deficit"] is True  # 20% < 30%

        # 4. 调度训练
        cmd = mgr.schedule_nightly()
        assert "train_qlora.py" in cmd

        # 5. 验证通过
        assert mgr.verify_and_rollback("✅ 训练完成") is True
        assert mgr.state["pending_pairs"] == 0

    def test_low_confidence_does_not_pollute(self, mgr):
        """低置信度样本不应污染 source_counts。"""
        mgr.add_pairs("corrections", [{"chosen": "c", "rejected": "r"}], 0.3)
        mgr.add_pairs("corrections", [{"chosen": "c", "rejected": "r"}], 0.9)
        assert mgr.state["pending_pairs"] == 1
        assert mgr.state["source_counts"]["corrections"] == 1
        assert mgr.state["total_rejected"] == 1
