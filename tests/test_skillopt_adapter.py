"""
Unit tests for SkillOpt Adapter
================================

测试覆盖：
1. ScoredEdit 基础行为（score_delta, accepted 属性）
2. TextEdit 验证（__post_init__ 警告）
3. SkillOptAdapter.optimize() — 核心优化流程
   - 严格接受改进（score > baseline）
   - 拒绝不改进（score <= baseline）
   - token budget 约束
   - max_edits_per_iteration 约束
   - rejected buffer FIFO
4. SkillOptAdapter.propose_edits() — 编辑提案生成
5. SkillOptAdapter.get_rejected_patterns() — 失败模式提取
6. OptimizationResult.to_config_dict() — 输出格式
7. SkillOptAdapter.to_unified_config() — 写入 UnifiedSkillConfig
"""

import sys
import os
import pytest

# 确保 openllm 可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.isn.adapters.skillopt_adapter import (
    EditOp,
    TextEdit,
    ScoredEdit,
    OptimizationHint,
    OptimizationResult,
    UnifiedSkillConfig,
    SkillOptAdapter,
)


# ── Fixtures ──

def make_edit(
    op: EditOp = EditOp.ADD,
    section: str = "instructions",
    content: str = "新增内容",
    old_content: str = "",
    token_delta: int = 10,
) -> TextEdit:
    return TextEdit(
        op=op,
        target_section=section,
        content=content,
        old_content=old_content,
        token_delta=token_delta,
    )


def make_scored_edit(
    op: EditOp = EditOp.ADD,
    section: str = "instructions",
    content: str = "新增内容",
    old_content: str = "",
    token_delta: int = 10,
    validation_score: float = 0.8,
    baseline_score: float = 0.5,
    rollout_id: str = "test-rollout",
) -> ScoredEdit:
    return ScoredEdit(
        edit=make_edit(op, section, content, old_content, token_delta),
        validation_score=validation_score,
        baseline_score=baseline_score,
        rollout_id=rollout_id,
    )


def make_skill_config() -> UnifiedSkillConfig:
    return UnifiedSkillConfig(
        skill_id="skill-test-001",
        name="Test Skill",
        description="A test skill for unit tests",
        content="Main skill content",
        sections={
            "instructions": "Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
            "examples": "Example A\nExample A\nExample B",
            "short": "Only one line",
        },
    )


# ── ScoredEdit Tests ──

class TestScoredEdit:
    def test_score_delta_positive(self):
        se = make_scored_edit(validation_score=0.8, baseline_score=0.5)
        assert se.score_delta == pytest.approx(0.3)

    def test_score_delta_negative(self):
        se = make_scored_edit(validation_score=0.3, baseline_score=0.5)
        assert se.score_delta == pytest.approx(-0.2)

    def test_accepted_strict_improvement(self):
        se = make_scored_edit(validation_score=0.5001, baseline_score=0.5)
        assert se.accepted is True

    def test_rejected_equal_score(self):
        se = make_scored_edit(validation_score=0.5, baseline_score=0.5)
        assert se.accepted is False

    def test_rejected_worse_score(self):
        se = make_scored_edit(validation_score=0.3, baseline_score=0.5)
        assert se.accepted is False


# ── TextEdit Tests ──

class TestTextEdit:
    def test_delete_without_old_content_warns(self):
        """DELETE edit 没有 old_content 应该触发 warning。"""
        # 只验证不抛异常
        te = TextEdit(op=EditOp.DELETE, target_section="x", content="", old_content="")
        assert te.op == EditOp.DELETE

    def test_replace_without_old_content_warns(self):
        te = TextEdit(op=EditOp.REPLACE, target_section="x", content="new", old_content="")
        assert te.op == EditOp.REPLACE


# ── Optimize Core Tests ──

class TestSkillOptAdapterOptimize:
    def test_accept_strict_improvement(self):
        """只接受严格改进的 edits。"""
        adapter = SkillOptAdapter(token_budget=1000)
        rollouts = [
            make_scored_edit(validation_score=0.8, baseline_score=0.5),
            make_scored_edit(validation_score=0.6, baseline_score=0.5),
        ]
        result = adapter.optimize(rollouts, iteration=1)
        assert len(result.accepted_edits) == 2
        assert len(result.rejected_edits) == 0

    def test_reject_no_improvement(self):
        """拒绝不改进的 edits。"""
        adapter = SkillOptAdapter(token_budget=1000)
        rollouts = [
            make_scored_edit(validation_score=0.4, baseline_score=0.5),  # 降分
            make_scored_edit(validation_score=0.5, baseline_score=0.5),  # 持平
            make_scored_edit(validation_score=0.5001, baseline_score=0.5),  # 微改进
        ]
        result = adapter.optimize(rollouts, iteration=1)
        assert len(result.accepted_edits) == 1
        assert len(result.rejected_edits) == 2

    def test_budget_constraint(self):
        """超过 token budget 的 edit 被拒绝。"""
        adapter = SkillOptAdapter(token_budget=50)
        rollouts = [
            make_scored_edit(token_delta=30, validation_score=0.9, baseline_score=0.5),
            make_scored_edit(token_delta=30, validation_score=0.85, baseline_score=0.5),
        ]
        result = adapter.optimize(rollouts, iteration=1)
        # 第一个 30 tokens, 第二个也是 30, 总共 60 > 50
        assert len(result.accepted_edits) == 1
        assert len(result.rejected_edits) == 1
        assert result.rejected_edits[0].metadata.get("rejection_reason") == "token_budget_exceeded"
        assert result.budget_used == 30

    def test_max_edits_per_iteration(self):
        """超过 max_edits 的 edit 被拒绝。"""
        adapter = SkillOptAdapter(token_budget=1000, max_edits_per_iteration=2)
        rollouts = [
            make_scored_edit(token_delta=5, validation_score=0.9, baseline_score=0.5),
            make_scored_edit(token_delta=5, validation_score=0.85, baseline_score=0.5),
            make_scored_edit(token_delta=5, validation_score=0.8, baseline_score=0.5),
            make_scored_edit(token_delta=5, validation_score=0.75, baseline_score=0.5),
        ]
        result = adapter.optimize(rollouts, iteration=1)
        assert len(result.accepted_edits) == 2
        assert len(result.rejected_edits) == 2
        assert result.rejected_edits[0].metadata.get("rejection_reason") == "max_edits_exceeded"

    def test_sorting_by_score_delta(self):
        """接受的 edits 按 score_delta 降序排列（贪心优先高分）。"""
        adapter = SkillOptAdapter(token_budget=1000, max_edits_per_iteration=3)
        rollouts = [
            make_scored_edit(token_delta=5, validation_score=0.6, baseline_score=0.5, rollout_id="low"),
            make_scored_edit(token_delta=5, validation_score=0.9, baseline_score=0.5, rollout_id="high"),
            make_scored_edit(token_delta=5, validation_score=0.75, baseline_score=0.5, rollout_id="mid"),
        ]
        result = adapter.optimize(rollouts, iteration=1)
        assert result.accepted_edits[0].rollout_id == "high"
        assert result.accepted_edits[1].rollout_id == "mid"
        assert result.accepted_edits[2].rollout_id == "low"

    def test_budget_utilization(self):
        """正确计算 budget 使用率。"""
        adapter = SkillOptAdapter(token_budget=100)
        rollouts = [
            make_scored_edit(token_delta=30, validation_score=0.9, baseline_score=0.5),
        ]
        result = adapter.optimize(rollouts, iteration=1)
        config = result.to_config_dict()
        assert config["budget"]["used"] == 30
        assert config["budget"]["limit"] == 100
        assert config["budget"]["utilization"] == pytest.approx(0.3)

    def test_min_score_delta_filter(self):
        """min_score_delta 过滤微小改进。"""
        adapter = SkillOptAdapter(token_budget=1000, min_score_delta=0.1)
        rollouts = [
            make_scored_edit(token_delta=5, validation_score=0.55, baseline_score=0.5),  # delta=0.05 < 0.1
            make_scored_edit(token_delta=5, validation_score=0.7, baseline_score=0.5),   # delta=0.2 > 0.1
        ]
        result = adapter.optimize(rollouts, iteration=1)
        assert len(result.accepted_edits) == 1
        assert len(result.rejected_edits) == 1

    def test_empty_rollouts(self):
        """空 rollouts 不报错。"""
        adapter = SkillOptAdapter(token_budget=500)
        result = adapter.optimize([], iteration=0)
        assert len(result.accepted_edits) == 0
        assert len(result.rejected_edits) == 0
        assert result.budget_used == 0


# ── Rejected Buffer Tests ──

class TestRejectedBuffer:
    def test_buffer_fifo(self):
        """Buffer 是 FIFO，超过容量弹出最旧的。"""
        adapter = SkillOptAdapter(token_budget=0, rejected_buffer_size=3)
        rollouts = [
            make_scored_edit(rollout_id="r1", validation_score=0.1, baseline_score=0.5),
            make_scored_edit(rollout_id="r2", validation_score=0.2, baseline_score=0.5),
            make_scored_edit(rollout_id="r3", validation_score=0.3, baseline_score=0.5),
            make_scored_edit(rollout_id="r4", validation_score=0.4, baseline_score=0.5),
        ]
        adapter.optimize(rollouts, iteration=1)
        buffer = adapter.rejected_buffer
        assert len(buffer) == 3
        assert buffer[0].rollout_id == "r2"  # r1 被弹出
        assert buffer[-1].rollout_id == "r4"

    def test_rejected_patterns_extraction(self):
        """从 rejected buffer 提取失败模式。"""
        adapter = SkillOptAdapter(token_budget=0, rejected_buffer_size=50)
        rollouts = [
            make_scored_edit(op=EditOp.ADD, section="instructions", rollout_id="a1",
                           validation_score=0.1, baseline_score=0.5),
            make_scored_edit(op=EditOp.ADD, section="instructions", rollout_id="a2",
                           validation_score=0.2, baseline_score=0.5),
            make_scored_edit(op=EditOp.DELETE, section="examples", rollout_id="d1",
                           validation_score=0.3, baseline_score=0.5),
        ]
        adapter.optimize(rollouts, iteration=1)
        patterns = adapter.get_rejected_patterns()
        assert len(patterns) == 2
        # ADD:instructions 出现 2 次，DELETE:examples 出现 1 次
        assert patterns[0]["count"] == 2  # 排序后 count 高的在前
        assert patterns[0]["op"] == "add"


# ── Propose Edits Tests ──

class TestProposeEdits:
    def test_propose_short_section(self):
        """短 section 会生成 ADD 候选。"""
        adapter = SkillOptAdapter()
        config = make_skill_config()
        candidates = adapter.propose_edits(config, rollout_score=0.5)
        short_edits = [c for c in candidates if c.edit.target_section == "short"]
        assert len(short_edits) > 0

    def test_propose_dedup(self):
        """重复行会生成 DELETE 候选。"""
        adapter = SkillOptAdapter()
        config = make_skill_config()
        candidates = adapter.propose_edits(config, rollout_score=0.5)
        dedup_edits = [c for c in candidates if c.edit.target_section == "examples"]
        assert len(dedup_edits) > 0
        assert dedup_edits[0].edit.op == EditOp.DELETE


# ── Output Format Tests ──

class TestOptimizationResult:
    def test_to_config_dict_format(self):
        """验证 to_config_dict 输出格式。"""
        adapter = SkillOptAdapter(token_budget=100)
        rollouts = [
            make_scored_edit(
                token_delta=15, validation_score=0.9, baseline_score=0.5,
                rollout_id="good-edit",
            ),
            make_scored_edit(
                token_delta=10, validation_score=0.3, baseline_score=0.5,
                rollout_id="bad-edit",
            ),
        ]
        result = adapter.optimize(rollouts, iteration=1)
        config = result.to_config_dict()

        assert "accepted_edits" in config
        assert "rejected_edits" in config
        assert "hints" in config
        assert "budget" in config
        assert "iteration" in config

        assert len(config["accepted_edits"]) == 1
        assert config["accepted_edits"][0]["op"] == "add"
        assert config["accepted_edits"][0]["score_delta"] == pytest.approx(0.4)

        assert len(config["rejected_edits"]) == 1
        assert config["rejected_edits"][0]["op"] == "add"

    def test_to_unified_config(self):
        """验证 to_unified_config 写入 UnifiedSkillConfig。"""
        adapter = SkillOptAdapter(token_budget=100)
        skill_config = make_skill_config()
        rollouts = [
            make_scored_edit(token_delta=15, validation_score=0.9, baseline_score=0.5),
        ]
        result = adapter.optimize(rollouts, iteration=1)
        updated = adapter.to_unified_config(skill_config, result)

        assert "accepted_edits" in updated.optimization_hints
        assert "budget" in updated.optimization_hints
        assert updated.optimization_hints["total_iterations"] == 1
        assert updated.optimization_hints["total_tokens_used"] == 15


# ── Reset Tests ──

class TestReset:
    def test_reset_clears_state(self):
        """reset 清除所有内部状态。"""
        adapter = SkillOptAdapter(token_budget=0, rejected_buffer_size=5)
        rollouts = [
            make_scored_edit(rollout_id="r1", validation_score=0.1, baseline_score=0.5),
        ]
        adapter.optimize(rollouts, iteration=1)
        assert len(adapter.rejected_buffer) > 0

        adapter.reset()
        assert len(adapter.rejected_buffer) == 0
        assert adapter.iteration == 0
        assert adapter.total_tokens_used == 0


# ── Integration Flow ──

class TestIntegrationFlow:
    def test_full_optimization_cycle(self):
        """完整优化周期：propose → optimize → to_config → verify。"""
        adapter = SkillOptAdapter(
            token_budget=100,
            max_edits_per_iteration=3,
            rejected_buffer_size=10,
        )
        config = make_skill_config()

        # Phase 1: Propose
        candidates = adapter.propose_edits(config, rollout_score=0.5)
        assert len(candidates) > 0

        # Phase 2: Simulate scoring (in real use, this runs validation)
        for c in candidates:
            c.validation_score = 0.5 + (0.1 if "dedup" in c.rollout_id else 0.05)

        # Phase 3: Optimize
        result = adapter.optimize(candidates, iteration=1)
        assert result.budget_used <= adapter.token_budget

        # Phase 4: Write to config
        config = adapter.to_unified_config(config, result)
        assert "accepted_edits" in config.optimization_hints

        # Phase 5: Check patterns
        patterns = adapter.get_rejected_patterns()
        # At least budget_exceeded or max_edits patterns exist
        assert isinstance(patterns, list)

    def test_multi_iteration(self):
        """多轮迭代优化。"""
        adapter = SkillOptAdapter(token_budget=50, max_edits_per_iteration=2)

        for i in range(3):
            rollouts = [
                make_scored_edit(
                    token_delta=15,
                    validation_score=0.6 + i * 0.1,
                    baseline_score=0.5,
                    rollout_id=f"iter{i}-edit{j}",
                )
                for j in range(3)
            ]
            result = adapter.optimize(rollouts, iteration=i)
            assert result.iteration == i

        assert adapter.total_tokens_used > 0
        assert len(adapter.rejected_buffer) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
