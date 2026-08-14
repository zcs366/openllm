"""
Unit tests for SkillOS Adapter
==============================

测试覆盖：
1. SkillUsageHistory — 使用历史数据模型
2. SkillScore — RL 风格多维度评分
3. CurationPolicy — 策展策略参数验证
4. SkillOSAdapter.score_skill() — 评分方法
5. SkillOSAdapter.recommend_action() — 单技能决策
6. SkillOSAdapter.curate() — 批量策展（任务组、容量、合并候选）
7. SkillOSAdapter.apply_decision() — 决策应用到 UnifiedSkillConfig
8. 边界情况：空历史、无数据、全部失败
"""

import sys
import os
import pytest
from datetime import datetime, timezone, timedelta

# Ensure openllm is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.isn.adapters.skillos_adapter import (
    CurationAction,
    SkillUsageRecord,
    SkillUsageHistory,
    SkillScore,
    CurationDecision,
    CurationPolicy,
    SkillOSAdapter,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

NOW = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
NOW_ISO = NOW.isoformat()


def make_record(
    success: bool = True,
    latency_ms: float = 50.0,
    token_cost: float = 100.0,
    quality_score: float = 0.8,
    days_ago: int = 0,
) -> SkillUsageRecord:
    """创建一条使用记录。"""
    ts = (NOW - timedelta(days=days_ago)).isoformat()
    return SkillUsageRecord(
        timestamp=ts,
        success=success,
        latency_ms=latency_ms,
        token_cost=token_cost,
        quality_score=quality_score,
    )


def make_history(
    name: str = "test-skill",
    calls: int = 10,
    success_rate: float = 0.9,
    task_group_id: str = "group-a",
    curator_score: float = 0.7,
    compression_reward: float = 0.6,
    total_token_size: int = 200,
    content_hash: str = "",
    days_ago_last: int = 1,
) -> SkillUsageHistory:
    """创建技能使用历史。"""
    records = []
    for i in range(calls):
        success = (i / max(calls, 1)) < success_rate
        records.append(make_record(
            success=success,
            days_ago=max(0, days_ago_last - i),
        ))
    return SkillUsageHistory(
        skill_name=name,
        task_group_id=task_group_id,
        records=records,
        curator_score=curator_score,
        compression_reward=compression_reward,
        total_token_size=total_token_size,
        content_hash=content_hash or f"hash-{name}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tests: SkillUsageHistory
# ══════════════════════════════════════════════════════════════════════════════

class TestSkillUsageHistory:
    def test_empty_history(self):
        h = SkillUsageHistory(skill_name="empty")
        assert h.total_calls == 0
        assert h.success_rate == 0.0
        assert h.avg_latency_ms == 0.0
        assert h.avg_token_cost == 0.0
        assert h.avg_quality_score == 0.0
        assert h.last_used_at is None

    def test_single_record(self):
        r = make_record(success=True, latency_ms=100.0, token_cost=200.0, quality_score=0.9)
        h = SkillUsageHistory(skill_name="single", records=[r])
        assert h.total_calls == 1
        assert h.success_rate == 1.0
        assert h.avg_latency_ms == 100.0
        assert h.avg_token_cost == 200.0
        assert h.avg_quality_score == 0.9

    def test_mixed_records(self):
        records = [
            make_record(success=True, token_cost=100.0, quality_score=0.8),
            make_record(success=True, token_cost=200.0, quality_score=0.6),
            make_record(success=False, token_cost=150.0, quality_score=0.4),
        ]
        h = SkillUsageHistory(skill_name="mixed", records=records)
        assert h.total_calls == 3
        assert abs(h.success_rate - 2 / 3) < 0.01
        assert abs(h.avg_token_cost - 150.0) < 0.01
        assert abs(h.avg_quality_score - 0.6) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# Tests: SkillScore
# ══════════════════════════════════════════════════════════════════════════════

class TestSkillScore:
    def test_to_dict(self):
        s = SkillScore(
            success_signal=0.9,
            efficiency_signal=0.7,
            recency_signal=0.5,
            uniqueness_signal=0.8,
            compression_signal=0.6,
            curator_signal=0.75,
            composite_score=0.72,
            confidence=0.85,
        )
        d = s.to_dict()
        assert d["success"] == 0.9
        assert d["composite"] == 0.72
        assert d["confidence"] == 0.85
        assert len(d) == 8

    def test_zero_score(self):
        s = SkillScore()
        d = s.to_dict()
        assert d["composite"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Tests: CurationPolicy
# ══════════════════════════════════════════════════════════════════════════════

class TestCurationPolicy:
    def test_default_valid(self):
        p = CurationPolicy()
        assert p.validate() == []

    def test_invalid_thresholds(self):
        p = CurationPolicy(retire_threshold=0.8, merge_threshold=0.5)
        errors = p.validate()
        assert any("retire_threshold" in e for e in errors)

    def test_invalid_weights(self):
        p = CurationPolicy(weight_success=0.5, weight_efficiency=0.5)
        errors = p.validate()
        # weights don't sum to 1.0
        assert any("weights" in e for e in errors)


# ══════════════════════════════════════════════════════════════════════════════
# Tests: SkillOSAdapter.score_skill()
# ══════════════════════════════════════════════════════════════════════════════

class TestScoreSkill:
    def test_high_quality_skill(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(
            calls=20, success_rate=1.0, curator_score=0.9,
            compression_reward=0.8, days_ago_last=1,
        )
        score = adapter.score_skill(h)
        assert score.success_signal == 1.0
        assert score.curator_signal == 0.9
        assert score.compression_signal == 0.8
        assert score.composite_score > 0.5
        assert score.confidence > 0.5

    def test_low_quality_skill(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(
            calls=5, success_rate=0.2, curator_score=0.1,
            compression_reward=0.1, days_ago_last=30,
        )
        score = adapter.score_skill(h)
        assert score.success_signal < 0.3
        assert score.curator_signal == 0.1
        assert score.composite_score < 0.3

    def test_empty_history_gives_zero(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = SkillUsageHistory(skill_name="empty")
        score = adapter.score_skill(h)
        assert score.composite_score <= 0.01
        assert score.confidence <= 0.01

    def test_recency_decay(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h_recent = make_history(name="recent", days_ago_last=1)
        h_old = make_history(name="old", days_ago_last=90)
        s_recent = adapter.score_skill(h_recent)
        s_old = adapter.score_skill(h_old)
        assert s_recent.recency_signal > s_old.recency_signal

    def test_efficiency_signal(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h_efficient = make_history(
            name="efficient",
            calls=10,
            success_rate=1.0,
        )
        h_efficient.records = [
            make_record(success=True, token_cost=10.0, quality_score=0.9)
            for _ in range(10)
        ]
        h_inefficient = make_history(
            name="inefficient",
            calls=10,
            success_rate=1.0,
        )
        h_inefficient.records = [
            make_record(success=True, token_cost=1000.0, quality_score=0.9)
            for _ in range(10)
        ]
        s_eff = adapter.score_skill(h_efficient)
        s_ineff = adapter.score_skill(h_inefficient)
        assert s_eff.efficiency_signal > s_ineff.efficiency_signal


# ══════════════════════════════════════════════════════════════════════════════
# Tests: SkillOSAdapter.recommend_action()
# ══════════════════════════════════════════════════════════════════════════════

class TestRecommendAction:
    def test_keep_high_quality(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(
            calls=20, success_rate=1.0, curator_score=0.9,
            compression_reward=0.8, days_ago_last=1,
        )
        decision = adapter.recommend_action(h)
        assert decision.action == CurationAction.KEEP
        assert decision.skill_name == "test-skill"

    def test_retire_low_quality(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        # Directly construct with ALL records old
        records = [
            SkillUsageRecord(
                timestamp=(NOW - timedelta(days=120 + i)).isoformat(),
                success=False,
                latency_ms=200,
                token_cost=500,
                quality_score=0.05,
            )
            for i in range(3)
        ]
        h = SkillUsageHistory(
            skill_name="test-skill",
            task_group_id="g1",
            records=records,
            curator_score=0.01,
            compression_reward=0.01,
            total_token_size=3000,
            content_hash="hash-test-skill",
        )
        decision = adapter.recommend_action(h)
        assert decision.action == CurationAction.RETIRE

    def test_merge_medium_quality(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(
            calls=10, success_rate=0.5, curator_score=0.35,
            compression_reward=0.3, days_ago_last=5,
        )
        decision = adapter.recommend_action(h)
        assert decision.action == CurationAction.MERGE

    def test_decision_has_score(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(calls=5, success_rate=0.8)
        decision = adapter.recommend_action(h)
        assert isinstance(decision.score, SkillScore)
        assert 0.0 <= decision.score.composite_score <= 1.0

    def test_decision_has_reason(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(calls=5, success_rate=0.8)
        decision = adapter.recommend_action(h)
        assert len(decision.reason) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Tests: SkillOSAdapter.curate() — Batch Curation
# ══════════════════════════════════════════════════════════════════════════════

class TestCurate:
    def test_empty_input(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        decisions = adapter.curate([])
        assert decisions == []

    def test_single_skill(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(calls=10, success_rate=1.0, curator_score=0.8)
        decisions = adapter.curate([h])
        assert len(decisions) == 1
        assert decisions[0].action in (CurationAction.KEEP, CurationAction.MERGE, CurationAction.RETIRE)

    def test_multiple_skills_same_group(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        skills = [
            make_history(name="skill-a", calls=20, success_rate=1.0, curator_score=0.9, task_group_id="g1"),
            make_history(name="skill-b", calls=15, success_rate=0.8, curator_score=0.7, task_group_id="g1"),
            make_history(name="skill-c", calls=5, success_rate=0.3, curator_score=0.2, task_group_id="g1"),
        ]
        decisions = adapter.curate(skills)
        assert len(decisions) == 3

        # skill-a should be kept
        da = next(d for d in decisions if d.skill_name == "skill-a")
        assert da.action == CurationAction.KEEP

    def test_group_capacity_limit(self):
        """超过 max_skills_per_group 的技能被强制 retire。"""
        policy = CurationPolicy(max_skills_per_group=2)
        adapter = SkillOSAdapter(policy=policy, reference_time=NOW)
        skills = [
            make_history(name=f"skill-{i}", calls=10, success_rate=0.4,
                        curator_score=0.3, task_group_id="g1",
                        days_ago_last=30)
            for i in range(4)
        ]
        decisions = adapter.curate(skills)
        # All 4 should get decisions
        assert len(decisions) == 4
        # At least 2 should be retired due to capacity (only 2 slots)
        retired = [d for d in decisions if d.action == CurationAction.RETIRE]
        assert len(retired) >= 2

    def test_uniqueness_detection(self):
        """重复内容的技能应该降低独特性分。"""
        adapter = SkillOSAdapter(reference_time=NOW)
        skills = [
            make_history(name="skill-a", calls=10, success_rate=0.9,
                        task_group_id="g1", content_hash="same-hash"),
            make_history(name="skill-b", calls=10, success_rate=0.9,
                        task_group_id="g1", content_hash="same-hash"),
            make_history(name="skill-c", calls=10, success_rate=0.9,
                        task_group_id="g1", content_hash="unique-hash"),
        ]
        decisions = adapter.curate(skills)
        da = next(d for d in decisions if d.skill_name == "skill-a")
        dc = next(d for d in decisions if d.skill_name == "skill-c")
        # skill-c (unique) should have higher uniqueness than skill-a (duplicate)
        assert dc.score.uniqueness_signal >= da.score.uniqueness_signal

    def test_different_groups(self):
        """不同任务组的技能独立策展。"""
        adapter = SkillOSAdapter(reference_time=NOW)
        # g1-good: recent, high success
        h_g1 = SkillUsageHistory(
            skill_name="g1-good",
            task_group_id="g1",
            records=[make_record(success=True, days_ago=i) for i in range(20, 0, -1)],
            curator_score=0.9,
            compression_reward=0.8,
            total_token_size=200,
            content_hash="hash-g1-good",
        )
        # g2-bad: very old, all failures, very large
        h_g2 = SkillUsageHistory(
            skill_name="g2-bad",
            task_group_id="g2",
            records=[
                SkillUsageRecord(
                    timestamp=(NOW - timedelta(days=200 + i)).isoformat(),
                    success=False, latency_ms=500, token_cost=2000,
                    quality_score=0.01,
                )
                for i in range(10)
            ],
            curator_score=0.01,
            compression_reward=0.01,
            total_token_size=5000,
            content_hash="hash-g2-bad",
        )
        decisions = adapter.curate([h_g1, h_g2])
        d_g1 = next(d for d in decisions if d.skill_name == "g1-good")
        d_g2 = next(d for d in decisions if d.skill_name == "g2-bad")
        assert d_g1.action == CurationAction.KEEP
        # g2-bad is the only skill in its group — gets MERGE (optimize), not RETIRE
        # because there's no replacement. Uniqueness=1.0 pushes it above retire threshold.
        assert d_g2.action == CurationAction.MERGE

    def test_merge_candidate_detection(self):
        """合并候选配对检测。"""
        adapter = SkillOSAdapter(reference_time=NOW)
        skills = [
            make_history(name="skill-merge-1", calls=8, success_rate=0.4,
                        curator_score=0.35, task_group_id="g1"),
            make_history(name="skill-merge-2", calls=6, success_rate=0.4,
                        curator_score=0.35, task_group_id="g1"),
        ]
        decisions = adapter.curate(skills)
        merge_decisions = [d for d in decisions if d.action == CurationAction.MERGE]
        # At least one should have a merge_target
        if merge_decisions:
            assert any(d.merge_target for d in merge_decisions)


# ══════════════════════════════════════════════════════════════════════════════
# Tests: SkillOSAdapter.apply_decision()
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyDecision:
    def _make_config(self):
        """创建简单的 mock config。"""
        from dataclasses import dataclass, field
        from typing import Any, Optional

        @dataclass
        class MockOptimizationHints:
            should_optimize: bool = False
            priority: int = 0
            reason: str = ""
            suggested_focus: str = ""
            last_analyzed_at: Optional[str] = None
            confidence: float = 0.0
            metadata: dict = field(default_factory=dict)

        @dataclass
        class MockConfig:
            name: str = "test"
            lifecycle_state: str = "active"
            optimization_hints: Optional[MockOptimizationHints] = None

            def transition(self, new_state: str):
                self.lifecycle_state = new_state
                return True

        return MockConfig()

    def test_apply_keep(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(calls=20, success_rate=1.0, curator_score=0.9)
        decision = adapter.recommend_action(h)
        config = self._make_config()
        adapter.apply_decision(config, decision)

        assert config.optimization_hints.should_optimize is False
        assert config.optimization_hints.suggested_focus == "maintain"
        assert "skillos_score" in config.optimization_hints.metadata

    def test_apply_merge(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(calls=8, success_rate=0.5, curator_score=0.35)
        decision = adapter.recommend_action(h)
        decision.merge_target = "better-skill"
        config = self._make_config()
        adapter.apply_decision(config, decision)

        assert config.optimization_hints.should_optimize is True
        assert config.optimization_hints.suggested_focus == "merge"
        assert config.optimization_hints.metadata["merge_target"] == "better-skill"

    def test_apply_retire(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        # All records very old, all failures
        records = [
            SkillUsageRecord(
                timestamp=(NOW - timedelta(days=180 + i)).isoformat(),
                success=False,
                latency_ms=200,
                token_cost=500,
                quality_score=0.05,
            )
            for i in range(3)
        ]
        h = SkillUsageHistory(
            skill_name="test-skill",
            task_group_id="g1",
            records=records,
            curator_score=0.01,
            compression_reward=0.01,
            total_token_size=3000,
            content_hash="hash-test-skill",
        )
        decision = adapter.recommend_action(h)
        config = self._make_config()
        adapter.apply_decision(config, decision)

        assert config.optimization_hints.should_optimize is True
        assert config.optimization_hints.suggested_focus == "retire"
        assert "retirement_reason" in config.optimization_hints.metadata

    def test_apply_creates_hints_if_none(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(calls=5, success_rate=0.8, curator_score=0.7)
        decision = adapter.recommend_action(h)
        config = self._make_config()
        config.optimization_hints = None
        adapter.apply_decision(config, decision)
        assert config.optimization_hints is not None


# ══════════════════════════════════════════════════════════════════════════════
# Tests: Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_adapter_init_invalid_policy(self):
        with pytest.raises(ValueError, match="Invalid curation policy"):
            SkillOSAdapter(policy=CurationPolicy(
                retire_threshold=0.9, merge_threshold=0.5,
            ))

    def test_history_no_quality_score(self):
        """无 quality_score 时用 success_rate 近似。"""
        adapter = SkillOSAdapter(reference_time=NOW)
        h = SkillUsageHistory(
            skill_name="no-quality",
            records=[make_record(success=True, quality_score=None)],
        )
        # avg_quality_score 为 0 (因为 None 被过滤)
        score = adapter.score_skill(h)
        assert score.efficiency_signal >= 0.0

    def test_history_no_last_used(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = SkillUsageHistory(skill_name="no-last-used", records=[])
        score = adapter.score_skill(h)
        assert score.recency_signal == 0.0

    def test_curation_history_tracking(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h1 = make_history(name="s1", calls=10, success_rate=0.9)
        h2 = make_history(name="s2", calls=5, success_rate=0.3)
        adapter.curate([h1, h2])
        assert len(adapter.curation_history) == 2

    def test_reset(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(calls=5, success_rate=0.8)
        adapter.curate([h])
        assert len(adapter.curation_history) > 0
        adapter.reset()
        assert len(adapter.curation_history) == 0

    def test_decision_to_dict(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(calls=10, success_rate=0.9)
        decision = adapter.recommend_action(h)
        d = decision.to_dict()
        assert "skill_name" in d
        assert "action" in d
        assert "score" in d
        assert "reason" in d


# ══════════════════════════════════════════════════════════════════════════════
# Tests: RL Signal Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestRLSignals:
    def test_success_rate_all_pass(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = SkillUsageHistory(
            skill_name="perfect",
            records=[make_record(success=True) for _ in range(10)],
        )
        s = adapter.score_skill(h)
        assert s.success_signal == 1.0

    def test_success_rate_all_fail(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = SkillUsageHistory(
            skill_name="failure",
            records=[make_record(success=False) for _ in range(10)],
        )
        s = adapter.score_skill(h)
        assert s.success_signal == 0.0

    def test_compression_reward_direct(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = SkillUsageHistory(
            skill_name="compressed",
            compression_reward=0.95,
        )
        s = adapter.score_skill(h)
        assert s.compression_signal == 0.95

    def test_compression_signal_from_size(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h_small = SkillUsageHistory(skill_name="small", total_token_size=50)
        h_large = SkillUsageHistory(skill_name="large", total_token_size=2000)
        s_small = adapter.score_skill(h_small)
        s_large = adapter.score_skill(h_large)
        assert s_small.compression_signal > s_large.compression_signal

    def test_confidence_increases_with_calls(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h_few = SkillUsageHistory(
            skill_name="few",
            records=[make_record(success=True) for _ in range(1)],
        )
        h_many = SkillUsageHistory(
            skill_name="many",
            records=[make_record(success=True) for _ in range(20)],
        )
        s_few = adapter.score_skill(h_few)
        s_many = adapter.score_skill(h_many)
        assert s_many.confidence > s_few.confidence

    def test_weighted_composite_sums_to_range(self):
        adapter = SkillOSAdapter(reference_time=NOW)
        h = make_history(calls=10, success_rate=0.8, curator_score=0.7)
        s = adapter.score_skill(h)
        # All signals are [0, 1], weights sum to 1.0, so composite must be [0, 1]
        assert 0.0 <= s.composite_score <= 1.0
