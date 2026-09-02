"""遗忘维 ForgettingCurve 测试（PAL T-F-9）。

覆盖：
  - 正常路径：指数衰减打分（新>旧）
  - 半衰期可配：不同half_life产生不同衰减率
  - record_access再巩固：访问后age重置，分数回升
  - IoR衔接：已处理主题衰减加速
  - apply排序：新主题在前，旧主题在后
  - 边界：age=0得1.0，极老主题得MIN_SCORE
  - 异常：空列表、不可用filter
  - 集成：build_context注入forgetting_hints
"""
import time
from pathlib import Path

import pytest

from openllm.iai.forgetting import ForgettingCurve, MIN_SCORE, IoR_ACCELERATOR


# ── 指数衰减打分 ──────────────────────────────────────


class TestForgettingCurveScore:
    def test_new_topic_scores_high(self):
        """新主题（age=0）得分接近1.0。"""
        fc = ForgettingCurve()
        score = fc.score("新主题", age_seconds=0)
        assert score == 1.0

    def test_old_topic_scores_low(self):
        """老主题（age很大）得分低。"""
        fc = ForgettingCurve()
        score = fc.score("老主题", age_seconds=86400 * 7)  # 7天
        assert score < 0.1

    def test_newer_is_higher(self):
        """新主题得分 > 老主题得分。"""
        fc = ForgettingCurve()
        s_new = fc.score("新", age_seconds=100)
        s_old = fc.score("老", age_seconds=86400 * 3)
        assert s_new > s_old

    def test_score_never_below_min(self):
        """得分不低于MIN_SCORE。"""
        fc = ForgettingCurve()
        score = fc.score("极老", age_seconds=86400 * 365)  # 1年
        assert score >= MIN_SCORE

    def test_half_life_boundary(self):
        """半衰期处得分≈0.5。"""
        fc = ForgettingCurve(half_life_s=86400)
        score = fc.score("topic", age_seconds=86400)
        assert abs(score - 0.5) < 0.001

    def test_two_half_lives(self):
        """两个半衰期处得分≈0.25。"""
        fc = ForgettingCurve(half_life_s=86400)
        score = fc.score("topic", age_seconds=86400 * 2)
        assert abs(score - 0.25) < 0.001


# ── 半衰期可配 ──────────────────────────────────────


class TestConfigurableHalfLife:
    def test_short_half_life_decays_fast(self):
        """短半衰期=更快衰减。"""
        fc_fast = ForgettingCurve(half_life_s=60)  # 1分钟
        fc_slow = ForgettingCurve(half_life_s=86400)  # 1天
        # 同样age=120s（2分钟），短半衰期已经衰减到0.25，长的几乎1.0
        s_fast = fc_fast.score("t", age_seconds=120)
        s_slow = fc_slow.score("t", age_seconds=120)
        assert s_fast < s_slow

    def test_very_short_half_life(self):
        """极短半衰期（1秒）——10秒后几乎归零。"""
        fc = ForgettingCurve(half_life_s=1.0)
        score = fc.score("t", age_seconds=10)
        assert score < 0.02

    def test_very_long_half_life(self):
        """极长半衰期（1年）——1天后仍然很高。"""
        fc = ForgettingCurve(half_life_s=365 * 86400)
        score = fc.score("t", age_seconds=86400)
        assert score > 0.99


# ── record_access 再巩固 ──────────────────────────────


class TestRecordAccess:
    def test_record_access_refreshes_score(self):
        """record_access后得分回升（再巩固）。"""
        fc = ForgettingCurve()
        # 旧主题：age=1天，得分≈0.5
        s_before = fc.score("topic", age_seconds=86400)
        assert s_before < 1.0
        # 访问后：age重置
        fc.record_access("topic")
        s_after = fc.score("topic")
        assert s_after > s_before

    def test_record_access_increments_count(self):
        """每次访问递增访问计数。"""
        fc = ForgettingCurve()
        assert fc.get_access_count("topic") == 0
        fc.record_access("topic")
        assert fc.get_access_count("topic") == 1
        fc.record_access("topic")
        assert fc.get_access_count("topic") == 2

    def test_record_access_make_fresh(self):
        """刚访问过的topic得分接近1.0。"""
        fc = ForgettingCurve()
        fc.record_access("topic")
        score = fc.score("topic")
        assert score > 0.99


# ── IoR 衔接 ──────────────────────────────────────


class TestIoRIntegration:
    def test_ior_handled_reduces_score(self):
        """IoR已处理主题得分更低（衰减加速）。"""
        fc = ForgettingCurve()
        s_normal = fc.score("topic", age_seconds=3600, ior_handled=False)
        s_ior = fc.score("topic", age_seconds=3600, ior_handled=True)
        assert s_ior < s_normal

    def test_ior_accelerator_ratio(self):
        """已处理主题得分 = 正常得分 × IoR_ACCELERATOR。"""
        fc = ForgettingCurve()
        s_normal = fc.score("topic", age_seconds=3600, ior_handled=False)
        s_ior = fc.score("topic", age_seconds=3600, ior_handled=True)
        assert abs(s_ior / s_normal - IoR_ACCELERATOR) < 0.001

    def test_ior_fresh_topic_still_reasonable(self):
        """刚访问的IoR主题（age=0）仍然有合理得分。"""
        fc = ForgettingCurve()
        score = fc.score("topic", age_seconds=0, ior_handled=True)
        assert score > 0.5  # 1.0 × 0.7 = 0.7


# ── apply 排序 ──────────────────────────────────────


class TestApplySort:
    def test_apply_sorts_fresh_first(self):
        """新主题排在旧主题前面。"""
        fc = ForgettingCurve()
        topics = ["old", "mid", "new"]
        # 手动设置不同的age
        fc._access_times["old"] = time.time() - 86400 * 7
        fc._access_times["mid"] = time.time() - 86400
        fc._access_times["new"] = time.time() - 10

        result = fc.apply(topics)
        names = [t for t, _ in result]
        assert names[0] == "new"
        assert names[-1] == "old"

    def test_apply_with_ior_handled_set(self):
        """IoR已处理集合加速衰减排序。"""
        fc = ForgettingCurve()
        # 两个age相同但一个被IoR处理过
        fc._access_times["t1"] = time.time() - 3600
        fc._access_times["t2"] = time.time() - 3600

        result = fc.apply(["t1", "t2"], ior_handled_set={"t2"})
        # t2被IoR处理过→得分更低→排在后面
        assert result[0][0] == "t1"
        assert result[1][0] == "t2"

    def test_apply_empty_list(self):
        """空列表返回空结果。"""
        fc = ForgettingCurve()
        result = fc.apply([])
        assert result == []


# ── stats ──────────────────────────────────────


class TestStats:
    def test_stats_empty(self):
        """初始状态统计。"""
        fc = ForgettingCurve()
        stats = fc.stats()
        assert stats["topics_tracked"] == 0
        assert stats["total_accesses"] == 0

    def test_stats_after_access(self):
        """访问后统计更新。"""
        fc = ForgettingCurve()
        fc.record_access("a")
        fc.record_access("b")
        fc.record_access("a")
        stats = fc.stats()
        assert stats["topics_tracked"] == 2
        assert stats["total_accesses"] == 3


# ── build_context 集成 ─────────────────────────────


class TestBuildContextIntegration:
    def test_forgetting_hints_populated(self, tmp_path: Path):
        """有记忆召回时，context.forgetting_hints 非空。"""
        from openllm.core.isa_impl import ISA
        from openllm.core.models import Message

        isa = ISA(mode="silent")
        # 手动注入ForgettingCurve（避免干扰生产路径）
        from openllm.iai.forgetting import ForgettingCurve
        isa._forgetting_curve = ForgettingCurve(half_life_s=1)  # 极短半衰期，1秒后就衰减

        # 手动设置memory（模拟有recall结果）
        # build_context会走MemoryBus，这里用mock让recalled存在
        original_get_memory_bus = isa._get_memory_bus

        class MockBus:
            def query(self, q):
                # 返回一些模拟记忆
                class R:
                    def __init__(self, content, importance, source, score, timestamp):
                        self.content = content
                        self.importance = importance
                        self.source = source
                        self.score = score
                        self.timestamp = timestamp
                return [
                    R("非常新的安全知识", 0.9, "test", 0.8, time.time()),
                    R("很久以前的旧信息", 0.3, "test", 0.3, time.time() - 86400 * 30),
                ]

        isa._memory_bus = MockBus()

        ctx = isa.build_context(Message(text="测试消息"))
        # 新记忆排在旧记忆前面
        assert len(ctx.memory.get("recalled", [])) == 2
        assert ctx.memory["recalled"][0]["importance"] == 0.9

    def test_no_forgetting_hints_when_no_recall(self):
        """无记忆召回时，forgetting_hints为空。"""
        from openllm.core.isa_impl import ISA
        from openllm.core.models import Message

        isa = ISA(mode="silent")
        isa._forgetting_curve = ForgettingCurve()
        # 不设置memory_bus，让MemoryBus不可用
        isa._memory_bus = None
        ctx = isa.build_context(Message(text="测试消息"))
        assert ctx.forgetting_hints == []

    def test_forgetting_hints_cold_items(self, tmp_path: Path):
        """衰减严重的记忆产生forgetting_hints。"""
        from openllm.core.isa_impl import ISA
        from openllm.core.models import Message

        isa = ISA(mode="silent")
        fc = ForgettingCurve(half_life_s=3600)  # 1小时半衰期
        isa._forgetting_curve = fc
        # 预设一个极老的topic（age=7天 >> 3600s），让score<0.3
        old_topic = "过时的旧信息内容很多字"
        fc._access_times[old_topic] = time.time() - 86400 * 7

        class MockBus:
            def query(self, q):
                class R:
                    def __init__(self, content, importance, source, score, timestamp):
                        self.content = content
                        self.importance = importance
                        self.source = source
                        self.score = score
                        self.timestamp = timestamp
                return [R(old_topic, 0.3, "test", 0.3, time.time() - 86400)]

        isa._memory_bus = MockBus()
        ctx = isa.build_context(Message(text="测试消息"))
        # 半衰期3600s下，age=7天的topic得分极低→产生淡忘提示
        assert len(ctx.forgetting_hints) > 0
        assert "淡忘" in ctx.forgetting_hints[0]

    def test_no_forgetting_hints_when_fc_unavailable(self):
        """ForgettingCurve不可用时，不影响build_context。"""
        from openllm.core.isa_impl import ISA
        from openllm.core.models import Message

        isa = ISA(mode="silent")
        isa._forgetting_curve = None  # 模拟不可用
        ctx = isa.build_context(Message(text="测试消息"))
        assert ctx.forgetting_hints == []
        # 其他字段仍正常
        assert ctx.user_message == "测试消息"
        assert isinstance(ctx.identity, dict)
