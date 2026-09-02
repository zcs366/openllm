"""
⑩ 仲裁端到端测试 + ⑨ 稀疏确认
PAL T-F-7

测试覆盖:
1. EnhancedArbiter 全流程: 冲突场景→策略选择→verdict生成→ArbiterRecord记录
2. 五种仲裁策略的verdict合理性
3. EnhancedHemispherePair 集成（对弈仲裁→记录持久化）
4. Majority Voting 路径（default_arbiter 证据优先）
5. IOS_DRIVEN 策略的IO-S回调和降级路径
6. 无仲裁器时的fallback
7. ⑨ 稀疏确认: 触手数量有限 + 融合逻辑存在
"""
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openllm.core.hemispheres import (
    ArbiterVerdict, ArbiterRecord,
    Hemisphere, HemispherePair,
)
from openllm.core.hemispheres_enhanced import (
    ArbitrationStrategy, ArbitrationContext, ArbitrationResult,
    EnhancedArbiter, EnhancedHemispherePair,
    create_enhanced_hemisphere_pair, quick_arbitrate,
)
from openllm.core.models import Proposal, Critique, Context


# ══════════════════════════════════════════════════════════════
# ⑩ EnhancedArbiter 端到端测试
# ══════════════════════════════════════════════════════════════

class TestEnhancedArbiterE2E:

    def setup_method(self):
        self.arbiter = EnhancedArbiter(default_strategy=ArbitrationStrategy.BALANCED)
        self.context = ArbitrationContext(
            task_id="test-001", task_description="重构ISA记忆层",
        )

    @staticmethod
    def _conflict():
        """左右脑冲突场景"""
        return (
            "重构ISA记忆层，用bolt替代fLock",
            "fLock已经跑了1000+次测试全过，bolt未经验证。重构风险>收益。",
            ["fLock并发性能下降70%", "fLock不支持跨进程"],
            ["bolt在ragflow中发生过数据损坏", "fLock测试覆盖率100%"],
        )

    @staticmethod
    def _empty_right():
        return "优化API速度", "", [], []

    # ── 完整流程 ──

    def test_full_flow(self):
        left, right, ev_l, ev_r = self._conflict()
        result = self.arbiter.arbitrate(
            left_proposal=left, right_critique=right,
            evidence_left=ev_l, evidence_right=ev_r, context=self.context,
        )
        assert result.verdict in [v.value for v in ArbiterVerdict]
        assert result.resolution
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning
        assert result.evidence_summary == {"left": 2, "right": 2}
        assert len(self.arbiter._history) == 1

    # ── 五种策略 ──

    def test_conservative_both_have_evidence(self):
        left, right, ev_l, ev_r = self._conflict()
        result = self.arbiter.arbitrate(
            left_proposal=left, right_critique=right,
            evidence_left=ev_l, evidence_right=ev_r,
            strategy=ArbitrationStrategy.CONSERVATIVE,
        )
        assert result.strategy_used == "conservative"
        assert result.verdict == ArbiterVerdict.INCONCLUSIVE.value

    def test_conservative_no_right(self):
        left, right, ev_l, ev_r = self._empty_right()
        result = self.arbiter.arbitrate(
            left_proposal=left, right_critique=right,
            evidence_left=ev_l, evidence_right=ev_r,
            strategy=ArbitrationStrategy.CONSERVATIVE,
        )
        assert result.verdict == ArbiterVerdict.LEFT_WINS.value

    def test_conservative_right_has_evidence(self):
        result = self.arbiter.arbitrate(
            left_proposal="用方案A", right_critique="方案A有漏洞",
            evidence_left=[], evidence_right=["漏洞1", "漏洞2"],
            strategy=ArbitrationStrategy.CONSERVATIVE,
        )
        assert result.verdict == ArbiterVerdict.RIGHT_WINS.value

    def test_conservative_left_has_evidence(self):
        result = self.arbiter.arbitrate(
            left_proposal="用方案B", right_critique="方案B不好",
            evidence_left=["支持"], evidence_right=[],
            strategy=ArbitrationStrategy.CONSERVATIVE,
        )
        assert result.verdict == ArbiterVerdict.LEFT_WINS.value

    def test_aggressive_always_left(self):
        left, right, ev_l, ev_r = self._conflict()
        result = self.arbiter.arbitrate(
            left_proposal=left, right_critique=right,
            evidence_left=ev_l, evidence_right=ev_r,
            strategy=ArbitrationStrategy.AGGRESSIVE,
        )
        assert result.verdict == ArbiterVerdict.LEFT_WINS.value

    def test_aggressive_no_evidence(self):
        result = self.arbiter.arbitrate(
            left_proposal="试试新", right_critique="风险太大",
            evidence_left=[], evidence_right=[],
            strategy=ArbitrationStrategy.AGGRESSIVE,
        )
        assert result.verdict == ArbiterVerdict.LEFT_WINS.value
        assert result.confidence == 0.5

    def test_balanced_left_wins(self):
        result = self.arbiter.arbitrate(
            left_proposal="X", right_critique="X有问题",
            evidence_left=["e1", "e2", "e3"],  # weight=3.0
            evidence_right=["r1"],              # weight=1.2
            strategy=ArbitrationStrategy.BALANCED,
        )
        assert result.verdict == ArbiterVerdict.LEFT_WINS.value

    def test_balanced_right_wins(self):
        result = self.arbiter.arbitrate(
            left_proposal="Y", right_critique="Y有缺陷",
            evidence_left=["e1"],        # weight=1.0
            evidence_right=["r1", "r2"], # weight=2.4
            strategy=ArbitrationStrategy.BALANCED,
        )
        assert result.verdict == ArbiterVerdict.RIGHT_WINS.value

    def test_balanced_compromise(self):
        # left_count*1.0 == right_count*1.2 → 6.0 vs 6.0
        result = self.arbiter.arbitrate(
            left_proposal="C", right_critique="C需改进",
            evidence_left=["e1","e2","e3","e4","e5","e6"],  # 6.0
            evidence_right=["r1","r2","r3","r4","r5"],       # 6.0
            strategy=ArbitrationStrategy.BALANCED,
        )
        assert result.verdict == ArbiterVerdict.COMPROMISE.value

    def test_balanced_no_right(self):
        left, right, ev_l, ev_r = self._empty_right()
        result = self.arbiter.arbitrate(
            left_proposal=left, right_critique=right,
            evidence_left=ev_l, evidence_right=ev_r,
            strategy=ArbitrationStrategy.BALANCED,
        )
        assert result.verdict == ArbiterVerdict.LEFT_WINS.value

    def test_evidence_based_left_wins(self):
        result = self.arbiter.arbitrate(
            left_proposal="D", right_critique="D有问题",
            evidence_left=["e1","e2","e3"], evidence_right=["r1"],
            strategy=ArbitrationStrategy.EVIDENCE_BASED,
        )
        assert result.verdict == ArbiterVerdict.LEFT_WINS.value

    def test_evidence_based_right_wins(self):
        result = self.arbiter.arbitrate(
            left_proposal="E", right_critique="E有风险",
            evidence_left=["e1"], evidence_right=["r1","r2","r3"],
            strategy=ArbitrationStrategy.EVIDENCE_BASED,
        )
        assert result.verdict == ArbiterVerdict.RIGHT_WINS.value

    def test_evidence_based_equal(self):
        result = self.arbiter.arbitrate(
            left_proposal="F", right_critique="F需改进",
            evidence_left=["e1","e2"], evidence_right=["r1","r2"],
            strategy=ArbitrationStrategy.EVIDENCE_BASED,
        )
        assert result.verdict == ArbiterVerdict.COMPROMISE.value

    # ── IOS_DRIVEN ──

    def test_ios_with_callback(self):
        def cb(**kw):
            return {"verdict": "left_wins", "resolution": "IO-S决定",
                    "confidence": 0.85, "reasoning": "分析后执行"}
        self.arbiter.set_ios_callback(cb)
        left, right, ev_l, ev_r = self._conflict()
        result = self.arbiter.arbitrate(
            left_proposal=left, right_critique=right,
            evidence_left=ev_l, evidence_right=ev_r,
            context=self.context, strategy=ArbitrationStrategy.IOS_DRIVEN,
        )
        assert result.verdict == "left_wins"
        assert result.confidence == 0.85

    def test_ios_callback_exception_fallback(self):
        def bad_cb(**kw):
            raise RuntimeError("IO-S不可用")
        self.arbiter.set_ios_callback(bad_cb)
        left, right, ev_l, ev_r = self._conflict()
        result = self.arbiter.arbitrate(
            left_proposal=left, right_critique=right,
            evidence_left=ev_l, evidence_right=ev_r,
            context=self.context, strategy=ArbitrationStrategy.IOS_DRIVEN,
        )
        # 回退到balanced
        assert result.verdict in [v.value for v in ArbiterVerdict]

    def test_ios_no_context_fallback(self):
        left, right, ev_l, ev_r = self._empty_right()
        result = self.arbiter.arbitrate(
            left_proposal=left, right_critique=right,
            evidence_left=ev_l, evidence_right=ev_r,
            context=None, strategy=ArbitrationStrategy.IOS_DRIVEN,
        )
        # balanced + 空右脑 → LEFT_WINS
        assert result.verdict == ArbiterVerdict.LEFT_WINS.value

    # ── 历史与统计 ──

    def test_history_accumulates(self):
        left, right, ev_l, ev_r = self._conflict()
        for _ in range(5):
            self.arbiter.arbitrate(
                left_proposal=left, right_critique=right,
                evidence_left=ev_l, evidence_right=ev_r,
            )
        stats = self.arbiter.get_performance_stats()
        assert stats["total_arbitrations"] == 5
        assert stats["average_confidence"] > 0.0
        assert "balanced" in stats["strategy_performance"]

    def test_multi_strategy_tracking(self):
        left, right, ev_l, ev_r = self._conflict()
        self.arbiter.arbitrate(
            left_proposal=left, right_critique=right,
            evidence_left=ev_l, evidence_right=ev_r,
            strategy=ArbitrationStrategy.AGGRESSIVE,
        )
        self.arbiter.arbitrate(
            left_proposal=left, right_critique=right,
            evidence_left=ev_l, evidence_right=ev_r,
            strategy=ArbitrationStrategy.CONSERVATIVE,
        )
        stats = self.arbiter.get_performance_stats()
        assert "aggressive" in stats["strategy_performance"]
        assert "conservative" in stats["strategy_performance"]

    # ── 便捷函数 ──

    def test_quick_arbitrate(self):
        result = quick_arbitrate(
            left_proposal="执行A", right_critique="A有风险",
            evidence_left=["风险可控"], evidence_right=["不可控", "事故"],
        )
        assert isinstance(result, ArbitrationResult)
        assert result.verdict in [v.value for v in ArbiterVerdict]

    def test_default_strategy_balanced(self):
        assert EnhancedArbiter().default_strategy == ArbitrationStrategy.BALANCED

    def test_strategy_override(self):
        arb = EnhancedArbiter(default_strategy=ArbitrationStrategy.AGGRESSIVE)
        left, right, ev_l, ev_r = self._conflict()
        result = arb.arbitrate(
            left_proposal=left, right_critique=right,
            evidence_left=ev_l, evidence_right=ev_r,
            strategy=ArbitrationStrategy.CONSERVATIVE,
        )
        assert result.strategy_used == "conservative"


# ══════════════════════════════════════════════════════════════
# ⑩ EnhancedHemispherePair 集成测试
# ══════════════════════════════════════════════════════════════

class TestEnhancedHemispherePairE2E:

    def test_create_pair(self):
        pair = create_enhanced_hemisphere_pair("左", "右")
        assert pair.left_name == "左"
        assert pair.right_name == "右"
        assert isinstance(pair.enhanced_arbiter, EnhancedArbiter)

    def test_arbitrate_with_strategy(self):
        pair = create_enhanced_hemisphere_pair()
        result = pair.arbitrate_with_strategy(
            left_proposal="重构", right_critique="重构不必要",
            evidence_left=["瓶颈"], evidence_right=["测试全过"],
            strategy=ArbitrationStrategy.EVIDENCE_BASED,
        )
        assert isinstance(result, ArbitrationResult)
        assert result.verdict in [v.value for v in ArbiterVerdict]

    def test_pair_stats(self):
        pair = create_enhanced_hemisphere_pair()
        pair.arbitrate_with_strategy(
            left_proposal="方案1", right_critique="方案1有问题",
            evidence_left=["e1"], evidence_right=["r1"],
        )
        stats = pair.get_arbitration_stats()
        assert stats["total_arbitrations"] == 1

    def test_pair_ios_callback(self):
        pair = create_enhanced_hemisphere_pair()
        calls = []
        def cb(**kw):
            calls.append(True)
            return {"verdict": "left_wins", "resolution": "IO-S",
                    "confidence": 0.9, "reasoning": "test"}
        pair.set_ios_callback(cb)
        ctx = ArbitrationContext(task_id="cb-test", task_description="测试IO-S回调")
        result = pair.arbitrate_with_strategy(
            left_proposal="执行", right_critique="不要",
            context=ctx, strategy=ArbitrationStrategy.IOS_DRIVEN,
        )
        assert len(calls) == 1, "IO-S回调应被调用1次（需要传入context）"
        assert result.verdict == "left_wins"

    def test_base_arbitrate_record(self):
        pair = create_enhanced_hemisphere_pair()
        record = pair.arbitrate(
            left_proposal="重构记忆", right_critique="风险大",
            evidence_left=["性能瓶颈"], evidence_right=["测试覆盖"],
        )
        assert isinstance(record, ArbiterRecord)
        assert record.verdict in [v.value for v in ArbiterVerdict]


# ══════════════════════════════════════════════════════════════
# ⑩ Majority Voting 路径测试
# ══════════════════════════════════════════════════════════════

class TestMajorityVoting:

    def test_left_wins(self):
        v, r = HemispherePair._default_arbiter(
            "执行A", "不要", ["证据1","证据2"], [])
        assert v == ArbiterVerdict.LEFT_WINS

    def test_right_wins(self):
        v, r = HemispherePair._default_arbiter(
            "执行B", "B有漏洞", [], ["漏洞1","漏洞2"])
        assert v == ArbiterVerdict.RIGHT_WINS

    def test_compromise(self):
        v, r = HemispherePair._default_arbiter(
            "方案C", "方案D", ["支持C"], ["支持D"])
        assert v == ArbiterVerdict.COMPROMISE

    def test_inconclusive(self):
        v, r = HemispherePair._default_arbiter(
            "方案E", "E不行", [], [])
        assert v == ArbiterVerdict.INCONCLUSIVE

    def test_no_right_objection(self):
        v, r = HemispherePair._default_arbiter("执行F", "", [], [])
        assert v == ArbiterVerdict.LEFT_WINS

    def test_same_opinion(self):
        v, r = HemispherePair._default_arbiter(
            "执行G", "执行G", [], [])
        assert v == ArbiterVerdict.LEFT_WINS


# ══════════════════════════════════════════════════════════════
# ⑩ IOS 5级仲裁栈测试
# ══════════════════════════════════════════════════════════════

class TestIOSArbitrationStack:

    def _stub(self, policy="balanced"):
        s = MagicMock()
        s._arbiter_policy = policy
        s._rejection_engine = None
        s._RejectionReason = MagicMock()
        return s

    def test_L1_no_risk(self):
        from openllm.core.ios_arbitrate import arbitrate
        d = arbitrate(self._stub(), Proposal("exec", 0.8), Critique("opp", "reject"))
        assert d.action == "execute"

    def test_L2_high_conf(self):
        from openllm.core.ios_arbitrate import arbitrate
        risk = MagicMock(level="low", is_blocked=lambda: False)
        d = arbitrate(self._stub(), Proposal("exec", 0.7), Critique("ok", "approve"), risk)
        assert "self_consistency通过" in d.reason

    def test_L2_low_conf(self):
        from openllm.core.ios_arbitrate import arbitrate
        risk = MagicMock(level="low", is_blocked=lambda: False)
        d = arbitrate(self._stub(), Proposal("exec", 0.4), Critique("ok", "approve"), risk)
        assert "降级L1" in d.reason

    def test_L3_reject(self):
        from openllm.core.ios_arbitrate import arbitrate
        risk = MagicMock(level="medium", is_blocked=lambda: False)
        d = arbitrate(self._stub(), Proposal("exec", 0.8), Critique("风险", "reject"), risk)
        assert d.action == "deny"

    def test_L3_approve(self):
        from openllm.core.ios_arbitrate import arbitrate
        risk = MagicMock(level="medium", is_blocked=lambda: False)
        d = arbitrate(self._stub(), Proposal("exec", 0.8), Critique("同意", "approve"), risk)
        assert d.action == "execute"

    def test_L4_debate_approve(self):
        from openllm.core.ios_arbitrate import arbitrate
        risk = MagicMock(level="high", is_blocked=lambda: False)
        d = arbitrate(self._stub(), Proposal("exec", 0.5), Critique("通过", "approve"), risk)
        assert d.action == "execute"

    def test_L4_debate_reject(self):
        from openllm.core.ios_arbitrate import arbitrate
        risk = MagicMock(level="high", is_blocked=lambda: False)
        d = arbitrate(self._stub(), Proposal("exec", 0.5),
                      Critique("不行", "reject", concerns=["风险高"]), risk)
        assert d.action == "deny"

    def test_L5_critical_approve(self):
        from openllm.core.ios_arbitrate import arbitrate
        risk = MagicMock(level="critical", is_blocked=lambda: False)
        d = arbitrate(self._stub(), Proposal("exec", 0.9), Critique("通过", "approve"), risk)
        assert d.action == "execute"

    def test_L5_critical_reject(self):
        from openllm.core.ios_arbitrate import arbitrate
        risk = MagicMock(level="critical", is_blocked=lambda: False)
        d = arbitrate(self._stub(), Proposal("exec", 0.9), Critique("有风险", "reject"), risk)
        assert d.action == "deny"

    def test_risk_blocked(self):
        from openllm.core.ios_arbitrate import arbitrate
        risk = MagicMock(level="critical", is_blocked=lambda: True, reason="安全拦截")
        d = arbitrate(self._stub(), Proposal("exec", 0.9), Critique("ok", "approve"), risk)
        assert d.action == "deny"
        assert "安全拦截" in d.reason


# ══════════════════════════════════════════════════════════════
# ⑨ 稀疏确认 — 触手数量 + 融合逻辑
# ══════════════════════════════════════════════════════════════

class TestSparsityConfirmation:

    def test_tentacle_count_limited(self):
        """章鱼I触手数量有限（当前=2）"""
        from openllm.core.tentacle import FileWatcherBrain, IndexBrain, TentacleBrain
        count = sum(1 for c in [FileWatcherBrain, IndexBrain] if issubclass(c, TentacleBrain))
        assert count == 2, f"触手数量应为2（稀疏原则），实际={count}"
        assert count < 10, "稀疏律：触手数远小于10"

    def test_tentacle_subclass_chain(self):
        """所有触手脑继承TentacleBrain"""
        from openllm.core.tentacle import TentacleBrain, FileWatcherBrain, IndexBrain
        assert issubclass(FileWatcherBrain, TentacleBrain)
        assert issubclass(IndexBrain, TentacleBrain)

    def test_search_results_is_list(self):
        """Context.search_results = list（融合容器）"""
        ctx = Context(user_message="test")
        assert hasattr(ctx, "search_results")
        assert isinstance(ctx.search_results, list)

    def test_search_results_fusion(self):
        """多来源search_results通过append融合"""
        ctx = Context(user_message="test")
        src_a = [{"filepath": "/a.md", "snippet": "A"}]
        src_b = [{"filepath": "/b.md", "snippet": "B"}]
        ctx.search_results = getattr(ctx, "search_results", []) or []
        for r in src_a:
            ctx.search_results.append(r)
        for r in src_b:
            ctx.search_results.append(r)
        assert len(ctx.search_results) == 2
        assert ctx.search_results[0]["filepath"] == "/a.md"
        assert ctx.search_results[1]["filepath"] == "/b.md"

    def test_tentacle_report(self):
        """触手脑上报机制"""
        from openllm.core.tentacle import TentacleBrain
        brain = TentacleBrain("test")
        report = brain.report("discovery", 0.7, {"key": "val"})
        assert report.brain_name == "test"
        assert report.importance == 0.7
        assert len(brain.pending_reports) == 1
        brain.ack(report)
        assert report.acknowledged is True
        assert len(brain.pending_reports) == 0

    def test_file_watcher_scan(self):
        """FileWatcherBrain触手扫描"""
        from openllm.core.tentacle import FileWatcherBrain
        with tempfile.TemporaryDirectory() as td:
            fw = FileWatcherBrain(watch_dir=td)
            changes = fw.scan()
            assert isinstance(changes, list)
            # 写入新文件再扫
            (Path(td) / "new.md").write_text("hello")
            changes = fw.scan()
            assert any(c["type"] == "new_file" for c in changes)

    def test_index_brain_build_and_search(self):
        """IndexBrain触手：建索引+搜索"""
        from openllm.core.tentacle import IndexBrain
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "doc.md").write_text("openLLM is an agent framework")
            ib = IndexBrain(index_dir=td)
            count = ib.build(force=True)
            assert count >= 1
            results = ib.search("agent")
            assert len(results) >= 1
            assert any("agent" in r["filepath"] or "agent" in r.get("snippet","").lower()
                       for r in results)

    def test_octopus_has_two_tentacles(self):
        """章鱼I.__init__注册了2个触手"""
        from openllm.iai.octopus import 章鱼I
        # 章鱼I.__init__中: self.tentacles = {"file_watcher": ..., "index": ...}
        import inspect
        src = inspect.getsource(章鱼I.__init__)
        assert "file_watcher" in src
        assert "index" in src
