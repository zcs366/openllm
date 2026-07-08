"""
Tests for Pipeline + Checkpoint + Sovereignty — P2/P3模块测试
============================================================
"""
import pytest
from openllm.pipeline import MessageQueue
from openllm.protocol import Protocol, MessageType, BodyName
from openllm.governance.checkpoint import ConstitutionalCheckpoint
from openllm.governance.sovereignty import SovereigntyDeclaration, ProposalStatus
from openllm.governance.precedent_log import PrecedentLog, ConflictType


class TestMessageQueue:
    def test_publish_and_subscribe(self):
        """发布-订阅基本功能。"""
        proto = Protocol()
        mq = MessageQueue(proto)
        received = []
        mq.subscribe(BodyName.IOS, handler=lambda e: received.append(e))
        mq.publish(source=BodyName.IAX, target=BodyName.IOS, msg_type=MessageType.HEARTBEAT_PING, payload={"t": 1})
        processed = mq.process_pending()
        assert processed == 1
        assert len(received) == 1
        assert received[0].source == "IAX"

    def test_multiple_subscribers(self):
        """多订阅者。"""
        proto = Protocol()
        mq = MessageQueue(proto)
        h1, h2 = [], []
        mq.subscribe(BodyName.IOS, handler=lambda e: h1.append(e))
        mq.subscribe(BodyName.IOS, handler=lambda e: h2.append(e))
        mq.publish(source=BodyName.IAX, target=BodyName.IOS, msg_type=MessageType.HEARTBEAT_PING, payload={})
        mq.process_pending()
        assert len(h1) == 1 and len(h2) == 1

    def test_statistics(self):
        """统计功能。"""
        proto = Protocol()
        mq = MessageQueue(proto)
        mq.subscribe(BodyName.IOS, handler=lambda e: None)
        mq.publish(source=BodyName.IAX, target=BodyName.IOS, msg_type=MessageType.HEARTBEAT_PING, payload={})
        mq.publish(source=BodyName.ISA, target=BodyName.IOS, msg_type=MessageType.MEMORY_QUERY, payload={})
        mq.process_pending()
        stats = mq.get_statistics()
        assert stats["total_processed"] == 2
        assert stats["total_errors"] == 0

    def test_handler_error(self):
        """处理器异常不影响队列。"""
        proto = Protocol()
        mq = MessageQueue(proto)
        def bad_handler(e): raise ValueError("boom")
        good_received = []
        mq.subscribe(BodyName.IOS, handler=bad_handler)
        mq.subscribe(BodyName.IOS, handler=lambda e: good_received.append(e))
        mq.publish(source=BodyName.IAX, target=BodyName.IOS, msg_type=MessageType.HEARTBEAT_PING, payload={})
        mq.process_pending()
        assert len(good_received) == 1
        assert mq.get_statistics()["total_errors"] == 1


class TestCheckpoint:
    def test_continue_recommendation(self):
        """继续积累判例。"""
        cp = ConstitutionalCheckpoint()
        r = cp.run_checkpoint(month=3, external_references=5, precedent_count=30, test_pass_rate=0.95)
        assert r.recommendation == "continue"
        assert r.signature != ""

    def test_pivot_recommendation(self):
        """6个月零引用→需要重新评估。"""
        cp = ConstitutionalCheckpoint()
        r = cp.run_checkpoint(month=6, external_references=0, precedent_count=50, test_pass_rate=0.9)
        assert r.recommendation == "pivot"

    def test_expand_recommendation(self):
        """判例充足+测试通过→扩展。"""
        cp = ConstitutionalCheckpoint()
        r = cp.run_checkpoint(month=2, external_references=2, precedent_count=100, test_pass_rate=0.9)
        assert r.recommendation == "expand"

    def test_trajectory(self):
        """检查点轨迹。"""
        cp = ConstitutionalCheckpoint()
        cp.run_checkpoint(month=1, external_references=0, precedent_count=10, test_pass_rate=0.8)
        cp.run_checkpoint(month=2, external_references=3, precedent_count=25, test_pass_rate=0.9)
        traj = cp.get_trajectory()
        assert traj["checkpoints"] == 2
        assert traj["ref_trend"] == "growing"


class TestSovereignty:
    def test_full_workflow(self):
        """完整提案-审查-投票流程。"""
        s = SovereigntyDeclaration()
        p = s.submit_proposal(title="拒绝权规范", content="...", author="team")
        assert p.status == ProposalStatus.DRAFT
        s.submit_comment(p.proposal_id, "community_A", "支持")
        s.submit_comment(p.proposal_id, "community_B", "有建议")
        s.vote(p.proposal_id, "community_A", approve=True)
        s.vote(p.proposal_id, "community_B", approve=True)
        p2 = s.get_proposal(p.proposal_id)
        assert p2.votes_for == 2 and p2.votes_against == 0


class TestPrecedentLog:
    def test_multi_day(self):
        """多天判例记录。"""
        log = PrecedentLog()
        for d in range(5):
            log.record(day_number=d+1, scenario=f"day {d+1}", conflict_type=ConflictType.EDGE_CASE,
                       articles_involved=["refusal"], agent_action="acted", agent_reasoning="reason",
                       human_verdict="agent_correct", lesson_learned="learned")
        assert log.verify_chain() is True
        stats = log.get_statistics()
        assert stats["total_precedents"] == 5
        assert stats["days_covered"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
