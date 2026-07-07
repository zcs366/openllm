"""FeedbackLoop 单元测试"""
import os
import json
import tempfile
import pytest
from pathlib import Path

from openllm.governance.feedback_loop import FeedbackRecord, FeedbackStore, FeedbackLoop


@pytest.fixture
def tmp_store(tmp_path):
    """创建临时反馈存储"""
    return str(tmp_path / "feedback.jsonl")


class TestFeedbackRecord:
    def test_create_record(self):
        rec = FeedbackRecord(
            source_body="ISN",
            target_body="IAX",
            round_number=1,
            score=0.85,
            issues=["心跳延迟"],
            suggestion="优化调度频率",
        )
        assert rec.source_body == "ISN"
        assert rec.target_body == "IAX"
        assert rec.round_number == 1
        assert rec.score == 0.85
        assert rec.issues == ["心跳延迟"]
        assert rec.suggestion == "优化调度频率"

    def test_hash_deterministic(self):
        rec1 = FeedbackRecord("ISN", "IAX", 1, 0.8, ["issue"], "fix", prev_hash="abc")
        rec2 = FeedbackRecord("ISN", "IAX", 1, 0.8, ["issue"], "fix", prev_hash="abc")
        assert rec1.compute_hash() == rec2.compute_hash()

    def test_hash_differs_with_prev(self):
        rec1 = FeedbackRecord("ISN", "IAX", 1, 0.8, [], "", prev_hash="")
        rec2 = FeedbackRecord("ISN", "IAX", 1, 0.8, [], "", prev_hash="abc")
        assert rec1.compute_hash() != rec2.compute_hash()

    def test_frozen(self):
        rec = FeedbackRecord("ISN", "IAX", 1, 0.8, [], "")
        with pytest.raises(AttributeError):
            rec.score = 0.9


class TestFeedbackStore:
    def test_append_and_read(self, tmp_store):
        store = FeedbackStore(tmp_store)
        rec = FeedbackRecord("ISN", "IAX", 1, 0.8, ["issue"], "suggestion")
        store.append(rec)
        recent = store.get_recent(5)
        assert len(recent) == 1
        assert recent[0]["source_body"] == "ISN"
        assert recent[0]["target_body"] == "IAX"

    def test_chain_verify(self, tmp_store):
        store = FeedbackStore(tmp_store)
        for i in range(5):
            rec = FeedbackRecord("ISN", "IAX", i, 0.8, [], "")
            store.append(rec)
        assert store.verify_chain() is True

    def test_chain_tamper_detection(self, tmp_store):
        store = FeedbackStore(tmp_store)
        for i in range(3):
            rec = FeedbackRecord("ISN", "IAX", i, 0.8, [], "")
            store.append(rec)

        # 篡改第二条记录的prev_hash
        lines = Path(tmp_store).read_text().strip().split("\n")
        rec = json.loads(lines[1])
        rec["prev_hash"] = "TAMPERED"
        lines[1] = json.dumps(rec)
        Path(tmp_store).write_text("\n".join(lines) + "\n")

        # 重新加载store验证
        store2 = FeedbackStore(tmp_store)
        assert store2.verify_chain() is False

    def test_get_for_body(self, tmp_store):
        store = FeedbackStore(tmp_store)
        store.append(FeedbackRecord("ISN", "IAX", 1, 0.8, [], ""))
        store.append(FeedbackRecord("ISN", "IAI", 1, 0.9, [], ""))
        store.append(FeedbackRecord("IKO", "IAX", 1, 0.7, [], ""))

        iax_feedback = store.get_for_body("IAX")
        assert len(iax_feedback) == 2
        assert all(r["target_body"] == "IAX" for r in iax_feedback)


class TestFeedbackLoop:
    def test_collect_feedback(self, tmp_store):
        loop = FeedbackLoop(tmp_store)
        body_outputs = {
            "IAX": {"heartbeat_ok": True, "tick_count": 10},
            "IAI": {"search_results": 5, "confidence": 0.9},
            "ISA": {"memory_count": 100, "entries": 50},
            "IOS": {"arbitrations": 3, "approvals": 2},
            "ISN": {"executions": 5, "error_rate": 0.1},
            "IKO": {"traces": 10, "ok_rate": 0.95},
        }
        records = loop.collect_feedback(body_outputs)
        # 6个体，每个有1个监督者，监督者也在输出中 => 6条反馈
        assert len(records) == 6
        assert all(isinstance(r, FeedbackRecord) for r in records)

    def test_apply_feedback(self, tmp_store):
        loop = FeedbackLoop(tmp_store)
        body_outputs = {
            "IAX": {"heartbeat_ok": True},
            "ISN": {"executions": 1, "error_rate": 0.1},
        }
        loop.collect_feedback(body_outputs)
        adjustments = loop.apply_feedback()
        assert "IAX" in adjustments
        assert "ISN" in adjustments

    def test_empty_output_detection(self, tmp_store):
        loop = FeedbackLoop(tmp_store)
        body_outputs = {
            "IAX": {},  # 空输出
            "ISN": {"executions": 1},
        }
        records = loop.collect_feedback(body_outputs)
        # IAX有空输出，ISN作为监督者评估IAX
        iax_record = [r for r in records if r.target_body == "IAX"]
        assert len(iax_record) == 1
        assert "输出为空" in iax_record[0].issues
        assert iax_record[0].score < 0.5

    def test_health_report(self, tmp_store):
        loop = FeedbackLoop(tmp_store)
        body_outputs = {
            "IAX": {"heartbeat_ok": True},
            "ISN": {"executions": 1, "error_rate": 0.1},
        }
        loop.collect_feedback(body_outputs)
        report = loop.get_health_report()
        assert report["round"] == 1
        assert report["chain_valid"] is True
        assert "IAX" in report["bodies"]
        assert "ISN" in report["bodies"]

    def test_degraded_status(self, tmp_store):
        loop = FeedbackLoop(tmp_store)
        # 多轮低分反馈
        for i in range(3):
            body_outputs = {
                "IAX": {"heartbeat_ok": False},  # 心跳异常，低分
                "ISN": {"executions": 1},
            }
            loop.collect_feedback(body_outputs)
        report = loop.get_health_report()
        # IAX连续心跳异常，应该是degraded或critical
        assert report["bodies"]["IAX"]["status"] in ("degraded", "critical")
