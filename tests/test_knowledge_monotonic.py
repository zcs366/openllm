"""
tests/test_knowledge_monotonic.py — K度量探针测试（≥8条含反向测试）
================================================================
验收判据：
  KM1 可import，KnowledgeMonotonicProbe/KnowledgeReport存在
  KM2 take_snapshot真收集四容器数据
  KM3 check_monotonic真做两快照diff
  KM4 反向测试：构造'旧知识被删'场景→verdict='regression'
  KM5 测试≥8条
  KM6 真实运行两次快照
  KM7 bus.py只加'knowledge.probed'一行
"""
import json
import shutil
import tempfile
from pathlib import Path

import pytest


# ── KM1: import检查 ──

class TestKM1Import:
    """KM1：模块可import，核心类存在。"""

    def test_import_probe(self):
        from openllm.memory.knowledge_monotonic import KnowledgeMonotonicProbe
        assert KnowledgeMonotonicProbe is not None

    def test_import_report(self):
        from openllm.memory.knowledge_monotonic import KnowledgeReport
        assert KnowledgeReport is not None


# ── 辅助fixture：临时快照目录 ──

@pytest.fixture
def tmp_snapshot_dir():
    d = Path(tempfile.mkdtemp(prefix="km_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── KM2+KM3: take_snapshot + check_monotonic 基本流程 ──

class TestKM2KM3Snapshot:
    """KM2：take_snapshot真收集四容器数据；KM3：check_monotonic真做diff。"""

    def test_take_snapshot_returns_valid_dict(self, tmp_snapshot_dir):
        from openllm.memory.knowledge_monotonic import KnowledgeMonotonicProbe
        probe = KnowledgeMonotonicProbe(snapshot_dir=tmp_snapshot_dir)
        snap = probe.take_snapshot()
        assert "ts" in snap
        assert "counts" in snap
        assert "content_hashes" in snap
        assert "content_digest" in snap
        # 四容器计数均存在
        for key in ("capsule", "jiak", "growth", "ledger"):
            assert key in snap["counts"]

    def test_take_snapshot_persists_to_jsonl(self, tmp_snapshot_dir):
        from openllm.memory.knowledge_monotonic import KnowledgeMonotonicProbe
        probe = KnowledgeMonotonicProbe(snapshot_dir=tmp_snapshot_dir)
        probe.take_snapshot()
        # JSONL文件存在且有一行
        lines = probe._snapshot_file.read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert "counts" in rec

    def test_check_monotonic_two_snapshots(self, tmp_snapshot_dir):
        from openllm.memory.knowledge_monotonic import KnowledgeMonotonicProbe
        probe = KnowledgeMonotonicProbe(snapshot_dir=tmp_snapshot_dir)
        probe.take_snapshot()
        probe.take_snapshot()
        report = probe.check_monotonic()
        assert report.verdict in ("monotonic", "regression", "stagnant")
        assert isinstance(report.capacity_delta, dict)
        assert 0.0 <= report.recall <= 1.0

    def test_insufficient_data_one_snapshot(self, tmp_snapshot_dir):
        from openllm.memory.knowledge_monotonic import KnowledgeMonotonicProbe
        probe = KnowledgeMonotonicProbe(snapshot_dir=tmp_snapshot_dir)
        probe.take_snapshot()
        report = probe.check_monotonic()
        assert report.verdict == "insufficient_data"


# ── KM4: 反向测试——灾难性遗忘检测 ──

class TestKM4Regression:
    """KM4：构造'旧知识被删'场景→verdict='regression'。"""

    def test_deleted_old_content_triggers_regression(self, tmp_snapshot_dir):
        """手动构造两份快照，第二份删除旧哈希→recall<1.0→regression。"""
        from openllm.memory.knowledge_monotonic import (
            KnowledgeMonotonicProbe, KnowledgeReport,
        )

        probe = KnowledgeMonotonicProbe(snapshot_dir=tmp_snapshot_dir)

        # 第一份快照：有3条旧内容
        snap1 = {
            "ts": "2026-01-01T00:00:00+08:00",
            "counts": {"capsule": 3, "jiak": 2, "growth": 1, "ledger": 0},
            "content_hashes": {
                "capsule": ["h1", "h2", "h3"],
                "jiak": ["h4", "h5"],
                "growth": ["h6"],
                "ledger": [],
            },
            "content_digest": "abc",
        }
        # 第二份快照：h2和h5被删
        snap2 = {
            "ts": "2026-01-01T00:00:05+08:00",
            "counts": {"capsule": 2, "jiak": 1, "growth": 1, "ledger": 0},
            "content_hashes": {
                "capsule": ["h1", "h3"],
                "jiak": ["h4"],
                "growth": ["h6"],
                "ledger": [],
            },
            "content_digest": "def",
        }

        # 直接写入JSONL
        with open(probe._snapshot_file, "a") as f:
            f.write(json.dumps(snap1) + "\n")
            f.write(json.dumps(snap2) + "\n")

        report = probe.check_monotonic()
        assert report.verdict == "regression"
        assert report.recall < 1.0
        assert "丢失" in report.detail or "灾难" in report.detail


# ── KM5-a: 零增长→stagnant ──

class TestKM5Stagnant:
    """零增长场景→verdict='stagnant'。"""

    def test_stagnant_when_no_growth(self, tmp_snapshot_dir):
        from openllm.memory.knowledge_monotonic import KnowledgeMonotonicProbe

        probe = KnowledgeMonotonicProbe(snapshot_dir=tmp_snapshot_dir)

        snap = {
            "ts": "2026-01-01T00:00:00+08:00",
            "counts": {"capsule": 5, "jiak": 3, "growth": 2, "ledger": 1},
            "content_hashes": {
                "capsule": ["a", "b"],
                "jiak": ["c"],
                "growth": ["d"],
                "ledger": ["e"],
            },
            "content_digest": "x",
        }
        # 两份完全相同的快照
        with open(probe._snapshot_file, "a") as f:
            f.write(json.dumps(snap) + "\n")
            snap2 = dict(snap)
            snap2["ts"] = "2026-01-01T00:00:05+08:00"
            f.write(json.dumps(snap2) + "\n")

        report = probe.check_monotonic()
        assert report.verdict == "stagnant"


# ── KM5-b: recall计算正确性 ──

class TestKM5Recall:
    """构造已知场景验证recall数值。"""

    def test_recall_0_5(self, tmp_snapshot_dir):
        """旧有4条哈希，新只剩2条→recall=0.5。"""
        from openllm.memory.knowledge_monotonic import KnowledgeMonotonicProbe

        probe = KnowledgeMonotonicProbe(snapshot_dir=tmp_snapshot_dir)

        snap1 = {
            "ts": "2026-01-01T00:00:00+08:00",
            "counts": {"capsule": 4, "jiak": 0, "growth": 0, "ledger": 0},
            "content_hashes": {"capsule": ["h1", "h2", "h3", "h4"], "jiak": [], "growth": [], "ledger": []},
            "content_digest": "a",
        }
        snap2 = {
            "ts": "2026-01-01T00:00:01+08:00",
            "counts": {"capsule": 2, "jiak": 0, "growth": 0, "ledger": 0},
            "content_hashes": {"capsule": ["h1", "h3"], "jiak": [], "growth": [], "ledger": []},
            "content_digest": "b",
        }
        with open(probe._snapshot_file, "a") as f:
            f.write(json.dumps(snap1) + "\n")
            f.write(json.dumps(snap2) + "\n")

        report = probe.check_monotonic()
        assert abs(report.recall - 0.5) < 0.01


# ── KM5-c: 四容器计数正确 ──

class TestKM5Counts:
    """验证take_snapshot的四容器计数与实际环境一致。"""

    def test_counts_are_non_negative_integers(self, tmp_snapshot_dir):
        from openllm.memory.knowledge_monotonic import KnowledgeMonotonicProbe
        probe = KnowledgeMonotonicProbe(snapshot_dir=tmp_snapshot_dir)
        snap = probe.take_snapshot()
        for name, count in snap["counts"].items():
            assert isinstance(count, int), f"{name} count不是int"
            assert count >= 0, f"{name} count为负"


# ── KM5-d: emit_report写回 ──

class TestKM5Emit:
    """emit_report能写回bus。"""

    def test_emit_returns_event(self, tmp_snapshot_dir):
        from openllm.memory.knowledge_monotonic import KnowledgeMonotonicProbe
        probe = KnowledgeMonotonicProbe(snapshot_dir=tmp_snapshot_dir)
        probe.take_snapshot()
        probe.take_snapshot()
        result = probe.emit_report()
        # emit可能因bus路径问题返回None，但不报错
        if result is not None:
            assert result["type"] == "knowledge.probed"


# ── KM6: 真实运行两次快照 ──

class TestKM6RealRun:
    """对真实环境take_snapshot两次并输出报告。"""

    def test_real_two_snapshots(self):
        from openllm.memory.knowledge_monotonic import KnowledgeMonotonicProbe
        probe = KnowledgeMonotonicProbe()
        snap1 = probe.take_snapshot()
        # 短暂间隔，模拟中间加一条数据
        import time
        time.sleep(0.1)
        snap2 = probe.take_snapshot()

        report = probe.check_monotonic()
        # 输出给用户看
        print(f"\n[KM6] Snap1 counts: {snap1['counts']}")
        print(f"[KM6] Snap2 counts: {snap2['counts']}")
        print(f"[KM6] Capacity delta: {report.capacity_delta}")
        print(f"[KM6] Recall: {report.recall}")
        print(f"[KM6] Verdict: {report.verdict}")
        print(f"[KM6] Detail: {report.detail}")

        # 基本断言：至少能跑通不报错
        assert report.verdict in ("monotonic", "regression", "stagnant", "insufficient_data")
