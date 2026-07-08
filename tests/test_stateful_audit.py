"""
StatefulAuditTrail 测试
========================

5个核心测试用例：
1. 单次记录 + cumulative_score正确
2. 跨session累积
3. 阈值触发告警
4. 分布统计
5. 归档功能
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from openllm.governance.stateful_audit import StatefulAuditTrail, AuditRecord, AlertRecord


def test_single_record():
    """T-SFL-TEST-1: 单次记录 + cumulative_score正确。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        trail = StatefulAuditTrail(audit_dir=Path(tmpdir), retention_days=90)

        rec = trail.record(
            session_id="test-session-1",
            event_type="tool_call",
            suspicion_score=0.3,
            content_hash="abc123",
            details={"tool": "web_search"},
        )

        assert rec.session_id == "test-session-1"
        assert rec.event_type == "tool_call"
        assert rec.suspicion_score == 0.3
        assert rec.cumulative_score == 0.3  # 首次 = 自身
        assert rec.content_hash == "abc123"

        # 验证JSONL已写入
        path = trail._records_path()
        assert path.exists()
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["session_id"] == "test-session-1"
        assert data["cumulative_score"] == 0.3

        print("✅ T-SFL-TEST-1: 单次记录 + cumulative_score正确")


def test_cross_session_cumulative():
    """T-SFL-TEST-2: 跨session累积。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        trail = StatefulAuditTrail(audit_dir=Path(tmpdir), retention_days=90)

        # Session A: 3次操作
        trail.record("session-A", "tool_call", 0.2)
        trail.record("session-A", "tool_call", 0.3)
        trail.record("session-A", "file_write", 0.5)

        # Session B: 1次操作
        trail.record("session-B", "tool_call", 0.1)

        cum = trail.get_cumulative_by_session()
        assert len(cum) == 2
        assert abs(cum["session-A"] - 1.0) < 0.001  # 0.2+0.3+0.5=1.0
        assert abs(cum["session-B"] - 0.1) < 0.001

        print("✅ T-SFL-TEST-2: 跨session累积")


def test_threshold_alert():
    """T-SFL-TEST-3: 阈值触发告警。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        trail = StatefulAuditTrail(audit_dir=Path(tmpdir), retention_days=90)

        # 累积到接近阈值
        trail.record("session-C", "tool_call", 0.8)
        trail.record("session-C", "tool_call", 0.8)
        trail.record("session-C", "tool_call", 0.8)  # cumulative=2.4

        # 未超阈值(3.0)
        alert = trail.check_threshold("session-C", threshold=3.0)
        assert alert is None

        # 继续累积，超阈值
        trail.record("session-C", "tool_call", 0.8)  # cumulative=3.2
        alert = trail.check_threshold("session-C", threshold=3.0)
        assert alert is not None
        assert alert.trigger_score > 3.0
        assert alert.threshold == 3.0
        assert alert.alert_type == "cumulative_threshold"

        # 验证告警已写入JSONL
        alerts_path = trail._alerts_path()
        assert alerts_path.exists()
        with open(alerts_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) >= 1

        print("✅ T-SFL-TEST-3: 阈值触发告警")


def test_distribution_stats():
    """T-SFL-TEST-4: 分布统计。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        trail = StatefulAuditTrail(audit_dir=Path(tmpdir), retention_days=90)

        # 生成20条记录，分数从0.0到1.0均匀分布
        for i in range(20):
            trail.record("session-D", "tool_call", i / 19.0)

        stats = trail.get_distribution()
        assert stats["total_records"] == 20
        assert stats["total_sessions"] == 1
        assert 0.0 <= stats["mean_score"] <= 1.0
        assert stats["p50"] >= 0.0
        assert stats["p95"] >= stats["p50"]
        assert stats["max_score"] >= stats["p95"]

        print(f"✅ T-SFL-TEST-4: 分布统计 (mean={stats['mean_score']:.3f}, "
              f"p50={stats['p50']:.3f}, p95={stats['p95']:.3f})")


def test_archive():
    """T-SFL-TEST-5: 归档功能。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        trail = StatefulAuditTrail(audit_dir=Path(tmpdir), retention_days=0)  # retention=0天=全部归档

        # 写入一些记录
        for i in range(5):
            trail.record("session-E", "tool_call", 0.1 * i)

        # 验证有记录
        assert trail._records_path().exists()

        # 执行归档（retention=0天=所有记录都过期）
        archived = trail.archive_old_records()
        assert archived == 5

        # 验证主文件已清空
        with open(trail._records_path()) as f:
            remaining = [l.strip() for l in f if l.strip()]
        assert len(remaining) == 0

        # 验证归档目录有文件
        archive_files = list(trail.archive_dir.glob("*.jsonl"))
        assert len(archive_files) >= 1

        print("✅ T-SFL-TEST-5: 归档功能")


def test_health_check():
    """T-SFL-TEST-6: 健康检查。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        trail = StatefulAuditTrail(audit_dir=Path(tmpdir), retention_days=90)

        # 样本不足
        result = trail.health_check()
        assert result["status"] == "warning"
        assert "样本不足" in result["message"]

        # 正常样本
        for i in range(15):
            trail.record("session-F", "tool_call", 0.1)
        result = trail.health_check()
        assert result["status"] == "healthy"

        print("✅ T-SFL-TEST-6: 健康检查")


if __name__ == "__main__":
    test_single_record()
    test_cross_session_cumulative()
    test_threshold_alert()
    test_distribution_stats()
    test_archive()
    test_health_check()
    print("\n🎉 全部6个测试通过！")
