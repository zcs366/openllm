"""test_audit_logger.py — 验证日志可审计测试。

测试项：
1. 日志记录：log_verification 写入正确格式
2. 日志读取：get_logs 返回最近N条
3. 导出：export 写出JSON文件
4. 边界：空文件、损坏行、limit=0
"""

import json
import os
import tempfile

import pytest

from openllm.core.audit_logger import AuditLogger


# ═══════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════

@pytest.fixture
def tmp_log(tmp_path):
    """使用临时路径的审计日志。"""
    return AuditLogger(log_path=str(tmp_path / "audit_log.jsonl"))


@pytest.fixture
def sample_claim():
    """样本验证数据。"""
    return {
        "claim": "地球是圆的",
        "method": "rule",
        "result": {"passed": True, "confidence": 0.99, "score": 0.95},
    }


# ═══════════════════════════════════════════════════════
# 测试：日志记录
# ═══════════════════════════════════════════════════════

class TestLogVerification:
    """测试 log_verification 方法。"""

    def test_log_writes_file(self, tmp_log, sample_claim):
        """记录后文件应存在。"""
        tmp_log.log_verification(**sample_claim)
        assert os.path.exists(tmp_log.log_path)

    def test_log_format(self, tmp_log, sample_claim):
        """日志格式应为有效JSONL。"""
        tmp_log.log_verification(**sample_claim)
        with open(tmp_log.log_path, "r") as f:
            line = f.readline().strip()
        record = json.loads(line)
        assert record["claim"] == "地球是圆的"
        assert record["method"] == "rule"
        assert record["result"]["passed"] is True
        assert "timestamp" in record
        assert record["logger_id"] == "openllm.audit_logger"

    def test_log_custom_timestamp(self, tmp_log):
        """自定义timestamp应被记录。"""
        ts = "2026-07-27T00:00:00Z"
        tmp_log.log_verification(
            claim="测试",
            method="test",
            result={"passed": True},
            timestamp=ts,
        )
        records = tmp_log.get_logs()
        assert records[0]["timestamp"] == ts

    def test_log_multiple(self, tmp_log):
        """多次记录应追加。"""
        for i in range(5):
            tmp_log.log_verification(
                claim=f"声明{i}",
                method="rule",
                result={"passed": True, "idx": i},
            )
        records = tmp_log.get_logs()
        assert len(records) == 5
        assert records[0]["claim"] == "声明0"
        assert records[4]["claim"] == "声明4"


# ═══════════════════════════════════════════════════════
# 测试：日志读取
# ═══════════════════════════════════════════════════════

class TestGetLogs:
    """测试 get_logs 方法。"""

    def test_empty_file(self, tmp_log):
        """文件不存在时返回空列表。"""
        assert tmp_log.get_logs() == []

    def test_limit(self, tmp_log):
        """limit应限制返回条数。"""
        for i in range(10):
            tmp_log.log_verification(
                claim=f"声明{i}",
                method="rule",
                result={"passed": True},
            )
        assert len(tmp_log.get_logs(limit=3)) == 3
        # 返回的是最后3条
        logs = tmp_log.get_logs(limit=3)
        assert logs[0]["claim"] == "声明7"
        assert logs[2]["claim"] == "声明9"

    def test_limit_exceeds_total(self, tmp_log):
        """limit大于总条数时返回全部。"""
        for i in range(3):
            tmp_log.log_verification(
                claim=f"声明{i}",
                method="rule",
                result={"passed": True},
            )
        assert len(tmp_log.get_logs(limit=100)) == 3

    def test_corrupted_line_skipped(self, tmp_log):
        """损坏行应被跳过。"""
        tmp_log.log_verification(
            claim="正常",
            method="rule",
            result={"passed": True},
        )
        # 写入损坏行
        with open(tmp_log.log_path, "a") as f:
            f.write("NOT_VALID_JSON\n")
        tmp_log.log_verification(
            claim="又一个正常",
            method="rule",
            result={"passed": True},
        )
        logs = tmp_log.get_logs()
        assert len(logs) == 2
        assert logs[0]["claim"] == "正常"
        assert logs[1]["claim"] == "又一个正常"


# ═══════════════════════════════════════════════════════
# 测试：导出
# ═══════════════════════════════════════════════════════

class TestExport:
    """测试 export 方法。"""

    def test_export_json(self, tmp_log, tmp_path):
        """导出应生成有效JSON文件。"""
        tmp_log.log_verification(
            claim="导出测试",
            method="rule",
            result={"passed": True},
        )
        export_path = str(tmp_path / "exported.json")
        tmp_log.export(export_path)
        assert os.path.exists(export_path)
        with open(export_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["claim"] == "导出测试"

    def test_export_empty(self, tmp_log, tmp_path):
        """空日志导出应生成空JSON数组。"""
        export_path = str(tmp_path / "export_empty.json")
        tmp_log.export(export_path)
        with open(export_path) as f:
            data = json.load(f)
        assert data == []

    def test_export_preserves_all_fields(self, tmp_log, tmp_path):
        """导出应保留所有字段。"""
        ts = "2026-07-27T12:00:00Z"
        tmp_log.log_verification(
            claim="字段测试",
            method="llm",
            result={"passed": False, "score": 0.3, "evidence": "无效"},
            timestamp=ts,
        )
        export_path = str(tmp_path / "fields.json")
        tmp_log.export(export_path)
        with open(export_path) as f:
            data = json.load(f)
        record = data[0]
        assert record["claim"] == "字段测试"
        assert record["method"] == "llm"
        assert record["result"]["passed"] is False
        assert record["result"]["score"] == 0.3
        assert record["timestamp"] == ts
        assert record["logger_id"] == "openllm.audit_logger"
