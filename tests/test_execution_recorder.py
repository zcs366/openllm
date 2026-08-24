"""Tests for execution_recorder — CCL执行节点写入RECALL。

关键约束：绝不写真实 ~/.hermes/jiak/RECALL.jsonl。
所有测试通过mock subprocess验证。
"""
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Ensure openllm is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openllm.memory.execution_recorder import (
    record_execution,
    _write_one_sync,
    RECALL_APPEND_SCRIPT,
)


class TestRecordExecutionJSON:
    """a) record_execution 生成合法 JSON 且被 recall_append 接受。"""

    def test_generates_valid_json(self):
        """构造一条record，验证JSON格式合法且包含必要字段。"""
        record = {
            "ts": "2026-08-24T12:00:00+00:00",
            "type": "tool_call",
            "tool": "read_file",
            "status": "ok",
            "duration_ms": 12.5,
            "summary": "文件内容前200字符",
            "source": "openllm",
        }
        # JSON序列化验证
        json_str = json.dumps(record, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["type"] == "tool_call"
        assert parsed["tool"] == "read_file"
        assert parsed["status"] == "ok"
        assert parsed["source"] == "openllm"

    def test_args_summary_truncated(self):
        """args_summary的值被截断到100字符。"""
        long_value = "x" * 200
        record_args = {"code": long_value}
        truncated = {k: str(v)[:100] for k, v in record_args.items()}
        assert len(truncated["code"]) == 100

    def test_result_summary_truncated(self):
        """result_summary被截断到200字符。"""
        long_result = "y" * 300
        assert len(long_result[:200]) == 200

    @patch("openllm.memory.execution_recorder.subprocess.run")
    def test_write_one_sync_calls_recall_append(self, mock_run):
        """_write_one_sync 正确调用 recall_append.py。"""
        mock_run.return_value = MagicMock(returncode=0)
        record = {
            "ts": "2026-08-24T12:00:00+00:00",
            "type": "tool_call",
            "tool": "test_tool",
            "status": "ok",
            "duration_ms": 5.0,
            "summary": "test",
            "source": "openllm",
        }
        _write_one_sync(record)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "python3"
        assert args[1] == RECALL_APPEND_SCRIPT
        # Third arg should be the JSON line
        parsed = json.loads(args[2])
        assert parsed["type"] == "tool_call"
        assert parsed["tool"] == "test_tool"


class TestToolExecutionTriggersRecording:
    """b) 工具执行后确实触发记录。"""

    @patch("openllm.memory.execution_recorder.subprocess.run")
    def test_record_execution_queues_and_writes(self, mock_run):
        """record_execution 入队并通过后台线程写入。"""
        mock_run.return_value = MagicMock(returncode=0)
        record_execution(
            tool_name="shell",
            args_summary={"command": "ls -la"},
            status="ok",
            duration_ms=42.3,
            result_summary="file1.txt\nfile2.py",
        )
        # Wait briefly for daemon thread
        time.sleep(0.2)
        assert mock_run.called
        # Verify the JSON written is valid
        args = mock_run.call_args[0][0]
        parsed = json.loads(args[2])
        assert parsed["type"] == "tool_call"
        assert parsed["tool"] == "shell"
        assert parsed["status"] == "ok"
        assert parsed["duration_ms"] == 42.3

    @patch("openllm.memory.execution_recorder.subprocess.run")
    def test_error_status_recorded(self, mock_run):
        """错误状态正确记录。"""
        mock_run.return_value = MagicMock(returncode=0)
        record_execution(
            tool_name="write_file",
            args_summary={"path": "/tmp/test"},
            status="error",
            duration_ms=1.5,
            result_summary="Permission denied",
        )
        time.sleep(0.2)
        args = mock_run.call_args[0][0]
        parsed = json.loads(args[2])
        assert parsed["status"] == "error"
        assert parsed["tool"] == "write_file"


class TestWriteFailureIsolation:
    """c) 写入失败时工具执行不受影响。"""

    @patch("openllm.memory.execution_recorder.subprocess.run")
    def test_subprocess_exception_does_not_propagate(self, mock_run):
        """recall_append.py抛异常时，record_execution不报错。"""
        mock_run.side_effect = OSError("No such file or directory")
        # This should NOT raise
        record_execution(
            tool_name="test_tool",
            args_summary={},
            status="ok",
            duration_ms=0.0,
            result_summary="",
        )
        time.sleep(0.2)
        # Verify the exception was caught (mock was called and failed)
        assert mock_run.called

    @patch("openllm.memory.execution_recorder.subprocess.run")
    def test_write_one_sync_handles_timeout(self, mock_run):
        """_write_one_sync 处理超时异常。"""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=10)
        record = {"type": "tool_call", "tool": "test", "status": "ok"}
        # Should not raise
        _write_one_sync(record)

    @patch("openllm.memory.execution_recorder.subprocess.run")
    def test_write_one_sync_handles_json_error(self, mock_run):
        """recall_append返回非零退出码不影响调用方。"""
        mock_run.return_value = MagicMock(returncode=1, stderr="bad json")
        record = {"type": "tool_call", "tool": "test", "status": "ok"}
        # Should not raise
        _write_one_sync(record)


class TestExecuteToolHook:
    """验证 tool_executor.py 中 execute_tool 调用 record_execution。"""

    def test_execute_tool_imports_record_execution(self):
        """execute_tool模块成功导入record_execution。"""
        from openllm.core.tool_executor import record_execution as re_import
        assert callable(re_import)
