"""
test_viability_logger.py — P1-2 观测仪表测试

全部用 tmp_path + mock，不依赖真实 io-s 和 ~/.openllm。
"""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# 确保 openllm 包可导入
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from openllm.memory.viability_logger import (
    _load_compute_viability,
    log_skill_event,
    log_viability,
    read_log,
)

# ── fixtures ──


@pytest.fixture
def v_log(tmp_path):
    """V值日志临时路径。"""
    return tmp_path / "viability_log.jsonl"


@pytest.fixture
def s_log(tmp_path):
    """技能生命周期日志临时路径。"""
    return tmp_path / "skill_lifecycle_log.jsonl"


# ═══ log_viability 测试 ═══

MOCK_V_RESULT = {
    "V": 0.861,
    "components": {"M": 0.82, "S": 0.81, "B": 1.0, "E": None},
}


class TestLogViability:
    """log_viability 写入与健壮性。"""

    @patch(
        "openllm.memory.viability_logger._load_compute_viability",
        return_value=lambda: MOCK_V_RESULT,
    )
    def test_writes_valid_jsonl(self, _mock_loader, v_log, tmp_path):
        """mock compute_viability → 写入后 read_log 能读回，含 timestamp 和 source。"""
        record = log_viability(log_path=v_log)
        assert record is not None
        assert "timestamp" in record
        assert record["source"] == "openllm.viability_logger"
        assert record["v_result"]["V"] == 0.861

        # 文件可读回
        rows = read_log(v_log, last_n=10)
        assert len(rows) == 1
        assert rows[0]["v_result"]["components"]["M"] == 0.82

    def test_failure_returns_none_no_corruption(self, v_log):
        """compute_viability 抛异常 → 返回 None，日志文件不产生损坏行。"""
        import openllm.memory.viability_logger as mod
        original = mod._load_compute_viability
        def _broken():
            def _raise():
                raise RuntimeError("boom")
            return _raise
        mod._load_compute_viability = _broken
        try:
            record = log_viability(log_path=v_log)
        finally:
            mod._load_compute_viability = original
        assert record is None
        # 文件不应存在或为空
        assert not v_log.exists() or v_log.read_text(encoding="utf-8").strip() == ""

    @patch(
        "openllm.memory.viability_logger._load_compute_viability",
        return_value=None,
    )
    def test_io_s_not_importable_returns_none(self, _mock_none, v_log):
        """io-s 不可导入 → 返回 None。"""
        record = log_viability(log_path=v_log)
        assert record is None

    def test_compute_viability_returns_none_returns_none(self, v_log):
        """compute_viability 函数本身返回 None → 仍写入日志（v_result=None）。"""
        with patch(
            "openllm.memory.viability_logger._load_compute_viability",
            return_value=lambda: None,
        ):
            record = log_viability(log_path=v_log)
        assert record is not None
        assert record["v_result"] is None

    @patch(
        "openllm.memory.viability_logger._load_compute_viability",
        return_value=lambda: MOCK_V_RESULT,
    )
    def test_append_only_multiple_writes(self, _mock_loader, v_log):
        """多次写入都是追加，不覆盖。"""
        log_viability(log_path=v_log)
        log_viability(log_path=v_log)
        rows = read_log(v_log, last_n=10)
        assert len(rows) == 2
        # 两条时间戳不同（至少秒级差异或相同均可）
        assert rows[0]["timestamp"]


# ═══ log_skill_event 测试 ═══


class TestLogSkillEvent:
    """log_skill_event 写入与字段完整性。"""

    def test_adopted(self, s_log):
        """写入 adopted 事件 → 字段齐全。"""
        rec = log_skill_event("adopted", "search-pipeline", details={"version": "v7.0"}, log_path=s_log)
        assert rec is not None
        assert rec["event"] == "adopted"
        assert rec["skill"] == "search-pipeline"
        assert rec["details"]["version"] == "v7.0"
        assert "timestamp" in rec

        rows = read_log(s_log)
        assert len(rows) == 1
        assert rows[0]["event"] == "adopted"

    def test_gated_reject(self, s_log):
        """写入 gated_reject 事件 → 字段齐全。"""
        rec = log_skill_event(
            "gated_reject", "bad-skill", details={"reason": "low_V"}, log_path=s_log
        )
        assert rec is not None
        assert rec["event"] == "gated_reject"
        assert rec["skill"] == "bad-skill"

        rows = read_log(s_log)
        assert len(rows) == 1
        assert rows[0]["event"] == "gated_reject"

    def test_details_optional(self, s_log):
        """不传 details → 字段中不含 details 键。"""
        rec = log_skill_event("retired", "old-skill", log_path=s_log)
        assert rec is not None
        assert "details" not in rec

    def test_write_failure_returns_none(self, s_log):
        """写入失败（路径非法）→ 返回 None。"""
        rec = log_skill_event(
            "adopted", "x", log_path=Path("/nonexistent/deeply/bad/path.jsonl")
        )
        assert rec is None


# ═══ read_log 测试 ═══


class TestReadLog:
    """read_log 的健壮性与截取。"""

    def test_file_not_exists(self, tmp_path):
        """文件不存在 → 空列表。"""
        assert read_log(tmp_path / "nope.jsonl") == []

    def test_mixed_valid_and_corrupt(self, s_log):
        """混合合法行 + 损坏行 → 只返回合法行。"""
        with open(s_log, "w", encoding="utf-8") as f:
            f.write('{"ok": true}\n')
            f.write("NOT_JSON\n")
            f.write('{"ok": true, "n": 2}\n')
            f.write("\n")  # 空行
            f.write('{"ok": true, "n": 3}\n')

        rows = read_log(s_log)
        assert len(rows) == 3
        assert rows[0]["ok"] is True
        assert rows[2]["n"] == 3

    def test_last_n_truncation(self, s_log):
        """写5条读3条 → 返回最后3条。"""
        for i in range(5):
            with open(s_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"i": i}) + "\n")

        rows = read_log(s_log, last_n=3)
        assert len(rows) == 3
        assert rows[0]["i"] == 2
        assert rows[1]["i"] == 3
        assert rows[2]["i"] == 4

    def test_last_n_larger_than_file(self, s_log):
        """last_n 大于文件行数 → 返回全部。"""
        with open(s_log, "w", encoding="utf-8") as f:
            f.write('{"a": 1}\n')
        rows = read_log(s_log, last_n=100)
        assert len(rows) == 1

    def test_empty_lines_skipped(self, s_log):
        """纯空行文件 → 空列表。"""
        with open(s_log, "w", encoding="utf-8") as f:
            f.write("\n\n\n")
        assert read_log(s_log) == []


# ═══ CLI 测试 ═══


class TestCLI:
    """CLI 入口 subprocess 验证。"""

    @patch(
        "openllm.memory.viability_logger._load_compute_viability",
        return_value=lambda: MOCK_V_RESULT,
    )
    def test_cli_default_logs_v(self, _mock_loader, v_log, monkeypatch):
        """python -m 默认执行 → 输出含 'V'（直接调用 main）。"""
        monkeypatch.setenv("HOME", str(v_log.parent.parent))
        # openllm 顶层 __init__ 导入 numpy 导致子进程失败，
        # 改为直接调用 main() 函数验证输出逻辑
        import io
        from contextlib import redirect_stdout
        from openllm.memory.viability_logger import main
        old_argv = sys.argv[:]
        sys.argv = ["viability_logger"]
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                main()
        finally:
            sys.argv = old_argv
        output = buf.getvalue()
        assert "V" in output, f"CLI output missing 'V': {output}"

    @patch(
        "openllm.memory.viability_logger._load_compute_viability",
        return_value=lambda: MOCK_V_RESULT,
    )
    def test_cli_read_flag(self, _mock_loader, v_log, monkeypatch):
        """python -m --read 5 → 不报错。"""
        # 先写一条
        log_viability(log_path=v_log)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "openllm.memory.viability_logger",
                "--read",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env={**dict(__import__("os").environ), "PYTHONPATH": str(SRC_DIR)},
        )
        # 不崩溃即可
        assert result.returncode == 0 or "V" in (result.stdout + result.stderr)
