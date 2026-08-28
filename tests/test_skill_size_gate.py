#!/usr/bin/env python3
"""skill_size_gate 单元测试。"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

# 确保 openllm 包可导入
OPENLLM_SRC = os.path.expanduser("~/projects/openllm/src")
if OPENLLM_SRC not in sys.path:
    sys.path.insert(0, OPENLLM_SRC)

from openllm.isn.skill_size_gate import (
    Grade,
    GateReport,
    GateResult,
    count_skill_lines,
    gate_directory,
    gate_file,
    main,
    render_report,
)


# ── helpers ───────────────────────────────────────────────────────
def _write_skill(directory: str, name: str, line_count: int) -> str:
    """在 directory 下创建 SKILL.md，精确 line_count 行。"""
    skill_dir = os.path.join(directory, name)
    os.makedirs(skill_dir, exist_ok=True)
    path = os.path.join(skill_dir, "SKILL.md")
    # 每行以换行符结尾，最后一行也换行（模拟正常文件）
    content = "\n".join(f"line {i}" for i in range(line_count)) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ── 测试 ──────────────────────────────────────────────────────────
class TestThreeGrades:
    """三档分级：100行=PASS，400行=WARN，600行=FAIL。"""

    def test_three_grades(self, tmp_path):
        _write_skill(str(tmp_path), "short", 100)
        _write_skill(str(tmp_path), "medium", 400)
        _write_skill(str(tmp_path), "long", 600)

        report = gate_directory(tmp_path)

        assert report.total == 3
        assert report.pass_count == 1
        assert report.warn_count == 1
        assert report.fail_count == 1

        grades = {r.grade for r in report.results}
        assert Grade.PASS in grades
        assert Grade.WARN in grades
        assert Grade.FAIL in grades

    def test_boundary_350_is_pass(self, tmp_path):
        _write_skill(str(tmp_path), "exact", 350)
        report = gate_directory(tmp_path)
        assert report.pass_count == 1
        assert report.results[0].grade == Grade.PASS

    def test_boundary_351_is_warn(self, tmp_path):
        _write_skill(str(tmp_path), "over", 351)
        report = gate_directory(tmp_path)
        assert report.warn_count == 1
        assert report.results[0].grade == Grade.WARN

    def test_boundary_500_is_warn(self, tmp_path):
        _write_skill(str(tmp_path), "edge", 500)
        report = gate_directory(tmp_path)
        assert report.warn_count == 1

    def test_boundary_501_is_fail(self, tmp_path):
        _write_skill(str(tmp_path), "over", 501)
        report = gate_directory(tmp_path)
        assert report.fail_count == 1
        assert report.results[0].grade == Grade.FAIL


class TestEmptyDirectory:
    """空目录→total=0，无异常。"""

    def test_empty_directory(self, tmp_path):
        report = gate_directory(tmp_path)
        assert report.total == 0
        assert report.pass_count == 0
        assert report.warn_count == 0
        assert report.fail_count == 0
        assert report.results == []


class TestNestedDirectories:
    """嵌套目录能被扫到。"""

    def test_nested(self, tmp_path):
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text("hello\nworld\n")

        report = gate_directory(tmp_path)
        assert report.total == 1
        assert report.results[0].lines == 2


class TestPycacheSkip:
    """__pycache__ 下的 SKILL.md 被跳过。"""

    def test_pycache_skipped(self, tmp_path):
        # 正常位置放一个
        normal = tmp_path / "my_skill"
        normal.mkdir()
        (normal / "SKILL.md").write_text("ok\n")

        # __pycache__ 里放一个
        cache = tmp_path / "my_skill" / "__pycache__"
        cache.mkdir()
        (cache / "SKILL.md").write_text("should be skipped\n")

        report = gate_directory(tmp_path)
        assert report.total == 1
        assert "SKILL.md" not in report.results[0].path or "__pycache__" not in report.results[0].path


class TestDotDirectorySkip:
    """以 '.' 开头的目录下的 SKILL.md 被跳过。"""

    def test_dotdir_skipped(self, tmp_path):
        dot = tmp_path / ".hidden_skill"
        dot.mkdir()
        (dot / "SKILL.md").write_text("hidden\n")

        report = gate_directory(tmp_path)
        assert report.total == 0


class TestCLIExitCode:
    """CLI退出码：有FAIL→1，全合格→0。"""

    def _run_main(self, argv):
        """直接调用 main 并捕获 SystemExit。"""
        try:
            main(argv)
            return 0  # main 没有 raise → 退出码 0
        except SystemExit as e:
            return e.code

    def test_exit_code_1_with_fail(self, tmp_path):
        _write_skill(str(tmp_path), "giant", 600)
        code = self._run_main(["--root", str(tmp_path), "--format", "json"])
        assert code == 1

    def test_exit_code_0_all_pass(self, tmp_path):
        _write_skill(str(tmp_path), "ok", 100)
        code = self._run_main(["--root", str(tmp_path), "--format", "json"])
        assert code == 0

    def test_exit_code_0_empty(self, tmp_path):
        code = self._run_main(["--root", str(tmp_path), "--format", "text"])
        assert code == 0


class TestCountSkillLines:
    """wc -l口径=换行符计数，末尾无换行的末行不计入。"""

    def test_two_lines_no_trailing_newline(self, tmp_path):
        p = tmp_path / "skill.md"
        p.write_text("aaa\nbbb")
        assert count_skill_lines(p) == 1

    def test_two_lines_with_trailing_newline(self, tmp_path):
        p = tmp_path / "skill.md"
        p.write_text("aaa\nbbb\n")
        assert count_skill_lines(p) == 2

    def test_single_line_no_newline(self, tmp_path):
        p = tmp_path / "skill.md"
        p.write_text("only one")
        assert count_skill_lines(p) == 0

    def test_single_line_with_newline(self, tmp_path):
        p = tmp_path / "skill.md"
        p.write_text("only one\n")
        assert count_skill_lines(p) == 1

    def test_empty_file(self, tmp_path):
        p = tmp_path / "skill.md"
        p.write_text("")
        assert count_skill_lines(p) == 0


class TestGateFile:
    """gate_file 返回正确的 GateResult。"""

    def test_gate_file(self, tmp_path):
        p = tmp_path / "skill.md"
        p.write_text("x\n" * 100)
        result = gate_file(p)
        assert result.lines == 100
        assert result.grade == Grade.PASS
        assert isinstance(result, GateResult)


class TestRenderReport:
    """render_report 输出可读。"""

    def test_render_with_all_grades(self, tmp_path):
        _write_skill(str(tmp_path), "a", 100)
        _write_skill(str(tmp_path), "b", 400)
        _write_skill(str(tmp_path), "c", 600)

        report = gate_directory(tmp_path)
        text = render_report(report)

        assert "PASS" in text
        assert "WARN" in text
        assert "FAIL" in text
        assert "600" in text or "c" in text

    def test_render_empty(self, tmp_path):
        report = gate_directory(tmp_path)
        text = render_report(report)
        assert "全部合格" in text


class TestToDict:
    """GateReport.to_dict() 返回正确结构。"""

    def test_to_dict(self, tmp_path):
        _write_skill(str(tmp_path), "a", 100)
        report = gate_directory(tmp_path)
        d = report.to_dict()

        assert isinstance(d, dict)
        assert d["total"] == 1
        assert d["pass_count"] == 1
        assert d["warn_count"] == 0
        assert d["fail_count"] == 0
        assert len(d["results"]) == 1
        assert d["results"][0]["grade"] == "PASS"
        assert d["results"][0]["lines"] == 100
