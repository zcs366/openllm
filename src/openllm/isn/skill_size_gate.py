#!/usr/bin/env python3
"""Skill出厂行数门 — 纯文本行数门控，零LLM成本。

两档阈值来源：2026-08-24 巨象锻造战役收官标准。
  - ROUTER_BRAIN_MAX=350（路由大脑标准线，≤350=PASS）
  - GIANT_MAX=500（巨象线，351-500=WARN，>500=FAIL）

用法：
    python -m openllm.isn.skill_size_gate --root ~/.hermes/skills --format json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import List, Optional

# ── 常量 ──────────────────────────────────────────────────────────
ROUTER_BRAIN_MAX: int = 350   # 路由大脑标准线
GIANT_MAX: int = 500          # 巨象线


# ── 枚举 ──────────────────────────────────────────────────────────
class Grade(Enum):
    PASS = "PASS"   # ≤ 350
    WARN = "WARN"   # 351-500
    FAIL = "FAIL"   # > 500


# ── 数据类 ────────────────────────────────────────────────────────
@dataclass
class GateResult:
    path: str
    lines: int
    grade: Grade


@dataclass
class GateReport:
    total: int = 0
    pass_count: int = 0
    warn_count: int = 0
    fail_count: int = 0
    results: List[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "results": [
                {"path": r.path, "lines": r.lines, "grade": r.grade.value}
                for r in self.results
            ],
        }


# ── 行数计算 ──────────────────────────────────────────────────────
def count_skill_lines(path: str | Path) -> int:
    """与 wc -l 完全一致：统计换行符个数。

    锻造台账87锻全部以 wc -l 记账，守门人必须说同一种语言。
    注意：末尾无换行的最后一行不计入（'aaa\\nbbb' = 1，与 wc -l 相同）。
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().count("\n")


# ── 门控 ──────────────────────────────────────────────────────────
def _classify(lines: int) -> Grade:
    if lines <= ROUTER_BRAIN_MAX:
        return Grade.PASS
    if lines <= GIANT_MAX:
        return Grade.WARN
    return Grade.FAIL


def gate_file(path: str | Path) -> GateResult:
    """对单个 SKILL.md 文件进行行数门控。"""
    p = Path(path)
    n = count_skill_lines(p)
    return GateResult(path=str(p), lines=n, grade=_classify(n))


def gate_directory(skills_root: str | Path) -> GateReport:
    """扫描 skills_root 下所有 SKILL.md，返回汇总报告。

    跳过 __pycache__ 和以 '.' 开头的目录。
    """
    root = Path(skills_root)
    report = GateReport()

    for p in root.rglob("SKILL.md"):
        # 跳过 __pycache__ 和 . 目录（只检查相对路径内的部分）
        try:
            rel_parts = p.relative_to(root).parts
        except ValueError:
            rel_parts = p.parts
        if any(part.startswith(".") or part == "__pycache__" for part in rel_parts):
            continue

        result = gate_file(p)
        report.results.append(result)
        report.total += 1

        if result.grade == Grade.PASS:
            report.pass_count += 1
        elif result.grade == Grade.WARN:
            report.warn_count += 1
        else:
            report.fail_count += 1

    return report


# ── 报告渲染 ──────────────────────────────────────────────────────
def render_report(report: GateReport, top_n: int = 10) -> str:
    """生成纯文本报告：三档计数 + FAIL全列 + WARN降序前top_n。"""
    lines: list[str] = []
    lines.append(f"=== Skill出厂行数门报告 ===")
    lines.append(f"总计: {report.total}  "
                 f"PASS(≤{ROUTER_BRAIN_MAX}): {report.pass_count}  "
                 f"WARN({ROUTER_BRAIN_MAX+1}-{GIANT_MAX}): {report.warn_count}  "
                 f"FAIL(>{GIANT_MAX}): {report.fail_count}")
    lines.append("")

    # FAIL 全列
    fails = [r for r in report.results if r.grade == Grade.FAIL]
    if fails:
        lines.append(f"── FAIL ({len(fails)}) ──")
        for r in sorted(fails, key=lambda x: -x.lines):
            rel = _relative(r.path)
            lines.append(f"  {r.lines:>5}行  {rel}")
        lines.append("")

    # WARN 按行数降序列前 top_n
    warns = [r for r in report.results if r.grade == Grade.WARN]
    if warns:
        sorted_warns = sorted(warns, key=lambda x: -x.lines)[:top_n]
        shown = min(top_n, len(warns))
        lines.append(f"── WARN (显示 {shown}/{len(warns)}) ──")
        for r in sorted_warns:
            rel = _relative(r.path)
            lines.append(f"  {r.lines:>5}行  {rel}")
        lines.append("")

    if not fails and not warns:
        lines.append("✅ 全部合格，无超标项。")

    return "\n".join(lines)


def _relative(path: str) -> str:
    """尽量返回相对于常见根目录的相对路径。"""
    p = Path(path)
    for candidate in [Path.home() / ".hermes" / "skills", Path.cwd()]:
        try:
            return str(p.relative_to(candidate))
        except ValueError:
            continue
    return str(p)


# ── CLI ───────────────────────────────────────────────────────────
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Skill出厂行数门 — 扫描SKILL.md行数，三档门控。"
    )
    parser.add_argument(
        "--root",
        default=os.path.expanduser("~/.hermes/skills"),
        help="skill根目录 (默认: ~/.hermes/skills)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="输出格式 (默认: text)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="WARN列表最多显示条数 (默认: 10)",
    )

    args = parser.parse_args(argv)
    report = gate_directory(args.root)

    if args.format == "json":
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_report(report, top_n=args.top_n))

    # 有 FAIL 返回 1，否则 0
    sys.exit(1 if report.fail_count > 0 else 0)


if __name__ == "__main__":
    main()
