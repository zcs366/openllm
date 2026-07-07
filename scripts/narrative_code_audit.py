#!/usr/bin/env python3
"""
叙事-代码季度对账脚本

阿佛洛狄忒启示：壮美叙事必须每季度用冷水浇。
不是质疑方向，是质疑进度——"我们说的那些壮美 things，
到底有多少变成了代码？"

用法：
    python3 scripts/narrative_code_audit.py
    python3 scripts/narrative_code_audit.py --threshold 60
"""
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path.home() / "projects" / "openllm"
NORTH_STAR = PROJECT_ROOT / "docs" / "NORTH_STAR.md"
SRC_DIR = PROJECT_ROOT / "src" / "openllm"

# 壮美承诺 → 代码验证规则
PROMISES = [
    {
        "id": 1,
        "promise": "Agent能说不（拒绝权）",
        "check_files": ["governance/rejection.py"],
        "check_patterns": [r"class.*Rejection", r"def.*reject", r"RejectionReason"],
    },
    {
        "id": 2,
        "promise": "六体自监督闭环运行",
        "check_files": ["governance/feedback_loop.py"],
        "check_patterns": [r"class FeedbackLoop", r"def collect_feedback", r"def apply_feedback"],
    },
    {
        "id": 3,
        "promise": "SSP v0.0.1发布",
        "check_files": ["docs/SSP-v0.0.1.md"],
        "check_patterns": [r"Self-Supervision Protocol"],
    },
    {
        "id": 4,
        "promise": "宪法层不可被Agent修改",
        "check_files": ["constitution.py"],
        "check_patterns": [r"CONSTITUTION", r"BLOCKED_FILES", r"check_constitution"],
    },
    {
        "id": 5,
        "promise": "好奇心最小实现（空闲期协议）",
        "check_files": ["core/idle_wander.py"],
        "check_patterns": [r"class IdleWanderer", r"def wander", r"WANDER_DOMAINS"],
    },
    {
        "id": 6,
        "promise": "人-Agent共栖关系验证",
        "check_files": [],
        "check_patterns": [],
    },
    {
        "id": 7,
        "promise": "自创生能力（Agent修改自身规则）",
        "check_files": [],
        "check_patterns": [],
    },
]


def check_file_exists(rel_path: str) -> bool:
    """检查文件是否存在"""
    full = SRC_DIR / rel_path
    if not full.exists():
        # 也检查docs目录
        full = PROJECT_ROOT / rel_path
    return full.exists()


def check_patterns_in_file(rel_path: str, patterns: list[str]) -> int:
    """检查文件中是否包含指定模式，返回匹配数"""
    full = SRC_DIR / rel_path
    if not full.exists():
        full = PROJECT_ROOT / rel_path
    if not full.exists():
        return 0

    try:
        content = full.read_text(errors="replace")
    except Exception:
        return 0

    matches = 0
    for pat in patterns:
        if re.search(pat, content):
            matches += 1
    return matches


def classify_status(promise: dict) -> str:
    """分类承诺状态"""
    if not promise["check_files"] and not promise["check_patterns"]:
        return "❌ 未开始"

    files_found = sum(1 for f in promise["check_files"] if check_file_exists(f))
    total_files = len(promise["check_files"])

    if total_files == 0:
        return "❌ 未开始"

    if files_found == total_files:
        # 检查模式匹配
        pattern_matches = 0
        total_patterns = 0
        for i, f in enumerate(promise["check_files"]):
            pats = promise["check_patterns"][i * 2:(i + 1) * 2] if len(promise["check_patterns"]) > i * 2 else promise["check_patterns"]
            total_patterns += len(pats)
            pattern_matches += check_patterns_in_file(f, pats)

        if total_patterns > 0 and pattern_matches >= total_patterns * 0.5:
            return "✅ 已实现"
        else:
            return "🟡 进行中"
    elif files_found > 0:
        return "🟡 进行中"
    else:
        return "❌ 未开始"


def main():
    threshold = 50
    if "--threshold" in sys.argv:
        idx = sys.argv.index("--threshold")
        if idx + 1 < len(sys.argv):
            threshold = int(sys.argv[idx + 1])

    print("=" * 60)
    print("叙事-代码对账报告")
    print(f"项目: openLLM · 阈值: {threshold}%")
    print("=" * 60)

    results = []
    for p in PROMISES:
        status = classify_status(p)
        results.append({"promise": p["promise"], "status": status})

    # 输出表格
    implemented = 0
    in_progress = 0
    not_started = 0

    for r in results:
        print(f"  {r['status']}  {r['promise']}")
        if "✅" in r["status"]:
            implemented += 1
        elif "🟡" in r["status"]:
            in_progress += 1
        else:
            not_started += 1

    total = len(results)
    impl_pct = (implemented / total * 100) if total > 0 else 0
    active_pct = ((implemented + in_progress) / total * 100) if total > 0 else 0

    print()
    print(f"统计: ✅{implemented} 🟡{in_progress} ❌{not_started}")
    print(f"已实现: {impl_pct:.0f}% · 含进行中: {active_pct:.0f}%")

    if not_started / total * 100 > threshold:
        print()
        print(f"⚠️ 降温警告: 未开始>{threshold}% ({not_started}/{total})")
        print("   叙事应该降温——诚实面对进度。")
        return 1
    else:
        print()
        print("✅ 进度健康。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
