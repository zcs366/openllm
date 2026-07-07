#!/usr/bin/env python3
"""
constitution.py 修改拦截器

赫淮斯托斯启示：宪法层代码必须张成市亲手写，永远不被Agent覆盖。
此脚本作为pre-commit hook或CI检查使用。

用法：
    # 作为pre-commit hook
    python3 scripts/protect_constitution.py

    # 作为CI检查
    python3 scripts/protect_constitution.py --ci

退出码：
    0 = 合宪（无修改或人类修改）
    1 = 违禁（Agent尝试修改宪法文件）
"""
import subprocess
import sys
from pathlib import Path

CONSTITUTION_PATH = Path(__file__).parent.parent / "src" / "openllm" / "constitution.py"

# 被保护的文件列表
PROTECTED_FILES = [
    "src/openllm/constitution.py",
]


def get_git_diff_files() -> list[str]:
    """获取git暂存区中被修改的文件"""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        pass
    return []


def get_git_author() -> str:
    """获取最后git commit的作者"""
    try:
        result = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def check_constitution_protection(ci_mode: bool = False) -> bool:
    """检查宪法文件是否被修改"""
    # 获取修改的文件列表
    if ci_mode:
        # CI模式：检查git diff（不限于暂存区）
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True, text=True, timeout=10,
                cwd=str(Path(__file__).parent.parent),
            )
            modified = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        except Exception:
            modified = []
    else:
        modified = get_git_diff_files()

    # 检查是否有被保护的文件
    violations = []
    for modified_file in modified:
        for protected in PROTECTED_FILES:
            if modified_file == protected or modified_file.endswith("/" + protected):
                violations.append(modified_file)

    if violations:
        author = get_git_author()
        print("🔴 宪法违禁——检测到对受保护文件的修改：")
        for v in violations:
            print(f"  文件: {v}")
        print(f"  作者: {author}")
        print()
        print("宪法层文件必须由张成市亲手修改。")
        print("如果是人类修改，请用 --force 跳过检查。")
        return False

    return True


def main():
    ci_mode = "--ci" in sys.argv
    force = "--force" in sys.argv

    if force:
        print("⚠️ --force 跳过宪法保护检查")
        return 0

    ok = check_constitution_protection(ci_mode)
    if ok:
        print("✅ 宪法文件未被修改——合宪")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
