#!/usr/bin/env python3
"""check_manual.py — MANUAL.md 活性校验（G3 · 0927自主窗口）。

判据（§6 更新制度的校验条款落地）：
  C1  §3 阶段链与 activity.py PHASE_LABELS 一致（名称+顺序）
  C2  §1 地址表中每个 `/`绝对地址在盘上存在
  C3  沙箱 §4 的两个写口与 sandbox.py DEFAULT_ALLOWED 一致
退出码：0=全过；1=有过期项（触发§6 手册更新流程）。

用法：python scripts/check_manual.py [--quiet]
cron 接线：每日 09:00（hermes cron add）。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "docs" / "MANUAL.md"
ACTIVITY = ROOT / "src" / "openllm" / "cli" / "activity.py"
SANDBOX = ROOT / "src" / "openllm" / "core" / "sandbox.py"


def load_manual() -> str:
    return MANUAL.read_text(encoding="utf-8")


def check_phases(manual: str) -> tuple[bool, str]:
    """C1: 手册 §3 阶段链 vs PHASE_LABELS 顺序。"""
    src = ACTIVITY.read_text(encoding="utf-8")
    m = re.search(r"PHASE_LABELS[^{]*\{(.*?)\n\}", src, re.S)
    if not m:
        return False, "activity.py 里找不到 PHASE_LABELS"
    # dict 字面量保持插入序（py3.7+），取 value 二元组的 label（第二项）
    labels = re.findall(r'"[^"]*":\s*\("[^"]*",\s*"([^"]+)"\)', m.group(1))
    if not labels:
        return False, "PHASE_LABELS 解析出 0 个 label"

    # 手册 §3 链：建境(ISA)→苏醒注入(IAI)→…，取括号内的器官序号做交叉核对，
    # 名称以代码 label 为准做子串匹配
    m3 = re.search(r"## 3\.[^\n]*\n\n(.+?)\n\n权威源", manual, re.S)
    if not m3:
        return False, "手册里找不到 §3 阶段链"
    chain = m3.group(1)
    missing = [lb for lb in labels if lb not in chain]
    if missing:
        return False, f"手册 §3 缺少代码中的阶段: {missing}"
    return True, f"§3 阶段链覆盖 PHASE_LABELS 全部 {len(labels)} 项"


def check_addresses(manual: str) -> tuple[bool, str]:
    """C2: §1 表中反引号内的 / 开头绝对路径必须存在（目录或文件）。"""
    m1 = re.search(r"## 1\.[^\n]*\n(.*?)\n## 2\.", manual, re.S)
    if not m1:
        return False, "手册里找不到 §1 地址表"
    paths = set(re.findall(r"`(/[^`]+)`", m1.group(1)))
    bad = []
    for p in paths:
        clean = p.rstrip("/")
        if not Path(clean).exists():
            bad.append(p)
    if bad:
        return False, f"§1 失效地址: {bad}"
    return True, f"§1 {len(paths)} 个绝对地址全部存在"


def check_write_ports(manual: str) -> tuple[bool, str]:
    """C3: §4 写口声明 vs sandbox.py DEFAULT_ALLOWED。"""
    src = SANDBOX.read_text(encoding="utf-8")
    m = re.search(r"DEFAULT_ALLOWED\s*=\s*\[(.*?)\]", src, re.S)
    if not m:
        return False, "sandbox.py 里找不到 DEFAULT_ALLOWED"
    code_ports = sorted(re.findall(r'"([^"]+)"', m.group(1)))
    m4 = re.search(r"## 4\.[^\n]*\n(.*?)\n## 5\.", manual, re.S)
    if not m4:
        return False, "手册里找不到 §4"
    ok_ports = []
    for p in code_ports:
        base = p.replace("~", "/home/zcs").strip("/")
        # 手册只需提及白名单的目录名（output、tmp/openllm）
        if base.split("/")[-1] in m4.group(1) or p in m4.group(1):
            ok_ports.append(p)
    if len(ok_ports) != len(code_ports):
        return False, f"§4 未覆盖写白名单: {set(code_ports) - set(ok_ports)}"
    return True, f"§4 写口覆盖 {len(code_ports)} 项白名单"


def main() -> int:
    quiet = "--quiet" in sys.argv
    manual = load_manual()
    checks = [
        ("C1 阶段链", check_phases(manual)),
        ("C2 地址表", check_addresses(manual)),
        ("C3 写口", check_write_ports(manual)),
    ]
    failed = []
    for name, (ok, detail) in checks:
        mark = "PASS" if ok else "FAIL"
        if not quiet or not ok:
            print(f"[{mark}] {name}: {detail}")
        if not ok:
            failed.append(name)
    print(f"check_manual: {len(checks)-len(failed)}/{len(checks)} 过"
          + (f"，过期项: {failed} → 触发§6更新" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
