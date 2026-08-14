"""
路由健康度检查脚本 — 考核机制自动化

日检级别：JSONL文件存在+非空+轮转正常
输出：健康/警告/告警 三级
退出码：0=健康, 1=警告, 2=告警
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime


LOG_PATH = Path("~/.openllm/router_log.jsonl").expanduser()
MAX_LOG_SIZE = 2 * 1024 * 1024  # 2MB未轮转=警告


def check_health():
    """执行健康检查，返回(级别, 摘要, 详情)。"""
    issues = []
    info = []

    # 1. 文件存在性
    if not LOG_PATH.exists():
        return "alert", "路由日志文件不存在", ["router_log.jsonl not found"]

    # 2. 文件非空
    size = LOG_PATH.stat().st_size
    if size == 0:
        return "alert", "路由日志文件为空", ["router_log.jsonl is 0 bytes"]

    info.append(f"日志大小: {size} bytes")

    # 3. 轮转检查
    if size > MAX_LOG_SIZE:
        issues.append(f"日志未轮转: {size} bytes > {MAX_LOG_SIZE} bytes")

    # 4. 格式完整性
    lines = []
    try:
        content = LOG_PATH.read_text(encoding="utf-8").strip()
        lines = content.split("\n") if content else []
        valid = 0
        invalid = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                json.loads(line)
                valid += 1
            except json.JSONDecodeError:
                invalid += 1
        info.append(f"有效记录: {valid}, 无效行: {invalid}")
        if invalid > 0:
            issues.append(f"JSON解析失败: {invalid}行")
    except Exception as e:
        issues.append(f"读取日志失败: {e}")
        invalid = 0
        valid = 0

    # 5. 路由统计分析
    if valid > 0:
        phase_stats = {}
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                for d in rec.get("decisions", []):
                    phase = d["phase"]
                    action = d["action"]
                    if phase not in phase_stats:
                        phase_stats[phase] = {"RUN": 0, "SKIP": 0, "DEGRADED": 0}
                    phase_stats[phase][action] = phase_stats[phase].get(action, 0) + 1
            except json.JSONDecodeError:
                continue

        # 分析各阶段
        for phase, counts in phase_stats.items():
            total = sum(counts.values())
            if total == 0:
                continue
            skip_rate = counts.get("SKIP", 0) / total
            degrade_rate = counts.get("DEGRADED", 0) / total

            if skip_rate > 0.5:
                issues.append(f"{phase} SKIP率过高: {skip_rate:.0%}")
            if skip_rate < 0.05 and phase in ("ACT", "OBSERVE"):
                info.append(f"{phase} SKIP率低: {skip_rate:.0%}（路由可能无效）")

        info.append(f"阶段统计: {json.dumps(phase_stats, ensure_ascii=False)}")

    # 判定级别
    if any("过高" in i or "失败" in i or "不存在" in i or "为空" in i for i in issues):
        return "alert", f"路由健康告警 ({len(issues)}项)", issues + info
    elif issues:
        return "warning", f"路由健康警告 ({len(issues)}项)", issues + info
    else:
        return "healthy", f"路由健康 ({valid}条记录)", info


def main():
    level, summary, details = check_health()

    icon = {"healthy": "🟢", "warning": "🟡", "alert": "🔴"}[level]
    print(f"\n{icon} 路由健康度检查 · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   {summary}")
    for d in details:
        print(f"   · {d}")
    print()

    sys.exit({"healthy": 0, "warning": 1, "alert": 2}[level])


if __name__ == "__main__":
    main()
