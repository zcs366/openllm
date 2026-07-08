"""hindsight_loop.py — 经验闭环

来自老IO-S syscall/hindsight_loop.py：
  - 子Agent执行完成后自动提取经验
  - 写入对应Process的hindsight字段
  - 下次同类任务自动注入经验
  - 七神划界（阿波罗）：hindsight只读不碰状态机
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("openllm.hindsight")

OPENLLM_HOME = Path.home() / ".openllm"
HINDSIGHT_DIR = OPENLLM_HOME / "hindsight"


def extract_hindsight(pid: str, goal: str, result: dict,
                      failure: str = "") -> dict:
    """从执行结果提取hindsight经验。

    持久化到HINDSIGHT_DIR供跨session复用。

    Returns: {experience, failure_pattern, alternative}
    """
    HINDSIGHT_DIR.mkdir(parents=True, exist_ok=True)

    # 简单的经验提取逻辑（不依赖外部planner）
    experience = ""
    failure_pattern = ""
    alternative = ""

    success = result.get("success", True)
    if success:
        execution_time = result.get("execution_time", 0)
        experience = (
            f"任务 '{goal[:80]}' 成功完成"
            f"（耗时{execution_time:.1f}s）")
    else:
        error = failure or result.get("error", "未知错误")
        failure_pattern = _classify_failure(error)
        experience = (
            f"任务 '{goal[:80]}' 失败: {error[:100]}")
        alternative = _suggest_alternative(failure_pattern)

    h = {
        "experience": experience,
        "failure_pattern": failure_pattern,
        "alternative": alternative,
    }

    # 持久化
    record = {
        "pid": pid,
        "goal_preview": goal[:120],
        "hindsight": h,
        "result": {
            k: v for k, v in result.items()
            if k in ("success", "execution_time", "token_count")
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    path = HINDSIGHT_DIR / (
        f"{pid}_{int(datetime.now().timestamp())}.json")
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2))

    return h


def inject_hindsight(pid: str) -> str:
    """为指定进程注入历史hindsight经验作为上下文。

    搜索HINDSIGHT_DIR中同pid的历史记录，
    返回格式化的经验文本。
    """
    HINDSIGHT_DIR.mkdir(parents=True, exist_ok=True)
    records = sorted(
        HINDSIGHT_DIR.glob(f"{pid}_*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True)

    if not records:
        return ""

    experiences = []
    for rp in records[:3]:
        try:
            rec = json.loads(rp.read_text())
            h = rec.get("hindsight", {})
            exp = h.get("experience", "")
            if exp:
                experiences.append(f"- {exp}")
            alt = h.get("alternative", "")
            if alt:
                experiences.append(f"  → 建议: {alt}")
        except (json.JSONDecodeError, IOError):
            continue

    if not experiences:
        return ""

    return "\n".join(["【历史经验参考】"] + experiences)


def get_recent_hindsight(limit: int = 5) -> list[dict]:
    """获取最近的hindsight经验（供审计/评估使用）。"""
    HINDSIGHT_DIR.mkdir(parents=True, exist_ok=True)
    records = sorted(
        HINDSIGHT_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for rp in records[:limit]:
        try:
            out.append(json.loads(rp.read_text()))
        except (json.JSONDecodeError, IOError):
            continue
    return out


def _classify_failure(error: str) -> str:
    """简单失败模式分类。"""
    error_lower = error.lower()
    if "timeout" in error_lower:
        return "timeout"
    if "permission" in error_lower or "denied" in error_lower:
        return "permission"
    if "not found" in error_lower:
        return "resource_missing"
    if "memory" in error_lower or "oom" in error_lower:
        return "resource_exhaustion"
    if "syntax" in error_lower or "parse" in error_lower:
        return "input_format"
    return "unknown"


def _suggest_alternative(failure_pattern: str) -> str:
    """根据失败模式建议替代方案。"""
    suggestions = {
        "timeout": "增加超时时间或拆分任务",
        "permission": "检查cap_policy或请求人工授权",
        "resource_missing": "检查资源路径或创建资源",
        "resource_exhaustion": "释放内存或减少并发",
        "input_format": "检查输入格式或使用不同解析器",
    }
    return suggestions.get(failure_pattern, "人工介入")
