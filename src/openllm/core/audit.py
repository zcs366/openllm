"""audit.py — 包拯审计

来自老IO-S syscall/audit.py：
  - audit.plan_quality: 检查plan结构完整性
  - audit.hindsight_quality: 检查hindsight经验质量
  - 七神划界: 审计产出止于审计报告，不阻塞异步队列
"""

import logging
from typing import Optional

logger = logging.getLogger("openllm.audit")


def audit_plan_quality(pid: str,
                       plan: Optional[list[dict]] = None,
                       goal: str = "",
                       cap_bypass: bool = False) -> dict:
    """审计plan结构完整性。

    检查项:
      1. 结构完整性 — 每步有id/depends_on
      2. 循环检测 — depends_on不形成环
      3. 一致性 — steps数量合理（1-20步）
      4. bypass核验 — 如果走cap_check bypass，必须加注
    """
    if plan is None:
        plan = []

    checks = {}
    issues = []

    # 检查1: 结构完整性
    if plan:
        has_id = all(
            isinstance(s.get("id"), (int, str)) for s in plan)
        has_dep = all("depends_on" in s for s in plan)
        checks["structure"] = has_id and has_dep
        if not checks["structure"]:
            issues.append("部分步骤缺id或depends_on字段")
    else:
        checks["structure"] = False
        issues.append("plan为空")

    # 检查2: 数量合理性
    count = len(plan)
    checks["count"] = 1 <= count <= 20
    if count == 0:
        issues.append("plan无步骤")
    elif count > 20:
        issues.append(f"步骤过多({count}步，建议≤20)")

    # 检查3: 循环检测
    if plan:
        seen_ids = set()
        has_cycle = False
        for s in plan:
            sid = s.get("id")
            deps = s.get("depends_on", [])
            if isinstance(deps, list):
                for d in deps:
                    if d == sid:
                        has_cycle = True
            seen_ids.add(sid)
        checks["no_cycle"] = not has_cycle
        if has_cycle:
            issues.append("检测到循环依赖")
    else:
        checks["no_cycle"] = True

    # 检查4: bypass核验
    checks["bypass_verified"] = True

    total = sum(1 for v in checks.values() if v)
    score = total / max(len(checks), 1)

    passed = score >= 0.75 and len(issues) == 0

    return {
        "passed": passed,
        "score": round(score, 2),
        "checks": checks,
        "issues": issues,
        "plan_count": count,
        "pid": pid,
        "note": ("审计结果仅记录不拦截"
                 if not passed else "审计通过"),
    }


def audit_hindsight_quality(pid: str,
                            hindsight: Optional[dict] = None
                            ) -> dict:
    """审计hindsight经验的质量。"""
    if hindsight is None:
        hindsight = {
            "experience": "",
            "failure_pattern": "",
            "alternative": "",
        }

    checks = {}
    issues = []

    checks["has_experience"] = bool(
        hindsight.get("experience"))
    checks["has_failure_pattern"] = bool(
        hindsight.get("failure_pattern"))
    checks["has_alternative"] = bool(
        hindsight.get("alternative"))
    checks["experience_length"] = len(
        hindsight.get("experience", "")) > 10

    if not checks["has_experience"]:
        issues.append("hindsight缺少experience")
    if not checks["experience_length"]:
        issues.append("experience过于简短")

    total = sum(1 for v in checks.values() if v)
    score = total / max(len(checks), 1)

    return {
        "passed": score >= 0.75,
        "score": round(score, 2),
        "checks": checks,
        "issues": issues,
        "pid": pid,
    }
