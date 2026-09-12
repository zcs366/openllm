"""
SubstrateEnforcer — ISN约束→IKO引导编译器
==========================================

2026-07-07 PAL P1-2 · 七神+核战队终裁
论文来源: Steerability via constraints (2607.02389)
设计原则: 约束应引导认知（in-context），非事后检查（post-hoc）

核心机制:
1. 读取ISN constraint_schema.json
2. 将约束编译为in-context提示
3. 注入到IKO输出模板中（生成前引导）
4. 保留post-hoc检查作为双重保障

约束分级:
- block: 硬拦截（不可违反）
- warn: 强制确认（需用户确认）
- info: 建议（可忽略）
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

CONSTRAINT_SCHEMA_PATH = Path.home() / ".hermes" / "scripts" / "isn_constraint_schema.json"


@dataclass
class CompiledConstraint:
    """编译后的约束。"""
    id: str
    category: str
    severity: str
    description: str
    check_method: str
    in_context_hint: str  # 注入到prompt的引导文本
    regex_pattern: Optional[str] = None  # 用于post-hoc检查的正则


@dataclass
class SubstrateGuide:
    """Substrate引导结果。"""
    content: str  # 注入到IKO模板的引导文本
    constraints_count: int
    block_count: int
    warn_count: int
    info_count: int
    categories: List[str]


class SubstrateEnforcer:
    """ISN约束→IKO引导编译器。

    核心流程:
    1. load_schema() — 加载constraint_schema.json
    2. compile() — 将约束编译为in-context提示
    3. guide() — 生成引导文本
    4. check() — post-hoc验证（双重保障）

    设计约束:
    - 延迟<10ms（编译结果可缓存）
    - 不修改constraint_schema.json（只读）
    - 新约束自动被编译（schema更新后重新compile）
    """

    def __init__(self, schema_path: Path = CONSTRAINT_SCHEMA_PATH):
        self.schema_path = schema_path
        self._constraints: List[CompiledConstraint] = []
        self._compiled_content: Optional[str] = None
        self._schema_hash: Optional[str] = None

    def load_schema(self) -> List[dict]:
        """加载constraint_schema.json。

        支持三种格式:
        1. 约束实例列表: [{"id": "c1", ...}, ...]
        2. {"constraints": [...]} 或 {"items": [...list...]}
        3. JSON Schema格式: {"examples": [[{约束实例}]]} — 取examples[0]
        """
        if not self.schema_path.exists():
            return []
        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        if isinstance(schema, list):
            return schema
        # 直接constraints/items是list的情况
        for key in ("constraints", "items"):
            val = schema.get(key)
            if isinstance(val, list) and val:
                return val
        # JSON Schema格式: examples是[[{约束实例}]]
        examples = schema.get("examples")
        if isinstance(examples, list) and examples:
            first = examples[0]
            if isinstance(first, list):
                return first
            if isinstance(first, dict):
                return [first]
        return []

    def compile(self) -> str:
        """将约束编译为in-context提示。

        Returns:
            编译后的引导文本
        """
        raw = self.load_schema()
        self._constraints = []

        # 按severity分组
        by_severity = {"block": [], "warn": [], "info": []}

        for item in raw:
            if not isinstance(item, dict):
                continue

            cid = item.get("id", "")
            category = item.get("category", "quality")
            severity = item.get("severity", "info")
            desc = item.get("description", "")
            check = item.get("check_method", "manual")

            # 生成in-context提示
            hint = self._make_hint(cid, category, severity, desc)

            # 生成post-hoc正则（简单模式匹配）
            regex = self._make_regex(desc)

            constraint = CompiledConstraint(
                id=cid,
                category=category,
                severity=severity,
                description=desc,
                check_method=check,
                in_context_hint=hint,
                regex_pattern=regex,
            )
            self._constraints.append(constraint)
            by_severity.get(severity, []).append(constraint)

        # 编译为引导文本
        parts = []
        parts.append("[ISN Substrate约束] 生成代码时遵守以下规则:")

        if by_severity["block"]:
            parts.append("⛔ 不可违反:")
            for c in by_severity["block"]:
                parts.append(f"  - {c.in_context_hint}")

        if by_severity["warn"]:
            parts.append("⚠️ 需确认:")
            for c in by_severity["warn"]:
                parts.append(f"  - {c.in_context_hint}")

        if by_severity["info"]:
            parts.append("💡 建议:")
            for c in by_severity["info"][:5]:  # info最多5条
                parts.append(f"  - {c.in_context_hint}")

        self._compiled_content = "\n".join(parts)
        return self._compiled_content

    def guide(self, task_type: str = "code") -> SubstrateGuide:
        """生成引导文本。

        Args:
            task_type: 任务类型（code|document|analysis）

        Returns:
            SubstrateGuide
        """
        if self._compiled_content is None:
            self.compile()

        # 按任务类型过滤相关约束
        relevant = [c for c in self._constraints if self._is_relevant(c, task_type)]

        block_count = sum(1 for c in relevant if c.severity == "block")
        warn_count = sum(1 for c in relevant if c.severity == "warn")
        info_count = sum(1 for c in relevant if c.severity == "info")
        categories = list(set(c.category for c in relevant if c.category))

        # 构建引导文本
        parts = [self._compiled_content]

        # 任务特定提示
        if task_type == "code":
            parts.append("\n[代码生成注意] 生成前检查:")
            parts.append("  1. 所有文件路径是否在白名单内")
            parts.append("  2. 是否使用了被禁止的函数（eval/exec等）")
            parts.append("  3. 输入是否经过验证")
        elif task_type == "document":
            parts.append("\n[文档生成注意] 生成前检查:")
            parts.append("  1. 格式是否符合规范")
            parts.append("  2. 是否包含必要的元数据")

        content = "\n".join(parts)

        return SubstrateGuide(
            content=content,
            constraints_count=len(relevant),
            block_count=block_count,
            warn_count=warn_count,
            info_count=info_count,
            categories=categories,
        )

    def check(self, generated_code: str) -> List[Dict]:
        """Post-hoc验证：检查生成的代码是否违反约束。

        Returns:
            [{"constraint_id": str, "severity": str, "message": str, "matched": str}]
        """
        violations = []
        for c in self._constraints:
            if c.regex_pattern and c.check_method in ("auto", "hybrid"):
                match = re.search(c.regex_pattern, generated_code)
                if match:
                    violations.append({
                        "constraint_id": c.id,
                        "severity": c.severity,
                        "message": f"违反约束 {c.id}: {c.description}",
                        "matched": match.group(0)[:100],
                    })
        return violations

    def _make_hint(self, cid: str, category: str, severity: str, desc: str) -> str:
        """生成单条约束的in-context提示。"""
        prefix = {
            "safety": "安全",
            "quality": "质量",
            "format": "格式",
            "boundary": "边界",
        }.get(category, category)

        return f"[{prefix}] {desc}"

    def _make_regex(self, desc: str) -> Optional[str]:
        """从约束描述生成简单的正则匹配模式。"""
        # 简单模式：检测常见危险函数
        danger_patterns = {
            "eval": r"\beval\s*\(",
            "exec": r"\bexec\s*\(",
            "os.system": r"\bos\.system\s*\(",
            "subprocess": r"\bsubprocess\.call\s*\(",
            "pickle": r"\bpickle\.loads?\s*\(",
            "shell": r"\bshell\s*=\s*True",
        }
        desc_lower = desc.lower()
        for keyword, pattern in danger_patterns.items():
            if keyword in desc_lower:
                return pattern
        return None

    def _is_relevant(self, constraint: CompiledConstraint, task_type: str) -> bool:
        """判断约束是否与任务类型相关。"""
        # safety约束总是相关
        if constraint.category == "safety":
            return True
        # format约束与code/document相关
        if constraint.category == "format" and task_type in ("code", "document"):
            return True
        # boundary约束总是相关
        if constraint.category == "boundary":
            return True
        # quality约束与所有类型相关
        if constraint.category == "quality":
            return True
        return False


# ── 便捷函数 ──

_default_enforcer: Optional[SubstrateEnforcer] = None


def get_enforcer() -> SubstrateEnforcer:
    """获取全局SubstrateEnforcer实例（懒加载）。"""
    global _default_enforcer
    if _default_enforcer is None:
        _default_enforcer = SubstrateEnforcer()
    return _default_enforcer


def get_guide(task_type: str = "code") -> str:
    """获取引导文本（便捷函数）。"""
    return get_enforcer().guide(task_type).content


def check_code(code: str) -> List[Dict]:
    """检查代码（便捷函数）。"""
    return get_enforcer().check(code)
