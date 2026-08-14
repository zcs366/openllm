"""
OpenSkill Adapter — 开放世界自进化框架 → UnifiedSkillConfig
===========================================================

OpenSkill 核心思想：技能可以从外部来源（文档、代码仓库、网页）中
自动学习和提取，并通过自建虚拟任务验证质量，最终生成可复用的
UnifiedSkillConfig 配置。

本模块实现：
  1. learn_from_docs() — 从文档中提取技能（API文档、教程、规范）
  2. learn_from_repo()  — 从代码仓库中提取技能（README、函数签名、模式）
  3. validate_skill()   — 通过虚拟任务验证技能质量
  4. SourceTraceability 自动填充（来源溯源）

关键概念：
  - SkillExtraction: 从来源中提取的原始技能片段
  - VirtualTask: 自建的验证任务，用于评估技能质量
  - SkillScore: 技能在虚拟任务上的表现评分
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("openllm.isn.openskill_adapter")


# ══════════════════════════════════════════════════════════════════════════════
# Enums & Data Structures
# ══════════════════════════════════════════════════════════════════════════════

class SourceType(str, Enum):
    """外部来源类型。"""
    DOCS = "docs"       # 文档（API文档、教程、规范）
    REPO = "repo"       # 代码仓库（README、源码、配置）
    WEBPAGE = "webpage" # 网页内容


class ExtractionConfidence(str, Enum):
    """提取置信度。"""
    HIGH = "high"       # 明确的技能模式（函数签名、步骤列表等）
    MEDIUM = "medium"   # 隐含的技能模式（段落推断、上下文分析）
    LOW = "low"         # 弱信号（关键词匹配、模式猜测）


@dataclass
class SkillExtraction:
    """从外部来源中提取的原始技能片段。

    这是 OpenSkill 提取管线的中间产物。每个 extraction 代表一个
    从来源中识别出的、潜在可复用的技能单元。
    """
    name: str                      # 技能名称
    description: str               # 技能描述
    source_type: SourceType        # 来源类型
    source_url: str = ""           # 来源 URL / 路径
    content: str = ""              # 提取的原始内容
    confidence: ExtractionConfidence = ExtractionConfidence.MEDIUM
    tags: List[str] = field(default_factory=list)
    author: str = ""               # 原始作者（如果有）
    license: str = ""              # 来源许可证
    version: str = "0.0.1"         # 版本
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """基于内容生成唯一指纹（用于去重）。"""
        raw = f"{self.name}:{self.description}:{self.source_url}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


@dataclass
class VirtualTask:
    """自建的虚拟验证任务。

    OpenSkill 的核心验证机制：为每个提取的技能构建一组
    虚拟任务，模拟真实使用场景，评估技能质量。
    """
    task_id: str
    description: str
    input_data: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    expected_pattern: str = ""   # 正则模式匹配
    difficulty: str = "medium"   # easy / medium / hard
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillScore:
    """技能在虚拟任务上的评分结果。"""
    task_id: str
    passed: bool
    score: float = 0.0       # 0.0 ~ 1.0
    latency_ms: float = 0.0
    error: str = ""
    output: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Extraction Rules — 从来源中提取技能的规则
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractionRule:
    """单条提取规则。"""
    pattern: str                  # 正则模式
    confidence: ExtractionConfidence
    tag: str                      # 匹配后附加的标签
    name_template: str = ""       # 技能名称模板（{0} 为匹配内容）
    description_template: str = ""  # 技能描述模板
    use_dotall: bool = False      # 是否启用 re.DOTALL（跨行匹配）


# 默认文档提取规则
DEFAULT_DOC_RULES: List[ExtractionRule] = [
    # API 端点模式
    ExtractionRule(
        pattern=r"(GET|POST|PUT|DELETE|PATCH)\s+(/[^\\s]+)",
        confidence=ExtractionConfidence.HIGH,
        tag="api-endpoint",
        name_template="api-{0}-{1}",
        description_template="API {0} endpoint: {1}",
    ),
    # 步骤列表模式 (1. xxx, 2. xxx)
    ExtractionRule(
        pattern=r"(?:^|\n)\s*\d+\.\s+(.+)",
        confidence=ExtractionConfidence.HIGH,
        tag="step-list",
        name_template="procedure",
        description_template="Procedure: {0}",
    ),
    # 代码块模式
    ExtractionRule(
        pattern=r"```(\w+)?\n(.*?)```",
        confidence=ExtractionConfidence.MEDIUM,
        tag="code-block",
        name_template="code-example-{0}",
        description_template="Code example in {0}",
        use_dotall=True,
    ),
    # "How to" 模式
    ExtractionRule(
        pattern=r"(?:How to|How do you|Guide to)\s+(.+?)(?:\.|$)",
        confidence=ExtractionConfidence.MEDIUM,
        tag="howto",
        name_template="howto-{0}",
        description_template="How-to: {0}",
    ),
    # 命令行模式
    ExtractionRule(
        pattern=r"\$\s+(.+)",
        confidence=ExtractionConfidence.HIGH,
        tag="cli-command",
        name_template="command-{0}",
        description_template="CLI command: {0}",
    ),
]

# 默认仓库提取规则
DEFAULT_REPO_RULES: List[ExtractionRule] = [
    # README 章节
    ExtractionRule(
        pattern=r"#{1,3}\s+(.+)",
        confidence=ExtractionConfidence.HIGH,
        tag="readme-section",
        name_template="repo-section-{0}",
        description_template="Repository section: {0}",
    ),
    # Python 函数定义
    ExtractionRule(
        pattern=r"def\s+(\w+)\s*\(([^)]*)\)",
        confidence=ExtractionConfidence.HIGH,
        tag="python-function",
        name_template="func-{0}",
        description_template="Function signature: {0}({1})",
    ),
    # Python 类定义
    ExtractionRule(
        pattern=r"class\s+(\w+)(?:\(([^)]*)\))?:",
        confidence=ExtractionConfidence.HIGH,
        tag="python-class",
        name_template="class-{0}",
        description_template="Class definition: {0}",
    ),
    # 配置键值对
    ExtractionRule(
        pattern=r"^(\w+)\s*[=:]\s*(.+)",
        confidence=ExtractionConfidence.MEDIUM,
        tag="config-entry",
        name_template="config-{0}",
        description_template="Configuration: {0} = {1}",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# OpenSkillAdapter — 核心适配器
# ══════════════════════════════════════════════════════════════════════════════

class OpenSkillAdapter:
    """
    OpenSkill → UnifiedSkillConfig 适配器。

    职责：
    1. 从外部来源（文档/仓库/网页）提取 transferable skills
    2. 通过自建虚拟任务验证技能质量
    3. 生成带 SourceTraceability 的 UnifiedSkillConfig
    4. 支持去重、合并、置信度过滤

    用法：
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.MEDIUM)

        # 从文档学习
        skills = adapter.learn_from_docs(doc_content, source_url="https://...")

        # 从仓库学习
        skills = adapter.learn_from_repo(repo_path, repo_url="https://github.com/...")

        # 验证技能
        results = adapter.validate_skill(skill, virtual_tasks)

        # 生成 UnifiedSkillConfig
        config = adapter.to_unified_config(skill, results)
    """

    def __init__(
        self,
        min_confidence: ExtractionConfidence = ExtractionConfidence.MEDIUM,
        max_skills_per_source: int = 20,
        auto_generate_tasks: bool = True,
        task_count: int = 3,
        doc_rules: Optional[List[ExtractionRule]] = None,
        repo_rules: Optional[List[ExtractionRule]] = None,
    ):
        """
        Args:
            min_confidence: 最低置信度阈值（低于此值的提取被丢弃）
            max_skills_per_source: 每个来源最多提取的技能数
            auto_generate_tasks: 是否自动生成验证虚拟任务
            task_count: 每个技能自动生成的虚拟任务数
            doc_rules: 自定义文档提取规则（None 使用默认规则）
            repo_rules: 自定义仓库提取规则（None 使用默认规则）
        """
        self.min_confidence = min_confidence
        self.max_skills_per_source = max_skills_per_source
        self.auto_generate_tasks = auto_generate_tasks
        self.task_count = task_count

        # 提取规则
        self._custom_doc_rules = doc_rules
        self._custom_repo_rules = repo_rules
        self._doc_rules = list(doc_rules) if doc_rules is not None else list(DEFAULT_DOC_RULES)
        self._repo_rules = list(repo_rules) if repo_rules is not None else list(DEFAULT_REPO_RULES)

        # 状态追踪
        self._extraction_log: List[Dict[str, Any]] = []
        self._fingerprint_cache: set = set()

    @property
    def extraction_log(self) -> List[Dict[str, Any]]:
        """只读访问提取日志。"""
        return list(self._extraction_log)

    @property
    def seen_fingerprints(self) -> set:
        """已见过的指纹集合（用于去重）。"""
        return set(self._fingerprint_cache)

    # ══════════════════════════════════════════════════════════════════════════
    # learn_from_docs — 从文档提取技能
    # ══════════════════════════════════════════════════════════════════════════

    def learn_from_docs(
        self,
        content: str,
        source_url: str = "",
        author: str = "",
        license: str = "",
    ) -> List[SkillExtraction]:
        """
        从文档内容中提取技能。

        解析 Markdown/纯文本文档，应用提取规则，过滤低置信度结果，
        去重后返回技能列表。

        Args:
            content: 文档文本内容
            source_url: 文档来源 URL
            author: 文档作者
            license: 文档许可证

        Returns:
            提取的技能列表
        """
        return self._extract_from_content(
            content=content,
            source_type=SourceType.DOCS,
            source_url=source_url,
            author=author,
            license=license,
            rules=self._doc_rules,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # learn_from_repo — 从代码仓库提取技能
    # ══════════════════════════════════════════════════════════════════════════

    def learn_from_repo(
        self,
        repo_content: Dict[str, str],
        repo_url: str = "",
        author: str = "",
        license: str = "",
    ) -> List[SkillExtraction]:
        """
        从代码仓库内容中提取技能。

        接受仓库文件映射 {文件名: 内容}，应用仓库提取规则。

        Args:
            repo_content: 仓库文件映射 {filename: content}，
                          支持 README.md, *.py, *.yaml, *.json, etc.
            repo_url: 仓库 URL
            author: 仓库作者
            license: 仓库许可证

        Returns:
            提取的技能列表
        """
        all_skills: List[SkillExtraction] = []

        for filename, content in repo_content.items():
            if not content or not content.strip():
                continue

            # 根据文件类型选择规则
            rules = self._select_rules_for_file(filename)
            skills = self._extract_from_content(
                content=content,
                source_type=SourceType.REPO,
                source_url=f"{repo_url}/blob/main/{filename}" if repo_url else filename,
                author=author,
                license=license,
                rules=rules,
            )
            all_skills.extend(skills)

        # 跨文件去重（使用独立集合，避免与 _extract_from_content 内部去重冲突）
        seen: set = set()
        deduped: List[SkillExtraction] = []
        for e in all_skills:
            if e.fingerprint not in seen:
                seen.add(e.fingerprint)
                deduped.append(e)

        logger.info(
            f"learn_from_repo: extracted {len(all_skills)} raw, "
            f"{len(deduped)} after dedup from {len(repo_content)} files"
        )
        return deduped

    # ══════════════════════════════════════════════════════════════════════════
    # learn_from_webpage — 从网页提取技能
    # ══════════════════════════════════════════════════════════════════════════

    def learn_from_webpage(
        self,
        content: str,
        source_url: str = "",
        author: str = "",
        license: str = "",
    ) -> List[SkillExtraction]:
        """
        从网页内容中提取技能。

        Args:
            content: 网页文本内容
            source_url: 网页 URL
            author: 网页作者
            license: 网页许可证

        Returns:
            提取的技能列表
        """
        return self._extract_from_content(
            content=content,
            source_type=SourceType.WEBPAGE,
            source_url=source_url,
            author=author,
            license=license,
            rules=self._doc_rules,  # 网页使用文档规则
        )

    # ══════════════════════════════════════════════════════════════════════════
    # validate_skill — 通过虚拟任务验证技能质量
    # ══════════════════════════════════════════════════════════════════════════

    def validate_skill(
        self,
        skill: SkillExtraction,
        virtual_tasks: Optional[List[VirtualTask]] = None,
    ) -> List[SkillScore]:
        """
        通过虚拟任务验证技能质量。

        如果未提供虚拟任务且 auto_generate_tasks=True，会自动生成。
        验证逻辑：对每个虚拟任务，检查技能内容是否匹配预期模式。

        Args:
            skill: 待验证的技能
            virtual_tasks: 虚拟任务列表（None 则自动生成）

        Returns:
            每个虚拟任务的评分结果
        """
        if virtual_tasks is None and self.auto_generate_tasks:
            virtual_tasks = self._generate_virtual_tasks(skill)

        if not virtual_tasks:
            return []

        scores: List[SkillScore] = []
        for task in virtual_tasks:
            score = self._score_skill_against_task(skill, task)
            scores.append(score)

        # 记录验证日志
        pass_rate = sum(1 for s in scores if s.passed) / max(len(scores), 1)
        logger.info(
            f"validate_skill [{skill.name}]: "
            f"{sum(1 for s in scores if s.passed)}/{len(scores)} passed "
            f"(pass_rate={pass_rate:.2f})"
        )

        return scores

    # ══════════════════════════════════════════════════════════════════════════
    # to_unified_config — 生成 UnifiedSkillConfig
    # ══════════════════════════════════════════════════════════════════════════

    def to_unified_config(
        self,
        skill: SkillExtraction,
        validation_scores: Optional[List[SkillScore]] = None,
    ) -> "UnifiedSkillConfig":
        """
        将 SkillExtraction 转换为 UnifiedSkillConfig。

        自动填充 SourceTraceability（来源溯源）。

        Args:
            skill: 提取的技能
            validation_scores: 验证评分（如果有）

        Returns:
            UnifiedSkillConfig with SourceTraceability populated
        """
        from openllm.isn.unified_skill_config import (
            SkillFrameworkSource,
            SkillLifecycleState,
            SourceTraceability,
            UnifiedSkillConfig,
        )

        # 构建 SourceTraceability
        trace = SourceTraceability(
            upstream_repo=skill.source_url,
            upstream_version=skill.version,
            forked_at=skill.metadata.get("extracted_at"),
            license=skill.license,
            original_author=skill.author,
            attribution=self._build_attribution(skill),
            modifications=skill.metadata.get("modifications", []),
        )

        # 计算质量分
        quality_score = None
        if validation_scores:
            passed = [s for s in validation_scores if s.passed]
            if validation_scores:
                quality_score = round(
                    sum(s.score for s in passed) / max(len(validation_scores), 1),
                    4,
                )

        return UnifiedSkillConfig(
            name=skill.name,
            description=skill.description,
            version=skill.version,
            author=skill.author,
            lifecycle_state=SkillLifecycleState.ACTIVE,
            source_framework=SkillFrameworkSource.OPENSKILL,
            domain_tags=skill.tags,
            source_traceability=trace,
            curator_score=quality_score,
            risk_level=self._assess_risk_level(skill, validation_scores),
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Internal: Extraction Pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_from_content(
        self,
        content: str,
        source_type: SourceType,
        source_url: str,
        author: str,
        license: str,
        rules: List[ExtractionRule],
    ) -> List[SkillExtraction]:
        """通用内容提取管线。"""
        raw_extractions: List[SkillExtraction] = []

        for rule in rules:
            flags = re.MULTILINE | (re.DOTALL if rule.use_dotall else 0)
            matches = re.findall(rule.pattern, content, flags)

            for match in matches:
                # 匹配可能是 tuple（多组捕获）或 string（单组捕获）
                if isinstance(match, tuple):
                    match_str = "_".join(str(m) for m in match if m)
                    format_args = match
                else:
                    match_str = match
                    format_args = (match,)

                # 生成名称和描述
                try:
                    name = rule.name_template.format(*format_args) if rule.name_template else match_str[:50]
                except (IndexError, KeyError):
                    name = match_str[:50]

                try:
                    desc = rule.description_template.format(*format_args) if rule.description_template else match_str[:200]
                except (IndexError, KeyError):
                    desc = match_str[:200]

                extraction = SkillExtraction(
                    name=name,
                    description=desc,
                    source_type=source_type,
                    source_url=source_url,
                    content=match_str[:2000],  # 截断过长内容
                    confidence=rule.confidence,
                    tags=[rule.tag],
                    author=author,
                    license=license,
                    metadata={
                        "rule_pattern": rule.pattern,
                        "raw_match": match_str[:500],
                        "extracted_at": self._now_iso(),
                    },
                )
                raw_extractions.append(extraction)

        # 置信度过滤
        filtered = self._filter_by_confidence(raw_extractions)

        # 去重
        deduped = self._deduplicate(filtered)

        # 数量限制
        result = deduped[:self.max_skills_per_source]

        # 记录日志
        self._extraction_log.append({
            "source_type": source_type.value,
            "source_url": source_url,
            "raw_count": len(raw_extractions),
            "filtered_count": len(filtered),
            "deduped_count": len(deduped),
            "final_count": len(result),
        })

        return result

    def _filter_by_confidence(
        self, extractions: List[SkillExtraction]
    ) -> List[SkillExtraction]:
        """按置信度过滤。"""
        confidence_order = {
            ExtractionConfidence.LOW: 0,
            ExtractionConfidence.MEDIUM: 1,
            ExtractionConfidence.HIGH: 2,
        }
        min_level = confidence_order.get(self.min_confidence, 0)
        return [
            e for e in extractions
            if confidence_order.get(e.confidence, 0) >= min_level
        ]

    def _deduplicate(
        self, extractions: List[SkillExtraction]
    ) -> List[SkillExtraction]:
        """基于指纹去重。"""
        result: List[SkillExtraction] = []
        for e in extractions:
            fp = e.fingerprint
            if fp not in self._fingerprint_cache:
                self._fingerprint_cache.add(fp)
                result.append(e)
        return result

    def _select_rules_for_file(self, filename: str) -> List[ExtractionRule]:
        """根据文件名选择提取规则。

        如果用户自定义了 repo_rules，直接返回所有自定义规则（不按文件类型过滤）。
        """
        if self._custom_repo_rules is not None:
            return list(self._custom_repo_rules)

        name_lower = filename.lower()

        if name_lower.endswith((".py", ".pyx")):
            # Python 文件：优先函数/类定义规则
            return [r for r in self._repo_rules if "python" in r.tag]
        elif name_lower.endswith((".yaml", ".yml", ".json", ".toml")):
            # 配置文件
            return [r for r in self._repo_rules if "config" in r.tag]
        elif name_lower in ("readme.md", "readme.rst", "readme.txt"):
            # README：章节标题规则
            return [r for r in self._repo_rules if "readme" in r.tag]
        else:
            # 默认使用全部规则
            return self._repo_rules

    # ══════════════════════════════════════════════════════════════════════════
    # Internal: Virtual Task Generation & Scoring
    # ══════════════════════════════════════════════════════════════════════════

    def _generate_virtual_tasks(
        self, skill: SkillExtraction
    ) -> List[VirtualTask]:
        """
        为技能自动生成虚拟验证任务。

        生成策略：
        1. 内容存在性检查（技能内容非空）
        2. 名称/描述一致性检查
        3. 来源溯源完整性检查
        """
        tasks: List[VirtualTask] = []

        # Task 1: 内容非空
        tasks.append(VirtualTask(
            task_id=f"{skill.fingerprint}-content-check",
            description="Skill content must be non-empty and substantive",
            input_data={"min_length": 10},
            difficulty="easy",
        ))

        # Task 2: 名称和描述质量
        tasks.append(VirtualTask(
            task_id=f"{skill.fingerprint}-desc-check",
            description="Skill name and description must be meaningful",
            input_data={"min_name_length": 2, "min_desc_length": 10},
            difficulty="easy",
        ))

        # Task 3: 来源可追溯
        tasks.append(VirtualTask(
            task_id=f"{skill.fingerprint}-trace-check",
            description="Skill must have traceable source URL or author",
            input_data={},
            difficulty="medium",
        ))

        return tasks[:self.task_count]

    def _score_skill_against_task(
        self, skill: SkillExtraction, task: VirtualTask
    ) -> SkillScore:
        """评分单个技能在单个虚拟任务上的表现。"""
        passed = False
        score = 0.0
        error = ""
        output = ""

        if "content-check" in task.task_id:
            min_len = task.input_data.get("min_length", 10)
            if len(skill.content) >= min_len:
                passed = True
                score = min(1.0, len(skill.content) / (min_len * 5))
                output = f"Content length: {len(skill.content)} >= {min_len}"
            else:
                error = f"Content too short: {len(skill.content)} < {min_len}"

        elif "desc-check" in task.task_id:
            min_name = task.input_data.get("min_name_length", 2)
            min_desc = task.input_data.get("min_desc_length", 10)
            name_ok = len(skill.name) >= min_name
            desc_ok = len(skill.description) >= min_desc
            if name_ok and desc_ok:
                passed = True
                score = 1.0
                output = f"Name({len(skill.name)}), Desc({len(skill.description)})"
            else:
                parts = []
                if not name_ok:
                    parts.append(f"name too short({len(skill.name)})")
                if not desc_ok:
                    parts.append(f"desc too short({len(skill.description)})")
                error = "; ".join(parts)

        elif "trace-check" in task.task_id:
            has_trace = bool(skill.source_url or skill.author)
            if has_trace:
                passed = True
                score = 1.0
                output = f"Trace: url={bool(skill.source_url)}, author={bool(skill.author)}"
            else:
                error = "No traceable source URL or author"

        else:
            error = f"Unknown task pattern: {task.task_id}"

        return SkillScore(
            task_id=task.task_id,
            passed=passed,
            score=score,
            output=output,
            error=error,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Internal: Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _build_attribution(self, skill: SkillExtraction) -> str:
        """构建归属声明。"""
        parts = []
        if skill.author:
            parts.append(f"Original author: {skill.author}")
        if skill.source_url:
            parts.append(f"Source: {skill.source_url}")
        if skill.license:
            parts.append(f"License: {skill.license}")
        return " | ".join(parts) if parts else "Extracted via OpenSkill"

    def _assess_risk_level(
        self,
        skill: SkillExtraction,
        validation_scores: Optional[List[SkillScore]] = None,
    ) -> str:
        """根据来源和验证结果评估风险级别。"""
        risk = "medium"  # 默认

        # 高置信度 + 验证通过 → low risk
        if skill.confidence == ExtractionConfidence.HIGH:
            risk = "low"

        # 低置信度 → high risk
        if skill.confidence == ExtractionConfidence.LOW:
            risk = "high"

        # 验证结果调整
        if validation_scores:
            pass_rate = sum(1 for s in validation_scores if s.passed) / max(len(validation_scores), 1)
            if pass_rate < 0.5:
                risk = "high"
            elif pass_rate >= 0.8:
                risk = "low" if risk != "critical" else risk

        return risk

    @staticmethod
    def _now_iso() -> str:
        """返回当前 ISO 时间戳。"""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def reset(self):
        """重置 adapter 状态。"""
        self._extraction_log.clear()
        self._fingerprint_cache.clear()
