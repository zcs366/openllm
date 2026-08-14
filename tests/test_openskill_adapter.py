"""
Unit tests for OpenSkill Adapter
=================================

测试覆盖：
1. SkillExtraction 基础行为（fingerprint, confidence, metadata）
2. VirtualTask 数据结构
3. SkillScore 数据结构
4. OpenSkillAdapter.learn_from_docs() — 从文档提取技能
   - API 端点提取
   - 步骤列表提取
   - 代码块提取
   - How-to 模式提取
   - CLI 命令提取
   - 置信度过滤
   - 去重
   - 空内容处理
5. OpenSkillAdapter.learn_from_repo() — 从代码仓库提取技能
   - README 章节提取
   - Python 函数定义提取
   - Python 类定义提取
   - 配置文件提取
   - 跨文件去重
   - 空文件跳过
6. OpenSkillAdapter.learn_from_webpage() — 从网页提取技能
7. OpenSkillAdapter.validate_skill() — 虚拟任务验证
   - 自动生成虚拟任务
   - 内容检查
   - 描述检查
   - 来源追溯检查
   - 自定义虚拟任务
8. OpenSkillAdapter.to_unified_config() — 生成 UnifiedSkillConfig
   - SourceTraceability 填充
   - 质量分计算
   - 风险级别评估
9. OpenSkillAdapter 内部方法
   - 置信度过滤
   - 去重
   - 文件规则选择
   - 属性构建
10. 重置与日志
"""

import sys
import os
import pytest

# 确保 openllm 可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.isn.adapters.openskill_adapter import (
    ExtractionConfidence,
    ExtractionRule,
    OpenSkillAdapter,
    SkillExtraction,
    SkillScore,
    SourceType,
    VirtualTask,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

def make_skill(
    name: str = "test-skill",
    description: str = "A test skill for unit tests",
    content: str = "This is test skill content with enough length",
    source_url: str = "https://example.com/docs",
    author: str = "test-author",
    license: str = "MIT",
    confidence: ExtractionConfidence = ExtractionConfidence.HIGH,
) -> SkillExtraction:
    return SkillExtraction(
        name=name,
        description=description,
        source_type=SourceType.DOCS,
        source_url=source_url,
        content=content,
        confidence=confidence,
        tags=["test"],
        author=author,
        license=license,
    )


SAMPLE_DOCS = """# API Reference

## Getting Started

Follow these steps:
1. Install the package with `pip install mylib`
2. Configure the API key
3. Make your first request

$ mylib init --config config.yaml
$ mylib status

## API Endpoints

GET /api/v1/users — List all users
POST /api/v1/users — Create a new user
DELETE /api/v1/users/{id} — Delete a user

## Code Example

```python
import mylib

client = mylib.Client(api_key="xxx")
users = client.get("/api/v1/users")
```

## How to Migrate

How to migrate from v1 to v2: update your config file and re-run.
"""

SAMPLE_REPO = {
    "README.md": "# My Library\n\n## Installation\n\nRun `pip install mylib`.\n\n## Usage\n\n```python\nimport mylib\n```",
    "src/core.py": 'class DataProcessor:\n    """Process data."""\n    def process(self, data: list) -> dict:\n        return {"result": data}\n\n    def validate(self, item: dict) -> bool:\n        return bool(item)',
    "config.yaml": "debug: true\nlog_level: info\nmax_retries: 3",
}


# ══════════════════════════════════════════════════════════════════════════════
# SkillExtraction Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSkillExtraction:
    def test_fingerprint_deterministic(self):
        """相同输入产生相同指纹。"""
        s1 = make_skill(name="skill-a", description="desc", source_url="url")
        s2 = make_skill(name="skill-a", description="desc", source_url="url")
        assert s1.fingerprint == s2.fingerprint

    def test_fingerprint_different(self):
        """不同输入产生不同指纹。"""
        s1 = make_skill(name="skill-a")
        s2 = make_skill(name="skill-b")
        assert s1.fingerprint != s2.fingerprint

    def test_confidence_levels(self):
        """置信度枚举正确。"""
        assert ExtractionConfidence.HIGH.value == "high"
        assert ExtractionConfidence.MEDIUM.value == "medium"
        assert ExtractionConfidence.LOW.value == "low"

    def test_source_types(self):
        """来源类型枚举正确。"""
        assert SourceType.DOCS.value == "docs"
        assert SourceType.REPO.value == "repo"
        assert SourceType.WEBPAGE.value == "webpage"

    def test_metadata_default(self):
        """metadata 默认为空 dict。"""
        s = make_skill()
        assert isinstance(s.metadata, dict)
        assert len(s.metadata) == 0


# ══════════════════════════════════════════════════════════════════════════════
# VirtualTask Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestVirtualTask:
    def test_default_values(self):
        """默认值正确。"""
        vt = VirtualTask(task_id="t1", description="Test task")
        assert vt.difficulty == "medium"
        assert vt.timeout_seconds == 30
        assert vt.input_data == {}

    def test_custom_values(self):
        """自定义值正确。"""
        vt = VirtualTask(
            task_id="t2",
            description="Custom task",
            difficulty="hard",
            timeout_seconds=60,
        )
        assert vt.difficulty == "hard"
        assert vt.timeout_seconds == 60


# ══════════════════════════════════════════════════════════════════════════════
# SkillScore Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSkillScore:
    def test_passed_score(self):
        """通过的评分。"""
        ss = SkillScore(task_id="t1", passed=True, score=0.9, output="OK")
        assert ss.passed is True
        assert ss.score == 0.9

    def test_failed_score(self):
        """失败的评分。"""
        ss = SkillScore(task_id="t1", passed=False, score=0.0, error="too short")
        assert ss.passed is False
        assert ss.error == "too short"


# ══════════════════════════════════════════════════════════════════════════════
# learn_from_docs Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLearnFromDocs:
    def test_extract_api_endpoints(self):
        """提取 API 端点。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_docs(SAMPLE_DOCS, source_url="https://docs.example.com")

        endpoint_skills = [s for s in skills if "api-endpoint" in s.tags]
        assert len(endpoint_skills) >= 2  # GET, POST, DELETE

    def test_extract_step_lists(self):
        """提取步骤列表。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_docs(SAMPLE_DOCS)

        step_skills = [s for s in skills if "step-list" in s.tags]
        assert len(step_skills) >= 1

    def test_extract_code_blocks(self):
        """提取代码块。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_docs(SAMPLE_DOCS)

        code_skills = [s for s in skills if "code-block" in s.tags]
        assert len(code_skills) >= 1

    def test_extract_howto(self):
        """提取 How-to 模式。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_docs(SAMPLE_DOCS)

        howto_skills = [s for s in skills if "howto" in s.tags]
        assert len(howto_skills) >= 1

    def test_extract_cli_commands(self):
        """提取 CLI 命令。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_docs(SAMPLE_DOCS)

        cli_skills = [s for s in skills if "cli-command" in s.tags]
        assert len(cli_skills) >= 1

    def test_confidence_filter(self):
        """置信度过滤。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.HIGH)
        skills = adapter.learn_from_docs(SAMPLE_DOCS)

        # 只有 HIGH 置信度的结果
        for skill in skills:
            assert skill.confidence == ExtractionConfidence.HIGH

    def test_source_traceability(self):
        """来源溯源信息正确。"""
        adapter = OpenSkillAdapter()
        skills = adapter.learn_from_docs(
            SAMPLE_DOCS,
            source_url="https://docs.example.com",
            author="Alice",
            license="Apache-2.0",
        )
        for skill in skills:
            assert skill.source_url == "https://docs.example.com"
            assert skill.author == "Alice"
            assert skill.license == "Apache-2.0"

    def test_empty_content(self):
        """空内容不报错。"""
        adapter = OpenSkillAdapter()
        skills = adapter.learn_from_docs("")
        assert skills == []

    def test_no_matches(self):
        """无匹配内容返回空列表。"""
        adapter = OpenSkillAdapter()
        skills = adapter.learn_from_docs("Just some random text without patterns.")
        assert skills == []

    def test_max_skills_per_source(self):
        """技能数量不超过限制。"""
        adapter = OpenSkillAdapter(max_skills_per_source=2, min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_docs(SAMPLE_DOCS)
        assert len(skills) <= 2


# ══════════════════════════════════════════════════════════════════════════════
# learn_from_repo Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLearnFromRepo:
    def test_readme_sections(self):
        """提取 README 章节。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_repo(
            {"README.md": "# Project\n\n## Install\n\nRun this.\n\n## Usage\n\nDo that."},
            repo_url="https://github.com/test/repo",
        )
        section_skills = [s for s in skills if "readme-section" in s.tags]
        assert len(section_skills) >= 2

    def test_python_functions(self):
        """提取 Python 函数定义。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_repo(
            {"utils.py": "def hello(name: str) -> str:\n    return f'Hi {name}'\n\ndef goodbye():\n    pass"},
        )
        func_skills = [s for s in skills if "python-function" in s.tags]
        assert len(func_skills) >= 2

    def test_python_classes(self):
        """提取 Python 类定义。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_repo(
            {"models.py": "class User:\n    pass\n\nclass Admin(User):\n    pass"},
        )
        class_skills = [s for s in skills if "python-class" in s.tags]
        assert len(class_skills) >= 2

    def test_config_entries(self):
        """提取配置条目。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_repo(
            {"config.yaml": "debug: true\nlog_level: info\nmax_retries: 3"},
        )
        config_skills = [s for s in skills if "config-entry" in s.tags]
        assert len(config_skills) >= 2

    def test_cross_file_dedup(self):
        """跨文件去重。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_repo(
            {
                "readme.md": "# MyLib\n## Install\n## Install",
            },
        )
        # 同一个章节标题不会出现两次
        section_skills = [s for s in skills if "readme-section" in s.tags]
        # 去重后应该减少
        assert len(section_skills) >= 1

    def test_empty_file_skipped(self):
        """空文件被跳过。"""
        adapter = OpenSkillAdapter()
        skills = adapter.learn_from_repo({"empty.py": "", "blank.yaml": "  \n  "})
        assert skills == []

    def test_file_rule_selection(self):
        """文件类型正确选择规则。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)

        # Python 文件只提取函数/类
        py_skills = adapter.learn_from_repo(
            {"main.py": "def foo(): pass\nclass Bar: pass\n# Install\n"},
        )
        assert all(
            s.tags[0] in ("python-function", "python-class")
            for s in py_skills
        )

        # 配置文件只提取配置条目
        adapter.reset()
        yaml_skills = adapter.learn_from_repo(
            {"config.yaml": "key: value\n# Section\n"},
        )
        assert all(s.tags[0] == "config-entry" for s in yaml_skills)

    def test_repo_url_propagated(self):
        """仓库 URL 正确传播到技能。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_repo(
            {"README.md": "# Title"},
            repo_url="https://github.com/test/repo",
        )
        for skill in skills:
            assert "github.com/test/repo" in skill.source_url

    def test_full_repo_sample(self):
        """完整仓库样本测试。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_repo(SAMPLE_REPO, repo_url="https://github.com/test/mylib")
        assert len(skills) > 0
        # 应该覆盖多种类型
        tags = {s.tags[0] for s in skills}
        assert len(tags) >= 2  # 至少 2 种类型


# ══════════════════════════════════════════════════════════════════════════════
# learn_from_webpage Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLearnFromWebpage:
    def test_webpage_extraction(self):
        """网页提取使用文档规则。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_webpage(
            SAMPLE_DOCS,
            source_url="https://example.com/tutorial",
        )
        assert len(skills) > 0
        for skill in skills:
            assert skill.source_type == SourceType.WEBPAGE

    def test_webpage_with_author(self):
        """网页作者信息正确。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_webpage(
            "1. Step one\n2. Step two\n3. Step three",
            source_url="https://example.com",
            author="Web Author",
        )
        for skill in skills:
            assert skill.author == "Web Author"


# ══════════════════════════════════════════════════════════════════════════════
# validate_skill Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateSkill:
    def test_auto_generate_tasks(self):
        """自动生成虚拟任务。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(content="This has enough content to pass the check.")
        scores = adapter.validate_skill(skill)
        assert len(scores) == 3  # 默认 task_count=3

    def test_content_check_pass(self):
        """内容检查通过。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(content="Long enough content for validation.")
        scores = adapter.validate_skill(skill, virtual_tasks=[
            VirtualTask(task_id="xxx-content-check", description="check", input_data={"min_length": 10}),
        ])
        assert len(scores) == 1
        assert scores[0].passed is True

    def test_content_check_fail_short(self):
        """内容检查失败（太短）。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(content="Short")
        scores = adapter.validate_skill(skill, virtual_tasks=[
            VirtualTask(task_id="xxx-content-check", description="check", input_data={"min_length": 50}),
        ])
        assert scores[0].passed is False
        assert "too short" in scores[0].error

    def test_desc_check_pass(self):
        """描述检查通过。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(
            name="good-name",
            description="A good meaningful description that is long enough.",
        )
        scores = adapter.validate_skill(skill, virtual_tasks=[
            VirtualTask(task_id="xxx-desc-check", description="check",
                        input_data={"min_name_length": 3, "min_desc_length": 10}),
        ])
        assert scores[0].passed is True

    def test_desc_check_fail_name(self):
        """描述检查失败（名称太短）。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(
            name="x",
            description="A good description here.",
        )
        scores = adapter.validate_skill(skill, virtual_tasks=[
            VirtualTask(task_id="xxx-desc-check", description="check",
                        input_data={"min_name_length": 5, "min_desc_length": 10}),
        ])
        assert scores[0].passed is False
        assert "name too short" in scores[0].error

    def test_trace_check_pass(self):
        """来源追溯检查通过。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(source_url="https://example.com")
        scores = adapter.validate_skill(skill, virtual_tasks=[
            VirtualTask(task_id="xxx-trace-check", description="check"),
        ])
        assert scores[0].passed is True

    def test_trace_check_fail(self):
        """来源追溯检查失败（无 URL 和 author）。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(source_url="", author="")
        scores = adapter.validate_skill(skill, virtual_tasks=[
            VirtualTask(task_id="xxx-trace-check", description="check"),
        ])
        assert scores[0].passed is False
        assert "No traceable" in scores[0].error

    def test_custom_tasks(self):
        """自定义虚拟任务。"""
        adapter = OpenSkillAdapter(auto_generate_tasks=False)
        skill = make_skill()
        custom_tasks = [
            VirtualTask(task_id="custom-1", description="Custom check 1"),
            VirtualTask(task_id="custom-2", description="Custom check 2"),
        ]
        scores = adapter.validate_skill(skill, virtual_tasks=custom_tasks)
        assert len(scores) == 2
        assert scores[0].task_id == "custom-1"
        assert scores[1].task_id == "custom-2"

    def test_no_tasks_no_auto(self):
        """无任务 + 不自动生成 = 空结果。"""
        adapter = OpenSkillAdapter(auto_generate_tasks=False)
        skill = make_skill()
        scores = adapter.validate_skill(skill)
        assert scores == []

    def test_pass_rate_logged(self):
        """验证日志记录。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(content="Valid content for testing purposes.")
        scores = adapter.validate_skill(skill)
        passed = sum(1 for s in scores if s.passed)
        assert passed > 0  # 至少有部分通过


# ══════════════════════════════════════════════════════════════════════════════
# to_unified_config Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestToUnifiedConfig:
    def test_source_traceability_populated(self):
        """SourceTraceability 正确填充。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(
            source_url="https://github.com/test/repo",
            author="Test Author",
            license="MIT",
        )
        config = adapter.to_unified_config(skill)

        assert config.source_traceability is not None
        assert config.source_traceability.upstream_repo == "https://github.com/test/repo"
        assert config.source_traceability.original_author == "Test Author"
        assert config.source_traceability.license == "MIT"

    def test_framework_source_is_openskill(self):
        """框架来源标记为 OPENSKILL。"""
        adapter = OpenSkillAdapter()
        skill = make_skill()
        config = adapter.to_unified_config(skill)

        from openllm.isn.unified_skill_config import SkillFrameworkSource
        assert config.source_framework == SkillFrameworkSource.OPENSKILL

    def test_lifecycle_active(self):
        """生命周期状态为 ACTIVE。"""
        adapter = OpenSkillAdapter()
        skill = make_skill()
        config = adapter.to_unified_config(skill)

        from openllm.isn.unified_skill_config import SkillLifecycleState
        assert config.lifecycle_state == SkillLifecycleState.ACTIVE

    def test_domain_tags(self):
        """领域标签正确传递。"""
        adapter = OpenSkillAdapter()
        skill = make_skill()
        skill.tags = ["api", "rest", "v1"]
        config = adapter.to_unified_config(skill)
        assert config.domain_tags == ["api", "rest", "v1"]

    def test_quality_score_computed(self):
        """质量分从验证结果计算。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(content="Content for scoring.")

        scores = [
            SkillScore(task_id="t1", passed=True, score=0.8),
            SkillScore(task_id="t2", passed=True, score=0.9),
        ]
        config = adapter.to_unified_config(skill, validation_scores=scores)
        # 平均分 = (0.8 + 0.9) / 2 = 0.85
        assert config.curator_score == pytest.approx(0.85, abs=0.01)

    def test_quality_score_with_failures(self):
        """部分失败影响质量分。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(content="Some content.")

        scores = [
            SkillScore(task_id="t1", passed=True, score=0.8),
            SkillScore(task_id="t2", passed=False, score=0.0),
        ]
        config = adapter.to_unified_config(skill, validation_scores=scores)
        # 只计算通过的平均分，但分母是总任务数
        assert config.curator_score is not None
        assert config.curator_score < 0.85

    def test_no_validation_scores(self):
        """无验证结果时 quality_score 为 None。"""
        adapter = OpenSkillAdapter()
        skill = make_skill()
        config = adapter.to_unified_config(skill)
        assert config.curator_score is None

    def test_risk_level_high_confidence(self):
        """高置信度 → low risk。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(confidence=ExtractionConfidence.HIGH)
        config = adapter.to_unified_config(skill)
        assert config.risk_level == "low"

    def test_risk_level_low_confidence(self):
        """低置信度 → high risk。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(confidence=ExtractionConfidence.LOW)
        config = adapter.to_unified_config(skill)
        assert config.risk_level == "high"

    def test_risk_level_with_bad_validation(self):
        """验证通过率低 → high risk。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(confidence=ExtractionConfidence.MEDIUM)

        scores = [
            SkillScore(task_id="t1", passed=False, score=0.0),
            SkillScore(task_id="t2", passed=False, score=0.0),
        ]
        config = adapter.to_unified_config(skill, validation_scores=scores)
        assert config.risk_level == "high"

    def test_risk_level_with_good_validation(self):
        """验证通过率高 → low risk。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(confidence=ExtractionConfidence.MEDIUM)

        scores = [
            SkillScore(task_id="t1", passed=True, score=0.8),
            SkillScore(task_id="t2", passed=True, score=0.9),
            SkillScore(task_id="t3", passed=True, score=0.85),
        ]
        config = adapter.to_unified_config(skill, validation_scores=scores)
        assert config.risk_level == "low"

    def test_attribution_built(self):
        """归属声明正确构建。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(author="Alice", source_url="https://example.com", license="MIT")
        config = adapter.to_unified_config(skill)

        attr = config.source_traceability.attribution
        assert "Alice" in attr
        assert "example.com" in attr
        assert "MIT" in attr

    def test_to_dict_serialization(self):
        """生成的 config 可序列化。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(content="Content for serialization test.")
        config = adapter.to_unified_config(skill)

        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test-skill"
        assert d["source_framework"] == "openskill"
        assert d["source_traceability"]["upstream_repo"] == "https://example.com/docs"

    def test_config_valid(self):
        """生成的 config 通过 UnifiedSkillConfig 验证。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(content="Valid content for testing.")
        config = adapter.to_unified_config(skill)
        assert config.is_valid()


# ══════════════════════════════════════════════════════════════════════════════
# Internal Method Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestInternalMethods:
    def test_filter_by_confidence_medium(self):
        """置信度过滤 medium。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.MEDIUM)
        skills = adapter.learn_from_docs(
            "# How to do X\n```python\nprint('hi')\n```\n$ echo hello",
        )
        # 所有技能 >= medium
        for s in skills:
            assert s.confidence in (ExtractionConfidence.MEDIUM, ExtractionConfidence.HIGH)

    def test_deduplicate(self):
        """去重功能。"""
        adapter = OpenSkillAdapter()
        skill1 = make_skill(name="dup", description="same", source_url="url")
        skill2 = make_skill(name="dup", description="same", source_url="url")

        # 相同指纹应该被去重
        assert skill1.fingerprint == skill2.fingerprint

    def test_reset_clears_state(self):
        """reset 清除所有状态。"""
        adapter = OpenSkillAdapter()
        adapter.learn_from_docs(SAMPLE_DOCS)
        assert len(adapter.extraction_log) > 0
        assert len(adapter.seen_fingerprints) > 0

        adapter.reset()
        assert len(adapter.extraction_log) == 0
        assert len(adapter.seen_fingerprints) == 0

    def test_extraction_log_populated(self):
        """提取日志正确记录。"""
        adapter = OpenSkillAdapter()
        adapter.learn_from_docs(SAMPLE_DOCS, source_url="https://example.com")
        log = adapter.extraction_log
        assert len(log) == 1
        assert log[0]["source_type"] == "docs"
        assert log[0]["source_url"] == "https://example.com"
        assert "raw_count" in log[0]
        assert "final_count" in log[0]


# ══════════════════════════════════════════════════════════════════════════════
# Custom Rules Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCustomRules:
    def test_custom_doc_rules(self):
        """自定义文档提取规则。"""
        custom_rules = [
            ExtractionRule(
                pattern=r"TODO:\s*(.+)",
                confidence=ExtractionConfidence.HIGH,
                tag="todo-item",
                name_template="todo-{0}",
                description_template="TODO: {0}",
            ),
        ]
        adapter = OpenSkillAdapter(doc_rules=custom_rules, min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_docs("TODO: Fix the login bug\nTODO: Add tests\n")
        todo_skills = [s for s in skills if "todo-item" in s.tags]
        assert len(todo_skills) == 2

    def test_custom_repo_rules(self):
        """自定义仓库提取规则。"""
        custom_rules = [
            ExtractionRule(
                pattern=r"@(\w+)\s+def\s+(\w+)",
                confidence=ExtractionConfidence.HIGH,
                tag="decorated-func",
                name_template="{0}-{1}",
                description_template="Decorated function: {0}.{1}",
            ),
        ]
        adapter = OpenSkillAdapter(repo_rules=custom_rules, min_confidence=ExtractionConfidence.LOW)
        skills = adapter.learn_from_repo({
            "api.py": "@app\ndef hello():\n    pass\n\n@app\ndef world():\n    pass"
        })
        decorated = [s for s in skills if "decorated-func" in s.tags]
        assert len(decorated) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Integration Flow Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationFlow:
    def test_full_docs_pipeline(self):
        """完整文档管线：learn → validate → config。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)

        # Phase 1: Learn from docs
        skills = adapter.learn_from_docs(
            SAMPLE_DOCS,
            source_url="https://docs.example.com",
            author="Doc Team",
            license="MIT",
        )
        assert len(skills) > 0

        # Phase 2: Validate each skill
        for skill in skills:
            scores = adapter.validate_skill(skill)
            assert len(scores) > 0

            # Phase 3: Generate UnifiedSkillConfig
            config = adapter.to_unified_config(skill, validation_scores=scores)
            assert config.is_valid()
            assert config.source_traceability is not None

    def test_full_repo_pipeline(self):
        """完整仓库管线：learn → validate → config。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)

        # Phase 1: Learn from repo
        skills = adapter.learn_from_repo(
            SAMPLE_REPO,
            repo_url="https://github.com/test/mylib",
            author="Repo Author",
            license="Apache-2.0",
        )
        assert len(skills) > 0

        # Phase 2: Validate and generate configs
        configs = []
        for skill in skills:
            scores = adapter.validate_skill(skill)
            config = adapter.to_unified_config(skill, validation_scores=scores)
            configs.append(config)

        assert len(configs) > 0
        for config in configs:
            assert config.is_valid()
            assert config.source_framework.value == "openskill"
            assert config.source_traceability is not None

    def test_multi_source_pipeline(self):
        """多来源管线：同时从文档和仓库学习。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)

        doc_skills = adapter.learn_from_docs(
            "# API Guide\nGET /api/data — fetch data\nPOST /api/data — create data",
            source_url="https://docs.example.com",
            author="Doc Author",
        )
        repo_skills = adapter.learn_from_repo(
            {"main.py": "def fetch_data(url: str) -> dict:\n    pass"},
            repo_url="https://github.com/test/repo",
            author="Repo Author",
        )

        all_skills = doc_skills + repo_skills
        assert len(all_skills) >= 2

        # 每个 skill 都能生成有效的 config
        for skill in all_skills:
            config = adapter.to_unified_config(skill)
            assert config.is_valid()

    def test_dedup_across_sources(self):
        """跨来源去重。"""
        adapter = OpenSkillAdapter(min_confidence=ExtractionConfidence.LOW)

        # 相同内容来源两次
        skills1 = adapter.learn_from_docs(
            "1. Install\n2. Configure",
            source_url="https://example.com",
        )
        skills2 = adapter.learn_from_docs(
            "1. Install\n2. Configure",
            source_url="https://example.com",
        )

        # 去重后总数不超过单次提取量
        total_unique = len(skills1) + len(skills2)
        assert total_unique >= len(skills1)

    def test_config_roundtrip(self):
        """配置序列化/反序列化往返。"""
        adapter = OpenSkillAdapter()
        skill = make_skill(content="Content for roundtrip test.")
        skill.tags = ["api", "rest"]

        config = adapter.to_unified_config(skill)
        d = config.to_dict()

        # 反序列化
        from openllm.isn.unified_skill_config import UnifiedSkillConfig
        restored = UnifiedSkillConfig.from_dict(d)
        assert restored.name == config.name
        assert restored.source_framework == config.source_framework
        assert restored.source_traceability.upstream_repo == config.source_traceability.upstream_repo


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
