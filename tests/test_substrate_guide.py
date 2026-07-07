"""SubstrateEnforcer 测试 — PAL P1-2
2026-07-07
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from openllm.governance.substrate import (
    SubstrateEnforcer,
    CompiledConstraint,
    SubstrateGuide,
)


class TestSubstrateEnforcer(unittest.TestCase):
    """SubstrateEnforcer 核心逻辑测试。"""

    def setUp(self):
        # 创建临时schema文件
        self.tmpdir = tempfile.mkdtemp()
        self.schema_path = Path(self.tmpdir) / "test_schema.json"

        # 测试用约束
        self.test_constraints = [
            {
                "id": "c1",
                "category": "safety",
                "description": "禁止使用eval()函数",
                "check_method": "auto",
                "severity": "block",
            },
            {
                "id": "c2",
                "category": "quality",
                "description": "所有公共函数必须有docstring",
                "check_method": "manual",
                "severity": "warn",
            },
            {
                "id": "c3",
                "category": "format",
                "description": "文件路径必须使用绝对路径",
                "check_method": "auto",
                "severity": "info",
            },
        ]

        with open(self.schema_path, "w") as f:
            json.dump(self.test_constraints, f)

        self.enforcer = SubstrateEnforcer(schema_path=self.schema_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_schema(self):
        """schema正确加载。"""
        raw = self.enforcer.load_schema()
        self.assertEqual(len(raw), 3)

    def test_compile_generates_content(self):
        """compile生成引导文本。"""
        content = self.enforcer.compile()
        self.assertIn("ISN Substrate约束", content)
        self.assertIn("eval", content)
        self.assertIn("docstring", content)

    def test_compile_groups_by_severity(self):
        """约束按severity分组。"""
        self.enforcer.compile()
        self.assertEqual(len(self.enforcer._constraints), 3)

        block = [c for c in self.enforcer._constraints if c.severity == "block"]
        warn = [c for c in self.enforcer._constraints if c.severity == "warn"]
        info = [c for c in self.enforcer._constraints if c.severity == "info"]

        self.assertEqual(len(block), 1)
        self.assertEqual(len(warn), 1)
        self.assertEqual(len(info), 1)

    def test_guide_returns_substrate_guide(self):
        """guide返回SubstrateGuide。"""
        guide = self.enforcer.guide(task_type="code")
        self.assertIsInstance(guide, SubstrateGuide)
        self.assertEqual(guide.constraints_count, 3)
        self.assertEqual(guide.block_count, 1)
        self.assertEqual(guide.warn_count, 1)
        self.assertEqual(guide.info_count, 1)

    def test_guide_includes_task_hints(self):
        """guide包含任务类型特定提示。"""
        guide = self.enforcer.guide(task_type="code")
        self.assertIn("代码生成注意", guide.content)

        guide_doc = self.enforcer.guide(task_type="document")
        self.assertIn("文档生成注意", guide_doc.content)

    def test_check_detects_violations(self):
        """check检测代码违规。"""
        self.enforcer.compile()
        bad_code = "result = eval(user_input)"
        violations = self.enforcer.check(bad_code)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["constraint_id"], "c1")
        self.assertEqual(violations[0]["severity"], "block")

    def test_check_clean_code(self):
        """check对干净代码无违规。"""
        self.enforcer.compile()
        clean_code = "result = safe_function(input)"
        violations = self.enforcer.check(clean_code)
        self.assertEqual(len(violations), 0)

    def test_is_relevant_safety_always(self):
        """safety约束总是相关。"""
        c = CompiledConstraint(
            id="c1", category="safety", severity="block",
            description="test", check_method="auto", in_context_hint="test"
        )
        self.assertTrue(self.enforcer._is_relevant(c, "code"))
        self.assertTrue(self.enforcer._is_relevant(c, "document"))
        self.assertTrue(self.enforcer._is_relevant(c, "analysis"))

    def test_empty_schema(self):
        """空schema不报错。"""
        with open(self.schema_path, "w") as f:
            json.dump([], f)
        enforcer = SubstrateEnforcer(schema_path=self.schema_path)
        content = enforcer.compile()
        self.assertIn("ISN Substrate约束", content)

    def test_missing_schema_file(self):
        """不存在的schema文件不报错。"""
        enforcer = SubstrateEnforcer(schema_path=Path("/nonexistent/schema.json"))
        content = enforcer.compile()
        self.assertIn("ISN Substrate约束", content)


if __name__ == "__main__":
    unittest.main()
