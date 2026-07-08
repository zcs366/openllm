"""
Tests for probing_trainer.py — 追问训练器单元测试
"""

import sys
import os

# 确保 src 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.iko.probing_trainer import ProbingTrainer


def test_m1_prompt_at_interval():
    """测试 M1 阶段（<30 sessions）每 5 次提示 1 次。"""
    trainer = ProbingTrainer()
    # session_count=5 → 5 % 5 == 0 且 > 0 → 提示
    assert trainer.should_prompt_probing("u1", 5) is True
    # session_count=10 → 提示
    assert trainer.should_prompt_probing("u1", 10) is True
    # session_count=3 → 不提示
    assert trainer.should_prompt_probing("u1", 3) is False


def test_m2_prompt_at_interval():
    """测试 M2 阶段（30-90 sessions）每 10 次提示 1 次。"""
    trainer = ProbingTrainer()
    # session_count=30 → 30 % 10 == 0 → 提示
    assert trainer.should_prompt_probing("u1", 30) is True
    # session_count=40 → 提示
    assert trainer.should_prompt_probing("u1", 40) is True
    # session_count=35 → 不提示
    assert trainer.should_prompt_probing("u1", 35) is False


def test_m3_no_prompt():
    """测试 M3 阶段（>90 sessions）不提示。"""
    trainer = ProbingTrainer()
    assert trainer.should_prompt_probing("u1", 91) is False
    assert trainer.should_prompt_probing("u1", 200) is False
    assert trainer.should_prompt_probing("u1", 1000) is False


def test_session_zero():
    """测试 session_count=0 不提示。"""
    trainer = ProbingTrainer()
    assert trainer.should_prompt_probing("u1", 0) is False


def test_suggestion_for_code_output():
    """测试代码输出的追问建议。"""
    trainer = ProbingTrainer()
    suggestion = trainer.get_probing_suggestion("```python\ndef foo():\n    pass\n```")
    assert "边界条件" in suggestion or "错误处理" in suggestion


def test_suggestion_for_long_output():
    """测试长输出的追问建议。"""
    trainer = ProbingTrainer()
    long_output = "A" * 600
    suggestion = trainer.get_probing_suggestion(long_output)
    assert "简洁" in suggestion or "表达方式" in suggestion


def test_suggestion_empty_output():
    """测试空输出返回空字符串。"""
    trainer = ProbingTrainer()
    assert trainer.get_probing_suggestion("") == ""
    assert trainer.get_probing_suggestion("   ") == ""


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
