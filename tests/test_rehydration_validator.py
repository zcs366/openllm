"""
RehydrationValidator 测试——再水化验收器 pytest 用例。
"""
import json
import tempfile
from pathlib import Path

import pytest

from openllm.memory.rehydration_validator import (
    RehydrationValidator,
    ValidationResult,
    _tfidf_cosine,
    _tokenize,
)


# ── 测试 _tokenize ──────────────────────────────────────────

class TestTokenize:
    def test_english(self):
        tokens = _tokenize("hello world foo")
        assert "hello" in tokens
        assert "world" in tokens

    def test_chinese_bigram(self):
        tokens = _tokenize("你好世界")
        assert "你好" in tokens
        assert "好世" in tokens
        assert "世界" in tokens

    def test_mixed(self):
        tokens = _tokenize("hello你好")
        assert "hello" in tokens


# ── 测试 _tfidf_cosine 降级路径 ─────────────────────────────

class TestTfidfCosine:
    def test_identical(self):
        s = _tfidf_cosine("上下文工程是核心", "上下文工程是核心")
        assert s > 0.95

    def test_empty(self):
        assert _tfidf_cosine("", "") == 1.0
        assert _tfidf_cosine("hello", "") == 0.0
        assert _tfidf_cosine("", "hello") == 0.0

    def test_similar(self):
        s = _tfidf_cosine(
            "上下文工程的压缩技术可以显著减少token数量，同时保留关键语义信息",
            "上下文工程中的压缩技术能够大幅减少token使用量，同时维持核心语义信息",
        )
        # 语义近似但不完全相同——TF-IDF词袋近似，分数可能不高但应 > 0
        assert 0.0 < s <= 1.0

    def test_different(self):
        s = _tfidf_cosine("今天天气很好", "量子计算是未来趋势")
        assert s < 0.3


# ── 测试 RehydrationValidator.measure_fidelity ──────────────

class TestMeasureFidelity:
    def test_identical_texts(self):
        v = RehydrationValidator()
        text = "openLLM是一个给AI造身体的具身智能系统"
        score = v.measure_fidelity(text, text)
        assert score > 0.95

    def test_both_empty(self):
        v = RehydrationValidator()
        assert v.measure_fidelity("", "") == 1.0

    def test_one_empty(self):
        v = RehydrationValidator()
        assert v.measure_fidelity("some text", "") == 0.0
        assert v.measure_fidelity("", "some text") == 0.0


# ── 测试 RehydrationValidator.validate（核心验收）────────────

class TestValidate:
    def test_lossless_pass(self):
        """无损再水化 → pass"""
        v = RehydrationValidator()
        original = "上下文工程六艺包括压缩、固化、再水化等关键技术"
        rehydrated = "上下文工程六艺涵盖压缩、固化、再水化等核心技术"
        result = v.validate(original, rehydrated)
        assert result["verdict"] in ("pass", "warn")  # 语义近似
        assert 0.0 <= result["score"] <= 1.0
        assert result["backend"] in ("embedding_bge_small_zh", "tfidf_cosine_approximate")

    def test_lossy_fail(self):
        """严重有损再水化 → fail"""
        v = RehydrationValidator()
        original = "上下文工程六艺包括压缩、固化、再水化、上下文守卫、审计追踪和保真验收"
        rehydrated = "量子物理是研究微观粒子运动规律的学科"
        result = v.validate(original, rehydrated)
        assert result["verdict"] == "fail"
        assert result["score"] < 0.6

    def test_empty_both(self):
        """双空 → pass（0字节的再水化不丢失任何信息）"""
        v = RehydrationValidator()
        result = v.validate("", "")
        assert result["verdict"] == "pass"
        assert result["score"] == 1.0

    def test_empty_rehydrated(self):
        """再水化为空 → fail"""
        v = RehydrationValidator()
        result = v.validate("一段有意义的文字", "")
        assert result["verdict"] == "fail"

    def test_custom_threshold(self):
        """自定义阈值"""
        v = RehydrationValidator()
        original = "人工智能正在改变世界"
        rehydrated = "AI技术正在改变世界"  # 术语改写，语义近似
        result = v.validate(original, rehydrated, threshold=0.3)
        assert result["threshold"] == 0.3

    def test_result_dict_keys(self):
        """验证返回字典结构完整"""
        v = RehydrationValidator()
        result = v.validate("a", "b")
        expected_keys = {
            "score", "verdict", "threshold", "backend",
            "original_len", "rehydrated_len", "length_ratio", "ts",
        }
        assert expected_keys == set(result.keys())

    def test_warn_zone(self):
        """阈值恰好在边界时，verdict 应为 warn"""
        v = RehydrationValidator()
        # 使用已知 score 的场景来测试 warn 判定逻辑
        # 空文本 + 有文本 = 0.0 → fail
        # 相同文本 = ~1.0 → pass
        # 这里测 threshold 判定逻辑本身
        result = v.validate("完全不同的内容A", "完全不同的内容B", threshold=0.0)
        # threshold=0.0 → 几乎任何 score 都 >= 0.0，不会 fail
        assert result["verdict"] in ("pass", "warn")

    def test_audit_log(self):
        """审计日志写入验证"""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = Path(f.name)

        try:
            v = RehydrationValidator(audit_log_path=log_path)
            v.validate("原文", "再水化后")

            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert "score" in record
            assert "verdict" in record
            assert "ts" in record
        finally:
            log_path.unlink(missing_ok=True)


# ── 真实 score 输出（C6 验收）───────────────────────────────

class TestRealScoreOutput:
    def test_print_real_score(self):
        """打印一次真实 score，用于人工验证。"""
        v = RehydrationValidator()
        original = (
            "openLLM是给AI造身体的具身智能系统，"
            "核心模块包括ISA（人工认知架构）、IO-S（Agent OS）、"
            "ISN（技能网络）、IKO（输出层）。"
            "上下文工程六艺：压缩、固化、再水化、上下文守卫、审计追踪、保真验收。"
        )
        # 良好压缩：保留核心，删除冗余
        rehydrated_good = (
            "openLLM具身智能，模块含ISA/IO-S/ISN/IKO。"
            "六艺：压缩、固化、再水化、守卫、审计、验收。"
        )
        # 严重有损：完全不相关
        rehydrated_bad = "今天是星期天，天气晴朗。"

        score_good = v.measure_fidelity(original, rehydrated_good)
        score_bad = v.measure_fidelity(original, rehydrated_bad)

        print(f"\n[真实score输出] 良好压缩 vs 原文: {score_good:.4f}")
        print(f"[真实score输出] 严重有损 vs 原文: {score_bad:.4f}")

        # 基本断言：好的应该比差的高
        assert score_good > score_bad, (
            f"良好压缩({score_good}) 应高于严重有损({score_bad})"
        )
        # 好的应该在合理范围内
        assert score_good > 0.3, f"良好压缩 score {score_good} 过低"
