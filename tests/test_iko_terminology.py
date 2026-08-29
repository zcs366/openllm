"""test_iko_terminology.py — P2-9: 验证IKO报告术语诚实化"""
import pytest


def test_report_key_renamed():
    """验证report()返回total_observations而非total_ticks"""
    from openllm.core.iko_impl import IKO
    iko = IKO()
    iko.trace("test", "ok")
    report = iko.report()
    assert "total_observations" in report
    assert "total_ticks" not in report
    assert report["total_observations"] == 1
    print("✓ report() key: total_observations")


def test_report_has_quality_sample_count():
    """验证report()包含quality_sample_count"""
    from openllm.core.iko_impl import IKO
    iko = IKO()
    iko.trace("test", "ok")
    report = iko.report()
    assert "quality_sample_count" in report
    assert isinstance(report["quality_sample_count"], int)
    print(f"✓ report() quality_sample_count={report['quality_sample_count']}")


def test_shutdown_uses_chinese_label():
    """验证关闭报告用'次观测'而非'tick'，质量分附样本数"""
    import io, contextlib
    from openllm.core.iko_impl import IKO
    iko = IKO()
    iko.trace("test", "ok")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        iko.shutdown()
    output = buf.getvalue()
    assert "次观测" in output, f"缺少'次观测': {output}"
    assert "tick" not in output.lower() or "tick" not in output, f"仍含'tick': {output}"
    assert "样本" in output, f"缺少'样本': {output}"
    print(f"✓ 关闭报告格式: {output.strip()}")


def test_quality_scorer_sample_count():
    """验证QualityScorer.sample_count正确计数"""
    from openllm.core.iko_impl import QualityScorer, OutputScene
    scorer = QualityScorer()
    assert scorer.sample_count == 0
    for _ in range(5):
        scorer.score("test output", OutputScene.CASUAL, {}, {})
    assert scorer.sample_count == 5
    print("✓ QualityScorer.sample_count正确")
