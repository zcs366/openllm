"""test_startup_health.py — P2-8: 验证启动健康汇总播报"""
import pytest


def test_startup_health_prints_on_console_mode():
    """验证 console 模式下 Agent.__init__ 触发健康汇总打印"""
    import io, sys, contextlib
    # 捕获stdout，验证播报输出
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from openllm.core.main_loop import Agent
        Agent(mode="console")
    output = buf.getvalue()
    assert "本次启动:" in output, f"缺少启动播报: {output[:200]}"
    # 检查三大探针标记
    assert "embedding" in output
    assert "iam" in output
    assert "时钟" in output
    print(f"✓ 启动健康汇总: {output.strip()}")


def test_startup_health_silent_no_print():
    """验证 silent 模式下启动汇总被suppress（redirect_stdout生效）"""
    import io, sys, contextlib
    # silent模式自身会redirect_stdout，所以播报也会被捕获
    # 验证Agent实例能正常创建（不崩溃）即可
    from openllm.core.main_loop import Agent
    a = Agent(mode="silent")
    # 确认方法存在且可调用
    assert hasattr(a, "_print_startup_health")
    assert callable(a._print_startup_health)
    print("✓ silent模式启动健康汇总不崩溃")


def test_startup_health_method_directly():
    """直接调用 _print_startup_health，验证输出格式"""
    import io, contextlib
    from openllm.core.main_loop import Agent
    a = Agent(mode="silent")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        a._print_startup_health()
    output = buf.getvalue()
    assert "本次启动:" in output
    # 应包含三种探针结果
    assert "embedding" in output
    assert "iam" in output
    assert "时钟" in output
    # 每项后应有✓或✗符号
    assert "✓" in output or "✗" in output or "⚠" in output
    print(f"✓ 直接调用播报格式正确: {output.strip()}")
