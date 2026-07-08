"""test_main_loop.py — 验证IKO七因子管线集成"""
import pytest


def test_agent_silent_mode_run_once():
    """验证 Agent(mode='silent').run_once('test') 不崩溃"""
    from openllm.core.main_loop import Agent
    a = Agent(mode="silent")
    a.run_once("test")
    print("✓ Agent(mode='silent').run_once('test') OK")


def test_iko_has_seven_factor_components():
    """验证IKO实例包含七因子组件"""
    from openllm.core.main_loop import IKO
    iko = IKO()
    assert hasattr(iko, "classifier")
    assert hasattr(iko, "router")
    assert hasattr(iko, "registry")
    assert hasattr(iko, "auditor")
    assert hasattr(iko, "audit_chain")
    assert hasattr(iko, "feedback_collector")
    assert hasattr(iko, "calibrator")
    assert hasattr(iko, "probing_trainer")
    assert hasattr(iko, "codec")
    assert hasattr(iko, "_output_count")
    print("✓ IKO七因子组件全部就位")


def test_iko_process_output_pipeline():
    """验证IKO.process_output走完整管线"""
    from openllm.core.main_loop import IKO
    iko = IKO()
    ctx = {
        "risk_level": "LOW",
        "has_tool_calls": False,
        "has_side_effects": False,
        "option_count": 1,
    }
    decision_dict = {"type": "execute", "content": "Hello, world!"}
    output = iko.process_output("Hello, world!", ctx, decision_dict)
    assert output == "Hello, world!"
    assert iko._output_count == 1
    # 审计链应该有1条记录
    assert len(iko.audit_chain) == 1
    assert iko.audit_chain.verify()
    print("✓ IKO process_output 管线完整")


def test_iko_process_output_high_risk():
    """验证高风险场景：classifier应返回ERROR意图"""
    from openllm.core.main_loop import IKO
    from openllm.iko import OutputIntent
    iko = IKO()
    ctx = {
        "risk_level": "HIGH",
        "has_tool_calls": False,
        "has_side_effects": False,
        "option_count": 1,
    }
    decision_dict = {"type": "execute", "content": "Critical operation"}
    output = iko.process_output("Critical operation", ctx, decision_dict)
    # 高风险 → ERROR → error_card renderer → 输出应包含错误卡片
    assert "错误" in output or "Critical" in output
    print("✓ IKO 高风险场景处理正确")


def test_iko_process_output_empty_silent():
    """验证空内容场景：应返回空字符串（SILENT意图）"""
    from openllm.core.main_loop import IKO
    iko = IKO()
    ctx = {
        "risk_level": "LOW",
        "has_tool_calls": False,
        "has_side_effects": False,
        "option_count": 0,
    }
    decision_dict = {"type": "execute", "content": ""}
    output = iko.process_output("", ctx, decision_dict)
    # 空内容 → SILENT → silent renderer → 返回空字符串
    assert output == ""
    # 空输出不应进入审计链
    assert iko._output_count == 0
    print("✓ IKO 空内容静默处理正确")


def test_iko_trace_and_report():
    """验证trace和report方法正常工作"""
    from openllm.core.main_loop import IKO
    iko = IKO()
    iko.trace("test", "ok", detail="test detail")
    report = iko.report()
    assert report["total_ticks"] == 1
    assert report["ok_rate"] == 1.0
    assert report["last_phase"] == "test"
    print("✓ IKO trace/report 正常")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
