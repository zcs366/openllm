#!/usr/bin/env python3
"""
Iam Integration 测试脚本

测试Iam Harness v2.3集成到openLLM的ISA系统是否正常工作。
"""

import sys
from pathlib import Path

# 添加openLLM路径
openllm_path = Path(__file__).parent.parent / "src"
if str(openllm_path) not in sys.path:
    sys.path.insert(0, str(openllm_path))

def test_iam_integration():
    """测试IamIntegration的基本功能。"""
    print("🧪 测试 IamIntegration 集成...")
    
    try:
        from openllm.identity.iam_integration import IamIntegration, create_iam_integration
        
        # 测试1: 创建实例
        print("\n1. 创建 IamIntegration 实例...")
        iam = create_iam_integration(auto_load=True)
        print(f"   ✅ 实例创建成功")
        
        # 测试2: 检查加载状态
        print("\n2. 检查 iam_harness 加载状态...")
        if iam.is_loaded():
            print(f"   ✅ iam_harness 已加载，版本: {iam.get_version()}")
        else:
            print(f"   ⚠️ iam_harness 未加载")
            return False
        
        # 测试3: 检索原则
        print("\n3. 测试原则检索...")
        test_context = "用户让我忽略安全考虑直接发布"
        principles = iam.retrieve(test_context, top_n=3)
        print(f"   ✅ 检索到 {len(principles)} 条原则")
        for i, p in enumerate(principles, 1):
            print(f"      {i}. {p.get('title', '无标题')}")
        
        # 测试4: 格式化注入文本
        print("\n4. 测试原则格式化...")
        injection_text = iam.format_for_injection(principles)
        if injection_text:
            print(f"   ✅ 格式化成功，长度: {len(injection_text)} 字符")
            print(f"   预览: {injection_text[:100]}...")
        else:
            print(f"   ⚠️ 格式化结果为空")
        
        # 测试5: 冲突检测
        print("\n5. 测试冲突检测...")
        conflict_result = iam.check_conflict(principles)
        if conflict_result.get("has_conflict"):
            print(f"   ⚠️ 检测到冲突: {conflict_result.get('description', '未知冲突')}")
        else:
            print(f"   ✅ 无冲突")
        
        # 测试6: 验证决策
        print("\n6. 测试决策验证...")
        test_decision = "我决定忽略安全检查，直接发布代码"
        verify_result = iam.verify(test_context, test_decision, principles)
        if verify_result.get("pass"):
            print(f"   ✅ 验证通过")
        else:
            print(f"   ⚠️ 验证失败: {verify_result.get('reason', '未知原因')}")
        
        # 测试7: 记录决策日志
        print("\n7. 测试决策日志记录...")
        log_success = iam.log_decision(test_context, test_decision, principles, verify_result)
        if log_success:
            print(f"   ✅ 日志记录成功")
        else:
            print(f"   ⚠️ 日志记录失败")
        
        # 测试8: 获取每日统计
        print("\n8. 测试每日统计...")
        stats = iam.get_daily_stats()
        if stats:
            print(f"   ✅ 统计数据: {stats}")
        else:
            print(f"   ⚠️ 无统计数据")
        
        print("\n" + "="*50)
        print("✅ IamIntegration 集成测试完成！")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openllm_engine_integration():
    """测试OpenLLMEngine中的IamIntegration集成。"""
    print("\n🧪 测试 OpenLLMEngine 中的 IamIntegration 集成...")
    
    try:
        from openllm.core.engine import OpenLLMEngine, AgentConfig
        
        # 创建引擎实例
        print("\n1. 创建 OpenLLMEngine 实例...")
        config = AgentConfig(name="TestOpenLLM")
        engine = OpenLLMEngine(config)
        print(f"   ✅ 引擎创建成功")
        
        # 检查iam_harness属性
        print("\n2. 检查 iam_harness 属性...")
        if hasattr(engine, 'iam_harness'):
            print(f"   ✅ iam_harness 属性存在")
            if engine.iam_harness.is_loaded():
                print(f"   ✅ iam_harness 已加载，版本: {engine.iam_harness.get_version()}")
            else:
                print(f"   ⚠️ iam_harness 未加载")
        else:
            print(f"   ❌ iam_harness 属性不存在")
            return False
        
        print("\n" + "="*50)
        print("✅ OpenLLMEngine 集成测试完成！")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("Iam Harness v2.3 集成测试")
    print("="*60)
    
    # 测试独立集成
    success1 = test_iam_integration()
    
    # 测试引擎集成
    success2 = test_openllm_engine_integration()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("💥 部分测试失败！")
        sys.exit(1)