#!/usr/bin/env python3
"""
技能自进化模块测试脚本

测试SkillEvolution的功能。
"""

import sys
from pathlib import Path

# 添加openLLM路径
openllm_path = Path(__file__).parent.parent / "src"
if str(openllm_path) not in sys.path:
    sys.path.insert(0, str(openllm_path))

def test_skill_evolution():
    """测试SkillEvolution的基本功能。"""
    print("🧪 测试 SkillEvolution...")
    
    try:
        from openllm.tools.skill_evolution import SkillEvolution, create_skill_evolution
        
        # 测试1: 创建实例
        print("\n1. 创建 SkillEvolution 实例...")
        evolution = create_skill_evolution(auto_load=True)
        print(f"   ✅ 实例创建成功")
        
        # 测试2: 检查加载状态
        print("\n2. 检查 world_model_evolver 加载状态...")
        if evolution.is_loaded():
            print(f"   ✅ world_model_evolver 已加载")
        else:
            print(f"   ⚠️ world_model_evolver 未加载")
            return False
        
        # 测试3: 记录预测
        print("\n3. 测试记录预测...")
        prediction = evolution.record_prediction(
            card_id="test-skill-001",
            prediction="这个技能会成功执行",
            confidence=0.8,
            context="测试上下文",
        )
        if prediction:
            print(f"   ✅ 预测记录成功")
            print(f"      预测ID: {prediction.get('prediction_id', 'N/A')}")
        else:
            print(f"   ⚠️ 预测记录失败")
        
        # 测试4: 记录观察
        print("\n4. 测试记录观察...")
        observation = evolution.record_observation(
            card_id="test-skill-001",
            actual="技能执行成功",
            prediction_ref=prediction.get("prediction_id", ""),
            success=True,
        )
        if observation:
            print(f"   ✅ 观察记录成功")
        else:
            print(f"   ⚠️ 观察记录失败")
        
        # 测试5: 记录不匹配
        print("\n5. 测试记录不匹配...")
        mismatch = evolution.record_mismatch(
            card_id="test-skill-001",
            prediction="技能会成功",
            actual="技能执行失败",
            severity="medium",
            context="测试不匹配",
        )
        if mismatch:
            print(f"   ✅ 不匹配记录成功")
        else:
            print(f"   ⚠️ 不匹配记录失败")
        
        # 测试6: 分析失败模式
        print("\n6. 测试分析失败模式...")
        patterns = evolution.analyze_failure_patterns("test-skill-001")
        print(f"   ✅ 分析完成，找到 {len(patterns)} 个失败模式")
        
        # 测试7: 提取规则
        print("\n7. 测试提取规则...")
        rules = evolution.extract_rules("test-skill-001")
        print(f"   ✅ 提取完成，找到 {len(rules)} 条规则")
        
        # 测试8: 获取置信度
        print("\n8. 测试获取置信度...")
        confidence = evolution.get_confidence("test-skill-001")
        print(f"   ✅ 置信度: {confidence:.2f}")
        
        # 测试9: 更新置信度
        print("\n9. 测试更新置置信度...")
        success = evolution.update_confidence(
            card_id="test-skill-001",
            confidence=0.9,
            reason="测试更新",
        )
        if success:
            print(f"   ✅ 置信度更新成功")
            new_confidence = evolution.get_confidence("test-skill-001")
            print(f"      新置信度: {new_confidence:.2f}")
        else:
            print(f"   ⚠️ 置信度更新失败")
        
        # 测试10: 获取进化状态
        print("\n10. 测试获取进化状态...")
        status = evolution.get_evolution_status("test-skill-001")
        print(f"   ✅ 进化状态: {status}")
        
        print("\n" + "="*50)
        print("✅ SkillEvolution 测试完成！")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_functions():
    """测试便捷函数。"""
    print("\n🧪 测试便捷函数...")
    
    try:
        from openllm.tools.skill_evolution import quick_predict, quick_observe
        
        # 测试快速预测
        print("\n1. 测试快速预测...")
        prediction = quick_predict(
            card_id="test-skill-002",
            prediction="快速预测测试",
            confidence=0.75,
        )
        if prediction:
            print(f"   ✅ 快速预测成功")
        else:
            print(f"   ⚠️ 快速预测失败")
        
        # 测试快速观察
        print("\n2. 测试快速观察...")
        observation = quick_observe(
            card_id="test-skill-002",
            actual="快速观察测试",
            prediction_ref=prediction.get("prediction_id", ""),
            success=True,
        )
        if observation:
            print(f"   ✅ 快速观察成功")
        else:
            print(f"   ⚠️ 快速观察失败")
        
        print("\n" + "="*50)
        print("✅ 便捷函数测试完成！")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("技能自进化模块测试")
    print("="*60)
    
    # 测试SkillEvolution
    success1 = test_skill_evolution()
    
    # 测试便捷函数
    success2 = test_quick_functions()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("💥 部分测试失败！")
        sys.exit(1)