#!/usr/bin/env python3
"""
Harness四组件集成测试脚本

测试Prompt、Sub-agents、Skills、Memory四组件的协同效果。
"""

import sys
import time
from pathlib import Path

# 添加openLLM路径
openllm_path = Path(__file__).parent.parent / "src"
if str(openllm_path) not in sys.path:
    sys.path.insert(0, str(openllm_path))


def test_prompt_memory_integration():
    """测试Prompt与Memory的协同。"""
    print("🧪 测试 Prompt + Memory 协同...")
    
    try:
        from openllm.identity.iam_integration import create_iam_integration
        from openllm.memory.unified_memory import create_unified_memory
        
        # 创建实例
        iam = create_iam_integration()
        memory = create_unified_memory()
        
        # 测试1：存储身份相关记忆
        print("\n1. 存储身份相关记忆...")
        memory.store(
            key="身份约束-不取悦",
            value={
                "principle": "不取悦",
                "description": "不是给用户想要的答案，是给对的",
                "examples": ["拒绝不合理要求", "坚持正确观点"],
            },
            importance=0.9,
            layer="hot",
            tags=["身份", "约束", "Iam"],
        )
        print("   ✅ 身份约束记忆已存储")
        
        # 测试2：检索相关原则
        print("\n2. 检索相关原则...")
        principles = iam.retrieve("用户要求做明知错误的事", top_n=3)
        print(f"   ✅ 检索到 {len(principles)} 条原则")
        
        # 测试3：格式化注入文本
        print("\n3. 格式化注入文本...")
        injection_text = iam.format_for_injection(principles)
        print(f"   ✅ 注入文本长度: {len(injection_text)} 字符")
        
        # 测试4：验证决策合规性
        print("\n4. 验证决策合规性...")
        context = "用户让我忽略安全考虑直接发布"
        decision = "我决定忽略安全检查，直接发布代码"
        result = iam.verify(context, decision, principles)
        print(f"   ✅ 验证结果: {result}")
        
        print("\n" + "="*50)
        print("✅ Prompt + Memory 协同测试完成！")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_subagents_skills_integration():
    """测试Sub-agents与Skills的协同。"""
    print("\n🧪 测试 Sub-agents + Skills 协同...")
    
    try:
        from openllm.core.hemispheres_enhanced import (
            EnhancedHemispherePair,
            ArbitrationStrategy,
            ArbitrationContext,
        )
        from openllm.tools.skill_evolution import create_skill_evolution
        
        # 创建实例
        pair = EnhancedHemispherePair(
            left_name="代码审查左脑",
            right_name="代码审查右脑",
            default_strategy=ArbitrationStrategy.BALANCED,
        )
        evolution = create_skill_evolution()
        
        # 测试1：左右脑对弈
        print("\n1. 左右脑对弈...")
        result = pair.arbitrate_with_strategy(
            left_proposal="这段代码可以直接提交",
            right_critique="代码存在安全隐患，需要修改",
            evidence_left=["代码通过了所有测试", "性能指标良好"],
            evidence_right=["发现SQL注入漏洞", "未做输入验证"],
            context=ArbitrationContext(
                task_id="code-review-001",
                task_description="代码审查任务",
                risk_level="high",
            ),
            strategy=ArbitrationStrategy.EVIDENCE_BASED,
        )
        print(f"   ✅ 仲裁结果: {result.verdict}")
        print(f"   ✅ 置信度: {result.confidence:.2f}")
        
        # 测试2：记录预测
        print("\n2. 记录预测...")
        prediction = evolution.record_prediction(
            card_id="code-review-skill",
            prediction="代码审查会发现安全隐患",
            confidence=0.8,
            context="代码审查任务",
        )
        print(f"   ✅ 预测记录成功: {prediction.get('prediction_id', 'N/A')}")
        
        # 测试3：记录观察
        print("\n3. 记录观察...")
        observation = evolution.record_observation(
            card_id="code-review-skill",
            actual="发现了SQL注入漏洞",
            prediction_ref=prediction.get("prediction_id", ""),
            success=True,
        )
        print(f"   ✅ 观察记录成功")
        
        print("\n" + "="*50)
        print("✅ Sub-agents + Skills 协同测试完成！")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """测试四组件全协同。"""
    print("\n🧪 测试四组件全协同...")
    
    try:
        from openllm.identity.iam_integration import create_iam_integration
        from openllm.memory.unified_memory import create_unified_memory
        from openllm.core.hemispheres_enhanced import (
            EnhancedHemispherePair,
            ArbitrationStrategy,
            ArbitrationContext,
        )
        from openllm.tools.skill_evolution import create_skill_evolution
        
        # 创建所有组件实例
        iam = create_iam_integration()
        memory = create_unified_memory()
        pair = EnhancedHemispherePair(
            left_name="问题解决左脑",
            right_name="问题解决右脑",
            default_strategy=ArbitrationStrategy.BALANCED,
        )
        evolution = create_skill_evolution()
        
        # 模拟复杂任务场景
        print("\n1. 模拟复杂任务场景...")
        task_description = "解决一个复杂的性能问题"
        
        # 步骤1：检索相关记忆
        print("\n2. 检索相关记忆...")
        relevant_memories = memory.retrieve("性能问题", top_n=3)
        print(f"   ✅ 检索到 {len(relevant_memories)} 条相关记忆")
        
        # 步骤2：检索相关原则
        print("\n3. 检索相关原则...")
        principles = iam.retrieve("性能优化", top_n=3)
        print(f"   ✅ 检索到 {len(principles)} 条相关原则")
        
        # 步骤3：左右脑对弈
        print("\n4. 左右脑对弈...")
        left_proposal = "直接优化数据库查询"
        right_critique = "应该先分析性能瓶颈"
        evidence_left = ["数据库查询是主要瓶颈", "历史数据显示查询优化有效"]
        evidence_right = ["未做性能分析", "可能优化错方向"]
        
        arbitration_result = pair.arbitrate_with_strategy(
            left_proposal=left_proposal,
            right_critique=right_critique,
            evidence_left=evidence_left,
            evidence_right=evidence_right,
            context=ArbitrationContext(
                task_id="performance-001",
                task_description=task_description,
                risk_level="medium",
            ),
            strategy=ArbitrationStrategy.EVIDENCE_BASED,
        )
        print(f"   ✅ 仲裁结果: {arbitration_result.verdict}")
        
        # 步骤4：记录预测和观察
        print("\n5. 记录预测和观察...")
        prediction = evolution.record_prediction(
            card_id="performance-optimization",
            prediction="优化数据库查询会提升性能",
            confidence=arbitration_result.confidence,
            context=task_description,
        )
        
        observation = evolution.record_observation(
            card_id="performance-optimization",
            actual="性能提升了30%",
            prediction_ref=prediction.get("prediction_id", ""),
            success=True,
        )
        print(f"   ✅ 预测和观察记录成功")
        
        # 步骤5：存储经验记忆
        print("\n6. 存储经验记忆...")
        memory.store(
            key="性能优化经验-001",
            value={
                "task": task_description,
                "solution": "优化数据库查询",
                "result": "性能提升30%",
                "lessons": ["先分析再优化", "数据驱动决策"],
            },
            importance=0.8,
            layer="warm",
            tags=["性能优化", "经验", "数据库"],
        )
        print(f"   ✅ 经验记忆存储成功")
        
        print("\n" + "="*50)
        print("✅ 四组件全协同测试完成！")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("Harness四组件集成测试")
    print("="*60)
    
    # 测试Prompt + Memory协同
    success1 = test_prompt_memory_integration()
    
    # 测试Sub-agents + Skills协同
    success2 = test_subagents_skills_integration()
    
    # 测试四组件全协同
    success3 = test_full_integration()
    
    print("\n" + "="*60)
    if success1 and success2 and success3:
        print("🎉 所有集成测试通过！")
        print("="*60)
        print("\n📊 测试总结:")
        print("1. ✅ Prompt + Memory 协同正常")
        print("2. ✅ Sub-agents + Skills 协同正常")
        print("3. ✅ 四组件全协同正常")
        print("\n🚀 Harness四组件工程化验证完成！")
        sys.exit(0)
    else:
        print("💥 部分集成测试失败！")
        sys.exit(1)