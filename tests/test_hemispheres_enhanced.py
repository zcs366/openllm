#!/usr/bin/env python3
"""
增强版左右脑对弈机制测试脚本

测试EnhancedHemispherePair和EnhancedArbiter的功能。
"""

import sys
from pathlib import Path

# 添加openLLM路径
openllm_path = Path(__file__).parent.parent / "src"
if str(openllm_path) not in sys.path:
    sys.path.insert(0, str(openllm_path))

def test_enhanced_arbiter():
    """测试EnhancedArbiter的基本功能。"""
    print("🧪 测试 EnhancedArbiter...")
    
    try:
        from openllm.core.hemispheres_enhanced import (
            EnhancedArbiter,
            ArbitrationStrategy,
            ArbitrationContext,
            ArbitrationResult,
        )
        
        # 测试1: 创建实例
        print("\n1. 创建 EnhancedArbiter 实例...")
        arbiter = EnhancedArbiter(ArbitrationStrategy.BALANCED)
        print(f"   ✅ 实例创建成功")
        
        # 测试2: 测试各种策略
        print("\n2. 测试各种仲裁策略...")
        
        test_cases = [
            {
                "name": "无反对意见",
                "left": "使用Python 3.12",
                "right": "",
                "ev_left": ["Python 3.12性能提升20%"],
                "ev_right": [],
            },
            {
                "name": "左脑有证据",
                "left": "使用FastAPI",
                "right": "应该用Flask",
                "ev_left": ["FastAPI性能更好", "自动文档生成"],
                "ev_right": [],
            },
            {
                "name": "右脑有证据",
                "left": "直接上线",
                "right": "需要先测试",
                "ev_left": [],
                "ev_right": ["上次没测试出bug了", "测试覆盖率只有30%"],
            },
            {
                "name": "双方都有证据",
                "left": "用Redis缓存",
                "right": "用Memcached",
                "ev_left": ["Redis支持更多数据结构", "持久化"],
                "ev_right": ["Memcached更简单", "内存效率更高"],
            },
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n   测试案例 {i}: {case['name']}")
            
            # 测试每种策略
            for strategy in ArbitrationStrategy:
                result = arbiter.arbitrate(
                    left_proposal=case["left"],
                    right_critique=case["right"],
                    evidence_left=case["ev_left"],
                    evidence_right=case["ev_right"],
                    strategy=strategy,
                )
                print(f"     {strategy.value}: {result.verdict} (置信度: {result.confidence:.2f})")
        
        # 测试3: 性能统计
        print("\n3. 测试性能统计...")
        stats = arbiter.get_performance_stats()
        print(f"   总仲裁次数: {stats['total_arbitrations']}")
        print(f"   平均置信度: {stats['average_confidence']:.2f}")
        print(f"   策略性能: {stats['strategy_performance']}")
        
        print("\n" + "="*50)
        print("✅ EnhancedArbiter 测试完成！")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_hemisphere_pair():
    """测试EnhancedHemispherePair的基本功能。"""
    print("\n🧪 测试 EnhancedHemispherePair...")
    
    try:
        from openllm.core.hemispheres_enhanced import (
            EnhancedHemispherePair,
            ArbitrationStrategy,
            ArbitrationContext,
        )
        
        # 测试1: 创建实例
        print("\n1. 创建 EnhancedHemispherePair 实例...")
        pair = EnhancedHemispherePair(
            left_name="左脑",
            right_name="右脑",
            default_strategy=ArbitrationStrategy.BALANCED,
        )
        print(f"   ✅ 实例创建成功")
        
        # 测试2: 测试增强仲裁
        print("\n2. 测试增强仲裁...")
        result = pair.arbitrate_with_strategy(
            left_proposal="使用Python 3.12",
            right_critique="应该用Python 3.11",
            evidence_left=["Python 3.12性能提升20%"],
            evidence_right=["Python 3.11更稳定"],
            context=ArbitrationContext(
                task_id="test-001",
                task_description="选择Python版本",
                risk_level="low",
            ),
            strategy=ArbitrationStrategy.EVIDENCE_BASED,
        )
        print(f"   裁决: {result.verdict}")
        print(f"   决议: {result.resolution}")
        print(f"   置信度: {result.confidence:.2f}")
        print(f"   推理: {result.reasoning}")
        
        # 测试3: 测试统计
        print("\n3. 测试统计...")
        stats = pair.get_arbitration_stats()
        print(f"   总仲裁次数: {stats['total_arbitrations']}")
        print(f"   平均置信度: {stats['average_confidence']:.2f}")
        
        print("\n" + "="*50)
        print("✅ EnhancedHemispherePair 测试完成！")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quick_arbitrate():
    """测试quick_arbitrate便捷函数。"""
    print("\n🧪 测试 quick_arbitrate 便捷函数...")
    
    try:
        from openllm.core.hemispheres_enhanced import (
            quick_arbitrate,
            ArbitrationStrategy,
        )
        
        # 测试快速仲裁
        print("\n1. 测试快速仲裁...")
        result = quick_arbitrate(
            left_proposal="使用Docker部署",
            right_critique="应该用K8s",
            evidence_left=["Docker更简单", "开发环境一致"],
            evidence_right=["K8s自动扩缩容", "高可用"],
            strategy=ArbitrationStrategy.BALANCED,
        )
        print(f"   裁决: {result.verdict}")
        print(f"   决议: {result.resolution}")
        print(f"   置信度: {result.confidence:.2f}")
        
        print("\n" + "="*50)
        print("✅ quick_arbitrate 测试完成！")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("增强版左右脑对弈机制测试")
    print("="*60)
    
    # 测试EnhancedArbiter
    success1 = test_enhanced_arbiter()
    
    # 测试EnhancedHemispherePair
    success2 = test_enhanced_hemisphere_pair()
    
    # 测试quick_arbitrate
    success3 = test_quick_arbitrate()
    
    print("\n" + "="*60)
    if success1 and success2 and success3:
        print("🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("💥 部分测试失败！")
        sys.exit(1)