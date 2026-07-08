#!/usr/bin/env python3
"""
统一记忆系统测试脚本

测试UnifiedMemory的功能。
"""

import sys
from pathlib import Path

# 添加openLLM路径
openllm_path = Path(__file__).parent.parent / "src"
if str(openllm_path) not in sys.path:
    sys.path.insert(0, str(openllm_path))

def test_unified_memory():
    """测试UnifiedMemory的基本功能。"""
    print("🧪 测试 UnifiedMemory...")
    
    try:
        from openllm.memory.unified_memory import UnifiedMemory, create_unified_memory
        
        # 测试1: 创建实例
        print("\n1. 创建 UnifiedMemory 实例...")
        memory = create_unified_memory()
        print(f"   ✅ 实例创建成功")
        
        # 测试2: 存储记忆
        print("\n2. 测试存储记忆...")
        
        # 热记忆
        hot_entry = memory.store(
            key="当前任务",
            value={"task": "测试统一记忆系统", "status": "进行中"},
            importance=0.9,
            layer="hot",
            tags=["测试", "记忆系统"],
        )
        print(f"   ✅ 热记忆存储成功: {hot_entry.key}")
        
        # 温记忆
        warm_entry = memory.store(
            key="决策-001",
            value={"decision": "使用Python 3.12", "reason": "性能更好"},
            importance=0.8,
            layer="warm",
            tags=["决策", "Python"],
        )
        print(f"   ✅ 温记忆存储成功: {warm_entry.key}")
        
        # 冷记忆
        cold_entry = memory.store(
            key="历史知识",
            value={"fact": "Python由Guido van Rossum创建", "year": 1991},
            importance=0.7,
            layer="cold",
            tags=["历史", "Python"],
        )
        print(f"   ✅ 冷记忆存储成功: {cold_entry.key}")
        
        # 测试3: 检索记忆
        print("\n3. 测试检索记忆...")
        
        # 检索所有层
        results = memory.retrieve("Python", top_n=10)
        print(f"   ✅ 检索到 {len(results)} 条记忆")
        for i, entry in enumerate(results, 1):
            print(f"      {i}. {entry.key} ({entry.layer}) - 温度: {entry.temperature():.2f}")
        
        # 检索指定层
        warm_results = memory.retrieve("决策", layer="warm", top_n=5)
        print(f"   ✅ 温记忆检索到 {len(warm_results)} 条")
        
        # 测试4: 获取指定记忆
        print("\n4. 测试获取指定记忆...")
        entry = memory.get("决策-001")
        if entry:
            print(f"   ✅ 获取成功: {entry.key}")
            print(f"      值: {entry.value}")
            print(f"      访问次数: {entry.access_count}")
        else:
            print(f"   ⚠️ 获取失败")
        
        # 测试5: 删除记忆
        print("\n5. 测试删除记忆...")
        success = memory.delete("测试删除")
        print(f"   ✅ 删除操作完成: {success}")
        
        # 测试6: 统计信息
        print("\n6. 测试统计信息...")
        stats = memory.get_stats()
        print(f"   ✅ 统计信息:")
        print(f"      热记忆: {stats['hot_memory_count']} 条")
        print(f"      温记忆: {stats['warm_memory_count']} 条")
        print(f"      冷记忆: {stats['cold_memory_count']} 条")
        print(f"      总计: {stats['total_memory_count']} 条")
        
        # 测试7: 检查点
        print("\n7. 测试检查点...")
        memory.checkpoint()
        print(f"   ✅ 检查点创建成功")
        
        # 测试8: 清理过期记忆
        print("\n8. 测试清理过期记忆...")
        memory.cleanup_expired()
        print(f"   ✅ 清理完成")
        
        print("\n" + "="*50)
        print("✅ UnifiedMemory 测试完成！")
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
        from openllm.memory.unified_memory import quick_store, quick_retrieve
        
        # 测试快速存储
        print("\n1. 测试快速存储...")
        entry = quick_store(
            key="快速测试",
            value={"test": "便捷函数测试"},
            importance=0.6,
            layer="warm",
        )
        print(f"   ✅ 快速存储成功: {entry.key}")
        
        # 测试快速检索
        print("\n2. 测试快速检索...")
        results = quick_retrieve("测试", top_n=5)
        print(f"   ✅ 快速检索到 {len(results)} 条记忆")
        
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
    print("统一记忆系统测试")
    print("="*60)
    
    # 测试UnifiedMemory
    success1 = test_unified_memory()
    
    # 测试便捷函数
    success2 = test_quick_functions()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("💥 部分测试失败！")
        sys.exit(1)