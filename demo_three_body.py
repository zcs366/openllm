#!/usr/bin/env python3
"""
openLLM三体架构完整演示

展示：感知→决策→执行三层流水线
数据：unified_memory 29条真实记忆
"""
import sys
sys.path.insert(0, '/home/zcs/projects/openllm/src')

from openllm.core.three_body_heartbeat import heartbeat
from openllm.core.perception import ISA, 章鱼I
from openllm.core.decision import IOS
from openllm.core.execution import ISN, IKO
from openllm.core.models import Message

def main():
    print("=" * 60)
    print("openLLM 三体架构演示")
    print("=" * 60)
    
    # 初始化六体
    isa = ISA(mode='silent')
    octopus = 章鱼I()
    ios = IOS()
    isn = ISN()
    iko = IKO()
    
    # 测试查询
    queries = [
        "openLLM三体架构",
        "ISA记忆系统",
        "ISN verify",
        "IOS拒绝权",
        "章鱼I搜索",
        "三体通信协议",
        "Harness四组件",
        "ISA ICE修复",
        "ISN Skills生态",
        "记忆召回",
        "治理层",
        "执行层",
        "感知层",
        "心跳循环",
        "降级追踪",
    ]
    
    success = 0
    total = len(queries)
    
    for q in queries:
        hc = heartbeat(isa, octopus, ios, isn, iko, Message(text=q))
        recalled = len(hc.memory.get("recalled", []))
        if recalled > 0:
            success += 1
            print(f"✅ Q: {q}")
            print(f"   召回: {recalled} items")
            if hc.memory.get("recalled"):
                for r in hc.memory["recalled"][:2]:
                    print(f"   - {r['key']}: {str(r['content'])[:60]}")
            print(f"   决策: {hc.decision.action if hc.decision else 'N/A'}")
            print(f"   输出: {hc.output[:60]}")
        else:
            print(f"❌ Q: {q} (无召回)")
        print()
    
    print("=" * 60)
    print(f"结果: {success}/{total} ({success/total*100:.0f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
