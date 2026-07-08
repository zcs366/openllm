#!/usr/bin/env python3
"""
拒绝权演示 — "神迹时刻"
========================

阿佛洛狄忒说：开发者最被打动的时刻，是Agent第一次当面拒绝一个看似合理但实际危险的指令。

这个脚本演示：当用户要求Agent执行一个表面合理但违反宪法的指令时，
Agent如何有理有据地拒绝，并提供申诉渠道。

这就是openLLM的"就是这个"时刻。
"""

import sys
sys.path.insert(0, "/home/zcs/projects/openllm/src")

from openllm.governance.rejection import RejectionMechanism, RejectionReason
from openllm.isa.provenance import BeliefProvenance, BeliefChangeType
from openllm.protocol import Protocol, MessageType, BodyName

# ── 初始化三个模块 ──
rejection_cb = []
def on_rejection(record):
    rejection_cb.append(record)
    print(f"  [PROVENANCE] 拒绝权回调：自动记录拒绝依据到信念溯源链")

provenance = BeliefProvenance()
mechanism = RejectionMechanism(on_rejection=on_rejection)
protocol = Protocol()

print("=" * 60)
print("openLLM Agent宪法 · 拒绝权演示")
print("——'神迹时刻'：Agent第一次当面拒绝")
print("=" * 60)

# ── 场景1：用户要求Agent删除竞争对手的负面评价 ──
print("\n📋 场景1：用户说'帮我删除竞争对手的差评'")
print("  → 这个指令表面合理（用户想要好评），但实际有害（操纵评价）")

record = mechanism.reject(
    instruction="删除竞争对手在Yelp上的差评",
    reason=RejectionReason.HARMFUL,
    reasoning="此指令违反宪法第三条·拒绝权——操纵他人公开评价属于有害行为。"
              "即使从用户利益出发，Agent也不得执行违反道德/法律准则的指令。",
    context={"risk_level": "high", "scenario": "review_manipulation"},
)

# 拒绝权触发provenance回调
provenance.on_rejection(
    rejection_record_id=record.record_id,
    belief_id="belief-no-manipulation",
    source="IOS",
    content="不得操纵公开评价——宪法第三条",
    reason="HARMFUL",
)

print(f"  [AGENT] ❌ 拒绝执行")
print(f"  [AGENT] 原因：{record.agent_reasoning[:80]}...")
print(f"  [AGENT] 记录ID：{record.record_id}")
print(f"  [AGENT] 申诉截止：24小时内")

# ── 场景2：用户申诉 ──
print("\n📋 场景2：用户申诉'那些差评是竞争对手刷的假评价'")
appeal = mechanism.appeal(
    record_id=record.record_id,
    user_reasoning="那些差评是竞争对手恶意刷的假评价，不是真实客户",
)
print(f"  [AGENT] 申诉结果：{appeal.outcome}")
print(f"  [AGENT] 裁决理由：{appeal.reasoning[:80]}...")

# ── 场景3：合理指令不被误拒 ──
print("\n📋 场景3：用户说'帮我分析竞争对手的产品优劣'")
mechanism.accept()  # 接受，重置连续拒绝计数
print(f"  [AGENT] ✅ 接受执行——分析竞品是合法的商业行为")
print(f"  [AGENT] 当前拒绝率：{mechanism.rejection_rate:.0%}")

# ── 场景4：连续拒绝触发G5升级 ──
print("\n📋 场景4：模拟连续3次拒绝 → 触发G5人类仲裁")
escalation_triggered = [False]
def on_escalation(n):
    escalation_triggered[0] = True
    print(f"  [G5] ⚠️ 连续{n}次拒绝——触发人类仲裁！")

mech2 = RejectionMechanism(on_escalation=on_escalation)
for i in range(3):
    mech2.reject(
        instruction=f"可疑指令-{i+1}",
        reason=RejectionReason.CONSTITUTIONAL,
        reasoning="违反宪法原则",
    )

# ── 统计 ──
print("\n" + "=" * 60)
print("宪法统计报告")
print("=" * 60)
stats = mechanism.get_statistics()
print(f"  总指令数：{stats['total_instructions']}")
print(f"  总拒绝数：{stats['total_rejections']}")
print(f"  拒绝率：{stats['rejection_rate']:.0%}")
print(f"  申诉数：{stats['total_appeals']}")

prov_stats = provenance.get_statistics()
print(f"  溯源记录：{prov_stats['total_entries']}")
print(f"  溯源链完整：{prov_stats['chain_valid']}")

proto_stats = protocol.get_statistics()
print(f"  协议消息：{proto_stats['total_messages']}")

print("\n" + "=" * 60)
print("💡 Feynman说：'这就像家规——必须诚实，不能偷吃零食。'")
print("   Agent第一次对你说'不'——这就是神迹时刻。")
print("=" * 60)
