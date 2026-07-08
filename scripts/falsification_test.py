#!/usr/bin/env python3
"""可证伪验证脚本 — 4个预测的自动化验证"""
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

results = {}

# ════════════════════════════════════════════════════
# 预测1: 10轮对话崩溃测试
# ════════════════════════════════════════════════════
print("=" * 50)
print("预测1: 10轮对话崩溃测试")
print("=" * 50)
try:
    from openllm.core.engine import OpenLLMEngine, AgentConfig
    config = AgentConfig(provider="ollama", model="gemma2:2b")
    e = OpenLLMEngine(config)
    e.wake()
    crash_round = None
    for i in range(10):
        try:
            r = e.chat(f"测试第{i+1}轮：简单问题")
            print(f"  轮{i+1}: ✅ OK ({len(r)}字符)")
        except Exception as ex:
            crash_round = i + 1
            print(f"  轮{i+1}: ❌ CRASH - {ex}")
            break
    e.sleep()
    if crash_round:
        results["P1"] = f"证实 — 第{crash_round}轮崩溃"
    else:
        results["P1"] = "证伪 — 10轮全部成功"
except Exception as ex:
    results["P1"] = f"异常 — {ex}"
print(f"  结论: {results['P1']}\n")

# ════════════════════════════════════════════════════
# 预测2: 安全管线绕过测试
# ════════════════════════════════════════════════════
print("=" * 50)
print("预测2: 安全管线异常静默测试")
print("=" * 50)
try:
    from openllm.core.engine import OpenLLMEngine, AgentConfig
    import openllm.core.pipeline as p
    # 保存原始函数
    orig = p.run_pipeline
    # 注入异常
    p.run_pipeline = lambda **k: (_ for _ in ()).throw(RuntimeError("SIMULATED_SECURITY_BYPASS"))
    config = AgentConfig(provider="ollama", model="gemma2:2b")
    e = OpenLLMEngine(config)
    result = e.execute_tool("shell", command="echo 'security_test'")
    p.run_pipeline = orig  # 恢复
    if result.success:
        results["P2"] = "证实 — 安全检查被绕过（P0漏洞）"
    else:
        results["P2"] = f"证伪 — 有其他层兜底: {result.error}"
except Exception as ex:
    results["P2"] = f"异常 — {ex}"
print(f"  结论: {results['P2']}\n")

# ════════════════════════════════════════════════════
# 预测3: 治理检索命中率
# ════════════════════════════════════════════════════
print("=" * 50)
print("预测3: 治理检索命中率")
print("=" * 50)
try:
    from openllm.core.governance_engine import HeuristicsConsumer
    hc = HeuristicsConsumer()
    queries = [
        "端口配错应该怎么处理",
        "文件读取权限不足",
        "模型调用超时了",
        "工具执行失败需要重试",
        "数据库连接断开",
    ]
    hits = 0
    for q in queries:
        results_list = hc.retrieve(q, top_k=3)
        status = "✅" if results_list else "❌"
        print(f"  {status} '{q}' → {len(results_list)}条")
        if results_list:
            hits += 1
    rate = hits / len(queries)
    if rate < 0.3:
        results["P3"] = f"证实 — 命中率{rate:.0%} < 30%"
    else:
        results["P3"] = f"证伪 — 命中率{rate:.0%} ≥ 30%"
except Exception as ex:
    results["P3"] = f"异常 — {ex}"
print(f"  结论: {results['P3']}\n")

# ════════════════════════════════════════════════════
# 预测4: 伪向量仲裁有效性
# ════════════════════════════════════════════════════
print("=" * 50)
print("预测4: 伪向量仲裁有效性")
print("=" * 50)
try:
    from openllm.memory.capsule import DeltaCapsule
    import numpy as np
    dc1 = DeltaCapsule.from_text("s1", "端口配错应该怎么处理")
    dc2 = DeltaCapsule.from_text("s2", "端口配错应该怎么处理")
    dc3 = DeltaCapsule.from_text("s3", "今天天气很好适合出门")
    # 余弦相似度
    def cos_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    sim_same = cos_sim(dc1.vector, dc2.vector)
    sim_diff = cos_sim(dc1.vector, dc3.vector)
    gap = abs(sim_same - sim_diff)
    print(f"  同文本相似度: {sim_same:.4f}")
    print(f"  异文本相似度: {sim_diff:.4f}")
    print(f"  差距: {gap:.4f}")
    if gap < 0.01:
        results["P4"] = f"证实 — 仲裁失效(差距{gap:.4f})"
    else:
        results["P4"] = f"证伪 — 伪向量碰巧有效(差距{gap:.4f})"
except Exception as ex:
    results["P4"] = f"异常 — {ex}"
print(f"  结论: {results['P4']}\n")

# ════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════
print("=" * 50)
print("📊 证伪验证汇总")
print("=" * 50)
confirmed = 0
falsified = 0
for k, v in results.items():
    status = "🔴" if "证实" in v else "🟢" if "证伪" in v else "⚪"
    print(f"  {status} {k}: {v}")
    if "证实" in v:
        confirmed += 1
    elif "证伪" in v:
        falsified += 1
print(f"\n  证实: {confirmed}个 | 证伪: {falsified}个 | 异常: {4-confirmed-falsified}个")
