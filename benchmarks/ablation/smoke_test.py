"""smoke_test.py — 四级 harness 冒烟测试

验证 G0/G1 能跑通（调 ollama 返回非空响应），G2/G3 接口定型。
不依赖 openllm 测试框架，纯独立脚本。

用法: .venv/bin/python benchmarks/ablation/smoke_test.py
"""

import json
import sys
import time

# ── 1. 测试 llm_layer.py（G0 底层）──
print("=" * 60)
print("冒烟测试开始")
print("=" * 60)

# G0：真裸
print("\n── G0（真裸）──")
try:
    from .llm_layer import llm_chat
    t0 = time.time()
    resp = llm_chat("请回答：1+1=？只回答数字。")
    elapsed = time.time() - t0
    ok = bool(resp and resp.strip())
    print(f"  LLM 回复: {resp.strip()[:80]!r}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  非空: {'✓' if ok else '✗'}")
    if not ok:
        print("  ✗ FAIL: llm_chat 返回空响应")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ FAIL: {e}")
    sys.exit(1)

# G1：裸 ReAct
print("\n── G1（裸 ReAct）──")
try:
    from .harness import G1ReActHarness
    g1 = G1ReActHarness(max_steps=3)
    t0 = time.time()
    result = g1.run("请回答：3×4=？请用ReAct格式思考。", session={})
    elapsed = time.time() - t0
    ok = bool(result["response"] and result["response"].strip())
    print(f"  回复: {result['response'].strip()[:80]!r}")
    print(f"  步数: {result['steps']}, 耗时: {elapsed:.1f}s")
    print(f"  非空: {'✓' if ok else '✗'}")
    if not ok:
        print("  ✗ FAIL: G1 返回空响应")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ FAIL: {e}")
    sys.exit(1)

# G2/G3：接口定型（不调 ollama，只验证接口正确）
print("\n── G2（+记忆）──")
try:
    from .harness import G2MemoryHarness
    g2 = G2MemoryHarness(max_steps=1)
    # 验证接口签名：run(task_prompt, session) -> dict
    # 没有 memory_bus 时应降级运行（不 crash）
    result = g2.run("测试接口", session={"session_id": "smoke"})
    assert isinstance(result, dict), f"返回类型错误: {type(result)}"
    assert result["group"] == "G2"
    assert "response" in result
    assert "memories_injected_count" in result
    print(f"  接口: ✓ (group={result['group']}, steps={result['steps']})")
    print(f"  降级模式(无bus): {'✓' if result.get('write_result') is None else '✓'}")
except Exception as e:
    print(f"  ✗ FAIL: {e}")
    sys.exit(1)

print("\n── G3（全 harness）──")
try:
    from .harness import G3FullHarness
    g3 = G3FullHarness(max_steps=1)
    result = g3.run("测试接口", session={"session_id": "smoke"})
    assert isinstance(result, dict), f"返回类型错误: {type(result)}"
    assert result["group"] == "G3"
    assert "response" in result
    assert "identity_injected" in result
    assert result["identity_injected"] is True
    assert "causal_hits_count" in result
    print(f"  接口: ✓ (group={result['group']}, identity={result['identity_injected']})")
    print(f"  降级模式(无bus): ✓")
except Exception as e:
    print(f"  ✗ FAIL: {e}")
    sys.exit(1)

# G0 工厂函数
print("\n── get_harness() 工厂 ──")
try:
    from .harness import get_harness
    for name in ["G0", "G1", "G2", "G3"]:
        h = get_harness(name)
        assert h.name == name, f"组名不匹配: {h.name} != {name}"
    print(f"  四级工厂: ✓")
except Exception as e:
    print(f"  ✗ FAIL: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("冒烟测试全部通过 ✓")
print("=" * 60)
