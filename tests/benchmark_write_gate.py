#!/usr/bin/env python3
"""Write Gate性能基准测试"""
import sys
import os
import time

sys.path.insert(0, os.path.expanduser("~/.hermes/jiak"))

from belief_update import write_gate, cosine_similarity_text, compute_jaccard

print("=" * 50)
print("Write Gate Performance Benchmark")
print("=" * 50)

# 测试数据
test_pairs = [
    ("端口配错应该是8080", "端口配错，应该是8080"),
    ("openLLM四体架构验证通过", "openLLM四体架构测试成功"),
    ("数据库连接超时", "数据库连接失败"),
    ("Python代码重构优化", "JavaScript测试编写"),
    ("今天天气很好适合出门", "openLLM四体架构验证通过"),
]

# 预热jieba
import jieba
list(jieba.cut("预热测试"))

# 测试1：单次调用性能
print("\n[1] 单次调用性能...")
times = []
for _ in range(100):
    start = time.perf_counter()
    for insight, opinion in test_pairs:
        write_gate(insight, opinion)
    elapsed = time.perf_counter() - start
    times.append(elapsed)

avg_time = sum(times) / len(times)
print(f"  平均100次调用: {avg_time*1000:.2f}ms")
print(f"  单次调用: {avg_time/len(test_pairs)*1000:.3f}ms")

# 测试2：cosine单独性能
print("\n[2] Cosine单独性能...")
start = time.perf_counter()
for _ in range(1000):
    for insight, opinion in test_pairs:
        cosine_similarity_text(insight, opinion)
elapsed = time.perf_counter() - start
print(f"  1000次cosine调用: {elapsed*1000:.2f}ms")
print(f"  单次cosine: {elapsed/1000*1000:.3f}ms")

# 测试3：jaccard单独性能
print("\n[3] Jaccard单独性能...")
start = time.perf_counter()
for _ in range(1000):
    for insight, opinion in test_pairs:
        compute_jaccard(insight, opinion)
elapsed = time.perf_counter() - start
print(f"  1000次jaccard调用: {elapsed*1000:.2f}ms")
print(f"  单次jaccard: {elapsed/1000*1000:.3f}ms")

# 评估
print("\n[4] 性能评估...")
if avg_time/len(test_pairs)*1000 < 1.0:
    print("  ✅ 性能优秀: <1ms/次")
elif avg_time/len(test_pairs)*1000 < 10.0:
    print("  ⚠️ 性能良好: <10ms/次")
else:
    print("  ❌ 性能不足: >10ms/次，需要优化")

print("\n" + "=" * 50)
print("Benchmark Complete")
print("=" * 50)
