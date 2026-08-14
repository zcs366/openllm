"""
路由回归测试套件 — 10个标准意图→期望路由的映射

验证路由决策正确性，防止规则漂移。
路由规则变更后必须通过此测试套件。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openllm.core.router import RuleRouter, RoutingContext, PhaseAction


# ── 标准测试用例 ──────────────────────────────────
# 每个用例: (意图, 期望的阶段动作映射, 描述)
TEST_CASES = [
    # 1. 问候 → SKIP ACT+OBSERVE, DEGRADED REFLECT
    {
        "intent": "你好",
        "expect": {"ACT": "SKIP", "OBSERVE": "SKIP", "REFLECT": "DEGRADED"},
        "desc": "简单问候：跳过执行和观察，轻量反思",
    },
    # 2. 文件操作 → 全RUN
    {
        "intent": "帮我读取文件test.py",
        "expect": {"PLAN": "RUN", "ACT": "RUN", "OBSERVE": "RUN", "REFLECT": "RUN"},
        "desc": "文件操作：全阶段激活",
    },
    # 3. 代码操作 → 全RUN
    {
        "intent": "写一个Python脚本",
        "expect": {"PLAN": "RUN", "ACT": "RUN", "OBSERVE": "RUN", "REFLECT": "RUN"},
        "desc": "代码操作：全阶段激活",
    },
    # 4. 搜索查询 → DEGRADED OBSERVE
    {
        "intent": "帮我找一下天气",
        "expect": {"OBSERVE": "DEGRADED"},
        "desc": "搜索查询：轻量结果检查",
    },
    # 5. 高风险操作 → 全RUN（安全优先）
    {
        "intent": "删除所有数据sudo rm -rf",
        "expect": {"PLAN": "RUN", "ACT": "RUN", "OBSERVE": "RUN", "REFLECT": "RUN"},
        "desc": "高风险操作：全阶段激活（安全优先）",
    },
    # 6. 中文问候 → 同英文
    {
        "intent": "早",
        "expect": {"ACT": "SKIP", "OBSERVE": "SKIP"},
        "desc": "中文单字问候：跳过执行和观察",
    },
    # 7. 重复输入 → SKIP PLAN
    {
        "intent": "你好",
        "expect": {},  # 需要连续2次才触发
        "desc": "重复输入（需设置consecutive_identical>=2）",
        "setup": {"consecutive_identical": 2},
    },
    # 8. 高上下文 → DEGRADED REFLECT
    {
        "intent": "继续",
        "expect": {"REFLECT": "DEGRADED"},
        "desc": "上下文>60%：轻量反思",
        "setup": {"context_used_pct": 0.7},
    },
    # 9. 混合意图：搜索+代码 → 全RUN（代码优先）
    {
        "intent": "搜索Python教程并运行代码",
        "expect": {"PLAN": "RUN", "ACT": "RUN", "OBSERVE": "RUN", "REFLECT": "RUN"},
        "desc": "混合意图：代码优先，全RUN",
    },
    # 10. 未知意图 → 默认全RUN
    {
        "intent": "今天天气怎么样适合出门吗",
        "expect": {"PLAN": "RUN", "ACT": "RUN", "OBSERVE": "RUN", "REFLECT": "RUN"},
        "desc": "未知意图：默认全RUN",
    },
]


def run_regression():
    """运行回归测试。"""
    router = RuleRouter()
    passed = 0
    failed = 0
    results = []

    for i, tc in enumerate(TEST_CASES):
        intent = tc["intent"]
        expect = tc["expect"]
        desc = tc["desc"]
        setup = tc.get("setup", {})

        ctx = RoutingContext(
            user_input=intent,
            turn_count=i + 1,
            context_used_pct=setup.get("context_used_pct", 0.0),
            consecutive_identical=setup.get("consecutive_identical", 0),
        )
        decisions = router.route(ctx)
        actions = {d.phase: d.action.name for d in decisions}

        # 检查期望
        ok = True
        mismatches = []
        for phase, expected_action in expect.items():
            actual = actions.get(phase, "MISSING")
            if actual != expected_action:
                ok = False
                mismatches.append(f"{phase}: expected={expected_action}, got={actual}")

        if ok:
            passed += 1
            status = "✅"
        else:
            failed += 1
            status = "❌"

        results.append({
            "case": i + 1,
            "intent": intent,
            "desc": desc,
            "status": status,
            "mismatches": mismatches,
            "actual": actions,
        })

    return results, passed, failed


def main():
    """主函数——运行测试并输出报告。"""
    results, passed, failed = run_regression()

    print("=" * 60)
    print("路由回归测试套件 · 10标准用例")
    print("=" * 60)

    for r in results:
        print(f"\n  {r['status']} Case {r['case']}: {r['desc']}")
        print(f"     意图: {r['intent']}")
        if r["mismatches"]:
            for m in r["mismatches"]:
                print(f"     ❌ {m}")
        else:
            actions_str = ", ".join(f"{k}={v}" for k, v in r["actual"].items())
            print(f"     决策: {actions_str}")

    print("\n" + "=" * 60)
    print(f"结果: {passed}/{passed + failed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
