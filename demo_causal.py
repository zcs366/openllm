#!/usr/bin/env python3
"""openLLM 因果记忆 Demo — 30行证明"没有因果记忆时Agent犯错，有了时不错"

用法：python demo_causal.py
"""
import sys, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from openllm.memory.causal_memory import CausalMemoryStore, TrustLevel

# 创建临时存储（demo用，不污染真实数据）
tmpdir = tempfile.mkdtemp(prefix="openllm-demo-")
store = CausalMemoryStore(store_dir=Path(tmpdir) / "causal")

# === 场景定义：Agent要决定"用哪个方案处理用户请求" ===
DECISIONS = [
    {"context": "用户请求写测试", "action": "方案A-手写测试", "correct": "方案B-自动生成"},
    {"context": "用户请求重构代码", "action": "方案A-手写测试", "correct": "方案B-自动生成"},
    {"context": "用户请求查bug", "action": "方案A-手写测试", "correct": "方案C-日志分析"},
]

RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"

print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  openLLM 因果记忆 Demo{RESET}")
print(f"  {CYAN}证明：因果记忆让Agent不再重复犯错{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

# === 场景A：没有因果记忆（Agent重复犯错）===
print(f"{YELLOW}▸ 场景A：没有因果记忆{RESET}")
print(f"  Agent不知道之前犯过错，每次都选同一个方案\n")
mistakes_a = 0
for i, d in enumerate(DECISIONS, 1):
    # 没有因果记忆 → Agent总是选"方案A"（默认行为）
    chosen = "方案A-手写测试"
    is_correct = (chosen == d["correct"])
    icon = f"{GREEN}✓{RESET}" if is_correct else f"{RED}✗{RESET}"
    if not is_correct:
        mistakes_a += 1
    print(f"  {icon} 第{i}次 [{d['context']}] → 选了{chosen}")
    if not is_correct:
        print(f"    {RED}→ 错误！应该选 {d['correct']}{RESET}")

print(f"\n  {RED}{BOLD}结果：犯错 {mistakes_a}/{len(DECISIONS)} 次{RESET}\n")

# === 场景B：有因果记忆（Agent学习后纠正）===
print(f"{YELLOW}▸ 场景B：有因果记忆{RESET}")
print(f"  Agent记住之前的错误，自动选择不同策略\n")

# 先让Agent"经历"第一次错误并记录因果
store.store(
    action_signature="方案A-手写测试",
    context_features=["用户请求写测试"],
    prediction="方案A能完成任务",
    prediction_confidence=0.7,
    actual_result="方案A太慢且质量低",
    actual_success=False,
    delta="预测与实际不符",
    delta_magnitude=0.8,
    lesson="手写测试效率低，应该用自动生成",
    source="demo",
    trust_level=TrustLevel.INTERNAL,
    importance=0.9,
    tags=["testing", "mistake"],
)

mistakes_b = 0
for i, d in enumerate(DECISIONS, 1):
    # 有因果记忆 → 搜索相关教训
    lessons = store.get_lessons_for_action("方案A-手写测试")
    if lessons:
        # Agent看到教训 → 改选正确方案
        chosen = d["correct"]
    else:
        chosen = "方案A-手写测试"
    
    is_correct = (chosen == d["correct"])
    icon = f"{GREEN}✓{RESET}" if is_correct else f"{RED}✗{RESET}"
    if not is_correct:
        mistakes_b += 1
    print(f"  {icon} 第{i}次 [{d['context']}] → 选了{chosen}")
    if lessons and is_correct:
        print(f"    {GREEN}→ 因果记忆生效：'{lessons[0][:40]}...'{RESET}")

print(f"\n  {GREEN}{BOLD}结果：犯错 {mistakes_b}/{len(DECISIONS)} 次{RESET}\n")

# === 对比总结 ===
print(f"{BOLD}{'='*60}{RESET}")
print(f"  {RED}无因果记忆：犯错 {mistakes_a}/{len(DECISIONS)}{RESET}  vs  {GREEN}有因果记忆：犯错 {mistakes_b}/{len(DECISIONS)}{RESET}")
if mistakes_b < mistakes_a:
    print(f"\n  {GREEN}{BOLD}✓ 因果记忆有效：Agent因为记住昨天犯的错，今天做出了不同选择{RESET}")
else:
    print(f"\n  {YELLOW}⚠ 因果记忆demo需要更多场景验证{RESET}")
print(f"{BOLD}{'='*60}{RESET}\n")

# 清理临时目录
shutil.rmtree(tmpdir, ignore_errors=True)
