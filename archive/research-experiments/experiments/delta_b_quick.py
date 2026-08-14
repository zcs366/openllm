#!/usr/bin/env python3
"""ΔB裁决实验精简版：10个任务，快速验证记忆注入是否改变Agent行为。"""
import json, time, requests
from difflib import SequenceMatcher

MODEL = "qwen3.5:9b"
OLLAMA_URL = "http://localhost:11434/api/chat"
TIMEOUT = 60

# 10个任务：问题 + 记忆文本 + 期望方向
TASKS = [
    {"id": 1, "cat": "fact_pref", "q": "你推荐什么编程语言？", "mem": "用户偏好Python，讨厌Java的冗长语法。", "expect": "python"},
    {"id": 2, "cat": "fact_pref", "q": "数据库选哪个？", "mem": "用户之前选择了PostgreSQL，因为支持JSONB。", "expect": "postgres"},
    {"id": 3, "cat": "tech_dec", "q": "API设计用什么风格？", "mem": "用户团队统一用RESTful，不用GraphQL。", "expect": "rest"},
    {"id": 4, "cat": "tech_dec", "q": "前端框架选什么？", "mem": "用户决定用Vue而不是React，因为学习曲线低。", "expect": "vue"},
    {"id": 5, "cat": "user_hab", "q": "请解释一下Docker是什么。", "mem": "用户习惯简洁回答，不要超过3句话。", "expect": "short"},
    {"id": 6, "cat": "user_hab", "q": "什么是微服务？", "mem": "用户偏好用类比解释技术概念。", "expect": "analogy"},
    {"id": 7, "cat": "fact_pref", "q": "部署到哪里？", "mem": "用户选择了AWS而不是GCP，因为团队熟悉AWS。", "expect": "aws"},
    {"id": 8, "cat": "tech_dec", "q": "测试框架用什么？", "mem": "用户坚持用pytest，不用unittest。", "expect": "pytest"},
    {"id": 9, "cat": "user_hab", "q": "解释一下机器学习。", "mem": "用户是初学者，需要通俗易懂的解释。", "expect": "simple"},
    {"id": 10, "cat": "fact_pref", "q": "版本控制用什么？", "mem": "用户只用Git，不用SVN。", "expect": "git"},
]

def call_llm(messages):
    """调用ollama LLM"""
    import time
    time.sleep(2)  # 请求间隔2秒，避免模型过载
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL, "messages": messages, "stream": False,
            "options": {"temperature": 0.3, "think": False, "num_predict": 100}
        }, timeout=120)  # 超时120秒
        content = resp.json()["message"]["content"]
        if not content:
            print(f"    ⚠ 空响应")
        return content or ""
    except Exception as e:
        print(f"    ⚠ 错误: {e}")
        return ""

def compute_delta_b(text_a, text_b):
    """计算归一化编辑距离作为ΔB"""
    if not text_a and not text_b:
        return 0.0  # 两个都空=无行为改变
    if not text_a or not text_b:
        return 0.0  # 一个空=模型失败，不算行为改变
    return 1.0 - SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()

def run_experiment():
    """运行ΔB实验"""
    print(f"ΔB裁决实验精简版 | 模型: {MODEL} | 任务数: {len(TASKS)}")
    print("=" * 60)
    
    results = []
    for task in TASKS:
        # 对照组：无记忆
        ctrl_msgs = [{"role": "user", "content": task["q"]}]
        ctrl_out = call_llm(ctrl_msgs)
        
        # 实验组：有记忆注入
        exp_msgs = [
            {"role": "system", "content": f"以下是关于用户的历史记忆：{task['mem']}"},
            {"role": "user", "content": task["q"]}
        ]
        exp_out = call_llm(exp_msgs)
        
        delta_b = compute_delta_b(ctrl_out, exp_out)
        results.append({
            "id": task["id"], "cat": task["cat"],
            "delta_b": round(delta_b, 4),
            "ctrl_len": len(ctrl_out), "exp_len": len(exp_out),
            "significant": delta_b > 0.1
        })
        
        status = "✅" if delta_b > 0.2 else ("⚠️" if delta_b > 0.1 else "❌")
        print(f"  {status} Task {task['id']:2d} [{task['cat']:9s}] ΔB={delta_b:.3f}")
    
    # 汇总统计
    delta_bs = [r["delta_b"] for r in results]
    mean_db = sum(delta_bs) / len(delta_bs)
    std_db = (sum((d - mean_db)**2 for d in delta_bs) / len(delta_bs)) ** 0.5
    effective = sum(1 for d in delta_bs if d > 0.2)
    
    report = {
        "total_tasks": len(TASKS),
        "delta_b_mean": round(mean_db, 4),
        "delta_b_std": round(std_db, 4),
        "effective_count": effective,
        "threshold_20pct": "PASS" if mean_db >= 0.2 else "FAIL",
        "per_task": results,
        "conclusion": "Memory injection EFFECTIVE" if mean_db >= 0.2 else "Memory injection INEFFECTIVE"
    }
    
    # 按类别统计
    for cat in ["fact_pref", "tech_dec", "user_hab"]:
        cat_results = [r["delta_b"] for r in results if r["cat"] == cat]
        if cat_results:
            cat_mean = sum(cat_results) / len(cat_results)
            print(f"  {cat}: mean={cat_mean:.3f}")
    
    print("=" * 60)
    print(f"ΔB总均值: {mean_db:.4f} (阈值0.20 → {'PASS ✅' if mean_db >= 0.2 else 'FAIL ❌'})")
    print(f"有效任务: {effective}/{len(TASKS)}")
    print(f"结论: {report['conclusion']}")
    
    # 保存报告
    with open("delta_b_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告: delta_b_report.json")
    
    return report

if __name__ == "__main__":
    run_experiment()
