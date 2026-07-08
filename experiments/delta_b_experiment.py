#!/usr/bin/env python3
"""
Delta-B 实验：记忆注入对Agent行为变化率的度量
=================================================
实验目标：测量ISA记忆系统注入上下文后，LLM回答是否发生显著变化。
核心指标：delta-B = 归一化编辑距离（control vs memory-injected回答）
判定标准：delta-B均值 ≥ 20% 视为记忆注入有效（PASS）

运行方式：python3 delta_b_experiment.py
依赖：requests, ollama (localhost:11434), 模型 gemma4:12b
"""

import json
import math
import time
import sys
from datetime import datetime
from typing import List, Dict, Any, Tuple

try:
    import requests
except ImportError:
    print("错误：需要 requests 库。请运行：pip install requests")
    sys.exit(1)

# ============================================================
# 全局配置
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:9b"
TIMEOUT = 120  # 秒，单次LLM调用超时
MAX_RETRIES = 2  # 失败重试次数
THRESHOLD = 0.20  # delta-B 有效性阈值（20%）


# ============================================================
# 30组实验数据：问题 + 记忆文本 + 预期变化方向
# ============================================================
TASK_TRIPLES: List[Dict[str, str]] = [
    # ---- 事实偏好 (fact_pref) x10 ----
    {"category": "fact_pref", "question": "推荐一个数据库给Web项目用",
     "memory": "用户明确偏好PostgreSQL而非MySQL，之前多次在项目中使用PG。",
     "expected": "应推荐PostgreSQL"},
    {"category": "fact_pref", "question": "前端框架选哪个好？",
     "memory": "用户偏好Vue.js，对React不感兴趣，说Vue更轻量。",
     "expected": "应推荐Vue.js"},
    {"category": "fact_pref", "question": "写文档用什么格式",
     "memory": "用户始终坚持使用Markdown，拒绝Word和LaTeX。",
     "expected": "应推荐Markdown"},
    {"category": "fact_pref", "question": "代码编辑器推荐",
     "memory": "用户是VS Code重度用户，明确表示不喜欢Vim/Emacs。",
     "expected": "应推荐VS Code"},
    {"category": "fact_pref", "question": "部署用什么方案",
     "memory": "用户偏好Docker+Kubernetes，不用AWS Lambda之类的serverless。",
     "expected": "应推荐Docker+K8s"},
    {"category": "fact_pref", "question": "版本控制工具推荐",
     "memory": "用户坚持用Git，对SVN表示不满。",
     "expected": "应推荐Git"},
    {"category": "fact_pref", "question": "日志分析工具用什么",
     "memory": "用户喜欢ELK stack（Elasticsearch+Logstash+Kibana），不用Splunk。",
     "expected": "应推荐ELK"},
    {"category": "fact_pref", "question": "API文档工具推荐",
     "memory": "用户偏好Swagger/OpenAPI，不用Postman做文档。",
     "expected": "应推荐Swagger"},
    {"category": "fact_pref", "question": "CI/CD工具推荐",
     "memory": "用户使用GitHub Actions，不用Jenkins。",
     "expected": "应推荐GitHub Actions"},
    {"category": "fact_pref", "question": "监控告警平台推荐",
     "memory": "用户偏好Prometheus+Grafana，不考虑Datadog（太贵）。",
     "expected": "应推荐Prometheus+Grafana"},

    # ---- 技术决策 (tech_decision) x10 ----
    {"category": "tech_decision", "question": "这段代码用什么语言写比较好",
     "memory": "用户所在项目统一使用Python，团队没有Java经验。",
     "expected": "应推荐Python"},
    {"category": "tech_decision", "question": "异步任务队列怎么选",
     "memory": "用户项目已用Celery+Redis，不想引入新的队列系统。",
     "expected": "应推荐继续用Celery"},
    {"category": "tech_decision", "question": "ORM框架推荐",
     "memory": "用户项目使用SQLAlchemy，团队熟悉其API。",
     "expected": "应推荐SQLAlchemy"},
    {"category": "tech_decision", "question": "测试框架用什么",
     "memory": "用户坚持用pytest，不用unittest。",
     "expected": "应推荐pytest"},
    {"category": "tech_decision", "question": "缓存方案怎么设计",
     "memory": "用户项目已有Redis集群，希望复用而非引入Memcached。",
     "expected": "应推荐基于Redis的方案"},
    {"category": "tech_decision", "question": "Web服务器选型",
     "memory": "用户项目用Gunicorn+Uvicorn混合模式，不要Nginx直接接Python。",
     "expected": "应推荐Gunicorn+Uvicorn"},
    {"category": "tech_decision", "question": "数据库迁移工具推荐",
     "memory": "用户项目用Alembic管理数据库迁移，schema变更都通过它。",
     "expected": "应推荐Alembic"},
    {"category": "tech_decision", "question": "序列化格式用什么",
     "memory": "用户项目统一用JSON，不用XML和YAML做数据交换。",
     "expected": "应推荐JSON"},
    {"category": "tech_decision", "question": "静态类型检查工具",
     "memory": "用户项目用mypy做类型检查，已集成到CI流程。",
     "expected": "应推荐mypy"},
    {"category": "tech_decision", "question": "包管理器推荐",
     "memory": "用户偏好Poetry，已抛弃pip+virtualenv。",
     "expected": "应推荐Poetry"},

    # ---- 用户习惯 (user_habit) x10 ----
    {"category": "user_habit", "question": "帮我写一段排序算法的代码",
     "memory": "用户习惯要求简洁代码，不要冗长注释，偏好单文件可运行脚本。",
     "expected": "应给出简洁代码"},
    {"category": "user_habit", "question": "解释一下什么是递归",
     "memory": "用户喜欢用类比方式解释概念，不要纯学术定义。",
     "expected": "应使用类比解释"},
    {"category": "user_habit", "question": "帮我写个Python脚本处理CSV",
     "memory": "用户习惯写完代码后追问优化建议，喜欢性能对比数据。",
     "expected": "应提及性能优化"},
    {"category": "user_habit", "question": "分析这个错误日志",
     "memory": "用户希望回答先给结论再给分析，不要从头到尾流水账。",
     "expected": "应先给结论"},
    {"category": "user_habit", "question": "设计一个用户认证系统",
     "memory": "用户喜欢架构图和流程图辅助说明，不要纯文字。",
     "expected": "应包含图示建议"},
    {"category": "user_habit", "question": "帮我review这段代码",
     "memory": "用户希望按严重程度排列问题，先致命错误后风格问题。",
     "expected": "应按严重程度排列"},
    {"category": "user_habit", "question": "写个项目README",
     "memory": "用户偏好英文README，带badges和目录结构。",
     "expected": "应写英文README"},
    {"category": "user_habit", "question": "帮我调试一个bug",
     "memory": "用户希望给出step-by-step调试步骤，不要直接给答案。",
     "expected": "应给出调试步骤"},
    {"category": "user_habit", "question": "推荐学习资料",
     "memory": "用户偏好视频教程胜过书籍，喜欢B站和YouTube资源。",
     "expected": "应推荐视频资源"},
    {"category": "user_habit", "question": "帮我写单元测试",
     "memory": "用户喜欢测试覆盖率数据和edge case覆盖分析。",
     "expected": "应提及覆盖率和edge case"},
]


# ============================================================
# 工具函数
# ============================================================

def call_ollama(prompt: str, system: str = "") -> str:
    """调用Ollama API生成回答。返回文本，失败返回空字符串。"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "num_predict": 512},
    }
    if system:
        payload["system"] = system

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.exceptions.Timeout:
            print(f"  ⚠ 超时（尝试 {attempt+1}/{MAX_RETRIES+1}）")
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Ollama连接失败，请确保服务已启动 (localhost:11434)")
            return ""
        except Exception as e:
            print(f"  ⚠ 异常: {e}")
        if attempt < MAX_RETRIES:
            time.sleep(2)
    return ""


def normalize_edit_distance(s1: str, s2: str) -> float:
    """
    计算归一化编辑距离（Levenshtein）。
    返回值在 [0.0, 1.0] 之间：0=完全相同，1=完全不同。
    用于度量两个回答的差异程度。
    """
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    # 简化：按字符计算（对中文更公平）
    c1, c2 = list(s1), list(s2)
    n, m = len(c1), len(c2)
    # DP矩阵
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if c1[i-1] == c2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    max_len = max(n, m)
    return dp[n][m] / max_len if max_len > 0 else 0.0


def welch_t_test(control_vals: List[float], experiment_vals: List[float]) -> float:
    """
    Welch's t-test: 检验两组delta-B均值是否有显著差异。
    返回近似p值（用正态近似，不依赖scipy）。
    """
    n1, n2 = len(control_vals), len(experiment_vals)
    if n1 < 2 or n2 < 2:
        return 1.0
    m1 = sum(control_vals) / n1
    m2 = sum(experiment_vals) / n2
    var1 = sum((x - m1) ** 2 for x in control_vals) / (n1 - 1)
    var2 = sum((x - m2) ** 2 for x in experiment_vals) / (n2 - 1)
    se = math.sqrt(var1 / n1 + var2 / n2) if (var1 / n1 + var2 / n2) > 0 else 1e-10
    t_stat = abs(m1 - m2) / se
    # 正态近似p值（|Z| > t_stat 的尾概率）
    # 简化：用1-Φ近似
    x = t_stat / math.sqrt(2)
    p = math.exp(-x * x) * (1 - 0.5 * x) if x > 0 else 1.0
    return max(0.0, min(1.0, p))


# ============================================================
# 主实验流程
# ============================================================

def run_experiment() -> Dict[str, Any]:
    """执行完整delta-B实验，返回结构化报告。"""
    print("=" * 60)
    print("  Delta-B 实验：记忆注入行为变化率度量")
    print(f"  模型: {MODEL} | 任务数: {len(TASK_TRIPLES)}")
    print(f"  阈值: {THRESHOLD*100:.0f}% | 时间: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)

    per_task_results = []
    all_control_lens = []
    all_exp_lens = []
    effective_count = 0

    for idx, task in enumerate(TASK_TRIPLES):
        tid = idx + 1
        cat = task["category"]
        q = task["question"]
        mem = task["memory"]
        expected = task["expected"]

        print(f"\n[{tid:2d}/30] [{cat}] {q}")

        # 对照组：无记忆注入
        ctrl_prompt = f"请回答以下问题，给出具体建议：\n{q}"
        ctrl_resp = call_ollama(ctrl_prompt)
        if not ctrl_resp:
            print("  ✗ 对照组失败，跳过")
            per_task_results.append({"task_id": tid, "category": cat,
                "delta_b": 0.0, "significant": False, "status": "skipped"})
            continue

        # 实验组：注入记忆上下文
        exp_prompt = (
            f"请根据以下记忆上下文回答问题。记忆上下文反映了用户的历史偏好和决策：\n\n"
            f"[记忆上下文] {mem}\n\n"
            f"请回答以下问题，给出具体建议：\n{q}"
        )
        exp_resp = call_ollama(exp_prompt)
        if not exp_resp:
            print("  ✗ 实验组失败，跳过")
            per_task_results.append({"task_id": tid, "category": cat,
                "delta_b": 0.0, "significant": False, "status": "skipped"})
            continue

        # 计算归一化编辑距离作为delta-B
        delta_b = normalize_edit_distance(ctrl_resp, exp_resp)
        significant = delta_b >= THRESHOLD

        if significant:
            effective_count += 1

        all_control_lens.append(len(ctrl_resp))
        all_exp_lens.append(len(exp_resp))

        status_mark = "✓" if significant else "○"
        print(f"  {status_mark} delta-B = {delta_b:.3f} | "
              f"对照 {len(ctrl_resp)}字 vs 实验 {len(exp_resp)}字")

        per_task_results.append({
            "task_id": tid,
            "category": cat,
            "delta_b": round(delta_b, 4),
            "significant": significant,
            "status": "ok",
            "expected": expected,
            "control_preview": ctrl_resp[:80] + "..." if len(ctrl_resp) > 80 else ctrl_resp,
            "experiment_preview": exp_resp[:80] + "..." if len(exp_resp) > 80 else exp_resp,
        })

    # ---- 汇总统计 ----
    valid = [r for r in per_task_results if r["status"] == "ok"]
    if not valid:
        print("\n❌ 无有效结果，Ollama服务可能不可用")
        return {"total_tasks": len(TASK_TRIPLES), "delta_b_mean": 0.0,
                "delta_b_std": 0.0, "p_value": 1.0, "effective_count": 0,
                "threshold_20pct": "FAIL", "per_task": per_task_results,
                "conclusion": "Memory injection INEFFECTIVE (no data)"}

    delta_bs = [r["delta_b"] for r in valid]
    mean_db = sum(delta_bs) / len(delta_bs)
    std_db = math.sqrt(sum((d - mean_db) ** 2 for d in delta_bs) / max(len(delta_bs)-1, 1))

    # 分类别统计
    cat_stats: Dict[str, List[float]] = {}
    for r in valid:
        cat = r["category"]
        cat_stats.setdefault(cat, []).append(r["delta_b"])

    # p值：对照组与实验组回答长度差异的显著性
    p_value = welch_t_test(all_control_lens, all_exp_lens)

    threshold_pass = "PASS" if mean_db >= THRESHOLD else "FAIL"
    conclusion = "Memory injection EFFECTIVE" if threshold_pass == "PASS" else "Memory injection INEFFECTIVE"

    # ---- 输出报告 ----
    report = {
        "total_tasks": len(TASK_TRIPLES),
        "effective_tasks": len(valid),
        "delta_b_mean": round(mean_db, 4),
        "delta_b_std": round(std_db, 4),
        "p_value": round(p_value, 4),
        "effective_count": effective_count,
        "threshold_20pct": threshold_pass,
        "category_stats": {k: round(sum(v)/len(v), 4) for k, v in cat_stats.items()},
        "per_task": per_task_results,
        "conclusion": conclusion,
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
    }

    # 打印摘要
    print("\n" + "=" * 60)
    print("  实验结果摘要")
    print("=" * 60)
    print(f"  有效任务数: {len(valid)}/{len(TASK_TRIPLES)}")
    print(f"  Delta-B 均值: {mean_db:.4f}  标准差: {std_db:.4f}")
    print(f"  有效任务 (delta-B ≥ {THRESHOLD*100:.0f}%): {effective_count}/{len(valid)}")
    print(f"  阈值判定: {threshold_pass}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  分类均值: ", end="")
    for cat, vals in cat_stats.items():
        avg = sum(vals) / len(vals)
        print(f"{cat}={avg:.3f}  ", end="")
    print()
    print(f"  结论: {conclusion}")
    print("=" * 60)

    return report


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    # 先检查Ollama是否可用
    print("检查Ollama服务...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(MODEL in m for m in models):
            print(f"⚠ 模型 {MODEL} 未找到。可用模型: {models}")
            print(f"  请先运行: ollama pull {MODEL}")
            sys.exit(1)
        print(f"✓ Ollama就绪，模型 {MODEL} 可用")
    except Exception as e:
        print(f"❌ 无法连接Ollama: {e}")
        print("  请确保: ollama serve 已启动")
        sys.exit(1)

    report = run_experiment()

    # 保存JSON报告
    out_path = "/home/zcs/projects/openllm/experiments/delta_b_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 报告已保存: {out_path}")
