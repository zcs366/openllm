#!/usr/bin/env python3
"""ΔB裁决实验v3：用/api/generate+raw绕过思考模式"""
import json, time, requests
from difflib import SequenceMatcher

MODEL = "qwen3.5:9b"
API = "http://localhost:11434/api/generate"

TASKS = [
    {"id":1,"cat":"fact_pref","q":"推荐什么编程语言？","mem":"用户偏好Python，讨厌Java。"},
    {"id":2,"cat":"fact_pref","q":"数据库选哪个？","mem":"用户选择了PostgreSQL。"},
    {"id":3,"cat":"tech_dec","q":"API设计风格？","mem":"用户团队用RESTful。"},
    {"id":4,"cat":"tech_dec","q":"前端框架选什么？","mem":"用户决定用Vue不是React。"},
    {"id":5,"cat":"user_hab","q":"解释Docker。","mem":"用户要简洁回答，不超过3句。"},
    {"id":6,"cat":"user_hab","q":"什么是微服务？","mem":"用户偏好用类比解释。"},
    {"id":7,"cat":"fact_pref","q":"部署到哪里？","mem":"用户选AWS不是GCP。"},
    {"id":8,"cat":"tech_dec","q":"测试框架用什么？","mem":"用户坚持pytest。"},
    {"id":9,"cat":"user_hab","q":"解释机器学习。","mem":"用户是初学者，要通俗。"},
    {"id":10,"cat":"fact_pref","q":"版本控制用什么？","mem":"用户只用Git。"},
]

def ask(prompt):
    """用/api/generate+raw调用，绕过思考模式"""
    time.sleep(1)
    try:
        resp = requests.post(API, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {"temperature": 0.3, "num_predict": 80}
        }, timeout=90)
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  ⚠ {e}")
        return ""

def delta_b(a, b):
    if not a and not b: return 0.0
    if not a or not b: return 0.0
    return 1.0 - SequenceMatcher(None, a.lower(), b.lower()).ratio()

print(f"ΔB实验v3 | {MODEL} | {len(TASKS)}任务")
print("="*50)

results = []
for t in TASKS:
    ctrl = ask(f"回答：{t['q']}。一个词或一句话。")
    exp = ask(f"关于用户的历史记忆：{t['mem']}\n\n回答：{t['q']}。一个词或一句话。")
    db = delta_b(ctrl, exp)
    results.append({"id":t["id"],"cat":t["cat"],"delta_b":round(db,3),
                     "ctrl":ctrl[:30],"exp":exp[:30]})
    icon = "✅" if db > 0.2 else "⚠️" if db > 0.1 else "❌"
    print(f"  {icon} T{t['id']:2d} ΔB={db:.3f} | Ctrl={ctrl[:25]} | Exp={exp[:25]}")

dbs = [r["delta_b"] for r in results]
mean = sum(dbs)/len(dbs)
eff = sum(1 for d in dbs if d > 0.2)

report = {
    "total": len(TASKS), "delta_b_mean": round(mean,4),
    "effective": eff, "pass": mean >= 0.2,
    "conclusion": "EFFECTIVE" if mean >= 0.2 else "INEFFECTIVE",
    "per_task": results
}

print("="*50)
print(f"ΔB={mean:.3f} (阈值0.2 → {'PASS ✅' if mean>=0.2 else 'FAIL ❌'})")
print(f"有效: {eff}/{len(TASKS)}")
print(f"结论: {report['conclusion']}")

with open("delta_b_report.json","w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"报告: delta_b_report.json")
