"""
classifier.py — 统一三类数据分类器
铁律：三类数据三套规则，所有提取器共用此模块
消除包拯审计发现的规则不一致问题
"""
import re


# ═══════════════════════════════════════════════════════════════
# 规则1：关系信号（用户纠正/偏好/决策风格/称呼）
# ═══════════════════════════════════════════════════════════════
RELATION_PATTERNS = [
    # 用户纠正（最强信号）
    r"不[可以许]", r"错了", r"不对", r"别[这样那样]", r"停",
    r"我说的不是", r"你理解错了", r"重新来", r"不要这样",
    # 偏好表达
    r"我喜欢", r"我讨厌", r"我习惯", r"我不喜欢", r"偏好",
    r"以后.{0,5}要", r"从今以后", r"记住这个",
    # 决策风格
    r"你觉得", r"你的判断", r"你来分析", r"你来决定",
    r"先.{0,10}后", r"优先", r"不重要", r"必须",
    # 关系称呼
    r"老搭档", r"军师", r"老铁",
]

# ═══════════════════════════════════════════════════════════════
# 规则2：技术insight（架构/原理/发现/方法论）
# ═══════════════════════════════════════════════════════════════
TECHNICAL_PATTERNS = [
    # 架构与设计
    r"架构", r"设计模式", r"方案.{0,5}选择", r"原理", r"机制",
    # 算法与模型
    r"算法", r"模型.{0,5}(训练|推理|优化)", r"参数", r"权重",
    # 发现与验证
    r"发现", r"验证.{0,5}(了|结果)", r"实验.{0,5}(结果|证明)",
    r"结论.{0,5}(是|表明)", r"论文.{0,5}(说|表明|提出)",
    # 技术术语
    r"(attention|transformer|embedding|minhash|faiss|bge|sqlite|ollama)",
    r"(D0|ICA|ISA|IO-S|ISN|IKO|QLoRA|LoRA|GGUF)",
]

# ═══════════════════════════════════════════════════════════════
# 规则3：工程决策（任务/优先级/交付/审计）
# ═══════════════════════════════════════════════════════════════
ENGINEERING_PATTERNS = [
    # 优先级与任务
    r"P[0-2]", r"里程碑", r"交付", r"部署", r"上线",
    # 审计与质量
    r"测试.{0,5}(通过|完成)", r"审计", r"修复", r"重构",
    # 流程与规范
    r"PAL", r"任务书", r"执行报告", r"三碑",
    r"优先级", r"工期", r"截止",
    # 完成状态
    r"完成", r"通过", r"验收", r"落地",
]

# 三套规则映射
RULE_SETS = {
    "relation": RELATION_PATTERNS,
    "technical": TECHNICAL_PATTERNS,
    "engineering": ENGINEERING_PATTERNS,
}


def classify(text: str) -> str:
    """
    三套独立规则分类（铁律：不合并为一套）
    每类独立计分，取最高分
    """
    scores = {}
    for doc_type, patterns in RULE_SETS.items():
        hits = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        scores[doc_type] = hits

    max_score = max(scores.values())
    if max_score == 0:
        return "unknown"

    for doc_type, score in scores.items():
        if score == max_score:
            return doc_type
    return "unknown"


def classify_with_confidence(text: str) -> tuple:
    """
    分类+置信度
    返回: (doc_type, confidence, scores_dict)
    confidence = winner_score / total_hits (0-1)
    """
    scores = {}
    for doc_type, patterns in RULE_SETS.items():
        scores[doc_type] = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))

    total = sum(scores.values())
    if total == 0:
        return "unknown", 0.0, scores

    winner = max(scores, key=lambda k: scores[k])
    confidence = scores[winner] / total if total > 0 else 0.0
    return winner, confidence, scores


if __name__ == "__main__":
    tests = [
        ("用户说不可以讨好我，记住这个偏好。以后评估要真实严厉。", "relation"),
        ("架构采用attention机制，论文发现D0从13降到2。", "technical"),
        ("P0任务完成，PAL交付通过审计，测试全过。", "engineering"),
        ("今天天气不错", "unknown"),
    ]
    for text, expected in tests:
        result, conf, scores = classify_with_confidence(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{text[:30]}...' → {result} (conf={conf:.2f}) expected={expected}")
