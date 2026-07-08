#!/usr/bin/env python3
"""
跨语言翻译对收集脚本

收集英文-中文翻译对，用于AGFT应用实验。
"""

import json
import os
from pathlib import Path

# 数据目录
DATA_DIR = Path(__file__).parent.parent / "data" / "cross_lingual"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 跨语言翻译对
TRANSLATION_PAIRS = [
    # 技术领域
    {
        "en": "The algorithm optimizes the performance of the system.",
        "zh": "该算法优化了系统的性能。",
        "domain": "technology",
        "difficulty": "medium",
    },
    {
        "en": "Machine learning models require large amounts of training data.",
        "zh": "机器学习模型需要大量的训练数据。",
        "domain": "technology",
        "difficulty": "medium",
    },
    {
        "en": "The neural network architecture consists of multiple layers.",
        "zh": "神经网络架构由多个层组成。",
        "domain": "technology",
        "difficulty": "medium",
    },
    {
        "en": "Deep learning has revolutionized natural language processing.",
        "zh": "深度学习彻底改变了自然语言处理。",
        "domain": "technology",
        "difficulty": "medium",
    },
    {
        "en": "The transformer model uses self-attention mechanisms.",
        "zh": "Transformer模型使用自注意力机制。",
        "domain": "technology",
        "difficulty": "hard",
    },
    
    # 科学领域
    {
        "en": "The experiment results confirm the hypothesis.",
        "zh": "实验结果证实了假设。",
        "domain": "science",
        "difficulty": "medium",
    },
    {
        "en": "Quantum computing promises to solve complex problems.",
        "zh": "量子计算有望解决复杂问题。",
        "domain": "science",
        "difficulty": "hard",
    },
    {
        "en": "The research team published their findings in a prestigious journal.",
        "zh": "研究团队在权威期刊上发表了他们的发现。",
        "domain": "science",
        "difficulty": "medium",
    },
    {
        "en": "Climate change poses significant challenges to global ecosystems.",
        "zh": "气候变化对全球生态系统构成了重大挑战。",
        "domain": "science",
        "difficulty": "hard",
    },
    {
        "en": "The new drug shows promising results in clinical trials.",
        "zh": "这种新药在临床试验中显示出有希望的结果。",
        "domain": "science",
        "difficulty": "medium",
    },
    
    # 商业领域
    {
        "en": "The company reported strong quarterly earnings.",
        "zh": "该公司报告了强劲的季度收益。",
        "domain": "business",
        "difficulty": "medium",
    },
    {
        "en": "Market competition drives innovation and efficiency.",
        "zh": "市场竞争推动创新和效率。",
        "domain": "business",
        "difficulty": "medium",
    },
    {
        "en": "The startup secured significant venture capital funding.",
        "zh": "这家初创公司获得了大量的风险投资。",
        "domain": "business",
        "difficulty": "medium",
    },
    {
        "en": "Digital transformation is reshaping traditional industries.",
        "zh": "数字化转型正在重塑传统行业。",
        "domain": "business",
        "difficulty": "hard",
    },
    {
        "en": "Supply chain disruptions have impacted global trade.",
        "zh": "供应链中断影响了全球贸易。",
        "domain": "business",
        "difficulty": "medium",
    },
    
    # 日常生活
    {
        "en": "The weather forecast predicts rain for tomorrow.",
        "zh": "天气预报预测明天会下雨。",
        "domain": "daily",
        "difficulty": "easy",
    },
    {
        "en": "I enjoy reading books in my free time.",
        "zh": "我喜欢在空闲时间读书。",
        "domain": "daily",
        "difficulty": "easy",
    },
    {
        "en": "The restaurant serves delicious Italian cuisine.",
        "zh": "这家餐厅提供美味的意大利菜。",
        "domain": "daily",
        "difficulty": "easy",
    },
    {
        "en": "She goes to the gym every morning to exercise.",
        "zh": "她每天早上去健身房锻炼。",
        "domain": "daily",
        "difficulty": "easy",
    },
    {
        "en": "The movie received excellent reviews from critics.",
        "zh": "这部电影获得了评论家的好评。",
        "domain": "daily",
        "difficulty": "easy",
    },
    
    # 学术领域
    {
        "en": "The thesis presents a novel approach to solving the problem.",
        "zh": "论文提出了一种解决该问题的新方法。",
        "domain": "academic",
        "difficulty": "hard",
    },
    {
        "en": "The literature review covers recent developments in the field.",
        "zh": "文献综述涵盖了该领域的最新进展。",
        "domain": "academic",
        "difficulty": "hard",
    },
    {
        "en": "The methodology section describes the experimental design.",
        "zh": "方法论部分描述了实验设计。",
        "domain": "academic",
        "difficulty": "hard",
    },
    {
        "en": "The results are statistically significant at the 0.05 level.",
        "zh": "结果在0.05水平上具有统计显著性。",
        "domain": "academic",
        "difficulty": "hard",
    },
    {
        "en": "The conclusion summarizes the key findings and implications.",
        "zh": "结论总结了主要发现和意义。",
        "domain": "academic",
        "difficulty": "medium",
    },
]


def save_translation_pairs(pairs, filename="translation_pairs.json"):
    """
    保存翻译对到文件
    
    Args:
        pairs: 翻译对列表
        filename: 文件名
    """
    filepath = DATA_DIR / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 保存 {len(pairs)} 个翻译对到 {filepath}")
    return filepath


def load_translation_pairs(filename="translation_pairs.json"):
    """
    从文件加载翻译对
    
    Args:
        filename: 文件名
        
    Returns:
        list: 翻译对列表
    """
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        print(f"⚠️ 文件不存在: {filepath}")
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    print(f"✅ 加载 {len(pairs)} 个翻译对从 {filepath}")
    return pairs


def split_dataset(pairs, train_ratio=0.8):
    """
    划分数据集
    
    Args:
        pairs: 翻译对列表
        train_ratio: 训练集比例
        
    Returns:
        tuple: (训练集, 评估集)
    """
    import random
    random.seed(42)
    
    # 随机打乱
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    # 划分
    split_idx = int(len(shuffled) * train_ratio)
    train_set = shuffled[:split_idx]
    eval_set = shuffled[split_idx:]
    
    print(f"✅ 数据集划分: 训练集 {len(train_set)} 个, 评估集 {len(eval_set)} 个")
    return train_set, eval_set


def main():
    """主函数"""
    print("="*60)
    print("跨语言翻译对收集")
    print("="*60)
    
    # 保存翻译对
    save_translation_pairs(TRANSLATION_PAIRS)
    
    # 加载翻译对
    pairs = load_translation_pairs()
    
    # 划分数据集
    train_set, eval_set = split_dataset(pairs)
    
    # 保存训练集和评估集
    save_translation_pairs(train_set, "train_set.json")
    save_translation_pairs(eval_set, "eval_set.json")
    
    print("\n" + "="*60)
    print("✅ 跨语言翻译对收集完成！")
    print("="*60)


if __name__ == "__main__":
    main()