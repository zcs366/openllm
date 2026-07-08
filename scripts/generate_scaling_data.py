#!/usr/bin/env python3
"""生成翻译对数据集 - 5/100/500/1000/5000/10000规模"""
import json, random, hashlib
from pathlib import Path

random.seed(42)
DATA_DIR = Path(__file__).parent.parent / "data" / "cross_lingual"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 基础翻译模板（5个领域×20条 = 100条种子）
TEMPLATES = {
    "tech": [
        ("The algorithm improves system performance.", "该算法提升了系统性能。"),
        ("Machine learning requires large datasets.", "机器学习需要大量数据集。"),
        ("Neural networks process information in layers.", "神经网络分层处理信息。"),
        ("Deep learning has transformed AI research.", "深度学习改变了人工智能研究。"),
        ("The transformer model uses self-attention.", "Transformer模型使用自注意力机制。"),
        ("Cloud computing enables scalable services.", "云计算支持可扩展服务。"),
        ("Cybersecurity protects digital infrastructure.", "网络安全保护数字基础设施。"),
        ("Quantum computing solves complex problems.", "量子计算解决复杂问题。"),
        ("Edge computing reduces latency.", "边缘计算降低延迟。"),
        ("Blockchain ensures data integrity.", "区块链确保数据完整性。"),
        ("Natural language processing understands text.", "自然语言处理理解文本。"),
        ("Computer vision analyzes images.", "计算机视觉分析图像。"),
        ("Reinforcement learning optimizes decisions.", "强化学习优化决策。"),
        ("Generative models create new content.", "生成模型创造新内容。"),
        ("Data mining reveals hidden patterns.", "数据挖掘揭示隐藏模式。"),
        ("IoT connects everyday devices.", "物联网连接日常设备。"),
        ("5G networks enable faster communication.", "5G网络实现更快通信。"),
        ("Autonomous vehicles navigate without drivers.", "自动驾驶车辆无需驾驶员。"),
        ("Robotics automates physical tasks.", "机器人自动化物理任务。"),
        ("APIs connect different software systems.", "API连接不同软件系统。"),
    ],
    "science": [
        ("The experiment confirmed the hypothesis.", "实验验证了假设。"),
        ("Climate change affects global ecosystems.", "气候变化影响全球生态系统。"),
        ("The new drug shows promising results.", "新药显示出有希望的结果。"),
        ("Research teams publish in top journals.", "研究团队在顶级期刊发表。"),
        ("Scientific methodology ensures reproducibility.", "科学方法确保可重复性。"),
        ("Biodiversity is declining worldwide.", "全球生物多样性正在下降。"),
        ("Renewable energy reduces carbon emissions.", "可再生能源减少碳排放。"),
        ("Genetic engineering modifies organisms.", "基因工程改造生物体。"),
        ("Astronomy studies celestial objects.", "天文学研究天体。"),
        ("Physics explores fundamental forces.", "物理学探索基本力。"),
        ("Chemistry studies matter and its changes.", "化学研究物质及其变化。"),
        ("Biology examines living organisms.", "生物学研究生命体。"),
        ("Geology studies Earth's structure.", "地质学研究地球结构。"),
        ("Ecology examines environmental interactions.", "生态学研究环境互动。"),
        ("Aerospace engineering designs aircraft.", "航空航天工程设计飞行器。"),
        ("Materials science develops new materials.", "材料科学开发新材料。"),
        ("Oceanography studies marine environments.", "海洋学研究海洋环境。"),
        ("Paleontology examines ancient life.", "古生物学研究古代生命。"),
        ("Meteorology predicts weather patterns.", "气象学预测天气模式。"),
        ("Virology studies viruses.", "病毒学研究病毒。"),
    ],
    "business": [
        ("Market competition drives innovation.", "市场竞争推动创新。"),
        ("The startup raised venture capital.", "初创公司获得了风险投资。"),
        ("Digital transformation reshapes industries.", "数字化转型重塑行业。"),
        ("Supply chains require efficient logistics.", "供应链需要高效物流。"),
        ("Customer experience builds brand loyalty.", "客户体验建立品牌忠诚。"),
        ("Financial markets respond to economic data.", "金融市场响应经济数据。"),
        ("E-commerce grows rapidly worldwide.", "电子商务全球快速增长。"),
        ("Management practices improve productivity.", "管理实践提高生产力。"),
        ("Marketing strategies target specific audiences.", "营销策略针对特定受众。"),
        ("Corporate governance ensures accountability.", "公司治理确保问责。"),
        ("Human resources manages talent.", "人力资源管理人才。"),
        ("Strategic planning guides organizations.", "战略规划指导组织。"),
        ("Data analytics informs business decisions.", "数据分析为商业决策提供信息。"),
        ("International trade connects economies.", "国际贸易连接经济。"),
        ("Investment portfolios diversify risk.", "投资组合分散风险。"),
        ("Operational efficiency reduces costs.", "运营效率降低成本。"),
        ("Brand identity differentiates products.", "品牌识别区分产品。"),
        ("Customer feedback improves services.", "客户反馈改进服务。"),
        ("Revenue models generate income.", "收入模式产生收入。"),
        ("Competitive advantage sustains growth.", "竞争优势维持增长。"),
    ],
    "daily": [
        ("The weather is nice today.", "今天天气很好。"),
        ("I enjoy reading books.", "我喜欢读书。"),
        ("The restaurant serves good food.", "这家餐厅的食物很好。"),
        ("She goes to the gym regularly.", "她定期去健身房。"),
        ("The movie received good reviews.", "电影获得了好评。"),
        ("Traffic was heavy this morning.", "今天早上交通很拥挤。"),
        ("The concert was amazing.", "音乐会非常精彩。"),
        ("He cooks dinner every night.", "他每晚做晚饭。"),
        ("The park is beautiful in spring.", "公园在春天很美。"),
        ("She travels to different countries.", "她去不同国家旅行。"),
        ("The library has many books.", "图书馆有很多书。"),
        ("We walked along the beach.", "我们沿着海滩散步。"),
        ("The children played in the garden.", "孩子们在花园里玩耍。"),
        ("He drove to work today.", "他今天开车上班。"),
        ("The flowers bloom in summer.", "花在夏天开放。"),
        ("She painted a beautiful picture.", "她画了一幅美丽的画。"),
        ("The dog chased the ball.", "狗追着球跑。"),
        ("They visited their grandparents.", "他们看望了祖父母。"),
        ("The train arrived on time.", "火车准时到达。"),
        ("He learned to play guitar.", "他学会了弹吉他。"),
    ],
    "academic": [
        ("The thesis presents a novel approach.", "论文提出了一种新方法。"),
        ("The literature review covers recent work.", "文献综述涵盖了近期工作。"),
        ("The methodology section describes the design.", "方法论部分描述了设计。"),
        ("Results are statistically significant.", "结果具有统计显著性。"),
        ("The conclusion summarizes key findings.", "结论总结了主要发现。"),
        ("The abstract provides a brief overview.", "摘要提供了简要概述。"),
        ("Peer review ensures research quality.", "同行评审确保研究质量。"),
        ("The introduction sets the context.", "引言设定了背景。"),
        ("Data collection followed strict protocols.", "数据收集遵循严格协议。"),
        ("The discussion interprets the results.", "讨论解读了结果。"),
        ("Limitations are acknowledged honestly.", "局限性被诚实地承认。"),
        ("Future work is suggested.", "建议了未来工作。"),
        ("The framework enables systematic analysis.", "框架支持系统分析。"),
        ("Assumptions are clearly stated.", "假设被明确说明。"),
        ("The model predicts outcomes accurately.", "模型准确预测结果。"),
        ("The theory explains observed phenomena.", "理论解释观察到的现象。"),
        ("Validation confirms the approach.", "验证确认了方法。"),
        ("The study contributes to the field.", "研究对该领域有贡献。"),
        ("Methodology must be transparent.", "方法论必须透明。"),
        ("Citations support the arguments.", "引用支持论点。"),
    ],
}

def generate_dataset(n_samples):
    """生成指定规模的数据集"""
    # 从种子池扩展
    all_pairs = []
    for domain, pairs in TEMPLATES.items():
        for en, zh in pairs:
            all_pairs.append({"en": en, "zh": zh, "domain": domain})
    
    # 如果需要更多数据，通过变换生成
    variations = {
        "tech": [
            "The system uses {} to optimize {}.",
            "Modern {} enables efficient {}.",
            "The framework supports {} and {}.",
        ],
        "science": [
            "Research on {} shows {}.",
            "Studies confirm that {} affects {}.",
            "The discovery of {} impacts {}.",
        ],
        "business": [
            "The company focuses on {} and {}.",
            "Market trends indicate {} in {}.",
            "Investment in {} drives {}.",
        ],
        "daily": [
            "She enjoys {} during {}.",
            "The {} was great for {}.",
            "He practiced {} every {}.",
        ],
        "academic": [
            "The paper examines {} through {}.",
            "Analysis of {} reveals {}.",
            "The study investigates {} using {}.",
        ],
    }
    
    zh_variations = {
        "tech": [
            "该系统使用{}来优化{}。",
            "现代{}支持高效的{}。",
            "该框架支持{}和{}。",
        ],
        "science": [
            "关于{}的研究表明{}。",
            "研究证实{}影响{}。",
            "{}的发现影响{}。",
        ],
        "business": [
            "该公司专注于{}和{}。",
            "市场趋势显示{}在{}方面。",
            "对{}的投资推动{}。",
        ],
        "daily": [
            "她喜欢在{}期间{}。",
            "这个{}对{}来说很棒。",
            "他每天{}练习{}。",
        ],
        "academic": [
            "本文通过{}研究{}。",
            "对{}的分析揭示{}。",
            "该研究使用{}调查{}。",
        ],
    }
    
    # 主题词
    topics = {
        "tech": ["algorithms", "data processing", "software development", "AI", "cloud infrastructure"],
        "science": ["genetics", "climate", "evolution", "physics", "biology"],
        "business": ["innovation", "growth", "efficiency", "strategy", "management"],
        "daily": ["reading", "exercise", "cooking", "travel", "music"],
        "academic": ["methodology", "analysis", "theory", "evidence", "framework"],
    }
    
    zh_topics = {
        "tech": ["算法", "数据处理", "软件开发", "人工智能", "云基础设施"],
        "science": ["遗传学", "气候", "进化", "物理学", "生物学"],
        "business": ["创新", "增长", "效率", "战略", "管理"],
        "daily": ["阅读", "锻炼", "烹饪", "旅行", "音乐"],
        "academic": ["方法论", "分析", "理论", "证据", "框架"],
    }
    
    periods = ["morning", "afternoon", "evening", "weekend", "holiday"]
    zh_periods = ["早上", "下午", "晚上", "周末", "假期"]
    
    # 扩展数据
    while len(all_pairs) < n_samples:
        domain = random.choice(list(TEMPLATES.keys()))
        template_idx = random.randint(0, len(variations[domain]) - 1)
        template = variations[domain][template_idx]
        zh_template = zh_variations[domain][template_idx]
        
        t1 = random.choice(topics[domain])
        t2 = random.choice(topics[domain])
        t1_zh = zh_topics[domain][topics[domain].index(t1)]
        t2_zh = zh_topics[domain][topics[domain].index(t2)]
        
        en = template.format(t1, t2)
        zh = zh_template.format(t1_zh, t2_zh)
        
        all_pairs.append({"en": en, "zh": zh, "domain": domain})
    
    # 随机打乱并截取
    random.shuffle(all_pairs)
    return all_pairs[:n_samples]


def main():
    sizes = [5, 100, 500, 1000, 5000, 10000]
    
    for n in sizes:
        pairs = generate_dataset(n)
        
        # 按比例划分训练集和评估集
        split_idx = int(len(pairs) * 0.9)
        train = pairs[:split_idx]
        eval_set = pairs[split_idx:]
        
        # 保存
        with open(DATA_DIR / f"train_{n}.json", "w", encoding="utf-8") as f:
            json.dump(train, f, ensure_ascii=False, indent=2)
        with open(DATA_DIR / f"eval_{n}.json", "w", encoding="utf-8") as f:
            json.dump(eval_set, f, ensure_ascii=False, indent=2)
        
        print(f"✅ n={n}: 训练集={len(train)}, 评估集={len(eval_set)}")


if __name__ == "__main__":
    main()
