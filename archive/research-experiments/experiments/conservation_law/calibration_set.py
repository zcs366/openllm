"""
校准集 — 子贡出品

100条中英文句子，覆盖：
- 日常对话
- 技术文档
- 文学作品
- 新闻报道
"""

CALIBRATION_SET = [
    # 日常对话 (20条)
    "你好，今天天气怎么样？",
    "我想喝一杯咖啡。",
    "明天我们去公园散步吧。",
    "这部电影真的很好看。",
    "你能帮我一个忙吗？",
    "我最近在学习编程。",
    "这个周末有什么计划？",
    "我觉得这个主意不错。",
    "让我们一起吃晚饭吧。",
    "你最近工作忙吗？",
    "我想买一本新书。",
    "这个地方真漂亮。",
    "你能推荐一家餐厅吗？",
    "我需要休息一下。",
    "这个问题很难回答。",
    "我很期待明天的会议。",
    "你觉得这个方案怎么样？",
    "我想去旅行。",
    "这个菜很好吃。",
    "我们下次再聊。",
    
    # 技术文档 (20条)
    "Transformer模型使用自注意力机制处理序列数据。",
    "Python是一种广泛使用的高级编程语言。",
    "机器学习算法可以从数据中自动学习模式。",
    "深度神经网络包含多个隐藏层。",
    "自然语言处理是人工智能的一个重要分支。",
    "卷积神经网络在图像识别任务中表现出色。",
    "强化学习通过与环境交互来学习最优策略。",
    "数据预处理是机器学习流水线中的关键步骤。",
    "模型评估需要使用独立的测试数据集。",
    "梯度下降算法用于优化神经网络的参数。",
    "正则化技术可以防止模型过拟合。",
    "批归一化可以加速神经网络的训练。",
    "注意力机制允许模型关注输入的不同部分。",
    "循环神经网络适合处理序列数据。",
    "生成对抗网络可以生成逼真的图像。",
    "迁移学习可以利用预训练模型的知识。",
    "超参数调优是模型优化的重要环节。",
    "特征工程可以提高模型的性能。",
    "交叉验证可以更准确地评估模型性能。",
    "集成学习方法可以提高预测的准确性。",
    
    # 文学作品 (20条)
    "人生若只如初见，何事秋风悲画扇。",
    "床前明月光，疑是地上霜。",
    "大漠孤烟直，长河落日圆。",
    "采菊东篱下，悠然见南山。",
    "山重水复疑无路，柳暗花明又一村。",
    "春眠不觉晓，处处闻啼鸟。",
    "白日依山尽，黄河入海流。",
    "欲穷千里目，更上一层楼。",
    "海内存知己，天涯若比邻。",
    "落霞与孤鹜齐飞，秋水共长天一色。",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The only thing we have to fear is fear itself.",
    "In the middle of difficulty lies opportunity.",
    "Knowledge is power.",
    "Actions speak louder than words.",
    "The pen is mightier than the sword.",
    "A journey of a thousand miles begins with a single step.",
    "Rome was not built in a day.",
    "When in Rome, do as the Romans do.",
    
    # 新闻报道 (20条)
    "全球气候变化问题日益严重，各国正在采取行动。",
    "人工智能技术正在改变各行各业的运作方式。",
    "国际金融市场出现剧烈波动，投资者保持谨慎。",
    "新能源汽车销量持续增长，传统车企加速转型。",
    "全球疫情形势依然复杂，疫苗接种工作持续推进。",
    "科技创新成为推动经济增长的重要引擎。",
    "教育改革政策引发社会广泛讨论。",
    "国际局势紧张，外交谈判成为焦点。",
    "环保组织呼吁加强环境保护力度。",
    "数字经济成为新的增长点。",
    "SpaceX successfully launched another batch of Starlink satellites.",
    "The Federal Reserve announced a new interest rate decision.",
    "Global supply chain disruptions continue to affect businesses.",
    "Climate change summit reaches historic agreement.",
    "Tech giants face increasing regulatory scrutiny.",
    "Renewable energy investments hit record highs.",
    "International trade tensions escalate between major economies.",
    "Healthcare systems worldwide prepare for winter challenges.",
    "Educational institutions adapt to hybrid learning models.",
    "Cybersecurity threats become more sophisticated.",
    
    # 哲学思考 (20条)
    "我思故我在。",
    "知识就是力量。",
    "认识你自己。",
    "万物皆流，无物常驻。",
    "道可道，非常道。",
    "知之为知之，不知为不知，是知也。",
    "己所不欲，勿施于人。",
    "学而不思则罔，思而不学则殆。",
    "温故而知新，可以为师矣。",
    "三人行，必有我师焉。",
    "The unexamined life is not worth living.",
    "I think, therefore I am.",
    "The only true wisdom is in knowing you know nothing.",
    "The greatest wealth is to live content with little.",
    "Happiness is not an ideal of reason but of imagination.",
    "The only way to do great work is to love what you do.",
    "In the middle of difficulty lies opportunity.",
    "The best time to plant a tree was 20 years ago.",
    "Life is what happens when you're busy making other plans.",
    "The future belongs to those who believe in the beauty of their dreams.",
]


def get_calibration_set():
    """获取校准集"""
    return CALIBRATION_SET


def get_calibration_stats():
    """获取校准集统计信息"""
    total = len(CALIBRATION_SET)
    chinese = sum(1 for s in CALIBRATION_SET if any('\u4e00' <= c <= '\u9fff' for c in s))
    english = total - chinese
    
    return {
        "total": total,
        "chinese": chinese,
        "english": english,
        "categories": {
            "日常对话": 20,
            "技术文档": 20,
            "文学作品": 20,
            "新闻报道": 20,
            "哲学思考": 20
        }
    }


if __name__ == "__main__":
    stats = get_calibration_stats()
    print("=== 校准集统计 ===")
    print(f"总数: {stats['total']}")
    print(f"中文: {stats['chinese']}")
    print(f"英文: {stats['english']}")
    print(f"\n分类:")
    for category, count in stats['categories'].items():
        print(f"  {category}: {count}")
