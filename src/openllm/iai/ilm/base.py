"""
ILM数据协议 — 所有数据源适配器输出同一种格式
ILM (Inferred Lifetime Memory) Data Protocol
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List
import time
import hashlib


class DocType(Enum):
    """数据类型：三类核心信号"""
    RELATION = "relation"      # 关系信号：用户纠正、偏好、决策风格
    TECHNICAL = "technical"    # 技术insight：架构决策、技术发现、方法论
    ENGINEERING = "engineering"  # 工程决策：任务分解、优先级、交付记录
    KNOWLEDGE = "knowledge"    # 知识：百科、论文、教程
    UNKNOWN = "unknown"        # 未分类


class SourceType(Enum):
    """数据来源"""
    SESSION = "session"        # state.db对话
    RECALL = "recall"          # RECALL.jsonl经验
    JIAK = "jiak"              # jiak卡片
    OUTPUT = "output"          # hermes/output研究文档
    WIKI = "wiki"              # I盘wiki知识库
    CODE = "code"              # 代码
    DELTA = "delta"            # Δ胶囊
    RESEARCH = "research"      # iah/iat/ita/io-s/isn/izu研究数据


@dataclass
class ILMDocument:
    """ILM统一文档格式 — 所有数据源适配器输出此格式"""
    source: str               # SourceType.value
    source_path: str          # 原始文件路径
    content: str              # 提取后的文本内容
    doc_type: str             # DocType.value
    timestamp: float          # 原始时间戳
    metadata: dict = field(default_factory=dict)  # 源特定元数据
    quality_score: float = 0.0  # 质量分数（过滤后填）
    domain: str = ""          # 领域标签（分类后填）
    doc_id: str = ""          # 唯一ID（自动生成）
    content_hash: str = ""    # 内容哈希（去重用）
    source_ref: str = ""      # 永久字段：指向原始来源（session_id+message_id 或文件路径+行号）

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = self._generate_id()
        if not self.content_hash:
            self.content_hash = self._generate_hash()

    def _generate_id(self) -> str:
        """基于来源+路径+时间戳生成唯一ID"""
        raw = f"{self.source}:{self.source_path}:{self.timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _generate_hash(self) -> str:
        """基于内容生成哈希（去重用）"""
        if not self.content:
            return ""
        return hashlib.md5(self.content.encode()).hexdigest()

    def to_dict(self) -> dict:
        """序列化为dict（用于JSONL写入）"""
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> 'ILMDocument':
        """从dict反序列化"""
        return ILMDocument(**{k: v for k, v in d.items()
                              if k in ILMDocument.__dataclass_fields__})


# 关系信号关键词（用于启发式分类）
RELATION_KEYWORDS = [
    # 用户纠正
    "不对", "错了", "不要", "不可以", "不许", "别", "停",
    "我说的不是", "你理解错了", "重新来",
    # 偏好表达
    "我喜欢", "我讨厌", "我习惯", "我不喜欢", "偏好",
    "以后", "从今以后", "记住",
    # 决策风格
    "先", "后", "优先", "不重要", "重要", "必须", "不要",
    "你觉得", "你的判断", "你来分析",
    # 关系信号
    "老搭档", "军师", "老铁", "兄弟",
]

# 技术insight关键词
TECHNICAL_KEYWORDS = [
    "架构", "设计", "方案", "原理", "机制", "算法",
    "模型", "训练", "推理", "优化", "性能",
    "发现", "验证", "实验", "结论", "论文",
    "D0", "attention", "transformer", "embedding",
    "MinHash", "FAISS", "BGE", "SQLite",
]

# 工程决策关键词
ENGINEERING_KEYWORDS = [
    "P0", "P1", "P2", "里程碑", "交付", "部署",
    "测试", "审计", "修复", "重构",
    "PAL", "任务书", "执行报告", "三碑",
    "优先级", "工期", "截止", "deadline",
    "完成", "通过", "验收", "上线",
]
