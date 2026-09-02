"""
heuristic_filter.py — 规则过滤层
工程法典①匠石：能用代码就别用模型。纯规则过滤。
"""
import re
from ..base import ILMDocument, DocType


# 系统提示模式（需要过滤的内容）
SYSTEM_PATTERNS = [
    r"^\[INST\]",           # Llama格式
    r"^\[SYS\]",            # 系统标记
    r"^<<SYS>>",            # Llama2系统
    r"^-{3,}",              # 分隔线
    r"^={3,}",              # 分隔线
    r"^#{1,2}\s*$",         # 空标题
    r"^\[MODE:.*\]",        # 模式标记
    r"^\[TOOLS:.*\]",       # 工具标记
    r"^<thinking>",         # 思考标签
    r"^\[system\]",         # system标记
]

# 重复内容模式
REPEAT_PATTERNS = [
    r"^(.{20,})\1{2,}",    # 同一句话重复3次以上
    r"^(.{5,})\1{5,}",     # 同一片段重复6次以上
]


def _is_system_content(text: str) -> bool:
    """检查是否为系统内容"""
    for pattern in SYSTEM_PATTERNS:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def _has_excessive_repetition(text: str) -> bool:
    """检查是否有过度重复"""
    for pattern in REPEAT_PATTERNS:
        if re.search(pattern, text):
            return True
    
    # 字符级重复检测
    if len(text) > 50:
        unique_ratio = len(set(text)) / len(text)
        if unique_ratio < 0.15:  # 超过85%是重复字符
            return True
    
    return False


def _is_too_short(text: str, min_len: int = 10) -> bool:
    """检查内容是否太短"""
    # 去掉空白后的有效内容
    stripped = re.sub(r'\s+', '', text)
    return len(stripped) < min_len


def _is_code_only(text: str) -> bool:
    """检查是否纯代码（没有自然语言说明）"""
    lines = text.strip().split('\n')
    if len(lines) < 3:
        return False
    
    code_indicators = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # 代码行特征：缩进、import、def、class、if、for等
        if (stripped.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ',
                               'while ', 'try:', 'except', 'return ', 'yield ',
                               'self.', 'print(', '#!/', '"""', "'''", 'async ',
                               'await ', 'with ', 'as ', 'elif ', 'else:', 'finally:'))
            or stripped.endswith((':'))
            or re.match(r'^[a-zA-Z_]+\s*=\s*', stripped)
            or re.match(r'^\s{4,}', line)):  # 多级缩进
            code_indicators += 1
    
    code_ratio = code_indicators / len([l for l in lines if l.strip()])
    return code_ratio > 0.8


def filter_document(doc: ILMDocument, 
                    min_len: int = 10,
                    max_len: int = 50000,
                    allow_code: bool = True,
                    filter_system: bool = True) -> tuple:
    """
    对单个ILMDocument应用启发式过滤
    
    Returns:
        (passed: bool, reason: str, doc: ILMDocument)
    """
    content = doc.content
    
    # 1. 长度检查
    if _is_too_short(content, min_len):
        return False, "too_short", doc
    
    # 2. 超长截断（不是拒绝，是截断）
    if len(content) > max_len:
        doc.content = content[:max_len]
        doc.metadata["truncated"] = True
    
    # 3. 系统内容过滤
    if filter_system and _is_system_content(content):
        return False, "system_content", doc
    
    # 4. 重复内容过滤
    if _has_excessive_repetition(content):
        return False, "excessive_repetition", doc
    
    # 5. 纯代码过滤（可选）
    if not allow_code and _is_code_only(content):
        return False, "code_only", doc
    
    return True, "passed", doc


def filter_batch(documents: list, 
                 min_len: int = 10,
                 max_len: int = 50000,
                 allow_code: bool = True) -> tuple:
    """
    批量过滤
    
    Returns:
        (passed_docs: list, stats: dict)
    """
    passed = []
    stats = {"total": 0, "passed": 0, "filtered": 0, "reasons": {}}
    
    for doc in documents:
        stats["total"] += 1
        ok, reason, filtered_doc = filter_document(doc, min_len, max_len, allow_code)
        
        if ok:
            passed.append(filtered_doc)
            stats["passed"] += 1
        else:
            stats["filtered"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
    
    return passed, stats


if __name__ == "__main__":
    # 测试过滤器
    test_docs = [
        ILMDocument("test", "test.py", "太短", "unknown", 0),
        ILMDocument("test", "test.py", "这是一个正常的技术讨论，关于架构设计和实现方案的选择。涉及多个技术栈的对比分析。", "technical", 0),
        ILMDocument("test", "test.py", "import os\nimport sys\nimport json\nimport re\nimport math\nimport time\nimport datetime\nimport hashlib", "unknown", 0),
        ILMDocument("test", "test.py", "用户说：不可以讨好我，评估要真实严厉。\n\n这是关系信号——用户的偏好和决策风格。", "relation", 0),
        ILMDocument("test", "test.py", "这是一段包含大量重复内容的文本。这是一段包含大量重复内容的文本。这是一段包含大量重复内容的文本。这是一段包含大量重复内容的文本。这是一段包含大量重复内容的文本。", "unknown", 0),
    ]
    
    passed, stats = filter_batch(test_docs)
    print(f"Total: {stats['total']}, Passed: {stats['passed']}, Filtered: {stats['filtered']}")
    print(f"Reasons: {stats['reasons']}")
    for doc in passed:
        print(f"  PASSED: {doc.content[:60]}...")
