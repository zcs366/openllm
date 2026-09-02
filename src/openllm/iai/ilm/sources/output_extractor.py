"""
output_extractor.py — hermes/output + I盘output → ILMDocument流
铁律：source_ref永久填充，Markdown文档按段落拆分
"""
import os
import re
import glob
from typing import Generator
from ..base import ILMDocument, SourceType, DocType


RELATION_KEYWORDS = [
    "纠正", "偏好", "记住", "以后", "风格", "关系", "老搭档",
]
TECHNICAL_KEYWORDS = [
    "架构", "发现", "验证", "实验", "论文", "算法",
    "模型", "D0", "attention", "transformer", "ISA",
]
ENGINEERING_KEYWORDS = [
    "P0", "P1", "交付", "完成", "测试", "审计",
    "PAL", "里程碑", "部署", "里程碑",
]


def _classify_text(text: str) -> str:
    r = sum(1 for kw in RELATION_KEYWORDS if kw in text)
    t = sum(1 for kw in TECHNICAL_KEYWORDS if kw in text)
    e = sum(1 for kw in ENGINEERING_KEYWORDS if kw in text)
    mx = max(r, t, e)
    if mx == 0:
        return DocType.UNKNOWN.value
    if r == mx:
        return DocType.RELATION.value
    if t == mx:
        return DocType.TECHNICAL.value
    return DocType.ENGINEERING.value


def _split_markdown(content: str, min_len: int = 50) -> list:
    """
    Markdown按段落拆分
    策略：按二级标题(##)拆分大段，每段保留标题作为上下文
    """
    sections = []
    current_section = []
    current_title = ""
    
    for line in content.split('\n'):
        if re.match(r'^#{1,2}\s+', line):
            # 遇到标题，保存当前段落
            if current_section:
                text = '\n'.join(current_section).strip()
                if len(text) >= min_len:
                    sections.append((current_title, text))
            current_title = line.strip('#').strip()
            current_section = [line]
        else:
            current_section.append(line)
    
    # 最后一段
    if current_section:
        text = '\n'.join(current_section).strip()
        if len(text) >= min_len:
            sections.append((current_title, text))
    
    # 如果没有标题拆分成功，按段落拆
    if not sections:
        paragraphs = re.split(r'\n\s*\n', content)
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) >= min_len:
                sections.append((f"paragraph_{i}", para))
    
    return sections


def _build_source_ref(file_path: str, line_offset: int = 0) -> str:
    """永久字段：source_ref = "output:{file_path}:{line_offset}" """
    # 只保留相对路径
    home = os.path.expanduser("~")
    rel = file_path.replace(home, "~")
    return f"output:{rel}:L{line_offset}"


def extract_output(dirs: list,
                   min_content_len: int = 50,
                   max_docs: int = 0) -> Generator[ILMDocument, None, None]:
    """
    从output目录提取Markdown研究文档
    
    铁律：
    - source_ref永远填充
    - Markdown按段落拆分（保留标题上下文）
    - 跳过非.md文件
    - 跳过INDEX.md等索引文件
    """
    count = 0
    skip_patterns = {"INDEX", "index", "README", "CHANGELOG"}
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            continue
        
        md_files = glob.glob(os.path.join(dir_path, "**/*.md"), recursive=True)
        
        for md_path in sorted(md_files):
            # 跳过索引文件
            basename = os.path.basename(md_path)
            if any(pat in basename for pat in skip_patterns):
                continue
            
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (IOError, UnicodeDecodeError):
                continue
            
            if len(content) < min_content_len:
                continue
            
            # 按段落拆分
            sections = _split_markdown(content, min_len=min_content_len)
            
            for line_offset, (title, section_content) in enumerate(sections):
                full_content = f"{title}\n\n{section_content}" if title else section_content
                
                doc = ILMDocument(
                    source=SourceType.OUTPUT.value,
                    source_path=md_path,
                    content=full_content,
                    doc_type=_classify_text(full_content),
                    timestamp=os.path.getmtime(md_path),
                    metadata={
                        "file_name": basename,
                        "section_title": title,
                        "file_size": os.path.getsize(md_path),
                    },
                    source_ref=_build_source_ref(md_path, line_offset),
                )
                yield doc
                count += 1
                
                if max_docs > 0 and count >= max_docs:
                    return
    
    print(f"[output_extractor] Extracted {count} sections from {len(dirs)} dirs")


if __name__ == "__main__":
    import sys
    dirs = [
        os.path.expanduser("~/hermes/output"),
        os.path.expanduser("/mnt/i/hermes/output"),
    ]
    
    total = 0
    for doc in extract_output(dirs, max_docs=20):
        total += 1
        print(f"[{total}] {doc.doc_type} | ref={doc.source_ref[:60]} | {doc.content[:60]}...")
