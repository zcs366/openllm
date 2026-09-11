#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corpus_from_riverbed.py — 河床文档→训练对提取器

扫描 /mnt/i/hermes/output/ 下指定主题目录的 .md 文件，
纯规则切分（零LLM），产出 instruction/output 训练对。

用法:
  python3 corpus_from_riverbed.py              # 标准：幂等追加
  python3 corpus_from_riverbed.py --dry-run    # 只统计不写
  python3 corpus_from_riverbed.py --reset      # 清index重跑

产出: /home/zcs/projects/isa/ilm/train_data/riverbed_knowledge.jsonl
索引: /home/zcs/projects/isa/ilm/train_data/riverbed_processed_index.json
"""
import json
import os
import re
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

# === 配置 ===
RIVERBED_ROOT = Path('/mnt/i/hermes/output')
TARGET_DIRS = [
    '818具神智能研究',
    '0905上下文工程学科',
    '安全工作',
]
OUTPUT_FILE = Path('/home/zcs/projects/isa/ilm/train_data/riverbed_knowledge.jsonl')
INDEX_FILE = Path('/home/zcs/projects/isa/ilm/train_data/riverbed_processed_index.json')

MIN_SECTION_CHARS = 200      # 节最低字数
MAX_OUTPUT_CHARS = 800       # output截断
MIN_FILE_CHARS = 100         # 文件级概览对的最低内容
LINK_THRESHOLD = 0.3         # 链接行占比超此值跳过


def file_hash(path: str) -> str:
    """内容哈希，用于幂等判断。"""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def load_index() -> dict:
    if INDEX_FILE.exists():
        with open(INDEX_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_index(idx: dict):
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)


def is_code_only(text: str) -> bool:
    """判断内容是否主要为代码块。"""
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    code_chars = sum(len(b) for b in code_blocks)
    total_chars = len(text)
    if total_chars == 0:
        return True
    return code_chars / total_chars > 0.8


def has_too_many_links(text: str) -> bool:
    """链接行占比过高则跳过。"""
    lines = text.strip().split('\n')
    if not lines:
        return True
    link_lines = sum(1 for l in lines if re.search(r'https?://', l))
    return (link_lines / len(lines)) > LINK_THRESHOLD


def clean_section(text: str) -> str:
    """清理节内容：去首尾空白、截断。"""
    text = text.strip()
    # 去除纯分隔线
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE).strip()
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + '…'
    return text


def extract_file_title(content: str) -> str:
    """从md提取文档标题（第一个#标题）。"""
    m = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ''


def extract_first_para(content: str) -> str:
    """提取标题后第一个非空段落。"""
    lines = content.split('\n')
    passed_title = False
    para_lines = []
    for line in lines:
        if not passed_title:
            if re.match(r'^#\s+', line):
                passed_title = True
            continue
        stripped = line.strip()
        if not stripped:
            if para_lines:
                break
            continue
        if re.match(r'^#{1,6}\s+', stripped):
            break
        para_lines.append(stripped)
    return ' '.join(para_lines)


def split_sections(content: str) -> list:
    """按 ## 标题切分节。返回 [(heading, body), ...]"""
    sections = []
    parts = re.split(r'^(#{2,3}\s+.+)$', content, flags=re.MULTILINE)
    # parts[0] = 标题前内容, then alternating: heading, body
    i = 1
    while i < len(parts) - 1:
        heading = re.sub(r'^#{2,3}\s+', '', parts[i]).strip()
        body = parts[i + 1]
        sections.append((heading, body))
        i += 2
    return sections


def process_file(filepath: str, file_idx: int) -> list:
    """处理单个md文件，返回训练对列表。"""
    try:
        with open(filepath, encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f'  ⚠ 读取失败 {filepath}: {e}')
        return []

    if len(content.strip()) < MIN_FILE_CHARS:
        return []

    pairs = []
    file_title = extract_file_title(content) or Path(filepath).stem
    rel_path = str(filepath)

    # 1. 文件级概览对
    first_para = extract_first_para(content)
    if first_para and len(first_para) >= 50:
        overview = clean_section(first_para)
        if overview and not is_code_only(overview):
            pairs.append({
                'instruction': f'关于{file_title}的核心内容是什么？',
                'output': overview,
                'source': f'riverbed:{rel_path}',
                'type': 'overview',
            })

    # 2. 节级训练对
    sections = split_sections(content)
    for heading, body in sections:
        cleaned = clean_section(body)
        if len(cleaned) < MIN_SECTION_CHARS:
            continue
        if is_code_only(cleaned):
            continue
        if has_too_many_links(cleaned):
            continue
        pairs.append({
            'instruction': f'关于{file_title}中「{heading}」的要点',
            'output': cleaned,
            'source': f'riverbed:{rel_path}',
            'type': 'section',
        })

    return pairs


def main():
    parser = argparse.ArgumentParser(description='河床文档→训练对提取器')
    parser.add_argument('--dry-run', action='store_true', help='只统计不写')
    parser.add_argument('--reset', action='store_true', help='清索引重跑')
    args = parser.parse_args()

    idx = {} if args.reset else load_index()
    all_pairs = []
    files_scanned = 0
    files_new = 0
    files_skipped = 0

    for dirname in TARGET_DIRS:
        dirpath = RIVERBED_ROOT / dirname
        if not dirpath.is_dir():
            print(f'⚠ 目录不存在: {dirpath}')
            continue
        md_files = sorted(dirpath.rglob('*.md'))
        print(f'📁 {dirname}: {len(md_files)} 个 .md 文件')
        for fp in md_files:
            fp_str = str(fp)
            files_scanned += 1
            fhash = file_hash(fp_str)
            if fp_str in idx and idx[fp_str] == fhash:
                files_skipped += 1
                continue
            pairs = process_file(fp_str, files_scanned)
            if pairs:
                all_pairs.extend(pairs)
                files_new += 1
                idx[fp_str] = fhash
                print(f'  ✓ {fp.name} → {len(pairs)} 对')
            else:
                idx[fp_str] = fhash  # 标记已处理，不再重复扫
                files_skipped += 1

    total_new = len(all_pairs)

    # 追加写入
    if not args.dry_run and total_new > 0:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        save_index(idx)

    # 统计已有总量
    existing_total = 0
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding='utf-8') as f:
            existing_total = sum(1 for _ in f)

    print(f'\n{"="*50}')
    print(f'📊 报告 ({datetime.now().strftime("%Y-%m-%d %H:%M")})')
    print(f'  扫描文件: {files_scanned}')
    print(f'  新处理:   {files_new}')
    print(f'  跳过(幂等): {files_skipped}')
    print(f'  本次产出: {total_new} 训练对')
    print(f'  文件总量: {existing_total} 训练对')
    print(f'  输出文件: {OUTPUT_FILE}')
    print(f'  索引文件: {INDEX_FILE}')

    # 抽样展示
    if all_pairs:
        print(f'\n📋 抽样 (前3条):')
        for i, p in enumerate(all_pairs[:3]):
            print(f'\n--- 第{i+1}条 ({p["type"]}) ---')
            print(f'  instruction: {p["instruction"][:80]}')
            print(f'  output: {p["output"][:120]}...')
            print(f'  source: {p["source"]}')

    return total_new


if __name__ == '__main__':
    main()
