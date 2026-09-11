#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""corpus_from_riverbed 单元测试 — ≥5条"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', ''))
from corpus_from_riverbed import (
    split_sections, clean_section, is_code_only, has_too_many_links,
    process_file, file_hash,
    MIN_SECTION_CHARS,
)

PASS = FAIL = 0

def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  ✅ {name}')
    else:
        FAIL += 1
        print(f'  ❌ {name}: {detail}')


LONG_SECTION = '这是一段足够长的技术内容。' * 20  # ~200 chars


def test_split_sections():
    """T1: ## 标题正确切分，### 也切分，空节包含在内。"""
    md = f"""# 标题

前言。

## 第一节

{LONG_SECTION}

## 第二节

{LONG_SECTION}

### 三级标题

{LONG_SECTION}

## 空节

"""
    sections = split_sections(md)
    check('T1a 切分数量≥3', len(sections) >= 3, f'got {len(sections)}')
    check('T1b 第一节标题', sections[0][0] == '第一节')
    check('T1c 三级标题被切', any('三级' in s[0] for s in sections))


def test_quality_gate_min_chars():
    """T2: 短节被跳过，长节通过。"""
    check('T2a 短节<MIN', len(clean_section('太短')) < MIN_SECTION_CHARS)
    check('T2b 长节>=MIN', len(clean_section(LONG_SECTION * 2)) >= MIN_SECTION_CHARS)


def test_quality_gate_code_only():
    """T3: 纯代码块节被过滤。"""
    code = '```python\nx = 1\n```\n```python\ny = 2\n```'
    check('T3a 纯代码=True', is_code_only(code))
    mixed = f'```\nprint(1)\n```\n\n正常中文内容。{LONG_SECTION}'
    check('T3b 混合=False', not is_code_only(mixed))


def test_quality_gate_links():
    """T4: 链接行占比高被跳过。"""
    linky = '\n'.join([f'https://example.com/{i}' for i in range(20)])
    check('T4a 链接过多=True', has_too_many_links(linky))
    clean = '正常文本。\n第二行。\n第三行。'
    check('T4b 正常文本=False', not has_too_many_links(clean))


def test_idempotent_hash():
    """T5: 同内容同哈希（幂等基础）。"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write('# 测试\n\n内容')
        p1 = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write('# 测试\n\n内容')
        p2 = f.name
    try:
        check('T5 同内容同hash', file_hash(p1) == file_hash(p2))
    finally:
        os.unlink(p1); os.unlink(p2)


def test_overview_pair():
    """T6: 文件标题+首段→概览对。"""
    md = f"""# 概览文档标题

这是首段核心内容。需要足够长。{LONG_SECTION}

## 第一节

{LONG_SECTION}
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(md)
        path = f.name
    try:
        pairs = process_file(path, 1)
        overviews = [p for p in pairs if p['type'] == 'overview']
        check('T6a 生成概览对', len(overviews) >= 1, f'got {len(overviews)}')
        if overviews:
            check('T6b instruction含核心', '核心内容' in overviews[0]['instruction'])
            check('T6c output非空', len(overviews[0]['output']) > 0)
            check('T6d source字段', overviews[0]['source'].startswith('riverbed:'))
    finally:
        os.unlink(path)


def test_format_and_json():
    """T7: 所有训练对格式正确、JSON合法。"""
    md = f"""# 格式测试文件

概览段落。{LONG_SECTION}

## 有效节

{LONG_SECTION}

## 另一节

{LONG_SECTION}
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
        f.write(md)
        path = f.name
    try:
        pairs = process_file(path, 1)
        check('T7a 产出≥2条', len(pairs) >= 2, f'got {len(pairs)}')
        for i, p in enumerate(pairs):
            check(f'T7b[{i}] 有instruction', 'instruction' in p and len(p['instruction']) > 0)
            check(f'T7c[{i}] 有output', 'output' in p and len(p['output']) > 0)
            check(f'T7d[{i}] 有source', 'source' in p)
            check(f'T7e[{i}] type合法', p['type'] in ('overview', 'section'))
            try:
                json.dumps(p, ensure_ascii=False)
                check(f'T7f[{i}] JSON合法', True)
            except Exception as e:
                check(f'T7f[{i}] JSON合法', False, str(e))
    finally:
        os.unlink(path)


if __name__ == '__main__':
    print('🧪 corpus_from_riverbed 单元测试\n')
    test_split_sections()
    test_quality_gate_min_chars()
    test_quality_gate_code_only()
    test_quality_gate_links()
    test_idempotent_hash()
    test_overview_pair()
    test_format_and_json()
    print(f'\n{"="*40}')
    print(f'结果: {PASS} 通过, {FAIL} 失败')
    sys.exit(1 if FAIL else 0)
