# -*- coding: utf-8 -*-
"""format_converter 测试 — 验证三类样本构造逻辑正确性。"""
import json
import os
import sqlite3
import sys
import tempfile

# 确保能 import format_converter
ilm_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ilm_dir not in sys.path:
    sys.path.insert(0, ilm_dir)

from ..format_converter import (
    CORRECTION_PATTERN, FormatConverter, _is_auto_message, _is_cron,
)


def _make_state_db(path):
    """构造测试用 state.db（1个session，含纠正对）。"""
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute('CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL)')
    cur.execute('''CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
        role TEXT, content TEXT, timestamp REAL)''')
    cur.execute("INSERT INTO sessions VALUES ('s1', 1.0)")
    msgs = [
        ('s1', 'user', '帮我写一个二分查找', 1.0),
        ('s1', 'assistant', '好的，这是二分查找：\n```python\ndef bs(a, x): ...\n```\n' * 3, 2.0),
        ('s1', 'user', '不对，你这个实现没有处理空列表！', 3.0),
        ('s1', 'assistant', '你说得对，空列表会IndexError。修正：\n```python\ndef bs(a, x):\n    if not a: return -1\n```\n加上空列表检查。', 4.0),
        # cron 噪音（必须被过滤）
        ('s1', 'user', '[IMPORTANT: You are running as a scheduled cron job] 处理日报', 5.0),
        ('s1', 'assistant', '好的，生成日报。', 6.0),
    ]
    cur.executemany(
        'INSERT INTO messages (session_id, role, content, timestamp) '
        'VALUES (?,?,?,?)', msgs)
    conn.commit()
    conn.close()


def _make_corpus_db(path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE documents (
        doc_id TEXT PRIMARY KEY, source TEXT, content TEXT, doc_type TEXT,
        domain TEXT, quality_score REAL, source_ref TEXT, metadata TEXT)''')
    content = '核心铁律：训练数据必须带正负样本对。' * 20
    cur.execute(
        "INSERT INTO documents VALUES ('d1','session',?,"
        " 'technical','',0.9,'state.db:test','{}')", (content,))
    conn.commit()
    conn.close()


def test_correction_pattern():
    assert CORRECTION_PATTERN.search('不对，你理解错了')
    assert CORRECTION_PATTERN.search('不是这样，应该是那样')
    assert not CORRECTION_PATTERN.search('这个方案不错')
    assert not CORRECTION_PATTERN.search('继续')


def test_noise_filters():
    assert _is_cron('[IMPORTANT: You are running as a scheduled cron job] x')
    assert _is_auto_message('[ASYNC DELEGATION BATCH COMPLETE] x')
    assert not _is_auto_message('老搭档，我们继续')


def test_converter_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        state_db = os.path.join(tmp, 'state.db')
        corpus_db = os.path.join(tmp, 'corpus.db')
        outdir = os.path.join(tmp, 'train_data')
        _make_state_db(state_db)
        _make_corpus_db(corpus_db)

        fc = FormatConverter(state_db, corpus_db, outdir,
                             max_corrections=100, max_instruction=100,
                             max_knowledge=100, session_limit=10)
        stats = fc.run()

        # corrections: 1对(neg+pos)=2条
        assert stats['corrections'] == 2, f"期望2条corrections, 实际{stats['corrections']}"
        # knowledge: 1条
        assert stats['knowledge'] == 1, f"期望1条knowledge, 实际{stats['knowledge']}"

        # 校验 JSONL 内容
        with open(os.path.join(outdir, 'corrections.jsonl'), encoding='utf-8') as f:
            lines = [l for l in f if not l.startswith('#')]
            items = [json.loads(l) for l in lines]
        tags = {i.get('rejected') for i in items}
        assert tags == {True, False}, 'corrections 应同时含 neg+pos'
        # 每条有 source_ref
        for i in items:
            assert i['source_ref'].startswith('state.db:s1:'), i['source_ref']
            assert i['type'] == 'correction'
        # 交替规则
        for i in items:
            conv = i['conversation']
            assert conv[0]['role'] == 'user'
            assert conv[1]['role'] == 'assistant'

        # 长度上限：超长样本应被过滤
        with open(os.path.join(outdir, 'corrections.jsonl'), encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                item = json.loads(line)
                for m in item['conversation']:
                    assert len(m['content']) <= 3000, '超长样本未过滤'

        # CONTEXT COMPACTION 注入不进训练数据
        for fname in ['corrections.jsonl', 'instruction_pairs.jsonl']:
            with open(os.path.join(outdir, fname), encoding='utf-8') as f:
                content = f.read()
            assert 'CONTEXT COMPACTION' not in content, f'{fname} 含压缩注入'
            assert 'Earlier turns were compacted' not in content, f'{fname} 含压缩注入'

        # cron 噪音未进入 instruction
        with open(os.path.join(outdir, 'instruction_pairs.jsonl'), encoding='utf-8') as f:
            content = f.read()
        assert 'scheduled cron job' not in content

        print('✅ 全部断言通过')


if __name__ == '__main__':
    test_correction_pattern()
    test_noise_filters()
    test_converter_end_to_end()
    print('✅ format_converter 测试全部通过')
