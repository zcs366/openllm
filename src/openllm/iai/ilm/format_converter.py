# -*- coding: utf-8 -*-
"""
format_converter.py — ILM Phase 1 核心模块
把原始对话数据转换为 LLM 微调训练数据（SFT 三元组），输出 JSONL。

ILM 食物链: 原始数据 → 清洗(corpus.db) → 格式化(本模块) → 训练(QLoRA) → 权重
本模块是"消化系统"的第二环：把清洗后的文档 + 原始对话重建为训练样本。

三类样本:
  1. correction  — 用户纠正对（最高价值：前答=rejected负样本, 后答=正样本）
  2. instruction — 指令-执行对（user指令 + assistant实质回答）
  3. knowledge   — 知识注入对（corpus.db技术/工程文档 → 启发式构造）

铁律:
  - 每条样本必须有 source_ref（永久来源字段）
  - conversation 必须 user/assistant 交替
  - cron 指令、纯工具噪音、空回复 一律不进训练数据
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import OrderedDict

# ── 纠正信号正则（用户明确否定/纠正）──
CORRECTION_PATTERN = re.compile(
    r'(不对|错了|不是.{0,6}(而是|是)|理解错|重新来|重来|我说的是|'
    r'你又|别[的用把]|不要[的用把]|停[一下止]|反了|反着|搞错|'
    r'说得不对|这不对|不是这样|方向[反错]|那不对)'
)

# ── 噪音过滤 ──
CRON_MARKERS = [
    '[IMPORTANT: You are running as a scheduled cron job',
    'DELIVERY: Your final response',
    'You are running as a scheduled cron job',
]
TOOL_NOISE_MARKERS = ['调用工具结果', 'tool result', 'Tool execution']
# 系统自动注入消息（role=user 但实为后台通知，非真人输入）
AUTO_MESSAGE_PREFIXES = [
    '[ASYNC', '[IMPORTANT', '[SYSTEM', '[OUT-OF-BAND',
    'ASYNC DELEGATION', 'Scheduled cron',
    '[CONTEXT COMPACTION', 'Earlier turns were compacted',
]
# 训练样本内容长度上限（字符）：超长样本拖垮训练
MAX_CONTENT_LEN = 3000

# 无意义短指令（instruction 类过滤）
TRIVIAL_INSTRUCTION = re.compile(
    r'^(好的|好|嗯|继续|ok|OK|收到|行|可以|对|是的|没错|谢谢|'
    r'再来|就这样|开始吧|开干|同意|赞|nice|good|perfect)$'
)


def _is_cron(content: str) -> bool:
    return any(m in content for m in CRON_MARKERS)


def _is_auto_message(content: str) -> bool:
    """系统自动注入消息（async委派/cron通知/带标记的系统消息）。"""
    return any(content.startswith(p) for p in AUTO_MESSAGE_PREFIXES)


def _is_tool_noise(content: str) -> bool:
    return any(m in content for m in TOOL_NOISE_MARKERS)


def _quality_score(assistant_content: str, user_content: str = '') -> float:
    """启发式质量评分 0-1。"""
    q = 0.5
    n = len(assistant_content)
    if n < 20:
        return 0.1
    if n > 100:
        q += 0.1
    if n > 500:
        q += 0.1
    if '```' in assistant_content:
        q += 0.15  # 有代码块
    if re.search(r'\n[-*] |\n#|^\d+\.', assistant_content):
        q += 0.1   # 有结构化列表/标题
    if '|' in assistant_content and '---' in assistant_content:
        q += 0.1   # 有表格
    if user_content and len(user_content) > 30:
        q += 0.05  # 用户指令具体
    return round(min(1.0, q), 2)


class FormatConverter:
    def __init__(self, state_db: str, corpus_db: str, outdir: str,
                 max_corrections: int, max_instruction: int,
                 max_knowledge: int, session_limit: int):
        self.state_db = state_db
        self.corpus_db = corpus_db
        self.outdir = outdir
        self.max_corrections = max_corrections
        self.max_instruction = max_instruction
        self.max_knowledge = max_knowledge
        self.session_limit = session_limit
        os.makedirs(outdir, exist_ok=True)

        self.corrections = []   # (quality, dict)
        self.instructions = []  # (quality, dict)
        self.knowledge = []     # (quality, dict)
        self.seen_hashes = set()

    # ── 主流程 ──
    def run(self) -> dict:
        t0 = time.time()
        stats = {'sessions_scanned': 0, 'corrections': 0, 'instructions': 0,
                 'knowledge': 0, 'deduped': 0}

        # 1. 从 state.db 提取对话
        sessions = self._load_sessions()
        stats['sessions_scanned'] = len(sessions)

        for sid in sessions:
            msgs = self._load_session_messages(sid)
            if not msgs:
                continue
            self._extract_corrections(sid, msgs)
            self._extract_instructions(sid, msgs)

        # 2. 从 corpus.db 提取知识
        self._extract_knowledge()

        # 3. 去重 + 截断 + 写出
        for bucket, max_n, fname in [
            (self.corrections, self.max_corrections, 'corrections.jsonl'),
            (self.instructions, self.max_instruction, 'instruction_pairs.jsonl'),
            (self.knowledge, self.max_knowledge, 'knowledge_triples.jsonl'),
        ]:
            before = len(bucket)
            bucket.sort(key=lambda x: x[0], reverse=True)
            bucket = bucket[:max_n]
            stats['deduped'] += before - len(bucket)
            self._write_jsonl(bucket, os.path.join(self.outdir, fname))

        stats['corrections'] = len(self.corrections)
        stats['instructions'] = len(self.instructions)
        stats['knowledge'] = len(self.knowledge)
        stats['elapsed'] = round(time.time() - t0, 1)
        return stats

    # ── state.db 加载 ──
    def _load_sessions(self) -> list:
        conn = sqlite3.connect(self.state_db)
        try:
            cur = conn.cursor()
            rows = cur.execute(
                'SELECT id FROM sessions ORDER BY started_at DESC'
            ).fetchall()
            sessions = [r[0] for r in rows]
            if self.session_limit > 0:
                sessions = sessions[:self.session_limit]
            return sessions
        finally:
            conn.close()

    def _load_session_messages(self, sid: str) -> list:
        """按时间序加载单个 session 的消息（排除 tool/system 噪声）。"""
        conn = sqlite3.connect(self.state_db)
        try:
            cur = conn.cursor()
            rows = cur.execute(
                'SELECT id, role, content, timestamp FROM messages '
                'WHERE session_id=? AND content IS NOT NULL '
                'ORDER BY timestamp ASC, id ASC',
                (sid,)
            ).fetchall()
            msgs = []
            for mid, role, content, ts in rows:
                content = (content or '').strip()
                if not content:
                    continue
                msgs.append({'id': mid, 'role': role, 'content': content, 'ts': ts})
            return msgs
        finally:
            conn.close()

    # ── 样本构造：correction ──
    def _extract_corrections(self, sid: str, msgs: list) -> None:
        """找纠正信号 user 消息，取前一条 assistant=rejected，后一条 assistant=positive。"""
        for i, m in enumerate(msgs):
            if m['role'] != 'user':
                continue
            content = m['content']
            if (_is_cron(content) or _is_auto_message(content)
                    or _is_tool_noise(content) or len(content) < 10):
                continue
            if len(content) > MAX_CONTENT_LEN:
                continue  # 超长（如context compaction注入）
            if not CORRECTION_PATTERN.search(content):
                continue
            # 找前一条 assistant（可能隔着 tool）
            prev_a = None
            for j in range(i - 1, -1, -1):
                if msgs[j]['role'] == 'assistant' and len(msgs[j]['content']) >= 20:
                    prev_a = msgs[j]
                    break
                if msgs[j]['role'] == 'user':
                    break
            if prev_a is None:
                continue
            # 找后一条 assistant
            next_a = None
            for k in range(i + 1, len(msgs)):
                if msgs[k]['role'] == 'assistant' and len(msgs[k]['content']) >= 20:
                    next_a = msgs[k]
                    break
                if msgs[k]['role'] == 'user':
                    break
            if next_a is None:
                continue
            # 长度上限：超长样本拖垮训练（单步极慢/卡死）
            if len(prev_a['content']) > 3000 or len(next_a['content']) > 3000:
                continue
            # 负样本（被纠正的回答）
            neg = {
                'conversation': [
                    {'role': 'user', 'content': content},
                    {'role': 'assistant', 'content': prev_a['content']},
                ],
                'type': 'correction', 'rejected': True,
                'source_ref': f'state.db:{sid}:{prev_a["id"]}',
                'quality': 0.3,
            }
            # 正样本（纠正后的回答）
            pos = {
                'conversation': [
                    {'role': 'user', 'content': content},
                    {'role': 'assistant', 'content': next_a['content']},
                ],
                'type': 'correction', 'rejected': False,
                'source_ref': f'state.db:{sid}:{next_a["id"]}',
                'quality': 0.9,
            }
            self._add_dedup(self.corrections, neg)
            self._add_dedup(self.corrections, pos)

    # ── 样本构造：instruction ──
    def _extract_instructions(self, sid: str, msgs: list) -> None:
        """user 指令 + 紧随其后的 assistant 实质回答。"""
        for i, m in enumerate(msgs):
            if m['role'] != 'user':
                continue
            content = m['content']
            if _is_cron(content) or _is_auto_message(content) or _is_tool_noise(content):
                continue
            if len(content) > MAX_CONTENT_LEN or len(content) < 6:
                continue
            if TRIVIAL_INSTRUCTION.match(content.strip()):
                continue
            # 找紧随其后的 assistant 实质回答
            for k in range(i + 1, len(msgs)):
                if msgs[k]['role'] == 'user':
                    break
                if msgs[k]['role'] == 'assistant':
                    ans = msgs[k]['content']
                    if len(ans) < 50:
                        break  # 太短不是实质回答
                    if len(ans) > 3000:
                        break  # 超长样本拖垮训练，跳过
                    q = _quality_score(ans, content)
                    if q >= 0.6:
                        sample = {
                            'conversation': [
                                {'role': 'user', 'content': content},
                                {'role': 'assistant', 'content': ans},
                            ],
                            'type': 'instruction',
                            'source_ref': f'state.db:{sid}:{msgs[k]["id"]}',
                            'quality': q,
                        }
                        self._add_dedup(self.instructions, sample)
                    break  # 只取紧跟的第一条 assistant

    # ── 样本构造：knowledge ──
    def _extract_knowledge(self) -> None:
        """从 corpus.db 技术/工程文档构造知识注入对。"""
        conn = sqlite3.connect(self.corpus_db)
        try:
            cur = conn.cursor()
            rows = cur.execute(
                "SELECT doc_id, content, doc_type, source_ref FROM documents "
                "WHERE doc_type IN ('technical','engineering') "
                "AND LENGTH(content) > 200"
            ).fetchall()
            for doc_id, content, doc_type, source_ref in rows:
                # 找含结论/方法的关键段落
                seg_pat = re.compile(
                    r'([^\n]*(?:核心|关键|结论|方法|教训|铁律|洞察)[^\n]*\n[^\n]*)')
                segs = seg_pat.findall(content)
                if segs:
                    seg = segs[0].strip()
                    if len(seg) > 40:
                        sample = {
                            'conversation': [
                                {'role': 'user', 'content':
                                    f'根据你的经验，总结一下：{seg[:80]}'},
                                {'role': 'assistant', 'content': seg},
                            ],
                            'type': 'knowledge',
                            'source_ref': source_ref or f'corpus.db:{doc_id}',
                            'quality': 0.8,
                        }
                        self._add_dedup(self.knowledge, sample)
                # 无结论段时用文档开头摘要
                else:
                    head = content[:200].strip()
                    if len(head) > 100:
                        sample = {
                            'conversation': [
                                {'role': 'user', 'content':
                                    f'阅读以下资料并总结要点：\n{head}'},
                                {'role': 'assistant', 'content': head},
                            ],
                            'type': 'knowledge',
                            'source_ref': source_ref or f'corpus.db:{doc_id}',
                            'quality': 0.6,
                        }
                        self._add_dedup(self.knowledge, sample)
        finally:
            conn.close()

    # ── 去重 + 收集 ──
    def _add_dedup(self, bucket: list, sample: dict) -> None:
        key = sample['conversation'][-1]['content']
        h = hashlib.md5(key.encode()).hexdigest()
        if h in self.seen_hashes:
            return
        self.seen_hashes.add(h)
        bucket.append((sample['quality'], sample))

    def _write_jsonl(self, bucket: list, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'# generated={time.strftime("%Y-%m-%d %H:%M:%S")} '
                    f'samples={len(bucket)} by format_converter.py\n')
            for _, sample in bucket:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description='ILM Phase 1: format_converter')
    ap.add_argument('--state-db', default=os.environ.get('ILM_STATE_DB', os.path.expanduser('~/.hermes/state.db')))
    ap.add_argument('--corpus-db', default=os.environ.get('ILM_CORPUS_DB', os.path.expanduser('~/projects/isa/ilm/corpus.db')))
    ap.add_argument('--outdir', default=os.environ.get('ILM_TRAIN_DATA', os.path.expanduser('~/projects/isa/ilm/train_data')))
    ap.add_argument('--max-corrections', type=int, default=1500)
    ap.add_argument('--max-instruction', type=int, default=4000)
    ap.add_argument('--max-knowledge', type=int, default=2000)
    ap.add_argument('--session-limit', type=int, default=0,
                    help='只处理前N个session（0=全部）')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    fc = FormatConverter(args.state_db, args.corpus_db, args.outdir,
                         args.max_corrections, args.max_instruction,
                         args.max_knowledge, args.session_limit)
    stats = fc.run()

    print(f"✅ format_converter 完成 | 耗时 {stats['elapsed']}s")
    print(f"  sessions扫描: {stats['sessions_scanned']}")
    print(f"  corrections:  {stats['corrections']} (rejected+positive)")
    print(f"  instructions: {stats['instructions']}")
    print(f"  knowledge:    {stats['knowledge']}")
    print(f"  去重截断:     {stats['deduped']} 条")
    print(f"  输出目录:     {args.outdir}")

    # 抽样打印检查
    if args.verbose:
        for fname, n in [('corrections.jsonl', 3), ('instruction_pairs.jsonl', 2)]:
            path = os.path.join(args.outdir, fname)
            if not os.path.exists(path):
                continue
            print(f"\n=== {fname} 样本 ===")
            shown = 0
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    s = json.loads(line)
                    u = s['conversation'][0]['content'][:80].replace('\n', ' ')
                    a = s['conversation'][-1]['content'][:120].replace('\n', ' ')
                    tag = 'NEG' if s.get('rejected') else 'POS'
                    print(f"[{tag}] U: {u}")
                    print(f"       A: {a} | src={s['source_ref']}")
                    shown += 1
                    if shown >= n:
                        break


if __name__ == '__main__':
    main()
