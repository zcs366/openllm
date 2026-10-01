"""test_query_builder — 检索 query 提炼层的钉子（2026-10-01）

覆盖：短句原样 / 长文提炼+主题词保留 / 中英混排 / 全停用词回退 /
空与None不抛异常 / 幂等 / isa_impl 接线（反证法目标用例）。
"""

import ast
import hashlib
import re
from pathlib import Path

import pytest

from openllm.isa.query_builder import build_query

REPO = Path(__file__).resolve().parents[1]

LONG_RAMBLE = (
    "嗯那个，我今天想跟你说一下啊，就是我们那个运输安全例会材料的事情吧，"
    "上次开会的时候领导说三超一疲劳的数据要重新统计一下，然后我就回去查了"
    "一下那个台账，发现里面的数据有点问题，就是有些司机的超速记录没有录进"
    "去，这个就很麻烦，因为月度报告下周就要交了，如果数据不对的话整个报告"
    "就白做了，所以我想着能不能帮我看看这个统计脚本哪里有问题，主要是那个"
    "疲劳驾驶的判定逻辑，我觉得阈值设得不对，应该是连续驾驶四小时就要预警，"
    "但是现在脚本里面写的是八小时，这样肯定不行嘛，还有那个超速的判定也是，"
    "高速和普通道路应该分开算，不能一刀切，你懂我意思吧。"
)


def test_short_text_returned_verbatim():
    """短句原样返回——不擅自改写。"""
    for s in ["今天天气怎么样？", "帮我看看这个bug", "openLLM 检索很烂"]:
        assert build_query(s) == s.strip()


def test_long_chinese_compressed_with_topic_terms():
    """≥200字口语废话 → ≤40%长度 且主题词在结果里。"""
    assert len(LONG_RAMBLE) >= 200
    q = build_query(LONG_RAMBLE)
    assert len(q) <= len(LONG_RAMBLE) * 0.4, f"提炼不够: {len(q)}/{len(LONG_RAMBLE)}"
    for topic in ("疲劳", "超速", "数据", "报告"):
        assert topic in q, f"主题词丢失: {topic} not in {q!r}"
    # 口语噪声词必须被过滤
    for noise in ("嗯", "那个", "你懂", "嘛"):
        assert noise not in q, f"噪声词残留: {noise} in {q!r}"
    # 主题词数量受 max_terms 约束
    assert len(q.split()) <= 12


def test_mixed_chinese_english():
    """中英混排：英文词形保留、小写化，两边停用词都被滤。"""
    text = ("Our retrieval pipeline in openLLM 的 recall_provider 检索效果很差，"
            "因为整条消息原文当 query 导致主题词被稀释，我们要加一层 query "
            "提炼用 jieba 分词去掉停用词，不然检索出来的东西完全不能用。")
    q = build_query(text)
    assert q and q != text
    assert "openllm" in q.lower()
    assert "检索" in q or "稀释" in q or "提炼" in q
    # 英文停用词 the/our/in 不得作为独立词出现
    words = set(w.strip("，。,.") for w in q.split())
    assert not words & {"the", "our", "in", "and", "because"}


def test_all_stopwords_falls_back_to_original():
    """全停用词/标点/空白 → 回退原文，不得空串。"""
    text = "的了吗呢啊呀哦，。！？；：、"
    out = build_query(text)
    assert out != ""
    assert out == text  # 短文本本就原样；即便长也绝不为空
    long_stop = ("我们的这个那个就是说了一下，也可以还是那么些事情，"
                 "不是就是这么的了对吧，其实也就那样了吧嗯啊哦。" * 4)
    out2 = build_query(long_stop)
    assert out2 != ""  # 若提炼为空也必须回退原文


@pytest.mark.parametrize("bad", [None, "", "   ", "\n\t ", 123, [], {}])
def test_empty_or_none_like_never_raises(bad):
    """空输入/None-like → 不抛异常。"""
    out = build_query(bad)
    assert isinstance(out, str)
    if bad in (None, 123, [], {}):
        assert out == ""


def test_idempotent_and_deterministic():
    """同一输入两次结果一致（纯函数，无随机）。"""
    a = build_query(LONG_RAMBLE)
    b = build_query(LONG_RAMBLE)
    assert a == b
    mixed = "Transformer attention head 的注意力机制在长上下文里会被噪声稀释。"
    assert build_query(mixed) == build_query(mixed)


# ── 反证法锚点：isa_impl.py 接线检查 ──────────────────────────
# 若把 isa_impl.py:128 改回 text=msg.text（或 build_query 退化为恒等），
# 本用例必须红——它钉住"检索词经过提炼层"这一事实，而非只钉函数本身。

def test_isa_impl_wires_build_query():
    src = (REPO / "src/openllm/core/isa_impl.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "BusQuery"]
    assert calls, "isa_impl.py 中找不到 BusQuery(...)"
    kw = {k.arg: k.value for k in calls[0].keywords}
    text_node = kw.get("text")
    assert isinstance(text_node, ast.Call), (
        "反证法：BusQuery(text=...) 必须是 build_query(...) 调用，"
        f"现在是 {ast.dump(text_node)} —— 整条消息原文直接当检索词的病灶复发"
    )
    assert getattr(text_node.func, "id", None) == "build_query"


def test_build_query_never_returns_empty_for_real_text():
    """契约总闸：任何非空输入 → 输出非空（空query=检索零结果）。"""
    corpus = [
        LONG_RAMBLE,
        "，。！？、；：""''（）",
        "the a an of to in for and",
        "a" * 500,
        "测" * 500,
        "Hello world this is a test of the emergency broadcast system " * 10,
    ]
    for text in corpus:
        out = build_query(text)
        assert isinstance(out, str) and out.strip() != "", f"返回空串: {text[:20]!r}"
