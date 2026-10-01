"""配平钉子（D-20261001-OTB 动作1/2）——两处改动都是「等价提速」，所以钉子验的是**等价性**：

A. tokenize 缓存：缓存路径的输出必须与「旧写法的直算路径」**逐字节相同**；
   且返回的 set 必须是每次现建的独立对象（防共享可变集合被下游改写）。
B. health_check TTL：TTL 内只探一次；**失败绝不缓存**（可以少探，不可以漏报死）。

反证法口径：把改动拆回旧写法，下列断言必须变红。
"""
import logging
import time
from types import SimpleNamespace

import pytest

from openllm.iai import octopus as octo_mod
from openllm.isa import memory_bus
from openllm.isa.memory_bus import tokenize


# ══════════════════════════════════════════════
# A. tokenize 等价性
# ══════════════════════════════════════════════

SAMPLES = [
    "1+1等于几？",
    "遗忘比记忆更胜任——记忆与遗忘中产生理解",
    "MemoryBus is the single write entry for ISA",
    "memory 总线 MemoryBus 混合 MEMORY 词条",
    "",
    "   ",
    "，。！？——……",
    "a",
    "两个字的词",
    "Agent Harness 十二大模块完全解析：上下文、工具、记忆、治理",
    "无标点的一长串中文文本用来测试分词器的边界行为是否稳定",
    "Mixed 中英 text with 标点，逗号、句号。",
]


def _reference_tokenize(text: str) -> set:
    """旧写法的直算路径（不经过任何缓存）——作为等价性基准。"""
    if not text:
        return set()
    jb = memory_bus._get_jieba()
    if jb is not None and jb is not False:
        words = list(jb.cut(text.lower()))
    else:
        words = text.lower().split()
    return {w for w in words if len(w) > 1 and w.strip()}


@pytest.mark.parametrize("text", SAMPLES)
def test_tokenize_matches_uncached_reference(text):
    """逐字节等价：缓存路径 == 直算路径。"""
    assert tokenize(text) == _reference_tokenize(text)


def test_tokenize_is_identical_on_repeat():
    """同一输入重复调用（命中缓存）结果不变。"""
    t = SAMPLES[9]
    first = tokenize(t)
    for _ in range(5):
        assert tokenize(t) == first


def test_cached_set_is_not_shared_between_callers():
    """改动最怕的隐患：把缓存的 set 直接发出去 → 下游一改，全库都脏。"""
    t = SAMPLES[1]
    a = tokenize(t)
    a.add("被污染的标记")
    b = tokenize(t)
    assert "被污染的标记" not in b
    assert b == _reference_tokenize(t)


def test_cache_actually_hits():
    """证明确实走的是缓存：重复调用后 hits 上升（否则改动等于没做）。"""
    memory_bus._split_words.cache_clear()
    t = SAMPLES[8]
    tokenize(t)
    info1 = memory_bus._split_words.cache_info()
    tokenize(t)
    info2 = memory_bus._split_words.cache_info()
    assert info2.hits > info1.hits, "重复分词没有命中缓存——配平未生效"


# ══════════════════════════════════════════════
# B. health_check TTL
# ══════════════════════════════════════════════

class _FakeProvider:
    def __init__(self, available=True, boom=False):
        self._available = available
        self.boom = boom
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        if self.boom:
            raise RuntimeError("provider down")
        return "pong"


def _brain(provider):
    """不经 __init__（触手脑很重），直接造最小宿主对象调未绑定方法。"""
    return SimpleNamespace(left=SimpleNamespace(provider=provider),
                           _health_cache=None, _health_ts=0.0)


def test_health_check_caches_success_within_ttl():
    p = _FakeProvider()
    b = _brain(p)
    assert octo_mod.章鱼I.health_check(b) == "FULL"
    assert octo_mod.章鱼I.health_check(b) == "FULL"
    assert p.calls == 1, "TTL 内应只探活一次（账目：%d 次）" % p.calls


def test_health_check_reprobes_after_ttl(monkeypatch):
    monkeypatch.setattr(octo_mod, "_HEALTH_TTL_SEC", 0.0)
    p = _FakeProvider()
    b = _brain(p)
    octo_mod.章鱼I.health_check(b)
    octo_mod.章鱼I.health_check(b)
    assert p.calls == 2, "TTL 过期后必须重新探活"


def test_health_check_failure_is_never_cached():
    """红线：探活失败必须当次可见——不许把「死了」缓存成「还活着」。"""
    p = _FakeProvider(boom=True)
    b = _brain(p)
    assert octo_mod.章鱼I.health_check(b) == "DEGRADED"
    assert octo_mod.章鱼I.health_check(b) == "DEGRADED"
    assert p.calls == 2, "失败被缓存了——红线破了"
    assert b._health_cache is None


def test_health_check_unavailable_does_not_probe():
    p = _FakeProvider(available=False)
    b = _brain(p)
    assert octo_mod.章鱼I.health_check(b) == "MINIMAL"
    assert p.calls == 0


def test_failure_after_success_is_visible_immediately():
    """先成功后失败：失败必须立刻盖掉成功缓存（不能因为缓存了 FULL 而漏报）。"""
    p = _FakeProvider()
    b = _brain(p)
    assert octo_mod.章鱼I.health_check(b) == "FULL"
    p.boom = True
    b._health_ts = 0.0          # 模拟 TTL 过期（不依赖真实等待）
    assert octo_mod.章鱼I.health_check(b) == "DEGRADED"
