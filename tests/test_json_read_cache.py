"""读盘缓存钉子（配平动作2 后半：检索层重复读盘）。

病（实测无 profiler）：一次问询 memory_bus.query 2.14s，其中 delta_capsule 1.29s +
jiak 0.84s = 99%，两者都在每次 search 时重新读盘。
药：按 (路径, mtime_ns, 大小) 缓存。**唯一的真风险是陈旧**——所以钉子主打这一条：
文件改了必须立刻读到新内容（mtime 键控），而不是靠"猜时间"（原先 jiak 是 5 秒 TTL）。
"""
import json
import os

from openllm.isa.memory_bus import read_json_cached, _read_json_keyed


def _write(p, obj, mtime):
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.utime(p, (mtime, mtime))     # 显式设 mtime，保证测试可判（不靠时钟精度）


def test_reads_same_content_as_plain_json(tmp_path):
    p = tmp_path / "a.json"
    _write(p, {"k": [1, 2, 3], "中文": "值"}, 1_700_000_000)
    assert read_json_cached(p) == json.loads(p.read_text(encoding="utf-8"))


def test_file_change_is_picked_up_immediately(tmp_path):
    """核心钉子：文件一改立刻重读——不允许陈旧（这正是原先 5 秒 TTL 的老毛病）。"""
    p = tmp_path / "a.json"
    _write(p, {"v": 1}, 1_700_000_000)
    assert read_json_cached(p)["v"] == 1
    _write(p, {"v": 2}, 1_700_000_060)
    assert read_json_cached(p)["v"] == 2, "改了文件却读到旧内容＝陈旧"
    _write(p, {"v": 3}, 1_700_000_120)
    assert read_json_cached(p)["v"] == 3


def test_repeat_read_hits_cache(tmp_path):
    p = tmp_path / "a.json"
    _write(p, {"v": 1}, 1_700_000_000)
    read_json_cached(p)                       # 先预热
    before = _read_json_keyed.cache_info().hits
    for _ in range(5):
        read_json_cached(p)
    assert _read_json_keyed.cache_info().hits >= before + 5


def test_missing_file_returns_none(tmp_path):
    assert read_json_cached(tmp_path / "nope.json") is None


def test_corrupt_json_returns_none_instead_of_raising(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{不是合法 json", encoding="utf-8")
    assert read_json_cached(p) is None


def test_same_size_different_mtime_is_a_different_cache_key(tmp_path):
    """同大小同路径但 mtime 不同 → 必须是两个键（否则 mtime 白设）。"""
    p = tmp_path / "a.json"
    _write(p, {"v": 1}, 1_700_000_000)
    read_json_cached(p)
    _write(p, {"v": 9}, 1_700_000_099)      # 长度与上面相同
    assert read_json_cached(p)["v"] == 9
