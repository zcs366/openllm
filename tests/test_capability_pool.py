"""test_capability_pool — 能力池钉子测试（tmp池目录，永不污染真池）"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openllm.isn.capability_pool import CapabilityPool, PoolEntry


@pytest.fixture()
def pool(tmp_path):
    return CapabilityPool(pool_dir=tmp_path / "pool")


def test_upsert_get_roundtrip(pool):
    pool.upsert(PoolEntry(kind="skill", name="foo", description="测试技能",
                          domain_tags=["t1"], source_path="/x/y"))
    e = pool.get("skill", "foo")
    assert e is not None and e.description == "测试技能" and e.domain_tags == ["t1"]
    assert pool.get("skill", "nope") is None


def test_list_all_and_bad_json_skipped(pool):
    pool.upsert(PoolEntry(kind="tool", name="t1", description="d"))
    pool.upsert(PoolEntry(kind="skill", name="s1", description="d"))
    (pool.tools_dir / "broken.json").write_text("{bad json", encoding="utf-8")
    all_entries = pool.list_all()
    assert len(all_entries) == 2  # 坏条目跳过不炸
    assert {e.kind for e in all_entries} == {"tool", "skill"}


def test_search_chinese_and_english(pool):
    pool.upsert(PoolEntry(kind="tool", name="paper_knowledge",
                          description="论文知识库检索"))
    pool.upsert(PoolEntry(kind="skill", name="web-fetch",
                          description="免费通用网页抓取器 web fetch"))
    hits = pool.search("论文 检索", top_k=2)
    assert hits and hits[0]["name"] == "paper_knowledge"
    hits2 = pool.search("web fetch", top_k=2)
    assert hits2 and hits2[0]["name"] == "web-fetch"


def test_assemble_format_and_budget(pool):
    pool.upsert(PoolEntry(kind="tool", name="fcrawl", description="网页爬虫。六大功能"))
    m = pool.assemble(["fcrawl"])
    assert m.startswith("- fcrawl:") and "六大功能" not in m  # 只取第一句
    pool.upsert(PoolEntry(kind="skill", name="a" * 200, description="x" * 100))
    m2 = pool.assemble(["a" * 200], max_chars=50)
    assert "截断" in m2


def test_usage_frequency_signal(pool):
    pool.upsert(PoolEntry(kind="tool", name="hot", description="d"))
    for _ in range(3):
        pool.assemble(["hot"])
    assert pool.frequent(min_count=3) == ["hot"]
    usage = json.loads((pool.pool_dir / "usage.json").read_text(encoding="utf-8"))
    assert usage["hot"] == 3


def test_import_hermes_skills_from_fake_dir(pool, tmp_path):
    fake = tmp_path / "skills" / "demo-skill"
    fake.mkdir(parents=True)
    (fake / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: 演示技能\nversion: 1.2.0\n---\n正文",
        encoding="utf-8")
    r = pool.import_hermes_skills(skills_dir=tmp_path / "skills")
    assert r["imported"] == ["demo-skill"]
    e = pool.get("skill", "demo-skill")
    assert e.version == "1.2.0" and e.source_framework == "hermes"
    # 幂等：重跑不重复入池
    r2 = pool.import_hermes_skills(skills_dir=tmp_path / "skills")
    assert r2["imported"] == [] and "已在池" in r2["skipped"][0]


def test_entry_json_roundtrip():
    e = PoolEntry(kind="tool", name="n", description="d", version="1.0",
                  domain_tags=["a"], source_path="/p", source_framework="openllm",
                  extra={"status": "active"})
    e2 = PoolEntry.from_json(e.to_json())
    assert e2 == e
