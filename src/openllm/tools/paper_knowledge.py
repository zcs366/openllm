"""
论文知识库 — 研究agent的外部知识源。

核心能力:
1. 论文索引（标题/摘要/标签/关联）
2. 知识图谱（论文→概念→假说的关联网络）
3. 前沿追踪（最新论文→已有假说的映射）
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Paper:
    """一篇论文的结构化记录。"""
    id: str
    title: str
    authors: str = ""
    year: int = 0
    venue: str = ""  # arXiv/NeurIPS/ICML/...
    abstract: str = ""
    tags: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    related_hypotheses: list[str] = field(default_factory=list)  # hypothesis IDs
    url: str = ""
    added_at: float = field(default_factory=time.time)

    def dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "authors": self.authors,
            "year": self.year, "venue": self.venue, "tags": self.tags,
            "key_findings": self.key_findings,
            "related_hypotheses": self.related_hypotheses,
        }


@dataclass
class Concept:
    """一个研究概念。"""
    name: str
    definition: str = ""
    papers: list[str] = field(default_factory=list)  # paper IDs
    related: list[str] = field(default_factory=list)  # other concept names

    def dict(self) -> dict:
        return {"name": self.name, "definition": self.definition,
                "papers": self.papers, "related": self.related}


class PaperKnowledge:
    """
    研究agent的论文知识库。

    区别于搜索引擎：搜索引擎找论文，知识库理解论文。
    每篇论文不只是URL——是发现+概念+关联的结构化知识。
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.papers: dict[str, Paper] = {}
        self.concepts: dict[str, Concept] = {}
        self._storage_dir = storage_dir or Path.home() / ".openllm" / "research"
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        p_path = self._storage_dir / "papers.json"
        if p_path.exists():
            try:
                for p in json.loads(p_path.read_text()):
                    valid = set(Paper.__dataclass_fields__.keys())
                    paper = Paper(**{k: v for k, v in p.items() if k in valid})
                    self.papers[paper.id] = paper
            except Exception:
                pass

        c_path = self._storage_dir / "concepts.json"
        if c_path.exists():
            try:
                for c in json.loads(c_path.read_text()):
                    valid = set(Concept.__dataclass_fields__.keys())
                    concept = Concept(**{k: v for k, v in c.items() if k in valid})
                    self.concepts[concept.name] = concept
            except Exception:
                pass

    def _save(self):
        p_path = self._storage_dir / "papers.json"
        p_path.write_text(json.dumps(
            [p.dict() for p in self.papers.values()],
            indent=2, ensure_ascii=False
        ))
        c_path = self._storage_dir / "concepts.json"
        c_path.write_text(json.dumps(
            [c.dict() for c in self.concepts.values()],
            indent=2, ensure_ascii=False
        ))

    # ── 论文管理 ──

    def add_paper(self, id: str, title: str, **kwargs) -> Paper:
        """添加一篇论文。"""
        valid = set(Paper.__dataclass_fields__.keys())
        paper = Paper(id=id, title=title, **{k: v for k, v in kwargs.items() if k in valid})
        self.papers[id] = paper
        self._save()
        return paper

    def search(self, query: str) -> list[Paper]:
        """按关键词搜索论文。匹配title/abstract/tags/key_findings。"""
        q = query.lower()
        results = []
        for p in self.papers.values():
            if (q in p.title.lower()
                or q in p.abstract.lower()
                or any(q in t.lower() for t in (p.tags or []))
                or any(q in f.lower() for f in (p.key_findings or []))):
                results.append(p)
        return results

    def link_to_hypothesis(self, paper_id: str, hypothesis_id: str):
        """将论文关联到假说。"""
        if paper_id in self.papers:
            p = self.papers[paper_id]
            if hypothesis_id not in p.related_hypotheses:
                p.related_hypotheses.append(hypothesis_id)
                self._save()

    # ── 概念管理 ──

    def add_concept(self, name: str, definition: str = "", papers: Optional[list] = None) -> Concept:
        """添加一个研究概念。"""
        concept = Concept(name=name, definition=definition, papers=papers or [])
        self.concepts[name] = concept
        self._save()
        return concept

    def find_related(self, concept_name: str) -> list[str]:
        """找到与某概念相关的所有概念。"""
        if concept_name not in self.concepts:
            return []
        c = self.concepts[concept_name]
        related = set(c.related)
        # 也找引用同一篇论文的概念
        for other in self.concepts.values():
            if other.name != concept_name and set(other.papers) & set(c.papers):
                related.add(other.name)
        return list(related)

    # ── 分析 ──

    def summary(self) -> dict:
        return {
            "total_papers": len(self.papers),
            "total_concepts": len(self.concepts),
            "papers_by_tag": self._tag_distribution(),
            "recent_papers": [p.dict() for p in sorted(
                self.papers.values(), key=lambda x: x.added_at, reverse=True
            )[:5]],
        }

    def _tag_distribution(self) -> dict:
        tags = {}
        for p in self.papers.values():
            for t in p.tags:
                tags[t] = tags.get(t, 0) + 1
        return dict(sorted(tags.items(), key=lambda x: -x[1])[:10])
