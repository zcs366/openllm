"""
ISN CapabilityPool — 能力池（pi思路落地）
==========================================

设计原则（2026-09-22 军令）：
  1. 池子在 openllm 代码外、上下文外——磁盘全量存 UnifiedSkillConfig json，零上下文成本。
  2. 用得到/经常用到才真正被装配（assemble）——装配单是唯一进上下文的通道。
  3. 以 Hermes 已有 tool/skill 为初始库存，但池子格式是 openLLM 自己的
     （UnifiedSkillConfig / ToolConfig 形态），不寄生 Hermes。

用法：
    from openllm.isn.capability_pool import CapabilityPool
    pool = CapabilityPool()                    # 默认 ~/.openllm/pool/
    pool.import_hermes_skills()                # 一次性：79个Hermes skill入池
    pool.upsert_tool("paper_knowledge", "...") # 库存武器入池
    hits = pool.search("论文知识检索", top_k=5)  # BM25式按需发现
    manifest = pool.assemble(["openllm-dogfood", "fcrawl"])  # 装配单→注入上下文
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# 池子根目录：能力目录≠用户数据，沙箱(HOME重定向)也该读到武器库清单。
# 默认 ~/.openllm/pool，OPENLLM_POOL_DIR 可覆盖（共享真池/测试隔离两用）。
# 在 CapabilityPool.__init__ 里实例化时解析（非导入期），env随时可设。
def _default_pool_dir() -> Path:
    env = os.environ.get("OPENLLM_POOL_DIR", "")
    return Path(env) if env else Path.home() / ".openllm" / "pool"
SKILLS_SUBDIR = "skills"
TOOLS_SUBDIR = "tools"

# 常用度计数文件（frequency信号：装过N次=经常用到→下次默认装配）
USAGE_DB = "usage.json"


@dataclass
class PoolEntry:
    """池子条目—— UnifiedSkillConfig 的池内投影（含工具）。"""
    kind: str                 # "skill" | "tool"
    name: str
    description: str
    version: str = "0.0.0"
    domain_tags: List[str] = field(default_factory=list)
    source_path: str = ""     # 正文/实现的真实路径（不复制正文，只留指针）
    source_framework: str = "hermes"   # hermes | openllm | external
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "kind": self.kind, "name": self.name,
            "description": self.description, "version": self.version,
            "domain_tags": self.domain_tags, "source_path": self.source_path,
            "source_framework": self.source_framework, "extra": self.extra,
        }

    @classmethod
    def from_json(cls, d: dict) -> "PoolEntry":
        return cls(kind=d.get("kind", "skill"), name=d["name"],
                   description=d.get("description", ""),
                   version=d.get("version", "0.0.0"),
                   domain_tags=d.get("domain_tags", []),
                   source_path=d.get("source_path", ""),
                   source_framework=d.get("source_framework", "hermes"),
                   extra=d.get("extra", {}))


def _tokenize(text: str) -> List[str]:
    """中英混合分词：英文按词、中文按2-gram。够BM25用，不引重依赖。"""
    text = (text or "").lower()
    en = re.findall(r"[a-z0-9_\-]{2,}", text)
    zh = re.findall(r"[\u4e00-\u9fff]+", text)
    zh_tokens: List[str] = []
    for seg in zh:
        if len(seg) == 1:
            zh_tokens.append(seg)
        else:
            zh_tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return en + zh_tokens


class CapabilityPool:
    """能力池：磁盘全量 + BM25式按需发现 + 装配单。线程安全。"""

    def __init__(self, pool_dir: Optional[Path] = None):
        self.pool_dir = Path(pool_dir) if pool_dir else _default_pool_dir()
        self.skills_dir = self.pool_dir / SKILLS_SUBDIR
        self.tools_dir = self.pool_dir / TOOLS_SUBDIR
        self.usage_path = self.pool_dir / USAGE_DB
        self._lock = threading.Lock()
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    # ── 存储 ─────────────────────────────────────────
    def _entry_path(self, kind: str, name: str) -> Path:
        sub = self.skills_dir if kind == "skill" else self.tools_dir
        safe = re.sub(r"[^A-Za-z0-9_\-\.]", "_", name)
        return sub / f"{safe}.json"

    def upsert(self, entry: PoolEntry) -> Path:
        p = self._entry_path(entry.kind, entry.name)
        with self._lock:
            p.write_text(json.dumps(entry.to_json(), ensure_ascii=False, indent=1),
                         encoding="utf-8")
        return p

    def get(self, kind: str, name: str) -> Optional[PoolEntry]:
        p = self._entry_path(kind, name)
        if not p.exists():
            return None
        return PoolEntry.from_json(json.loads(p.read_text(encoding="utf-8")))

    def list_all(self, kind: Optional[str] = None) -> List[PoolEntry]:
        out: List[PoolEntry] = []
        dirs = ([self.skills_dir] if kind in (None, "skill") else []) + \
               ([self.tools_dir] if kind in (None, "tool") else [])
        for d in dirs:
            for p in sorted(d.glob("*.json")):
                try:
                    out.append(PoolEntry.from_json(json.loads(p.read_text(encoding="utf-8"))))
                except Exception:
                    continue  # 坏条目跳过，不炸全池
        return out

    # ── Hermes 摄入 ──────────────────────────────────
    def import_hermes_skills(self, skills_dir: Optional[Path] = None,
                             force: bool = False) -> dict:
        """扫 Hermes skills/*/SKILL.md frontmatter → 入池。正文不复制，留指针。"""
        src = Path(skills_dir) if skills_dir else Path.home() / ".hermes" / "skills"
        imported, skipped = [], []
        if not src.exists():
            return {"imported": [], "skipped": [f"{src} 不存在"]}
        for skill_md in sorted(src.glob("*/SKILL.md")):
            try:
                raw = skill_md.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                skipped.append(f"{skill_md.parent.name}: {e}")
                continue
            meta = self._parse_frontmatter(raw)
            name = meta.get("name") or skill_md.parent.name
            if not meta.get("description"):
                skipped.append(f"{name}: 无description")
                continue
            if not force and self.get("skill", name) is not None:
                skipped.append(f"{name}: 已在池")
                continue
            tags = meta.get("tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in re.split(r"[,\[\]]", tags) if t.strip()]
            self.upsert(PoolEntry(
                kind="skill", name=name,
                description=meta.get("description", "")[:500],
                version=meta.get("version", "0.0.0"),
                domain_tags=tags,
                source_path=str(skill_md.parent),
                source_framework="hermes",
                extra={"has_linked_files": any(skill_md.parent.iterdir() and
                        (x.name not in ("SKILL.md", "__pycache__")) for x in skill_md.parent.iterdir())},
            ))
            imported.append(name)
        return {"imported": imported, "skipped": skipped}

    @staticmethod
    def _parse_frontmatter(raw: str) -> dict:
        """极简 YAML frontmatter 解析（name/description/version/tags 行级）。"""
        meta: Dict[str, Any] = {}
        if not raw.startswith("---"):
            return meta
        try:
            fm = raw.split("---")[1]
        except IndexError:
            return meta
        for line in fm.splitlines():
            m = re.match(r"^(\w[\w\-]*)\s*:\s*(.*)$", line)
            if not m:
                continue
            k, v = m.group(1), m.group(2).strip()
            if k in ("name", "description", "version"):
                meta[k] = v.strip("\"'")
            elif k == "tags":
                meta[k] = v
        return meta

    # ── BM25式搜索 ───────────────────────────────────
    def search(self, query: str, top_k: int = 5,
               kind: Optional[str] = None) -> List[dict]:
        entries = self.list_all(kind=kind)
        if not entries:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        docs = [_tokenize(f"{e.name} {e.description} {' '.join(e.domain_tags)}")
                for e in entries]
        n_docs = len(docs)
        avgdl = sum(len(d) for d in docs) / n_docs or 1.0
        df: Dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        k1, b = 1.5, 0.75
        scored = []
        for e, d in zip(entries, docs):
            score, tf_map = 0.0, {}
            for t in d:
                tf_map[t] = tf_map.get(t, 0) + 1
            for t in q_tokens:
                if t not in tf_map:
                    continue
                idf = math.log(1 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
                score += idf * (tf_map[t] * (k1 + 1)) / (
                    tf_map[t] + k1 * (1 - b + b * len(d) / avgdl))
            if score > 0:
                scored.append({"name": e.name, "kind": e.kind,
                               "description": e.description,
                               "source_path": e.source_path,
                               "score": round(score, 3)})
        scored.sort(key=lambda x: -x["score"])
        return scored[:top_k]

    # ── 装配（唯一进上下文的通道）────────────────────
    def assemble(self, names: List[str], max_chars: int = 1200) -> str:
        """按名单装配：name+一句话描述，紧凑清单形态。超预算截断并报数。"""
        usage = self._load_usage()
        lines, used = [], 0
        for n in names:
            e = self.get("skill", n) or self.get("tool", n)
            if e is None:
                continue
            line = f"- {e.name}: {e.description.split('。')[0][:80]}"
            if used + len(line) > max_chars:
                lines.append(f"... [装配预算{max_chars}字符已满，剩余条目截断]")
                break
            lines.append(line)
            used += len(line)
            usage[n] = usage.get(n, 0) + 1
        self._save_usage(usage)
        return "\n".join(lines)

    def frequent(self, min_count: int = 3, top_k: int = 8) -> List[str]:
        """经常用到的能力名单（frequency信号）。"""
        usage = self._load_usage()
        pairs = sorted(usage.items(), key=lambda x: -x[1])
        return [n for n, c in pairs if c >= min_count][:top_k]

    # ── 使用计数 ─────────────────────────────────────
    def _load_usage(self) -> dict:
        if self.usage_path.exists():
            try:
                return json.loads(self.usage_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_usage(self, usage: dict) -> None:
        with self._lock:
            self.usage_path.write_text(json.dumps(usage, ensure_ascii=False),
                                       encoding="utf-8")

    def stats(self) -> dict:
        skills = self.list_all(kind="skill")
        tools = self.list_all(kind="tool")
        return {"skills": len(skills), "tools": len(tools),
                "pool_dir": str(self.pool_dir)}
