#!/usr/bin/env python3
"""
OpenLLM 产出根 — 固定输出路径管理
iko铁律第①条：一个根，永不漂移。

用法:
    from openllm.core.output_root import OUT, iko, iai, isa, ios, isn, docs
    iko.ticks / "ticks.jsonl"
    docs.白皮书 / "xxx.md"
"""

from pathlib import Path

# ═══ 唯一产出根 ═══
OUT = Path.home() / ".openllm" / "output"
OUT.mkdir(parents=True, exist_ok=True)


class _SubDir:
    """子目录——惰性创建"""
    def __init__(self, *parts):
        self._path = OUT.joinpath(*parts)
    def __truediv__(self, other):
        p = self._path / other
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    def __str__(self): return str(self._path)
    def __repr__(self): return f"OutputDir({self._path})"
    @property
    def path(self): return self._path


# 各子系统产出目录
iko   = _SubDir("iko")
iai   = _SubDir("iai")
isa   = _SubDir("isa")
ios   = _SubDir("ios")
isn   = _SubDir("isn")
docs  = _SubDir("docs")

# 常用子路径
iko_ticks     = iko / "ticks"
iai_ckpt      = iai / "checkpoints"
iai_pred      = iai / "predictions"
docs_白皮书   = docs / "白皮书"
docs_立项书   = docs / "立项书"
docs_备忘录   = docs / "备忘录"
docs_PAL      = docs / "PAL"
docs_审计     = docs / "审计"
