"""
openLLM MCP Memory Server
=========================
将 openLLM 的记忆系统（Δ胶囊 + 因果记忆）通过 MCP 协议暴露给外部 Agent。
stdio JSON-RPC 传输，外部 Agent（Hermes/Claude Code/Codex）可调用。

三个工具：
  memory_write  — 写入一条记忆（同时写入 Δ胶囊 + 因果库）
  memory_read   — 按 session_id 或 query 读取记忆
  memory_search — 关键词/标签搜索记忆

启动方式：
  cd /home/zcs/projects/openllm
  PYTHONPATH=src .venv/bin/python -m openllm.bridge.mcp_memory_server
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── 路径处理：让 PYTHONPATH src 正确 import openllm ──
_project_root = Path(__file__).resolve().parents[2]  # src/openllm/bridge/ → project root
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ── openLLM 内部模块 ──
from openllm.memory.capsule import (
    MemoryOS,
    TextCapsule,
    DeltaCapsule,
    CAPSULE_DIR,
)
from openllm.memory.causal_memory import (
    CausalMemoryStore,
    TrustLevel,
    get_causal_store,
)
from openllm.memory.temperature_engine import (
    calculate_temperature,
    temperature_state,
)

# ── MCP v2 ──
from mcp.server.mcpserver import MCPServer


# ═══════════════════════════════════════════════════════════════
# 初始化存储后端
# ═══════════════════════════════════════════════════════════════

# Δ胶囊：直接读写现有 capsules 目录
_capsule_dir = Path(CAPSULE_DIR).expanduser()
_memory_os = MemoryOS(capsule_dir=_capsule_dir)

# 因果记忆：全局单实例
_causal_store: Optional[CausalMemoryStore] = None


def _get_causal() -> CausalMemoryStore:
    global _causal_store
    if _causal_store is None:
        _causal_store = get_causal_store()
    return _causal_store


# ═══════════════════════════════════════════════════════════════
# MCP Server
# ═══════════════════════════════════════════════════════════════

server = MCPServer(
    name="openllm-memory",
    description="openLLM 记忆服务器 — 暴露 Δ胶囊 + 因果记忆的读写搜索接口",
)


def _capsule_to_entry(data: dict) -> dict:
    """将 v06 胶囊 dict 转为标准记忆条目格式。"""
    ts = data.get("timestamp", 0)
    imp = 5.0
    acc = 0
    for d in data.get("decisions", []):
        meta = d.get("metadata", {})
        if "importance" in meta and meta["importance"] is not None:
            imp = float(meta["importance"])
        if "access_count" in meta:
            acc = int(meta["access_count"])

    now = time.time()
    days = max(0, int((now - ts) / 86400)) if ts else 0
    temp = calculate_temperature(imp, days, "insight", acc)
    state = temperature_state(temp)

    insights = data.get("insights", [])
    content_parts = []
    for d in data.get("decisions", []):
        s = d.get("summary", str(d))
        if s:
            content_parts.append(s)
    for i in insights:
        if i:
            content_parts.append(i)
    content = " | ".join(content_parts) if content_parts else data.get("session_id", "")

    return {
        "memory_id": data.get("session_id", ""),
        "content": content,
        "temperature": temp,
        "state": state,
        "importance": imp,
        "access_count": acc,
        "created_at": ts,
        "source": "capsule",
        "outputs": data.get("outputs", []),
        "unresolved": data.get("unresolved", []),
    }


def _causal_to_entry(mem) -> dict:
    """将 CausalMemory 对象转为标准记忆条目格式。"""
    temp = mem.temperature()
    state = temperature_state(temp)
    return {
        "memory_id": mem.memory_id,
        "content": mem.lesson or mem.delta or mem.actual_result,
        "action": mem.action_signature,
        "prediction": mem.prediction,
        "actual": mem.actual_result,
        "delta": mem.delta,
        "temperature": round(temp, 2),
        "state": state,
        "importance": mem.importance,
        "created_at": mem.created_at,
        "source": "causal",
        "trust": mem.trust_level.value,
        "tags": mem.tags,
        "session_id": mem.session_id,
    }


def _keyword_in_capsule(data: dict, keyword: str) -> bool:
    """简单关键词匹配：检查胶囊文本内容是否包含关键词。"""
    kw = keyword.lower()
    for d in data.get("decisions", []):
        s = str(d.get("summary", d)).lower()
        if kw in s:
            return True
    for i in data.get("insights", []):
        if kw in str(i).lower():
            return True
    for o in data.get("outputs", []):
        if kw in str(o).lower():
            return True
    return False


# ═══════════════════════════════════════════════════════════════
# Tool: memory_write
# ═══════════════════════════════════════════════════════════════

@server.tool(
    name="memory_write",
    description="写入一条记忆到 openLLM 记忆系统（Δ胶囊 + 因果记忆）",
)
def memory_write(
    content: str,
    importance: float = 5.0,
    memory_type: str = "insight",
    tags: list[str] | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """写入记忆。

    Args:
        content: 记忆内容文本
        importance: 重要性 1-10 (默认5)
        memory_type: 记忆类型 insight/preference/event/noise
        tags: 可选标签列表
        session_id: 可选会话ID
    """
    ts = time.time()
    sid = session_id or f"mcp_{int(ts * 1000)}"
    importance = max(1.0, min(10.0, importance))

    # 1. 写入 Δ胶囊
    text_capsule = TextCapsule(
        session_id=sid,
        timestamp=ts,
        decisions=[{"summary": content, "metadata": {"importance": importance}}],
        insights=[content] if memory_type == "insight" else [],
    )
    capsule_path = _memory_os.write(text_capsule)

    # 2. 写入因果记忆
    causal = _get_causal()
    mem = causal.store(
        action_signature=f"mcp_write:{content[:50]}",
        context_features=tags or [],
        prediction="",
        prediction_confidence=0.0,
        actual_result=content,
        actual_success=True,
        delta="",
        delta_magnitude=0.0,
        lesson=content,
        source="mcp_memory_server",
        trust_level=TrustLevel.TRUSTED,
        importance=importance / 10.0,  # 归一化到 0-1
        tags=tags or [],
        session_id=sid,
    )

    temp = calculate_temperature(importance, 0, memory_type)
    state = temperature_state(temp)

    return {
        "memory_id": mem.memory_id,
        "session_id": sid,
        "capsule_path": capsule_path,
        "temperature": temp,
        "state": state,
        "importance": importance,
        "created_at": ts,
        "status": "ok",
    }


# ═══════════════════════════════════════════════════════════════
# Tool: memory_read
# ═══════════════════════════════════════════════════════════════

@server.tool(
    name="memory_read",
    description="读取记忆条目（按 session_id 或 query）",
)
def memory_read(
    query: str = "",
    session_id: str = "",
    max_results: int = 10,
) -> dict[str, Any]:
    """读取记忆。

    Args:
        query: 搜索查询文本（关键词匹配胶囊内容）
        session_id: 指定会话ID读取该会话的所有记忆
        max_results: 最大返回条数
    """
    entries: list[dict] = []

    # 路径1: 按 session_id 读取胶囊
    if session_id:
        # 直接按文件名读取特定 session 的胶囊
        capsule_file = _capsule_dir / f"v06_{session_id}.json"
        if capsule_file.exists():
            try:
                data = json.loads(capsule_file.read_text(encoding="utf-8"))
                entries.append(_capsule_to_entry(data))
            except Exception:
                pass
        # 如果没找到指定 session，读最新胶囊作为 fallback
        if not entries:
            capsule_data = _memory_os.read()
            if capsule_data and capsule_data.get("status") != "empty":
                entries.append(_capsule_to_entry(capsule_data))

        # 同时查因果记忆
        causal = _get_causal()
        causal_mems = causal.get_by_session(session_id)
        for m in causal_mems[:max_results]:
            entries.append(_causal_to_entry(m))

    # 路径2: 按 query 搜索
    elif query:
        # 搜索胶囊
        capsule_files = sorted(
            _capsule_dir.glob("v06_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in capsule_files[:50]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if _keyword_in_capsule(data, query):
                    entries.append(_capsule_to_entry(data))
                    if len(entries) >= max_results:
                        break
            except Exception:
                continue

        # 搜索因果记忆
        causal = _get_causal()
        kw_list = [w.strip() for w in query.split() if w.strip()]
        causal_mems = causal.search(
            context_features=kw_list,
            tags=kw_list,
            max_results=max_results,
        )
        for m in causal_mems:
            entries.append(_causal_to_entry(m))

    else:
        # 无参数：返回最新胶囊
        capsule_data = _memory_os.read()
        if capsule_data and capsule_data.get("status") != "empty":
            entries.append(_capsule_to_entry(capsule_data))

    # 去重（按 memory_id）
    seen = set()
    unique = []
    for e in entries:
        mid = e.get("memory_id", "")
        if mid and mid not in seen:
            seen.add(mid)
            unique.append(e)
        elif not mid:
            unique.append(e)

    return {
        "total": len(unique[:max_results]),
        "entries": unique[:max_results],
    }


# ═══════════════════════════════════════════════════════════════
# Tool: memory_search
# ═══════════════════════════════════════════════════════════════

@server.tool(
    name="memory_search",
    description="关键词/标签搜索记忆（搜索 Δ胶囊 + 因果记忆）",
)
def memory_search(
    keyword: str,
    max_results: int = 10,
) -> dict[str, Any]:
    """搜索记忆。

    Args:
        keyword: 搜索关键词
        max_results: 最大返回条数
    """
    entries: list[dict] = []

    # 搜索胶囊
    capsule_files = sorted(
        _capsule_dir.glob("v06_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for f in capsule_files[:100]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if _keyword_in_capsule(data, keyword):
                entries.append(_capsule_to_entry(data))
        except Exception:
            continue

    # 搜索因果记忆
    causal = _get_causal()
    kw_list = [w.strip() for w in keyword.split() if w.strip()]
    causal_mems = causal.search(
        context_features=kw_list,
        tags=kw_list,
        max_results=max_results,
    )
    for m in causal_mems:
        entries.append(_causal_to_entry(m))

    # 去重 + 截断
    seen = set()
    unique = []
    for e in entries:
        mid = e.get("memory_id", "")
        if mid and mid not in seen:
            seen.add(mid)
            unique.append(e)
        elif not mid:
            unique.append(e)
        if len(unique) >= max_results:
            break

    return {
        "total": len(unique),
        "keyword": keyword,
        "entries": unique,
    }


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    """stdio MCP 服务器入口。"""
    import asyncio
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
