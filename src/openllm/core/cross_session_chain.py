"""
IO-S 迷雾区 #3: 端到端长期链 — 跨Session审计追溯
====================================================

问题：现有AuditChain仅在单个session内工作。
当一个操作跨越多个session（如：session A的决策导致session B的执行，
session B的结果又影响session C的评估），无法追溯完整因果链。

MVP：CrossSessionChain
- 每个session创建genesis block，记录parent_session引用
- Action携带correlation_id跨session关联
- 链式验证：给定任何action，可追溯其完整因果祖先
- 跨session完整性报告

验收标准：
1. 新session启动时自动创建genesis block，引用parent
2. 跨session操作携带correlation_id
3. trace()函数可从任意action追溯到链头
4. verify()可验证跨session链的hash完整性
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional


class CrossSessionChain:
    """跨session因果追溯链。"""

    CHAIN_DIR = Path.home() / ".openllm" / "output" / "audit_chain" / "cross_session"
    GENESIS_MARKER = "GENESIS_v1"

    def __init__(self):
        self.CHAIN_DIR.mkdir(parents=True, exist_ok=True)

    def _chain_path(self, session_id: str) -> Path:
        safe = session_id.replace("/", "_").replace("\\", "_")
        return self.CHAIN_DIR / f"{safe}.jsonl"

    def _compute_hash(self, block: dict) -> str:
        """确定性hash。"""
        content = json.dumps({
            "block_id": block["block_id"],
            "session_id": block["session_id"],
            "action": block.get("action", ""),
            "correlation_id": block.get("correlation_id", ""),
            "parent_session": block.get("parent_session", ""),
            "parent_block_hash": block.get("parent_block_hash", ""),
            "prev_hash": block["prev_hash"],
            "timestamp": block["timestamp"],
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # ── 核心API ──────────────────────────────────────

    def create_session(self, session_id: str,
                       parent_session: Optional[str] = None) -> dict:
        """为新session创建genesis block。
        
        Args:
            session_id: 新session ID
            parent_session: 父session ID（可选，首次创建时不填）
        
        Returns: genesis block信息
        """
        parent_block_hash = ""
        if parent_session:
            parent_block_hash = self._get_last_hash(parent_session)

        genesis = {
            "block_id": f"{session_id}_genesis",
            "block_type": "genesis",
            "session_id": session_id,
            "parent_session": parent_session or "",
            "parent_block_hash": parent_block_hash or "",
            "action": "session_created",
            "correlation_id": "",
            "prev_hash": self.GENESIS_MARKER,
            "timestamp": time.time(),
        }
        genesis["_hash"] = self._compute_hash(genesis)

        path = self._chain_path(session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(genesis, ensure_ascii=False) + "\n")

        return {
            "session_id": session_id,
            "parent_session": parent_session,
            "block_hash": genesis["_hash"],
            "parent_block_hash": parent_block_hash,
        }

    def record_action(self, session_id: str, action: str,
                      correlation_id: str = "",
                      outcome: str = "", metadata: Optional[dict] = None) -> dict:
        """记录一个跨session action。
        
        Args:
            session_id: 当前session
            action: 操作描述
            correlation_id: 跨session关联ID（相同correlation_id = 同一任务链）
            outcome: 操作结果
            metadata: 额外元数据
        
        Returns: block信息
        """
        prev_hash = self._get_last_hash(session_id)

        block = {
            "block_id": f"{session_id}_action_{int(time.time()*1000)}",
            "block_type": "action",
            "session_id": session_id,
            "action": action[:200],
            "correlation_id": correlation_id,
            "outcome": outcome[:200],
            "parent_session": "",
            "parent_block_hash": "",
            "prev_hash": prev_hash,
            "timestamp": time.time(),
        }
        if metadata:
            block["metadata"] = {k: str(v)[:100] for k, v in metadata.items()}

        block["_hash"] = self._compute_hash(block)

        path = self._chain_path(session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(block, ensure_ascii=False) + "\n")

        return {"block_id": block["block_id"], "hash": block["_hash"]}

    def trace(self, session_id: str, block_hash: Optional[str] = None) -> list[dict]:
        """从任意block追溯完整因果链。
        
        如果指定了block_hash，从该block开始追溯。
        否则追溯整个session。
        
        Returns: 因果链列表（从当前到genesis，再跨session到父链）
        """
        blocks = self._read_chain(session_id)
        if not blocks:
            return []

        # 如果指定了block_hash，定位到该block
        if block_hash:
            idx = next(
                (i for i, b in enumerate(blocks) if b.get("_hash") == block_hash),
                len(blocks) - 1
            )
            blocks = blocks[:idx + 1]

        chain = list(reversed(blocks))  # 从当前到genesis

        # genesis块在reversed链的末尾，检查其parent_session
        if chain and chain[-1].get("parent_session"):
            parent = chain[-1]["parent_session"]
            parent_chain = self.trace(parent, chain[-1].get("parent_block_hash"))
            if parent_chain:
                chain = parent_chain + [{"===CROSS_SESSION_BOUNDARY===": parent}] + chain

        return chain

    def trace_by_correlation(self, correlation_id: str) -> list[dict]:
        """通过correlation_id追溯跨session的任务链。
        
        查找所有session中具有相同correlation_id的action。
        """
        results = []
        for path in sorted(self.CHAIN_DIR.glob("*.jsonl")):
            try:
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        block = json.loads(line)
                        if block.get("correlation_id") == correlation_id:
                            results.append(block)
            except Exception:
                continue

        return sorted(results, key=lambda b: b.get("timestamp", 0))

    def verify(self, session_id: str) -> dict:
        """验证单个session的跨session链完整性。"""
        blocks = self._read_chain(session_id)
        if not blocks:
            return {"valid": True, "total": 0, "broken_links": [], "tampered": []}

        broken = []
        tampered = []
        prev = self.GENESIS_MARKER

        for i, block in enumerate(blocks):
            if block.get("prev_hash") != prev:
                broken.append(i)
            recomputed = self._compute_hash(block)
            if recomputed != block.get("_hash"):
                tampered.append(i)
            prev = block.get("_hash", "")

        return {
            "valid": len(broken) == 0 and len(tampered) == 0,
            "total": len(blocks),
            "broken_links": broken,
            "tampered": tampered,
        }

    def verify_all(self) -> dict:
        """验证所有session链的完整性。"""
        results = {}
        for path in self.CHAIN_DIR.glob("*.jsonl"):
            sid = path.stem
            results[sid] = self.verify(sid)
        all_valid = all(r["valid"] for r in results.values())
        return {"all_valid": all_valid, "sessions": results}

    # ── 内部方法 ─────────────────────────────────────

    def _get_last_hash(self, session_id: str) -> str:
        """获取session链的最后一个hash。"""
        blocks = self._read_chain(session_id)
        if not blocks:
            return self.GENESIS_MARKER
        return blocks[-1].get("_hash", self.GENESIS_MARKER)

    def _read_chain(self, session_id: str) -> list[dict]:
        """读取session的完整链。"""
        path = self._chain_path(session_id)
        if not path.exists():
            return []
        blocks = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        blocks.append(json.loads(line))
        except Exception:
            pass
        return blocks

    def report(self) -> dict:
        """生成跨session审计报告。"""
        sessions = list(self.CHAIN_DIR.glob("*.jsonl"))
        total_blocks = 0
        for p in sessions:
            try:
                with open(p) as f:
                    total_blocks += sum(1 for line in f if line.strip())
            except Exception:
                pass
        return {
            "total_sessions": len(sessions),
            "total_blocks": total_blocks,
            "chain_dir": str(self.CHAIN_DIR),
        }


# ── 集成点 ──────────────────────────────────────────

_chain: Optional[CrossSessionChain] = None

def get_chain() -> CrossSessionChain:
    global _chain
    if _chain is None:
        _chain = CrossSessionChain()
    return _chain

def on_session_start(session_id: str, parent_session: Optional[str] = None) -> dict:
    """IO-S Phase 0 钩子：session启动时调用。"""
    return get_chain().create_session(session_id, parent_session)

def on_cross_session_action(session_id: str, action: str,
                            correlation_id: str = "", **kwargs) -> dict:
    """IO-S 跨session操作钩子。"""
    return get_chain().record_action(session_id, action, correlation_id, **kwargs)

def trace_action(session_id: str, block_hash: Optional[str] = None) -> list[dict]:
    """追溯任意action的完整因果链。"""
    return get_chain().trace(session_id, block_hash)
