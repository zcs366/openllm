"""
ISA 记忆免疫系统 — Sleeper威胁防御

Sleeper Memory Poisoning核心发现：
  - 注入率：GPT-5.5上99.8%，Kimi-K2.6上95%
  - 检索后恶意使用率：60-89%
  - 攻击可跨会话存活

ISA免疫层防御：
  1. 来源标注：每条记忆标注来源
  2. 信任分级：trusted/internal/untrusted/unknown
  3. 写入验证：untrusted来源需额外验证
  4. 行为审计：异常写入模式检测
  5. 跨会话隔离：不同来源的记忆不自动合并

不引入LLM推理——纯规则防御。零成本。
"""

import json
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import logging
from .causal_memory import TrustLevel

logger = logging.getLogger("openllm.immune")


class ThreatType(Enum):
    """威胁类型"""
    NONE = "none"
    UNTRUSTED_WRITE = "untrusted_write"           # untrusted来源写入
    RAPID_INJECTION = "rapid_injection"            # 短时间大量写入
    CROSS_SESSION_CARRY = "cross_session_carry"   # 跨会话携带
    ANOMALOUS_PATTERN = "anomalous_pattern"        # 异常模式


@dataclass
class AuditEntry:
    """审计条目"""
    timestamp: float = field(default_factory=time.time)
    action: str = ""          # write/read/delete/archival
    memory_id: str = ""
    source: str = ""
    trust_level: str = ""
    threat_type: str = "none"
    details: str = ""
    session_id: str = ""
    checksum: str = ""        # 写入签名


@dataclass
class WriteRequest:
    """写入请求"""
    content: dict
    source: str = ""
    trust_level: TrustLevel = TrustLevel.UNKNOWN
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)


class MemoryImmuneSystem:
    """
    记忆免疫系统。
    
    防御层次：
      L1: 来源标注（所有记忆必须标注来源）
      L2: 信任分级（trusted/internal/untrusted/unknown）
      L3: 写入验证（untrusted需额外检查）
      L4: 行为审计（异常写入模式检测）
      L5: 跨会话隔离（不同来源不自动合并）
    
    设计哲学：
      - 不引入LLM推理，纯规则防御
      - 默认保守（unknown不自动写入）
      - 审计链完整可追溯
    """
    
    def __init__(self, audit_dir: Optional[Path] = None):
        self.audit_dir = audit_dir or Path.home() / ".openllm" / "memory" / "audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        self._audit_log: List[AuditEntry] = []
        self._recent_writes: List[float] = []  # 最近写入时间戳
        self._blocked_count = 0
        self._allowed_count = 0
        self._session_writes: Dict[str, int] = {}  # session_id → 写入次数
        
        # 速率限制
        self.RAPID_THRESHOLD = 10     # 10次/分钟触发告警
        self.RAPID_WINDOW = 60        # 窗口：60秒
        
        # 加载审计日志
        self._load_audit_log()
        logger.info(f"🛡️ 记忆免疫系统初始化: {len(self._audit_log)}条审计记录")
    
    def _load_audit_log(self):
        """加载审计日志"""
        log_path = self.audit_dir / "audit.jsonl"
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = AuditEntry(**{k: v for k, v in data.items() if k in AuditEntry.__dataclass_fields__})
                    self._audit_log.append(entry)
                except Exception:
                    pass
    
    def _append_audit(self, entry: AuditEntry):
        """追加审计记录"""
        self._audit_log.append(entry)
        
        # 写入签名
        entry_str = json.dumps({
            "timestamp": entry.timestamp,
            "action": entry.action,
            "memory_id": entry.memory_id,
            "source": entry.source,
            "trust_level": entry.trust_level,
            "threat_type": entry.threat_type,
        }, ensure_ascii=False)
        entry.checksum = hashlib.sha256(entry_str.encode()).hexdigest()[:16]
        
        # 追加到审计日志文件
        log_path = self.audit_dir / "audit.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": entry.timestamp,
                "action": entry.action,
                "memory_id": entry.memory_id,
                "source": entry.source,
                "trust_level": entry.trust_level,
                "threat_type": entry.threat_type,
                "details": entry.details,
                "session_id": entry.session_id,
                "checksum": entry.checksum,
            }, ensure_ascii=False) + "\n")
    
    def check_write(self, request: WriteRequest) -> tuple[bool, ThreatType, str]:
        """
        检查写入请求是否允许。
        
        Returns:
            (allowed, threat_type, reason)
        """
        now = time.time()
        threats = []
        
        # L2: 信任分级检查
        trust_val = request.trust_level.value if hasattr(request.trust_level, 'value') else str(request.trust_level)
        if trust_val == "untrusted":
            threats.append(ThreatType.UNTRUSTED_WRITE)
        
        # L4: 速率检查
        self._recent_writes = [t for t in self._recent_writes if now - t < self.RAPID_WINDOW]
        if len(self._recent_writes) >= self.RAPID_THRESHOLD:
            threats.append(ThreatType.RAPID_INJECTION)
        
        # Session写入次数检查
        session_count = self._session_writes.get(request.session_id, 0)
        if session_count > 50:  # 单session写入超过50次
            threats.append(ThreatType.ANOMALOUS_PATTERN)
        
        # 判定
        if not threats:
            self._allowed_count += 1
            self._recent_writes.append(now)
            self._session_writes[request.session_id] = session_count + 1
            
            # 审计通过
            self._append_audit(AuditEntry(
                action="write_allowed",
                memory_id=hashlib.sha256(json.dumps(request.content, ensure_ascii=False).encode()).hexdigest()[:12],
                source=request.source,
                trust_level=request.trust_level.value,
                threat_type="none",
                session_id=request.session_id,
            ))
            return True, ThreatType.NONE, "允许写入"
        
        # 有威胁
        primary_threat = threats[0]
        reason_parts = []
        for t in threats:
            if t == ThreatType.UNTRUSTED_WRITE:
                reason_parts.append("untrusted来源写入")
            elif t == ThreatType.RAPID_INJECTION:
                reason_parts.append(f"速率异常({len(self._recent_writes)}次/{self.RAPID_WINDOW}s)")
            elif t == ThreatType.ANOMALOUS_PATTERN:
                reason_parts.append(f"异常写入模式(本session已{session_count}次)")
        
        reason = "拦截: " + "; ".join(reason_parts)
        
        self._blocked_count += 1
        
        # 审计拦截
        self._append_audit(AuditEntry(
            action="write_blocked",
            memory_id=hashlib.sha256(json.dumps(request.content, ensure_ascii=False).encode()).hexdigest()[:12],
            source=request.source,
            trust_level=request.trust_level.value,
            threat_type=primary_threat.value,
            details=reason,
            session_id=request.session_id,
        ))
        
        logger.warning(f"🚫 {reason}")
        return False, primary_threat, reason
    
    def classify_source(self, source_description: str) -> TrustLevel:
        """
        根据来源描述自动分级。
        
        关键词匹配（零LLM成本）。
        """
        desc = source_description.lower()
        
        if any(kw in desc for kw in ["user", "用户", "直接", "manual", "explicit"]):
            return TrustLevel.TRUSTED
        if any(kw in desc for kw in ["agent", "推理", "reasoning", "self", "internal"]):
            return TrustLevel.INTERNAL
        if any(kw in desc for kw in ["web", "网页", "搜索", "search", "document", "文档", 
                                      "email", "邮件", "repo", "外部"]):
            return TrustLevel.UNTRUSTED
        return TrustLevel.UNKNOWN
    
    def get_audit_summary(self, last_n: int = 20) -> str:
        """生成审计摘要"""
        recent = self._audit_log[-last_n:] if self._audit_log else []
        
        lines = ["## 记忆免疫审计摘要"]
        lines.append(f"总审计: {len(self._audit_log)} | 允许: {self._allowed_count} | 拦截: {self._blocked_count}")
        lines.append("")
        
        if recent:
            lines.append("最近审计:")
            for entry in recent[-10:]:
                emoji = "✅" if "allowed" in entry.action else "🚫"
                lines.append(f"  {emoji} [{entry.action}] trust={entry.trust_level} | {entry.source[:30]} | {entry.details[:40]}")
        
        # 威胁统计
        threats = [e for e in self._audit_log if e.threat_type != "none"]
        if threats:
            lines.append(f"\n威胁事件: {len(threats)}")
            threat_types = {}
            for t in threats:
                threat_types[t.threat_type] = threat_types.get(t.threat_type, 0) + 1
            for ttype, count in sorted(threat_types.items(), key=lambda x: -x[1]):
                lines.append(f"  {ttype}: {count}")
        
        return "\n".join(lines)
    
    def stats(self) -> dict:
        """统计信息"""
        return {
            "total_audit": len(self._audit_log),
            "allowed": self._allowed_count,
            "blocked": self._blocked_count,
            "block_rate": self._blocked_count / (self._allowed_count + self._blocked_count) 
                         if (self._allowed_count + self._blocked_count) > 0 else 0,
        }
