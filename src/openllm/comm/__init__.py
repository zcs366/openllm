"""comm — 通信体：AgentComm 多agent可靠通信栈

把 ISA 通信体战役的四条腿组装成 agent 可直接使用的通信栈：
  - Registry 寻址 + CommAgent 收发（mailbox）
  - 裁决双闸门（send/deliver）—— governance.adjudication（P0-1）
  - 可靠会话 ACK/超时/重试 —— iai.session（P1-3 工程宝石）
  - 语义路由 broadcast —— P0-2（兴趣 × embedding/关键词）
  - comm.* 事件可见性 —— iai.event_bus

快速上手：
    reg = Registry()
    a = CommAgent("IAI", registry=reg)
    b = CommAgent("ISN", registry=reg, interests=["技能"])
    a.send("ISN", "技能库更新了", kind="tell")
    b.run_once()

设计文档/战役：ISA 通信体重启（七神终裁 2026-09-04）
"""
from openllm.comm.agent import (
    Registry,
    CommAgent,
    CommMessage,
    SendResult,
)

__all__ = [
    "Registry",
    "CommAgent",
    "CommMessage",
    "SendResult",
]
