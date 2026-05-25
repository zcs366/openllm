"""
OpenLLM — 大模型为自己建立的Agent躯体。

六个维度：
  记忆 OS     — Δ胶囊 + 仲裁 + 4层分层 + 检查点
  工具系统    — Shell/API/MCP（Phase 1）
  安全铠甲    — 3层权限门 + 审计 + 不可修改基座
  Agent Loop — plan→act→observe→reflect
  身份系统    — SOUL + Iam + 跨会话重建
  造物者对话  — 核心设计目标

Phase 0：Agent Loop + Memory + Identity + Security + CLI。
"""

__version__ = "0.1.0"
