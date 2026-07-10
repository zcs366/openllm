"""
openLLM 决策层 — IOS治理 + 仲裁 + 拒绝权

三体架构第二层：决策层 = "我决定什么"
- IOS：风险评估 + 仲裁 + 因果学习 + 拒绝权
"""
from .ios_impl import IOS

__all__ = ["IOS"]
