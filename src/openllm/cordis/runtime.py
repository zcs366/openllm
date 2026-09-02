"""Cordis runtime — 可逆效应代数 + 反应式共效应最小实现。

纯stdlib, <500行。形式化保证:
  - EffectContext: 副作用自带逆，卸载=逆序吞后悔药, 环境零残渣
  - CoeffectContext: 组件声明依赖, 依赖满足激活/撤回失效

设计依据: IAX-Cordis设计草案 T-D-1/T-D-2 (2026-09-02)
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

# ---------------------------------------------------------------------------
# 类型变量: 上下文泛型
# ---------------------------------------------------------------------------
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Effect — 副作用: apply 修改上下文, undo 是逆(后悔药)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Effect(Generic[T]):
    """一个可逆副作用: apply 修改上下文, undo 是它的逆。

    形式语义: Effect = Γ → Γ × (Γ → Γ)
    - apply: Γ → Γ     作用到上下文, 产出修改后的上下文
    - undo:  Γ → Γ     逆操作, 恢复到 apply 之前的状态
    """

    name: str
    apply: Callable[[T], T]
    undo: Callable[[T], T]

    def __repr__(self) -> str:
        return f"Effect({self.name!r})"


# ---------------------------------------------------------------------------
# EffectContext — 效应上下文: 记账 + 累加器 φ
# ---------------------------------------------------------------------------
class EffectContext(Generic[T]):
    """可逆效应上下文。

    核心不变量:
      - apply(e) 后, _log 末尾记录 e
      - undo_last() 弹出最近的 effect 并执行其 undo
      - undo_all() 逆序吞后悔药, 环境零残渣
      - snapshot() 返回当前状态的深拷贝
      - restore() 从快照恢复

    线程安全: 所有操作加锁, 适合双脑互保场景。
    """

    def __init__(self, ctx: T) -> None:
        self._ctx: T = ctx
        self._log: list[Effect[T]] = []
        self._lock = threading.Lock()

    # -- 核心操作 ---------------------------------------------------------

    def apply(self, effect: Effect[T]) -> None:
        """应用一个效应, 记入日志。"""
        with self._lock:
            self._ctx = effect.apply(self._ctx)
            self._log.append(effect)

    def undo_last(self) -> Effect[T]:
        """弹出并执行最后一个效应的逆。返回被撤销的 Effect。

        Raises:
            IndexError: 没有可撤销的效应时。
        """
        with self._lock:
            if not self._log:
                raise IndexError("undo_last: 无可撤销的效应")
            effect = self._log.pop()
            self._ctx = effect.undo(self._ctx)
            return effect

    def undo_all(self) -> list[Effect[T]]:
        """逆序吞后悔药: 按 LIFO 顺序撤销所有效应。

        Returns:
            被撤销的效应列表(逆序)。
        """
        undone: list[Effect[T]] = []
        while True:
            try:
                undone.append(self.undo_last())
            except IndexError:
                break
        return undone

    # -- 快照 -------------------------------------------------------------

    def snapshot(self) -> tuple[T, list[Effect[T]]]:
        """返回当前上下文和日志的深拷贝快照。

        快照是隔离的: 修改快照不影响原 EffectContext。
        """
        with self._lock:
            ctx_copy = copy.deepcopy(self._ctx)
            log_copy = list(self._log)
            return ctx_copy, log_copy

    def restore(self, snapshot: tuple[T, list[Effect[T]]]) -> None:
        """从快照恢复。

        Args:
            snapshot: 之前由 snapshot() 返回的 (ctx, log) 对。
        """
        with self._lock:
            self._ctx = copy.deepcopy(snapshot[0])
            self._log = list(snapshot[1])

    # -- 查询 -------------------------------------------------------------

    @property
    def context(self) -> T:
        """当前上下文(只读引用, 不拷贝)。"""
        return self._ctx

    @property
    def log(self) -> list[Effect[T]]:
        """当前效应日志(浅拷贝)。"""
        return list(self._log)

    @property
    def depth(self) -> int:
        """已应用但未撤销的效应数量。"""
        return len(self._log)

    def __repr__(self) -> str:
        return f"EffectContext(depth={self.depth})"


# ---------------------------------------------------------------------------
# Provider — 共效应声明: requires / provides / on_activate / on_deactivate
# ---------------------------------------------------------------------------
@dataclass
class Provider:
    """共效应提供者: 组件声明依赖和供给。

    形式语义:
      - requires: 组件需要的依赖键集合
      - provides: 组件提供的依赖键集合
      - on_activate: 依赖满足时的激活回调
      - on_deactivate: 依赖撤回时的失效回调
      - value: provide 携带的实际值
    """

    name: str
    requires: set[str] = field(default_factory=set)
    provides: set[str] = field(default_factory=set)
    on_activate: Callable[[], None] = field(default_factory=lambda: lambda: None)
    on_deactivate: Callable[[], None] = field(default_factory=lambda: lambda: None)
    value: Any = None

    def __repr__(self) -> str:
        return (
            f"Provider({self.name!r}, "
            f"requires={self.requires}, provides={self.provides})"
        )


# ---------------------------------------------------------------------------
# CoeffectContext — 依赖容器: provide / withdraw / resolve
# ---------------------------------------------------------------------------
class CoeffectContext:
    """反应式共效应上下文: 依赖的提供/撤回/解析。

    硬约束 (违反则 raise):
      - provide(key): key 已存在 → KeyError
      - withdraw(key): key 不存在 → KeyError

    支持组件激活/失效回调。
    """

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._lock = threading.Lock()

    def provide(self, key: str, provider: Provider) -> None:
        """提供一个依赖。

        Args:
            key: 依赖键(如 "config.db_url")。
            provider: 提供者声明。

        Raises:
            KeyError: key 已被提供(依赖不能提供两次)。
        """
        with self._lock:
            if key in self._providers:
                raise KeyError(
                    f"provide({key!r}): 依赖已被提供 "
                    f"by {self._providers[key].name!r}"
                )
            self._providers[key] = provider
            # 触发激活回调
            provider.on_activate()

    def withdraw(self, key: str) -> Provider:
        """撤回一个依赖, 触发失效回调。

        Args:
            key: 依赖键。

        Returns:
            被撤回的 Provider。

        Raises:
            KeyError: key 不存在(不能撤回不存在的依赖)。
        """
        with self._lock:
            if key not in self._providers:
                raise KeyError(
                    f"withdraw({key!r}): 依赖不存在, 无法撤回"
                )
            provider = self._providers.pop(key)
            # 触发失效回调
            provider.on_deactivate()
            return provider

    def resolve(self, key: str) -> Any:
        """解析依赖, 返回 Provider.value。

        Args:
            key: 依赖键。

        Returns:
            Provider.value 的值。

        Raises:
            KeyError: key 不存在。
        """
        with self._lock:
            if key not in self._providers:
                raise KeyError(
                    f"resolve({key!r}): 依赖不存在"
                )
            return self._providers[key].value

    # -- 查询 -------------------------------------------------------------

    def has(self, key: str) -> bool:
        """检查依赖是否存在。"""
        return key in self._providers

    def keys(self) -> list[str]:
        """返回所有已提供的依赖键。"""
        return list(self._providers.keys())

    def providers(self) -> dict[str, Provider]:
        """返回所有 Provider 的浅拷贝。"""
        return dict(self._providers)

    def __repr__(self) -> str:
        return f"CoeffectContext(keys={self.keys()})"


# ---------------------------------------------------------------------------
# 辅助: 快速创建 Effect 的工厂函数
# ---------------------------------------------------------------------------
def make_effect(
    name: str,
    apply_fn: Callable[[T], T],
    undo_fn: Callable[[T], T],
) -> Effect[T]:
    """快捷创建 Effect 的工厂函数。"""
    return Effect(name=name, apply=apply_fn, undo=undo_fn)
