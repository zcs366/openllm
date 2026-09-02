"""Cordis runtime 单元测试。

覆盖:
  1. Effect 应用→undo 往返一致
  2. 多 effect undo_all 逆序恢复
  3. CoeffectContext provide 重复 raise / withdraw 不存在 raise / resolve
  4. 写文件 Effect: apply 写文件 → undo_all 恢复原状 (M2 验收标准)
  5. snapshot / restore 隔离验证
  6. Provider 激活/失效回调
  7. 边界: undo 空栈 IndexError
  8. 线程安全
"""

import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

from openllm.cordis.runtime import (
    CoeffectContext,
    Effect,
    EffectContext,
    Provider,
    make_effect,
)


# ===========================================================================
# 辅助: 简单的 dict 上下文 Effect
# ===========================================================================

_SENTINEL = object()


def _set_effect(key: str, value: object) -> Effect[dict[str, Any]]:
    """创建一个设置 dict[key]=value 的 Effect, undo 恢复原值。"""
    _original: dict[str, Any] = {}

    def apply_fn(d: dict[str, Any]) -> dict[str, Any]:
        d = dict(d)
        _original[key] = d.get(key, _SENTINEL)
        d[key] = value
        return d

    def undo_fn(d: dict[str, Any]) -> dict[str, Any]:
        d = dict(d)
        orig = _original.get(key, _SENTINEL)
        if orig is _SENTINEL:
            d.pop(key, None)
        else:
            d[key] = orig
        return d

    return make_effect(name=f"set_{key}", apply_fn=apply_fn, undo_fn=undo_fn)


def _add_counter_effect(n: int) -> Effect[dict[str, Any]]:
    """创建一个 counter += n 的 Effect, undo 恢复原值。"""
    _original: dict[str, Any] = {}

    def apply_fn(d: dict[str, Any]) -> dict[str, Any]:
        d = dict(d)
        _original["counter"] = d.get("counter", _SENTINEL)
        d["counter"] = d.get("counter", 0) + n
        return d

    def undo_fn(d: dict[str, Any]) -> dict[str, Any]:
        d = dict(d)
        orig = _original.get("counter", _SENTINEL)
        if orig is _SENTINEL:
            d.pop("counter", None)
        else:
            d["counter"] = orig
        return d

    return make_effect(name=f"add_{n}", apply_fn=apply_fn, undo_fn=undo_fn)


# ===========================================================================
# 1. Effect 应用→undo 往返一致
# ===========================================================================


class TestEffectRoundTrip:
    """Effect apply + undo 应恢复到原始状态。"""

    def test_single_effect_round_trip(self) -> None:
        ctx = EffectContext({"x": 1, "y": 2})
        e = _set_effect("x", 42)
        ctx.apply(e)
        assert ctx.context["x"] == 42
        ctx.undo_last()
        assert ctx.context["x"] == 1
        assert ctx.context["y"] == 2

    def test_multiple_effects_sequential_undo(self) -> None:
        ctx: EffectContext[dict[str, Any]] = EffectContext({"counter": 0})
        ctx.apply(_add_counter_effect(10))
        assert ctx.context["counter"] == 10
        ctx.apply(_add_counter_effect(20))
        assert ctx.context["counter"] == 30
        ctx.undo_last()
        assert ctx.context["counter"] == 10
        ctx.undo_last()
        assert ctx.context["counter"] == 0

    def test_counter_add_effect_preserves_other_keys(self) -> None:
        ctx: EffectContext[dict[str, Any]] = EffectContext({"x": 1, "y": 2})
        ctx.apply(_add_counter_effect(5))
        assert ctx.context["counter"] == 5
        assert ctx.context["x"] == 1
        assert ctx.context["y"] == 2
        # undo_all 恢复: counter 回到不存在(undo 恢复原始值 _SENTINEL → 删除)
        undone = ctx.undo_all()
        assert len(undone) == 1
        assert "counter" not in ctx.context
        assert ctx.context["x"] == 1
        assert ctx.context["y"] == 2


# ===========================================================================
# 2. 多 effect undo_all 逆序恢复
# ===========================================================================


class TestUndoAllReverseOrder:
    """undo_all 必须逆序吞后悔药。"""

    def test_three_effects_undo_all(self) -> None:
        ctx: EffectContext[dict[str, Any]] = EffectContext({"x": 1})
        ctx.apply(_set_effect("x", "a"))
        ctx.apply(_set_effect("x", "b"))
        ctx.apply(_set_effect("x", "c"))
        assert ctx.context["x"] == "c"
        assert ctx.depth == 3
        undone = ctx.undo_all()
        assert len(undone) == 3
        assert ctx.context["x"] == 1
        assert ctx.depth == 0

    def test_five_counter_effects_undo_all(self) -> None:
        """五个 counter 效果, undo_all 逆序恢复。"""
        ctx: EffectContext[dict[str, Any]] = EffectContext({"counter": 0})
        effects = []
        for _ in range(5):
            e = _add_counter_effect(1)
            effects.append(e)
            ctx.apply(e)
        assert ctx.context["counter"] == 5

        undone = ctx.undo_all()
        assert len(undone) == 5
        # 验证 LIFO: 最后 apply 的最先 undo
        assert undone == effects[::-1]
        assert ctx.context["counter"] == 0

    def test_undo_all_from_empty(self) -> None:
        ctx = EffectContext("hello")
        undone = ctx.undo_all()
        assert undone == []
        assert ctx.context == "hello"


# ===========================================================================
# 3. CoeffectContext 依赖管理
# ===========================================================================


class TestCoeffectContext:
    """provide / withdraw / resolve 硬约束。"""

    def test_provide_and_resolve(self) -> None:
        cc = CoeffectContext()
        p = Provider(name="db", value="postgres://localhost/test")
        cc.provide("config.db_url", p)
        assert cc.resolve("config.db_url") == "postgres://localhost/test"
        assert cc.has("config.db_url")

    def test_provide_duplicate_raises(self) -> None:
        cc = CoeffectContext()
        cc.provide("k", Provider(name="a", value=1))
        with pytest.raises(KeyError, match="已被提供"):
            cc.provide("k", Provider(name="b", value=2))

    def test_withdraw_nonexistent_raises(self) -> None:
        cc = CoeffectContext()
        with pytest.raises(KeyError, match="不存在"):
            cc.withdraw("no_such_key")

    def test_resolve_nonexistent_raises(self) -> None:
        cc = CoeffectContext()
        with pytest.raises(KeyError, match="不存在"):
            cc.resolve("no_such_key")

    def test_withdraw_removes_provider(self) -> None:
        cc = CoeffectContext()
        p = Provider(name="tmp", value=42)
        cc.provide("x", p)
        withdrawn = cc.withdraw("x")
        assert withdrawn is p
        assert not cc.has("x")
        assert cc.keys() == []

    def test_provide_withdraw_reprovide(self) -> None:
        """withdraw 后可以重新 provide 同一个 key。"""
        cc = CoeffectContext()
        cc.provide("k", Provider(name="a", value=1))
        cc.withdraw("k")
        cc.provide("k", Provider(name="b", value=2))
        assert cc.resolve("k") == 2

    def test_multiple_keys(self) -> None:
        cc = CoeffectContext()
        cc.provide("a", Provider(name="pa", value="alpha"))
        cc.provide("b", Provider(name="pb", value="beta"))
        cc.provide("c", Provider(name="pc", value="gamma"))
        assert set(cc.keys()) == {"a", "b", "c"}
        assert cc.resolve("b") == "beta"


# ===========================================================================
# 4. Provider 激活/失效回调
# ===========================================================================


class TestProviderCallbacks:
    """on_activate / on_deactivate 回调触发验证。"""

    def test_activate_called_on_provide(self) -> None:
        called: list[str] = []
        p = Provider(
            name="cb",
            value=1,
            on_activate=lambda: called.append("activate"),
            on_deactivate=lambda: called.append("deactivate"),
        )
        cc = CoeffectContext()
        cc.provide("k", p)
        assert called == ["activate"]

    def test_deactivate_called_on_withdraw(self) -> None:
        called: list[str] = []
        p = Provider(
            name="cb",
            value=1,
            on_activate=lambda: called.append("activate"),
            on_deactivate=lambda: called.append("deactivate"),
        )
        cc = CoeffectContext()
        cc.provide("k", p)
        cc.withdraw("k")
        assert called == ["activate", "deactivate"]

    def test_no_activate_on_duplicate_provide(self) -> None:
        """重复 provide 应 raise, 不触发第二次 activate。"""
        called: list[str] = []
        p1 = Provider(name="a", value=1, on_activate=lambda: called.append("a1"))
        p2 = Provider(name="b", value=2, on_activate=lambda: called.append("a2"))
        cc = CoeffectContext()
        cc.provide("k", p1)
        with pytest.raises(KeyError):
            cc.provide("k", p2)
        assert called == ["a1"]


# ===========================================================================
# 5. 写文件 Effect: apply 写文件 → undo_all 恢复原状 (M2 验收标准)
# ===========================================================================


class TestFileEffectUndo:
    """最小演示: 写文件 Effect, undo_all 后文件恢复原状。"""

    def test_write_file_undo_restores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            original_content = "hello world\n"
            filepath.write_text(original_content)

            new_content = "MODIFIED CONTENT\n"

            def file_apply(_ctx: dict[str, Any]) -> dict[str, Any]:
                filepath.write_text(new_content)
                return {"written": True}

            def file_undo(_ctx: dict[str, Any]) -> dict[str, Any]:
                filepath.write_text(original_content)
                return {"written": False}

            e = Effect(name="write_file", apply=file_apply, undo=file_undo)
            ctx: EffectContext[dict[str, Any]] = EffectContext({"written": False})

            ctx.apply(e)
            assert filepath.read_text() == new_content
            assert ctx.context["written"] is True

            undone = ctx.undo_all()
            assert len(undone) == 1
            assert filepath.read_text() == original_content
            assert ctx.context["written"] is False

    def test_multiple_file_effects_undo_all(self) -> None:
        """多个文件操作, undo_all 逆序恢复所有文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_a = Path(tmpdir) / "a.txt"
            file_b = Path(tmpdir) / "b.txt"
            orig_a = "original A"
            orig_b = "original B"
            file_a.write_text(orig_a)
            file_b.write_text(orig_b)

            def apply_a(_ctx: dict[str, Any]) -> dict[str, Any]:
                file_a.write_text("MODIFIED A")
                return _ctx

            def undo_a(_ctx: dict[str, Any]) -> dict[str, Any]:
                file_a.write_text(orig_a)
                return _ctx

            def apply_b(_ctx: dict[str, Any]) -> dict[str, Any]:
                file_b.write_text("MODIFIED B")
                return _ctx

            def undo_b(_ctx: dict[str, Any]) -> dict[str, Any]:
                file_b.write_text(orig_b)
                return _ctx

            ctx: EffectContext[dict[str, Any]] = EffectContext({})
            ctx.apply(Effect(name="write_a", apply=apply_a, undo=undo_a))
            ctx.apply(Effect(name="write_b", apply=apply_b, undo=undo_b))

            assert file_a.read_text() == "MODIFIED A"
            assert file_b.read_text() == "MODIFIED B"

            undone = ctx.undo_all()
            assert len(undone) == 2
            assert file_a.read_text() == orig_a
            assert file_b.read_text() == orig_b

    def test_file_undo_with_nested_effects(self) -> None:
        """嵌套文件操作: A → B → C, undo_all 逆序 C → B → A。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = Path(tmpdir) / "chain.txt"
            fp.write_text("state_0")

            states: list[str] = []

            def make_file_effect(name: str, content: str) -> Effect[dict[str, Any]]:
                def apply_fn(_ctx: dict[str, Any]) -> dict[str, Any]:
                    fp.write_text(content)
                    states.append(f"apply_{name}")
                    return _ctx

                def undo_fn(_ctx: dict[str, Any]) -> dict[str, Any]:
                    states.append(f"undo_{name}")
                    return _ctx

                return Effect(name=name, apply=apply_fn, undo=undo_fn)

            ctx: EffectContext[dict[str, Any]] = EffectContext({})
            ctx.apply(make_file_effect("A", "state_1"))
            assert fp.read_text() == "state_1"
            ctx.apply(make_file_effect("B", "state_2"))
            assert fp.read_text() == "state_2"
            ctx.apply(make_file_effect("C", "state_3"))
            assert fp.read_text() == "state_3"

            undone = ctx.undo_all()
            assert len(undone) == 3
            assert [e.name for e in undone] == ["C", "B", "A"]
            assert states == [
                "apply_A", "apply_B", "apply_C",
                "undo_C", "undo_B", "undo_A",
            ]


# ===========================================================================
# 6. snapshot / restore 隔离验证
# ===========================================================================


class TestSnapshotRestore:
    """snapshot 返回隔离拷贝, restore 恢复到快照时刻。"""

    def test_snapshot_isolation(self) -> None:
        ctx: EffectContext[dict[str, Any]] = EffectContext({"x": 1})
        snap = ctx.snapshot()
        ctx.apply(_set_effect("x", 999))
        assert ctx.context["x"] == 999
        assert snap[0]["x"] == 1

    def test_restore_from_snapshot(self) -> None:
        ctx: EffectContext[dict[str, Any]] = EffectContext({"x": 1})
        ctx.apply(_set_effect("x", 10))
        snap = ctx.snapshot()  # 快照时刻: x=10, depth=1
        ctx.apply(_set_effect("x", 20))
        ctx.apply(_set_effect("x", 30))
        assert ctx.context["x"] == 30
        assert ctx.depth == 3

        # 恢复到快照时刻 (x=10, depth=1)
        ctx.restore(snap)
        assert ctx.context["x"] == 10
        assert ctx.depth == 1

    def test_snapshot_log_preserved(self) -> None:
        ctx: EffectContext[dict[str, Any]] = EffectContext({"counter": 0})
        ctx.apply(_add_counter_effect(1))
        snap = ctx.snapshot()
        assert len(snap[1]) == 1
        # 修改快照日志不影响原
        snap[1].append(
            Effect(name="fake", apply=lambda x: x, undo=lambda x: x)
        )
        assert len(ctx.log) == 1


# ===========================================================================
# 7. 边界情况
# ===========================================================================


class TestEdgeCases:
    """边界条件测试。"""

    def test_undo_empty_raises_index_error(self) -> None:
        ctx = EffectContext(42)
        with pytest.raises(IndexError, match="无可撤销"):
            ctx.undo_last()

    def test_depth_tracking(self) -> None:
        ctx: EffectContext[dict[str, Any]] = EffectContext({"counter": 0})
        assert ctx.depth == 0
        ctx.apply(_add_counter_effect(1))
        assert ctx.depth == 1
        ctx.apply(_add_counter_effect(2))
        assert ctx.depth == 2
        ctx.undo_last()
        assert ctx.depth == 1
        ctx.undo_all()
        assert ctx.depth == 0

    def test_repr(self) -> None:
        ctx: EffectContext[dict[str, Any]] = EffectContext({"counter": 0})
        assert "depth=0" in repr(ctx)
        ctx.apply(_add_counter_effect(1))
        assert "depth=1" in repr(ctx)

    def test_make_effect_factory(self) -> None:
        e = make_effect(
            name="double",
            apply_fn=lambda x: x * 2,
            undo_fn=lambda x: x // 2,
        )
        ctx = EffectContext(5)
        ctx.apply(e)
        assert ctx.context == 10
        ctx.undo_last()
        assert ctx.context == 5

    def test_effect_repr(self) -> None:
        e = _set_effect("x", 1)
        assert "set_x" in repr(e)

    def test_log_returns_copy(self) -> None:
        ctx: EffectContext[dict[str, Any]] = EffectContext({"counter": 0})
        ctx.apply(_add_counter_effect(1))
        log = ctx.log
        log.pop()
        assert len(ctx.log) == 1

    def test_context_property(self) -> None:
        ctx: EffectContext[dict[str, Any]] = EffectContext({"a": 1})
        assert ctx.context["a"] == 1

    def test_empty_context(self) -> None:
        ctx = EffectContext(None)
        assert ctx.context is None
        assert ctx.depth == 0
        undone = ctx.undo_all()
        assert undone == []


# ===========================================================================
# 8. 线程安全
# ===========================================================================


class TestThreadSafety:
    """并发 apply 不应崩溃。"""

    def test_concurrent_applies(self) -> None:
        ctx: EffectContext[dict[str, Any]] = EffectContext({"counter": 0})
        errors: list[Exception] = []

        def worker(n: int) -> None:
            try:
                for _ in range(10):
                    ctx.apply(_add_counter_effect(1))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert ctx.depth == 50
        assert ctx.context["counter"] == 50
