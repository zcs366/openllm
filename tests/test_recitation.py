"""
RecitationEngine 测试 — 诵经引擎（T1排版艺动态臂 · Manus课4移植）
==================================================================

覆盖：
1. test_recitation_rhythm —— 每N步诵经一次的节奏
2. test_no_goals_no_recitation —— 无目标/全完成不诵经
3. test_append_only_prefix —— recite_into前缀逐条不动（KV-cache纪律）
4. test_controlled_variation —— 模板轮转（防模式锁死）
5. test_max_cap —— 诵经上限防洪泛
6. test_goal_management —— 目标增删改查
7. test_stats —— P-Manus-1 A/B验证度量位
8. test_metadata —— 诵经消息元数据完整性
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.expanduser("~/projects/openllm/src"))

from openllm.message import Message
from openllm.recitation import (
    RecitationConfig,
    RecitationEngine,
    DEFAULT_TEMPLATES,
)


def _make_engine(every=5, cap=50, seed=42):
    eng = RecitationEngine(RecitationConfig(
        every_n_steps=every, max_recitations=cap, seed=seed))
    eng.set_goals(["下载论文", "精读原文", "写宪章v0.3"])
    return eng


def _step(eng):
    """on_step且断言非None（Pyright收窄Optional）。"""
    msg = eng.on_step()
    assert msg is not None, "预期诵经点却返回None"
    return msg


class TestRhythm(unittest.TestCase):
    def test_recitation_rhythm(self):
        """每5步诵经一次：步5/10/15有诵经，其余无"""
        eng = _make_engine(every=5)
        fired = []
        for step in range(1, 21):
            msg = eng.on_step()
            fired.append((step, msg is not None))
        fired_steps = [s for s, f in fired if f]
        self.assertEqual(fired_steps, [5, 10, 15, 20])
        self.assertEqual(eng.stats.recitations, 4)

    def test_every_step_mode(self):
        eng = _make_engine(every=1)
        msgs = [eng.on_step() for _ in range(3)]
        self.assertTrue(all(m is not None for m in msgs))

    def test_no_goals_no_recitation(self):
        eng = RecitationEngine(RecitationConfig(every_n_steps=1))
        # 未设目标
        self.assertIsNone(eng.on_step())
        # 全完成
        eng.set_goals(["唯一目标"])
        eng.mark_done(0)
        self.assertIsNone(eng.on_step())
        self.assertEqual(eng.stats.recitations, 0)


class TestAppendOnly(unittest.TestCase):
    def test_append_only_prefix(self):
        """recite_into的前缀必须逐条不动（同对象引用）——KV-cache纪律"""
        eng = _make_engine(every=2)
        prefix = [Message(role="system", content="sys"),
                  Message(role="user", content="u1"),
                  Message(role="assistant", content="a1"),
                  Message(role="user", content="u2")]
        ids_before = [id(m) for m in prefix]
        out = prefix
        recited = 0
        for _ in range(6):  # 步2/4/6诵经
            out = eng.recite_into(out)
        # 原前缀4条对象引用必须还在开头且未变
        for i, mid in enumerate(ids_before):
            self.assertEqual(id(out[i]), mid, f"前缀第{i}条被改动——违反append-only")
        # 诵经消息全在尾部
        recitations = [m for m in out if m.metadata.get("recitation")]
        self.assertEqual(len(recitations), 3)
        self.assertEqual(recitations[0], out[4])  # 第一条诵经紧跟前缀
        # 原列表对象未被就地修改
        self.assertEqual(len(prefix), 4)

    def test_no_recitation_returns_same_list(self):
        eng = _make_engine(every=10)
        msgs = [Message(role="user", content="x")]
        out = eng.recite_into(msgs)
        self.assertIs(out, msgs)  # 不诵经时原样返回（零拷贝）


class TestControlledVariation(unittest.TestCase):
    def test_template_rotation(self):
        """连续诵经使用不同模板（轮转，防模式锁死）"""
        eng = _make_engine(every=1)
        contents = [_step(eng).content for _ in range(len(DEFAULT_TEMPLATES) * 2)]
        # 6次诵经，3个模板——每个模板至少出现一次
        # 通过metadata验证轮转
        eng2 = _make_engine(every=1)
        tpl_idxs = [_step(eng2).metadata["template_idx"] for _ in range(6)]
        self.assertEqual(sorted(set(tpl_idxs)), [0, 1, 2])
        # 相邻两次不应是同一模板（轮转非随机重复）
        for a, b in zip(tpl_idxs, tpl_idxs[1:]):
            self.assertNotEqual(a, b)

    def test_deterministic_with_seed(self):
        """同seed两个引擎模板序列一致（可复现）"""
        e1 = _make_engine(every=1, seed=7)
        e2 = _make_engine(every=1, seed=7)
        s1 = [_step(e1).metadata["template_idx"] for _ in range(9)]
        s2 = [_step(e2).metadata["template_idx"] for _ in range(9)]
        self.assertEqual(s1, s2)

    def test_variation_is_formatting_not_semantic(self):
        """所有模板都含全部待办目标（信息内容恒等，变化仅排版级）"""
        eng = _make_engine(every=1)
        for _ in range(6):
            content = _step(eng).content
            for goal in ["下载论文", "精读原文", "写宪章v0.3"]:
                self.assertIn(goal, content)


class TestMaxCap(unittest.TestCase):
    def test_cap_stops_recitation(self):
        eng = _make_engine(every=1, cap=3)
        msgs = [eng.on_step() for _ in range(10)]
        fired = [m for m in msgs if m is not None]
        self.assertEqual(len(fired), 3)
        self.assertEqual(eng.stats.skipped_at_cap, 7)
        self.assertEqual(eng.stats.recitations, 3)


class TestGoalManagement(unittest.TestCase):
    def test_set_and_mark(self):
        eng = RecitationEngine(RecitationConfig(every_n_steps=1))
        eng.set_goals(["A", "B", "C"])
        self.assertEqual(eng.pending_goals, ["A", "B", "C"])
        eng.mark_done(1)
        self.assertEqual(eng.pending_goals, ["A", "C"])
        self.assertEqual(eng.done_goals, ["B"])
        self.assertFalse(eng.all_done)
        eng.mark_done(0)
        eng.mark_done(2)
        self.assertTrue(eng.all_done)
        self.assertEqual(eng.stats.goals_done, 3)

    def test_mark_done_idempotent(self):
        eng = RecitationEngine()
        eng.set_goals(["A"])
        eng.mark_done(0)
        eng.mark_done(0)  # 重复标记不重复计数
        self.assertEqual(eng.stats.goals_done, 1)

    def test_mark_out_of_range(self):
        eng = RecitationEngine()
        eng.set_goals(["A"])
        self.assertFalse(eng.mark_done(5))
        self.assertFalse(eng.mark_done(-1))

    def test_add_goal(self):
        eng = RecitationEngine()
        eng.set_goals(["A"])
        idx = eng.add_goal("B")
        self.assertEqual(idx, 1)
        self.assertEqual(eng.pending_goals, ["A", "B"])

    def test_add_goal_empty_raises(self):
        eng = RecitationEngine()
        with self.assertRaises(ValueError):
            eng.add_goal("   ")

    def test_set_goals_filters_empty(self):
        eng = RecitationEngine()
        eng.set_goals(["A", "", "  ", "B"])
        self.assertEqual(eng.pending_goals, ["A", "B"])

    def test_recitation_reflects_progress(self):
        """诵经内容随进度更新（done计数变化，不依赖特定模板措辞）"""
        eng = _make_engine(every=1)
        m1 = _step(eng)
        eng.mark_done(0)
        m2 = _step(eng)
        # metadata.pending 是措辞无关的进度真值：3→2
        self.assertEqual(m1.metadata["pending"], 3)
        self.assertEqual(m2.metadata["pending"], 2)
        # 内容层面：完成项从待办列表消失，未完成项仍在
        self.assertIn("[ ] 下载论文", m1.content)
        self.assertNotIn("[ ] 下载论文", m2.content)
        self.assertIn("[ ] 精读原文", m2.content)


class TestStatsAndMetadata(unittest.TestCase):
    def test_stats(self):
        eng = _make_engine(every=5)
        for _ in range(12):
            eng.on_step()
        s = eng.get_stats()
        self.assertEqual(s["steps"], 12)
        self.assertEqual(s["recitations"], 2)
        self.assertEqual(s["last_recited_step"], 10)
        self.assertEqual(s["pending_goals"], 3)
        self.assertEqual(s["every_n_steps"], 5)

    def test_metadata_fields(self):
        eng = _make_engine(every=1)
        msg = _step(eng)
        md = msg.metadata
        self.assertTrue(md["recitation"])
        self.assertEqual(md["recitation_no"], 1)
        self.assertEqual(md["step"], 1)
        self.assertIn("template_idx", md)
        self.assertEqual(md["pending"], 3)
        self.assertEqual(msg.role, "system")

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            RecitationConfig(every_n_steps=0)
        with self.assertRaises(ValueError):
            RecitationConfig(role="tool")
        with self.assertRaises(ValueError):
            RecitationConfig(templates=[])


if __name__ == "__main__":
    unittest.main(verbosity=2)
