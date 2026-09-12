"""2026-09-10 接骨：会话标识贯通——一个生命，一个账本。

病（二驾实测）：
  · sleep() 每次新铸 f"s{now}" 写一个新胶囊文件 → 同一次生命的记忆被劈成两本账
  · memory_write 追加到"当前加载的胶囊"（可能是别家会话留下的）
  · 唤醒只念 mtime 最新的那一本 → 会话中记下的东西永久沉默
  · sleep 用自己那点 decisions/insights 直接覆盖本会话胶囊 → 把会话中
    memory_write 存进去的洞察整段抹掉
  · 模型回声（"模型最后回应：……"）塞进 insights → 每次苏醒听见自己上一轮的独白

钉（每条测试对应一个症状，症状复现即红）：
  ① wake 认领恢复到的会话  ② sleep 不另开账本  ③ 工具与 sleep 同账本
  ④ sleep 合并不覆盖  ⑤ 回声不占 insights  ⑥ 条目有上限  ⑦ 重启后念得出记得的事
"""

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


# ── 测试夹具 ────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _cheap_delta(monkeypatch):
    """接骨测试不测嵌入：Δ向量走固定向量，免得每条测试加载 sentence-transformers。"""
    from openllm.isa import capsule as cap

    def _fast(cls, session_id, text):
        return cls(session_id=session_id,
                   vector=np.ones(cap.CAPSULE_DIM, dtype=np.float32))

    monkeypatch.setattr(cap.DeltaCapsule, "from_text", classmethod(_fast))


@pytest.fixture
def sandbox_home():
    """引擎的 MemoryOS/KV 层默认落 ~/.openllm，必须改 HOME 免得污染真实记忆库。"""
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "fakehome"
        (home / ".openllm").mkdir(parents=True)
        old = os.environ.get("HOME")
        os.environ["HOME"] = str(home)
        try:
            yield tmp
        finally:
            if old is not None:
                os.environ["HOME"] = old


def _engine(tmp, name="OpenLLM"):
    from openllm.core.engine import OpenLLMEngine, AgentConfig
    os.environ.setdefault("OPENLLM_SECURITY_LEVEL", "3")
    return OpenLLMEngine(AgentConfig(capsule_dir=tmp, provider="deepseek",
                                     model="nonexistent", name=name))


def _seed_capsule(caps_dir, session_id, insights=None, ts=None):
    """在库里预置一个胶囊（模拟上一次生命留下的账本）。"""
    from openllm.isa.capsule import TextCapsule
    cap = TextCapsule(session_id=session_id,
                      timestamp=ts if ts is not None else time.time() - 100,
                      insights=list(insights or []))
    (Path(caps_dir) / f"v06_{session_id}.json").write_text(
        json.dumps(cap.to_dict(), ensure_ascii=False), encoding="utf-8")
    return cap


def _v06_files(caps_dir):
    return sorted(p.name for p in Path(caps_dir).glob("v06_*.json"))


def _newest_capsule(caps_dir):
    files = list(Path(caps_dir).glob("v06_*.json"))
    assert files, "库里一个胶囊都没有"
    newest = max(files, key=lambda p: json.loads(p.read_text(encoding="utf-8"))
                 .get("timestamp", 0.0))
    return json.loads(newest.read_text(encoding="utf-8")), newest


# ── ① 唤醒认领 ──────────────────────────────────────────────

class TestWakeAdopts:
    def test_wake_adopts_restored_session_id(self, sandbox_home):
        """恢复到的胶囊是谁的会话，本次生命就续在谁的账上。"""
        tmp = sandbox_home
        _seed_capsule(tmp, "sOLD_LIFE")
        e = _engine(tmp)
        minted = e.session_id
        e.wake()
        assert e.session_id == "sOLD_LIFE", (
            f"唤醒后没认领被恢复的会话：仍是 {e.session_id}（铸造时={minted}）"
        )

    def test_wake_with_empty_store_keeps_minted_id(self, sandbox_home):
        """空库（第一次生命）：保留铸造的会话标识，不认领空气。"""
        tmp = sandbox_home
        e = _engine(tmp)
        minted = e.session_id
        e.wake()
        assert e.session_id == minted


# ── ② / ④ 一个生命一个账本 ──────────────────────────────────

class TestOneLedgerPerLife:
    def test_sleep_does_not_mint_new_capsule(self, sandbox_home):
        """症状②：sleep 另开新账 → 库里出现两个 v06 文件。"""
        tmp = sandbox_home
        _seed_capsule(tmp, "sOLD_LIFE")
        before = _v06_files(tmp)
        e = _engine(tmp)
        e.wake()
        e.sleep()
        after = _v06_files(tmp)
        assert after == ["v06_sOLD_LIFE.json"], (
            f"sleep 另开了账本：{before} → {after}"
        )

    def test_tool_write_and_sleep_share_one_capsule(self, sandbox_home):
        """症状③：工具写入落一本、sleep 落另一本。"""
        tmp = sandbox_home
        _seed_capsule(tmp, "sOLD_LIFE")
        e = _engine(tmp)
        e.wake()
        assert e.execute_tool("memory_write", key="线路",
                              content="宜宾到成都").success
        e.sleep()
        assert _v06_files(tmp) == ["v06_sOLD_LIFE.json"], (
            f"工具与 sleep 分了家：{_v06_files(tmp)}"
        )

    def test_sleep_merge_does_not_erase_insights(self, sandbox_home):
        """症状④：sleep 覆盖本会话胶囊，把会话中记下的洞察抹掉。"""
        tmp = sandbox_home
        _seed_capsule(tmp, "sOLD_LIFE")
        e = _engine(tmp)
        e.wake()
        e.execute_tool("memory_write", key="车牌", content="川Q12345")
        e.sleep()
        newest, path = _newest_capsule(tmp)
        joined = "\n".join(newest.get("insights", []))
        assert "川Q12345" in joined, (
            f"sleep 把会话中记下的洞察抹了（{path.name}）：{newest.get('insights')}"
        )

    def test_capsule_id_is_engine_session_id(self, sandbox_home):
        """落盘文件名必须就是引擎的会话标识（而非另铸一个）。

        先 wake 认领一个旧会话：旧代码 sleep 另铸 f"s{now}"，若与构造同秒会
        同名巧合通过——认领后再铸必然不同名，钉子才锋利。
        """
        tmp = sandbox_home
        _seed_capsule(tmp, "sOLD_LIFE")
        e = _engine(tmp)
        e.wake()
        e.sleep()
        assert _v06_files(tmp) == ["v06_sOLD_LIFE.json"], (
            f"文件名与会话标识不符：{_v06_files(tmp)} vs {e.session_id}"
        )


# ── ⑤ / ⑥ 唤醒内容要干净 ────────────────────────────────────

class TestWakeContent:
    def test_model_echo_not_in_insights(self, sandbox_home):
        """症状⑤：模型回声进了 insights → 苏醒时听见自己上一轮独白。"""
        from openllm.core.engine import Message
        tmp = sandbox_home
        e = _engine(tmp)
        e.wake()
        e._history = [Message(role="user", content="你好"),
                      Message(role="assistant", content="我是品尝师，这是上一轮的独白")]
        e.sleep()
        newest, _ = _newest_capsule(tmp)
        joined = "\n".join(newest.get("insights", []))
        assert "模型最后回应" not in joined, f"回声污染了 insights: {joined}"
        # 回声应该落在 outputs（留档但不当洞察念）
        assert any("独白" in o for o in newest.get("outputs", [])), (
            f"回声连 outputs 都没留：{newest.get('outputs')}"
        )

    def test_wake_prints_remembered_fact(self, sandbox_home):
        """症状⑦：重启后 wake() 必须念出会话中记下的事（本文件的验收线）。"""
        tmp = sandbox_home
        e1 = _engine(tmp)
        e1.wake()
        e1.execute_tool("memory_write", key="老搭档的车", content="蓝色东风天龙")
        e1.sleep()

        e2 = _engine(tmp)              # 模拟重启：新进程、新引擎
        out = e2.wake()
        assert "蓝色东风天龙" in out, f"重启后 wake 没念出记住的事：\n{out}"

    def test_insights_capped(self, sandbox_home):
        """症状⑥：只追加不截断 → 胶囊无限膨胀。

        必须同时断言"最新的洞察还在"：否则旧代码另开的新账本恰好没有洞察，
        这条测试会以 len=0 ≤ 20 蒙混过关。
        """
        tmp = sandbox_home
        e = _engine(tmp)
        e.wake()
        for i in range(25):
            e.execute_tool("memory_write", key=f"k{i}", content=f"v{i}")
        e.sleep()
        newest, _ = _newest_capsule(tmp)
        ins = newest.get("insights", [])
        assert len(ins) <= 20, f"insights 没有截断：{len(ins)} 条"
        assert any("v24" in i for i in ins), (
            f"最新写入的洞察不在唤醒账本里（说明落到了别处）：{ins[-3:]}"
        )
