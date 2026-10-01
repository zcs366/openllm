"""timeutil 收敛钉子 — 6 点位脏数据回归（审计 P1-5/P1-6，2026-10-01）。

每点位一条脏数据钉子：该模块真实读到的文件里混入 ISO 串/数值串时间戳
（其余行正常）→ 断言：
  ① 不抛异常；
  ② 正常记录照旧被返回/计数；
  ③ 脏数据按其真实时间归类，不被当成"最旧/缺失"错误处理。

点位覆盖：
  1. isa/providers/jiak_provider.py  卡级 created_at ISO 雷（真库已中 10 张）
  2. ios/verification_ledger.py      数值 ts 混入 TypeError 炸穿
  3. identity/closure_stress.py      growth.jsonl ISO ts 击穿 _load_growth_recent
  4. core/main_loop.py               降级日志 ISO ts → 整轮计数清零（双重静默）
     + 环状连通钉子：degradation_trace 真实写入 → main_loop 计数可见
  5. iai/event_bus.py                load_from_jsonl / cleanup_expired TypeError
  6. iai/ior_filter.py               get_recent sort key TypeError

反证：拆回 jiak / event_bus / main_loop 旧写法，钉子必须红在行为
（断言失败或被测调用抛错），而非测试数据准备阶段。
"""
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from openllm.isa.timeutil import coerce_ts


def _iso(offset_s: float) -> str:
    """本机时区 naive ISO 串（coerce_ts 按本地时间解析，与 time.time() 同基准）。"""
    return datetime.fromtimestamp(time.time() + offset_s).isoformat(timespec="seconds")


# ═══════════════════════════════════════════════════════════
# 点位 1：jiak_provider — 卡级 created_at ISO 雷
# ═══════════════════════════════════════════════════════════

class TestJiakProviderIsoNail:
    """真库 jiak 卡级 created_at 已是 ISO 串（delivery_bridge 写入）；
    旧 _compute_temperature 裸 (time.time()-created) 遇 ISO 即 TypeError，
    整个 provider 被 bus 吞掉静默降级。"""

    CARD_ISO_EPOCH = None  # set in fixture

    def _make_jiak(self, tmp_path):
        iso_recent = _iso(-3600.0)  # 1 小时前
        self.CARD_ISO_EPOCH = coerce_ts(iso_recent)
        index = {"cards": {"c1": {"keywords": ["nailjiak"], "importance": 0.7,
                                  "topic": "convergence"}}}
        card = {
            "created_at": iso_recent,  # 卡级 ISO——真库 10 张实测形态
            "opinions": [
                {"id": "o1", "status": "alive", "text": "no own ts, falls back to card ISO",
                 "importance": 0.8},
                {"id": "o2", "status": "alive", "text": "own ts is ISO string",
                 "created_at": iso_recent, "importance": 0.8},
            ],
        }
        (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
        (tmp_path / "cards").mkdir()
        (tmp_path / "cards" / "c1.json").write_text(json.dumps(card), encoding="utf-8")
        return tmp_path

    def test_search_survives_iso_created_at(self, tmp_path):
        from openllm.isa.memory_bus import Query
        from openllm.isa.providers.jiak_provider import JiakProvider

        jiak_dir = self._make_jiak(tmp_path)
        provider = JiakProvider(jiak_dir=jiak_dir)
        # ① 不抛异常（旧写法：o2 的 ISO created → (time.time()-str) TypeError）
        records = provider.search(Query(text="nailjiak", top_k=5))
        # ② 正常记录照旧返回
        assert len(records) == 2, f"ISO created_at 击穿致记录丢失: {len(records)}"
        # ③ ISO 按真实时间归类：timestamp 是解析后的 epoch float，非 ISO 串非 0
        for rec in records:
            assert isinstance(rec.timestamp, float)
            assert rec.timestamp == pytest.approx(self.CARD_ISO_EPOCH)
        by_id = {r.record_id.split(":")[-1]: r for r in records}
        # o2 带 ISO created_at：温度按真实 1h 龄衰减（≈0.97），
        # 不是"解析失败归0→中性0.5"或击穿丢整卡
        assert 0.9 < by_id["o2"].temperature < 1.0, \
            f"ISO 时间被当成缺失(0.5)或最旧处理: {by_id['o2'].temperature}"
        # o1 无意见级 created_at：维持既有"缺失→中性0.5"语义不变
        assert by_id["o1"].temperature == 0.5

    def test_iso_card_not_silently_zeroed(self, tmp_path):
        """反证锚点：ISO created_at 若被 coerce 失败归 0，temperature 恰为 0.5——
        断言其不等于 0.5，钉住'按真实时间归类'。"""
        from openllm.isa.memory_bus import Query
        from openllm.isa.providers.jiak_provider import JiakProvider

        provider = JiakProvider(jiak_dir=self._make_jiak(tmp_path))
        records = provider.search(Query(text="nailjiak"))
        o2 = next(r for r in records if r.record_id.endswith("o2"))
        assert o2.temperature != 0.5
        expected = math.exp(-0.029 * 1.0)  # 1 小时龄
        assert o2.temperature == pytest.approx(expected, rel=0.05)


# ═══════════════════════════════════════════════════════════
# 点位 2：verification_ledger — 数值 ts TypeError 炸穿
# ═══════════════════════════════════════════════════════════

class TestVerificationLedgerNumericTsNail:
    def _ledger(self, tmp_path):
        from openllm.ios.verification_ledger import VerificationLedger
        return VerificationLedger(path=tmp_path / "ledger.jsonl")

    def _append_raw(self, tmp_path, record: dict):
        with (tmp_path / "ledger.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_numeric_ts_does_not_break_fresh(self, tmp_path):
        ledger = self._ledger(tmp_path)
        ledger.record(target="/a.py", step="pytest tests/test_a.py", ok=True)
        # 脏数据：外部混入 JSON 数值 ts（float，非 ISO str）——旧写法
        # datetime.fromisoformat(float) 抛 TypeError，except 只捕 ValueError 炸穿
        fresh_num = {"ts": time.time() - 60, "evidence_id": "ext000000001",
                     "target": "/b.py", "kind": "ad_hoc", "step": "verify_syntax",
                     "ok": True}
        self._append_raw(tmp_path, fresh_num)
        # ① 不抛异常 ② 正常 ISO 证据照旧找到
        ev = ledger.fresh_evidence_for("/a.py", max_age_s=3600)
        assert ev is not None and ev.target == "/a.py"
        # ③ 数值 ts 按其真实时间（1 分钟前）归为新鲜证据，不被丢弃
        ev_b = ledger.fresh_evidence_for("/b.py", max_age_s=3600)
        assert ev_b is not None, "数值 ts 未按真实时间归类（新鲜证据被丢）"

    def test_numeric_string_and_garbage_ts(self, tmp_path):
        ledger = self._ledger(tmp_path)
        ledger.record(target="/c.py", step="verify", ok=True)
        # ts 为纯数字字符串：coerce 转 float，真实时间在窗口内 → 新鲜
        self._append_raw(tmp_path, {"ts": str(time.time() - 120),
                                    "evidence_id": "ext000000002", "target": "/d.py",
                                    "kind": "ad_hoc", "step": "check", "ok": True})
        # ts 为不可解析垃圾串 + 正常 ISO 证据：脏行跳过，旧好行仍返回（continue 语义）
        good_iso = ledger.record(target="/e.py", step="pytest -q", ok=True)
        self._append_raw(tmp_path, {"ts": "not-a-timestamp", "evidence_id": "ext000000003",
                                    "target": "/e.py", "kind": "ad_hoc", "step": "junk",
                                    "ok": True})
        assert ledger.has_fresh_evidence("/d.py") is True
        ev = ledger.fresh_evidence_for("/e.py")
        assert ev is not None and ev.evidence_id == good_iso

    def test_verify_claims_counts_numeric_ts(self, tmp_path):
        ledger = self._ledger(tmp_path)
        self._append_raw(tmp_path, {"ts": time.time() - 30, "evidence_id": "ext000000004",
                                    "target": "/f.py", "kind": "canonical",
                                    "step": "pytest tests/test_f.py", "ok": True})
        # ① 不抛异常 ③ 数值 ts 证据按真实新鲜计入 fresh_records
        res = ledger.verify_claims_vs_evidence(["pytest tests/test_f.py"])
        assert res["matched"] == ["pytest tests/test_f.py"]
        assert res["fresh_records"] == 1
        assert res["total_records"] == 1


# ═══════════════════════════════════════════════════════════
# 点位 3：closure_stress — growth.jsonl ISO ts 击穿
# ═══════════════════════════════════════════════════════════

class TestClosureStressIsoNail:
    def test_dirty_line_does_not_kill_recent_window(self, tmp_path):
        from openllm.identity.closure_stress import ClosureStressProbe

        probe = ClosureStressProbe(embed_fn=lambda a, b: 0.9,
                                   growth_path=tmp_path / "growth.jsonl")
        lines = [
            # ① 首行即脏：8 天前的 ISO ts——旧写法 str>=float TypeError，
            #    内层只捕 JSONDecodeError、外层只捕 OSError → 整轮炸穿
            json.dumps({"content": "iso-expired", "ts": _iso(-8 * 86400)}),
            json.dumps({"content": "iso-recent", "ts": _iso(-3600)}),
            json.dumps({"content": "float-recent", "ts": time.time() - 7200}),
            json.dumps({"content": "numeric-string-ts", "ts": str(time.time() - 60)}),
            '{"content": "broken-json", "ts": <<<nope>>}',
        ]
        (tmp_path / "growth.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        recs = probe._load_growth_recent(days=7)  # ① 不抛异常
        contents = {r.get("content") for r in recs}
        # ② 正常记录照旧返回
        assert contents == {"iso-recent", "float-recent", "numeric-string-ts"}
        # ③ 8 天前 ISO 按其真实时间归类被排除——不是当最旧丢弃好行，
        #    也不是解析失败归 0 混入窗口
        assert "iso-expired" not in contents
        assert "broken-json" not in contents


# ═══════════════════════════════════════════════════════════
# 点位 4：main_loop 降级计数 + 环状连通
# ═══════════════════════════════════════════════════════════

def _patch_home(monkeypatch, tmp_path):
    real_home = Path.home
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return real_home


def _startup_line(monkeypatch, tmp_path, capsys):
    """跑 Agent._print_startup_health（fake self 只给 clock=None），返回 stdout。"""
    from openllm.core.main_loop import Agent
    _patch_home(monkeypatch, tmp_path)
    Agent._print_startup_health(SimpleNamespace(clock=None))
    return capsys.readouterr().out


class TestMainLoopDegradedCountNail:
    def _write_log(self, tmp_path, lines):
        d = tmp_path / ".openllm" / "output"
        d.mkdir(parents=True, exist_ok=True)
        (d / "degradation_log.jsonl").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")

    def test_iso_line_does_not_zero_the_count(self, monkeypatch, tmp_path, capsys):
        now = time.time()
        self._write_log(tmp_path, [
            # ① 首行脏 ISO ts——旧写法 TypeError 上抛，外层 except Exception
            #    吞掉 → 后续正常行不再扫描 → 计数清零、横幅永不显示（双重静默）
            json.dumps({"timestamp": _iso(-3600), "body": "ISA", "phase": "query"}),
            json.dumps({"timestamp": now - 7200, "body": "IAI", "phase": "publish"}),
            json.dumps({"timestamp": now - 100, "body": "IOS", "phase": "arbitrate"}),
            json.dumps({"timestamp": now - 8 * 86400, "body": "ISN", "phase": "stale"}),
        ])
        out = _startup_line(monkeypatch, tmp_path, capsys)
        # ② 正常行照旧计数 ③ ISO 行按真实时间（1h 前）计入 → 共 3 次
        assert "近24h降级3次" in out, f"降级计数被脏数据静默清零:\n{out}"
        assert "ISN" not in out  # 8 天前的不属近 24h，不应混入

    def test_ring_connectivity_trace_writer_visible(self, monkeypatch, tmp_path, capsys):
        """环状连通：memory_bus._trace_degradation_safe 的真实写入格式
        （core.degradation_trace.trace_degradation 落的时间戳=epoch float）
        必须被 main_loop 启动横幅如实计数。"""
        from openllm.core import degradation_trace

        d = tmp_path / ".openllm" / "output"
        d.mkdir(parents=True, exist_ok=True)
        log = d / "degradation_log.jsonl"
        monkeypatch.setattr(degradation_trace, "LOG_PATH", log)
        degradation_trace.trace_degradation("JiakProvider", "provider_query",
                                            TypeError("int - str"), "iso-created_at")
        # 写入端产物即读取端输入——同一路径，不复制不转格式
        out = _startup_line(monkeypatch, tmp_path, capsys)
        assert "近24h降级1次" in out, f"写入→读取环断裂:\n{out}"


# ═══════════════════════════════════════════════════════════
# 点位 5：event_bus — load_from_jsonl / cleanup_expired
# ═══════════════════════════════════════════════════════════

def _evt(ts, etype="heartbeat.ping", source="IAX"):
    return {"event_id": f"evt-{etype}-{ts}", "source": source, "type": etype,
            "timestamp": ts, "entropy_score": 0.1, "brain_id": "default",
            "payload": {}}


class TestEventBusDirtyTsNail:
    def _bus(self, tmp_path):
        from openllm.iai.event_bus import EventBus
        return EventBus(log_dir=tmp_path / "events", ttl_hours=24)

    def _write_events(self, tmp_path, lines):
        d = tmp_path / "events"
        d.mkdir(parents=True, exist_ok=True)
        (d / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_load_from_jsonl_survives_iso_ts(self, tmp_path):
        now = time.time()
        # ① 首行脏 ISO ts——旧写法 TypeError，except 只捕 JSONDecodeError/KeyError
        #    → 炸穿整个 load_from_jsonl（新总线一启动就死）
        self._write_events(tmp_path, [
            json.dumps(_evt(_iso(-3600), etype="iso_recent")),
            json.dumps(_evt(now - 7200, etype="float_recent")),
            json.dumps(_evt(now - 90 * 86400, etype="float_expired")),
            json.dumps(_evt(str(now - 100), etype="numstr_recent")),
        ])
        bus = self._bus(tmp_path)
        loaded = bus.load_from_jsonl()  # ① 不抛异常
        types = {e["type"] for e in bus.get_history(limit=100)}
        # ② 正常 float 记录照旧加载
        assert "float_recent" in types
        # ③ ISO/数值串按其真实时间归类：在窗口内的加载（3条），90天前的排除
        assert loaded == 3, f"应按真实时间加载3条，实得{loaded}: {types}"
        assert "float_expired" not in types

    def test_cleanup_expired_survives_iso_ts(self, tmp_path):
        now = time.time()
        self._write_events(tmp_path, [
            json.dumps(_evt(_iso(-3600), etype="iso_recent")),
            json.dumps(_evt(now - 7200, etype="float_recent")),
            json.dumps(_evt(now - 90 * 86400, etype="float_old")),
        ])
        bus = self._bus(tmp_path)
        cleaned = bus.cleanup_expired()  # ① 不抛异常
        assert cleaned == 1  # 只清 90 天前的——ISO 真实新鲜行不被误清
        kept = (tmp_path / "events" / "events.jsonl").read_text(encoding="utf-8")
        assert "iso_recent" in kept and "float_recent" in kept
        assert "float_old" not in kept


# ═══════════════════════════════════════════════════════════
# 点位 6：ior_filter — get_recent sort key
# ═══════════════════════════════════════════════════════════

class TestIorFilterSortKeyNail:
    def test_get_recent_survives_iso_ts(self, tmp_path):
        from openllm.iai.ior_filter import IoRFilter
        import hashlib

        def h(t):
            return hashlib.sha256(" ".join(t.lower().split()).encode()).hexdigest()[:8]

        path = tmp_path / "ior_topics.jsonl"
        now = time.time()
        # ① 首行脏 ISO ts——旧写法 sort key 混合 str/float 比较抛 TypeError 外溢
        lines = [
            {"topic": "iso-newest", "hash": h("iso-newest"), "timestamp": _iso(-60)},
            {"topic": "float-mid", "hash": h("float-mid"), "timestamp": now - 7200},
            {"topic": "float-oldest", "hash": h("float-oldest"),
             "timestamp": now - 86400},
        ]
        path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines)
                        + "\n", encoding="utf-8")
        ior = IoRFilter(path=path)
        recent = ior.get_recent(limit=10)  # ① 不抛异常
        # ② 正常记录照旧返回
        assert set(recent) == {"iso-newest", "float-mid", "float-oldest"}
        # ③ ISO 按真实时间（1 分钟前）归为最新，而非解析失败归 0 排最末
        assert recent[0] == "iso-newest", f"ISO 未按真实时间排序: {recent}"
        assert recent[1] == "float-mid"

    def test_unparseable_ts_sinks_to_oldest(self, tmp_path):
        from openllm.iai.ior_filter import IoRFilter
        import hashlib

        def h(t):
            return hashlib.sha256(" ".join(t.lower().split()).encode()).hexdigest()[:8]

        path = tmp_path / "ior_topics.jsonl"
        now = time.time()
        lines = [
            {"topic": "good", "hash": h("good"), "timestamp": now - 10},
            {"topic": "junk-ts", "hash": h("junk-ts"), "timestamp": "not-a-timestamp"},
        ]
        path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines)
                        + "\n", encoding="utf-8")
        ior = IoRFilter(path=path)
        recent = ior.get_recent(limit=5)  # ① 不抛
        # ② 好记录返回 ③ 不可解析→0→按最旧排后（不被当最新插队）
        assert recent == ["good", "junk-ts"]

