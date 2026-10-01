"""写侧时间戳收口钉子（成市拍板 2026-10-01：canonical = epoch float，读侧兼容旧数据）。

覆盖四组：
E1 门房往返 —— validate_line 对 ISO串/数值串/float/缺失 四种输入统一产出 ts=float，
   且规范化发生在 integrity_hash 之前（新行才哈希，历史行绝不回写）。
E2 enforce_capacity 止血 —— 5100 行混格式（真老 float ts + 真新 ISO ts + 新鲜
   float + 不可解析）：被清洗的必须是真最老的；float 旧行不再被误判 age=1000；
   未知 ts 的行不得被优先洗掉（未知=最新=保护）。
E3 三个直写者 —— stateful_audit / isn_signal / recall_provider.store()：
   字段名统一 ts（epoch float），能走门房的走门房（端到端断最终行）。
E4 ISO 假设读者 —— card_size_guard / jika_audit / isa_context_editor：
   双格式容忍，float ts 不再击穿读取通道。

红线：全部写测试落在 tmp 副本；不碰真实 ~/.hermes/jiak/RECALL.jsonl。
"""
import calendar
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


REAL_HOME = Path(os.environ.get("HOME") or str(Path.home()))
GATE_SRC = REAL_HOME / ".hermes" / "jiak" / "scripts" / "recall_append.py"


def _load_gate(name="recall_append_under_test"):
    """按文件路径独立加载门房（scripts/recall_append.py）。"""
    spec = importlib.util.spec_from_file_location(name, str(GATE_SRC))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 收集期导入（此时 conftest 的 HOME=_FAKE_HOME，.hermes 软链完好——
# providers/__init__ 会经 sys.path 拉 ~/.hermes/scripts/source_index；
# 测试函数里再导会被 tmp_jiak 的 HOME 重定向打断）
from openllm.isa import memory_bus  # noqa: E402
from openllm.isa.providers import recall_provider as rp  # noqa: E402


# ════════════════════════════════════════════════════════════════
# E1 门房往返钉子
# ════════════════════════════════════════════════════════════════

class TestGateRoundTrip:
    def setup_method(self):
        self.gate = _load_gate()

    def test_iso_string_becomes_epoch_float(self):
        iso = "2026-09-01T00:00:00Z"
        expected = calendar.timegm((2026, 9, 1, 0, 0, 0))
        obj = self.gate.validate_line(json.dumps(
            {"type": "inject", "ts": iso, "content": "x"}))
        assert obj is not None
        assert isinstance(obj["ts"], float)
        assert abs(obj["ts"] - expected) < 1.0, f"ts={obj['ts']} expected≈{expected}"

    def test_numeric_string_becomes_float(self):
        obj = self.gate.validate_line(json.dumps(
            {"type": "inject", "ts": "1234567890.5", "content": "x"}))
        assert obj is not None
        assert isinstance(obj["ts"], float)
        assert obj["ts"] == pytest.approx(1234567890.5)

    def test_float_passes_through_unchanged(self):
        obj = self.gate.validate_line(json.dumps(
            {"type": "inject", "ts": 1767196800.25, "content": "x"}))
        assert obj is not None
        assert isinstance(obj["ts"], float)
        assert obj["ts"] == 1767196800.25

    def test_missing_ts_filled_with_now(self):
        t0 = time.time()
        obj = self.gate.validate_line(json.dumps(
            {"type": "inject", "content": "x"}))
        assert obj is not None
        assert isinstance(obj["ts"], float)
        assert abs(obj["ts"] - t0) < 5.0, "缺失 ts 应补当前时间"

    def test_unparseable_ts_filled_with_now_not_crash(self):
        obj = self.gate.validate_line(json.dumps(
            {"type": "inject", "ts": "not-a-date", "content": "x"}))
        assert obj is not None, "不可解析 ts 不得炸门房"
        assert isinstance(obj["ts"], float)
        assert abs(obj["ts"] - time.time()) < 5.0

    def test_legacy_timestamp_field_moved_to_ts(self):
        obj = self.gate.validate_line(json.dumps(
            {"type": "inject", "timestamp": 1767196800.0, "content": "x"}))
        assert obj is not None
        assert "timestamp" not in obj, "旧字段名必须收口到 ts，防止双字段分叉"
        assert obj["ts"] == 1767196800.0

    def test_hash_computed_after_canonicalization(self):
        """规范化发生在哈希之前：ISO 与等值 float 输入产出同一 integrity_hash。"""
        iso = "2026-09-01T00:00:00Z"
        epoch = calendar.timegm((2026, 9, 1, 0, 0, 0))
        o1 = self.gate.validate_line(json.dumps(
            {"type": "inject", "content": "same", "ts": iso}))
        o2 = self.gate.validate_line(json.dumps(
            {"type": "inject", "content": "same", "ts": float(epoch)}))
        assert o1["integrity_hash"] == o2["integrity_hash"], \
            "哈希若先于规范化，两种输入形态会产出不同 hash"

    def test_missing_type_still_rejected(self):
        # HEAD 既有契约：缺 type 抛 ValueError（validate_line 只捕 JSONDecodeError），
        # 收口改动不得把它变成静默通过
        with pytest.raises(ValueError):
            self.gate.validate_line(json.dumps({"content": "x"}))


# ════════════════════════════════════════════════════════════════
# E2 enforce_capacity 止血钉子
# ════════════════════════════════════════════════════════════════

class TestEnforceCapacity:
    def test_mixed_5100_rows_evict_truly_oldest_only(self, tmp_path):
        gate = _load_gate("gate_cap")
        now = time.time()
        rows = []

        def row(tag, ts_val):
            return json.dumps({"type": "event", "content": tag,
                               "ts": ts_val, "importance": 0.5})

        # 60 行真老：float ts（canonical！30天前）——必须最先被清洗
        old_epoch = now - 30 * 86400
        for _ in range(60):
            rows.append(row("OLD_FLOAT", old_epoch))
        # 25 行真新：float ts（30分钟前）——不得被误判 age=1000
        for _ in range(25):
            rows.append(row("FRESH_FLOAT", now - 1800.0))
        # 25 行未知：不可解析 ts——未知=最新，必须受保护
        for _ in range(25):
            rows.append(row("UNKNOWN_TS", "not-a-date"))
        # 4990 填充行：ISO ts 1小时前（最冷的合法淘汰对象）
        iso1h = datetime.fromtimestamp(now - 3600, tz=timezone.utc).isoformat()
        for _ in range(4990):
            rows.append(row("FILL_ISO", iso1h))
        assert len(rows) == 5100

        recall = tmp_path / "RECALL.jsonl"
        recall.write_text("\n".join(rows) + "\n", encoding="utf-8")
        archive_dir = tmp_path / "archive"
        monkey_mod_path = str(recall)

        # 只重定向模块常量到 tmp 副本（真实库绝不入径）
        gate.RECALL_PATH = monkey_mod_path
        gate.RECALL_ARCHIVE_DIR = str(archive_dir)

        assert gate.MAX_RECALL == 5000, "上限设计不得改变"
        evicted = gate.enforce_capacity()
        assert evicted == 100, f"5100→应洗100行, got {evicted}"

        kept_tags = []
        for line in recall.read_text(encoding="utf-8").splitlines():
            if line.strip():
                kept_tags.append(json.loads(line)["content"])
        evicted_tags = []
        arc_files = list(archive_dir.glob("*.jsonl"))
        assert len(arc_files) == 1
        for line in arc_files[0].read_text(encoding="utf-8").splitlines():
            if line.strip():
                evicted_tags.append(json.loads(line)["content"])

        from collections import Counter
        kc, ec = Counter(kept_tags), Counter(evicted_tags)
        # ① 被清洗的是真最老的（60 行 OLD_FLOAT 全灭）
        assert ec["OLD_FLOAT"] == 60, "真最老行必须最先被洗"
        assert kc["OLD_FLOAT"] == 0
        # ② float 新行不再被误判 age=1000 → 全部存活
        assert kc["FRESH_FLOAT"] == 25, \
            "新鲜 float ts 行被当成最老清洗 = 止血失败（原缺陷）"
        assert ec["FRESH_FLOAT"] == 0
        # ③ 未知 ts 行不被优先洗掉
        assert kc["UNKNOWN_TS"] == 25, "未知年龄被当最老 = 违反'未知=最新'保护"
        assert ec["UNKNOWN_TS"] == 0
        # 剩余量与最低清洗量设计不变
        assert sum(kc.values()) == 5000
        assert kc["FILL_ISO"] == 4990 - 40  # 60+40=100 满最低清洗量

    def test_bad_json_row_not_evicted_first(self, tmp_path):
        """坏 JSON 行=年龄未知→受保护，不再是旧代码的 -1 哨兵（最先洗）。"""
        gate = _load_gate("gate_cap2")
        now = time.time()
        rows = []
        # 填充行=真冷（3天前, temp≈0.29 < 受保护行≈0.50）：淘汰应优先落在它们
        iso = datetime.fromtimestamp(now - 3 * 86400, tz=timezone.utc).isoformat()
        rows.append("{this is not valid json}")  # 坏行（放最老位置=旧 -1 哨兵靶心）
        for _ in range(5100 - 1):
            rows.append(json.dumps({"type": "event", "content": "fill",
                                    "ts": iso, "importance": 0.5}))
        recall = tmp_path / "RECALL.jsonl"
        recall.write_text("\n".join(rows) + "\n", encoding="utf-8")
        gate.RECALL_PATH = str(recall)
        gate.RECALL_ARCHIVE_DIR = str(tmp_path / "archive")
        gate.enforce_capacity()
        evicted_text = "".join(
            p.read_text(encoding="utf-8") for p in
            (tmp_path / "archive").glob("*.jsonl"))
        assert "this is not valid json" not in evicted_text, \
            "坏行被优先清洗=旧 -1 哨兵复活"


# ════════════════════════════════════════════════════════════════
# E3 三个直写者字段名/路由钉子
# ════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_jiak(monkeypatch, tmp_path):
    """HOME→tmp + 伪造 jiak 库（真实库绝不入径）。"""
    jiak = tmp_path / ".hermes" / "jiak"
    jiak.mkdir(parents=True)
    (jiak / "RECALL.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    return jiak


class TestStatefulAuditWriter:
    def _make_trail(self, tmp_path, monkeypatch):
        from openllm.ios import stateful_audit
        monkeypatch.setattr(stateful_audit, "DEFAULT_AUDIT_DIR",
                            tmp_path / "audit")
        return stateful_audit.StatefulAuditTrail()

    def test_alert_written_with_ts_float_no_timestamp(self, tmp_jiak, monkeypatch, tmp_path):
        """门房不可用→回退直写：行里必须是 ts float 而非 timestamp。"""
        trail = self._make_trail(tmp_path, monkeypatch)
        trail._append_recall_alert("s1", 5.0, 3.0)
        lines = [l for l in (tmp_jiak / "RECALL.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert "timestamp" not in rec
        assert isinstance(rec["ts"], float) and rec["ts"] > 1_600_000_000

    def test_alert_routes_through_gate_end_to_end(self, tmp_jiak, monkeypatch, tmp_path):
        """门房存在→走门房：最终行带门房签名且 ts 为 float。"""
        gate_copy = tmp_jiak / "scripts" / "recall_append.py"
        gate_copy.parent.mkdir(parents=True, exist_ok=True)
        gate_copy.write_text(GATE_SRC.read_text(encoding="utf-8"),
                             encoding="utf-8")
        trail = self._make_trail(tmp_path, monkeypatch)
        trail._append_recall_alert("s2", 9.0, 3.0)
        lines = [l for l in (tmp_jiak / "RECALL.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1, "应经门房落一行"
        rec = json.loads(lines[0])
        assert "_written_by" in rec, "无门房签名=没走门房"
        assert "timestamp" not in rec
        assert isinstance(rec["ts"], float)
        assert rec["type"] == "security_alert"

    def test_archive_old_records_survives_iso_and_float(self, tmp_path, monkeypatch):
        """读侧比较点（:332）过 coerce_ts：ISO 字符串行不再击穿归档。"""
        from openllm.ios import stateful_audit
        monkeypatch.setattr(stateful_audit, "DEFAULT_AUDIT_DIR",
                            tmp_path / "audit")
        trail = stateful_audit.StatefulAuditTrail(retention_days=90)
        now = time.time()
        mixed = [
            # ISO 字符串时间戳（旧写法 str<float → TypeError → 归档整体失败）
            json.dumps({"session_id": "a", "timestamp":
                        datetime.fromtimestamp(now - 100 * 86400,
                                               tz=timezone.utc).isoformat()}),
            # epoch float（canonical）
            json.dumps({"session_id": "b", "timestamp": now - 100 * 86400}),
            # 新鲜行（应保留）
            json.dumps({"session_id": "c", "timestamp": now}),
        ]
        p = trail._records_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(mixed) + "\n", encoding="utf-8")
        archived = trail.archive_old_records()
        assert archived == 2, f"两条超期行(ISO串+float)都应归档, got {archived}"
        kept = [json.loads(l) for l in
                p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert [r["session_id"] for r in kept] == ["c"]


class TestIsnSignalWriter:
    def test_record_to_recall_fallback_writes_ts_float(self, tmp_jiak, monkeypatch):
        from openllm.core import isn_signal
        monkeypatch.setattr(isn_signal, "RECALL_PATH",
                            str(tmp_jiak / "RECALL.jsonl"))
        isn_signal.record_to_recall(
            {"type": "dream_insight", "from": "ISN", "payload": {"p": 1}})
        lines = [l for l in (tmp_jiak / "RECALL.jsonl").read_text(
            encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert "timestamp" not in rec
        assert isinstance(rec["ts"], float)

    def test_record_to_recall_routes_through_gate(self, tmp_jiak, monkeypatch, tmp_path):
        """门房存在→不再裸写：payload 交门房（stub 记录 argv）。"""
        from openllm.core import isn_signal
        monkeypatch.setattr(isn_signal, "RECALL_PATH",
                            str(tmp_jiak / "RECALL.jsonl"))
        scripts = tmp_jiak / "scripts"
        scripts.mkdir(parents=True, exist_ok=True)
        marker = tmp_path / "gate_marker.json"
        stub = scripts / "recall_append.py"
        stub.write_text(
            "import sys, json\n"
            f"open(r{str(marker)!r},'w').write(sys.argv[1])\n",
            encoding="utf-8")
        isn_signal.record_to_recall(
            {"type": "t", "from": "ISN", "payload": {}})
        assert marker.exists(), "没走门房"
        passed = json.loads(marker.read_text(encoding="utf-8"))
        assert isinstance(passed["ts"], float) and "timestamp" not in passed
        assert (tmp_jiak / "RECALL.jsonl").read_text(encoding="utf-8") == "", \
            "走门房路径后不得再裸写"


class TestRecallProviderStore:
    def test_store_builds_canonical_ts_field(self, tmp_jiak, monkeypatch):
        """store() 已走门房(validate_and_append)——断它交给门房的记录是 ts float。"""
        captured = {}

        def fake_append(record, *a, **kw):
            captured["record"] = record
            return {"success": True, "id": "x"}

        monkeypatch.setattr(rp, "_recall_append_loaded", True)
        monkeypatch.setattr(rp, "_recall_append", fake_append)
        provider = rp.RecallProvider(recall_path=tmp_jiak / "RECALL.jsonl")
        result = provider.store(memory_bus.WriteRequest(
            content="nail content", source="test"))
        assert result.success
        rec = captured["record"]
        assert "timestamp" not in rec, "字段名必须收口为 ts"
        assert isinstance(rec["ts"], float)


# ════════════════════════════════════════════════════════════════
# E4 ISO 假设读者双格式容忍钉子
# ════════════════════════════════════════════════════════════════

def _load_script_module(fname, name):
    path = REAL_HOME / ".hermes" / "jiak" / "scripts" / fname
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)  # 同目录依赖（jiak_meta_filter 等）
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestCardSizeGuardReader:
    def test_archive_old_recall_float_and_iso(self, tmp_path):
        csg = _load_script_module("card_size_guard.py", "csg_nail")
        now = time.time()
        rows = [
            json.dumps({"type": "e", "tag": "old_float",
                        "ts": now - 10 * 86400}),
            json.dumps({"type": "e", "tag": "fresh_float",
                        "ts": now - 3600}),
            json.dumps({"type": "e", "tag": "old_iso",
                        "ts": datetime.fromtimestamp(now - 10 * 86400,
                                                     tz=timezone.utc).isoformat()}),
            json.dumps({"type": "e", "tag": "garbage", "ts": "???"}),
            json.dumps({"type": "e", "tag": "no_ts"}),
        ]
        recall = tmp_path / "RECALL.jsonl"
        recall.write_text("\n".join(rows) + "\n", encoding="utf-8")
        csg.RECALL_PATH = recall
        csg.ARCHIVE_DIR = tmp_path / "archive"
        csg.archive_old_recall()
        kept = [json.loads(l)["tag"] for l in
                recall.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert sorted(kept) == ["fresh_float", "garbage", "no_ts"], \
            f"已知超期(ISO+float)归档、未知/新鲜保留, kept={kept}"


class TestJikaAuditReader:
    def test_parse_ts_accepts_float_numeric_iso(self):
        ja = _load_script_module("jika_audit.py", "ja_nail")
        now = time.time()
        dt = ja.parse_ts(now)
        assert dt is not None, "epoch float 不得返回 None（旧写法只认 ISO 串）"
        assert abs((dt - datetime.now()).total_seconds()) < 60
        dt2 = ja.parse_ts(str(now))
        assert dt2 is not None and abs((dt2 - datetime.now()).total_seconds()) < 60
        dt3 = ja.parse_ts("2026-09-01T00:00:00")
        assert dt3 == datetime(2026, 9, 1, 0, 0, 0)
        assert ja.parse_ts("not-a-date") is None
        assert ja.parse_ts(None) is None

    def test_rec_ts_field_chain(self):
        ja = _load_script_module("jika_audit.py", "ja_nail2")
        assert ja._rec_ts({"ts": 123.0}) == 123.0
        assert ja._rec_ts({"timestamp": "iso"}) == "iso"
        assert ja._rec_ts({}) == ""


class TestSleepConsolidationReader:
    def test_to_epoch_double_format(self):
        sc = _load_script_module("sleep_consolidation.py", "sc_nail")
        now = time.time()
        assert sc._to_epoch(now) == pytest.approx(now)
        assert sc._to_epoch(str(now)) == pytest.approx(now)
        assert sc._to_epoch("not-a-date") is None
        assert sc._to_epoch("") is None
        assert sc._to_epoch(None) is None

    def test_dissipate_float_ts_aged_row_can_dissipate(self):
        """旧写法 float ts → except → age=0 → 永不可耗散（巩固失明）。
        现 float 真老行必须参与年龄判定。"""
        sc = _load_script_module("sleep_consolidation.py", "sc_nail2")
        now_iso = datetime.now(timezone.utc).isoformat()
        recs = [
            {"content": "stale-float", "ts": time.time() - 60 * 86400,
             "access_count": 0, "trust": 0.3},
            {"content": "stale-iso", "ts": (
                datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),
             "access_count": 0, "trust": 0.3},
            {"content": "fresh", "ts": now_iso,
             "access_count": 0, "trust": 0.3},
            {"content": "unknown", "ts": "???",
             "access_count": 0, "trust": 0.3},
        ]
        retained, archived = sc.dissipate(recs)
        tags = {r["content"] for r in archived}
        assert "stale-float" in tags, "float ts 真老行不可耗散=旧写法失明复活"
        assert "stale-iso" in tags
        assert "fresh" not in tags
        assert "unknown" not in tags, "时间未知=最新保护，不得被耗散"


class TestIsaContextEditorReader:
    def test_load_records_survives_float_ts(self, tmp_path):
        ice = _load_script_module("isa_context_editor.py", "ice_nail")
        now = time.time()
        rows = [
            json.dumps({"type": "event", "content": "fresh float kw",
                        "ts": now - 3600}),
            json.dumps({"type": "event", "content": "ancient float kw",
                        "ts": now - 90 * 86400}),
            json.dumps({"type": "event", "content": "ancient iso kw",
                        "ts": datetime.fromtimestamp(now - 90 * 86400,
                                                     tz=timezone.utc).isoformat()}),
            json.dumps({"type": "event", "content": "no ts kw"}),
        ]
        recall = tmp_path / "RECALL.jsonl"
        recall.write_text("\n".join(rows) + "\n", encoding="utf-8")
        sel = ice.RecallMemorySelector(recall_path=recall)
        recs = sel._load_records(max_age_days=30)
        contents = [r["content"] for r in recs]
        assert "fresh float kw" in contents, "float ts 击穿读取通道=旧写法复活"
        assert "no ts kw" in contents, "时间未知应保留"
        assert "ancient float kw" not in contents
        assert "ancient iso kw" not in contents

    def test_select_card_id_no_typeerror_on_float(self, tmp_path):
        """card_id=f"recall-{ts[:10]}" 对 float 抛 TypeError——现必须双格式。"""
        ice = _load_script_module("isa_context_editor.py", "ice_nail3")
        now = time.time()
        recall = tmp_path / "RECALL.jsonl"
        recall.write_text(json.dumps(
            {"type": "event", "content": "alpha beta kw",
             "ts": now - 3600}) + "\n", encoding="utf-8")
        sel = ice.RecallMemorySelector(recall_path=recall)
        topic = ice.TopicPrediction(
            keywords=[("alpha", 1.0), ("beta", 1.0)], source_turns=1,
            elapsed_ms=0.0)
        cands = sel.select(topic, top_k=3)
        assert len(cands) == 1
        expect_day = datetime.fromtimestamp(now - 3600).strftime("%Y-%m-%d")
        assert cands[0].card_id == f"recall-{expect_day}"
