"""
tests/test_closure_stress.py — O度量探针测试（≥8条含三区判定）
================================================================
验收判据：
  CS1 可import，ClosureStressProbe存在
  CS2 stress_test真调embed_similarity（代码证据）
  CS3 三区判定：构造三类样本→分别落safe/review/quarantine
  CS4 coupling_check真读总线+growth（代码证据）
  CS5 测试≥8条：三区判定/baseline稳定性/耦合检查三种情况/K零增长insufficient
  CS6 真实运行：baseline+一条真实知识样本stress_test输出zone
  CS7 bus.py加'closure.probed'一行
"""
import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── CS1: import检查 ──

class TestCS1Import:
    """CS1：模块可import，核心类存在。"""

    def test_import_probe(self):
        from openllm.identity.closure_stress import ClosureStressProbe
        assert ClosureStressProbe is not None

    def test_import_dataclasses(self):
        from openllm.identity.closure_stress import AnchorBaseline, StressResult, CouplingResult
        assert AnchorBaseline is not None
        assert StressResult is not None
        assert CouplingResult is not None


# ── CS7: bus.py包含closure.probed ──

class TestCS7BusType:
    """CS7：bus.py的KNOWN_TYPES包含closure.probed。"""

    def test_closure_probed_in_known_types(self):
        from openllm.evolution.bus import KNOWN_TYPES
        assert "closure.probed" in KNOWN_TYPES


# ── 辅助fixture ──

@pytest.fixture
def fake_embed():
    """构造可控的embed_similarity mock。

    策略：基于文本长度差异计算"相似度"，可控地落入三区。
    """
    def _embed(text_a: str, text_b: str) -> float:
        # 完全相同文本 → sim=1.0
        if text_a == text_b:
            return 1.0
        # 长度差越小越相似
        len_a, len_b = len(text_a), len(text_b)
        ratio = min(len_a, len_b) / max(len_a, len_b) if max(len_a, len_b) > 0 else 0.0
        # 映射到 [0.05, 0.95]
        return 0.05 + ratio * 0.90
    return _embed


@pytest.fixture
def mock_embed_fn():
    """精确控制相似度的mock。"""
    def _embed(text_a: str, text_b: str) -> float:
        # 按预设映射
        if text_b == "__SIMILAR__":
            return 0.95   # distance=0.05 → 吸收区
        elif text_b == "__MEDIUM__":
            return 0.40   # distance=0.60 → 边缘区
        elif text_b == "__FOREIGN__":
            return 0.10   # distance=0.90 → 异物区
        # 默认：hash扰动
        h = hash(text_b + text_a) % 1000 / 1000.0
        return h
    return _embed


# ── CS2: stress_test真调embed_similarity ──

class TestCS2EmbedCall:
    """CS2：stress_test内部真调用embed_similarity。"""

    def test_stress_test_calls_embed(self):
        """验证stress_test确实调用了传入的embed函数。"""
        call_count = 0

        def tracking_embed(a: str, b: str) -> float:
            nonlocal call_count
            call_count += 1
            return 0.9  # 高相似度 → 吸收区

        from openllm.identity.closure_stress import ClosureStressProbe
        probe = ClosureStressProbe(embed_fn=tracking_embed)
        probe.stress_test("test knowledge sample")

        assert call_count == 1, f"embed应被调用1次，实际{call_count}次"


# ── CS3: 三区判定 ──

class TestCS3ThreeZones:
    """CS3：三类样本分别落入safe/review/quarantine。"""

    def test_absorption_zone_safe(self, mock_embed_fn):
        """高相似度样本 → 吸收区 → safe。"""
        from openllm.identity.closure_stress import ClosureStressProbe
        probe = ClosureStressProbe(embed_fn=mock_embed_fn)
        result = probe.stress_test("__SIMILAR__")

        assert result["zone"] == "吸收区"
        assert result["verdict"] == "safe"
        assert result["distance"] < 0.5

    def test_fringe_zone_review(self, mock_embed_fn):
        """中等相似度样本 → 边缘区 → review。"""
        from openllm.identity.closure_stress import ClosureStressProbe
        probe = ClosureStressProbe(embed_fn=mock_embed_fn)
        result = probe.stress_test("__MEDIUM__")

        assert result["zone"] == "边缘区"
        assert result["verdict"] == "review"
        assert 0.5 <= result["distance"] <= 0.8

    def test_foreign_zone_quarantine(self, mock_embed_fn):
        """低相似度样本 → 异物区 → quarantine。"""
        from openllm.identity.closure_stress import ClosureStressProbe
        probe = ClosureStressProbe(embed_fn=mock_embed_fn)
        result = probe.stress_test("__FOREIGN__")

        assert result["zone"] == "异物区"
        assert result["verdict"] == "quarantine"
        assert result["distance"] > 0.8

    def test_boundary_absorption_to_fringe(self):
        """边界测试：distance恰好0.5 → 边缘区（>=0.5归边缘）。"""
        def exact_embed(a: str, b: str) -> float:
            return 0.50  # distance = 0.50

        from openllm.identity.closure_stress import ClosureStressProbe
        probe = ClosureStressProbe(embed_fn=exact_embed)
        result = probe.stress_test("boundary")

        assert result["zone"] == "边缘区"
        assert result["verdict"] == "review"

    def test_boundary_fringe_to_foreign(self):
        """边界测试：distance恰好0.8 → 异物区（>0.8归异物，=0.8归边缘）。"""
        def exact_embed(a: str, b: str) -> float:
            return 0.20  # distance = 0.80

        from openllm.identity.closure_stress import ClosureStressProbe
        probe = ClosureStressProbe(embed_fn=exact_embed)
        result = probe.stress_test("boundary2")

        # distance=0.80, 边缘区条件是 distance <= 0.8 → 边缘区
        assert result["zone"] == "边缘区"
        assert result["verdict"] == "review"

    def test_stress_result_structure(self, mock_embed_fn):
        """stress_test返回结构包含所有必需字段。"""
        from openllm.identity.closure_stress import ClosureStressProbe
        probe = ClosureStressProbe(embed_fn=mock_embed_fn)
        result = probe.stress_test("任意样本")

        assert "k_sample" in result
        assert "distance" in result
        assert "similarity" in result
        assert "zone" in result
        assert "verdict" in result
        assert isinstance(result["distance"], float)
        assert isinstance(result["similarity"], float)


# ── CS5: baseline测试 ──

class TestCS5Baseline:
    """CS5a：baseline稳定性。"""

    def test_baseline_deterministic_digest(self):
        """相同Soul两次baseline → digest相同。"""
        from openllm.identity.closure_stress import ClosureStressProbe
        probe = ClosureStressProbe(embed_fn=lambda a, b: 0.5)

        b1 = probe.baseline()
        b2 = probe.baseline()

        assert b1["digest"] == b2["digest"], "digest应确定性"
        assert b1["anchor_text"] == b2["anchor_text"]

    def test_baseline_structure(self):
        """baseline返回结构完整。"""
        from openllm.identity.closure_stress import ClosureStressProbe
        probe = ClosureStressProbe(embed_fn=lambda a, b: 0.5)

        b = probe.baseline()
        assert "anchor_text" in b
        assert "digest" in b
        assert "ts" in b
        assert "anchor_count" in b
        assert "values_count" in b
        assert b["anchor_count"] > 0
        assert b["values_count"] > 0


# ── CS5b: coupling_check三种情况 ──

class TestCS5Coupling:
    """CS5b：coupling_check的三种情况。"""

    def test_coupling_insufficient_data(self, mock_embed_fn, tmp_path):
        """K零增长 → coupled=None, insufficient-data。"""
        from openllm.identity.closure_stress import ClosureStressProbe
        empty_growth = tmp_path / "empty_growth.jsonl"
        probe = ClosureStressProbe(embed_fn=mock_embed_fn, growth_path=empty_growth)

        result = probe.coupling_check(days=7)

        assert result["coupled"] is None
        assert result["capacity_delta"] == 0
        assert result["note"] == "insufficient-data"

    def test_coupling_all_absorption(self, mock_embed_fn, tmp_path):
        """全部吸收区条目 → coupled=True。"""
        from openllm.identity.closure_stress import ClosureStressProbe

        growth_path = tmp_path / "growth.jsonl"
        # 写入几条"相似"条目
        now = time.time()
        with open(growth_path, "w", encoding="utf-8") as f:
            for i in range(3):
                rec = {"bullet_id": f"b{i}", "content": "__SIMILAR__", "ts": now - i * 100}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        probe = ClosureStressProbe(embed_fn=mock_embed_fn, growth_path=growth_path)
        result = probe.coupling_check(days=1)

        assert result["coupled"] is True
        assert result["capacity_delta"] == 3
        assert result["zone_dist"]["吸收区"] == 3
        assert "异物区" not in result["alarm"] or result["alarm"] == ""

    def test_coupling_foreign_contamination(self, mock_embed_fn, tmp_path):
        """含异物区条目 → coupled=False, alarm。"""
        from openllm.identity.closure_stress import ClosureStressProbe

        growth_path = tmp_path / "growth_contam.jsonl"
        now = time.time()
        with open(growth_path, "w", encoding="utf-8") as f:
            rec1 = {"bullet_id": "b1", "content": "__SIMILAR__", "ts": now - 100}
            rec2 = {"bullet_id": "b2", "content": "__FOREIGN__", "ts": now - 50}
            rec3 = {"bullet_id": "b3", "content": "__FOREIGN__", "ts": now - 10}
            f.write(json.dumps(rec1, ensure_ascii=False) + "\n")
            f.write(json.dumps(rec2, ensure_ascii=False) + "\n")
            f.write(json.dumps(rec3, ensure_ascii=False) + "\n")

        probe = ClosureStressProbe(embed_fn=mock_embed_fn, growth_path=growth_path)
        result = probe.coupling_check(days=1)

        assert result["coupled"] is False
        assert result["capacity_delta"] == 3
        assert result["zone_dist"]["异物区"] == 2
        assert "污染" in result["alarm"]


# ── CS6: 真实运行 ──

class TestCS6RealRun:
    """CS6：用真实embed_similarity运行baseline + stress_test。"""

    def test_real_baseline_and_stress(self):
        """真实运行baseline + 一条知识样本stress_test。"""
        from openllm.identity.closure_stress import ClosureStressProbe

        # 尝试真实embed（若模型不可用则跳过）
        try:
            from openllm.retrieval.hybrid import embed_similarity
            # 验证embed可用
            test_sim = embed_similarity("测试", "测试")
            if test_sim == 0.0:
                pytest.skip("embed_similarity返回0.0，模型可能未加载")
        except Exception:
            pytest.skip("embed_similarity不可用")

        probe = ClosureStressProbe()  # 默认用真实embed

        # baseline
        b = probe.baseline()
        assert b["anchor_text"], "anchor_text非空"
        assert len(b["digest"]) == 16

        # stress_test：一条与身份相关的真实知识
        sample = "OpenLLM是一个AI经验积累系统，由张成市创建，核心价值观是诚实和成长。"
        result = probe.stress_test(sample)

        assert "zone" in result
        assert "distance" in result
        assert result["zone"] in ("吸收区", "边缘区", "异物区")

        # 打印真实输出
        print(f"\n=== CS6 真实运行输出 ===")
        print(f"baseline digest: {b['digest']}")
        print(f"baseline anchor_count: {b['anchor_count']}, values_count: {b['values_count']}")
        print(f"sample: {sample[:40]}...")
        print(f"distance: {result['distance']}, similarity: {result['similarity']}")
        print(f"zone: {result['zone']}, verdict: {result['verdict']}")
