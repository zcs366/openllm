#!/usr/bin/env python3
"""布局监控哨兵 v0.1（CKA Sentinel）——循环深度系列的工程化交付。

功能：对 loop 模型（Huginn/lrd 接口族）的潜状态做在线布局监控：
  1. 基线期：收集 N_batch 个批次的语义布局（Gram 矩阵），建立基线 CKA 矩阵集
  2. 监控期：每个新批次的布局与基线比 CKA——布局突变（< 阈值）= 告警
  3. 输出：结构化 JSON 告警（不写叙述文字——审计类产出由数字说话）

设计依据（20260910-0911 实验系列）：
  - 语义布局（CKA 0.96）跨循环稳定 → 是可靠监控面
  - 地址（维度/位置）不存在 → 放弃 neuron-level 监控
  - 容量守恒限于训练分布 → D₀ 同时监控，出分布即降（lrd 实测腰斩）
  - **D₀(R) 剖面 = 输入分布指纹**（0911 三池对照）：模板池 4.14 / 混合池 1.78 / 自然池 1.11
    → 第二通道，5 次 forward 即可给输入类型打指纹，并检测输入分布突变

通道：
  1. CKA 布局（逐批）：LAYOUT_SHIFT
  2. D₀ 容量（逐批）：CAPACITY_COLLAPSE
  3. D₀(R) 剖面（抽样，需多次 forward）：PROFILE_SHIFT + 指纹分类

用法：
    from openllm.core.ckal_sentinel import CKALSentinel
    s = CKALSentinel(baseline_batches=8, cka_floor=0.85, d0_drop=0.5)
    s.observe(latent_batch)   # 每个推理批次后调用（[B,S,E]或[B,E]）
    s.report()                # 随时取状态 JSON

    # 第三通道（抽样调用，非每批）：
    d0r = [d0(latents_at_R) for R in (1,2,4,16,32)]   # 同一输入、不同循环深度
    s.check_profile(d0r)      # → {"fingerprint": "template", "collapse": 0.71, "alerts": [...]}
"""
import json
import numpy as np
from collections import deque


def _mean_pool(x, mask=None):
    """[B,S,E]→[B,E]（mask过padding）；[B,E]直通。"""
    if x.ndim == 2:
        return x
    if mask is None:
        return x.mean(axis=1)
    m = mask.astype(x.dtype)
    return (x * m[..., None]).sum(1) / np.clip(m.sum(1), 1, None)[..., None]


def cka(X, Y):
    """线性 CKA（居中 Gram 矩阵夹角）。"""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Xc.T @ Yc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(num / max(den, 1e-12))


def d0(X, thresh=0.95):
    """PCA-95% 本征维度（token级样本≥300时可靠）。"""
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    ev = s ** 2
    cum = np.cumsum(ev) / max(ev.sum(), 1e-12)
    return int(np.searchsorted(cum, thresh) + 1)


# ---------- D₀(R) 剖面指纹（第三通道）----------
# 来源：20260911 三池对照实测（N=4000 处，R=(1,2,4,16,32)，已归一化 D₀(R)/D₀(R1)）
# 文档：河床 0911三项目立项/布局语义学第一期裁决-D0R剖面由输入同质度支配-20260911.md
D0R_FINGERPRINTS = {
    "template": [1.000, 0.627, 0.353, 0.286, 0.289],   # SVO 模板池：剧烈压缩（比值 4.14）
    "mixed":    [1.000, 0.800, 0.672, 0.596, 0.595],   # 混合格式池：中度压缩（比值 1.78）
    "prose":    [1.000, 0.952, 0.969, 0.940, 0.940],   # 自然散文池：基本平坦（比值 1.11）
}
D0R_SHIFT = 0.30   # 剖面塌缩率相对基线的变化阈值（绝对值）


def profile_shape(d0r):
    """D₀(R) 列表 → 归一化剖面（除以首项）；首项为 0 时返回 None。"""
    d0r = [float(x) for x in d0r]
    if not d0r or d0r[0] <= 0:
        return None
    return [x / d0r[0] for x in d0r]


def profile_collapse(d0r):
    """塌缩率 = 1 - D₀(R_last)/D₀(R_first)。模板池≈0.71，自然池≈0.06。"""
    s = profile_shape(d0r)
    return None if s is None else float(1.0 - s[-1])


def classify_profile(d0r, fingerprints=None):
    """最近邻指纹分类 → (名称, 欧氏距离, 全部距离表)。"""
    s = profile_shape(d0r)
    if s is None:
        return None, None, {}
    fps = fingerprints or D0R_FINGERPRINTS
    dists = {k: float(np.linalg.norm(np.array(s) - np.array(v))) for k, v in fps.items()}
    best = min(dists, key=dists.get)
    return best, dists[best], dists


class CKALSentinel:
    """布局监控哨兵：CKA 布局 + D₀ 容量 + D₀(R) 剖面三通道。"""

    def __init__(self, baseline_batches=8, cka_floor=0.85, d0_drop=0.5,
                 min_tokens=300, max_batches=200):
        self.baseline_batches = baseline_batches
        self.cka_floor = cka_floor
        self.d0_drop = d0_drop
        self.min_tokens = min_tokens
        self.max_batches = max_batches

        self._baseline_ckas = []      # 基线期内部两两 CKA（估自然波动）
        self._baseline_grams = []     # 基线 Gram 矩阵
        self._baseline_d0 = None
        self.state = "warming"        # warming → armed
        self._n_batches = 0
        self._alerts = deque(maxlen=50)
        self._recent_ckas = deque(maxlen=max_batches)
        self._recent_d0s = deque(maxlen=max_batches)
        self._baseline_profile = None
        self._baseline_collapse = None

    # ---------- 内部 ----------
    def _gram(self, X):
        Xc = X - X.mean(0, keepdims=True)
        G = Xc @ Xc.T
        return G / max(np.linalg.norm(G), 1e-12)

    def _tok_pool(self, x, mask):
        """拍平到 token 级（D₀ 用），同时出句级池化（CKA 用）。"""
        flat = []
        pooled = []
        if x.ndim == 3:
            mask = np.ones(x.shape[:2], dtype=x.dtype) if mask is None else mask
            for b in range(x.shape[0]):
                tk = x[b][mask[b].astype(bool)]
                flat.append(tk)
                pooled.append(tk.mean(0))
            X_tok = np.concatenate(flat, 0)
        else:
            X_tok, pooled = x, list(x)
        return X_tok, np.array(pooled)

    # ---------- 主接口 ----------
    def observe(self, x, mask=None):
        """每推理批次调用。x:[B,S,E] 或 [B,E]；mask:[B,S]（可选）。"""
        self._n_batches += 1
        X_tok, pooled = self._tok_pool(np.asarray(x), mask)

        # D₀ 通道（样本够才算）
        d0_now = None
        if X_tok.shape[0] >= self.min_tokens:
            d0_now = d0(X_tok)
            self._recent_d0s.append(d0_now)

        # 基线期
        if self.state == "warming":
            self._baseline_grams.append(self._gram(pooled))
            if len(self._baseline_grams) >= 2:
                self._baseline_ckas.append(cka(
                    self._baseline_grams[-2], self._baseline_grams[-1]))
            if len(self._baseline_grams) >= self.baseline_batches:
                self._baseline_d0 = d0_now or (
                    d0(X_tok) if X_tok.shape[0] >= self.min_tokens else None)
                self.state = "armed"
            return {"state": self.state, "d0": d0_now}

        # 监控期：与基线最后一帧比 CKA
        c = cka(self._baseline_grams[-1], self._gram(pooled))
        self._recent_ckas.append(c)

        alerts = []
        if c < self.cka_floor:
            alerts.append({
                "type": "LAYOUT_SHIFT", "cka": round(c, 4),
                "floor": self.cka_floor,
                "meaning": "语义布局突变——分布外输入或模型状态异常",
            })
        if d0_now is not None and self._baseline_d0:
            ratio = d0_now / self._baseline_d0
            if ratio < self.d0_drop:
                alerts.append({
                    "type": "CAPACITY_COLLAPSE", "d0_now": d0_now,
                    "d0_baseline": self._baseline_d0, "ratio": round(ratio, 3),
                    "meaning": "本征维度塌缩——循环出训练分布或表征退化",
                })
            self._recent_d0s.append(d0_now)

        for a in alerts:
            a["batch"] = self._n_batches
            self._alerts.append(a)

        return {"state": self.state, "cka": round(c, 4), "d0": d0_now,
                "alerts": alerts}

    # ---------- 查询 ----------
    def report(self):
        """结构化状态（审计用：只有数字，没有叙述）。"""
        rc = list(self._recent_ckas)
        rd = list(self._recent_d0s)
        return {
            "state": self.state,
            "batches_observed": self._n_batches,
            "baseline_batches": len(self._baseline_grams),
            "baseline_d0": self._baseline_d0,
            "baseline_internal_cka": {
                "mean": round(float(np.mean(self._baseline_ckas)), 4) if self._baseline_ckas else None,
                "min": round(float(np.min(self._baseline_ckas)), 4) if self._baseline_ckas else None,
            },
            "recent_cka": {
                "n": len(rc),
                "mean": round(float(np.mean(rc)), 4) if rc else None,
                "min": round(float(np.min(rc)), 4) if rc else None,
            },
            "recent_d0": {
                "n": len(rd),
                "mean": round(float(np.mean(rd)), 1) if rd else None,
                "min": min(rd) if rd else None,
            },
            "alerts_total": len(self._alerts),
            "alerts_recent": list(self._alerts)[-5:],
        }

    # ---------- 第三通道：D₀(R) 剖面 ----------
    def arm_profile(self, reference_d0r):
        """设立剖面基线（同分布输入上测一次，5 次 forward）。"""
        self._baseline_profile = profile_shape(reference_d0r)
        self._baseline_collapse = profile_collapse(reference_d0r)
        return {"baseline_profile": self._baseline_profile,
                "baseline_collapse": self._baseline_collapse,
                "fingerprint": classify_profile(reference_d0r)[0]}

    def check_profile(self, d0r):
        """抽样检查：输入类型指纹 + 相对基线的剖面突变告警。

        d0r: 同一输入在 R=(1,2,4,16,32) 下的 D₀ 列表。
        """
        name, dist, dists = classify_profile(d0r)
        collapse = profile_collapse(d0r)
        alerts = []
        if (self._baseline_collapse is not None and collapse is not None
                and abs(collapse - self._baseline_collapse) > D0R_SHIFT):
            alerts.append({
                "type": "PROFILE_SHIFT",
                "collapse_baseline": round(self._baseline_collapse, 3),
                "collapse_now": round(collapse, 3),
                "delta": round(collapse - self._baseline_collapse, 3),
                "meaning": "输入分布结构突变（同质度骤变）——输入类型改变或数据源污染",
            })
        for a in alerts:
            a["batch"] = self._n_batches
            self._alerts.append(a)
        return {"fingerprint": name, "fingerprint_dist": round(dist, 4) if dist is not None else None,
                "collapse": round(collapse, 3) if collapse is not None else None,
                "profile": [round(v, 3) for v in (profile_shape(d0r) or [])],
                "all_dists": {k: round(v, 4) for k, v in dists.items()},
                "alerts": alerts}

    def alerts(self):
        return list(self._alerts)


# ---------- 自测（mock 数据，可独立运行） ----------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    s = CKALSentinel(baseline_batches=4, cka_floor=0.85, d0_drop=0.5)

    # 基线期：低秩子空间随机批
    base_dir = rng.standard_normal((20, 64))
    for _ in range(4):
        batch = base_dir @ rng.standard_normal((64, 768)) * 0.1 + rng.standard_normal((20, 768)) * 0.01
        s.observe(batch)

    # 正常批（同分布）
    r1 = s.observe(base_dir @ rng.standard_normal((64, 768)) * 0.1 + rng.standard_normal((20, 768)) * 0.01)
    print("normal batch:", r1)

    # 异常批：换子空间（布局突变）
    other_dir = rng.standard_normal((20, 64))
    r2 = s.observe(other_dir @ rng.standard_normal((64, 768)) * 0.1 + rng.standard_normal((20, 768)) * 0.01)
    print("anomaly batch:", r2)

    # 塌缩批：降维 90%
    collapsed = base_dir[:, :6] @ rng.standard_normal((6, 768)) * 0.1 + rng.standard_normal((20, 768)) * 0.005
    r3 = s.observe(collapsed)
    print("collapse batch:", r3)

    rep = s.report()
    print("\nreport:", json.dumps(rep, ensure_ascii=False, indent=1)[:600])
    assert r2.get("alerts"), "布局突变应触发告警"

    # --- 第三通道：D₀(R) 剖面指纹（免GPU的合成自测）---
    # 合成三型剖面（取自 0911 实测形状的近似）
    d0r_template = [1000, 630, 350, 290, 290]
    d0r_prose = [1000, 950, 970, 940, 940]
    d0r_mixed = [1000, 795, 660, 585, 585]
    assert classify_profile(d0r_template)[0] == "template"
    assert classify_profile(d0r_prose)[0] == "prose"
    assert classify_profile(d0r_mixed)[0] == "mixed"
    assert abs(profile_collapse(d0r_template) - 0.71) < 0.02
    assert abs(profile_collapse(d0r_prose) - 0.06) < 0.02
    s2 = CKALSentinel(); s2._n_batches = 5
    s2.arm_profile(d0r_prose)
    assert not s2.check_profile(d0r_prose)["alerts"], "同分布不应误报"
    assert s2.check_profile(d0r_template)["alerts"], "剖面突变应告警"
    print("PROFILE CHANNEL SELF-TEST PASS")
    print("\nSELF-TEST PASS")
