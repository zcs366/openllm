"""
P0-紧急: G3b + G2 + B + C + E 串行执行
按v2.0实验设计严格执行。否决门优先。

G3b: 球面均匀分布基线 (否决门)
G2:  随机初始化 vs 训练后
B:   残差更新独立D₀ + ‖f‖/‖h‖
C:   CCA子空间对齐 + 置换检验
E:   ln_f有效秩 + weight tying
"""

import torch
import numpy as np
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from sklearn.cross_decomposition import CCA
import json, os, time, gc
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.dirname(__file__)
np.random.seed(42)

# ── 标准化工具 (v2.0协议) ──

def pca_95(matrix):
    """SVD-based PCA-95% → D₀"""
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    var = s ** 2
    var_cum = np.cumsum(var) / var.sum()
    return int(np.searchsorted(var_cum, 0.95) + 1)

def normalize(h):
    """L2归一化"""
    return h / np.linalg.norm(h, axis=-1, keepdims=True)


# ── 校准集 ──

def generate_texts(n=50):
    import random
    random.seed(42)
    templates = [
        "The {adj} {noun} {verb} over the {adj2} {noun2} near the {place}.",
        "Scientists discovered that {noun} can {verb} when exposed to {noun2}.",
        "The {adj} revolution in {field} is transforming how we {verb}.",
        "According to recent research, {noun} {verb} more efficiently than {noun2}.",
        "In the {adj} world of {field}, {noun} remains the most {adj2} {noun2}.",
    ]
    adjs = ['rapid', 'complex', 'emerging', 'innovative', 'fundamental']
    nouns = ['technology', 'algorithm', 'neuron', 'genome', 'ecosystem']
    verbs = ['transforms', 'accelerates', 'generates', 'optimizes', 'disrupts']
    nouns2 = ['data', 'energy', 'light', 'matter', 'signal']
    places = ['laboratory', 'observatory', 'university', 'workstation', 'deep sea']
    fields = ['AI', 'quantum computing', 'biotech', 'nanotech', 'neuroscience']
    
    texts = []
    for i in range(n):
        t = random.choice(templates)
        for placeholder, choices in [('{adj}', adjs), ('{noun}', nouns), ('{verb}', verbs),
                                      ('{noun2}', nouns2), ('{place}', places), ('{field}', fields),
                                      ('{adj2}', adjs)]:
            t = t.replace(placeholder, random.choice(choices))
        texts.append(t)
    return texts


def extract_hidden_states(model, tokenizer, texts, max_length=128):
    """提取所有层的hidden states"""
    all_hidden = {}
    
    for idx, text in enumerate(texts):
        inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        
        for li, h in enumerate(out.hidden_states):
            if li not in all_hidden:
                all_hidden[li] = []
            all_hidden[li].append(h[0].cpu().float().numpy())
    
    return all_hidden


# ═══════════════════════════════════════════════════════════════
# G3b: 球面均匀分布基线 (否决门)
# ═══════════════════════════════════════════════════════════════

def run_G3b():
    print("=" * 70)
    print("G3b: 球面均匀分布基线 (否决门·5min)")
    print("=" * 70)
    
    t0 = time.time()
    
    # 球面均匀分布: Marsaglia方法
    N_points = 200  # 匹配典型seq_len
    random_sphere = np.random.randn(N_points, 768)
    random_sphere = normalize(random_sphere)
    
    D0_uniform = pca_95(random_sphere)
    
    # 多次采样验证稳定性
    D0_samples = []
    for _ in range(10):
        pts = np.random.randn(N_points, 768)
        pts = normalize(pts)
        D0_samples.append(pca_95(pts))
    
    D0_mean = np.mean(D0_samples)
    D0_std = np.std(D0_samples)
    
    elapsed = time.time() - t0
    
    print(f"\n球面均匀分布 (N={N_points}, dim=768):")
    print(f"  D₀ = {D0_uniform}")
    print(f"  10次采样: {D0_mean:.1f} ± {D0_std:.1f}")
    print(f"  耗时: {elapsed:.1f}s")
    
    # 否决门判定
    if D0_uniform <= 110:
        print(f"\n🔴 否决门: D₀_uniform={D0_uniform} ≤ 110 → 项目终止")
        print(f"  103是球面几何artifact，非GPT-2性质")
        return False, D0_uniform
    elif D0_uniform >= 250:
        print(f"\n⭑ 通过: D₀_uniform={D0_uniform} ≥ 250 → 均匀分布给250+维")
        print(f"  103明显更小→GPT-2有结构偏置")
        return True, D0_uniform
    else:
        print(f"\n⚠️ 边界: D₀_uniform={D0_uniform} (110-250之间)")
        print(f"  继续执行，但结果需谨慎解读")
        return True, D0_uniform


# ═══════════════════════════════════════════════════════════════
# G2: 随机初始化 vs 训练后
# ═══════════════════════════════════════════════════════════════

def run_G2(tokenizer, texts):
    print("\n" + "=" * 70)
    print("G2: 随机初始化 vs 训练后")
    print("=" * 70)
    
    t0 = time.time()
    
    # 随机初始化
    config = GPT2Config()
    model_random = GPT2Model(config)
    model_random.eval()
    
    hidden_random = extract_hidden_states(model_random, tokenizer, texts)
    
    # 训练后
    model_trained = GPT2Model.from_pretrained('gpt2')
    model_trained.eval()
    
    hidden_trained = extract_hidden_states(model_trained, tokenizer, texts)
    
    del model_random, model_trained
    gc.collect()
    
    elapsed = time.time() - t0
    
    # 测量D₀
    print(f"\n{'Layer':>6} | {'D₀(random)':>11} {'D₀(trained)':>12} {'Delta':>8}")
    print("-" * 50)
    
    d0_random_list = []
    d0_trained_list = []
    
    for li in sorted(hidden_random.keys()):
        # 拼接所有文本
        h_rand = np.vstack(hidden_random[li])
        h_train = np.vstack(hidden_trained[li])
        
        h_rand_norm = normalize(h_rand)
        h_train_norm = normalize(h_train)
        
        d0_rand = pca_95(h_rand_norm)
        d0_train = pca_95(h_train_norm)
        
        d0_random_list.append(d0_rand)
        d0_trained_list.append(d0_train)
        
        delta = d0_train - d0_rand
        marker = " ***" if abs(delta) > 20 else ""
        print(f"L{li:4d} | {d0_rand:11d} {d0_train:12d} {delta:>+8d}{marker}")
    
    arr_rand = np.array(d0_random_list)
    arr_train = np.array(d0_trained_list)
    
    print(f"\n随机初始化: mean={np.mean(arr_rand):.1f}, std={np.std(arr_rand):.1f}")
    print(f"训练后:     mean={np.mean(arr_train):.1f}, std={np.std(arr_train):.1f}")
    print(f"比率:       {np.mean(arr_train)/np.mean(arr_rand):.2f}")
    print(f"耗时: {elapsed:.1f}s")
    
    return {
        'd0_random': d0_random_list,
        'd0_trained': d0_trained_list,
        'mean_random': float(np.mean(arr_rand)),
        'mean_trained': float(np.mean(arr_train)),
        'ratio': float(np.mean(arr_train) / np.mean(arr_rand)),
    }


# ═══════════════════════════════════════════════════════════════
# B: 残差更新独立D₀ + ‖f‖/‖h‖
# ═══════════════════════════════════════════════════════════════

def run_B(tokenizer, texts):
    print("\n" + "=" * 70)
    print("B: 残差更新独立D₀ + ‖f‖/‖h‖")
    print("=" * 70)
    
    t0 = time.time()
    
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()
    
    hidden = extract_hidden_states(model, tokenizer, texts)
    del model
    gc.collect()
    
    elapsed = time.time() - t0
    
    print(f"\n{'Layer':>6} | {'D₀(f)':>7} {'D₀(h)':>7} {'‖f‖/‖h‖':>10} {'‖f‖_mean':>10} {'‖h‖_mean':>10}")
    print("-" * 60)
    
    results = []
    
    for li in range(11):  # L0→L1 ... L10→L11
        h_in = np.vstack(hidden[li])      # 残差流进入
        h_out = np.vstack(hidden[li + 1])  # 残差流离开
        
        f_l = h_out - h_in  # 残差更新
        
        f_norm = normalize(f_l)
        h_norm = normalize(h_out)
        
        d0_f = pca_95(f_norm)
        d0_h = pca_95(h_norm)
        
        # 相对更新幅度
        norm_f = np.mean(np.linalg.norm(f_l, axis=-1))
        norm_h = np.mean(np.linalg.norm(h_in, axis=-1))
        rel_update = norm_f / norm_h if norm_h > 0 else 0
        
        print(f"L{li:4d} | {d0_f:7d} {d0_h:7d} {rel_update:10.4f} {norm_f:10.4f} {norm_h:10.4f}")
        
        results.append({
            'layer': li,
            'd0_f': d0_f,
            'd0_h': d0_h,
            'rel_update': float(rel_update),
            'norm_f': float(norm_f),
            'norm_h': float(norm_h),
        })
    
    # 汇总
    d0_f_arr = np.array([r['d0_f'] for r in results])
    rel_arr = np.array([r['rel_update'] for r in results])
    
    print(f"\nD₀(f) 均值: {np.mean(d0_f_arr):.1f} ± {np.std(d0_f_arr):.1f}")
    print(f"‖f‖/‖h‖ 均值: {np.mean(rel_arr):.4f}")
    
    if np.mean(rel_arr) < 0.10:
        print(f"⚠️ ‖f‖/‖h‖ < 0.10 → D₀(f_norm)解释需谨慎")
    
    return results


# ═══════════════════════════════════════════════════════════════
# C: CCA子空间对齐 + 置换检验
# ═══════════════════════════════════════════════════════════════

def run_C(tokenizer, texts):
    print("\n" + "=" * 70)
    print("C: CCA子空间对齐 + 置换检验")
    print("=" * 70)
    
    t0 = time.time()
    
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()
    
    hidden = extract_hidden_states(model, tokenizer, texts)
    del model
    gc.collect()
    
    elapsed = time.time() - t0
    
    n_components = 20
    
    print(f"\n{'Pair':>8} | {'ρ_real':>8} {'ρ_null':>8} {'ratio':>8} | {'判定':>10}")
    print("-" * 55)
    
    cca_results = []
    
    for li in range(11):
        X = normalize(np.vstack(hidden[li]))
        Y = normalize(np.vstack(hidden[li + 1]))
        N = X.shape[0]
        
        # 真实CCA
        cca_real = CCA(n_components=n_components)
        X_c, Y_c = cca_real.fit_transform(X, Y)
        rho_real = np.array([np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)])
        mean_rho_real = np.mean(rho_real[:10])
        
        # 置换检验
        X_shuffled = X[np.random.permutation(N)]
        cca_null = CCA(n_components=n_components)
        Xs_c, Y_c = cca_null.fit_transform(X_shuffled, Y)
        rho_null = np.array([np.corrcoef(Xs_c[:, k], Y_c[:, k])[0, 1] for k in range(n_components)])
        mean_rho_null = np.mean(rho_null[:10])
        
        ratio = mean_rho_real / mean_rho_null if mean_rho_null > 0 else float('inf')
        
        if ratio > 3:
            verdict = "确认对齐"
        elif ratio > 1.5:
            verdict = "弱对齐"
        else:
            verdict = "artifact"
        
        print(f"L{li:2d}→L{li+1:2d} | {mean_rho_real:8.3f} {mean_rho_null:8.3f} {ratio:8.1f}x | {verdict:>10}")
        
        cca_results.append({
            'pair': f'L{li}→L{li+1}',
            'rho_real': float(mean_rho_real),
            'rho_null': float(mean_rho_null),
            'ratio': float(ratio),
        })
    
    # 跨远距离
    X_L0 = normalize(np.vstack(hidden[0]))
    Y_L10 = normalize(np.vstack(hidden[10]))
    
    cca_far = CCA(n_components=20)
    X_c, Y_c = cca_far.fit_transform(X_L0, Y_L10)
    rho_far = np.array([np.corrcoef(X_c[:, k], Y_c[:, k])[0, 1] for k in range(20)])
    
    print(f"\nL0→L10 (远距离): ρ前10={np.mean(rho_far[:10]):.3f}, ρ前5={np.mean(rho_far[:5]):.3f}")
    
    # 判定
    ratios = [r['ratio'] for r in cca_results]
    if np.mean(ratios) > 3 and np.mean(rho_far[:10]) > 0.7:
        print("⭑ 场景A: 同一空间，远距离缓慢旋转")
    elif np.mean(ratios) > 3 and np.mean(rho_far[:10]) < 0.3:
        print("场景C: 空间缓慢漂移，深层已基本独立")
    elif np.mean(ratios) < 1.5:
        print("🔴 CCA是artifact。高维小样本抬高了所有ρ")
    
    return cca_results


# ═══════════════════════════════════════════════════════════════
# E: ln_f有效秩 + weight tying
# ═══════════════════════════════════════════════════════════════

def run_E(tokenizer, texts):
    print("\n" + "=" * 70)
    print("E: ln_f有效秩 + weight tying")
    print("=" * 70)
    
    t0 = time.time()
    
    model = GPT2Model.from_pretrained('gpt2')
    model.eval()
    
    # E1: embedding有效秩 (weight tying: lm_head = wte.T)
    W = model.wte.weight.detach().cpu().numpy()  # [50257, 768]
    u, s, vt = np.linalg.svd(W, full_matrices=False)
    var = s ** 2
    var_cum = np.cumsum(var) / var.sum()
    D0_lmhead = int(np.searchsorted(var_cum, 0.95) + 1)
    
    print(f"E1: embedding/lm_head D₀ = {D0_lmhead}")
    
    # E2: ln_f前后的D₀变化
    hidden = extract_hidden_states(model, tokenizer, texts[:10])  # 10条就够
    
    h_L11 = np.vstack(hidden[11])  # L11输出 (ln_f之前)
    h_L11_norm = normalize(h_L11)
    D0_pre = pca_95(h_L11_norm)
    
    # ln_f之后
    h_tensor = torch.tensor(h_L11, dtype=torch.float32)
    with torch.no_grad():
        h_post_ln = model.ln_f(h_tensor).numpy()
    h_post_ln_norm = normalize(h_post_ln)
    D0_post = pca_95(h_post_ln_norm)
    
    print(f"E2: ln_f前 D₀ = {D0_pre}, ln_f后 D₀ = {D0_post}")
    
    # E3: ln_f的gamma分析
    gamma = model.ln_f.weight.detach().cpu().numpy()
    near_zero = np.sum(np.abs(gamma) < 0.01)
    
    print(f"E3: Gamma near-zero: {near_zero}/768")
    print(f"    Gamma range: [{gamma.min():.4f}, {gamma.max():.4f}]")
    print(f"    Gamma mean: {gamma.mean():.4f}, std: {gamma.std():.4f}")
    
    # 判定
    if D0_lmhead > 50 and D0_pre > 50 and D0_post < 40:
        print(f"\n⭑ ln_f将{D0_pre}维压缩到{D0_post}维。来源是LayerNorm缩放效应")
    elif D0_pre < 40:
        print(f"\n⚠️ ln_f之前D₀已经低({D0_pre})→L11→L12的FFN做了压缩")
    
    elapsed = time.time() - t0
    print(f"耗时: {elapsed:.1f}s")
    
    return {
        'D0_lmhead': D0_lmhead,
        'D0_pre_ln': D0_pre,
        'D0_post_ln': D0_post,
        'gamma_near_zero': int(near_zero),
        'gamma_range': [float(gamma.min()), float(gamma.max())],
    }


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("P0-紧急: v2.0实验设计完整执行")
    print("=" * 70)
    
    texts = generate_texts(50)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    all_results = {}
    
    # ── G3b 否决门 ──
    passed, D0_uniform = run_G3b()
    all_results['G3b'] = {'passed': passed, 'D0_uniform': D0_uniform}
    
    if not passed:
        print("\n🔴 G3b否决。项目终止。不浪费后续token。")
        save_results(all_results)
        return
    
    # ── G2 ──
    g2_result = run_G2(tokenizer, texts)
    all_results['G2'] = g2_result
    
    # ── B ──
    b_result = run_B(tokenizer, texts)
    all_results['B'] = b_result
    
    # ── C ──
    c_result = run_C(tokenizer, texts)
    all_results['C'] = c_result
    
    # ── E ──
    e_result = run_E(tokenizer, texts)
    all_results['E'] = e_result
    
    save_results(all_results)


def save_results(results):
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj
    
    out_path = os.path.join(OUTPUT_DIR, 'p0_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
