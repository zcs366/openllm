"""
K_W Phase 2 — 扩展d值 + Shannon修正验证 + 分布直方图
实验工程师出品。不评论，只出数据。

实验内容：
1. 扩展d值: [1, 2, 4, 8, 16, 32, 64]
2. 线性版: K_W = d × SNR vs Shannon版: C = d × log₂(1+SNR)
3. K_W分布直方图（每层）——检测双峰/多峰
4. 固定SNR阈值反推d（解耦d-SNR）
5. 每层单独输出
"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.decomposition import PCA
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))
from calibration_set import get_calibration_set

# ── 配置 ──
D_VALUES = [1, 2, 4, 8, 16, 32, 64]
CALIBRATION_SIZE = 100  # 全量校准集
OUTPUT_DIR = os.path.dirname(__file__)

def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
    model.eval()
    return tokenizer, model

def extract_attentions(model, tokenizer, text):
    """提取各层平均注意力矩阵 (seq_len, seq_len)"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    attentions = outputs.attentions  # (layers, heads, seq, seq)
    matrices = {}
    for i in range(len(attentions)):
        attn = attentions[i][0].mean(dim=0)  # avg heads → (seq, seq)
        matrices[i] = attn.numpy()
    return matrices

def measure_snr(attention_matrix, d):
    """
    PCA-based SNR测量。
    attention_matrix: (seq_len, seq_len)
    d: 保留维度数
    """
    seq_len, dim = attention_matrix.shape
    max_d = min(seq_len, dim)
    actual_d = min(d, max_d)
    
    if actual_d <= 0:
        return 0.0, 0.0, 0.0
    
    pca = PCA(n_components=actual_d)
    pca.fit(attention_matrix)
    signal_var = np.sum(pca.explained_variance_)
    total_var = np.sum(np.var(attention_matrix, axis=0))
    noise_var = total_var - signal_var
    
    if noise_var <= 0:
        return float('inf'), signal_var, noise_var
    
    snr = signal_var / noise_var
    return snr, signal_var, noise_var

def kw_linear(d, snr):
    """旧版: K_W = d × SNR"""
    return d * snr

def kw_shannon(d, snr):
    """Shannon版: C = d × log₂(1 + SNR)"""
    return d * np.log2(1 + snr)

def find_d_for_snr_threshold(attention_matrix, snr_threshold, d_values):
    """固定SNR阈值，找最小d使得 SNR ≥ snr_threshold"""
    for d in sorted(d_values):
        snr, _, _ = measure_snr(attention_matrix, d)
        if snr >= snr_threshold:
            return d, snr
    return None, None

def find_snr_for_d_threshold(attention_matrix, d_fixed):
    """固定d，直接返回对应SNR"""
    snr, _, _ = measure_snr(attention_matrix, d_fixed)
    return snr

def main():
    print("=" * 70)
    print("K_W Phase 2 — 扩展d值 + Shannon修正 + 分布检测")
    print("=" * 70)
    
    tokenizer, model = load_model()
    calibration = get_calibration_set()
    print(f"模型: GPT-2 (12层, 768维)")
    print(f"校准集: {len(calibration)}条")
    print(f"d值: {D_VALUES}")
    print()
    
    # ── 存储结果 ──
    num_layers = len(model.h)
    # layer_data[layer_idx][d] = {'snr': [...], 'kw_linear': [...], 'kw_shannon': [...]}
    layer_data = {i: {d: {'snr': [], 'kw_linear': [], 'kw_shannon': []} for d in D_VALUES} 
                  for i in range(num_layers)}
    
    # ── 采集 ──
    for idx, text in enumerate(calibration):
        if idx % 20 == 0:
            print(f"  样本 {idx}/{len(calibration)}...")
        
        matrices = extract_attentions(model, tokenizer, text)
        
        for layer_idx in range(num_layers):
            attn = matrices[layer_idx]
            seq_len = attn.shape[0]
            
            for d in D_VALUES:
                if d >= seq_len:
                    # d太大，跳过（退化）
                    continue
                
                snr, sig, noise = measure_snr(attn, d)
                if snr == float('inf') or np.isnan(snr):
                    continue
                
                kw_l = kw_linear(d, snr)
                kw_s = kw_shannon(d, snr)
                
                layer_data[layer_idx][d]['snr'].append(snr)
                layer_data[layer_idx][d]['kw_linear'].append(kw_l)
                layer_data[layer_idx][d]['kw_shannon'].append(kw_s)
    
    print("\n采集完成。\n")
    
    # ═══════════════════════════════════════════════════════════════
    # 实验 1: 每层 K_W vs d 曲线（线性 + Shannon）
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("实验 1: K_W vs d 曲线（每层）")
    print("=" * 70)
    
    summary_table = []
    
    for layer_idx in range(num_layers):
        print(f"\n--- Layer {layer_idx} ---")
        print(f"{'d':>4} | {'SNR_mean':>10} {'SNR_std':>10} | {'K_W(linear)':>12} {'K_W(Shannon)':>12} | {'n':>4}")
        print("-" * 75)
        
        row = {'layer': layer_idx, 'by_d': {}}
        
        for d in D_VALUES:
            data = layer_data[layer_idx][d]
            n = len(data['snr'])
            if n < 3:
                print(f"{d:4d} | {'N/A':>10} {'N/A':>10} | {'N/A':>12} {'N/A':>12} | {n:4d}")
                continue
            
            snr_arr = np.array(data['snr'])
            kw_l_arr = np.array(data['kw_linear'])
            kw_s_arr = np.array(data['kw_shannon'])
            
            snr_m, snr_s = np.mean(snr_arr), np.std(snr_arr)
            kw_l_m, kw_l_s = np.mean(kw_l_arr), np.std(kw_l_arr)
            kw_s_m, kw_s_s = np.mean(kw_s_arr), np.std(kw_s_arr)
            
            print(f"{d:4d} | {snr_m:10.4f} {snr_s:10.4f} | {kw_l_m:12.4f} {kw_s_m:12.4f} | {n:4d}")
            
            row['by_d'][d] = {
                'snr_mean': float(snr_m), 'snr_std': float(snr_s),
                'kw_linear_mean': float(kw_l_m), 'kw_linear_std': float(kw_l_s),
                'kw_shannon_mean': float(kw_s_m), 'kw_shannon_std': float(kw_s_s),
                'n': n
            }
        
        summary_table.append(row)
    
    # ═══════════════════════════════════════════════════════════════
    # 实验 2: K_W分布直方图（每层·检测双峰/多峰）
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("实验 2: K_W分布检测（双峰/多峰）")
    print("=" * 70)
    
    for layer_idx in range(num_layers):
        # 取d=4的数据（中间值，信号适中）
        d_ref = 4
        if d_ref not in layer_data[layer_idx]:
            continue
        kw_l = layer_data[layer_idx][d_ref]['kw_linear']
        if len(kw_l) < 10:
            continue
        
        arr = np.array(kw_l)
        n_bins = min(20, max(5, len(arr) // 3))
        counts, edges = np.histogram(arr, bins=n_bins)
        
        print(f"\nLayer {layer_idx} (d=4, linear K_W):")
        print(f"  n={len(arr)}, mean={np.mean(arr):.2f}, std={np.std(arr):.2f}")
        print(f"  min={np.min(arr):.2f}, max={np.max(arr):.2f}, median={np.median(arr):.2f}")
        
        # 检测双峰：用中位数分割，检查两半的均值差异
        med = np.median(arr)
        left = arr[arr < med]
        right = arr[arr >= med]
        
        if len(left) > 3 and len(right) > 3:
            diff = abs(np.mean(right) - np.mean(left))
            pooled_std = np.sqrt((np.std(left)**2 + np.std(right)**2) / 2)
            cohens_d = diff / pooled_std if pooled_std > 0 else 0
            
            bimodal = "YES" if cohens_d > 0.8 else ("MAYBE" if cohens_d > 0.4 else "NO")
            print(f"  双峰检测: Cohen's d={cohens_d:.3f} → {bimodal}")
            print(f"  左半: mean={np.mean(left):.2f}, n={len(left)}")
            print(f"  右半: mean={np.mean(right):.2f}, n={len(right)}")
        
        # ASCII直方图
        max_count = max(counts) if max(counts) > 0 else 1
        print(f"  分布直方图:")
        for i, c in enumerate(counts):
            bar_len = int(40 * c / max_count)
            lo, hi = edges[i], edges[i+1]
            print(f"    [{lo:7.1f},{hi:7.1f}) {'█' * bar_len} ({c})")
    
    # ═══════════════════════════════════════════════════════════════
    # 实验 3: Shannon修正验证——固定SNR阈值反推d
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("实验 3: 固定SNR阈值 → 反推d（解耦d-SNR）")
    print("=" * 70)
    
    SNR_THRESHOLDS = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    
    print(f"\n{'SNR阈值':>10} |", end="")
    for layer_idx in range(num_layers):
        print(f" L{layer_idx:2d}", end="")
    print(" |")
    print("-" * (12 + num_layers * 5 + 3))
    
    threshold_d_data = {}
    
    for thr in SNR_THRESHOLDS:
        threshold_d_data[thr] = {}
        print(f"{thr:10.1f} |", end="")
        
        for layer_idx in range(num_layers):
            # 收集该层所有样本在各d值下的SNR
            # 找最小d使得 SNR >= thr
            found_d = None
            for d in sorted(D_VALUES):
                snrs = layer_data[layer_idx][d]['snr']
                if len(snrs) == 0:
                    continue
                mean_snr = np.mean(snrs)
                if mean_snr >= thr:
                    found_d = d
                    break
            
            threshold_d_data[thr][layer_idx] = found_d
            
            if found_d is not None:
                print(f" {found_d:3d}", end="")
            else:
                print(f"  --", end="")
        
        print(" |")
    
    print("\n解读: 数字=使mean(SNR)≥阈值的最小d; '--'=即使d=64也不够")
    
    # ═══════════════════════════════════════════════════════════════
    # 实验 4: 固定d → 直接看SNR随d变化
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("实验 4: 固定d → SNR(d) 曲线（验证d-SNR耦合）")
    print("=" * 70)
    
    print(f"\n如果d和SNR耦合，增大d时SNR应下降（总方差不变，信号分母增大）")
    print()
    
    for layer_idx in [0, 5, 11]:  # 首/中/末层
        print(f"Layer {layer_idx}:")
        print(f"  {'d':>4} | {'mean(SNR)':>10} | {'mean(KW_l)':>10} | {'mean(KW_s)':>10}")
        print(f"  " + "-" * 45)
        
        prev_snr = None
        for d in D_VALUES:
            data = layer_data[layer_idx][d]
            if len(data['snr']) == 0:
                continue
            m_snr = np.mean(data['snr'])
            m_kw_l = np.mean(data['kw_linear'])
            m_kw_s = np.mean(data['kw_shannon'])
            
            snr_trend = ""
            if prev_snr is not None:
                ratio = m_snr / prev_snr if prev_snr > 0 else float('inf')
                if ratio < 0.9:
                    snr_trend = " ↓"
                elif ratio > 1.1:
                    snr_trend = " ↑"
            prev_snr = m_snr
            
            print(f"  {d:4d} | {m_snr:10.4f} | {m_kw_l:10.4f} | {m_kw_s:10.4f}{snr_trend}")
    
    # ═══════════════════════════════════════════════════════════════
    # 实验 5: 线性 vs Shannon — 哪个更守恒？
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("实验 5: 守恒性比较 — 线性 vs Shannon (CV对比)")
    print("=" * 70)
    
    print(f"\n{'Layer':>6} | {'CV(linear)':>10} | {'CV(Shannon)':>11} | {'更优':>6}")
    print("-" * 45)
    
    cv_comparison = []
    
    for layer_idx in range(num_layers):
        all_kw_l = []
        all_kw_s = []
        
        for d in D_VALUES:
            data = layer_data[layer_idx][d]
            all_kw_l.extend(data['kw_linear'])
            all_kw_s.extend(data['kw_shannon'])
        
        if len(all_kw_l) < 10:
            continue
        
        arr_l = np.array(all_kw_l)
        arr_s = np.array(all_kw_s)
        
        cv_l = np.std(arr_l) / np.mean(arr_l) if np.mean(arr_l) > 0 else float('inf')
        cv_s = np.std(arr_s) / np.mean(arr_s) if np.mean(arr_s) > 0 else float('inf')
        
        better = "SHANNON" if cv_s < cv_l else "LINEAR"
        print(f"L{layer_idx:5d} | {cv_l:10.4f} | {cv_s:11.4f} | {better:>6}")
        
        cv_comparison.append({
            'layer': layer_idx,
            'cv_linear': float(cv_l),
            'cv_shannon': float(cv_s),
            'better': better,
            'n': len(all_kw_l)
        })
    
    # ═══════════════════════════════════════════════════════════════
    # 实验 6: 深层 vs 浅层的K_W分布对比
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("实验 6: 深层 vs 浅层 K_W分布对比 (d=4)")
    print("=" * 70)
    
    d_ref = 4
    shallow_layers = [0, 1, 2, 3]
    deep_layers = [8, 9, 10, 11]
    
    for group_name, layers in [("浅层 L0-L3", shallow_layers), ("深层 L8-L11", deep_layers)]:
        all_kw = []
        for li in layers:
            all_kw.extend(layer_data[li][d_ref]['kw_linear'])
        
        if len(all_kw) < 5:
            continue
        
        arr = np.array(all_kw)
        print(f"\n{group_name} (d=4, linear):")
        print(f"  n={len(arr)}, mean={np.mean(arr):.2f}, std={np.std(arr):.2f}")
        print(f"  CV={np.std(arr)/np.mean(arr):.4f}")
        print(f"  min={np.min(arr):.2f}, max={np.max(arr):.2f}")
        
        # 分位数
        q = np.percentile(arr, [10, 25, 50, 75, 90])
        print(f"  P10={q[0]:.2f}, P25={q[1]:.2f}, P50={q[2]:.2f}, P75={q[3]:.2f}, P90={q[4]:.2f}")
    
    # ═══════════════════════════════════════════════════════════════
    # 保存全部结果
    # ═══════════════════════════════════════════════════════════════
    output = {
        'd_values': D_VALUES,
        'summary_table': summary_table,
        'threshold_d_data': {str(k): v for k, v in threshold_d_data.items()},
        'cv_comparison': cv_comparison,
    }
    
    out_path = os.path.join(OUTPUT_DIR, 'kw_phase2_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n结果已保存: {out_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
