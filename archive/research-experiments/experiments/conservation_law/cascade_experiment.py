"""
三体串联验证 — L10守恒核假说检验

假说:
  L10→L11 = 守恒核边界。干扰此处→输出剧烈崩塌。
  L0→L1 = 弹性区。干扰此处→输出恢复。

方法:
  1. 基线: 正常前向，记录最终logits
  2. 注入: 在layer_N output处加高斯噪声，再喂给layer_N+1
  3. 度量: KL散度 / 余弦距离 / top-1命中率变化
  4. 扫描噪声强度: SNR ∈ {100, 50, 20, 10, 5, 2, 1, 0.5}
"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from torch.nn import functional as F
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))
from calibration_set import get_calibration_set

# ── 配置 ──
NOISE_SNR_VALUES = [100, 50, 20, 10, 5, 2, 1, 0.5]
INJECTION_SITES = {
    'L0→L1': 0,   # 浅层边界: hook on layer 0 output
    'L10→L11': 10, # 深层边界: hook on layer 10 output
}
CALIBRATION_SIZE = 50  # 子集，控制时间
OUTPUT_DIR = os.path.dirname(__file__)


def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
    model.eval()
    return tokenizer, model


def get_baseline_logits(model, tokenizer, text, max_length=128):
    """获取基线logits和hidden states"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = inputs['input_ids']
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 最后一个token的logits (通过lm_head)
    # GPT2Model没有lm_head，需要手动加
    # hidden_state: (batch, seq, hidden)
    last_hidden = outputs.last_hidden_state[:, -1, :]  # (batch, hidden)
    
    return {
        'input_ids': input_ids,
        'last_hidden': last_hidden,
        'hidden_all': outputs.last_hidden_state,
        'attentions': outputs.attentions,
    }


def inject_noise_and_measure(model, tokenizer, text, hook_layer_idx, snr, max_length=128):
    """
    在指定层输出注入高斯噪声，测量最终输出扰动。
    
    hook_layer_idx: 在该层output处注入噪声
    snr: 信噪比 (信号功率/噪声功率)
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    input_ids = inputs['input_ids']
    
    captured_output = [None]
    noise_added = [False]
    
    def hook_fn(module, input, output):
        """捕获layer output并在其上加噪声"""
        # output is a single Tensor: (batch, seq, hidden)
        hidden_state = output
        
        # 计算信号功率
        signal_power = torch.mean(hidden_state ** 2).item()
        
        # 计算噪声标准差: SNR = signal_power / noise_power → noise_std = sqrt(signal_power / snr)
        noise_std = np.sqrt(signal_power / snr) if snr > 0 else 0
        
        # 加高斯噪声
        noise = torch.randn_like(hidden_state) * noise_std
        corrupted = hidden_state + noise
        
        captured_output[0] = corrupted
        noise_added[0] = True
        
        return corrupted
    
    # 注册hook
    target_layer = model.h[hook_layer_idx]
    hook = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    hook.remove()
    
    # 最终输出
    last_hidden = outputs.last_hidden_state[:, -1, :]
    
    return {
        'last_hidden': last_hidden,
        'noise_added': noise_added[0],
    }


def measure_perturbation(baseline_hidden, perturbed_hidden):
    """测量两个hidden state之间的扰动"""
    b = baseline_hidden.squeeze(0)
    p = perturbed_hidden.squeeze(0)
    
    # 余弦距离
    cos_sim = F.cosine_similarity(b.unsqueeze(0), p.unsqueeze(0)).item()
    cos_dist = 1 - cos_sim
    
    # L2距离 (归一化)
    l2_dist = torch.norm(b - p).item()
    l2_rel = l2_dist / torch.norm(b).item()
    
    # 逐元素相关性
    corr = torch.corrcoef(torch.stack([b, p]))[0, 1].item()
    
    return {
        'cosine_distance': cos_dist,
        'cosine_similarity': cos_sim,
        'l2_distance': l2_dist,
        'l2_relative': l2_rel,
        'correlation': corr,
    }


def main():
    print("=" * 70)
    print("三体串联验证 — L10守恒核假说")
    print("=" * 70)
    
    tokenizer, model = load_model()
    calibration = get_calibration_set()[:CALIBRATION_SIZE]
    print(f"模型: GPT-2 (12层)")
    print(f"校准集: {len(calibration)}条")
    print(f"注入点: {list(INJECTION_SITES.keys())}")
    print(f"噪声SNR: {NOISE_SNR_VALUES}")
    print()
    
    results = {site: {snr: [] for snr in NOISE_SNR_VALUES} for site in INJECTION_SITES}
    
    for idx, text in enumerate(calibration):
        if idx % 10 == 0:
            print(f"  样本 {idx}/{len(calibration)}...")
        
        # 基线
        baseline = get_baseline_logits(model, tokenizer, text)
        baseline_hidden = baseline['last_hidden']
        
        # 对每个注入点
        for site_name, hook_layer_idx in INJECTION_SITES.items():
            for snr in NOISE_SNR_VALUES:
                try:
                    perturbed = inject_noise_and_measure(
                        model, tokenizer, text, hook_layer_idx, snr
                    )
                    
                    if perturbed['noise_added']:
                        metrics = measure_perturbation(
                            baseline_hidden, perturbed['last_hidden']
                        )
                        results[site_name][snr].append(metrics)
                except Exception as e:
                    pass
    
    # ═══════════════════════════════════════════════════════════════
    # 输出结果
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("实验结果: 输出扰动 vs 噪声强度")
    print("=" * 70)
    
    for site_name in INJECTION_SITES:
        print(f"\n{'─' * 60}")
        print(f"注入点: {site_name}")
        print(f"{'─' * 60}")
        print(f"{'SNR':>6} | {'cos_dist':>9} {'cos_sim':>9} | {'l2_rel':>8} {'corr':>8} | {'n':>3}")
        print("-" * 60)
        
        for snr in NOISE_SNR_VALUES:
            data = results[site_name][snr]
            if not data:
                print(f"{snr:6.1f} | {'N/A':>9} {'N/A':>9} | {'N/A':>8} {'N/A':>8} |   0")
                continue
            
            cos_d = np.mean([d['cosine_distance'] for d in data])
            cos_s = np.mean([d['cosine_similarity'] for d in data])
            l2_r = np.mean([d['l2_relative'] for d in data])
            corr = np.mean([d['correlation'] for d in data])
            n = len(data)
            
            print(f"{snr:6.1f} | {cos_d:9.4f} {cos_s:9.4f} | {l2_r:8.4f} {corr:8.4f} | {n:3d}")
    
    # ═══════════════════════════════════════════════════════════════
    # 核心比较: L0→L1 vs L10→L11
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("核心比较: 浅层(L0→L1) vs 深层(L10→L11)")
    print("=" * 70)
    
    print(f"\n{'SNR':>6} | {'cos_d(L0→L1)':>13} {'cos_d(L10→L11)':>15} | {'比值':>8} | {'判定':>8}")
    print("-" * 70)
    
    verdicts = []
    
    for snr in NOISE_SNR_VALUES:
        d0 = results['L0→L1'][snr]
        d10 = results['L10→L11'][snr]
        
        if not d0 or not d10:
            continue
        
        cos_0 = np.mean([d['cosine_distance'] for d in d0])
        cos_10 = np.mean([d['cosine_distance'] for d in d10])
        
        ratio = cos_10 / cos_0 if cos_0 > 0 else float('inf')
        
        if ratio > 2.0:
            verdict = "L10崩溃"
        elif ratio > 1.3:
            verdict = "L10更敏"
        elif ratio < 0.7:
            verdict = "L0更敏"
        elif ratio < 0.5:
            verdict = "L0崩溃"
        else:
            verdict = "相当"
        
        print(f"{snr:6.1f} | {cos_0:13.4f} {cos_10:15.4f} | {ratio:8.2f}x | {verdict:>8}")
        
        verdicts.append({'snr': snr, 'ratio': ratio, 'verdict': verdict})
    
    # ═══════════════════════════════════════════════════════════════
    # 临界点分析: SNR阈值在哪里崩塌？
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 70)
    print("临界点分析: 扰动突变阈值")
    print("=" * 70)
    
    for site_name in INJECTION_SITES:
        print(f"\n{site_name}:")
        prev_cos_d = None
        for snr in NOISE_SNR_VALUES:
            data = results[site_name][snr]
            if not data:
                continue
            cos_d = np.mean([d['cosine_distance'] for d in data])
            
            if prev_cos_d is not None and prev_cos_d > 0:
                jump = cos_d / prev_cos_d
                marker = " ← 突变" if jump > 2.0 else ""
                print(f"  SNR={snr:5.1f}: cos_dist={cos_d:.4f} (×{jump:.2f}){marker}")
            else:
                print(f"  SNR={snr:5.1f}: cos_dist={cos_d:.4f}")
            
            prev_cos_d = cos_d
    
    # ═══════════════════════════════════════════════════════════════
    # 保存
    # ═══════════════════════════════════════════════════════════════
    
    output = {
        'config': {
            'snr_values': NOISE_SNR_VALUES,
            'injection_sites': INJECTION_SITES,
            'calibration_size': CALIBRATION_SIZE,
        },
        'results': results,
        'verdicts': verdicts,
    }
    
    # 转换numpy类型
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
    
    out_path = os.path.join(OUTPUT_DIR, 'cascade_experiment_results.json')
    with open(out_path, 'w') as f:
        json.dump(to_serializable(output), f, indent=2)
    
    print(f"\n结果已保存: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
