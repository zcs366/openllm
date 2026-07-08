"""
P0: 跨架构归一化D₀验证

测量各架构每层的归一化D₀（PCA on normalized attention matrices）
- BERT-base (encoder)
- GPT-2 (decoder)
- Llama-3.2-1B (decoder)
- Qwen2.5-3B (decoder)

D₀(norm) = 保持95%方差的最小PCA维度（归一化注意力矩阵）
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    GPT2Tokenizer, GPT2Model,
    BertTokenizer, BertModel,
    LlamaTokenizer, LlamaModel,
    Qwen2Tokenizer, Qwen2Model,
)
from sklearn.decomposition import PCA
import json
import os
import gc
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.dirname(__file__)

# 校准集（复用）
from calibration_set import get_calibration_set

CALIBRATION_SUBSET = 30  # 每模型30条
VARIANCE_THRESHOLD = 0.95  # 95%方差 → D₀


def measure_d0_norm(attention_matrix, threshold=0.95):
    """
    归一化注意力矩阵后测D₀。
    attention_matrix: (seq_len, seq_len) 或 (seq_len, hidden_dim)
    
    D₀ = 保持threshold方差的最小PCA维度
    """
    # 行归一化（每行变成概率分布）
    row_sums = np.sum(attention_matrix, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    normalized = attention_matrix / row_sums
    
    # 去掉全零行
    non_zero_rows = np.any(normalized > 0, axis=1)
    if np.sum(non_zero_rows) < 10:
        return 0, 0, []
    
    normalized = normalized[non_zero_rows]
    
    # PCA
    n_components = min(normalized.shape[0], normalized.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(normalized)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    # 找D₀: 第一个达到threshold的维度
    d0 = np.searchsorted(cumvar, threshold) + 1
    d0 = min(d0, n_components)
    
    # 有效维度（>1%方差的成分数）
    effective_dims = np.sum(pca.explained_variance_ratio_ > 0.01)
    
    return d0, effective_dims, cumvar.tolist()


def extract_attention_gpt2(model, tokenizer, text, max_length=128):
    """GPT-2: 提取各层注意力矩阵"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    
    matrices = {}
    if outputs.attentions is not None:
        for i, attn in enumerate(outputs.attentions):
            # attn: (batch, heads, seq, seq) → avg heads → (seq, seq)
            matrices[i] = attn[0].mean(dim=0).cpu().numpy()
    return matrices


def extract_attention_bert(model, tokenizer, text, max_length=128):
    """BERT: 提取各层注意力矩阵"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    matrices = {}
    if outputs.attentions is not None:
        for i, attn in enumerate(outputs.attentions):
            matrices[i] = attn[0].mean(dim=0).cpu().numpy()
    return matrices


def extract_attention_decoder(model, tokenizer, text, max_length=128):
    """通用decoder (Llama/Qwen): 提取各层注意力矩阵"""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    matrices = {}
    if outputs.attentions is not None:
        for i, attn in enumerate(outputs.attentions):
            matrices[i] = attn[0].mean(dim=0).cpu().numpy()
    return matrices


def run_model_experiment(model_name, load_fn, extract_fn, tokenizer_cls, model_cls, model_kwargs):
    """对单个模型运行D₀测量"""
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"{'='*60}")
    
    # 加载
    tokenizer, model = load_fn(tokenizer_cls, model_cls, model_kwargs)
    config = model.config
    num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 0))
    hidden = getattr(config, 'hidden_size', getattr(config, 'n_embd', 0))
    print(f"  层数: {num_layers}, 隐藏维度: {hidden}")
    
    calibration = get_calibration_set()[:CALIBRATION_SUBSET]
    
    # 每层收集D₀值
    layer_d0 = {i: [] for i in range(num_layers)}
    layer_eff = {i: [] for i in range(num_layers)}
    
    for idx, text in enumerate(calibration):
        if idx % 10 == 0:
            print(f"  样本 {idx}/{len(calibration)}...")
        
        try:
            matrices = extract_fn(model, tokenizer, text)
            
            for layer_idx, attn in matrices.items():
                if layer_idx >= num_layers:
                    continue
                d0, eff, _ = measure_d0_norm(attn)
                if d0 > 0:
                    layer_d0[layer_idx].append(d0)
                    layer_eff[layer_idx].append(eff)
        except Exception as e:
            pass
    
    # 汇总
    results = {}
    print(f"\n  {'Layer':>6} | {'D₀(mean)':>9} {'D₀(std)':>8} {'D₀(CV)':>8} | {'EffD':>6} | {'n':>3}")
    print(f"  " + "-" * 55)
    
    all_d0 = []
    for layer_idx in range(num_layers):
        d0_vals = layer_d0[layer_idx]
        eff_vals = layer_eff[layer_idx]
        
        if len(d0_vals) < 3:
            results[layer_idx] = {'d0_mean': None, 'd0_std': None, 'n': len(d0_vals)}
            print(f"  L{layer_idx:4d} | {'N/A':>9} {'N/A':>8} {'N/A':>8} | {'N/A':>6} | {len(d0_vals):3d}")
            continue
        
        d0_arr = np.array(d0_vals)
        eff_arr = np.array(eff_vals)
        
        mean_d0 = np.mean(d0_arr)
        std_d0 = np.std(d0_arr)
        cv_d0 = std_d0 / mean_d0 if mean_d0 > 0 else 0
        mean_eff = np.mean(eff_arr)
        
        results[layer_idx] = {
            'd0_mean': float(mean_d0),
            'd0_std': float(std_d0),
            'd0_cv': float(cv_d0),
            'eff_dim': float(mean_eff),
            'n': len(d0_vals),
            'd0_values': [int(v) for v in d0_vals],
        }
        
        all_d0.extend(d0_vals)
        
        print(f"  L{layer_idx:4d} | {mean_d0:9.2f} {std_d0:8.2f} {cv_d0:8.4f} | {mean_eff:6.1f} | {len(d0_vals):3d}")
    
    # 整体统计
    if all_d0:
        all_arr = np.array(all_d0)
        overall = {
            'mean': float(np.mean(all_arr)),
            'std': float(np.std(all_arr)),
            'cv': float(np.std(all_arr) / np.mean(all_arr)) if np.mean(all_arr) > 0 else 0,
            'min': int(np.min(all_arr)),
            'max': int(np.max(all_arr)),
            'median': float(np.median(all_arr)),
        }
        print(f"\n  整体 D₀(norm): {overall['mean']:.2f} ± {overall['std']:.2f} (CV={overall['cv']:.4f})")
        print(f"  范围: [{overall['min']}, {overall['max']}], 中位数: {overall['median']:.1f}")
    else:
        overall = None
    
    # 释放显存
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {'layers': results, 'overall': overall}


# ── 加载函数 ──

def load_gpt2(tok_cls, model_cls, kwargs):
    tok = tok_cls.from_pretrained('gpt2')
    model = model_cls.from_pretrained('gpt2', output_attentions=True)
    model.eval()
    return tok, model

def load_bert(tok_cls, model_cls, kwargs):
    tok = tok_cls.from_pretrained('bert-base-uncased')
    model = model_cls.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()
    return tok, model

def load_llama(tok_cls, model_cls, kwargs):
    tok = tok_cls.from_pretrained('meta-llama/Llama-3.2-1B')
    model = model_cls.from_pretrained('meta-llama/Llama-3.2-1B', output_attentions=True)
    model.eval()
    return tok, model

def load_qwen(tok_cls, model_cls, kwargs):
    tok = tok_cls.from_pretrained('Qwen/Qwen2.5-3B')
    model = model_cls.from_pretrained('Qwen/Qwen2.5-3B', output_attentions=True)
    model.eval()
    return tok, model


# ── 主实验 ──

def main():
    print("=" * 70)
    print("P0: 跨架构归一化D₀验证")
    print("=" * 70)
    print(f"校准集: {CALIBRATION_SUBSET}条")
    print(f"方差阈值: {VARIANCE_THRESHOLD}")
    
    experiments = [
        ("GPT-2 (decoder)", load_gpt2, extract_attention_gpt2, GPT2Tokenizer, GPT2Model, {}),
        ("BERT-base (encoder)", load_bert, extract_attention_bert, BertTokenizer, BertModel, {}),
        ("Llama-3.2-1B (decoder)", load_llama, extract_attention_decoder, LlamaTokenizer, LlamaModel, {}),
        ("Qwen2.5-3B (decoder)", load_qwen, extract_attention_decoder, Qwen2Tokenizer, Qwen2Model, {}),
    ]
    
    all_results = {}
    
    for name, load_fn, extract_fn, tok_cls, model_cls, kwargs in experiments:
        try:
            result = run_model_experiment(name, load_fn, extract_fn, tok_cls, model_cls, kwargs)
            all_results[name] = result
        except Exception as e:
            print(f"\n  ❌ {name} 失败: {e}")
            all_results[name] = {'error': str(e)}
    
    # ═══════════════════════════════════════════════════════════════
    # 跨架构汇总比较
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("跨架构D₀(norm)汇总比较")
    print("=" * 70)
    
    print(f"\n{'模型':>25} | {'架构':>8} | {'D₀(mean)':>9} {'±std':>7} {'CV':>7} | {'层数':>4}")
    print("-" * 75)
    
    comparison = []
    
    for name, result in all_results.items():
        if 'error' in result:
            print(f"{name:>25} | {'?':>8} | {'ERROR':>9}")
            continue
        
        ov = result.get('overall')
        if ov is None:
            continue
        
        arch = "encoder" if "BERT" in name or "bert" in name else "decoder"
        n_layers = len([v for v in result['layers'].values() if v.get('d0_mean') is not None])
        
        print(f"{name:>25} | {arch:>8} | {ov['mean']:9.2f} ±{ov['std']:6.2f} {ov['cv']:7.4f} | {n_layers:4d}")
        
        comparison.append({
            'model': name,
            'architecture': arch,
            'd0_mean': ov['mean'],
            'd0_std': ov['std'],
            'd0_cv': ov['cv'],
            'layers': n_layers,
        })
    
    # 验证预测
    print("\n" + "=" * 70)
    print("预测验证")
    print("=" * 70)
    
    predictions = {
        'GPT-2 (decoder)': (100, 110),
        'BERT-base (encoder)': (10, 16),
        'Llama-3.2-1B (decoder)': (100, 110),
        'Qwen2.5-3B (decoder)': (100, 110),
    }
    
    for item in comparison:
        name = item['model']
        if name in predictions:
            lo, hi = predictions[name]
            actual = item['d0_mean']
            hit = lo <= actual <= hi
            status = "✅" if hit else "❌"
            print(f"  {status} {name}: 预测[{lo},{hi}], 实际={actual:.2f}")
    
    # 保存
    output = {
        'config': {
            'calibration_size': CALIBRATION_SUBSET,
            'variance_threshold': VARIANCE_THRESHOLD,
        },
        'comparison': comparison,
        'details': all_results,
    }
    
    out_path = os.path.join(OUTPUT_DIR, 'p0_cross_architecture_d0.json')
    
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
    
    with open(out_path, 'w') as f:
        json.dump(to_serializable(output), f, indent=2)
    
    print(f"\n结果已保存: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
