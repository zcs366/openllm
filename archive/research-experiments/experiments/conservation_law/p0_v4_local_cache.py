"""
P0 v4: 跨架构归一化D₀验证 (I盘本地缓存)

模型路径:
- GPT-2: gpt2 (HF缓存)
- BERT: /mnt/i/hermes/modelscope-cache/AI-ModelScope/bert-base-uncased/
- Llama: /mnt/i/.cache/modelscope/LLM-Research/Llama-3___2-1B/
- Qwen: 跳过(7B太大)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import json, os, gc, time
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.dirname(__file__)
CALIBRATION_SIZE = 100
MAX_LENGTH = 128
VARIANCE_THRESHOLD = 0.95

MODEL_PATHS = {
    "GPT-2 (decoder)": "gpt2",
    "BERT-base (encoder)": "/mnt/i/hermes/modelscope-cache/AI-ModelScope/bert-base-uncased",
    "Llama-3.2-1B (decoder)": "/mnt/i/.cache/modelscope/LLM-Research/Llama-3___2-1B",
}


def generate_calibration_texts(n=100):
    import random
    random.seed(42)
    templates = [
        "The {adj} {noun} {verb} over the {adj2} {noun2} near the {place}.",
        "Scientists discovered that {noun} can {verb} when exposed to {noun2}.",
        "The {adj} revolution in {field} is transforming how we {verb}.",
        "According to recent research, {noun} {verb} more efficiently than {noun2}.",
        "In the {adj} world of {field}, {noun} remains the most {adj2} {noun2}.",
        "The government announced new policies to {verb} the growing {noun} crisis.",
        "Many experts believe that {field} will {verb} completely within {number} years.",
        "The {adj} {noun} was found to {verb} under extreme {noun2} conditions.",
        "A new study reveals that {noun} and {noun2} are closely related through {field}.",
        "The future of {field} depends on our ability to {verb} {adj} {noun}.",
    ]
    adjs = ['rapid', 'complex', 'emerging', 'innovative', 'fundamental', 'remarkable', 
            'significant', 'unprecedented', 'critical', 'advanced']
    nouns = ['technology', 'algorithm', 'neuron', 'genome', 'ecosystem', 'quantum', 
             'entropy', 'catalyst', 'membrane', 'frequency']
    verbs = ['transforms', 'accelerates', 'generates', 'optimizes', 'disrupts', 
             'evolves', 'synthesizes', 'catalyzes', 'propagates', 'fluctuates']
    nouns2 = ['data', 'energy', 'light', 'matter', 'signal', 'pattern', 'wave', 
              'field', 'force', 'flow']
    places = ['laboratory', 'observatory', 'university', 'workstation', 
              '数据中心', '深海', '太空站', '量子计算机']
    fields = ['artificial intelligence', 'quantum computing', 'biotechnology', 
              'nanotechnology', 'astrophysics', 'neuroscience', 'materials science', 
              'climate science']
    numbers = ['five', 'ten', 'twenty', 'fifty', 'a hundred']
    
    texts = []
    for i in range(n):
        t = random.choice(templates)
        t = t.replace('{adj}', random.choice(adjs))
        t = t.replace('{noun}', random.choice(nouns))
        t = t.replace('{verb}', random.choice(verbs))
        t = t.replace('{noun2}', random.choice(nouns2))
        t = t.replace('{place}', random.choice(places))
        t = t.replace('{field}', random.choice(fields))
        t = t.replace('{number}', random.choice(numbers))
        t = t.replace('{adj2}', random.choice(adjs))
        texts.append(t)
    return texts


def measure_d0_hidden_states(model, tokenizer, texts, max_length=128):
    num_layers = None
    all_hidden = {}
    
    for idx, text in enumerate(texts):
        if idx % 20 == 0:
            print(f"    样本 {idx}/{len(texts)}...")
        
        inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
        
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        
        if num_layers is None:
            num_layers = len(out.hidden_states)
        
        for li, h in enumerate(out.hidden_states):
            if li not in all_hidden:
                all_hidden[li] = []
            h_np = h[0].cpu().float().numpy()
            all_hidden[li].append(h_np)
    
    results = {}
    for li in sorted(all_hidden.keys()):
        all_h = np.vstack(all_hidden[li])
        norms = np.linalg.norm(all_h, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        all_h_norm = all_h / norms
        
        max_comp = min(200, min(all_h_norm.shape) - 1)
        pca = PCA(n_components=max_comp)
        pca.fit(all_h_norm)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        d0 = int(np.searchsorted(cumvar, VARIANCE_THRESHOLD) + 1)
        eff_dims = int(np.sum(pca.explained_variance_ratio_ > 0.01))
        
        results[li] = {
            'd0': d0,
            'eff_dims': eff_dims,
            'total_tokens': all_h.shape[0],
            'hidden_dim': all_h.shape[1],
        }
    
    return results, num_layers


def run_experiment(name, model_path, texts):
    print(f"\n{'='*60}")
    print(f"模型: {name}")
    print(f"{'='*60}")
    
    try:
        print(f"  加载: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, output_hidden_states=True)
        model.eval()
        
        config = model.config
        num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 0))
        hidden = getattr(config, 'hidden_size', getattr(config, 'n_embd', 0))
        print(f"  层数: {num_layers}, 隐藏维度: {hidden}")
        
        t0 = time.time()
        results, actual_layers = measure_d0_hidden_states(model, tokenizer, texts, MAX_LENGTH)
        elapsed = time.time() - t0
        
        print(f"\n  {'Layer':>6} | {'D₀':>5} {'EffD':>5} | {'Tokens':>7} {'Hidden':>6}")
        print(f"  " + "-" * 40)
        
        d0_values = []
        for li in sorted(results.keys()):
            r = results[li]
            print(f"  L{li:4d} | {r['d0']:5d} {r['eff_dims']:5d} | {r['total_tokens']:7d} {r['hidden_dim']:6d}")
            if 0 < li < actual_layers:
                d0_values.append(r['d0'])
        
        if d0_values:
            arr = np.array(d0_values)
            mean_d0 = np.mean(arr)
            std_d0 = np.std(arr)
            cv = std_d0 / mean_d0 if mean_d0 > 0 else 0
            print(f"\n  D₀(norm) L1-L{actual_layers-1}: {mean_d0:.1f} ± {std_d0:.1f} (CV={cv:.4f})")
            print(f"  范围: [{int(np.min(arr))}, {int(np.max(arr))}], 中位数: {int(np.median(arr))}")
        else:
            mean_d0 = std_d0 = cv = 0
        
        print(f"  耗时: {elapsed:.1f}s")
        
        del model, tokenizer
        gc.collect()
        
        return {
            'name': name,
            'architecture': 'encoder' if 'bert' in name.lower() else 'decoder',
            'num_layers': num_layers,
            'hidden_dim': hidden,
            'd0_mean': float(mean_d0),
            'd0_std': float(std_d0),
            'd0_cv': float(cv),
            'd0_min': int(np.min(arr)) if len(arr) > 0 else 0,
            'd0_max': int(np.max(arr)) if len(arr) > 0 else 0,
            'layers': {str(k): v for k, v in results.items()},
            'elapsed': elapsed,
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'name': name, 'error': str(e)}


def main():
    print("=" * 70)
    print("P0 v4: 跨架构归一化D₀ (I盘本地缓存)")
    print("=" * 70)
    
    texts = generate_calibration_texts(CALIBRATION_SIZE)
    
    all_results = []
    for name, path in MODEL_PATHS.items():
        result = run_experiment(name, path, texts)
        all_results.append(result)
    
    # 汇总
    print("\n" + "=" * 70)
    print("跨架构D₀(norm)汇总")
    print("=" * 70)
    
    print(f"\n{'模型':>25} | {'架构':>8} | {'D₀(mean)':>9} {'±std':>7} {'CV':>7} | {'范围':>10}")
    print("-" * 78)
    
    for r in all_results:
        if 'error' in r:
            print(f"{r['name']:>25} | {'?':>8} | {'ERROR':>9}  ({r['error'][:50]})")
            continue
        print(f"{r['name']:>25} | {r['architecture']:>8} | {r['d0_mean']:9.1f} ±{r['d0_std']:6.1f} {r['d0_cv']:7.4f} | [{r['d0_min']},{r['d0_max']}]")
    
    # 预测验证
    print("\n预测验证:")
    predictions = {
        'GPT-2 (decoder)': (100, 150),
        'BERT-base (encoder)': (10, 25),
        'Llama-3.2-1B (decoder)': (100, 150),
    }
    
    for r in all_results:
        name = r['name']
        if 'error' in r or name not in predictions:
            continue
        lo, hi = predictions[name]
        actual = r['d0_mean']
        hit = lo <= actual <= hi
        status = "✅" if hit else "❌"
        print(f"  {status} {name}: 预测[{lo},{hi}], 实际={actual:.1f}")
    
    # 保存
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
    
    out_path = os.path.join(OUTPUT_DIR, 'p0_v4_cross_architecture_d0.json')
    with open(out_path, 'w') as f:
        json.dump(to_serializable({'config': {'calibration_size': CALIBRATION_SIZE}, 'results': all_results}), f, indent=2)
    
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
