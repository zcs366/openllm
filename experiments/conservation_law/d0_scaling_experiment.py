"""
D₀ Scaling Experiment: Random vs Trained
测试 D₀(normalized) 随校准集大小的收敛行为
"""
import torch, numpy as np, sys, time
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from sklearn.decomposition import PCA

# ── 校准文本生成 ──────────────────────────────────
def generate_texts(n):
    """生成n条多样化文本"""
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
    adjs = ['rapid', 'complex', 'emerging', 'innovative', 'fundamental', 'remarkable', 'significant', 'unprecedented', 'critical', 'advanced']
    nouns = ['technology', 'algorithm', 'neuron', 'genome', 'ecosystem', 'quantum', 'entropy', 'catalyst', 'membrane', 'frequency']
    verbs = ['transforms', 'accelerates', 'generates', 'optimizes', 'disrupts', 'evolves', 'synthesizes', 'catalyzes', 'propagates', 'fluctuates']
    nouns2 = ['data', 'energy', 'light', 'matter', 'signal', 'pattern', 'wave', 'field', 'force', 'flow']
    places = ['laboratory', 'observatory', 'university', 'workstation', '数据中心', '深海', '太空站', '量子计算机']
    fields = ['artificial intelligence', 'quantum computing', 'biotechnology', 'nanotechnology', 'astrophysics', 'neuroscience', 'materials science', 'climate science']
    numbers = ['five', 'ten', 'twenty', 'fifty', 'a hundred']

    import random
    random.seed(42)
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

# ── D₀ 测量 ──────────────────────────────────────
def measure_d0_batch(texts, model, tokenizer, max_length=128, batch_size=32):
    """批量测量归一化D₀"""
    num_layers = None
    all_hidden = {}  # {layer: list of numpy arrays}

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for t in batch:
            inp = tokenizer(t, return_tensors='pt', truncation=True, max_length=max_length)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            if num_layers is None:
                num_layers = len(out.hidden_states)
            for li, h in enumerate(out.hidden_states):
                if li not in all_hidden:
                    all_hidden[li] = []
                all_hidden[li].append(h[0].numpy())  # (seq_len, hidden_dim)

        if (i + batch_size) % 500 == 0 or i + batch_size >= len(texts):
            sys.stdout.write(f"\r  Processed {min(i+batch_size, len(texts))}/{len(texts)}")
            sys.stdout.flush()

    print()

    # Compute D0(normalized) for each layer
    d0_dict = {}
    for li in sorted(all_hidden.keys()):
        all_h = np.vstack(all_hidden[li])
        norms = np.linalg.norm(all_h, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        all_h_norm = all_h / norms

        max_comp = min(200, min(all_h_norm.shape) - 1)
        pca = PCA(n_components=max_comp)
        pca.fit(all_h_norm)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        d0 = int(np.searchsorted(cumvar, 0.95) + 1)
        d0_dict[li] = d0

    return d0_dict

# ── 主实验 ────────────────────────────────────────
def main():
    sizes = [100, 1000, 5000, 10000]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Random model
    config = GPT2Config()
    model_random = GPT2Model(config)
    model_random.eval()

    # Trained model
    model_trained = GPT2Model.from_pretrained('gpt2')
    model_trained.eval()

    results = {}

    for n in sizes:
        print(f"\n{'='*60}")
        print(f"  N = {n} texts")
        print(f"{'='*60}")

        texts = generate_texts(n)

        t0 = time.time()
        print(f"  Random model (step 0)...")
        d0_random = measure_d0_batch(texts, model_random, tokenizer)
        t_random = time.time() - t0

        t0 = time.time()
        print(f"  Trained model...")
        d0_trained = measure_d0_batch(texts, model_trained, tokenizer)
        t_trained = time.time() - t0

        results[n] = {'random': d0_random, 'trained': d0_trained}

        # Print comparison
        print(f"\n  {'Layer':>6} {'Random':>8} {'Trained':>8} {'Delta':>8}")
        print(f"  {'-'*35}")
        for L in sorted(d0_random.keys()):
            r = d0_random[L]
            t = d0_trained[L]
            delta = t - r
            marker = ' ***' if abs(delta) > 20 else ''
            print(f"  L{L:>4} {r:>8} {t:>8} {delta:>+8}{marker}")

        r_vals = [d0_random[L] for L in sorted(d0_random.keys()) if 0 < L < 12]
        t_vals = [d0_trained[L] for L in sorted(d0_trained.keys()) if 0 < L < 12]
        print(f"\n  Random  (L1-L11): mean={np.mean(r_vals):.1f}, std={np.std(r_vals):.1f}, CV={np.std(r_vals)/np.mean(r_vals):.4f}")
        print(f"  Trained (L1-L11): mean={np.mean(t_vals):.1f}, std={np.std(t_vals):.1f}, CV={np.std(t_vals)/np.mean(t_vals):.4f}")
        print(f"  Time: random={t_random:.1f}s, trained={t_trained:.1f}s")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"  SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"  {'N':>6} {'D0_rand':>10} {'D0_train':>10} {'Delta':>8} {'CV_rand':>8} {'CV_train':>8}")
    print(f"  {'-'*55}")
    for n in sizes:
        r_vals = [results[n]['random'][L] for L in sorted(results[n]['random'].keys()) if 0 < L < 12]
        t_vals = [results[n]['trained'][L] for L in sorted(results[n]['trained'].keys()) if 0 < L < 12]
        dr = np.mean(r_vals)
        dt = np.mean(t_vals)
        cvr = np.std(r_vals)/np.mean(r_vals)
        cvt = np.std(t_vals)/np.mean(t_vals)
        print(f"  {n:>6} {dr:>10.1f} {dt:>10.1f} {dt-dr:>+8.1f} {cvr:>8.4f} {cvt:>8.4f}")

if __name__ == '__main__':
    main()
