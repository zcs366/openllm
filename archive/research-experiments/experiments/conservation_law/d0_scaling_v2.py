"""
D₀ Scaling: N=100, 1000, 5000, 10000
Output → /tmp/d0_scaling_results.json
"""
import torch, numpy as np, time, sys, random, json
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from sklearn.decomposition import PCA

OUT = "/tmp/d0_scaling_results.json"
random.seed(42)

templates = [
    'The {a} {n} {v} over the {a2} {n2} near the {p}.',
    'Scientists discovered that {n} can {v} when exposed to {n2}.',
    'The {a} revolution in {f} is transforming how we {v}.',
    'According to recent research, {n} {v} more efficiently than {n2}.',
    'In the {a} world of {f}, {n} remains the most {a2} {n2}.',
    'The government announced new policies to {v} the growing {n} crisis.',
    'Many experts believe that {f} will {v} completely within {num} years.',
    'The {a} {n} was found to {v} under extreme {n2} conditions.',
    'A new study reveals that {n} and {n2} are closely related through {f}.',
    'The future of {f} depends on our ability to {v} {a} {n}.',
]
adjs = ['rapid','complex','emerging','innovative','fundamental','remarkable','significant','unprecedented','critical','advanced']
nouns = ['technology','algorithm','neuron','genome','ecosystem','quantum','entropy','catalyst','membrane','frequency']
verbs = ['transforms','accelerates','generates','optimizes','disrupts','evolves','synthesizes','catalyzes','propagates','fluctuates']
nouns2 = ['data','energy','light','matter','signal','pattern','wave','field','force','flow']
places = ['laboratory','observatory','university','workstation','center','station','facility','platform']
fields = ['artificial intelligence','quantum computing','biotechnology','nanotechnology','astrophysics','neuroscience','materials science','climate science']
numbers = ['five','ten','twenty','fifty','a hundred']

def gen_texts(n):
    texts = []
    for i in range(n):
        t = random.choice(templates)
        for k,v in [('{a}',random.choice(adjs)),('{n}',random.choice(nouns)),('{v}',random.choice(verbs)),
                    ('{n2}',random.choice(nouns2)),('{p}',random.choice(places)),('{f}',random.choice(fields)),
                    ('{num}',random.choice(numbers)),('{a2}',random.choice(adjs))]:
            t = t.replace(k,v)
        texts.append(t)
    return texts

def measure_d0(texts, model, tokenizer, label=""):
    all_hidden = {}
    for i, t in enumerate(texts):
        inp = tokenizer(t, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        for li, h in enumerate(out.hidden_states):
            if li not in all_hidden:
                all_hidden[li] = []
            all_hidden[li].append(h[0].numpy())
        if (i+1) % 1000 == 0:
            with open(OUT+".progress", 'a') as f:
                f.write(f"  {label}: {i+1}/{len(texts)}\n")

    d0_dict = {}
    for li in sorted(all_hidden.keys()):
        all_h = np.vstack(all_hidden[li])
        norms = np.linalg.norm(all_h, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        h_norm = all_h / norms
        pca = PCA(n_components=min(200, min(h_norm.shape)-1))
        pca.fit(h_norm)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        d0_dict[li] = int(np.searchsorted(cumvar, 0.95) + 1)
    return d0_dict

# Main
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config()
model_r = GPT2Model(config); model_r.eval()
model_t = GPT2Model.from_pretrained('gpt2'); model_t.eval()

results = {}
for n in [100, 1000, 5000, 10000]:
    # Clear progress
    with open(OUT+".progress", 'a') as f:
        f.write(f"\n=== N={n} started ===\n")

    texts = gen_texts(n)

    t0 = time.time()
    dr = measure_d0(texts, model_r, tokenizer, f"R-{n}")
    tr = time.time() - t0

    t0 = time.time()
    dt = measure_d0(texts, model_t, tokenizer, f"T-{n}")
    tt = time.time() - t0

    rv = [dr[L] for L in sorted(dr.keys()) if 0 < L < 12]
    tv = [dt[L] for L in sorted(dt.keys()) if 0 < L < 12]

    results[str(n)] = {
        'random': {str(L): dr[L] for L in sorted(dr.keys())},
        'trained': {str(L): dt[L] for L in sorted(dt.keys())},
        'random_mean': float(np.mean(rv)),
        'trained_mean': float(np.mean(tv)),
        'random_cv': float(np.std(rv)/np.mean(rv)),
        'trained_cv': float(np.std(tv)/np.mean(tv)),
        'time_random': tr,
        'time_trained': tt,
    }

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)

    with open(OUT+".progress", 'a') as f:
        f.write(f"  N={n} done: R={np.mean(rv):.1f} T={np.mean(tv):.1f} tR={tr:.0f}s tT={tt:.0f}s\n")

# Final summary
with open(OUT+".progress", 'a') as f:
    f.write("\n=== SUMMARY ===\n")
    for n in [100, 1000, 5000, 10000]:
        r = results[str(n)]
        f.write(f"  N={n:>5}: R={r['random_mean']:.1f}(CV={r['random_cv']:.4f}) T={r['trained_mean']:.1f}(CV={r['trained_cv']:.4f}) Δ={r['trained_mean']-r['random_mean']:+.1f}\n")
    f.write("DONE\n")
