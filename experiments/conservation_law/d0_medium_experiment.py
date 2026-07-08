"""
GPT-2-medium D₀ Scaling Experiment
d_model=1024, n_layers=24, n_heads=16
"""
import torch, numpy as np, time, sys, random, json
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
from sklearn.decomposition import PCA

OUT = "/tmp/d0_medium_results.json"
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
        pca = PCA(n_components=min(256, min(h_norm.shape)-1))
        pca.fit(h_norm)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        d0_dict[li] = int(np.searchsorted(cumvar, 0.95) + 1)
    return d0_dict

# Load GPT-2-medium
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
config = GPT2Config.from_pretrained('gpt2-medium')
print(f"GPT-2-medium: d_model={config.n_embd}, n_layers={config.n_layer}, n_heads={config.n_head}")
sys.stdout.flush()

model_r = GPT2Model(config); model_r.eval()
model_t = GPT2Model.from_pretrained('gpt2-medium'); model_t.eval()

# Use N=5000 (proven to converge)
n = 5000
with open(OUT+".progress", 'w') as f:
    f.write(f"GPT-2-medium: d_model={config.n_embd}, n_layers={config.n_layer}\n")
    f.write(f"Prediction: D0≈167, ΣD0≈4008, D0/d_model≈16.3%\n\n")

texts = gen_texts(n)

t0 = time.time()
dr = measure_d0(texts, model_r, tokenizer, f"R-med")
tr = time.time() - t0
with open(OUT+".progress", 'a') as f:
    f.write(f"Random done in {tr:.0f}s\n")

t0 = time.time()
dt = measure_d0(texts, model_t, tokenizer, f"T-med")
tt = time.time() - t0
with open(OUT+".progress", 'a') as f:
    f.write(f"Trained done in {tt:.0f}s\n")

# Results
import numpy as np
rv = [dr[L] for L in sorted(dr.keys()) if 0 < L < len(dr)-1]
tv = [dt[L] for L in sorted(dt.keys()) if 0 < L < len(dt)-1]

results = {
    'model': 'gpt2-medium',
    'd_model': config.n_embd,
    'n_layers': config.n_layer,
    'n_heads': config.n_head,
    'random': {str(L): dr[L] for L in sorted(dr.keys())},
    'trained': {str(L): dt[L] for L in sorted(dt.keys())},
    'random_mean_internal': float(np.mean(rv)),
    'trained_mean_internal': float(np.mean(tv)),
    'random_cv': float(np.std(rv)/np.mean(rv)),
    'trained_cv': float(np.std(tv)/np.mean(tv)),
    'sum_random': sum(dr.values()),
    'sum_trained': sum(dt.values()),
    'time_random': tr,
    'time_trained': tt,
}

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
with open(OUT+".progress", 'a') as f:
    f.write(f"\n=== RESULTS ===\n")
    f.write(f"{'L':>3} {'Rand':>6} {'Train':>6} {'Δ':>6}\n")
    for L in sorted(dr.keys()):
        d = dt[L] - dr[L]
        m = ' ***' if abs(d) > 30 else ''
        f.write(f"L{L:>2} {dr[L]:>6} {dt[L]:>6} {d:>+6}{m}\n")
    f.write(f"\nRandom  internal: {np.mean(rv):.1f}+/-{np.std(rv):.1f} CV={np.std(rv)/np.mean(rv):.4f}\n")
    f.write(f"Trained internal: {np.mean(tv):.1f}+/-{np.std(tv):.1f} CV={np.std(tv)/np.mean(tv):.4f}\n")
    f.write(f"ΣR={sum(dr.values())} ΣT={sum(dt.values())}\n")
    f.write(f"D0/d_model: R={np.mean(rv)/config.n_embd:.4f} T={np.mean(tv)/config.n_embd:.4f}\n")
    f.write(f"Prediction: D0≈167, actual R={np.mean(rv):.1f} T={np.mean(tv):.1f}\n")
    f.write("DONE\n")

print("Done. Results in", OUT)
