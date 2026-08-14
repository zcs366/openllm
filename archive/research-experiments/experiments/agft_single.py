#!/usr/bin/env python3
"""单规模AGFT实验 - n=5/100/500/1000/5000/10000"""
import gc, torch, json, time, sys
from pathlib import Path

MODEL_PATH = '/mnt/i/hermes/models/Qwen2.5-7B-Instruct'
DATA_DIR = Path('data/cross_lingual')
OUTPUT_DIR = Path('output/agft_experiment')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
print(f'数据规模: n={n}')

# 加载数据
with open(DATA_DIR / f'eval_{n}.json') as f:
    eval_data = json.load(f)[:20]
eval_en = [d['en'] for d in eval_data]
eval_zh = [d['zh'] for d in eval_data]

with open(DATA_DIR / f'train_{n}.json') as f:
    train_data = json.load(f)[:n]
print(f'训练集: {len(train_data)}, 评估集: {len(eval_data)}')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

def load():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
                                                device_map='auto', trust_remote_code=True)
    return mdl, tok

def infer(mdl, tok, text):
    prompt = f'Translate English to Chinese. Output ONLY the Chinese translation.\n\nEnglish: {text}\nChinese:'
    inp = tok(prompt, return_tensors='pt', max_length=256, truncation=True)
    inp = {k: v.to(mdl.device) for k, v in inp.items()}
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=60, do_sample=False)
    return tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip().split('\n')[0][:50]

def eval_bleu(preds, refs):
    return sum(1 for p, r in zip(preds, refs) if p[:10] in r or r[:10] in p) / len(preds) if preds else 0

results = []

# ── AGFT (冻结16层) ──
print('\n=== AGFT(16层) ===')
t0 = time.time()
m, t = load()
for i, layer in enumerate(m.model.layers):
    if i < 16:
        for p in layer.parameters():
            p.requires_grad = False
for p in m.model.embed_tokens.parameters():
    p.requires_grad = False
tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'可训练: {tp/1e9:.2f}B')

base_preds = [infer(m, t, en) for en in eval_en[:10]]
base_bleu = eval_bleu(base_preds, eval_zh[:10])
print(f'基线BLEU: {base_bleu:.4f}')

m.train()
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=2e-5)
losses = []
for ep in range(2):
    ep_loss = 0
    for i, item in enumerate(train_data):
        full = f'Translate English to Chinese:\n{item["en"]}\nChinese: {item["zh"]}'
        inp = t(full, return_tensors='pt', max_length=256, truncation=True)
        inp = {k: v.to(m.device) for k, v in inp.items()}
        inp['labels'] = inp['input_ids'].clone()
        loss = m(**inp).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        ep_loss += loss.item()
    avg = ep_loss / len(train_data)
    losses.append(avg)
    print(f'  Epoch {ep+1}: loss={avg:.4f}')

m.eval()
ft_preds = [infer(m, t, en) for en in eval_en[:10]]
ft_bleu = eval_bleu(ft_preds, eval_zh[:10])
print(f'微调后BLEU: {ft_bleu:.4f}')
results.append({'config': 'AGFT(16层)', 'base_bleu': base_bleu, 'ft_bleu': ft_bleu, 'losses': losses, 'time': time.time()-t0})
del m; gc.collect(); torch.cuda.empty_cache()

# ── LoRA ──
print('\n=== LoRA(r=8) ===')
t0 = time.time()
m, t = load()
m = get_peft_model(m, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.0,
                                  target_modules=['q_proj','k_proj','v_proj','o_proj']))
tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'可训练: {tp/1e6:.1f}M')

base_preds = [infer(m, t, en) for en in eval_en[:10]]
base_bleu = eval_bleu(base_preds, eval_zh[:10])
print(f'基线BLEU: {base_bleu:.4f}')

m.train()
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=2e-5)
losses = []
for ep in range(3):
    ep_loss = 0
    for i, item in enumerate(train_data):
        full = f'Translate English to Chinese:\n{item["en"]}\nChinese: {item["zh"]}'
        inp = t(full, return_tensors='pt', max_length=256, truncation=True)
        inp = {k: v.to(m.device) for k, v in inp.items()}
        inp['labels'] = inp['input_ids'].clone()
        loss = m(**inp).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        ep_loss += loss.item()
    avg = ep_loss / len(train_data)
    losses.append(avg)
    print(f'  Epoch {ep+1}: loss={avg:.4f}')

m.eval()
ft_preds = [infer(m, t, en) for en in eval_en[:10]]
ft_bleu = eval_bleu(ft_preds, eval_zh[:10])
print(f'微调后BLEU: {ft_bleu:.4f}')
results.append({'config': 'LoRA(r=8)', 'base_bleu': base_bleu, 'ft_bleu': ft_bleu, 'losses': losses, 'time': time.time()-t0})
del m; gc.collect(); torch.cuda.empty_cache()

# 保存
out = {'n': n, 'train_size': len(train_data), 'eval_size': len(eval_data), 'configs': results}
with open(OUTPUT_DIR / f'scaling_{n}.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f'\n{"="*60}')
print(f'n={n} 完成')
for r in results:
    print(f'  {r["config"]}: 基线={r["base_bleu"]:.4f}, 微调后={r["ft_bleu"]:.4f}, 改进={r["ft_bleu"]-r["base_bleu"]:.4f}, 耗时={r["time"]:.0f}s')
