#!/usr/bin/env python3
"""AGFT实验 v3 - 4bit量化"""
import gc, torch, json
from pathlib import Path

MODEL_PATH = '/mnt/i/hermes/models/Qwen2.5-7B-Instruct'
DATA_DIR = Path('data/cross_lingual')
OUTPUT_DIR = Path('output/agft_experiment')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / 'eval_set.json') as f:
    eval_data = json.load(f)[:5]
with open(DATA_DIR / 'train_set.json') as f:
    train_data = json.load(f)[:10]

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

def load():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map='auto', trust_remote_code=True)
    return mdl, tok

def infer(mdl, tok, text):
    prompt = f'Translate English to Chinese. Output ONLY the Chinese translation.\n\nEnglish: {text}\nChinese:'
    inp = tok(prompt, return_tensors='pt', max_length=256, truncation=True)
    inp = {k: v.to(mdl.device) for k, v in inp.items()}
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=80, do_sample=False)
    return tok.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip().split('\n')[0].strip()

results = []

# 基线
print('=== 基线 ===')
m, t = load()
preds = [infer(m, t, d['en']) for d in eval_data]
for i, d in enumerate(eval_data):
    print(f'  EN: {d["en"][:40]}... → ZH: {preds[i][:40]}')
del m; gc.collect(); torch.cuda.empty_cache()

# AGFT冻结16层
print('\n=== AGFT(16层) ===')
m, t = load()
for i, layer in enumerate(m.model.layers):
    if i < 16:
        for p in layer.parameters():
            p.requires_grad = False
for p in m.model.embed_tokens.parameters():
    p.requires_grad = False
tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'可训练: {tp/1e9:.2f}B')
m.train()
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=2e-5)
for ep in range(2):
    for item in train_data:
        full = f'Translate English to Chinese:\n{item["en"]}\nChinese: {item["zh"]}'
        inp = t(full, return_tensors='pt', max_length=256, truncation=True)
        inp = {k: v.to(m.device) for k, v in inp.items()}
        inp['labels'] = inp['input_ids'].clone()
        loss = m(**inp).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f'  Epoch {ep+1}, Loss: {loss.item():.4f}')
m.eval()
preds_ft = [infer(m, t, d['en']) for d in eval_data]
for i, d in enumerate(eval_data):
    print(f'  EN: {d["en"][:40]}... → ZH: {preds_ft[i][:40]}')
results.append({'exp': 'AGFT(16层)', 'preds': preds_ft})
del m; gc.collect(); torch.cuda.empty_cache()

# LoRA
print('\n=== LoRA(r=8) ===')
m, t = load()
m = get_peft_model(m, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.0, target_modules=['q_proj','k_proj','v_proj','o_proj']))
tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f'可训练: {tp/1e6:.1f}M')
m.train()
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, m.parameters()), lr=2e-5)
for ep in range(3):
    for item in train_data:
        full = f'Translate English to Chinese:\n{item["en"]}\nChinese: {item["zh"]}'
        inp = t(full, return_tensors='pt', max_length=256, truncation=True)
        inp = {k: v.to(m.device) for k, v in inp.items()}
        inp['labels'] = inp['input_ids'].clone()
        loss = m(**inp).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f'  Epoch {ep+1}, Loss: {loss.item():.4f}')
m.eval()
preds_lora = [infer(m, t, d['en']) for d in eval_data]
for i, d in enumerate(eval_data):
    print(f'  EN: {d["en"][:40]}... → ZH: {preds_lora[i][:40]}')
results.append({'exp': 'LoRA(r=8)', 'preds': preds_lora})
del m; gc.collect(); torch.cuda.empty_cache()

with open(OUTPUT_DIR / 'agft_final_results.json', 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('\n✅ 全部实验完成')
