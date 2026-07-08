#!/usr/bin/env python3
"""AGFT规模梯度实验 - 5/100/500/1000/5000/10000"""
import gc, torch, json, time, sys
from pathlib import Path

MODEL_PATH = '/mnt/i/hermes/models/Qwen2.5-7B-Instruct'
DATA_DIR = Path('data/cross_lingual')
OUTPUT_DIR = Path('output/agft_experiment')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

def load_model():
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

def train_loop(mdl, tok, data, lr=2e-5, epochs=2, batch_log=10):
    mdl.train()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, mdl.parameters()), lr=lr)
    losses = []
    for ep in range(epochs):
        ep_loss = 0
        for i, item in enumerate(data):
            full = f'Translate English to Chinese:\n{item["en"]}\nChinese: {item["zh"]}'
            inp = tok(full, return_tensors='pt', max_length=256, truncation=True)
            inp = {k: v.to(mdl.device) for k, v in inp.items()}
            inp['labels'] = inp['input_ids'].clone()
            loss = mdl(**inp).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            if i % batch_log == 0 and i > 0:
                print(f'    ep{ep+1} step{i}: loss={loss.item():.4f}')
        avg = ep_loss / len(data)
        losses.append(avg)
        print(f'  Epoch {ep+1}: avg_loss={avg:.4f}')
    return losses

def run_one_config(mdl, tok, train_data, eval_en, eval_zh, config_name, freeze_layers=0, use_lora=False):
    print(f'\n--- {config_name} ---')
    t0 = time.time()
    
    if use_lora:
        mdl = get_peft_model(mdl, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                                               lora_dropout=0.0, target_modules=['q_proj','k_proj','v_proj','o_proj']))
        tp = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
        print(f'LoRA: 可训练 {tp/1e6:.1f}M')
    elif freeze_layers > 0:
        for i, layer in enumerate(mdl.model.layers):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False
        for p in mdl.model.embed_tokens.parameters():
            p.requires_grad = False
        tp = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
        print(f'AGFT冻结{freeze_layers}层: 可训练 {tp/1e9:.2f}B')
    
    # 基线评估
    base_preds = [infer(mdl, tok, en) for en in eval_en[:10]]
    base_bleu = eval_bleu(base_preds, eval_zh[:10])
    print(f'  基线BLEU: {base_bleu:.4f}')
    
    # 微调
    losses = train_loop(mdl, tok, train_data)
    
    # 微调后评估
    mdl.eval()
    ft_preds = [infer(mdl, tok, en) for en in eval_en[:10]]
    ft_bleu = eval_bleu(ft_preds, eval_zh[:10])
    print(f'  微调后BLEU: {ft_bleu:.4f}')
    
    elapsed = time.time() - t0
    print(f'  耗时: {elapsed:.0f}s')
    
    return {
        'config': config_name,
        'base_bleu': base_bleu,
        'ft_bleu': ft_bleu,
        'improvement': ft_bleu - base_bleu,
        'losses': losses,
        'time_s': elapsed,
    }


def main():
    sizes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [5, 100, 500]
    
    all_results = []
    
    for n in sizes:
        print(f'\n{"="*60}')
        print(f'数据规模: n={n}')
        print(f'{"="*60}')
        
        # 加载数据
        with open(DATA_DIR / f'eval_{n}.json') as f:
            eval_data = json.load(f)[:20]  # 最多评估20条
        eval_en = [d['en'] for d in eval_data]
        eval_zh = [d['zh'] for d in eval_data]
        
        with open(DATA_DIR / f'train_{n}.json') as f:
            train_data = json.load(f)
        print(f'训练集: {len(train_data)}, 评估集: {len(eval_data)}')
        
        results_per_size = {'n': n, 'configs': []}
        
        # ── AGFT (冻结16层) ──
        m, t = load_model()
        r = run_one_config(m, t, train_data[:n], eval_en, eval_zh, 'AGFT(16层)', freeze_layers=16)
        results_per_size['configs'].append(r)
        del m; gc.collect(); torch.cuda.empty_cache()
        
        # ── LoRA ──
        m, t = load_model()
        r = run_one_config(m, t, train_data[:n], eval_en, eval_zh, 'LoRA(r=8)', use_lora=True)
        results_per_size['configs'].append(r)
        del m; gc.collect(); torch.cuda.empty_cache()
        
        all_results.append(results_per_size)
        
        # 中间保存
        with open(OUTPUT_DIR / f'scaling_results_{n}.json', 'w') as f:
            json.dump(results_per_size, f, ensure_ascii=False, indent=2)
        print(f'\n✅ n={n} 完成')
    
    # 汇总保存
    with open(OUTPUT_DIR / 'scaling_all_results.json', 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    print(f'\n{"="*60}')
    print('规模梯度汇总')
    print(f'{"="*60}')
    print(f'{"n":>8} | {"AGFT基线":>8} {"AGFT微调":>8} {"LoRA基线":>8} {"LoRA微调":>8}')
    print('-' * 60)
    for r in all_results:
        agft = r['configs'][0]
        lora = r['configs'][1]
        print(f'{r["n"]:>8} | {agft["base_bleu"]:>8.4f} {agft["ft_bleu"]:>8.4f} {lora["base_bleu"]:>8.4f} {lora["ft_bleu"]:>8.4f}')
    
    print(f'\n✅ 全部完成，结果保存到 {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
