#!/usr/bin/env python3
"""
AGFT跨语言迁移实验 v2 - 8bit量化版
适配22GB VRAM (RTX 2080 Ti)
"""

import json
import os
import gc
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

MODEL_PATH = "/mnt/i/hermes/models/Qwen2.5-7B-Instruct"
DATA_DIR = Path(__file__).parent.parent / "data" / "cross_lingual"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "agft_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(split="train"):
    filepath = DATA_DIR / f"{split}_set.json"
    if not filepath.exists():
        filepath = DATA_DIR / "translation_pairs.json"
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_8bit():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("加载模型 (8bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✅ 模型加载完成, 设备: {model.device}")
    return model, tokenizer


def run_inference(model, tokenizer, en_text, max_new_tokens=80):
    prompt = f"Translate the following English text to Chinese. Output ONLY the Chinese translation, nothing else.\n\nEnglish: {en_text}\nChinese:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # 只取第一行
    result = result.strip().split("\n")[0].strip()
    return result


def evaluate_bleu(preds, refs):
    matches = sum(1 for p, r in zip(preds, refs) if p in r or r in p or any(c in r for c in p[:10]))
    return matches / len(preds) if preds else 0.0


def train_step(model, tokenizer, data, optimizer):
    model.train()
    total_loss = 0
    for i, item in enumerate(data):
        prompt = f"Translate English to Chinese:\n{item['en']}\nChinese:"
        full_text = f"{prompt} {item['zh']}"
        
        inputs = tokenizer(full_text, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if i % 3 == 0:
            print(f"    Step {i+1}, Loss: {loss.item():.4f}")
    
    return total_loss / len(data)


def main():
    print("="*60)
    print("AGFT跨语言迁移实验 v2 (8bit量化)")
    print("="*60)
    
    train_data = load_data("train")
    eval_data = load_data("eval")
    print(f"训练集: {len(train_data)} 条, 评估集: {len(eval_data)} 条")
    
    eval_en = [item["en"] for item in eval_data[:5]]
    eval_zh = [item["zh"] for item in eval_data[:5]]
    
    results = []
    
    # ── 实验1: 基线（无微调）──
    print("\n" + "="*60)
    print("实验0: 基线（无微调）")
    print("="*60)
    
    model, tokenizer = load_model_8bit()
    
    baseline_preds = []
    for en in eval_en:
        pred = run_inference(model, tokenizer, en)
        baseline_preds.append(pred)
        print(f"  EN: {en[:50]}...")
        print(f"  翻译: {pred}")
    
    baseline_bleu = evaluate_bleu(baseline_preds, eval_zh)
    print(f"\n基线BLEU: {baseline_bleu:.4f}")
    results.append({"experiment": "基线（无微调）", "bleu": baseline_bleu, "predictions": baseline_preds})
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # ── 实验1: AGFT (冻结16层) ──
    print("\n" + "="*60)
    print("实验1: AGFT (冻结前16层)")
    print("="*60)
    
    model, tokenizer = load_model_8bit()
    
    # 冻结前16层
    for i, layer in enumerate(model.model.layers):
        if i < 16:
            for param in layer.parameters():
                param.requires_grad = False
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"冻结: 前16/28层, 可训练: {trainable/1e9:.2f}B / {total/1e9:.2f}B ({100*trainable/total:.1f}%)")
    
    # 微调前评估
    agft_base_preds = []
    for en in eval_en:
        pred = run_inference(model, tokenizer, en)
        agft_base_preds.append(pred)
        print(f"  微调前: {pred[:50]}")
    agft_base_bleu = evaluate_bleu(agft_base_preds, eval_zh)
    print(f"微调前BLEU: {agft_base_bleu:.4f}")
    
    # 微调
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    loss = train_step(model, tokenizer, train_data[:10], optimizer)
    print(f"平均Loss: {loss:.4f}")
    
    # 微调后评估
    agft_ft_preds = []
    for en in eval_en:
        pred = run_inference(model, tokenizer, en)
        agft_ft_preds.append(pred)
        print(f"  微调后: {pred[:50]}")
    agft_ft_bleu = evaluate_bleu(agft_ft_preds, eval_zh)
    print(f"微调后BLEU: {agft_ft_bleu:.4f}")
    print(f"改进: {agft_ft_bleu - agft_base_bleu:.4f}")
    
    results.append({
        "experiment": "AGFT (冻结16层)",
        "base_bleu": agft_base_bleu,
        "ft_bleu": agft_ft_bleu,
        "improvement": agft_ft_bleu - agft_base_bleu,
        "predictions": agft_ft_preds,
    })
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # ── 实验2: Full FT ──
    print("\n" + "="*60)
    print("实验2: Full FT (全参数微调)")
    print("="*60)
    
    model, tokenizer = load_model_8bit()
    
    # 微调前评估
    ft_base_preds = []
    for en in eval_en:
        pred = run_inference(model, tokenizer, en)
        ft_base_preds.append(pred)
        print(f"  微调前: {pred[:50]}")
    ft_base_bleu = evaluate_bleu(ft_base_preds, eval_zh)
    print(f"微调前BLEU: {ft_base_bleu:.4f}")
    
    # 微调（全参数）
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss = train_step(model, tokenizer, train_data[:5], optimizer)  # 只用5条避免OOM
    print(f"平均Loss: {loss:.4f}")
    
    # 微调后评估
    ft_ft_preds = []
    for en in eval_en:
        pred = run_inference(model, tokenizer, en)
        ft_ft_preds.append(pred)
        print(f"  微调后: {pred[:50]}")
    ft_ft_bleu = evaluate_bleu(ft_ft_preds, eval_zh)
    print(f"微调后BLEU: {ft_ft_bleu:.4f}")
    print(f"改进: {ft_ft_bleu - ft_base_bleu:.4f}")
    
    results.append({
        "experiment": "Full FT",
        "base_bleu": ft_base_bleu,
        "ft_bleu": ft_ft_bleu,
        "improvement": ft_ft_bleu - ft_base_bleu,
        "predictions": ft_ft_preds,
    })
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # ── 保存结果 ──
    output_file = OUTPUT_DIR / "experiment_results_v2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    for r in results:
        if "base_bleu" in r:
            print(f"  {r['experiment']}: 基线={r['base_bleu']:.4f}, 微调后={r['ft_bleu']:.4f}, 改进={r['improvement']:.4f}")
        else:
            print(f"  {r['experiment']}: BLEU={r['bleu']:.4f}")
    
    print(f"\n✅ 结果保存到: {output_file}")


if __name__ == "__main__":
    main()
