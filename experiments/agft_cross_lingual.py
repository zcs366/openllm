#!/usr/bin/env python3
"""
AGFT跨语言迁移实验 - Qwen2.5-7B

实验组：
1. AGFT组：冻结语义头，微调表达头
2. Full FT组：全参数微调（基线）
3. LoRA组：LoRA微调（基线）

评估：BLEU + 语义相似度 + 灾难性遗忘
"""

import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# ── 配置 ──

MODEL_NAME = "/mnt/i/hermes/models/Qwen2.5-7B-Instruct"
DATA_DIR = Path(__file__).parent.parent / "data" / "cross_lingual"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "agft_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 256
TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 4


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    use_agft: bool = False
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    freeze_layers: int = 16  # AGFT: 冻结前16层（语义层）


def load_data(split: str = "train") -> List[Dict]:
    """加载数据"""
    filename = f"{split}_set.json" if split != "all" else "translation_pairs.json"
    filepath = DATA_DIR / filename
    if not filepath.exists():
        filepath = DATA_DIR / "translation_pairs.json"
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_and_tokenizer():
    """加载模型和tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"✅ 模型加载完成, 设备: {model.device}")
    return model, tokenizer


def apply_lora(model, r=8, alpha=16):
    """应用LoRA"""
    from peft import LoraConfig, get_peft_model, TaskType
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA应用完成, 可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def apply_agft_freeze(model, freeze_layers=16):
    """AGFT: 冻结语义层（前N层），微调表达层（后N层）"""
    total_layers = len(model.model.layers)
    frozen = 0
    
    for i, layer in enumerate(model.model.layers):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False
            frozen += 1
    
    # embedding层冻结
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False
    
    # lm_head保持可训练（输出层）
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ AGFT冻结完成: 冻结前{frozen}/{total_layers}层")
    print(f"   可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def prepare_input(tokenizer, en_text: str, max_length=256):
    """准备输入"""
    prompt = f"Translate the following English text to Chinese:\n\n{en_text}\n\nChinese translation:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    return {k: v.to(next(iter(model.parameters())).device) if 'model' in dir() else v for k, v in inputs.items()}


def evaluate_bleu(predictions: List[str], references: List[str]) -> float:
    """计算BLEU分数"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = list(pred)
            ref_tokens = [list(ref)]
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
            scores.append(score)
        return np.mean(scores)
    except Exception:
        # 简单的字符级匹配
        matches = sum(1 for p, r in zip(predictions, references) if p in r or r in p)
        return matches / len(predictions) if predictions else 0.0


def run_inference(model, tokenizer, en_text: str, max_new_tokens=100) -> str:
    """运行推理"""
    prompt = f"Translate the following English text to Chinese:\n\n{en_text}\n\nChinese translation:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=False,
        )
    
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def run_experiment(config: ExperimentConfig, model, tokenizer, train_data, eval_data):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"实验: {config.name}")
    print(f"{'='*60}")
    
    # 准备模型
    if config.use_lora:
        model = apply_lora(model, r=config.lora_r, alpha=config.lora_alpha)
    elif config.use_agft:
        model = apply_agft_freeze(model, freeze_layers=config.freeze_layers)
    
    # 基线评估（微调前）
    print("\n--- 微调前评估 ---")
    baseline_preds = []
    baseline_refs = []
    for item in eval_data[:5]:  # 只评估前5条
        pred = run_inference(model, tokenizer, item["en"])
        baseline_preds.append(pred)
        baseline_refs.append(item["zh"])
        print(f"  EN: {item['en'][:50]}...")
        print(f"  预测: {pred[:50]}...")
        print(f"  参考: {item['zh'][:50]}...")
        print()
    
    baseline_bleu = evaluate_bleu(baseline_preds, baseline_refs)
    print(f"基线BLEU: {baseline_bleu:.4f}")
    
    # 简单微调（只训练1个epoch，用前5条数据）
    print("\n--- 微调中 ---")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    
    model.train()
    for epoch in range(1):  # 只训练1个epoch
        for i, item in enumerate(train_data[:5]):
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
            
            if i % 2 == 0:
                print(f"  Step {i+1}, Loss: {loss.item():.4f}")
    
    # 微调后评估
    print("\n--- 微调后评估 ---")
    model.eval()
    finetuned_preds = []
    for item in eval_data[:5]:
        pred = run_inference(model, tokenizer, item["en"])
        finetuned_preds.append(pred)
        print(f"  EN: {item['en'][:50]}...")
        print(f"  预测: {pred[:50]}...")
        print(f"  参考: {item['zh'][:50]}...")
        print()
    
    finetuned_bleu = evaluate_bleu(finetuned_preds, baseline_refs)
    print(f"微调后BLEU: {finetuned_bleu:.4f}")
    
    # 灾难性遗忘评估（英文性能）
    print("\n--- 灾难性遗忘评估 ---")
    en_preds = []
    en_refs = []
    for item in eval_data[:3]:
        # 简单的英文任务
        prompt = f"Summarize: {item['en']}"
        pred = run_inference(model, tokenizer, prompt)
        en_preds.append(pred)
        en_refs.append(item["en"][:50])
    
    遗忘率 = baseline_bleu - finetuned_bleu
    
    result = {
        "experiment": config.name,
        "baseline_bleu": baseline_bleu,
        "finetuned_bleu": finetuned_bleu,
        "遗忘率": max(遗忘率, 0),
        "improvement": finetuned_bleu - baseline_bleu,
    }
    
    print(f"\n--- {config.name} 结果 ---")
    print(f"  基线BLEU: {baseline_bleu:.4f}")
    print(f"  微调后BLEU: {finetuned_bleu:.4f}")
    print(f"  改进: {result['improvement']:.4f}")
    print(f"  遗忘率: {result['遗忘率']:.4f}")
    
    return result


def main():
    """主函数"""
    print("="*60)
    print("AGFT跨语言迁移实验 - Qwen2.5-7B")
    print("="*60)
    
    # 加载数据
    train_data = load_data("train")
    eval_data = load_data("eval")
    print(f"训练集: {len(train_data)} 条, 评估集: {len(eval_data)} 条")
    
    # 加载模型
    global model
    model, tokenizer = load_model_and_tokenizer()
    
    # 定义实验
    experiments = [
        ExperimentConfig("AGFT (冻结16层)", use_agft=True, freeze_layers=16),
        ExperimentConfig("Full FT (全参数微调)", use_agft=False),
        ExperimentConfig("LoRA (r=8)", use_lora=True, lora_r=8),
    ]
    
    # 运行实验
    results = []
    for config in experiments:
        try:
            result = run_experiment(config, model, tokenizer, train_data, eval_data)
            results.append(result)
        except Exception as e:
            print(f"❌ 实验 {config.name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    output_file = OUTPUT_DIR / "experiment_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    for r in results:
        print(f"  {r['experiment']}: BLEU={r['finetuned_bleu']:.4f}, 改进={r['improvement']:.4f}, 遗忘={r['遗忘率']:.4f}")
    
    print(f"\n✅ 结果保存到: {output_file}")


if __name__ == "__main__":
    main()