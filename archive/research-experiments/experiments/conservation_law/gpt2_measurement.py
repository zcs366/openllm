"""
GPT-2守恒律测量 — 子贡出品

测量GPT-2各层的K_W值，验证守恒律d×SNR≈K_W

步骤：
1. 加载GPT-2模型
2. 提取各层注意力矩阵
3. 测量每个d值的SNR
4. 计算K_W = d × SNR
5. 画出K_W曲线
"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import json
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))
from snr_measurement import measure_kw_curve, analyze_kw_stability
from calibration_set import get_calibration_set


def load_gpt2():
    """加载GPT-2模型和分词器"""
    print("加载GPT-2模型...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
    model.eval()
    print(f"模型加载完成。层数: {len(model.h)}, 隐藏维度: {model.config.n_embd}")
    return tokenizer, model


def extract_attention_matrices(model, tokenizer, text, layer_indices=None):
    """
    提取指定层的注意力矩阵
    
    Args:
        model: GPT-2模型
        tokenizer: 分词器
        text: 输入文本
        layer_indices: 要提取的层索引列表
    
    Returns:
        attention_matrices: 各层的注意力矩阵字典
    """
    # 分词
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取注意力矩阵
    attentions = outputs.attentions  # tuple of (batch, heads, seq_len, seq_len)
    
    if layer_indices is None:
        layer_indices = list(range(len(attentions)))
    
    attention_matrices = {}
    for layer_idx in layer_indices:
        # 取第一个batch，平均所有注意力头
        attn = attentions[layer_idx][0]  # (heads, seq_len, seq_len)
        attn_avg = attn.mean(dim=0)  # (seq_len, seq_len)
        attention_matrices[layer_idx] = attn_avg.numpy()
    
    return attention_matrices


def measure_all_layers(tokenizer, model, calibration_set, d_values=None):
    """
    测量所有层的K_W值
    
    Args:
        tokenizer: 分词器
        model: GPT-2模型
        calibration_set: 校准集
        d_values: 要测试的d值列表
    
    Returns:
        results: 各层各d值的测量结果
    """
    if d_values is None:
        d_values = [64, 32, 16, 8, 4, 2, 1]
    
    num_layers = len(model.h)
    results = {
        "layers": {},
        "d_values": d_values,
        "calibration_size": len(calibration_set)
    }
    
    # 对每个校准样本
    for sample_idx, text in enumerate(calibration_set):
        if sample_idx % 10 == 0:
            print(f"处理样本 {sample_idx}/{len(calibration_set)}...")
        
        # 提取各层注意力矩阵
        attention_matrices = extract_attention_matrices(model, tokenizer, text)
        
        # 对每层
        for layer_idx, attn_matrix in attention_matrices.items():
            if layer_idx not in results["layers"]:
                results["layers"][layer_idx] = {
                    "kw_values": {d: [] for d in d_values},
                    "snr_values": {d: [] for d in d_values}
                }
            
            # 测量K_W曲线
            curve_results = measure_kw_curve(attn_matrix, d_values)
            
            # 记录结果
            for i, d in enumerate(d_values):
                results["layers"][layer_idx]["kw_values"][d].append(curve_results["kw_values"][i])
                results["layers"][layer_idx]["snr_values"][d].append(curve_results["snr_values"][i])
    
    return results


def analyze_results(results):
    """
    分析测量结果
    
    Args:
        results: 测量结果
    
    Returns:
        analysis: 分析结果
    """
    analysis = {
        "layers": {},
        "overall": {}
    }
    
    all_kw_values = []
    
    for layer_idx, layer_data in results["layers"].items():
        layer_analysis = {
            "d_values": {},
            "mean_kw": {},
            "std_kw": {},
            "cv_kw": {}
        }
        
        for d in results["d_values"]:
            kw_values = layer_data["kw_values"][d]
            # 过滤掉inf和nan值
            valid_kw_values = [v for v in kw_values if v != float('inf') and v != float('-inf') and not (isinstance(v, float) and v != v)]
            
            if valid_kw_values:
                kw_array = np.array(valid_kw_values)
                
                layer_analysis["d_values"][d] = {
                    "mean": float(np.mean(kw_array)),
                    "std": float(np.std(kw_array)),
                    "cv": float(np.std(kw_array) / np.mean(kw_array)) if np.mean(kw_array) > 0 else float('inf'),
                    "valid_count": len(valid_kw_values),
                    "total_count": len(kw_values)
                }
                
                all_kw_values.extend(valid_kw_values)
            else:
                layer_analysis["d_values"][d] = {
                    "mean": float('nan'),
                    "std": float('nan'),
                    "cv": float('nan'),
                    "valid_count": 0,
                    "total_count": len(kw_values)
                }
        
        # 计算该层的整体统计（只用有效值）
        all_layer_kw = []
        for d in results["d_values"]:
            kw_values = layer_data["kw_values"][d]
            valid_kw_values = [v for v in kw_values if v != float('inf') and v != float('-inf') and not (isinstance(v, float) and v != v)]
            all_layer_kw.extend(valid_kw_values)
        
        if all_layer_kw:
            layer_analysis["overall"] = {
                "mean": float(np.mean(all_layer_kw)),
                "std": float(np.std(all_layer_kw)),
                "cv": float(np.std(all_layer_kw) / np.mean(all_layer_kw)) if np.mean(all_layer_kw) > 0 else float('inf'),
                "valid_count": len(all_layer_kw)
            }
        else:
            layer_analysis["overall"] = {
                "mean": float('nan'),
                "std": float('nan'),
                "cv": float('nan'),
                "valid_count": 0
            }
        
        analysis["layers"][layer_idx] = layer_analysis
    
    # 整体统计（只用有效值）
    if all_kw_values:
        all_kw_array = np.array(all_kw_values)
        analysis["overall"] = {
            "mean": float(np.mean(all_kw_array)),
            "std": float(np.std(all_kw_array)),
            "cv": float(np.std(all_kw_array) / np.mean(all_kw_array)) if np.mean(all_kw_array) > 0 else float('inf'),
            "min": float(np.min(all_kw_array)),
            "max": float(np.max(all_kw_array)),
            "valid_count": len(all_kw_values)
        }
    else:
        analysis["overall"] = {
            "mean": float('nan'),
            "std": float('nan'),
            "cv": float('nan'),
            "min": float('nan'),
            "max": float('nan'),
            "valid_count": 0
        }
    
    return analysis


def convert_to_serializable(obj):
    """将numpy类型转换为JSON可序列化的类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def main():
    """主函数"""
    print("=== GPT-2守恒律测量 ===\n")
    
    # 加载模型
    tokenizer, model = load_gpt2()
    
    # 获取校准集
    calibration_set = get_calibration_set()
    print(f"校准集大小: {len(calibration_set)}")
    
    # 定义d值（使用较小的值，确保对所有样本有效）
    # 注意力矩阵是(seq_len, seq_len)，seq_len通常在5-100之间
    # 所以d必须小于最小seq_len
    d_values = [4, 2, 1]
    print(f"测试d值: {d_values}\n")
    
    # 测量所有层
    print("开始测量...")
    results = measure_all_layers(tokenizer, model, calibration_set, d_values)
    
    # 分析结果
    print("\n分析结果...")
    analysis = analyze_results(results)
    
    # 打印结果
    print("\n=== 测量结果 ===")
    print(f"层数: {len(results['layers'])}")
    print(f"校准样本数: {results['calibration_size']}")
    
    print("\n各层K_W均值:")
    for layer_idx in sorted(results["layers"].keys()):
        layer_analysis = analysis["layers"][layer_idx]
        print(f"  Layer {layer_idx:2d}: K_W = {layer_analysis['overall']['mean']:.2f} ± {layer_analysis['overall']['std']:.2f}")
    
    print(f"\n整体K_W: {analysis['overall']['mean']:.2f} ± {analysis['overall']['std']:.2f}")
    print(f"变异系数: {analysis['overall']['cv']:.4f}")
    print(f"范围: [{analysis['overall']['min']:.2f}, {analysis['overall']['max']:.2f}]")
    
    # 保存结果
    output_path = "/home/zcs/projects/openllm/experiments/conservation_law/gpt2_results.json"
    
    # 转换为JSON可序列化的类型
    serializable_results = convert_to_serializable(results)
    serializable_analysis = convert_to_serializable(analysis)
    
    with open(output_path, 'w') as f:
        json.dump({
            "results": serializable_results,
            "analysis": serializable_analysis
        }, f, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    
    # 判断守恒律是否成立
    print("\n=== 守恒律验证 ===")
    cv = analysis['overall']['cv']
    if cv < 0.1:
        print(f"✅ 守恒律成立！K_W变异系数 = {cv:.4f} < 0.1")
    elif cv < 0.3:
        print(f"⚠️ 守恒律可能成立。K_W变异系数 = {cv:.4f} (0.1 < cv < 0.3)")
    else:
        print(f"❌ 守恒律不成立。K_W变异系数 = {cv:.4f} > 0.3")


if __name__ == "__main__":
    main()
