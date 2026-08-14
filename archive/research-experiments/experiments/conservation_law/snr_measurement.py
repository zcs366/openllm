"""
SNR测量工具 — 鲁班出品

测量注意力矩阵的信噪比（Signal-to-Noise Ratio）

SNR定义：
- 信号 = 主成分方差（PCA前d个成分）
- 噪声 = 剩余方差（PCA后768-d个成分）
- SNR = 信号方差 / 噪声方差

守恒律：d × SNR ≈ K_W (常数)
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Optional
import json


def measure_snr(attention_matrix: np.ndarray, d: int) -> Tuple[float, float, float]:
    """
    测量注意力矩阵的SNR
    
    Args:
        attention_matrix: 注意力矩阵 (seq_len, hidden_dim)
        d: 压缩维度（PCA保留的维度数）
    
    Returns:
        snr: 信噪比
        signal_var: 信号方差
        noise_var: 噪声方差
    """
    # 真正的总方差 = 每个特征的方差之和
    total_var = np.sum(np.var(attention_matrix, axis=0))
    
    # 用d维PCA获取信号方差（前d个特征值之和）
    max_d = min(attention_matrix.shape)
    actual_d = min(d, max_d)
    pca = PCA(n_components=actual_d)
    pca.fit(attention_matrix)
    signal_var = np.sum(pca.explained_variance_)
    
    # 噪声 = 总方差 - 信号方差
    noise_var = total_var - signal_var
    
    # 计算SNR
    snr = signal_var / noise_var if noise_var > 0 else float('inf')
    
    return snr, signal_var, noise_var


def measure_kw(attention_matrix: np.ndarray, d: int, use_shannon: bool = True) -> float:
    """
    测量信道容量 C = d × log₂(1 + SNR) （Shannon公式）
    或旧版 K_W = d × SNR （线性近似）

    Args:
        attention_matrix: 注意力矩阵
        d: 压缩维度
        use_shannon: True用Shannon公式，False用旧线性公式

    Returns:
        capacity: 信道容量或K_W
    """
    snr, _, _ = measure_snr(attention_matrix, d)
    if use_shannon:
        return d * np.log2(1 + snr)
    else:
        return d * snr


def measure_kw_curve(attention_matrix: np.ndarray, 
                     d_values: list = None) -> dict:
    """
    测量K_W随d变化的曲线
    
    Args:
        attention_matrix: 注意力矩阵 (seq_len, hidden_dim)
        d_values: 要测试的d值列表
    
    Returns:
        results: 包含所有测量结果的字典
    """
    seq_len, hidden_dim = attention_matrix.shape
    max_d = min(seq_len, hidden_dim)
    
    if d_values is None:
        # 默认从max_d到1的压缩曲线
        d_values = [max_d, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    
    # 过滤掉超过max_d的值，并确保d < max_d（留出噪声空间）
    d_values = [d for d in d_values if d < max_d]
    
    results = {
        "d_values": d_values,
        "snr_values": [],
        "kw_values": [],
        "signal_var_values": [],
        "noise_var_values": []
    }
    
    for d in d_values:
        snr, signal_var, noise_var = measure_snr(attention_matrix, d)
        # Shannon信道容量: C = d × log₂(1 + SNR)
        kw = d * np.log2(1 + snr)
        
        results["snr_values"].append(snr)
        results["kw_values"].append(kw)
        results["signal_var_values"].append(signal_var)
        results["noise_var_values"].append(noise_var)
    
    return results


def analyze_kw_stability(kw_values: list) -> dict:
    """
    分析K_W的稳定性
    
    Args:
        kw_values: K_W值列表
    
    Returns:
        analysis: 稳定性分析结果
    """
    kw_array = np.array(kw_values)
    
    analysis = {
        "mean": float(np.mean(kw_array)),
        "std": float(np.std(kw_array)),
        "min": float(np.min(kw_array)),
        "max": float(np.max(kw_array)),
        "cv": float(np.std(kw_array) / np.mean(kw_array)) if np.mean(kw_array) > 0 else float('inf'),
        "range": float(np.max(kw_array) - np.min(kw_array)),
        "stability": "stable" if np.std(kw_array) / np.mean(kw_array) < 0.1 else "unstable"
    }
    
    return analysis


def main():
    """测试函数"""
    # 生成随机注意力矩阵（模拟）
    np.random.seed(42)
    seq_len = 100
    hidden_dim = 768
    
    # 模拟一个有结构的注意力矩阵
    # 前10个维度有信号，其余是噪声
    signal = np.random.randn(seq_len, 10) * 5  # 强信号
    noise = np.random.randn(seq_len, hidden_dim - 10) * 1  # 弱噪声
    attention_matrix = np.hstack([signal, noise])
    
    print("=== SNR测量测试 ===\n")
    
    # 测试不同d值
    d_values = [768, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    results = measure_kw_curve(attention_matrix, d_values)
    
    print("d\tSNR\t\tK_W")
    print("-" * 40)
    for i, d in enumerate(d_values):
        snr = results["snr_values"][i]
        kw = results["kw_values"][i]
        print(f"{d}\t{snr:.4f}\t\t{kw:.4f}")
    
    # 分析稳定性
    analysis = analyze_kw_stability(results["kw_values"])
    
    print(f"\n=== K_W稳定性分析 ===")
    print(f"均值: {analysis['mean']:.4f}")
    print(f"标准差: {analysis['std']:.4f}")
    print(f"变异系数: {analysis['cv']:.4f}")
    print(f"范围: {analysis['range']:.4f}")
    print(f"稳定性: {analysis['stability']}")
    
    # 保存结果
    output_path = "/home/zcs/projects/openllm/experiments/conservation_law/test_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            "results": results,
            "analysis": analysis
        }, f, indent=2)
    
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
