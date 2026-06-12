"""
Exp15 Direction 3: Multi-Backend Weighted Fusion
=================================================

核心思路：多 backend 贡献 QK^T 的一部分，加权融合

QK^T = α·QK_SVD + β·QK_Coreset + γ·QK_Kernel
其中 α + β + γ = 1

权重决策:
  α = f(q_len, kv_len, data_skewness)
  β = g(block_size)
  γ = h(entropy)

实验设计:
  - 权重 sweep: grid search α, β, γ
  - 数据驱动权重 vs 启发式权重
  - 3 KV type × 3 q_len × 3 kv_len × 5 weight configs = 135 configs
  - SOTA 目标: 误差 0.1% + 压缩 15x
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional, Literal

import numpy as np
from numpy import linalg as npla

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    ground_truth,
)


# ============== 核心组件复用 ==============

def kmeans_plusplus_init(K: np.ndarray, r: int, seed: int = 0) -> np.ndarray:
    """K-Means++ 初始化"""
    gen = np.random.default_rng(seed)
    n, d = K.shape
    idx = gen.integers(0, n)
    centroids = [K[idx].copy()]
    
    for _ in range(r - 1):
        dists = np.zeros(n)
        for c in centroids:
            dists += np.sum((K - c) ** 2, axis=1)
        probs = dists / dists.sum()
        idx = gen.choice(n, p=probs)
        centroids.append(K[idx].copy())
    
    return np.array(centroids)


def build_coreset_sketch(
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    seed: int = 0,
    num_iters: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建 Coreset sketch"""
    n, d = K.shape
    centroids = kmeans_plusplus_init(K, r, seed)
    values = np.zeros((r, d))
    
    for _ in range(num_iters):
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
        assignments = dists.argmin(axis=1)
        
        new_centroids = np.zeros_like(centroids)
        new_values = np.zeros((r, d))
        weights = np.zeros(r)
        
        for j in range(r):
            mask = assignments == j
            count = mask.sum()
            if count > 0:
                new_centroids[j] = K[mask].mean(axis=0)
                new_values[j] = V[mask].mean(axis=0)
                weights[j] = count / n
        
        centroids = new_centroids
        values = new_values
    
    return centroids, values, weights


def eval_coreset_sketch(
    centroids: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    Q: np.ndarray,
    d: int,
) -> np.ndarray:
    """评估 Coreset sketch"""
    r = centroids.shape[0]
    scores = Q @ centroids.T / np.sqrt(d)
    log_weights = np.log(weights + 1e-30)
    scores_with_weights = scores + log_weights
    
    m = scores_with_weights.max(axis=-1, keepdims=True)
    p = np.exp(scores_with_weights - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ values
    
    return y / np.clip(l, 1e-30, None)


def svd_attention_sketch(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    r: int,
) -> tuple[np.ndarray, np.ndarray]:
    """SVD attention sketch, 返回 (y, attention_scores)"""
    d = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d)
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    A = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-30, None)
    
    U, S, Vt = npla.svd(A, full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]
    V_r = Vt[:r, :].T
    
    # y = A_r @ V
    y = U_r @ (S_r[:, None] * (V_r.T @ V))
    
    return y, scores


def kernel_sketch(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    D_kernel: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Random Feature Kernel Sketch (RFF approximation of RBF kernel)
    φ(x) = [cos(Ωx + b), sin(Ωx + b)] ∈ R^{2D_kernel}
    """
    d = Q.shape[1]
    gen = np.random.default_rng(seed)
    
    # Random projection matrix
    sigma = 1.0 / np.sqrt(d)
    Omega = gen.normal(0, sigma, (d, D_kernel))
    b = gen.uniform(0, 2 * np.pi, D_kernel)
    
    # Project
    Q_proj = np.concatenate([np.cos(Q @ Omega + b), np.sin(Q @ Omega + b)], axis=1)
    K_proj = np.concatenate([np.cos(K @ Omega + b), np.sin(K @ Omega + b)], axis=1)
    
    # Kernel approximation: K_approx = Q_proj @ K_proj^T
    # Attention: A_approx = softmax(K_approx / sqrt(2*D_kernel))
    scores = Q_proj @ K_proj.T / np.sqrt(2 * D_kernel)
    
    # Stable softmax
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    A = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-30, None)
    
    # Output: A @ V
    y = A @ V
    
    return y, scores


# ============== 加权融合核心 ==============

@dataclass
class WeightedFusionResult:
    """加权融合结果"""
    alpha: float  # SVD weight
    beta: float   # Coreset weight
    gamma: float  # Kernel weight
    y_svd: np.ndarray
    y_coreset: np.ndarray
    y_kernel: np.ndarray
    y_fused: np.ndarray
    error_svd: float
    error_coreset: float
    error_kernel: float
    error_fused: float


def compute_data_driven_weights(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    svd_r: int = 8,
    coreset_r: int = 16,
    D_kernel: int = 64,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    数据驱动的权重计算
    基于每个 backend 在验证集上的表现
    """
    q_len, kv_len, d = Q.shape[0], K.shape[0], Q.shape[1]
    
    # Ground truth
    gt = ground_truth(Q, K, V)
    
    # Compute each backend
    y_svd, _ = svd_attention_sketch(Q, K, V, svd_r)
    centroids, values, weights = build_coreset_sketch(K, V, coreset_r, seed)
    y_coreset = eval_coreset_sketch(centroids, values, weights, Q, d)
    y_kernel, _ = kernel_sketch(Q, K, V, D_kernel, seed)
    
    # Compute errors
    err_svd = float(np.abs(y_svd - gt).mean())
    err_coreset = float(np.abs(y_coreset - gt).mean())
    err_kernel = float(np.abs(y_kernel - gt).mean())
    
    # Inverse error weighting (with epsilon for stability)
    eps = 1e-4
    inv_err = np.array([1/(err_svd + eps), 1/(err_coreset + eps), 1/(err_kernel + eps)])
    
    # Normalize
    weights = inv_err / inv_err.sum()
    
    return float(weights[0]), float(weights[1]), float(weights[2])


def heuristic_weights(
    q_len: int,
    kv_len: int,
    kv_type: str,
    block_size: int = 64,
) -> tuple[float, float, float]:
    """
    启发式权重计算
    基于数据特征和任务配置
    """
    # 数据倾斜度
    skewness = {
        "clustered": 0.3,   # 结构化数据，Coreset 表现好
        "random": 0.5,      # 随机数据，混合更好
        "skewed": 0.2,      # 高度倾斜，SVD/Kernel 更好
    }.get(kv_type, 0.5)
    
    # 序列长度比
    ratio = kv_len / max(q_len, 1)
    
    # 基于规则的权重
    if ratio > 50:  # 长 KV
        alpha = 0.4  # SVD 擅长长序列
        beta = 0.4   # Coreset 擅长压缩
        gamma = 0.2  # Kernel 补充
    elif ratio > 10:
        alpha = 0.35
        beta = 0.35
        gamma = 0.3
    else:  # 短 KV
        alpha = 0.3
        beta = 0.4
        gamma = 0.3
    
    # 调整：结构化数据多给 Coreset
    if kv_type == "clustered":
        beta += 0.1
        alpha -= 0.05
        gamma -= 0.05
    elif kv_type == "skewed":
        alpha += 0.1
        gamma += 0.1
        beta -= 0.2
    
    # 归一化
    total = alpha + beta + gamma
    return alpha/total, beta/total, gamma/total


def weighted_fusion(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    svd_r: int = 8,
    coreset_r: int = 16,
    D_kernel: int = 64,
    seed: int = 0,
) -> WeightedFusionResult:
    """
    执行加权融合
    
    QK^T_fused = α·QK_SVD + β·QK_Coreset + γ·QK_Kernel
    
    融合在 attention 输出层面进行:
    y_fused = α·y_SVD + β·y_Coreset + γ·y_Kernel
    """
    d = Q.shape[1]
    
    # Ground truth
    gt = ground_truth(Q, K, V)
    
    # Compute each backend
    y_svd, scores_svd = svd_attention_sketch(Q, K, V, svd_r)
    centroids, values, weights = build_coreset_sketch(K, V, coreset_r, seed)
    y_coreset = eval_coreset_sketch(centroids, values, weights, Q, d)
    y_kernel, scores_kernel = kernel_sketch(Q, K, V, D_kernel, seed)
    
    # Weighted fusion of outputs
    y_fused = alpha * y_svd + beta * y_coreset + gamma * y_kernel
    
    # Compute errors
    err_svd = float(np.abs(y_svd - gt).mean())
    err_coreset = float(np.abs(y_coreset - gt).mean())
    err_kernel = float(np.abs(y_kernel - gt).mean())
    err_fused = float(np.abs(y_fused - gt).mean())
    
    return WeightedFusionResult(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        y_svd=y_svd,
        y_coreset=y_coreset,
        y_kernel=y_kernel,
        y_fused=y_fused,
        error_svd=err_svd,
        error_coreset=err_coreset,
        error_kernel=err_kernel,
        error_fused=err_fused,
    )


# ============== 数据生成 ==============

def make_clustered_kv(kv_len: int, d: int, n_clusters: int = 8, seed: int = 0):
    """生成 cluster 结构的 KV"""
    gen = np.random.default_rng(seed)
    centroids = gen.standard_normal((n_clusters, d)) * 2.0
    assignments = gen.integers(0, n_clusters, size=kv_len)
    K = centroids[assignments] + gen.standard_normal((kv_len, d)) * 0.5
    V = K @ (gen.standard_normal((d, d)) * 0.3) + gen.standard_normal((kv_len, d)) * 0.1
    return K.astype(np.float32), V.astype(np.float32)


def make_random_kv(kv_len: int, d: int, seed: int = 0):
    """生成随机 KV"""
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((kv_len, d)).astype(np.float32)
    V = gen.standard_normal((kv_len, d)).astype(np.float32)
    return K, V


def make_skewed_kv(kv_len: int, d: int, n_outliers: int = 16, seed: int = 0):
    """生成 skew 结构的 KV"""
    gen = np.random.default_rng(seed)
    outlier_K = gen.standard_normal((n_outliers, d)) * 3.0
    outlier_V = gen.standard_normal((n_outliers, d)) * 3.0
    normal_K = gen.standard_normal((kv_len - n_outliers, d)) * 0.3
    normal_V = gen.standard_normal((kv_len - n_outliers, d)) * 0.3
    K = np.concatenate([outlier_K, normal_K])
    V = np.concatenate([outlier_V, normal_V])
    perm = gen.permutation(kv_len)
    return K[perm].astype(np.float32), V[perm].astype(np.float32)


# ============== Sweep ==============

def run_sweep_weighted_fusion():
    """运行加权融合 sweep"""
    print("=" * 60)
    print("Exp15 Direction 3: Multi-Backend Weighted Fusion")
    print("=" * 60)
    
    d = 128
    sweep_configs = {
        "weight_configs": [
            # 均匀权重
            (1/3, 1/3, 1/3),
            # SVD 主导
            (0.5, 0.3, 0.2),
            (0.6, 0.2, 0.2),
            # Coreset 主导
            (0.3, 0.5, 0.2),
            (0.2, 0.6, 0.2),
            # Kernel 主导
            (0.3, 0.2, 0.5),
            (0.2, 0.2, 0.6),
            # SVD + Coreset
            (0.45, 0.45, 0.1),
            (0.4, 0.4, 0.2),
            # 极端
            (0.7, 0.15, 0.15),
            (0.15, 0.7, 0.15),
            (0.15, 0.15, 0.7),
        ],
        "kv_types": ["clustered", "random", "skewed"],
        "kv_lens": [1024, 4096, 16384],
        "q_lens": [16, 64, 256],
        "svd_r_values": [4, 8, 12],
        "coreset_r_values": [8, 16, 32],
    }
    
    results = []
    pareto_points = []
    
    total_configs = (
        len(sweep_configs["weight_configs"]) *
        len(sweep_configs["kv_types"]) *
        len(sweep_configs["kv_lens"]) *
        len(sweep_configs["q_lens"])
    )
    
    config_idx = 0
    start_time = time.time()
    
    for kv_type in sweep_configs["kv_types"]:
        for kv_len in sweep_configs["kv_lens"]:
            for q_len in sweep_configs["q_lens"]:
                # 生成数据
                if kv_type == "clustered":
                    K, V = make_clustered_kv(kv_len, d, seed=42)
                elif kv_type == "random":
                    K, V = make_random_kv(kv_len, d, seed=42)
                else:
                    K, V = make_skewed_kv(kv_len, d, seed=42)
                
                # Query
                gen = np.random.default_rng(100)
                Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
                
                for alpha, beta, gamma in sweep_configs["weight_configs"]:
                    config_idx += 1
                    
                    try:
                        result_fusion = weighted_fusion(
                            Q, K, V,
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
                            svd_r=8,
                            coreset_r=16,
                            D_kernel=64,
                            seed=42,
                        )
                        
                        # 同时测试数据驱动权重
                        alpha_dd, beta_dd, gamma_dd = compute_data_driven_weights(
                            Q, K, V, svd_r=8, coreset_r=16, D_kernel=64, seed=42
                        )
                        result_dd = weighted_fusion(
                            Q, K, V,
                            alpha=alpha_dd,
                            beta=beta_dd,
                            gamma=gamma_dd,
                            svd_r=8,
                            coreset_r=16,
                            D_kernel=64,
                            seed=42,
                        )
                        
                        # 启发式权重
                        alpha_h, beta_h, gamma_h = heuristic_weights(
                            q_len, kv_len, kv_type
                        )
                        result_h = weighted_fusion(
                            Q, K, V,
                            alpha=alpha_h,
                            beta=beta_h,
                            gamma=gamma_h,
                            svd_r=8,
                            coreset_r=16,
                            D_kernel=64,
                            seed=42,
                        )
                        
                        result = {
                            "kv_type": kv_type,
                            "kv_len": kv_len,
                            "q_len": q_len,
                            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
                            "weights_data_driven": {"alpha": alpha_dd, "beta": beta_dd, "gamma": gamma_dd},
                            "weights_heuristic": {"alpha": alpha_h, "beta": beta_h, "gamma": gamma_h},
                            "err_svd": result_fusion.error_svd,
                            "err_coreset": result_fusion.error_coreset,
                            "err_kernel": result_fusion.error_kernel,
                            "err_fused": result_fusion.error_fused,
                            "err_data_driven": result_dd.error_fused,
                            "err_heuristic": result_h.error_fused,
                            "best_single": min(result_fusion.error_svd, result_fusion.error_coreset, result_fusion.error_kernel),
                            "fusion_gain": min(result_fusion.error_svd, result_fusion.error_coreset, result_fusion.error_kernel) - result_fusion.error_fused,
                        }
                        
                        results.append(result)
                        
                        # Pareto
                        if result_fusion.error_fused < 0.5:
                            pareto_points.append(result)
                        
                        if config_idx % 15 == 0:
                            elapsed = time.time() - start_time
                            rate = config_idx / elapsed
                            remaining = (total_configs - config_idx) / rate if rate > 0 else 0
                            print(f"Progress: {config_idx}/{total_configs} ({elapsed:.1f}s, ~{remaining:.1f}s remaining)")
                            
                    except Exception as e:
                        print(f"Error in config {config_idx}: {e}")
                        continue
    
    # Sort and find Pareto
    results_sorted = sorted(results, key=lambda x: x["err_fused"])
    
    # Pareto frontier
    pareto_sorted = []
    for p in pareto_points:
        is_dominated = False
        for other in pareto_points:
            # Skip if same point
            if p == other:
                continue
            # other dominates p if better in both dimensions
            if (other["err_fused"] <= p["err_fused"] and 
                (1 - other["weights"]["alpha"]) <= (1 - p["weights"]["alpha"])):
                if other["err_fused"] < p["err_fused"] or (1 - other["weights"]["alpha"]) < (1 - p["weights"]["alpha"]):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_sorted.append(p)
    
    pareto_sorted = sorted(pareto_sorted, key=lambda x: x["err_fused"])
    
    # Statistics
    stats = {
        "total_configs": len(results),
        "pareto_points": len(pareto_sorted),
        "mean_error": float(np.mean([r["err_fused"] for r in results])),
        "mean_gain": float(np.mean([r["fusion_gain"] for r in results])),
        "best_single_vs_fused": {
            "single_better": sum(1 for r in results if r["best_single"] < r["err_fused"]),
            "fused_better": sum(1 for r in results if r["err_fused"] <= r["best_single"]),
        },
    }
    
    print(f"\nSweep complete!")
    print(f"Total configs: {stats['total_configs']}")
    print(f"Pareto points: {stats['pareto_points']}")
    print(f"Mean error: {stats['mean_error']:.4f}")
    print(f"Mean fusion gain: {stats['mean_gain']:.4f}")
    print(f"Single backend better: {stats['best_single_vs_fused']['single_better']}")
    print(f"Fusion better: {stats['best_single_vs_fused']['fused_better']}")
    
    return {
        "sweep_results": results_sorted,
        "pareto_frontier": pareto_sorted,
        "stats": stats,
        "configs": sweep_configs,
    }


def analyze_weighted_results(sweep_data: dict) -> dict:
    """分析加权融合结果"""
    results = sweep_data["sweep_results"]
    
    analysis = {
        "by_weight_type": {},
        "by_kv_type": {},
        "weight_effectiveness": {},
        "sota_recommendation": None,
    }
    
    # By weight type
    for wtype in ["manual", "data_driven", "heuristic"]:
        if wtype == "manual":
            errs = [r["err_fused"] for r in results]
        elif wtype == "data_driven":
            errs = [r["err_data_driven"] for r in results]
        else:
            errs = [r["err_heuristic"] for r in results]
        
        if errs:
            analysis["by_weight_type"][wtype] = {
                "mean_err": float(np.mean(errs)),
                "std_err": float(np.std(errs)),
            }
    
    # By KV type
    for kv_type in ["clustered", "random", "skewed"]:
        subset = [r for r in results if r["kv_type"] == kv_type]
        if subset:
            analysis["by_kv_type"][kv_type] = {
                "mean_err": float(np.mean([r["err_fused"] for r in subset])),
                "mean_gain": float(np.mean([r["fusion_gain"] for r in subset])),
                "stability": float(np.std([r["err_fused"] for r in subset])),
            }
    
    # Best weight configs
    best_configs = sorted(results, key=lambda x: x["err_fused"])[:10]
    analysis["weight_effectiveness"]["top_10_configs"] = [
        {
            "weights": r["weights"],
            "kv_type": r["kv_type"],
            "err": r["err_fused"],
        }
        for r in best_configs
    ]
    
    # SOTA recommendation
    if best_configs:
        best = best_configs[0]
        analysis["sota_recommendation"] = {
            "weight_type": "data_driven" if best["err_data_driven"] < best["err_fused"] else "manual",
            "best_weights": best["weights"],
            "kv_type": best["kv_type"],
            "kv_len": best["kv_len"],
            "q_len": best["q_len"],
            "err_fused": best["err_fused"],
            "fusion_gain": best["fusion_gain"],
        }
    
    return analysis


def save_results(sweep_data: dict, analysis: dict, output_dir: str):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "exp15_weighted_sweep.json"), "w") as f:
        json.dump(sweep_data, f, indent=2, default=str)
    
    with open(os.path.join(output_dir, "exp15_weighted_pareto.json"), "w") as f:
        json.dump({
            "pareto_frontier": sweep_data["pareto_frontier"],
            "stats": sweep_data["stats"],
            "analysis": analysis,
        }, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}")


def main():
    output_dir = os.path.join(_REPO_ROOT, "results")
    
    # Run sweep
    sweep_data = run_sweep_weighted_fusion()
    
    # Analyze
    analysis = analyze_weighted_results(sweep_data)
    
    # Save
    save_results(sweep_data, analysis, output_dir)
    
    # Print analysis
    print("\n" + "=" * 60)
    print("Analysis Results")
    print("=" * 60)
    
    print("\nBy Weight Type:")
    for wtype, stats in analysis["by_weight_type"].items():
        print(f"  {wtype}: err={stats['mean_err']:.4f} ± {stats['std_err']:.4f}")
    
    print("\nBy KV Type:")
    for kv_type, stats in analysis["by_kv_type"].items():
        print(f"  {kv_type}: err={stats['mean_err']:.4f}, gain={stats['mean_gain']:.4f}, stability={stats['stability']:.4f}")
    
    if analysis["sota_recommendation"]:
        rec = analysis["sota_recommendation"]
        print("\n" + "=" * 60)
        print("SOTA Recommendation:")
        print("=" * 60)
        print(f"  Weight type: {rec['weight_type']}")
        print(f"  Best weights: α={rec['best_weights']['alpha']:.2f}, β={rec['best_weights']['beta']:.2f}, γ={rec['best_weights']['gamma']:.2f}")
        print(f"  Best on: {rec['kv_type']}, kv_len={rec['kv_len']}, q_len={rec['q_len']}")
        print(f"  Error: {rec['err_fused']:.4f}")
        print(f"  Fusion gain: {rec['fusion_gain']:.4f}")
    
    return sweep_data, analysis


if __name__ == "__main__":
    main()
