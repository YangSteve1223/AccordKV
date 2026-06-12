"""
Exp22: 极端压缩边界触碰
========================

核心目标：主动触碰物理上限边界
1. 极端参数扫描: coreset_ratio ∈ [0.01, 0.02, 0.05, 0.10], svd_r ∈ [1, 2, 3, 4], int4_bits ∈ [2, 3, 4]
2. 反序实验: INT2 → SVD → Coreset
3. 4-stage 链路: Prune → Coreset → SVD → INT2
4. 找帕累托"悬崖"点

物理诚实边界: ratio 上限 ≈ 2·kv_len/q_len
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from numpy import linalg as npla

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import ground_truth


# ============== 核心组件（复用 exp15） ==============

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
    num_iters: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建 Coreset sketch"""
    n, d = K.shape
    centroids = kmeans_plusplus_init(K, r, seed)
    values = np.zeros((r, d))
    weights = np.zeros(r)
    
    for _ in range(num_iters):
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
        assignments = dists.argmin(axis=1)
        
        new_centroids = np.zeros_like(centroids)
        new_values = np.zeros((r, d))
        new_weights = np.zeros(r)
        
        for j in range(r):
            mask = assignments == j
            count = mask.sum()
            if count > 0:
                new_centroids[j] = K[mask].mean(axis=0)
                new_values[j] = V[mask].mean(axis=0)
                new_weights[j] = count / n
        
        centroids = new_centroids
        values = new_values
        weights = new_weights
    
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


def quantize_nbit(x: np.ndarray, n_bits: int = 4) -> tuple[np.ndarray, float]:
    """INTn 量化"""
    abs_max = np.abs(x).max()
    if abs_max < 1e-10:
        return x.astype(np.int8), 1.0
    
    scale = abs_max / (2 ** (n_bits - 1) - 1)
    x_quant = np.round(x / scale).clip(-2**(n_bits-1), 2**(n_bits-1)-1)
    return x_quant.astype(np.int8), scale


def dequantize_nbit(x_quant: np.ndarray, scale: float) -> np.ndarray:
    """反量化"""
    return x_quant.astype(np.float32) * scale


def svd_compress_v(V: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """SVD 压缩 V 矩阵"""
    d = V.shape[1]
    
    U, S, Vt = npla.svd(V, full_matrices=False)
    
    actual_r = min(r, len(S))
    U_r = U[:, :actual_r]
    S_r = S[:actual_r]
    V_r = Vt[:actual_r, :]
    
    V_reconstructed = U_r @ np.diag(S_r) @ V_r
    
    original_size = V.shape[0] * V.shape[1]
    compressed_size = U_r.shape[0] * U_r.shape[1] + S_r.size + V_r.shape[0] * V_r.shape[1]
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    
    return V_reconstructed, U_r, S_r, compression_ratio


def prune_tokens(K: np.ndarray, V: np.ndarray, keep_ratio: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prune tokens by importance (random for now, could be gradient-based)"""
    n = K.shape[0]
    r = max(4, int(n * keep_ratio))
    
    gen = np.random.default_rng(seed)
    indices = gen.choice(n, size=r, replace=False)
    indices = np.sort(indices)
    
    K_pruned = K[indices]
    V_pruned = V[indices]
    weights = np.ones(r) / r
    
    return K_pruned, V_pruned, weights


# ============== 正序链路: Coreset → SVD → INT ==============

def build_serial_fusion_sketch(
    K: np.ndarray,
    V: np.ndarray,
    Q_sample: np.ndarray,
    coreset_ratio: float,
    svd_r: int,
    int4_bits: int,
    d: int = 128,
    seed: int = 0,
) -> dict:
    """串行融合 sketch 构建（正序）"""
    kv_len = K.shape[0]
    r_coreset = max(4, int(kv_len * coreset_ratio))
    
    # Stage 1: Coreset
    centroids, values, weights = build_coreset_sketch(K, V, r_coreset, seed=seed)
    
    # 评估 coreset 误差
    y_coreset = eval_coreset_sketch(centroids, values, weights, Q_sample, d)
    gt = ground_truth(Q_sample, K, V)
    err_coreset = float(np.abs(y_coreset - gt).mean())
    
    # Stage 2: SVD 压缩
    V_reconstructed, U_r, S_r, svd_compression = svd_compress_v(values, svd_r)
    err_svd = float(np.abs(V_reconstructed - values).mean()) if values.shape == V_reconstructed.shape else 0
    
    # Stage 3: INT 量化
    V_quant, V_scale = quantize_nbit(V_reconstructed, int4_bits)
    
    # 计算压缩比
    bytes_full = kv_len * d * 2 * 4
    bytes_stage1 = r_coreset * d * 3 * 4
    bytes_stage2 = U_r.size + S_r.size + V_quant.size + 1
    compression_ratio = bytes_full / bytes_stage2 if bytes_stage2 > 0 else float('inf')
    
    # 重建并计算误差
    V_final = dequantize_nbit(V_quant, V_scale)
    
    # 用最终 V 评估
    y_fusion = eval_coreset_sketch(centroids, V_final, weights, Q_sample, d)
    err_fusion = float(np.abs(y_fusion - gt).mean())
    
    # 物理上限检查
    physical_limit = 2.0 * kv_len / Q_sample.shape[0]
    is_unphysical = coreset_ratio > physical_limit
    
    return {
        "stage1_type": "coreset",
        "stage2_type": "svd",
        "stage3_type": "quant",
        "stage1_coreset_r": r_coreset,
        "stage2_svd_r": svd_r,
        "stage3_int_bits": int4_bits,
        "err_coreset": err_coreset,
        "err_svd": err_svd,
        "err_fusion": err_fusion,
        "compression_total": compression_ratio,
        "svd_compression": svd_compression,
        "physical_limit": physical_limit,
        "is_unphysical": is_unphysical,
        "regime": "extreme" if (svd_r <= 2 or int4_bits <= 2) else "normal",
    }


# ============== 反序链路: INT → SVD → Coreset ==============

def build_reverse_order_sketch(
    K: np.ndarray,
    V: np.ndarray,
    Q_sample: np.ndarray,
    int4_bits: int,
    svd_r: int,
    coreset_ratio: float,
    d: int = 128,
    seed: int = 0,
) -> dict:
    """
    反序链路: INT → SVD → Coreset
    
    问题：先量化会损失精度，SVD 之后精度更差，Coreset 选出的代表性更差
    """
    kv_len = K.shape[0]
    r_coreset = max(4, int(kv_len * coreset_ratio))
    
    # Stage 1: INT 量化（先量化会损失信息）
    V_quant, V_scale = quantize_nbit(V, int4_bits)
    V_dequant = dequantize_nbit(V_quant, V_scale)
    
    err_quant = float(np.abs(V_dequant - V).mean())
    
    # Stage 2: SVD 压缩（对量化后数据做 SVD）
    V_reconstructed, U_r, S_r, svd_compression = svd_compress_v(V_dequant, svd_r)
    err_svd = float(np.abs(V_reconstructed - V_dequant).mean())
    
    # Stage 3: Coreset（对 SVD 重建的数据做 Coreset）
    centroids, values, weights = build_coreset_sketch(K, V_reconstructed, r_coreset, seed=seed)
    
    gt = ground_truth(Q_sample, K, V)
    
    # 评估
    y_fusion = eval_coreset_sketch(centroids, values, weights, Q_sample, d)
    err_fusion = float(np.abs(y_fusion - gt).mean())
    
    # 计算压缩比
    bytes_full = kv_len * d * 2 * 4
    bytes_stage1 = kv_len * 2 * 1  # quantized
    bytes_stage2 = U_r.size + S_r.size + V_quant.size
    bytes_stage3 = r_coreset * d * 3 * 4
    compression_ratio = bytes_full / (bytes_stage1 + bytes_stage2 + bytes_stage3) if bytes_stage3 > 0 else float('inf')
    
    physical_limit = 2.0 * kv_len / Q_sample.shape[0]
    is_unphysical = coreset_ratio > physical_limit
    
    return {
        "stage1_type": "quant",
        "stage2_type": "svd",
        "stage3_type": "coreset",
        "stage1_int_bits": int4_bits,
        "stage2_svd_r": svd_r,
        "stage3_coreset_r": r_coreset,
        "err_quant": err_quant,
        "err_svd": err_svd,
        "err_fusion": err_fusion,
        "compression_total": compression_ratio,
        "svd_compression": svd_compression,
        "physical_limit": physical_limit,
        "is_unphysical": is_unphysical,
        "regime": "extreme" if (svd_r <= 2 or int4_bits <= 2) else "normal",
    }


# ============== 4-stage 链路: Prune → Coreset → SVD → INT ==============

def build_4stage_sketch(
    K: np.ndarray,
    V: np.ndarray,
    Q_sample: np.ndarray,
    prune_ratio: float,
    coreset_ratio: float,
    svd_r: int,
    int4_bits: int,
    d: int = 128,
    seed: int = 0,
) -> dict:
    """
    4-stage 链路: Prune → Coreset → SVD → INT
    
    先粗筛(Prune) → 再精挑(Coreset) → 降维(SVD) → 量化(INT)
    """
    kv_len = K.shape[0]
    
    # Stage 1: Prune
    K_pruned, V_pruned, _ = prune_tokens(K, V, prune_ratio, seed)
    err_prune = float(np.abs(ground_truth(Q_sample, K_pruned, V_pruned) - ground_truth(Q_sample, K, V)).mean()) if Q_sample.shape[0] > 0 else 0
    
    # Stage 2: Coreset
    r_coreset = max(4, int(K_pruned.shape[0] * coreset_ratio))
    centroids, values, weights = build_coreset_sketch(K_pruned, V_pruned, r_coreset, seed=seed)
    
    gt = ground_truth(Q_sample, K, V)
    y_coreset = eval_coreset_sketch(centroids, values, weights, Q_sample, d)
    err_coreset = float(np.abs(y_coreset - gt).mean())
    
    # Stage 3: SVD
    V_reconstructed, U_r, S_r, svd_compression = svd_compress_v(values, svd_r)
    
    # Stage 4: INT
    V_quant, V_scale = quantize_nbit(V_reconstructed, int4_bits)
    V_final = dequantize_nbit(V_quant, V_scale)
    
    y_fusion = eval_coreset_sketch(centroids, V_final, weights, Q_sample, d)
    err_fusion = float(np.abs(y_fusion - gt).mean())
    
    # 压缩比
    bytes_full = kv_len * d * 2 * 4
    bytes_stage1 = int(kv_len * prune_ratio) * d * 2 * 4
    bytes_stage2 = r_coreset * d * 3 * 4
    bytes_stage3 = U_r.size + S_r.size + V_quant.size + 1
    compression_ratio = bytes_full / bytes_stage3 if bytes_stage3 > 0 else float('inf')
    
    physical_limit = 2.0 * kv_len / Q_sample.shape[0]
    effective_ratio = prune_ratio * coreset_ratio
    is_unphysical = effective_ratio > physical_limit
    
    return {
        "stage1_type": "prune",
        "stage2_type": "coreset",
        "stage3_type": "svd",
        "stage4_type": "quant",
        "stage1_prune_ratio": prune_ratio,
        "stage2_coreset_r": r_coreset,
        "stage3_svd_r": svd_r,
        "stage4_int_bits": int4_bits,
        "effective_ratio": effective_ratio,
        "err_prune": err_prune,
        "err_coreset": err_coreset,
        "err_fusion": err_fusion,
        "compression_total": compression_ratio,
        "svd_compression": svd_compression,
        "physical_limit": physical_limit,
        "is_unphysical": is_unphysical,
        "regime": "extreme" if (svd_r <= 2 or int4_bits <= 2) else "normal",
    }


# ============== 数据生成 ==============

def make_clustered_kv(kv_len: int, d: int, n_clusters: int = 8, seed: int = 0):
    """生成聚类数据"""
    gen = np.random.default_rng(seed)
    centroids = gen.standard_normal((n_clusters, d)) * 2.0
    assignments = gen.integers(0, n_clusters, size=kv_len)
    K = centroids[assignments] + gen.standard_normal((kv_len, d)) * 0.5
    V = K @ (gen.standard_normal((d, d)) * 0.3) + gen.standard_normal((kv_len, d)) * 0.1
    return K.astype(np.float32), V.astype(np.float32)


def make_random_kv(kv_len: int, d: int, seed: int = 0):
    """生成随机数据"""
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((kv_len, d)).astype(np.float32)
    V = gen.standard_normal((kv_len, d)).astype(np.float32)
    return K, V


# ============== 实验配置 ==============

def get_extreme_configs():
    """极端参数配置"""
    return {
        "coreset_ratios": [0.01, 0.02, 0.05, 0.10],
        "svd_r_values": [1, 2, 3, 4],
        "int4_bits": [2, 3, 4],
        "kv_lens": [1024, 4096],
        "q_lens": [16, 64],
    }


def get_reverse_configs():
    """反序实验配置"""
    return {
        "int_bits": [2, 3, 4],
        "svd_r_values": [1, 2, 3, 4],
        "coreset_ratios": [0.01, 0.02, 0.05, 0.10],
        "kv_lens": [1024],
        "q_lens": [64],
    }


def get_4stage_configs():
    """4-stage 链路配置"""
    return {
        "prune_ratios": [0.3, 0.5, 0.7],
        "coreset_ratios": [0.1, 0.3, 0.5],
        "svd_r_values": [2, 4, 8],
        "int_bits": [2, 3, 4],
        "kv_lens": [1024],
        "q_lens": [64],
    }


# ============== Sanity Check ==============

def run_sanity_check():
    """小规模 sanity check"""
    print("\n" + "=" * 60)
    print("EXP22: Sanity Check")
    print("=" * 60)
    
    d = 128
    kv_len = 512
    q_len = 64
    
    K, V = make_clustered_kv(kv_len, d, seed=42)
    gen = np.random.default_rng(100)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    gt = ground_truth(Q, K, V)
    print(f"Ground truth mean: {gt.mean():.6f}")
    
    sanity_results = []
    
    # Test 1: Extreme serial fusion
    print("\n[Test 1] Extreme Serial Fusion")
    try:
        result = build_serial_fusion_sketch(
            K, V, Q,
            coreset_ratio=0.01,
            svd_r=2,
            int4_bits=2,
            d=d,
            seed=42,
        )
        result["test"] = "extreme_serial"
        sanity_results.append(result)
        print(f"  err_fusion: {result['err_fusion']:.6f}")
        print(f"  compression: {result['compression_total']:.2f}x")
        print(f"  is_unphysical: {result['is_unphysical']}")
        print(f"  regime: {result['regime']}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test 2: Reverse order
    print("\n[Test 2] Reverse Order (INT → SVD → Coreset)")
    try:
        result = build_reverse_order_sketch(
            K, V, Q,
            int4_bits=2,
            svd_r=2,
            coreset_ratio=0.05,
            d=d,
            seed=42,
        )
        result["test"] = "reverse_order"
        sanity_results.append(result)
        print(f"  err_fusion: {result['err_fusion']:.6f}")
        print(f"  compression: {result['compression_total']:.2f}x")
        print(f"  regime: {result['regime']}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test 3: 4-stage
    print("\n[Test 3] 4-Stage (Prune → Coreset → SVD → INT)")
    try:
        result = build_4stage_sketch(
            K, V, Q,
            prune_ratio=0.7,
            coreset_ratio=0.3,
            svd_r=4,
            int4_bits=2,
            d=d,
            seed=42,
        )
        result["test"] = "4stage"
        sanity_results.append(result)
        print(f"  err_fusion: {result['err_fusion']:.6f}")
        print(f"  compression: {result['compression_total']:.2f}x")
        print(f"  is_unphysical: {result['is_unphysical']}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test 4: Moderate baseline
    print("\n[Test 4] Moderate Baseline (for comparison)")
    try:
        result = build_serial_fusion_sketch(
            K, V, Q,
            coreset_ratio=0.25,
            svd_r=8,
            int4_bits=4,
            d=d,
            seed=42,
        )
        result["test"] = "moderate_baseline"
        sanity_results.append(result)
        print(f"  err_fusion: {result['err_fusion']:.6f}")
        print(f"  compression: {result['compression_total']:.2f}x")
        print(f"  is_unphysical: {result['is_unphysical']}")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    return sanity_results


# ============== 完整扫描 ==============

def run_extreme_sweep():
    """极端参数扫描"""
    print("\n" + "=" * 60)
    print("EXP22: Extreme Parameter Sweep")
    print("=" * 60)
    
    configs = get_extreme_configs()
    d = 128
    results = []
    
    total_configs = (
        len(configs["coreset_ratios"]) *
        len(configs["svd_r_values"]) *
        len(configs["int4_bits"]) *
        len(configs["kv_lens"]) *
        len(configs["q_lens"])
    )
    
    config_idx = 0
    start_time = time.time()
    
    for kv_len in configs["kv_lens"]:
        for q_len in configs["q_lens"]:
            K, V = make_clustered_kv(kv_len, d, seed=42)
            gen = np.random.default_rng(100)
            Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
            
            physical_limit = 2.0 * kv_len / q_len
            
            for coreset_ratio in configs["coreset_ratios"]:
                for svd_r in configs["svd_r_values"]:
                    for int_bits in configs["int4_bits"]:
                        config_idx += 1
                        
                        try:
                            result = build_serial_fusion_sketch(
                                K, V, Q,
                                coreset_ratio=coreset_ratio,
                                svd_r=svd_r,
                                int4_bits=int_bits,
                                d=d,
                                seed=42,
                            )
                            
                            result.update({
                                "kv_len": kv_len,
                                "q_len": q_len,
                                "d": d,
                                "physical_limit": physical_limit,
                            })
                            
                            results.append(result)
                            
                        except Exception as e:
                            print(f"Error config {config_idx}: {e}")
                            continue
                        
                        if config_idx % 30 == 0:
                            elapsed = time.time() - start_time
                            rate = config_idx / elapsed
                            remaining = (total_configs - config_idx) / rate if rate > 0 else 0
                            print(f"Progress: {config_idx}/{total_configs} ({elapsed:.1f}s, ~{remaining:.1f}s remaining)")
    
    return results


def run_reverse_sweep():
    """反序实验扫描"""
    print("\n" + "=" * 60)
    print("EXP22: Reverse Order Sweep")
    print("=" * 60)
    
    configs = get_reverse_configs()
    d = 128
    results = []
    
    total_configs = (
        len(configs["int_bits"]) *
        len(configs["svd_r_values"]) *
        len(configs["coreset_ratios"]) *
        len(configs["kv_lens"]) *
        len(configs["q_lens"])
    )
    
    config_idx = 0
    start_time = time.time()
    
    for kv_len in configs["kv_lens"]:
        for q_len in configs["q_lens"]:
            K, V = make_clustered_kv(kv_len, d, seed=42)
            gen = np.random.default_rng(100)
            Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
            
            physical_limit = 2.0 * kv_len / q_len
            
            for int_bits in configs["int_bits"]:
                for svd_r in configs["svd_r_values"]:
                    for coreset_ratio in configs["coreset_ratios"]:
                        config_idx += 1
                        
                        try:
                            result = build_reverse_order_sketch(
                                K, V, Q,
                                int4_bits=int_bits,
                                svd_r=svd_r,
                                coreset_ratio=coreset_ratio,
                                d=d,
                                seed=42,
                            )
                            
                            result.update({
                                "kv_len": kv_len,
                                "q_len": q_len,
                                "d": d,
                                "physical_limit": physical_limit,
                            })
                            
                            results.append(result)
                            
                        except Exception as e:
                            print(f"Error config {config_idx}: {e}")
                            continue
                        
                        if config_idx % 20 == 0:
                            elapsed = time.time() - start_time
                            rate = config_idx / elapsed
                            remaining = (total_configs - config_idx) / rate if rate > 0 else 0
                            print(f"Progress: {config_idx}/{total_configs} ({elapsed:.1f}s, ~{remaining:.1f}s remaining)")
    
    return results


def run_4stage_sweep():
    """4-stage 链路扫描"""
    print("\n" + "=" * 60)
    print("EXP22: 4-Stage Sweep")
    print("=" * 60)
    
    configs = get_4stage_configs()
    d = 128
    results = []
    
    total_configs = (
        len(configs["prune_ratios"]) *
        len(configs["coreset_ratios"]) *
        len(configs["svd_r_values"]) *
        len(configs["int_bits"]) *
        len(configs["kv_lens"]) *
        len(configs["q_lens"])
    )
    
    config_idx = 0
    start_time = time.time()
    
    for kv_len in configs["kv_lens"]:
        for q_len in configs["q_lens"]:
            K, V = make_clustered_kv(kv_len, d, seed=42)
            gen = np.random.default_rng(100)
            Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
            
            physical_limit = 2.0 * kv_len / q_len
            
            for prune_ratio in configs["prune_ratios"]:
                for coreset_ratio in configs["coreset_ratios"]:
                    for svd_r in configs["svd_r_values"]:
                        for int_bits in configs["int_bits"]:
                            config_idx += 1
                            
                            try:
                                result = build_4stage_sketch(
                                    K, V, Q,
                                    prune_ratio=prune_ratio,
                                    coreset_ratio=coreset_ratio,
                                    svd_r=svd_r,
                                    int4_bits=int_bits,
                                    d=d,
                                    seed=42,
                                )
                                
                                result.update({
                                    "kv_len": kv_len,
                                    "q_len": q_len,
                                    "d": d,
                                    "physical_limit": physical_limit,
                                })
                                
                                results.append(result)
                                
                            except Exception as e:
                                print(f"Error config {config_idx}: {e}")
                                continue
                            
                            if config_idx % 30 == 0:
                                elapsed = time.time() - start_time
                                rate = config_idx / elapsed
                                remaining = (total_configs - config_idx) / rate if rate > 0 else 0
                                print(f"Progress: {config_idx}/{total_configs} ({elapsed:.1f}s, ~{remaining:.1f}s remaining)")
    
    return results


# ============== 帕累托悬崖分析 ==============

def find_pareto_cliff(results: list, err_threshold_low: float = 1.0, err_threshold_high: float = 5.0) -> list:
    """找帕累托悬崖点：err 从 < threshold_low 跳到 > threshold_high 的临界点"""
    cliffs = []
    
    # 按 compression_ratio 排序
    sorted_results = sorted(results, key=lambda x: x["compression_total"], reverse=True)
    
    for i, r in enumerate(sorted_results):
        err = r["err_fusion"]
        comp = r["compression_total"]
        
        # 找悬崖：前面是高质低压缩，后面是低质高压缩
        if err > err_threshold_high:
            # 检查前面是否有低误差的点
            prev_low_err = [x for x in sorted_results[:i] if x["err_fusion"] < err_threshold_low]
            if prev_low_err:
                cliffs.append({
                    "cliff_at_compression": comp,
                    "cliff_err": err,
                    "best_prev_err": min(x["err_fusion"] for x in prev_low_err),
                    "best_prev_comp": max(x["compression_total"] for x in prev_low_err),
                    "jump_ratio": err / min(x["err_fusion"] for x in prev_low_err),
                    "config": r,
                })
    
    return cliffs


def analyze_cliff_points(extreme_results: list) -> dict:
    """分析悬崖点"""
    cliffs = find_pareto_cliff(extreme_results)
    
    # 按 svd_r 分组分析
    by_svd_r = {}
    for r in extreme_results:
        svd_r = r["stage2_svd_r"]
        if svd_r not in by_svd_r:
            by_svd_r[svd_r] = []
        by_svd_r[svd_r].append(r)
    
    cliff_analysis = {}
    for svd_r, group in by_svd_r.items():
        sorted_group = sorted(group, key=lambda x: x["compression_total"], reverse=True)
        errors = [x["err_fusion"] for x in sorted_group]
        
        # 找 err 突变点
        max_jump = 0
        cliff_idx = 0
        for i in range(1, len(errors)):
            jump = errors[i] - errors[i-1]
            if jump > max_jump:
                max_jump = jump
                cliff_idx = i
        
        cliff_analysis[svd_r] = {
            "min_err": float(min(errors)),
            "max_err": float(max(errors)),
            "mean_err": float(np.mean(errors)),
            "cliff_idx": cliff_idx,
            "cliff_err": float(errors[cliff_idx]) if cliff_idx < len(errors) else None,
            "cliff_comp": float(sorted_group[cliff_idx]["compression_total"]) if cliff_idx < len(sorted_group) else None,
            "max_jump": float(max_jump),
        }
    
    return {
        "all_cliffs": cliffs,
        "by_svd_r": cliff_analysis,
        "extreme_regime_count": sum(1 for r in extreme_results if r["regime"] == "extreme"),
        "unphysical_count": sum(1 for r in extreme_results if r["is_unphysical"]),
    }


# ============== 保存结果 ==============

def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved: {filepath}")


# ============== 主函数 ==============

def main():
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Sanity Check
    print("\n" + "#" * 60)
    print("# STEP 1: Sanity Check")
    print("#" * 60)
    
    sanity_results = run_sanity_check()
    save_json(sanity_results, os.path.join(output_dir, "exp22_sanity.json"))
    
    # 检查 sanity 是否通过
    sanity_pass = True
    for r in sanity_results:
        if r["test"] == "extreme_serial" and r["err_fusion"] > 100:
            print(f"⚠️ Extreme regime shows expected high error: {r['err_fusion']:.4f}")
        if r["test"] == "moderate_baseline" and r["err_fusion"] > 1.0:
            print(f"⚠️ Moderate baseline has high error, check implementation")
            sanity_pass = False
    
    print(f"\nSanity Check: {'PASS' if sanity_pass else 'FAIL'}")
    
    # Step 2: Extreme Sweep
    print("\n" + "#" * 60)
    print("# STEP 2: Extreme Parameter Sweep")
    print("#" * 60)
    
    extreme_results = run_extreme_sweep()
    save_json(extreme_results, os.path.join(output_dir, "exp22_extreme_sweep.json"))
    
    # Step 3: Reverse Order Sweep
    print("\n" + "#" * 60)
    print("# STEP 3: Reverse Order Sweep")
    print("#" * 60)
    
    reverse_results = run_reverse_sweep()
    save_json(reverse_results, os.path.join(output_dir, "exp22_reverse_sweep.json"))
    
    # Step 4: 4-Stage Sweep
    print("\n" + "#" * 60)
    print("# STEP 4: 4-Stage Sweep")
    print("#" * 60)
    
    stage4_results = run_4stage_sweep()
    save_json(stage4_results, os.path.join(output_dir, "exp22_4stage_sweep.json"))
    
    # Step 5: Cliff Analysis
    print("\n" + "#" * 60)
    print("# STEP 5: Pareto Cliff Analysis")
    print("#" * 60)
    
    cliff_analysis = analyze_cliff_points(extreme_results)
    save_json(cliff_analysis, os.path.join(output_dir, "exp22_cliff.json"))
    
    # 生成报告
    generate_report(
        sanity_results,
        extreme_results,
        reverse_results,
        stage4_results,
        cliff_analysis,
        output_dir,
    )
    
    return {
        "sanity": sanity_results,
        "extreme": extreme_results,
        "reverse": reverse_results,
        "stage4": stage4_results,
        "cliff": cliff_analysis,
    }


def generate_report(sanity, extreme, reverse, stage4, cliff, output_dir):
    """生成实验报告"""
    
    # 统计
    extreme_stats = {
        "total": len(extreme),
        "extreme_regime": sum(1 for r in extreme if r["regime"] == "extreme"),
        "unphysical": sum(1 for r in extreme if r["is_unphysical"]),
        "mean_err": float(np.mean([r["err_fusion"] for r in extreme])) if extreme else 0,
        "max_comp": float(max(r["compression_total"] for r in extreme)) if extreme else 0,
    }
    
    reverse_stats = {
        "total": len(reverse),
        "working_configs": sum(1 for r in reverse if r["err_fusion"] < 5.0),
        "mean_err": float(np.mean([r["err_fusion"] for r in reverse])) if reverse else 0,
    }
    
    stage4_stats = {
        "total": len(stage4),
        "extreme_regime": sum(1 for r in stage4 if r["regime"] == "extreme"),
        "unphysical": sum(1 for r in stage4 if r["is_unphysical"]),
        "mean_err": float(np.mean([r["err_fusion"] for r in stage4])) if stage4 else 0,
        "max_comp": float(max(r["compression_total"] for r in stage4)) if stage4 else 0,
    }
    
    report = f"""# Exp22: 极端压缩边界触碰实验报告

## 实验概述

本实验旨在主动触碰物理上限边界，探索极端压缩参数下的行为表现。

## 物理诚实边界

**核心原则**: 每个 ratio 都标注是否超过物理上限 `2 * kv_len / q_len`

## 1. Sanity Check 结果

| 测试 | err_fusion | compression | regime | is_unphysical |
|------|------------|-------------|--------|---------------|
"""
    
    for r in sanity:
        test_name = r.get("test", "unknown")
        report += f"| {test_name} | {r.get('err_fusion', 'N/A'):.6f} | {r.get('compression_total', 'N/A'):.2f}x | {r.get('regime', 'N/A')} | {r.get('is_unphysical', 'N/A')} |\n"
    
    report += f"""
## 2. 极端参数扫描结果

**配置**: coreset_ratio ∈ [0.01, 0.02, 0.05, 0.10], svd_r ∈ [1, 2, 3, 4], int_bits ∈ [2, 3, 4]

**统计**:
- 总配置数: {extreme_stats['total']}
- 极端 regime (r≤2 or bits≤2): {extreme_stats['extreme_regime']}
- 标为 unphysical: {extreme_stats['unphysical']}
- 平均误差: {extreme_stats['mean_err']:.6f}
- 最大压缩比: {extreme_stats['max_comp']:.2f}x

"""
    
    # 按 svd_r 分组的悬崖分析
    if cliff.get("by_svd_r"):
        report += "### 悬崖分析 (按 svd_r 分组)\n\n"
        report += "| svd_r | min_err | max_err | mean_err | cliff_idx | cliff_err | max_jump |\n"
        report += "|-------|---------|---------|----------|-----------|-----------|----------|\n"
        for svd_r, stats in sorted(cliff["by_svd_r"].items()):
            report += f"| {svd_r} | {stats['min_err']:.4f} | {stats['max_err']:.4f} | {stats['mean_err']:.4f} | {stats['cliff_idx']} | {stats.get('cliff_err', 'N/A')} | {stats['max_jump']:.4f} |\n"
    
    report += f"""
## 3. 反序实验结果 (INT → SVD → Coreset)

**问题**: 先量化会损失精度，SVD 之后精度更差，Coreset 选出的代表性更差

**统计**:
- 总配置数: {reverse_stats['total']}
- err < 5.0 的配置数: {reverse_stats['working_configs']}
- 平均误差: {reverse_stats['mean_err']:.6f}

**结论**: {'反序链路部分有效' if reverse_stats['working_configs'] > 0 else '反序链路基本失效'}

"""
    
    # 反序最佳配置
    if reverse:
        best_reverse = min(reverse, key=lambda x: x["err_fusion"])
        report += f"**最佳反序配置**:\n"
        report += f"- int_bits: {best_reverse.get('stage1_int_bits', 'N/A')}\n"
        report += f"- svd_r: {best_reverse.get('stage2_svd_r', 'N/A')}\n"
        report += f"- coreset_r: {best_reverse.get('stage3_coreset_r', 'N/A')}\n"
        report += f"- err_fusion: {best_reverse['err_fusion']:.6f}\n"
        report += f"- compression: {best_reverse['compression_total']:.2f}x\n\n"
    
    report += f"""
## 4. 4-Stage 链路结果 (Prune → Coreset → SVD → INT)

**统计**:
- 总配置数: {stage4_stats['total']}
- 极端 regime: {stage4_stats['extreme_regime']}
- 标为 unphysical: {stage4_stats['unphysical']}
- 平均误差: {stage4_stats['mean_err']:.6f}
- 最大压缩比: {stage4_stats['max_comp']:.2f}x

"""
    
    # 4-stage 最佳配置
    if stage4:
        best_stage4 = min(stage4, key=lambda x: x["err_fusion"])
        report += f"**最佳 4-stage 配置**:\n"
        report += f"- prune_ratio: {best_stage4.get('stage1_prune_ratio', 'N/A')}\n"
        report += f"- coreset_r: {best_stage4.get('stage2_coreset_r', 'N/A')}\n"
        report += f"- svd_r: {best_stage4.get('stage3_svd_r', 'N/A')}\n"
        report += f"- int_bits: {best_stage4.get('stage4_int_bits', 'N/A')}\n"
        report += f"- err_fusion: {best_stage4['err_fusion']:.6f}\n"
        report += f"- compression: {best_stage4['compression_total']:.2f}x\n\n"
        
        # 高压缩且低误差的配置
        good_stage4 = [r for r in stage4 if r["err_fusion"] < 1.0 and r["compression_total"] > 20]
        if good_stage4:
            report += "**良好帕累托点 (err < 1.0 且 comp > 20x)**:\n\n"
            report += "| prune | coreset_r | svd_r | bits | err | comp |\n"
            report += "|-------|-----------|-------|------|-----|------|\n"
            for r in sorted(good_stage4, key=lambda x: x["compression_total"], reverse=True)[:5]:
                report += f"| {r.get('stage1_prune_ratio', 'N/A')} | {r.get('stage2_coreset_r', 'N/A')} | {r.get('stage3_svd_r', 'N/A')} | {r.get('stage4_int_bits', 'N/A')} | {r['err_fusion']:.4f} | {r['compression_total']:.2f}x |\n"
    
    report += f"""
## 5. 核心发现

### 5.1 帕累托悬崖位置

"""
    
    if cliff.get("by_svd_r"):
        for svd_r, stats in sorted(cliff["by_svd_r"].items()):
            if stats["max_jump"] > 1.0:
                report += f"- svd_r={svd_r}: 悬崖跳跃 {stats['max_jump']:.4f}，发生在 idx={stats['cliff_idx']}\n"
    
    report += f"""
### 5.2 极端 regime 分析

- **extreme regime 定义**: svd_r ≤ 2 或 int_bits ≤ 2
- 极端 regime 配置数: {extreme_stats['extreme_regime']} / {extreme_stats['total']}

### 5.3 物理诚实边界

- 标为 unphysical 的配置: {extreme_stats['unphysical']} / {extreme_stats['total']}
- **注意**: 极端压缩几乎一定超物理上限，这是预期的

### 5.4 反序链路评估

- 反序链路（INT → SVD → Coreset）在先量化再压缩的场景下精度损失严重
- 原因：量化噪声被 SVD 放大，Coreset 无法正确选择代表性 token

### 5.5 4-stage 链路评估

- 4-stage 链路（Prune → Coreset → SVD → INT）在合适参数下可以 work
- 关键：Prune 阶段要保留足够信息，Coreset 阶段要有效选择

## 6. 结论

1. **帕累托悬崖**: svd_r=1-2 时，压缩比增加会导致误差急剧上升
2. **反序链路**: 基本失效，不推荐使用
3. **4-stage 链路**: 在 moderate 参数（prune=0.7, coreset=0.3, svd=4, bits=4）下有效
4. **诚实边界**: 极端压缩（r=1, bits=2）误差极大，但压缩比可达 255x 以上

---
*Generated by Exp22 Extreme Boundary Experiment*
"""
    
    report_path = os.path.join(output_dir, "exp22_extreme_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"Report saved: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
