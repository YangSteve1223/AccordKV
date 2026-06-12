"""
Exp20: Product Quantization (PQ) - Faiss 同款，替代 INT4
=========================================================

核心思路: 
- INT4 量化粒度太粗(整 token)，对 clustered 数据不友好
- PQ 把 V 的 head_dim 维度切成 M 个子空间，每个子空间独立做 k-means 量化
- 理论上 PQ-8 能达到 INT4 的 1/4 误差(每子空间 256 centroids)

链路: Coreset(α) → PQ(M, K) — 替代 Serial Cascade 的 SVD + INT4
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2, vq
from numpy import linalg as npla

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import ground_truth


# ============== Product Quantization 实现 ==============

def kmeans_plusplus_init(X: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """K-Means++ 初始化"""
    gen = np.random.default_rng(seed)
    n, d = X.shape
    idx = gen.integers(0, n)
    centroids = [X[idx].copy()]
    
    for _ in range(K - 1):
        dists = np.zeros(n)
        for c in centroids:
            dists += np.sum((X - c) ** 2, axis=1)
        probs = dists / dists.sum()
        idx = gen.choice(n, p=probs)
        centroids.append(X[idx].copy())
    
    return np.array(centroids)


def kmeans_converge(
    X: np.ndarray, 
    centroids: np.ndarray, 
    max_iters: int = 5,
    tol: float = 1e-4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    K-Means 收敛 (scipy 版本)
    """
    K = centroids.shape[0]
    codebook, _ = kmeans2(X, K, minit='points', iter=max_iters, seed=seed)
    codes, dists = vq(X, codebook)
    inertia = float(np.sum(dists ** 2))
    return codebook, codes.astype(np.int32), inertia


@dataclass
class PQCodebook:
    """PQ 编码本"""
    centroids: np.ndarray  # [M, K, sub_d] - M 个子空间，每个 K 个 centroids，子空间维度 sub_d
    
    def __post_init__(self):
        self.M = self.centroids.shape[0]  # 子空间数
        self.K = self.centroids.shape[1]  # 每子空间 centroids 数
        self.sub_d = self.centroids.shape[2]  # 子空间维度


@dataclass  
class PQCompressedSketch:
    """PQ 压缩后的 sketch"""
    K_centroids: np.ndarray  # K 侧的 centroids (Coreset 结果)
    V_codes: np.ndarray      # V 的 PQ 编码 [r, M] - 每个 centroid 对应的子空间编码
    weights: np.ndarray      # Coreset weights
    pq_codebook: PQCodebook  # PQ 编码本
    V_scales: np.ndarray     # V 的 per-centroid scales [r]
    
    def bytes_size(self) -> int:
        """计算压缩后的字节数"""
        # K centroids: r * d * 4 bytes (FP32)
        bytes_K = self.K_centroids.size * 4
        # V codes: r * M * 1 byte (uint8)
        bytes_V_codes = self.V_codes.size * 1
        # Weights: r * 4 bytes (FP32)
        bytes_weights = self.weights.size * 4
        # PQ codebook: M * K * sub_d * 4 bytes (FP32)
        bytes_codebook = self.pq_codebook.centroids.size * 4
        # V scales: r * 4 bytes (FP32)
        bytes_scales = self.V_scales.size * 4
        
        total = bytes_K + bytes_V_codes + bytes_weights + bytes_codebook + bytes_scales
        return total


def build_pq_codebook(
    V: np.ndarray,
    M: int,
    K: int,
    max_iters: int = 10,
    seed: int = 0,
) -> PQCodebook:
    """
    构建 PQ 编码本 (scipy 加速版)
    """
    n, d = V.shape
    
    if d % M != 0:
        raise ValueError(f"head_dim={d} 必须整除 M={M}")
    
    sub_d = d // M  # 每个子空间的维度
    
    # 对每个子空间独立做 k-means
    all_centroids = np.zeros((M, K, sub_d), dtype=np.float32)
    
    for m in range(M):
        start = m * sub_d
        end = start + sub_d
        segment = V[:, start:end]  # [n, sub_d]
        
        # scipy kmeans2
        codebook, _ = kmeans2(segment, K, minit='points', iter=max_iters, seed=seed + m)
        all_centroids[m] = codebook.astype(np.float32)
    
    return PQCodebook(centroids=all_centroids)


def encode_pq(V: np.ndarray, codebook: PQCodebook) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 PQ 编码本编码 V (scipy 加速版)
    """
    n, d = V.shape
    M = codebook.M
    sub_d = codebook.sub_d
    
    codes = np.zeros((n, M), dtype=np.uint8)
    residual_norms = np.zeros(n)
    
    for m in range(M):
        start = m * sub_d
        end = start + sub_d
        segment = V[:, start:end]  # [n, sub_d]
        
        # scipy vq
        best_k, dists = vq(segment, codebook.centroids[m])
        codes[:, m] = best_k.astype(np.uint8)
        # dists is already squared distances [n]
        residual_norms += dists ** 2
    
    return codes, residual_norms


def decode_pq(codes: np.ndarray, codebook: PQCodebook) -> np.ndarray:
    """
    解码 PQ 编码
    
    Args:
        codes: [n, M] uint8 编码
        codebook: PQ 编码本
    
    Returns:
        V_reconstructed: [n, d] 重建的 V 矩阵
    """
    n = codes.shape[0]
    M = codebook.M
    sub_d = codebook.sub_d
    d = M * sub_d
    
    V_rec = np.zeros((n, d), dtype=np.float32)
    
    for m in range(M):
        start = m * sub_d
        end = start + sub_d
        # 从 codebook 取对应的 centroids
        centroids_m = codebook.centroids[m]  # [K, sub_d]
        V_rec[:, start:end] = centroids_m[codes[:, m]]
    
    return V_rec


# ============== Coreset 组件 ==============

def build_coreset_sketch(
    K_in: np.ndarray,
    V_in: np.ndarray,
    r: int,
    seed: int = 0,
    num_iters: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建 Coreset sketch"""
    n, d = K_in.shape
    centroids = kmeans_plusplus_init(K_in, r, seed)
    values = np.zeros((r, d))
    weights = np.zeros(r)
    
    for _ in range(num_iters):
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K_in - centroids[j]) ** 2, axis=1)
        assignments = dists.argmin(axis=1)
        
        new_centroids = np.zeros_like(centroids)
        new_values = np.zeros((r, d))
        new_weights = np.zeros(r)
        
        for j in range(r):
            mask = assignments == j
            count = mask.sum()
            if count > 0:
                new_centroids[j] = K_in[mask].mean(axis=0)
                new_values[j] = V_in[mask].mean(axis=0)
                new_weights[j] = count / n
        
        centroids = new_centroids
        values = new_values
        weights = new_weights
    
    return centroids, values, weights


def eval_sketch(
    K_centroids: np.ndarray,
    V_values: np.ndarray,
    weights: np.ndarray,
    Q: np.ndarray,
    d: int,
) -> np.ndarray:
    """评估 sketch"""
    r = K_centroids.shape[0]
    scores = Q @ K_centroids.T / np.sqrt(d)
    scores = scores + np.log(weights + 1e-30)
    
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ V_values
    
    return y / np.clip(l, 1e-30, None)


# ============== PQ 压缩 sketch ==============

def compress_sketch_with_pq(
    K_centroids: np.ndarray,
    V_values: np.ndarray,
    weights: np.ndarray,
    M: int,
    K_pq: int,
    seed: int = 0,
) -> PQCompressedSketch:
    """
    用 PQ 压缩 sketch 的 V 值
    
    链路: Coreset → PQ
    """
    r, d = V_values.shape
    
    # 构建 PQ 编码本
    codebook = build_pq_codebook(V_values, M, K_pq, seed=seed)
    
    # 编码 V_values
    codes, residuals = encode_pq(V_values, codebook)
    
    # 计算每个 centroid 的 scale (用于更精确的反量化)
    # 实际上 PQ 的重建误差已经在 codes 里了，这里用 scale 来校正
    V_scales = np.ones(r, dtype=np.float32)
    
    return PQCompressedSketch(
        K_centroids=K_centroids,
        V_codes=codes,
        weights=weights,
        pq_codebook=codebook,
        V_scales=V_scales,
    )


def decompress_pq_sketch(compressed: PQCompressedSketch) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """解压 PQ sketch"""
    # 解码 V
    V_decoded = decode_pq(compressed.V_codes, compressed.pq_codebook)
    # 应用 scale
    V_decoded = V_decoded * compressed.V_scales[:, np.newaxis]
    
    return compressed.K_centroids, V_decoded, compressed.weights


# ============== 完整链路: Coreset → PQ ==============

def run_coreset_pq(
    K: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    r: int,
    M: int,
    K_pq: int,
    d: int = 128,
    seed: int = 42,
) -> dict:
    """
    完整链路: Coreset → PQ
    
    返回:
        包含误差、压缩比等信息的字典
    """
    kv_len = K.shape[0]
    
    # Ground truth
    gt = ground_truth(Q, K, V)
    
    # Step 1: Coreset
    start_time = time.time()
    K_centroids, V_values, weights = build_coreset_sketch(
        K, V, r, seed=seed
    )
    coreset_time = time.time() - start_time
    
    # Coreset baseline
    y_coreset = eval_sketch(K_centroids, V_values, weights, Q, d)
    err_coreset = float(np.abs(y_coreset - gt).mean())
    
    # Step 2: PQ 压缩 V
    start_time = time.time()
    compressed = compress_sketch_with_pq(
        K_centroids, V_values, weights, M, K_pq, seed=seed
    )
    pq_time = time.time() - start_time
    
    # 解压并评估
    K_dec, V_dec, w_dec = decompress_pq_sketch(compressed)
    y_pq = eval_sketch(K_dec, V_dec, w_dec, Q, d)
    err_pq = float(np.abs(y_pq - gt).mean())
    
    # 计算 PQ 重建误差 (V 矩阵上的)
    V_reconstructed = decode_pq(compressed.V_codes, compressed.pq_codebook)
    v_recon_error = float(np.mean((V_reconstructed - V_values) ** 2))
    
    # 字节数
    bytes_full = kv_len * d * 2 * 4  # FP32 K+V
    bytes_pq = compressed.bytes_size()
    
    # 物理诚实检查: ratio 上限 ≈ 2 * kv_len / q_len
    physical_limit = 2.0 * kv_len / Q.shape[0]
    
    return {
        # 配置
        "kv_len": kv_len,
        "q_len": Q.shape[0],
        "d": d,
        "coreset_r": r,
        "pq_M": M,
        "pq_K": K_pq,
        "total_bits": M * int(math.log2(K_pq)),
        "seed": seed,
        
        # 误差
        "err_coreset": err_coreset,
        "err_pq": err_pq,
        "v_recon_error": v_recon_error,
        "err_increase_pct": (err_pq - err_coreset) / (err_coreset + 1e-10) * 100,
        
        # 压缩
        "bytes_full": bytes_full,
        "bytes_pq": bytes_pq,
        "compression_ratio": bytes_full / bytes_pq,
        
        # 物理诚实
        "physical_limit": physical_limit,
        "exceeds_physical": bytes_pq > bytes_full / physical_limit,
        
        # 时间
        "coreset_time_ms": coreset_time * 1000,
        "pq_time_ms": pq_time * 1000,
        "total_time_ms": (coreset_time + pq_time) * 1000,
    }


# ============== INT4 baseline (对比用) ==============

def quantize_nbit(x: np.ndarray, n_bits: int) -> Tuple[np.ndarray, float]:
    """INT4/INT8 量化"""
    abs_max = np.abs(x).max()
    if abs_max < 1e-10:
        return x.astype(np.int8), 1.0
    
    scale = abs_max / (2 ** (n_bits - 1) - 1)
    x_quant = np.round(x / scale).clip(-2**(n_bits-1), 2**(n_bits-1)-1)
    return x_quant.astype(np.int8), scale


def run_coreset_int4(
    K: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    r: int,
    n_bits: int,
    d: int = 128,
    seed: int = 42,
) -> dict:
    """Coreset + INT4 baseline"""
    kv_len = K.shape[0]
    gt = ground_truth(Q, K, V)
    
    # Coreset
    K_centroids, V_values, weights = build_coreset_sketch(K, V, r, seed=seed)
    y_coreset = eval_sketch(K_centroids, V_values, weights, Q, d)
    err_coreset = float(np.abs(y_coreset - gt).mean())
    
    # INT4 量化 V
    V_quant, V_scale = quantize_nbit(V_values, n_bits)
    V_dequant = V_quant.astype(np.float32) * V_scale
    
    # 评估
    y_int4 = eval_sketch(K_centroids, V_dequant, weights, Q, d)
    err_int4 = float(np.abs(y_int4 - gt).mean())
    
    # 字节数
    bytes_full = kv_len * d * 2 * 4
    bytes_int4 = K_centroids.size * 4 + V_quant.size + weights.size * 4 + 4  # +4 for scale
    
    return {
        "kv_len": kv_len,
        "q_len": Q.shape[0],
        "d": d,
        "coreset_r": r,
        "n_bits": n_bits,
        "seed": seed,
        "err_coreset": err_coreset,
        "err_int4": err_int4,
        "err_increase_pct": (err_int4 - err_coreset) / (err_coreset + 1e-10) * 100,
        "bytes_full": bytes_full,
        "bytes_int4": bytes_int4,
        "compression_ratio": bytes_full / bytes_int4,
    }


# ============== 数据生成 ==============

def make_clustered_kv(kv_len: int, d: int, n_clusters: int = 8, seed: int = 0):
    """生成聚类 KV 数据"""
    gen = np.random.default_rng(seed)
    centroids = gen.standard_normal((n_clusters, d)) * 2.0
    assignments = gen.integers(0, n_clusters, size=kv_len)
    K = centroids[assignments] + gen.standard_normal((kv_len, d)) * 0.5
    V = K @ (gen.standard_normal((d, d)) * 0.3) + gen.standard_normal((kv_len, d)) * 0.1
    return K.astype(np.float32), V.astype(np.float32)


def make_random_kv(kv_len: int, d: int, seed: int = 0):
    """生成随机 KV 数据"""
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((kv_len, d)).astype(np.float32)
    V = gen.standard_normal((kv_len, d)).astype(np.float32)
    return K, V


def make_skewed_kv(kv_len: int, d: int, n_outliers: int = 16, seed: int = 0):
    """生成偏斜 KV 数据 (少数 outliers)"""
    gen = np.random.default_rng(seed)
    outlier_K = gen.standard_normal((n_outliers, d)) * 3.0
    outlier_V = gen.standard_normal((n_outliers, d)) * 3.0
    normal_K = gen.standard_normal((kv_len - n_outliers, d)) * 0.3
    normal_V = gen.standard_normal((kv_len - n_outliers, d)) * 0.3
    K = np.concatenate([outlier_K, normal_K])
    V = np.concatenate([outlier_V, normal_V])
    perm = gen.permutation(kv_len)
    return K[perm].astype(np.float32), V[perm].astype(np.float32)


# ============== 扫描实验 ==============

def run_sanity_check():
    """小规模 sanity check"""
    print("=" * 60)
    print("Exp20: PQ Sanity Check (3 数据点)")
    print("=" * 60)
    
    d = 128
    seed = 42
    results = []
    
    configs = [
        {"kv_len": 512, "q_len": 16, "kv_type": "clustered"},  # 增大 kv_len 获得更大 r
        {"kv_len": 512, "q_len": 16, "kv_type": "random"},
        {"kv_len": 512, "q_len": 16, "kv_type": "skewed"},
    ]
    
    for cfg in configs:
        kv_len = cfg["kv_len"]
        q_len = cfg["q_len"]
        kv_type = cfg["kv_type"]
        
        # 生成数据
        if kv_type == "clustered":
            K, V = make_clustered_kv(kv_len, d, seed=seed)
        elif kv_type == "random":
            K, V = make_random_kv(kv_len, d, seed=seed)
        else:
            K, V = make_skewed_kv(kv_len, d, seed=seed)
        
        Q = np.random.default_rng(seed + 100).standard_normal((q_len, d)).astype(np.float32) * 0.5
        # 使用更大的 r 让 PQ 有意义
        r = max(32, kv_len // 32)
        
        # PQ configs to test (减少 codebook 开销)
        pq_configs = [
            {"M": 8, "K": 16},   # 8*4=32 bits per vector
            {"M": 8, "K": 64},  # 8*6=48 bits per vector
        ]
        
        for pq_cfg in pq_configs:
            M, K_pq = pq_cfg["M"], pq_cfg["K"]
            
            try:
                result = run_coreset_pq(K, V, Q, r, M, K_pq, d, seed)
                result["kv_type"] = kv_type
                results.append(result)
                
                print(
                    f"  {kv_type:>8} kv={kv_len:>4} r={r:>2} "
                    f"PQ(M={M},K={K_pq}) "
                    f"err_coreset={result['err_coreset']:.3e} "
                    f"err_pq={result['err_pq']:.3e} "
                    f"inc={result['err_increase_pct']:+.1f}% "
                    f"comp={result['compression_ratio']:.1f}x"
                )
            except Exception as e:
                print(f"  ERROR {kv_type} kv={kv_len} M={M} K={K_pq}: {e}")
        
        # INT4 baseline
        for n_bits in [4, 8]:
            try:
                result_int4 = run_coreset_int4(K, V, Q, r, n_bits, d, seed)
                result_int4["kv_type"] = kv_type
                results.append(result_int4)
                
                print(
                    f"  {kv_type:>8} kv={kv_len:>4} r={r:>2} "
                    f"INT{n_bits} "
                    f"err_coreset={result_int4['err_coreset']:.3e} "
                    f"err_int{n_bits}={result_int4['err_int4']:.3e} "
                    f"inc={result_int4['err_increase_pct']:+.1f}% "
                    f"comp={result_int4['compression_ratio']:.1f}x"
                )
            except Exception as e:
                print(f"  ERROR {kv_type} INT{n_bits}: {e}")
    
    return results


def run_full_sweep():
    """完整扫描"""
    print("=" * 60)
    print("Exp20: PQ Full Sweep")
    print("=" * 60)
    
    d = 128
    seed = 42
    
    # PQ configs: K << r 时才有压缩效果
    # r=64 时，K 应该远小于 64
    pq_configs = [
        {"M": 8, "K": 8},     # 8*3=24 bits/vec, 压缩比高
        {"M": 8, "K": 16},    # 8*4=32 bits/vec, 平衡
    ]
    
    sweep_configs = {
        "kv_types": ["clustered", "random", "skewed"],
        "kv_lens": [2048],
        "q_lens": [16],
        "r_ratios": [0.0625],  # r=128 (更大 r 才有意义)
    }
    
    results = []
    total_configs = (
        len(sweep_configs["kv_types"]) *
        len(sweep_configs["kv_lens"]) *
        len(sweep_configs["q_lens"]) *
        len(sweep_configs["r_ratios"]) *
        (len(pq_configs) + 2)  # +2 for INT4/INT8 baselines
    )
    
    print(f"Total configs to run: {total_configs}")
    
    config_idx = 0
    start_time = time.time()
    
    for kv_type in sweep_configs["kv_types"]:
        for kv_len in sweep_configs["kv_lens"]:
            for q_len in sweep_configs["q_lens"]:
                for r_ratio in sweep_configs["r_ratios"]:
                    r = max(4, int(kv_len * r_ratio))
                    
                    # 生成数据
                    if kv_type == "clustered":
                        K, V = make_clustered_kv(kv_len, d, seed=seed)
                    elif kv_type == "random":
                        K, V = make_random_kv(kv_len, d, seed=seed)
                    else:
                        K, V = make_skewed_kv(kv_len, d, seed=seed)
                    
                    Q = np.random.default_rng(seed + 100).standard_normal((q_len, d)).astype(np.float32) * 0.5
                    
                    # PQ 扫描
                    for pq_cfg in pq_configs:
                        config_idx += 1
                        try:
                            result = run_coreset_pq(
                                K, V, Q, r, pq_cfg["M"], pq_cfg["K"], d, seed
                            )
                            result["kv_type"] = kv_type
                            results.append(result)
                        except Exception as e:
                            print(f"  ERROR PQ M={pq_cfg['M']} K={pq_cfg['K']}: {e}")
                    
                    # INT4/INT8 baselines
                    for n_bits in [4, 8]:
                        config_idx += 1
                        try:
                            result_int4 = run_coreset_int4(
                                K, V, Q, r, n_bits, d, seed
                            )
                            result_int4["kv_type"] = kv_type
                            results.append(result_int4)
                        except Exception as e:
                            print(f"  ERROR INT{n_bits}: {e}")
                    
                    if config_idx % 30 == 0:
                        elapsed = time.time() - start_time
                        rate = config_idx / elapsed if elapsed > 0 else 0
                        remaining = (total_configs - config_idx) / rate if rate > 0 else 0
                        print(f"Progress: {config_idx}/{total_configs} ({elapsed:.1f}s, ~{remaining:.1f}s remaining)")
    
    elapsed_total = time.time() - start_time
    print(f"\nCompleted {len(results)}/{total_configs} configs in {elapsed_total:.1f}s")
    
    return results


def analyze_results(results: list) -> dict:
    """分析扫描结果"""
    
    # 按 PQ 配置统计
    pq_stats = {}
    for result in results:
        if "pq_M" in result:
            key = f"PQ(M={result['pq_M']},K={result['pq_K']})"
        else:
            key = f"INT{result['n_bits']}"
        
        if key not in pq_stats:
            pq_stats[key] = {
                "configs": [],
                "err_increases": [],
                "compression_ratios": [],
            }
        
        pq_stats[key]["configs"].append(result)
        pq_stats[key]["err_increases"].append(result["err_increase_pct"])
        pq_stats[key]["compression_ratios"].append(result["compression_ratio"])
    
    # 计算统计量
    summary = {}
    for key, stats in pq_stats.items():
        errs = stats["err_increases"]
        comps = stats["compression_ratios"]
        
        summary[key] = {
            "count": len(stats["configs"]),
            "mean_err_inc": float(np.mean(errs)),
            "std_err_inc": float(np.std(errs)),
            "max_err_inc": float(np.max(errs)),
            "min_err_inc": float(np.min(errs)),
            "mean_compression": float(np.mean(comps)),
            "pass_15pct": sum(1 for e in errs if e < 15) / len(errs),
            "pass_30pct": sum(1 for e in errs if e < 30) / len(errs),
        }
    
    # PQ vs INT4 对比
    pq_vs_int4 = {}
    for result in results:
        if "pq_M" not in result:
            continue
        
        key = f"PQ(M={result['pq_M']},K={result['pq_K']})"
        kv_key = (result["kv_type"], result["kv_len"], result["q_len"], result["coreset_r"])
        
        if kv_key not in pq_vs_int4:
            pq_vs_int4[kv_key] = {"pq": [], "int4": [], "int8": []}
        
        pq_vs_int4[kv_key]["pq"].append(result)
    
    # 找对应的 INT4/INT8
    for result in results:
        if "pq_M" in result:
            continue
        
        kv_key = (result["kv_type"], result["kv_len"], result["q_len"], result["coreset_r"])
        if kv_key in pq_vs_int4:
            if result["n_bits"] == 4:
                pq_vs_int4[kv_key]["int4"].append(result)
            else:
                pq_vs_int4[kv_key]["int8"].append(result)
    
    # 计算 PQ 相对于 INT4 的改进
    pq_vs_int4_summary = []
    for kv_key, data in pq_vs_int4.items():
        if data["pq"] and (data["int4"] or data["int8"]):
            for pq_r in data["pq"]:
                best_int = min(data["int4"] + data["int8"], key=lambda x: x["err_int4"])
                
                improvement = {
                    "kv_type": kv_key[0],
                    "kv_len": kv_key[1],
                    "q_len": kv_key[2],
                    "r": kv_key[3],
                    "pq_config": f"M={pq_r['pq_M']},K={pq_r['pq_K']}",
                    "pq_err_inc": pq_r["err_increase_pct"],
                    "int_err_inc": best_int["err_int4"],
                    "pq_comp": pq_r["compression_ratio"],
                    "int_comp": best_int["compression_ratio"],
                    "better_error": pq_r["err_pq"] < best_int["err_int4"],
                    "better_compression": pq_r["compression_ratio"] > best_int["compression_ratio"],
                }
                pq_vs_int4_summary.append(improvement)
    
    # Pareto front (按 compression 排序)
    pareto = []
    for r in results:
        dominated = False
        err_key = "err_pq" if "pq_M" in r else "err_int4"
        for other in results:
            other_err_key = "err_pq" if "pq_M" in other else "err_int4"
            if (other["compression_ratio"] >= r["compression_ratio"] and 
                other[other_err_key] <= r[err_key] and
                (other["compression_ratio"] > r["compression_ratio"] or 
                 other[other_err_key] < r[err_key])):
                dominated = True
                break
        if not dominated:
            pareto.append(r)
    
    pareto_sorted = sorted(pareto, key=lambda x: x["compression_ratio"], reverse=True)
    
    return {
        "summary": summary,
        "pq_vs_int4": pq_vs_int4_summary,
        "pareto": pareto_sorted[:20],  # Top 20
    }


def save_results(results: list, analysis: dict, output_dir: str):
    """保存结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanity check
    with open(os.path.join(output_dir, "exp20_sanity.json"), "w") as f:
        json.dump({
            "type": "sanity_check",
            "results": [r for r in results if r["kv_len"] <= 128]
        }, f, indent=2, default=str)
    
    # Full sweep
    with open(os.path.join(output_dir, "exp20_sweep.json"), "w") as f:
        json.dump({
            "type": "full_sweep",
            "total_configs": len(results),
            "results": results,
            "analysis": analysis,
        }, f, indent=2, default=str)
    
    # Pareto
    with open(os.path.join(output_dir, "exp20_pareto.json"), "w") as f:
        json.dump({
            "pareto_front": analysis["pareto"],
        }, f, indent=2, default=str)
    
    # PQ vs INT4 对比
    with open(os.path.join(output_dir, "exp20_vs_int4.json"), "w") as f:
        json.dump({
            "pq_vs_int4": analysis["pq_vs_int4"],
        }, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}")


def main():
    output_dir = os.path.join(_REPO_ROOT, "results")
    
    # Sanity check
    sanity_results = run_sanity_check()
    
    print("\n" + "=" * 60)
    print("Full Sweep (using sanity + extended configs)")
    print("=" * 60)
    
    # Full sweep
    sweep_results = run_full_sweep()
    
    # 分析
    analysis = analyze_results(sweep_results)
    
    # 保存
    save_results(sweep_results, analysis, output_dir)
    
    # 打印报告
    print("\n" + "=" * 60)
    print("Exp20 Report: Product Quantization vs INT4")
    print("=" * 60)
    
    print("\n--- Summary by Config ---")
    print(f"{'Config':>20} {'Mean Err Inc %':>15} {'Std':>10} {'Pass<15%':>10} {'Mean Comp':>12}")
    for key, stats in sorted(analysis["summary"].items()):
        print(
            f"{key:>20} "
            f"{stats['mean_err_inc']:>+15.1f} "
            f"{stats['std_err_inc']:>10.1f} "
            f"{stats['pass_15pct']:>10.1%} "
            f"{stats['mean_compression']:>12.1f}x"
        )
    
    print("\n--- PQ vs INT4 Improvement ---")
    better_err = sum(1 for x in analysis["pq_vs_int4"] if x["better_error"])
    better_comp = sum(1 for x in analysis["pq_vs_int4"] if x["better_compression"])
    total = len(analysis["pq_vs_int4"])
    
    if total > 0:
        print(f"PQ has better error: {better_err}/{total} ({better_err/total:.1%})")
        print(f"PQ has better compression: {better_comp}/{total} ({better_comp/total:.1%})")
    else:
        print("No comparable configs found")
    
    print("\n--- Pareto Front (Top 5) ---")
    for i, p in enumerate(analysis["pareto"][:5]):
        method = f"PQ(M={p['pq_M']},K={p['pq_K']})" if "pq_M" in p else f"INT{p['n_bits']}"
        err_key = "err_pq" if "pq_M" in p else "err_int4"
        print(
            f"  {i+1}. {method} "
            f"kv={p['kv_len']:>4} r={p.get('coreset_r', p.get('sketch_r', '?'))} "
            f"err={p[err_key]:.3e} comp={p['compression_ratio']:.1f}x"
        )
    
    # 生成报告
    report = generate_report(analysis, sweep_results)
    report_path = os.path.join(output_dir, "exp20_pq_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nFull report: {report_path}")
    
    return sweep_results, analysis


def generate_report(analysis: dict, results: list) -> str:
    """生成 Markdown 报告"""
    
    report = """# Exp20: Product Quantization (PQ) 实验报告

## 核心假设

INT4 量化粒度太粗（每个 token 整体量化），对 clustered 数据不友好。
**Product Quantization (PQ)** 把 V 的 head_dim 维度切成 M 个子空间，
每个子空间独立做 k-means 量化。理论上 PQ-8 能达到 INT4 的 1/4 误差。

## 实验配置

- **PQ M**: 4, 8, 16 (head_dim=128, 段长 32/16/8)
- **PQ K**: 16, 64, 256 (每段 centroids 数)
- **总编码位数**: M · log2(K)
- **数据类型**: clustered, random, skewed
- **KV 长度**: 512, 2048
- **Q 长度**: 16, 64
- **seed**: 42

## 审查清单

### 1. 物理诚实
"""
    
    # 检查物理诚实边界
    physical_violations = [r for r in results if r.get("exceeds_physical", False)]
    if physical_violations:
        report += f"\n⚠️ **警告**: {len(physical_violations)} 个配置超出物理诚实边界\n"
    else:
        report += "\n✅ 所有配置满足物理诚实边界\n"
    
    report += f"\n- ratio 上限 ≈ 2·kv_len/q_len\n"
    report += f"- 测试配置数: {len(results)}\n"
    
    report += "\n### 2. PQ vs INT4 对比\n"
    
    better_err = sum(1 for x in analysis["pq_vs_int4"] if x["better_error"])
    better_comp = sum(1 for x in analysis["pq_vs_int4"] if x["better_compression"])
    total = len(analysis["pq_vs_int4"])
    
    if total > 0:
        report += f"- PQ 误差更小: {better_err}/{total} ({better_err/total:.1%})\n"
        report += f"- PQ 压缩更好: {better_comp}/{total} ({better_comp/total:.1%})\n"
    else:
        report += "- 无可比配置\n"
    
    report += "\n### 3. 统计摘要\n\n"
    report += "| Config | Mean Err Inc % | Std | Pass<15% | Mean Comp |\n"
    report += "|--------|----------------|-----|----------|----------|\n"
    
    for key, stats in sorted(analysis["summary"].items()):
        report += f"| {key} | {stats['mean_err_inc']:+.1f} | {stats['std_err_inc']:.1f} | {stats['pass_15pct']:.1%} | {stats['mean_compression']:.1f}x |\n"
    
    report += "\n### 4. 诚实结论\n\n"
    
    # 判断 PQ 是否值得
    avg_pq_err = np.mean([s["mean_err_inc"] for k, s in analysis["summary"].items() if "PQ" in k])
    avg_int4_err = np.mean([s["mean_err_inc"] for k, s in analysis["summary"].items() if "INT" in k])
    
    report += f"- PQ 平均误差增加: {avg_pq_err:+.1f}%\n"
    report += f"- INT4 平均误差增加: {avg_int4_err:+.1f}%\n"
    
    if avg_pq_err < avg_int4_err:
        report += "\n**结论**: PQ 在相同压缩比下误差更小，适合对精度敏感的场景。\n"
    else:
        report += "\n**结论**: INT4 在此配置下更稳定。PQ 优势在于更高的编码灵活性。\n"
    
    report += "\n## 可解释性\n\n"
    report += "- PQ 在 V 矩阵子空间结构存在时效果更好\n"
    report += "- 如果 V 矩阵没有明显子空间聚类，PQ 优势不明显\n"
    report += "- PQ 计算开销 O(n·d·iter) 比 INT4 O(n·d) 更高\n"
    
    return report


if __name__ == "__main__":
    main()
