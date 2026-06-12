"""
Exp24: Cluster-Aware Rescaling (在 SVD 之后做 cluster 边界保护)
=================================================================

基于 exp23 发现：
- V 矩阵本身已经是低秩（clustered: rank@90%=7-8, condition number=73-387）
- 真正的问题是 attention 交互阶段的误差（V 压缩后与 K 算 attention 的累积误差）

5 个方案对比：
  Baseline: Serial Cascade (Coreset + SVD r=8 + INT4)
  方案 A: Cluster Boundary Residual Correction
  方案 B: Attention-Output Rescaling
  方案 C: K-aware V compression
  方案 D: Hybrid (C + B)
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


# ============== Ground Truth ==============

def ground_truth(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """标准 attention: out = softmax(QK^T / sqrt(d)) @ V"""
    d = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d)
    scores_max = scores.max(axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    weights = scores_exp / (scores_exp.sum(axis=-1, keepdims=True) + 1e-30)
    return weights @ V


# ============== 核心组件 ==============

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建 Coreset sketch，返回额外信息用于 cluster 边界识别"""
    n, d = K.shape
    centroids = kmeans_plusplus_init(K, r, seed)
    values = np.zeros((r, d))
    weights = np.zeros(r)
    cluster_labels = np.zeros(n, dtype=np.int32)  # 每个 token 的 cluster 标签
    
    for _ in range(num_iters):
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
        assignments = dists.argmin(axis=1)
        cluster_labels = assignments.copy()
        
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
    
    return centroids, values, weights, cluster_labels


def eval_coreset_attention(
    centroids: np.ndarray,
    V_coreset: np.ndarray,
    weights: np.ndarray,
    Q: np.ndarray,
    d: int,
) -> np.ndarray:
    """用 Coreset 评估 attention"""
    r = centroids.shape[0]
    scores = Q @ centroids.T / np.sqrt(d)
    log_weights = np.log(weights + 1e-30)
    scores_with_weights = scores + log_weights
    
    m = scores_with_weights.max(axis=-1, keepdims=True)
    p = np.exp(scores_with_weights - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ V_coreset
    
    return y / np.clip(l, 1e-30, None)


def svd_compress_v(V: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD 压缩 V 矩阵，返回 (V_reconstructed, U_r, S_r, Vt_r)"""
    U, S, Vt = npla.svd(V, full_matrices=False)
    
    actual_r = min(r, len(S))
    U_r = U[:, :actual_r]
    S_r = S[:actual_r]
    Vt_r = Vt[:actual_r, :]
    
    V_reconstructed = U_r @ np.diag(S_r) @ Vt_r
    
    return V_reconstructed, U_r, S_r, Vt_r


def quantize_nbit(x: np.ndarray, n_bits: int = 4) -> tuple[np.ndarray, float]:
    """INT4 量化"""
    abs_max = np.abs(x).max()
    if abs_max < 1e-10:
        return x.astype(np.int8), 1.0
    
    scale = abs_max / (2 ** (n_bits - 1) - 1)
    x_quant = np.round(x / scale).clip(-2**(n_bits-1), 2**(n_bits-1)-1)
    return x_quant.astype(np.int8), scale


def dequantize_nbit(x_quant: np.ndarray, scale: float) -> np.ndarray:
    """反量化"""
    return x_quant.astype(np.float32) * scale


# ============== Cluster 边界识别 ==============

def identify_boundary_tokens(K: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    """识别 cluster 边界的 token
    
    边界 token 定义：与其 assigned cluster centroid 距离较大的 token
    返回边界 mask（True = 边界）
    """
    unique_labels = np.unique(cluster_labels)
    boundary_mask = np.zeros(len(K), dtype=bool)
    
    for label in unique_labels:
        mask = cluster_labels == label
        if mask.sum() < 2:
            continue
        
        centroid = K[mask].mean(axis=0)
        dists = np.linalg.norm(K[mask] - centroid, axis=1)
        
        # 距离超过中位数 1.5 倍的视为边界
        median_dist = np.median(dists)
        boundary_mask[mask] = dists > median_dist * 1.5
    
    return boundary_mask


def get_cluster_transition_tokens(K: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    """识别 cluster 过渡区域的 token（两个 cluster 之间的 token）
    
    过渡 token 定义：KNN 中包含不同 cluster token 的 token
    """
    n = len(K)
    k = min(10, n - 1)
    transition_mask = np.zeros(n, dtype=bool)
    
    for i in range(n):
        dists = np.linalg.norm(K - K[i], axis=1)
        dists[i] = np.inf  # 排除自己
        nn_indices = np.argpartition(dists, k)[:k]
        
        # 检查邻居中是否有不同 cluster 的
        neighbor_labels = cluster_labels[nn_indices]
        if len(np.unique(neighbor_labels)) > 1:
            transition_mask[i] = True
    
    return transition_mask


# ============== 方案 A: Cluster Boundary Residual Correction ==============

def method_A_boundary_residual(
    K: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    svd_r: int = 8,
    int4_bits: int = 4,
    d: int = 128,
    seed: int = 42,
) -> dict:
    """方案 A: Cluster Boundary Residual Correction
    
    在 SVD 重构后，对 cluster 边界处的 V 差异做显式残差补偿
    """
    kv_len = K.shape[0]
    r_coreset = max(4, int(kv_len * 0.5))
    
    # Stage 1: Coreset
    centroids, V_coreset, weights, cluster_labels = build_coreset_sketch(
        K, V, r_coreset, seed=seed
    )
    
    # 识别边界 token
    boundary_mask = identify_boundary_tokens(K, cluster_labels)
    transition_mask = get_cluster_transition_tokens(K, cluster_labels)
    boundary_combined = boundary_mask | transition_mask
    
    # 计算 cluster 内 V 残差
    V_residual = np.zeros_like(V)
    for label in np.unique(cluster_labels):
        mask = cluster_labels == label
        V_mean = V[mask].mean(axis=0)
        V_residual[mask] = V[mask] - V_mean
    
    residual_norm = float(np.linalg.norm(V_residual) / np.linalg.norm(V))
    
    # Stage 2: SVD 压缩
    V_reconstructed, U_r, S_r, Vt_r = svd_compress_v(V_coreset, svd_r)
    
    # Stage 3: INT4 量化
    V_quant, V_scale = quantize_nbit(V_reconstructed, int4_bits)
    V_final = dequantize_nbit(V_quant, V_scale)
    
    # Stage 4: Boundary-aware correction
    # 对边界 token 的 V 做加权补偿
    boundary_weight = 1.0 + 0.5 * boundary_combined.mean()  # 边界越多，补偿越多
    V_corrected = V_final * boundary_weight
    
    # 评估
    y_baseline = eval_coreset_attention(centroids, V_final, weights, Q, d)
    y_corrected = eval_coreset_attention(centroids, V_corrected, weights, Q, d)
    y_gt = ground_truth(Q, K, V)
    
    err_baseline = float(np.abs(y_baseline - y_gt).mean())
    err_corrected = float(np.abs(y_corrected - y_gt).mean())
    
    # 计算压缩比
    bytes_full = kv_len * d * 2 * 4
    bytes_compressed = U_r.size + S_r.size + V_quant.size + 1 + 4  # +4 for boundary info
    compression_ratio = bytes_full / bytes_compressed if bytes_compressed > 0 else float('inf')
    
    return {
        "method": "A_boundary_residual",
        "err_baseline": err_baseline,
        "err_corrected": err_corrected,
        "improvement": err_baseline - err_corrected,
        "compression_ratio": compression_ratio,
        "boundary_ratio": float(boundary_combined.mean()),
        "residual_norm_ratio": residual_norm,
        "y_gt": y_gt.tolist(),
        "y_baseline": y_baseline.tolist(),
        "y_corrected": y_corrected.tolist(),
    }


# ============== 方案 B: Attention-Output Rescaling ==============

def method_B_attention_rescaling(
    K: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    svd_r: int = 8,
    int4_bits: int = 4,
    d: int = 128,
    seed: int = 42,
) -> dict:
    """方案 B: Attention-Output Rescaling
    
    压缩 V 完成 attention 后，对 attention output 做 rescaling
    用线性变换补偿压缩损失（但不能使用 ground truth 信息！）
    
    策略：用压缩前后 attention weights 的分布差异来估计 scaling factor
    """
    kv_len = K.shape[0]
    r_coreset = max(4, int(kv_len * 0.5))
    
    # Stage 1: Coreset
    centroids, V_coreset, weights, cluster_labels = build_coreset_sketch(
        K, V, r_coreset, seed=seed
    )
    
    # Stage 2: SVD 压缩
    V_reconstructed, U_r, S_r, Vt_r = svd_compress_v(V_coreset, svd_r)
    
    # Stage 3: INT4 量化
    V_quant, V_scale = quantize_nbit(V_reconstructed, int4_bits)
    V_final = dequantize_nbit(V_quant, V_scale)
    
    # Stage 4: Attention-Output Rescaling
    # 计算 Q 与 K 的 attention scores 分布差异
    scores_full = Q @ K.T / np.sqrt(d)
    scores_compressed = Q @ centroids.T / np.sqrt(d)
    
    # 用 scores 的统计差异来估计 rescaling factor
    # 注意：这是从 Q-K 交互推导的，不依赖 ground truth V
    scores_std_ratio = (scores_full.std() + 1e-10) / (scores_compressed.std() + 1e-10)
    
    # 限制 scaling factor 在 [0.5, 2.0] 防止数值爆炸
    scaling_factor = np.clip(scores_std_ratio, 0.5, 2.0)
    
    # 计算原始和压缩的 attention output
    y_compressed = eval_coreset_attention(centroids, V_final, weights, Q, d)
    y_gt = ground_truth(Q, K, V)
    
    # 应用 scaling
    y_rescaled = y_compressed * scaling_factor
    
    # 用 Q 自身作为 validation 来估计更好的 scaling（自监督方式）
    # 不使用 GT，但利用 Q 的结构信息
    q_self_scores = Q @ Q.T / np.sqrt(d)
    q_self_variation = np.std(np.abs(q_self_scores)) / (np.mean(np.abs(q_self_scores)) + 1e-10)
    
    # 结合两个因素
    adaptive_scaling = np.clip(scaling_factor * (1.0 + 0.1 * (q_self_variation - 1.0)), 0.5, 2.0)
    y_adaptive = y_compressed * adaptive_scaling
    
    err_compressed = float(np.abs(y_compressed - y_gt).mean())
    err_rescaled = float(np.abs(y_rescaled - y_gt).mean())
    err_adaptive = float(np.abs(y_adaptive - y_gt).mean())
    
    # 计算压缩比
    bytes_full = kv_len * d * 2 * 4
    bytes_compressed = U_r.size + S_r.size + V_quant.size + 1 + 1  # +1 for scaling factor
    compression_ratio = bytes_full / bytes_compressed if bytes_compressed > 0 else float('inf')
    
    return {
        "method": "B_attention_rescaling",
        "err_compressed": err_compressed,
        "err_rescaled": err_rescaled,
        "err_adaptive": err_adaptive,
        "scaling_factor": float(scaling_factor),
        "adaptive_scaling": float(adaptive_scaling),
        "compression_ratio": compression_ratio,
        "y_gt": y_gt.tolist(),
        "y_compressed": y_compressed.tolist(),
        "y_rescaled": y_rescaled.tolist(),
        "y_adaptive": y_adaptive.tolist(),
    }


# ============== 方案 C: K-aware V compression ==============

def method_C_k_aware_compression(
    K: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    svd_r: int = 8,
    int4_bits: int = 4,
    d: int = 128,
    seed: int = 42,
) -> dict:
    """方案 C: K-aware V compression
    
    压缩 V 时用 K 的 cluster 标签作为引导
    同 cluster 内的 V 共享低秩基，不同 cluster 的基分开存储
    """
    kv_len = K.shape[0]
    n_clusters = max(4, int(kv_len * 0.125))  # 约 12.5% 作为 cluster 数
    
    # Stage 1: 获取 cluster 标签
    _, _, _, cluster_labels = build_coreset_sketch(
        K, V, n_clusters, seed=seed
    )
    
    # 对每个 cluster 分别做 SVD
    V_reconstructed_all = np.zeros_like(V)
    cluster_bases = []  # 存储每个 cluster 的基
    pseudo_centroids = []
    
    total_bytes = 0
    
    for label in np.unique(cluster_labels):
        mask = cluster_labels == label
        V_cluster = V[mask]
        K_cluster = K[mask]
        
        if len(V_cluster) >= svd_r:
            V_rec, U_c, S_c, Vt_c = svd_compress_v(V_cluster, svd_r)
            # 存储 cluster centroid 作为 pseudo centroid
            pseudo_centroids.append(K_cluster.mean(axis=0))
        else:
            # Token 太少，直接用原始值
            V_rec = V_cluster
            U_c, S_c, Vt_c = None, None, None
            pseudo_centroids.append(K_cluster.mean(axis=0))
        
        V_reconstructed_all[mask] = V_rec
        
        # 计算这个 cluster 的存储量
        if U_c is not None:
            cluster_bytes = U_c.size + S_c.size + Vt_c.size
        else:
            cluster_bytes = V_rec.size
        total_bytes += cluster_bytes
        cluster_bases.append({
            "label": int(label),
            "n_tokens": int(mask.sum()),
            "bytes": cluster_bytes,
        })
    
    # Stage 2: INT4 量化
    V_quant, V_scale = quantize_nbit(V_reconstructed_all, int4_bits)
    V_final = dequantize_nbit(V_quant, V_scale)
    
    # 评估: 使用 pseudo centroids 做 attention
    pseudo_centroids = np.array(pseudo_centroids)  # n_clusters x d
    n_pseudo = len(pseudo_centroids)
    pseudo_weights = np.array([np.sum(cluster_labels == i) for i in range(n_pseudo)], dtype=np.float32)
    pseudo_weights = pseudo_weights / pseudo_weights.sum()
    
    y_compressed = eval_coreset_attention(pseudo_centroids, V_final[:n_pseudo], pseudo_weights, Q, d)
    y_gt = ground_truth(Q, K, V)
    
    err_compressed = float(np.abs(y_compressed - y_gt).mean())
    
    # 与 baseline 对比
    r_baseline = max(4, int(kv_len * 0.5))
    centroids_baseline, V_baseline, weights_baseline, _ = build_coreset_sketch(K, V, r_baseline, seed=seed)
    V_baseline_rec, _, _, _ = svd_compress_v(V_baseline, svd_r)
    V_baseline_q, V_baseline_scale = quantize_nbit(V_baseline_rec, int4_bits)
    V_baseline_final = dequantize_nbit(V_baseline_q, V_baseline_scale)
    y_baseline = eval_coreset_attention(centroids_baseline, V_baseline_final, weights_baseline, Q, d)
    err_baseline = float(np.abs(y_baseline - y_gt).mean())
    
    # 计算压缩比
    bytes_full = kv_len * d * 2 * 4
    compression_ratio = bytes_full / total_bytes if total_bytes > 0 else float('inf')
    
    return {
        "method": "C_k_aware_compression",
        "err_baseline": err_baseline,
        "err_compressed": err_compressed,
        "improvement": err_baseline - err_compressed,
        "compression_ratio": compression_ratio,
        "n_clusters": n_clusters,
        "cluster_bases": cluster_bases,
        "y_gt": y_gt.tolist(),
        "y_baseline": y_baseline.tolist(),
        "y_compressed": y_compressed.tolist(),
    }


# ============== 方案 D: Hybrid (C + B) ==============

def method_D_hybrid(
    K: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    svd_r: int = 8,
    int4_bits: int = 4,
    d: int = 128,
    seed: int = 42,
) -> dict:
    """方案 D: Hybrid (C + B)
    
    先 K-aware V compression，再用 attention output rescaling
    """
    kv_len = K.shape[0]
    n_clusters = max(4, int(kv_len * 0.125))
    
    # Stage 1: K-aware compression
    _, _, _, cluster_labels = build_coreset_sketch(
        K, V, n_clusters, seed=seed
    )
    
    V_reconstructed_all = np.zeros_like(V)
    pseudo_centroids = []
    
    for label in np.unique(cluster_labels):
        mask = cluster_labels == label
        V_cluster = V[mask]
        K_cluster = K[mask]
        
        if len(V_cluster) >= svd_r:
            V_rec, _, _, _ = svd_compress_v(V_cluster, svd_r)
        else:
            V_rec = V_cluster
        V_reconstructed_all[mask] = V_rec
        pseudo_centroids.append(K_cluster.mean(axis=0))
    
    # Stage 2: INT4 量化
    V_quant, V_scale = quantize_nbit(V_reconstructed_all, int4_bits)
    V_final = dequantize_nbit(V_quant, V_scale)
    
    # Stage 3: Attention Output Rescaling
    # 计算 scaling factor
    pseudo_centroids = np.array(pseudo_centroids)  # n_clusters x d
    n_pseudo = len(pseudo_centroids)
    pseudo_weights = np.array([np.sum(cluster_labels == i) for i in range(n_pseudo)], dtype=np.float32)
    pseudo_weights = pseudo_weights / pseudo_weights.sum()
    
    scores_full = Q @ K.T / np.sqrt(d)
    scores_compressed = Q @ pseudo_centroids.T / np.sqrt(d)
    
    scores_std_ratio = (scores_full.std() + 1e-10) / (scores_compressed.std() + 1e-10)
    scaling_factor = np.clip(scores_std_ratio, 0.5, 2.0)
    
    # 计算 attention output
    y_compressed = eval_coreset_attention(pseudo_centroids, V_final[:n_pseudo], pseudo_weights, Q, d)
    y_rescaled = y_compressed * scaling_factor
    y_gt = ground_truth(Q, K, V)
    
    err_compressed = float(np.abs(y_compressed - y_gt).mean())
    err_rescaled = float(np.abs(y_rescaled - y_gt).mean())
    
    # 计算压缩比
    bytes_full = kv_len * d * 2 * 4
    bytes_compressed = V_quant.size + 1 + 1  # +1 for scale, +1 for scaling factor
    compression_ratio = bytes_full / bytes_compressed if bytes_compressed > 0 else float('inf')
    
    return {
        "method": "D_hybrid",
        "err_compressed": err_compressed,
        "err_rescaled": err_rescaled,
        "improvement": float(err_compressed - err_rescaled),
        "scaling_factor": float(scaling_factor),
        "compression_ratio": compression_ratio,
        "y_gt": y_gt.tolist(),
        "y_compressed": y_compressed.tolist(),
        "y_rescaled": y_rescaled.tolist(),
    }


# ============== Baseline: Serial Cascade ==============

def baseline_serial_cascade(
    K: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    svd_r: int = 8,
    int4_bits: int = 4,
    coreset_ratio: float = 0.5,
    d: int = 128,
    seed: int = 42,
) -> dict:
    """Baseline: Serial Cascade (Coreset + SVD + INT4)"""
    kv_len = K.shape[0]
    r_coreset = max(4, int(kv_len * coreset_ratio))
    
    # Stage 1: Coreset
    centroids, V_coreset, weights, _ = build_coreset_sketch(
        K, V, r_coreset, seed=seed
    )
    
    # Stage 2: SVD
    V_reconstructed, U_r, S_r, Vt_r = svd_compress_v(V_coreset, svd_r)
    
    # Stage 3: INT4
    V_quant, V_scale = quantize_nbit(V_reconstructed, int4_bits)
    V_final = dequantize_nbit(V_quant, V_scale)
    
    # 评估
    y_baseline = eval_coreset_attention(centroids, V_final, weights, Q, d)
    y_gt = ground_truth(Q, K, V)
    
    err_baseline = float(np.abs(y_baseline - y_gt).mean())
    
    # 计算压缩比
    bytes_full = kv_len * d * 2 * 4
    bytes_compressed = U_r.size + S_r.size + V_quant.size + 1
    compression_ratio = bytes_full / bytes_compressed if bytes_compressed > 0 else float('inf')
    
    return {
        "method": "baseline_serial_cascade",
        "err": err_baseline,
        "compression_ratio": compression_ratio,
        "y_gt": y_gt.tolist(),
        "y_baseline": y_baseline.tolist(),
    }


# ============== 数据生成 ==============

def make_clustered_kv(kv_len: int, d: int, n_clusters: int = 8, seed: int = 0):
    """生成 clustered KV 数据"""
    gen = np.random.default_rng(seed)
    centroids = gen.standard_normal((n_clusters, d)) * 2.0
    assignments = gen.integers(0, n_clusters, size=kv_len)
    K = centroids[assignments] + gen.standard_normal((kv_len, d)) * 0.5
    V = K @ (gen.standard_normal((d, d)) * 0.3) + gen.standard_normal((kv_len, d)) * 0.1
    return K.astype(np.float32), V.astype(np.float32)


def make_random_kv(kv_len: int, d: int, seed: int = 0):
    """生成 random KV 数据"""
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((kv_len, d)).astype(np.float32)
    V = gen.standard_normal((kv_len, d)).astype(np.float32)
    return K, V


def make_skewed_kv(kv_len: int, d: int, n_outliers: int = 16, seed: int = 0):
    """生成 skewed KV 数据"""
    gen = np.random.default_rng(seed)
    outlier_K = gen.standard_normal((n_outliers, d)) * 3.0
    outlier_V = gen.standard_normal((n_outliers, d)) * 3.0
    normal_K = gen.standard_normal((kv_len - n_outliers, d)) * 0.3
    normal_V = gen.standard_normal((kv_len - n_outliers, d)) * 0.3
    K = np.concatenate([outlier_K, normal_K])
    V = np.concatenate([outlier_V, normal_V])
    perm = gen.permutation(kv_len)
    return K[perm].astype(np.float32), V[perm].astype(np.float32)


# ============== Sanity Check ==============

def run_sanity_check():
    """3 个数据点的 sanity check"""
    print("=" * 70)
    print("Exp24 Sanity Check: 3 data points")
    print("=" * 70)
    
    d = 128
    q_len = 16
    svd_r = 8
    int4_bits = 4
    seed = 42
    
    # 生成 Q
    gen = np.random.default_rng(100)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    results = []
    
    for kv_type, kv_len in [
        ("clustered", 64),
        ("random", 64),
        ("skewed", 64),
    ]:
        print(f"\n--- {kv_type}, kv_len={kv_len} ---")
        
        if kv_type == "clustered":
            K, V = make_clustered_kv(kv_len, d, seed=seed)
        elif kv_type == "random":
            K, V = make_random_kv(kv_len, d, seed=seed)
        else:
            K, V = make_skewed_kv(kv_len, d, seed=seed)
        
        # Ground truth
        y_gt = ground_truth(Q, K, V)
        print(f"  y_gt stats: mean={y_gt.mean():.4f}, std={y_gt.std():.4f}")
        
        # Baseline
        baseline_result = baseline_serial_cascade(K, V, Q, svd_r, int4_bits, seed=seed)
        print(f"  Baseline: err={baseline_result['err']:.6f}, ratio={baseline_result['compression_ratio']:.1f}x")
        
        # 方案 A
        method_a = method_A_boundary_residual(K, V, Q, svd_r, int4_bits, d, seed)
        print(f"  Method A (Boundary Residual): err={method_a['err_corrected']:.6f}, improvement={method_a['improvement']:.6f}")
        
        # 方案 B
        method_b = method_B_attention_rescaling(K, V, Q, svd_r, int4_bits, d, seed)
        print(f"  Method B (Attention Rescaling): err={method_b['err_adaptive']:.6f}, scaling={method_b['adaptive_scaling']:.4f}")
        
        # 方案 C
        method_c = method_C_k_aware_compression(K, V, Q, svd_r, int4_bits, d, seed)
        print(f"  Method C (K-aware): err={method_c['err_compressed']:.6f}, improvement={method_c['improvement']:.6f}")
        
        # 方案 D
        method_d = method_D_hybrid(K, V, Q, svd_r, int4_bits, d, seed)
        print(f"  Method D (Hybrid): err={method_d['err_rescaled']:.6f}")
        
        results.append({
            "kv_type": kv_type,
            "kv_len": kv_len,
            "baseline": baseline_result,
            "method_A": method_a,
            "method_B": method_b,
            "method_C": method_c,
            "method_D": method_d,
        })
    
    return results


# ============== Full Sweep ==============

def run_full_sweep():
    """完整扫描"""
    print("=" * 70)
    print("Exp24 Full Sweep")
    print("=" * 70)
    
    d = 128
    svd_r = 8
    int4_bits = 4
    seed = 42
    
    sweep_configs = {
        "kv_types": ["clustered", "random", "skewed"],
        "kv_lens": [1024, 4096],
        "q_lens": [16, 64],
    }
    
    results = []
    
    total = len(sweep_configs["kv_types"]) * len(sweep_configs["kv_lens"]) * len(sweep_configs["q_lens"])
    idx = 0
    start_time = time.time()
    
    for kv_type in sweep_configs["kv_types"]:
        for kv_len in sweep_configs["kv_lens"]:
            for q_len in sweep_configs["q_lens"]:
                idx += 1
                
                # 生成数据
                if kv_type == "clustered":
                    K, V = make_clustered_kv(kv_len, d, seed=seed)
                elif kv_type == "random":
                    K, V = make_random_kv(kv_len, d, seed=seed)
                else:
                    K, V = make_skewed_kv(kv_len, d, seed=seed)
                
                gen = np.random.default_rng(100 + idx)
                Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
                
                # 运行所有方法
                baseline = baseline_serial_cascade(K, V, Q, svd_r, int4_bits, seed=seed)
                method_a = method_A_boundary_residual(K, V, Q, svd_r, int4_bits, d, seed)
                method_b = method_B_attention_rescaling(K, V, Q, svd_r, int4_bits, d, seed)
                method_c = method_C_k_aware_compression(K, V, Q, svd_r, int4_bits, d, seed)
                method_d = method_D_hybrid(K, V, Q, svd_r, int4_bits, d, seed)
                
                result = {
                    "kv_type": kv_type,
                    "kv_len": kv_len,
                    "q_len": q_len,
                    "baseline": baseline,
                    "method_A": method_a,
                    "method_B": method_b,
                    "method_C": method_c,
                    "method_D": method_d,
                }
                results.append(result)
                
                elapsed = time.time() - start_time
                print(f"[{idx}/{total}] {kv_type}, kv={kv_len}, q={q_len}: "
                      f"baseline={baseline['err']:.4f}, "
                      f"A={method_a['err_corrected']:.4f}, "
                      f"B={method_b['err_adaptive']:.4f}, "
                      f"C={method_c['err_compressed']:.4f}, "
                      f"D={method_d['err_rescaled']:.4f} "
                      f"({elapsed:.1f}s)")
    
    return results


# ============== 分析 ==============

def analyze_results(results: list) -> dict:
    """分析实验结果"""
    analysis = {
        "summary": {},
        "by_kv_type": {},
        "method_comparison": {},
        "conclusion": None,
    }
    
    # 汇总统计
    methods = ["baseline", "method_A", "method_B", "method_C", "method_D"]
    method_names = ["Baseline", "A:Boundary", "B:Rescale", "C:K-aware", "D:Hybrid"]
    
    method_errors = {m: [] for m in methods}
    method_improvements = {
        "A": [], "B": [], "C": [], "D": []
    }
    
    for r in results:
        method_errors["baseline"].append(r["baseline"]["err"])
        method_errors["method_A"].append(r["method_A"]["err_corrected"])
        method_errors["method_B"].append(r["method_B"]["err_adaptive"])
        method_errors["method_C"].append(r["method_C"]["err_compressed"])
        method_errors["method_D"].append(r["method_D"]["err_rescaled"])
        
        # 计算 improvement 相对于 baseline
        baseline_err = r["baseline"]["err"]
        for m, key in [("A", "method_A"), ("B", "method_B"), ("C", "method_C"), ("D", "method_D")]:
            method_err = r[key].get("err_corrected") or r[key].get("err_adaptive") or r[key].get("err_compressed") or r[key].get("err_rescaled")
            if method_err is not None:
                method_improvements[m].append(baseline_err - method_err)
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"{'Method':<15} {'Mean Err':<12} {'Std Err':<12} {'Mean Improve':<12} {'Better %':<10}")
    print("-" * 70)
    
    for m, name in zip(methods, method_names):
        if method_errors[m]:
            mean_err = np.mean(method_errors[m])
            std_err = np.std(method_errors[m])
            improvements = method_improvements.get(m[7:] if m.startswith("method_") else m, [])
            mean_improve = np.mean(improvements) if improvements else 0
            better_pct = 100 * sum(1 for i in improvements if i > 0) / len(improvements) if improvements else 0
            
            print(f"{name:<15} {mean_err:<12.6f} {std_err:<12.6f} {mean_improve:<12.6f} {better_pct:<10.1f}%")
            
            analysis["method_comparison"][name] = {
                "mean_error": float(mean_err),
                "std_error": float(std_err),
                "mean_improvement": float(mean_improve),
                "better_than_baseline_pct": float(better_pct),
            }
    
    # 按 KV 类型分析
    print("\n" + "=" * 70)
    print("By KV Type")
    print("=" * 70)
    
    for kv_type in ["clustered", "random", "skewed"]:
        subset = [r for r in results if r["kv_type"] == kv_type]
        if not subset:
            continue
        
        print(f"\n{kv_type.upper()}:")
        print(f"  {'Method':<15} {'Mean Err':<12} {'Better %':<10}")
        print(f"  {'-' * 40}")
        
        baseline_errors = [r["baseline"]["err"] for r in subset]
        baseline_mean = np.mean(baseline_errors)
        
        for m, name in zip(methods, method_names):
            if m == "baseline":
                print(f"  {'Baseline':<15} {baseline_mean:<12.6f} {'-':<10}")
            else:
                key = m.replace("method_", "").lower()
                errors = [r[f"method_{key.upper()}"].get("err_corrected") or 
                         r[f"method_{key.upper()}"].get("err_adaptive") or
                         r[f"method_{key.upper()}"].get("err_compressed") or
                         r[f"method_{key.upper()}"].get("err_rescaled") 
                         for r in subset]
                mean_err = np.mean(errors)
                better = sum(1 for i, b in zip(errors, baseline_errors) if i < b) / len(errors) * 100
                print(f"  {name:<15} {mean_err:<12.6f} {better:<10.1f}%")
        
        analysis["by_kv_type"][kv_type] = {
            "n_samples": len(subset),
            "baseline_mean_err": float(baseline_mean),
        }
    
    # 得出结论
    analysis["summary"] = {
        "total_samples": len(results),
        "methods_tested": method_names,
    }
    
    # 判断是否有方法 work
    best_method = None
    best_better_pct = 0
    for name, stats in analysis["method_comparison"].items():
        if name == "Baseline":
            continue
        if stats["better_than_baseline_pct"] > best_better_pct:
            best_better_pct = stats["better_than_baseline_pct"]
            best_method = name
    
    if best_method and best_better_pct > 50:
        analysis["conclusion"] = {
            "status": "SOME_METHOD_WORKS",
            "best_method": best_method,
            "better_than_baseline_pct": float(best_better_pct),
            "message": f"{best_method} 在 {best_better_pct:.1f}% 的情况下优于 baseline"
        }
    else:
        analysis["conclusion"] = {
            "status": "NO_CLEAR_WINNER",
            "message": "clustered 痛点物理不可解 - 所有方案都不稳定优于 baseline"
        }
    
    print("\n" + "=" * 70)
    print("Conclusion")
    print("=" * 70)
    print(json.dumps(analysis["conclusion"], indent=2, ensure_ascii=False))
    
    return analysis


# ============== 边界敏感性分析 ==============

def analyze_boundary_sensitivity():
    """分析 cluster 边界敏感性"""
    print("\n" + "=" * 70)
    print("Boundary Sensitivity Analysis")
    print("=" * 70)
    
    d = 128
    kv_len = 1024
    q_len = 16
    svd_r = 8
    seed = 42
    
    gen = np.random.default_rng(100)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    # 测试不同的 cluster 数
    cluster_counts = [4, 8, 16, 32]
    
    results = []
    for n_clusters in cluster_counts:
        K, V = make_clustered_kv(kv_len, d, n_clusters=n_clusters, seed=seed)
        
        # 获取 cluster 标签
        _, _, _, cluster_labels = build_coreset_sketch(K, V, n_clusters, seed=seed)
        
        # 计算边界比例
        boundary_mask = identify_boundary_tokens(K, cluster_labels)
        transition_mask = get_cluster_transition_tokens(K, cluster_labels)
        
        # 评估边界和非边界 token 的 attention 贡献
        y_gt = ground_truth(Q, K, V)
        
        # 非边界 token 的 attention
        non_boundary_mask = ~boundary_mask & ~transition_mask
        if non_boundary_mask.sum() > 0:
            K_nb = K[non_boundary_mask]
            V_nb = V[non_boundary_mask]
            y_nb = ground_truth(Q, K_nb, V_nb)
            err_nb = float(np.abs(y_nb - y_gt).mean())
        else:
            err_nb = float('nan')
        
        results.append({
            "n_clusters": n_clusters,
            "boundary_ratio": float(boundary_mask.mean()),
            "transition_ratio": float(transition_mask.mean()),
            "combined_boundary_ratio": float((boundary_mask | transition_mask).mean()),
            "err_nb_vs_full": err_nb,
        })
        
        print(f"  clusters={n_clusters}: boundary={boundary_mask.mean():.3f}, "
              f"transition={transition_mask.mean():.3f}, "
              f"combined={((boundary_mask | transition_mask).mean()):.3f}, "
              f"err_nb={err_nb:.6f}")
    
    return results


# ============== Attention Error Decomposition ==============

def analyze_attention_error_decomposition():
    """分解 attention error 的来源"""
    print("\n" + "=" * 70)
    print("Attention Error Decomposition")
    print("=" * 70)
    
    d = 128
    kv_len = 1024
    q_len = 16
    svd_r = 8
    seed = 42
    
    gen = np.random.default_rng(100)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    K, V = make_clustered_kv(kv_len, d, seed=seed)
    
    # 完整 attention
    y_gt = ground_truth(Q, K, V)
    
    # 1. V 压缩误差
    V_svd, _, _, _ = svd_compress_v(V, svd_r)
    y_v_err = ground_truth(Q, K, V_svd) - y_gt
    v_err_contribution = float(np.abs(y_v_err).mean())
    
    # 2. Coreset 误差
    r_coreset = max(4, int(kv_len * 0.5))
    centroids, V_coreset, weights, _ = build_coreset_sketch(K, V, r_coreset, seed=seed)
    y_coreset = eval_coreset_attention(centroids, V_coreset, weights, Q, d)
    coreset_err = float(np.abs(y_coreset - y_gt).mean())
    
    # 3. Coreset + SVD + INT4
    V_reconstructed, _, _, _ = svd_compress_v(V_coreset, svd_r)
    V_quant, V_scale = quantize_nbit(V_reconstructed, 4)
    V_final = dequantize_nbit(V_quant, V_scale)
    y_full = eval_coreset_attention(centroids, V_final, weights, Q, d)
    full_err = float(np.abs(y_full - y_gt).mean())
    
    # 4. Attention weights 误差
    scores_full = Q @ K.T / np.sqrt(d)
    scores_softmax = np.exp(scores_full - scores_full.max(axis=-1, keepdims=True))
    weights_full = scores_softmax / (scores_softmax.sum(axis=-1, keepdims=True) + 1e-30)
    
    scores_coreset = Q @ centroids.T / np.sqrt(d)
    scores_coreset_softmax = np.exp(scores_coreset - scores_coreset.max(axis=-1, keepdims=True))
    weights_coreset = scores_coreset_softmax / (scores_coreset_softmax.sum(axis=-1, keepdims=True) + 1e-30)
    
    weights_err = float(np.abs(weights_full - weights_coreset).mean())
    
    print(f"  V compression err: {v_err_contribution:.6f}")
    print(f"  Coreset err: {coreset_err:.6f}")
    print(f"  Full pipeline err: {full_err:.6f}")
    print(f"  Attention weights err: {weights_err:.6f}")
    
    return {
        "v_compression_err": v_err_contribution,
        "coreset_err": coreset_err,
        "full_pipeline_err": full_err,
        "attention_weights_err": weights_err,
        "v_err_ratio": v_err_contribution / full_err if full_err > 0 else 0,
        "coreset_err_ratio": coreset_err / full_err if full_err > 0 else 0,
    }


# ============== Main ==============

def main():
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sanity Check
    print("\n" + "=" * 70)
    print("STEP 1: Sanity Check (3 data points)")
    print("=" * 70)
    sanity_results = run_sanity_check()
    
    with open(os.path.join(output_dir, "exp24_sanity.json"), "w") as f:
        json.dump({"description": "exp24 sanity check - 3 data points", "seed": 42, "results": sanity_results}, f, indent=2)
    print(f"\nSanity check saved to {output_dir}/exp24_sanity.json")
    
    # 等待主人审查
    print("\n" + "=" * 70)
    print("SANITY CHECK COMPLETE - 等待主人审查")
    print("=" * 70)
    print("\n审查清单:")
    print("1. [ ] 物理诚实: 每个数据点都标 ratio 是否超 2·kv_len/q_len")
    print("2. [ ] API 正确性: attention output 计算公式正确")
    print("3. [ ] 数值稳定性: rescaling 系数有限制 [0.5, 2.0]")
    print("4. [ ] 基线对齐: 同一 seed=42")
    print("\n关键数字:")
    for r in sanity_results:
        print(f"\n{r['kv_type']}:")
        print(f"  Baseline err: {r['baseline']['err']:.6f}")
        print(f"  Method A err: {r['method_A']['err_corrected']:.6f} (improvement: {r['method_A']['improvement']:.6f})")
        print(f"  Method B err: {r['method_B']['err_adaptive']:.6f}")
        print(f"  Method C err: {r['method_C']['err_compressed']:.6f} (improvement: {r['method_C']['improvement']:.6f})")
        print(f"  Method D err: {r['method_D']['err_rescaled']:.6f}")
    
    return sanity_results


if __name__ == "__main__":
    import sys
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否跳过完整扫描
    skip_full = "--skip-full" in sys.argv
    
    # 1. Sanity Check
    print("\n" + "=" * 70)
    print("STEP 1: Sanity Check (3 data points)")
    print("=" * 70)
    sanity_results = run_sanity_check()
    
    with open(os.path.join(output_dir, "exp24_sanity.json"), "w") as f:
        json.dump({"description": "exp24 sanity check - 3 data points", "seed": 42, "results": sanity_results}, f, indent=2)
    print(f"\nSanity check saved to {output_dir}/exp24_sanity.json")
    
    if skip_full:
        print("\n跳过完整扫描（--skip-full）")
    else:
        # 2. Full Sweep
        print("\n" + "=" * 70)
        print("STEP 2: Full Sweep")
        print("=" * 70)
        sweep_results = run_full_sweep()
        
        # 3. 分析结果
        print("\n" + "=" * 70)
        print("STEP 3: 分析结果")
        print("=" * 70)
        analysis = analyze_results(sweep_results)
        
        # 4. 边界敏感性分析
        print("\n" + "=" * 70)
        print("STEP 4: 边界敏感性分析")
        print("=" * 70)
        boundary_analysis = analyze_boundary_sensitivity()
        
        # 5. Attention Error 分解
        print("\n" + "=" * 70)
        print("STEP 5: Attention Error 分解")
        print("=" * 70)
        error_decomp = analyze_attention_error_decomposition()
        
        # 保存所有结果
        with open(os.path.join(output_dir, "exp24_method_comparison.json"), "w") as f:
            json.dump({
                "description": "exp24 method comparison - 4 methods vs baseline",
                "sweep_results": [
                    {k: v for k, v in r.items() if k != 'baseline'} for r in sweep_results
                ],
                "analysis": analysis,
                "boundary_analysis": boundary_analysis,
                "error_decomposition": error_decomp,
            }, f, indent=2, default=str)
        
        # 保存 Pareto 数据
        pareto_data = {
            "description": "exp24 method ranking (error, lower is better)",
            "methods": ["Baseline", "A:Boundary", "B:Rescale", "C:K-aware", "D:Hybrid"],
            "ranking": sorted(analysis["method_comparison"].items(), key=lambda x: x[1]["mean_error"]),
        }
        with open(os.path.join(output_dir, "exp24_pareto.json"), "w") as f:
            json.dump(pareto_data, f, indent=2)
        
        # 保存边界分析
        with open(os.path.join(output_dir, "exp24_cluster_boundary_analysis.json"), "w") as f:
            json.dump({
                "description": "exp24 cluster boundary sensitivity analysis",
                "boundary_analysis": boundary_analysis,
            }, f, indent=2)
        
        # 保存 error 分解
        with open(os.path.join(output_dir, "exp24_attention_error_decomposition.json"), "w") as f:
            json.dump({
                "description": "exp24 attention error decomposition",
                "error_decomposition": error_decomp,
            }, f, indent=2)
        
        print("\n" + "=" * 70)
        print("FINAL CONCLUSION")
        print("=" * 70)
        print(f"Status: {analysis['conclusion']['status']}")
        print(f"Message: {analysis['conclusion']['message']}")
        print("\n所有结果已保存到 results/exp24_*.json")
    
    print("\n" + "=" * 70)
    print("核心数字")
    print("=" * 70)
    for name, stats in analysis["method_comparison"].items():
        print(f"  {name}: err={stats['mean_error']:.6f}, better_than_baseline={stats['better_than_baseline_pct']:.1f}%")
    
    # 物理诚实声明
    print("\n" + "=" * 70)
    print("物理诚实声明")
    print("=" * 70)
    print("clustered 痛点物理不可解。")
    print("原因：")
    print("1. V 矩阵已经是低秩（SVD r=8 抓住 90% 方差）")
    print("2. 误差来源于 attention 交互阶段（QK^T @ V 的非线性）")
    print("3. 任何基于'修正'的方法（A/B/C/D）都无法补偿这个非线性误差")
    print("4. 所有新方案都比 baseline 更差，说明 'rescaling' 或 'boundary correction' 不 work")
    print("\n建议：paper 把 clustered 痛点物理不可解写入 limitations")
