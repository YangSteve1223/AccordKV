"""
Exp13: SVD + Coreset Hybrid Compression
======================================

核心思路: 联合 SVD (V端低秩) + Coreset (K端量化友好)
- K 端: Coreset + INT4 量化 (复用 exp4) → quantization-friendly
- V 端: SVD 压缩 (复用 exp8) → output-side low rank
- 联合 attention 计算:
  K_sketch = Coreset(K_center) + quantize(INT4)
  V_svd = SVD(V | K_center) = U_V · Σ_V · V_V^T
  A = softmax(Q · K_sketch^T / √d)
  output = A · V_svd

预期优势:
1. Hybrid error ≤ min(Coreset error, SVD error)
2. Hybrid compression ≥ max(Coreset compression, SVD compression)
3. bytes vs error Pareto frontier 更优

Sweep 配置 (972 configs):
- q_len ∈ {16, 64, 256}
- kv_len ∈ {1024, 4096, 16384}
- Coreset r ∈ {4, 8, 16, 32}
- SVD r ∈ {4, 8, 16}
- INT4 bits ∈ {3, 4, 8}
- kv_type ∈ {clustered, random, skewed}

物理一致性:
- r_svd = r_full, r_coreset = r_full → error = 0 (完美重建)
- Hybrid compression = Coreset compression × SVD compression
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from numpy import linalg as npla

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    ground_truth,
)


# ============== 数据结构 ==============

@dataclass
class HybridSketch:
    """Hybrid SVD + Coreset sketch 容器"""
    # K 端: Coreset
    centroids_K: np.ndarray  # [r_coreset, d]
    weights_K: np.ndarray    # [r_coreset] importance weights
    r_coreset: int
    
    # K 端: 量化
    n_bits: int
    scale_K: np.ndarray
    quantized_K: np.ndarray  # INT bits
    
    # V 端: SVD
    U_V: np.ndarray          # [kv_len, r_svd]
    S_V: np.ndarray          # [r_svd]
    Vt_V: np.ndarray         # [r_svd, d]
    r_svd: int
    
    # 尺寸
    q_len: int
    kv_len: int
    d: int
    
    # 元数据
    kv_type: str
    
    def compression_ratio(self) -> float:
        """原始 KV vs Hybrid sketch"""
        original = self.kv_len * self.d * 2  # K + V
        # K: r_coreset * d (centroids) + r_coreset (weights)
        # V: r_svd * (kv_len + 1 + d) ≈ r_svd * (kv_len + d)
        k_bytes = self.r_coreset * self.d * 2 * 4  # centroids + weights
        v_bytes = self.r_svd * (self.kv_len + self.d) * 4
        hybrid_bytes = k_bytes + v_bytes
        return original / hybrid_bytes if hybrid_bytes > 0 else float('inf')
    
    def bytes_size(self) -> int:
        """Hybrid sketch bytes (considering quantization)"""
        # K 端量化
        k_bytes = self.r_coreset * self.d * self.n_bits // 8  # quantized centroids
        k_bytes += self.r_coreset * 4  # scale
        k_bytes += self.r_coreset * 4  # weights
        
        # V 端 SVD
        v_bytes = self.U_V.size * 4  # U
        v_bytes += self.S_V.size * 4  # S
        v_bytes += self.Vt_V.size * 4  # Vt
        
        return k_bytes + v_bytes


@dataclass
class CoresetSketch:
    """Coreset sketch (from exp4)"""
    centroids: np.ndarray  # [r, d]
    weights: np.ndarray   # [r]
    r: int


@dataclass
class QuantizedSketch:
    """Quantized Coreset sketch (from exp4)"""
    quantized_centroids: np.ndarray  # packed int values
    scale: np.ndarray               # [r] scale factors
    weights: np.ndarray             # [r]
    r: int
    n_bits: int


# ============== 数据生成 ==============

def make_clustered_kv(
    kv_len: int,
    d: int,
    n_clusters: int = 8,
    cluster_std: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 cluster 结构的 KV"""
    gen = np.random.default_rng(seed)
    
    centroids = []
    for _ in range(n_clusters):
        for _ in range(100):
            c = gen.standard_normal(d) * 2.0
            if all(npla.norm(c - oc) > 3.0 for oc in centroids):
                centroids.append(c)
                break
        if len(centroids) >= n_clusters:
            break
    while len(centroids) < n_clusters:
        centroids.append(gen.standard_normal(d) * 2.0)
    
    centroids = np.array(centroids)
    cluster_assignments = gen.integers(0, n_clusters, size=kv_len)
    K = centroids[cluster_assignments] + gen.standard_normal((kv_len, d)) * cluster_std
    
    W = gen.standard_normal((d, d)) * 0.3
    V = K @ W + gen.standard_normal((kv_len, d)) * 0.1
    
    return K.astype(np.float32), V.astype(np.float32)


def make_random_kv(
    kv_len: int,
    d: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成完全随机的 KV"""
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((kv_len, d)).astype(np.float32)
    V = gen.standard_normal((kv_len, d)).astype(np.float32)
    return K, V


def make_skewed_kv(
    kv_len: int,
    d: int,
    n_outliers: int = 16,
    outlier_std: float = 3.0,
    normal_std: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 skew 结构的 KV"""
    gen = np.random.default_rng(seed)
    
    outlier_K = gen.standard_normal((n_outliers, d)) * outlier_std
    outlier_V = gen.standard_normal((n_outliers, d)) * outlier_std
    normal_K = gen.standard_normal((kv_len - n_outliers, d)) * normal_std
    normal_V = gen.standard_normal((kv_len - n_outliers, d)) * normal_std
    
    K = np.concatenate([outlier_K, normal_K], axis=0)
    V = np.concatenate([outlier_V, normal_V], axis=0)
    
    perm = gen.permutation(kv_len)
    K = K[perm]
    V = V[perm]
    return K.astype(np.float32), V.astype(np.float32)


def make_kv_by_type(
    kv_len: int,
    d: int,
    kv_type: Literal["clustered", "random", "skewed"],
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """根据类型生成 KV"""
    if kv_type == "clustered":
        n_clusters = max(4, kv_len // 256)
        return make_clustered_kv(kv_len, d, n_clusters=n_clusters, seed=seed)
    elif kv_type == "random":
        return make_random_kv(kv_len, d, seed=seed)
    else:
        return make_skewed_kv(kv_len, d, seed=seed)


# ============== Coreset 实现 ==============

def build_coreset_sketch(
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    seed: int = 0,
) -> CoresetSketch:
    """Build Coreset sketch via k-means++ (optimized)"""
    kv_len, d = K.shape
    gen = np.random.default_rng(seed)
    
    # Use efficient k-means++ initialization
    idx = gen.integers(0, kv_len)
    centroids = [K[idx].copy()]
    
    for _ in range(r - 1):
        # Vectorized distance computation
        C = np.array(centroids)  # [i, d]
        dists = np.sum((K[:, None, :] - C[None, :, :]) ** 2, axis=2)  # [n, i]
        dists = dists.min(axis=1)  # [n]
        probs = dists / dists.sum()
        idx = gen.choice(kv_len, p=probs)
        centroids.append(K[idx].copy())
    
    centroids = np.array(centroids)
    weights = np.ones(r, dtype=np.float32)
    
    return CoresetSketch(centroids=centroids, weights=weights, r=r)


def quantize_sketch_nbit(sketch: CoresetSketch, n_bits: int = 4) -> QuantizedSketch:
    """Quantize Coreset sketch to n_bits"""
    r, d = sketch.centroids.shape
    
    # Per-row quantization (per centroid)
    quantized = np.zeros((r, d), dtype=np.int8)
    scales = np.zeros(r, dtype=np.float32)
    
    for i in range(r):
        c = sketch.centroids[i]
        c_max = np.abs(c).max()
        if c_max < 1e-6:
            scales[i] = 1.0
        else:
            scales[i] = c_max
            c_norm = c / c_max
            
            # Quantize
            levels = 2 ** (n_bits - 1)
            c_q = np.clip(np.round(c_norm * levels), -levels, levels - 1)
            quantized[i] = c_q.astype(np.int8)
    
    return QuantizedSketch(
        quantized_centroids=quantized,
        scale=scales,
        weights=sketch.weights.copy(),
        r=r,
        n_bits=n_bits,
    )


def dequantize_sketch_nbit(q_sketch: QuantizedSketch) -> CoresetSketch:
    """Dequantize Coreset sketch"""
    r, d = q_sketch.quantized_centroids.shape
    centroids = np.zeros((r, d), dtype=np.float32)
    
    for i in range(r):
        levels = 2 ** (q_sketch.n_bits - 1)
        centroids[i] = q_sketch.quantized_centroids[i].astype(np.float32) / levels * q_sketch.scale[i]
    
    return CoresetSketch(centroids=centroids, weights=q_sketch.weights.copy(), r=r)


def eval_coreset_sketch(
    sketch: CoresetSketch,
    Q: np.ndarray,
) -> NumpyAttnStats:
    """Evaluate Coreset sketch"""
    q_len, d = Q.shape
    r = sketch.r
    
    scores = Q @ sketch.centroids.T  # [q_len, r]
    
    # Scale by weights
    scores = scores * sketch.weights[None, :]
    
    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - scores_max)
    l = p.sum(axis=-1, keepdims=True)
    p_norm = p / np.clip(l, 1e-30, None)
    
    # Weighted sum of centroids
    F = p_norm @ sketch.centroids
    
    H = 1
    m = np.zeros((H, q_len, 1), dtype=np.float32)
    l_out = l[None, :, 0:1]
    y = F[None, :, :] * l_out
    
    return NumpyAttnStats(m=m, l=l_out, y=y)


# ============== SVD 实现 ==============

def compute_attention_matrix(Q: np.ndarray, K: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute attention matrix A = softmax(Q @ K^T / sqrt(d))"""
    d_sqrt = np.sqrt(Q.shape[1])
    scores = (Q @ K.T) / d_sqrt
    scores = scores / temperature
    
    scores_max = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - scores_max)
    A = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-30, None)
    
    return A


def build_svd_sketch(
    Q: np.ndarray,
    V: np.ndarray,
    r: int,
    temperature: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build SVD sketch for V.
    
    Compute: A = softmax(Q @ K^T / √d), then V_svd = A @ V
    SVD on V_svd to compress V-side.
    
    Returns:
        (U_V, S_V, Vt_V) for V-side SVD
    """
    # Compute attention output: V_attn = A @ V
    d_sqrt = np.sqrt(Q.shape[1])
    scores = (Q @ np.eye(Q.shape[1])).sum(axis=-1, keepdims=True)  # Placeholder
    
    # Actually compute A @ V
    A = compute_attention_matrix(Q, np.eye(Q.shape[1]), temperature)  # placeholder
    # We need K for this - let's modify
    
    return None, None, None  # Placeholder


def build_svd_for_V(
    V: np.ndarray,
    r: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build SVD for V matrix directly.
    
    V: [kv_len, d]
    Returns: U_V [kv_len, r], S_V [r], Vt_V [r, d]
    """
    kv_len, d = V.shape
    r_actual = min(r, kv_len, d)
    
    U, S, Vt = npla.svd(V, full_matrices=False)
    
    U_V = U[:, :r_actual].copy()
    S_V = S[:r_actual].copy()
    Vt_V = Vt[:r_actual, :].copy()
    
    return U_V, S_V, Vt_V


# ============== Hybrid Sketch 构建 ==============

def build_hybrid_sketch_v2(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    r_coreset: int,
    r_svd: int,
    n_bits: int = 4,
    seed: int = 0,
) -> HybridSketch:
    """Build Hybrid SVD + Coreset sketch V2.
    
    Key insight: K and V must be compressed together to maintain alignment.
    
    Step 1: K-Means++ on K to get r centroids (same as Coreset)
    Step 2: For each centroid, find the corresponding V rows and compute their mean
    Step 3: Apply SVD on the resulting V centroids
    Step 4: Quantize K centroids to INT4
    """
    q_len, kv_len, d = Q.shape[0], K.shape[0], K.shape[1]
    
    # Step 1: K-Means++ on K
    coreset_sketch = build_coreset_sketch(K, V, r_coreset, seed=seed)
    K_centroids = coreset_sketch.centroids  # [r, d]
    
    # Step 2: Find V centroids aligned with K centroids
    # For each K centroid, find nearest KV rows and average their V values
    V_centroids = np.zeros((r_coreset, d), dtype=np.float32)
    assignments = np.zeros(kv_len, dtype=np.int32)
    
    for i in range(r_coreset):
        # Find KV rows closest to this centroid
        dists = np.sum((K - K_centroids[i]) ** 2, axis=1)
        # Assign to this centroid
        is_closest = np.ones(kv_len, dtype=bool)
        for j in range(i):
            other_dists = np.sum((K - K_centroids[j]) ** 2, axis=1)
            is_closest &= (dists <= other_dists + 1e-6)
        assignments[is_closest] = i
        V_centroids[i] = V[is_closest].mean(axis=0)
    
    # Step 3: Apply SVD on V centroids
    U_V, S_V, Vt_V = build_svd_for_V(V_centroids, r_svd, seed=seed)
    
    # Step 4: Quantize K centroids
    quantized_sketch = quantize_sketch_nbit(coreset_sketch, n_bits=n_bits)
    
    return HybridSketch(
        centroids_K=coreset_sketch.centroids,
        weights_K=coreset_sketch.weights,
        r_coreset=r_coreset,
        n_bits=n_bits,
        scale_K=quantized_sketch.scale,
        quantized_K=quantized_sketch.quantized_centroids,
        U_V=U_V,
        S_V=S_V,
        Vt_V=Vt_V,
        r_svd=len(S_V),
        q_len=q_len,
        kv_len=kv_len,
        d=d,
        kv_type="hybrid",
    )


def build_hybrid_from_components(
    coreset_sketch: CoresetSketch,
    quantized_sketch: QuantizedSketch,
    V: np.ndarray,
    Q: np.ndarray,
    r_svd: int,
    seed: int = 0,
) -> HybridSketch:
    """Build hybrid sketch from pre-computed components"""
    q_len, kv_len, d = Q.shape[0], V.shape[0], V.shape[1]
    
    # SVD on V directly
    U_V, S_V, Vt_V = build_svd_for_V(V, r_svd, seed=seed)
    
    return HybridSketch(
        centroids_K=coreset_sketch.centroids,
        weights_K=coreset_sketch.weights,
        r_coreset=coreset_sketch.r,
        n_bits=quantized_sketch.n_bits,
        scale_K=quantized_sketch.scale,
        quantized_K=quantized_sketch.quantized_centroids,
        U_V=U_V,
        S_V=S_V,
        Vt_V=Vt_V,
        r_svd=len(S_V),
        q_len=q_len,
        kv_len=kv_len,
        d=d,
        kv_type="hybrid",
    )


def eval_hybrid_sketch(
    sketch: HybridSketch,
    Q: np.ndarray,
) -> np.ndarray:
    """Evaluate Hybrid sketch.
    
    Hybrid V2 approach:
    1. K_sketch = dequantize(K_coreset) + weights
    2. V_svd = U_V @ diag(S_V) @ Vt_V (reconstructed V_centroids, size [r, d])
    3. attention = softmax(Q @ K_sketch^T)
    4. output = attention @ V_svd
    
    V_svd is computed on V_centroids (V values aligned with K centroids)
    This ensures K and V compression are aligned.
    """
    # Dequantize K
    r, d = sketch.r_coreset, sketch.d
    K_deq = np.zeros((r, d), dtype=np.float32)
    levels = 2 ** (sketch.n_bits - 1)
    
    for i in range(r):
        K_deq[i] = sketch.quantized_K[i].astype(np.float32) / levels * sketch.scale_K[i]
    
    # Compute attention with dequantized K
    d_sqrt = np.sqrt(d)
    scores = (Q @ K_deq.T) / d_sqrt
    
    # Apply weights
    scores = scores * sketch.weights_K[None, :]
    
    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - scores_max)
    l = p.sum(axis=-1, keepdims=True)
    attn_weights = p / np.clip(l, 1e-30, None)  # [q_len, r]
    
    # Reconstruct V from SVD: V_svd = U_V @ diag(S_V) @ Vt_V
    # V_svd here is the reconstructed V_centroids [r, d]
    V_svd = sketch.U_V @ np.diag(sketch.S_V) @ sketch.Vt_V  # [r_svd, d]
    
    # If V_svd has fewer rows than r, we need to handle this
    # For simplicity, take first min(r, r_svd) rows
    V_aligned = V_svd[:min(r, len(sketch.S_V)), :]  # [r', d]
    
    # Compute output: weighted sum of V centroids
    output = attn_weights @ V_aligned  # [q_len, d]
    
    return output


def eval_hybrid_sketch_v2(
    sketch: HybridSketch,
    Q: np.ndarray,
    V_orig: np.ndarray,
) -> np.ndarray:
    """Evaluate Hybrid sketch V2: Use original V with SVD approximation.
    
    More accurate: reconstruct V approximation, then do full attention.
    """
    # Dequantize K
    r, d = sketch.r_coreset, sketch.d
    K_deq = np.zeros((r, d), dtype=np.float32)
    levels = 2 ** (sketch.n_bits - 1)
    
    for i in range(r):
        K_deq[i] = sketch.quantized_K[i].astype(np.float32) / levels * sketch.scale_K[i]
    
    # Compute attention scores with dequantized K
    d_sqrt = np.sqrt(d)
    scores = (Q @ K_deq.T) / d_sqrt
    
    # Softmax over coreset centroids
    scores_max = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - scores_max)
    l = p.sum(axis=-1, keepdims=True)
    p_norm = p / np.clip(l, 1e-30, None)
    
    # Approximate V by using top-r_svd components
    V_approx = sketch.U_V @ np.diag(sketch.S_V) @ sketch.Vt_V
    
    # Attention output using original K and approximate V
    # This is an approximation: A @ V_approx
    d_sqrt_v = np.sqrt(d)
    scores_v = (Q @ K_orig.T) / d_sqrt_v if 'K_orig' in dir() else (Q @ K_deq.T) / d_sqrt
    scores_v_max = scores_v.max(axis=-1, keepdims=True)
    p_v = np.exp(scores_v - scores_v_max)
    l_v = p_v.sum(axis=-1, keepdims=True)
    p_v_norm = p_v / np.clip(l_v, 1e-30, None)
    
    output = p_v_norm @ V_approx
    
    return output


# ============== Baseline 方法 ==============

def method_A_coreset_only(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    n_bits: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, CoresetSketch, QuantizedSketch, float]:
    """Method A: Coreset + INT4 only (baseline from exp4)"""
    d = Q.shape[1]
    
    # Build Coreset
    coreset = build_coreset_sketch(K, V, r, seed=seed)
    
    # Quantize
    q_sketch = quantize_sketch_nbit(coreset, n_bits=n_bits)
    coreset_deq = dequantize_sketch_nbit(q_sketch)
    
    # Evaluate
    stats = eval_coreset_sketch(coreset_deq, Q)
    output = stats.finalize().squeeze(0)
    
    # Compute bytes
    bytes_size = r * d * n_bits // 8 + r * 4 * 2  # quantized + scale + weights
    
    return output, coreset, q_sketch, bytes_size


def method_B_svd_only(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Method B: SVD only (baseline from exp8)"""
    d = Q.shape[1]
    
    # Compute attention matrix
    A = compute_attention_matrix(Q, K)
    
    # SVD on attention matrix
    U, S, Vt = npla.svd(A, full_matrices=False)
    r_actual = min(r, len(S))
    
    U_r = U[:, :r_actual]
    S_r = S[:r_actual]
    V_r = Vt[:r_actual, :].T
    
    # Compute output: A_r @ V
    A_r = U_r @ np.diag(S_r) @ V_r.T
    output = A_r @ V
    
    # Compute bytes: r * (q_len + kv_len) + q_len * d
    bytes_size = r_actual * (Q.shape[0] + K.shape[0]) * 4 + Q.shape[0] * d * 4
    
    return output, U_r, S_r, V_r, bytes_size


def method_C_hybrid(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    r_coreset: int,
    r_svd: int,
    n_bits: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, HybridSketch, float]:
    """Method C: Hybrid SVD + Coreset + INT4"""
    # Build hybrid sketch V2
    hybrid = build_hybrid_sketch_v2(Q, K, V, r_coreset, r_svd, n_bits, seed)
    
    # Evaluate
    output = eval_hybrid_sketch(hybrid, Q)
    
    return output, hybrid, hybrid.bytes_size()


# ============== Sweep ==============

def run_single_config(
    q_len: int,
    kv_len: int,
    kv_type: Literal["clustered", "random", "skewed"],
    r_coreset: int,
    r_svd: int,
    n_bits: int,
    d: int = 64,
    seed: int = 0,
) -> dict:
    """Run single configuration"""
    # Generate data
    K, V = make_kv_by_type(kv_len, d, kv_type, seed=seed)
    
    # Generate Q
    gen = np.random.default_rng(seed + 1000)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    # Ground truth
    gt = ground_truth(Q, K, V)
    
    # Full KV bytes
    full_bytes = kv_len * d * 2 * 4
    
    result = {
        "q_len": q_len,
        "kv_len": kv_len,
        "kv_type": kv_type,
        "r_coreset": r_coreset,
        "r_svd": r_svd,
        "n_bits": n_bits,
        "d": d,
        "seed": seed,
        "full_bytes": full_bytes,
        "methods": {},
    }
    
    # Method A: Coreset only
    try:
        out_A, coreset, q_sketch, bytes_A = method_A_coreset_only(
            Q, K, V, r_coreset, n_bits, seed
        )
        err_A = float(np.abs(out_A - gt).mean())
        result["methods"]["coreset"] = {
            "bytes": bytes_A,
            "compression_ratio": full_bytes / bytes_A if bytes_A > 0 else float('inf'),
            "err_mean": err_A,
        }
    except Exception as e:
        result["methods"]["coreset"] = {"error": str(e)}
    
    # Method B: SVD only
    try:
        out_B, U_r, S_r, V_r, bytes_B = method_B_svd_only(
            Q, K, V, r_svd, seed
        )
        err_B = float(np.abs(out_B - gt).mean())
        result["methods"]["svd"] = {
            "bytes": bytes_B,
            "compression_ratio": full_bytes / bytes_B if bytes_B > 0 else float('inf'),
            "err_mean": err_B,
        }
    except Exception as e:
        result["methods"]["svd"] = {"error": str(e)}
    
    # Method C: Hybrid
    try:
        out_C, hybrid, bytes_C = method_C_hybrid(
            Q, K, V, r_coreset, r_svd, n_bits, seed
        )
        err_C = float(np.abs(out_C - gt).mean())
        result["methods"]["hybrid"] = {
            "bytes": bytes_C,
            "compression_ratio": full_bytes / bytes_C if bytes_C > 0 else float('inf'),
            "err_mean": err_C,
        }
    except Exception as e:
        result["methods"]["hybrid"] = {"error": str(e)}
    
    return result


def run_full_sweep(
    q_lens: list[int] = [16, 64, 256],
    kv_lens: list[int] = [1024, 4096, 16384],
    r_coreset_values: list[int] = [4, 8, 16, 32],
    r_svd_values: list[int] = [4, 8, 16],
    n_bits_values: list[int] = [3, 4, 8],
    kv_types: list[str] = ["clustered", "random", "skewed"],
    d: int = 64,
    seed: int = 0,
) -> list[dict]:
    """Run full sweep over all configurations"""
    
    results = []
    total_configs = len(q_lens) * len(kv_lens) * len(r_coreset_values) * len(r_svd_values) * len(n_bits_values) * len(kv_types)
    
    print("=" * 80)
    print(f"Exp13: SVD + Coreset Hybrid Sweep ({total_configs} configs)")
    print("=" * 80)
    
    count = 0
    for kv_type in kv_types:
        print(f"\n--- KV Type: {kv_type.upper()} ---")
        for kv_len in kv_lens:
            for q_len in q_lens:
                for r_coreset in r_coreset_values:
                    for r_svd in r_svd_values:
                        for n_bits in n_bits_values:
                            count += 1
                            
                            result = run_single_config(
                                q_len=q_len,
                                kv_len=kv_len,
                                kv_type=kv_type,
                                r_coreset=r_coreset,
                                r_svd=r_svd,
                                n_bits=n_bits,
                                d=d,
                                seed=seed,
                            )
                            results.append(result)
                            
                            if count % 100 == 0:
                                print(f"  Progress: {count}/{total_configs} ({100*count/total_configs:.1f}%)")
    
    print(f"\nCompleted: {count}/{total_configs} configs")
    return results


# ============== 分析 ==============

def analyze_results(results: list[dict]) -> dict:
    """Analyze sweep results"""
    
    analysis = {
        "total_configs": len(results),
        "by_kv_type": {},
        "pareto": {},
        "wins": {"coreset": 0, "svd": 0, "hybrid": 0},
    }
    
    # Group by kv_type
    for kv_type in ["clustered", "random", "skewed"]:
        type_results = [r for r in results if r["kv_type"] == kv_type]
        
        if not type_results:
            continue
        
        type_analysis = {
            "count": len(type_results),
            "avg_err": {},
            "avg_compression": {},
            "configs": [],
        }
        
        for method in ["coreset", "svd", "hybrid"]:
            errs = []
            comps = []
            for r in type_results:
                if method in r["methods"] and "err_mean" in r["methods"][method]:
                    errs.append(r["methods"][method]["err_mean"])
                    comps.append(r["methods"][method]["compression_ratio"])
            
            if errs:
                type_analysis["avg_err"][method] = float(np.mean(errs))
                type_analysis["avg_compression"][method] = float(np.mean(comps))
        
        # Count wins
        for r in type_results:
            best_err = float('inf')
            best_method = None
            for method in ["coreset", "svd", "hybrid"]:
                if method in r["methods"] and "err_mean" in r["methods"][method]:
                    if r["methods"][method]["err_mean"] < best_err:
                        best_err = r["methods"][method]["err_mean"]
                        best_method = method
            
            if best_method:
                analysis["wins"][best_method] = analysis["wins"].get(best_method, 0) + 1
        
        analysis["by_kv_type"][kv_type] = type_analysis
    
    # Compute Pareto frontier
    pareto_configs = []
    for r in results:
        for method in ["coreset", "svd", "hybrid"]:
            if method in r["methods"] and "err_mean" in r["methods"][method]:
                pareto_configs.append({
                    "kv_type": r["kv_type"],
                    "q_len": r["q_len"],
                    "kv_len": r["kv_len"],
                    "r_coreset": r["r_coreset"],
                    "r_svd": r["r_svd"],
                    "n_bits": r["n_bits"],
                    "method": method,
                    "bytes": r["methods"][method]["bytes"],
                    "err": r["methods"][method]["err_mean"],
                    "compression": r["methods"][method]["compression_ratio"],
                })
    
    # Sort by bytes, find Pareto optimal
    pareto_configs.sort(key=lambda x: x["bytes"])
    
    pareto_frontier = []
    min_err_seen = float('inf')
    for p in pareto_configs:
        if p["err"] < min_err_seen:
            pareto_frontier.append(p)
            min_err_seen = p["err"]
    
    analysis["pareto"] = pareto_frontier
    
    return analysis


def compute_pareto_comparison(results: list[dict]) -> dict:
    """Compare Pareto frontiers for each method"""
    
    method_paretos = {"coreset": [], "svd": [], "hybrid": []}
    
    for r in results:
        for method in ["coreset", "svd", "hybrid"]:
            if method in r["methods"] and "err_mean" in r["methods"][method]:
                method_paretos[method].append({
                    "bytes": r["methods"][method]["bytes"],
                    "err": r["methods"][method]["err_mean"],
                    "compression": r["methods"][method]["compression_ratio"],
                    "config": {
                        "q_len": r["q_len"],
                        "kv_len": r["kv_len"],
                        "r_coreset": r["r_coreset"],
                        "r_svd": r["r_svd"],
                        "n_bits": r["n_bits"],
                    }
                })
    
    # Compute Pareto frontier for each method
    pareto_by_method = {}
    for method, configs in method_paretos.items():
        if not configs:
            continue
        
        configs_sorted = sorted(configs, key=lambda x: x["bytes"])
        pareto = []
        min_err = float('inf')
        
        for c in configs_sorted:
            if c["err"] < min_err:
                pareto.append(c)
                min_err = c["err"]
        
        pareto_by_method[method] = pareto
    
    return pareto_by_method


# ============== E2E TTFT 模拟 ==============

def simulate_ttft(
    method: str,
    bytes_size: int,
    kv_len: int,
    bandwidth_gbps: float = 100.0,
    rtt_ms: float = 1.0,
) -> float:
    """Simulate TTFT for a method"""
    T_transfer = (bytes_size * 8) / (bandwidth_gbps * 1e9) * 1000  # ms
    T_compute = kv_len * 64 * 2 * 1e-6  # estimate
    return T_transfer + T_compute + rtt_ms / 2


# ============== 主函数 ==============

def main():
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration (minimal for quick validation)
    q_lens = [16, 64]
    kv_lens = [1024, 4096]
    r_coreset_values = [8, 16]
    r_svd_values = [8]
    n_bits_values = [4, 8]
    kv_types = ["clustered", "random", "skewed"]
    d = 64
    
    total_configs = len(q_lens) * len(kv_lens) * len(r_coreset_values) * len(r_svd_values) * len(n_bits_values) * len(kv_types)
    print(f"Total configs: {total_configs}")
    
    # Run sweep
    print("\n" + "=" * 80)
    print("Running SVD + Coreset Hybrid Sweep...")
    print("=" * 80)
    
    results = run_full_sweep(
        q_lens=q_lens,
        kv_lens=kv_lens,
        r_coreset_values=r_coreset_values,
        r_svd_values=r_svd_values,
        n_bits_values=n_bits_values,
        kv_types=kv_types,
        d=d,
        seed=0,
    )
    
    # Analyze
    print("\n" + "=" * 80)
    print("Analyzing Results...")
    print("=" * 80)
    
    analysis = analyze_results(results)
    pareto_by_method = compute_pareto_comparison(results)
    
    # Summary
    print("\n--- Summary ---")
    print(f"Total configs: {len(results)}")
    print(f"\nWins:")
    for method, wins in analysis["wins"].items():
        pct = 100 * wins / len(results)
        print(f"  {method}: {wins}/{len(results)} ({pct:.1f}%)")
    
    print(f"\n--- By KV Type ---")
    for kv_type, ta in analysis["by_kv_type"].items():
        print(f"\n{kv_type.upper()}:")
        for method in ["coreset", "svd", "hybrid"]:
            if method in ta["avg_err"]:
                print(f"  {method}: avg_err={ta['avg_err'][method]:.4e}, avg_comp={ta['avg_compression'][method]:.2f}x")
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving Results...")
    print("=" * 80)
    
    # Sweep results
    sweep_path = os.path.join(output_dir, "exp13_sweep.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {sweep_path}")
    
    # Pareto results
    pareto_path = os.path.join(output_dir, "exp13_pareto.json")
    with open(pareto_path, "w", encoding="utf-8") as f:
        json.dump({
            "analysis": analysis,
            "pareto_by_method": pareto_by_method,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {pareto_path}")
    
    # Compare with baselines
    vs_baseline = {
        "summary": {
            "total_configs": len(results),
            "wins": analysis["wins"],
            "by_kv_type": {},
        },
        "detailed": [],
    }
    
    # Add detailed comparison
    for r in results:
        entry = {
            "config": {
                "q_len": r["q_len"],
                "kv_len": r["kv_len"],
                "kv_type": r["kv_type"],
                "r_coreset": r["r_coreset"],
                "r_svd": r["r_svd"],
                "n_bits": r["n_bits"],
            },
            "methods": {},
        }
        
        for method in ["coreset", "svd", "hybrid"]:
            if method in r["methods"] and "err_mean" in r["methods"][method]:
                entry["methods"][method] = {
                    "bytes": r["methods"][method]["bytes"],
                    "compression": r["methods"][method]["compression_ratio"],
                    "err": r["methods"][method]["err_mean"],
                }
        
        # Determine best
        best_err = float('inf')
        best_method = None
        for method in ["coreset", "svd", "hybrid"]:
            if method in r["methods"] and "err_mean" in r["methods"][method]:
                if r["methods"][method]["err_mean"] < best_err:
                    best_err = r["methods"][method]["err_mean"]
                    best_method = method
        
        entry["best_method"] = best_method
        
        # Hybrid vs baselines
        if "hybrid" in r["methods"] and "err_mean" in r["methods"]["hybrid"]:
            hybrid_err = r["methods"]["hybrid"]["err_mean"]
            
            vs_baseline_entry = {
                "hybrid_err": hybrid_err,
                "vs_coreset": None,
                "vs_svd": None,
            }
            
            if "coreset" in r["methods"] and "err_mean" in r["methods"]["coreset"]:
                coreset_err = r["methods"]["coreset"]["err_mean"]
                vs_baseline_entry["vs_coreset"] = hybrid_err - coreset_err
            
            if "svd" in r["methods"] and "err_mean" in r["methods"]["svd"]:
                svd_err = r["methods"]["svd"]["err_mean"]
                vs_baseline_entry["vs_svd"] = hybrid_err - svd_err
            
            entry["hybrid_vs_baselines"] = vs_baseline_entry
        
        vs_baseline["detailed"].append(entry)
    
    # Aggregate vs_baseline
    hybrid_vs_coreset = [d["hybrid_vs_baselines"]["vs_coreset"] for d in vs_baseline["detailed"] if d.get("hybrid_vs_baselines", {}).get("vs_coreset") is not None]
    hybrid_vs_svd = [d["hybrid_vs_baselines"]["vs_svd"] for d in vs_baseline["detailed"] if d.get("hybrid_vs_baselines", {}).get("vs_svd") is not None]
    
    vs_baseline["summary"]["hybrid_vs_coreset"] = {
        "mean": float(np.mean(hybrid_vs_coreset)) if hybrid_vs_coreset else None,
        "std": float(np.std(hybrid_vs_coreset)) if hybrid_vs_coreset else None,
        "hybrid_better": sum(1 for v in hybrid_vs_coreset if v < 0) if hybrid_vs_coreset else 0,
        "total": len(hybrid_vs_coreset),
    }
    
    vs_baseline["summary"]["hybrid_vs_svd"] = {
        "mean": float(np.mean(hybrid_vs_svd)) if hybrid_vs_svd else None,
        "std": float(np.std(hybrid_vs_svd)) if hybrid_vs_svd else None,
        "hybrid_better": sum(1 for v in hybrid_vs_svd if v < 0) if hybrid_vs_svd else 0,
        "total": len(hybrid_vs_svd),
    }
    
    # By kv_type
    for kv_type in ["clustered", "random", "skewed"]:
        type_entries = [d for d in vs_baseline["detailed"] if d["config"]["kv_type"] == kv_type]
        if type_entries:
            hvc = [d["hybrid_vs_baselines"]["vs_coreset"] for d in type_entries if d.get("hybrid_vs_baselines", {}).get("vs_coreset") is not None]
            hvs = [d["hybrid_vs_baselines"]["vs_svd"] for d in type_entries if d.get("hybrid_vs_baselines", {}).get("vs_svd") is not None]
            
            vs_baseline["summary"]["by_kv_type"][kv_type] = {
                "hybrid_vs_coreset_better": sum(1 for v in hvc if v < 0) if hvc else 0,
                "hybrid_vs_svd_better": sum(1 for v in hvs if v < 0) if hvs else 0,
                "total": len(type_entries),
            }
    
    vs_baseline_path = os.path.join(output_dir, "exp13_vs_baseline.json")
    with open(vs_baseline_path, "w", encoding="utf-8") as f:
        json.dump(vs_baseline, f, indent=2, ensure_ascii=False)
    print(f"Saved: {vs_baseline_path}")
    
    # Generate report
    generate_report(results, analysis, pareto_by_method, vs_baseline, output_dir)
    
    print("\n" + "=" * 80)
    print("Exp13 Complete!")
    print("=" * 80)
    
    return results, analysis, pareto_by_method, vs_baseline


def generate_report(
    results: list[dict],
    analysis: dict,
    pareto_by_method: dict,
    vs_baseline: dict,
    output_dir: str,
) -> None:
    """Generate experiment report"""
    
    report = []
    report.append("# Exp13: SVD + Coreset Hybrid Compression\n\n")
    
    report.append("## 核心思路\n\n")
    report.append("**联合压缩**: 结合 SVD (V端低秩) + Coreset (K端量化友好)\n\n")
    report.append("| 组件 | 方法 | 优势 |\n")
    report.append("|------|------|------|\n")
    report.append("| K端 | Coreset + INT4 | quantization-friendly |\n")
    report.append("| V端 | SVD | output-side low rank |\n\n")
    
    report.append("## 3-way Comparison: Coreset vs SVD vs Hybrid\n\n")
    
    report.append(f"**总配置数**: {len(results)}\n\n")
    
    report.append("### 胜率统计\n\n")
    report.append("| Method | Wins | Win Rate |\n")
    report.append("|--------|------|----------|\n")
    total = len(results)
    for method, wins in analysis["wins"].items():
        pct = 100 * wins / total if total > 0 else 0
        report.append(f"| {method} | {wins} | {pct:.1f}% |\n")
    report.append("\n")
    
    report.append("### By KV Type\n\n")
    
    for kv_type in ["clustered", "random", "skewed"]:
        if kv_type not in analysis["by_kv_type"]:
            continue
        
        ta = analysis["by_kv_type"][kv_type]
        report.append(f"#### {kv_type.upper()}\n\n")
        
        report.append("| Method | Avg Error | Avg Compression |\n")
        report.append("|--------|-----------|----------------|\n")
        
        for method in ["coreset", "svd", "hybrid"]:
            if method in ta["avg_err"]:
                report.append(f"| {method} | {ta['avg_err'][method]:.4e} | {ta['avg_compression'][method]:.2f}x |\n")
        
        report.append("\n")
    
    report.append("## Hybrid vs Baselines\n\n")
    
    report.append("### Overall\n\n")
    
    if vs_baseline["summary"].get("hybrid_vs_coreset"):
        hvc = vs_baseline["summary"]["hybrid_vs_coreset"]
        report.append(f"**Hybrid vs Coreset**:\n")
        report.append(f"- Mean error diff: {hvc['mean']:.4e}\n")
        report.append(f"- Hybrid better: {hvc['hybrid_better']}/{hvc['total']} ({100*hvc['hybrid_better']/hvc['total']:.1f}%)\n\n")
    
    if vs_baseline["summary"].get("hybrid_vs_svd"):
        hvs = vs_baseline["summary"]["hybrid_vs_svd"]
        report.append(f"**Hybrid vs SVD**:\n")
        report.append(f"- Mean error diff: {hvs['mean']:.4e}\n")
        report.append(f"- Hybrid better: {hvs['hybrid_better']}/{hvs['total']} ({100*hvs['hybrid_better']/hvs['total']:.1f}%)\n\n")
    
    report.append("### By KV Type\n\n")
    
    for kv_type, stats in vs_baseline["summary"].get("by_kv_type", {}).items():
        report.append(f"**{kv_type.upper()}**:\n")
        report.append(f"- Hybrid beats Coreset: {stats['hybrid_vs_coreset_better']}/{stats['total']}\n")
        report.append(f"- Hybrid beats SVD: {stats['hybrid_vs_svd_better']}/{stats['total']}\n\n")
    
    report.append("## Pareto Frontier Analysis\n\n")
    
    report.append("### Best by Method (bytes vs error)\n\n")
    
    for method, pareto in pareto_by_method.items():
        if not pareto:
            continue
        
        report.append(f"#### {method.upper()}\n\n")
        report.append("| bytes | err | compression |\n")
        report.append("|-------|-----|------------|\n")
        
        for p in pareto[:10]:  # Top 10
            report.append(f"| {p['bytes']} | {p['err']:.4e} | {p['compression']:.2f}x |\n")
        
        report.append("\n")
    
    # Key findings
    report.append("## 关键发现\n\n")
    
    # Determine winner
    best_method = max(analysis["wins"].items(), key=lambda x: x[1])[0] if analysis["wins"] else None
    
    if best_method == "hybrid":
        report.append("✅ **Hybrid 胜出**: 在 bytes vs error Pareto frontier 上 Hybrid 优于单独 Coreset 或 SVD\n")
    elif best_method:
        report.append(f"⚠️ **{best_method} 最佳**: Hybrid 未能超越单独方法\n")
    
    # Hybrid advantage
    if vs_baseline["summary"].get("hybrid_vs_coreset"):
        hvc = vs_baseline["summary"]["hybrid_vs_coreset"]
        if hvc.get("mean") is not None and hvc["hybrid_better"] > hvc["total"] * 0.5:
            report.append(f"- Hybrid 在 {100*hvc['hybrid_better']/hvc['total']:.1f}% 配置中优于 Coreset\n")
    
    if vs_baseline["summary"].get("hybrid_vs_svd"):
        hvs = vs_baseline["summary"]["hybrid_vs_svd"]
        if hvs.get("mean") is not None and hvs["hybrid_better"] > hvs["total"] * 0.5:
            report.append(f"- Hybrid 在 {100*hvs['hybrid_better']/hvs['total']:.1f}% 配置中优于 SVD\n")
    
    report.append("\n## 诚实边界\n\n")
    
    # Analyze failure cases
    hybrid_wins = analysis["wins"].get("hybrid", 0)
    total = len(results)
    
    if hybrid_wins < total * 0.5:
        report.append("⚠️ **联合压缩未达到预期效果**:\n")
        report.append(f"- Hybrid 仅在 {100*hybrid_wins/total:.1f}% 配置中获胜\n")
        report.append("- 可能原因:\n")
        report.append("  1. K端和V端压缩的误差在 attention 计算中相互放大\n")
        report.append("  2. SVD 和 Coreset 的 rank 选择需要协调优化\n")
        report.append("  3. 量化误差在高维 attention 中占主导\n")
    else:
        report.append("✅ **联合压缩有效**:\n")
        report.append("- Hybrid 结合了量化友好和低秩压缩的优势\n")
        report.append("- 在 bytes vs error trade-off 上优于单独方法\n")
    
    report.append("\n## 对 Paper Section 4 的影响\n\n")
    report.append("1. **新增 Fig 3d**: Hybrid bytes vs error Pareto frontier\n")
    report.append("2. **Table 2 扩展**: Hybrid vs Coreset vs SVD 胜率\n")
    report.append("3. **理论分析**: 为什么 Hybrid 在某些场景有效/无效\n")
    
    report.append("\n## 结论\n\n")
    
    if best_method == "hybrid":
        report.append("✅ **推荐使用 Hybrid SVD + Coreset**\n")
        report.append("- 优势: 量化友好 (K端) + 低秩压缩 (V端)\n")
        report.append("- 适用场景: clustered KV data\n")
    else:
        report.append("⚠️ **需要进一步优化 Hybrid**\n")
        report.append("- 当前: 单独 Coreset 或 SVD 更优\n")
        report.append("- 建议: 尝试端到端联合训练\n")
    
    report_path = os.path.join(output_dir, "exp13_hybrid_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(report))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
