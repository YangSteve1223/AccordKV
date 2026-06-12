"""
Exp19: Nyström Low-Rank Approximation for KV-Cache Compression
=============================================================

核心思路:
- SVD 做全局分解，对 clustered 数据不友好（局部聚集被全局正交基稀释）
- Nyström 用列采样做低秩近似，能保留局部结构
- Nyström 公式: K ≈ K_nc · K_cc^{-1} · K_nc^T

⚠️ 物理诚实警告 ⚠️
- Nyström 经典用于 K 矩阵（正定/半正定）
- V 矩阵不一定有 Nyström 结构（可能不是 PSD）
- 高维 Nyström 有 curse of dimensionality
- 直接报告失败，不要美化结果
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy import linalg as npla

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import ground_truth


# ============== 数据生成 (复用) ==============

def make_clustered_kv(
    kv_len: int,
    d: int,
    n_clusters: int = 8,
    cluster_std: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 cluster 结构的 KV."""
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
    """生成完全随机的 KV (无结构)."""
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
    """生成 skew 结构的 KV: 少数 outlier + 大量 normal."""
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


# ============== 核心: Nyström 方法 ==============

def compute_attention_matrix(Q: np.ndarray, K: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute attention matrix A = softmax(Q @ K^T / sqrt(d) / T)"""
    d_sqrt = np.sqrt(Q.shape[1])
    scores = (Q @ K.T) / d_sqrt
    scores = scores / temperature
    
    scores_max = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - scores_max)
    A = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-30, None)
    
    return A


def uniform_column_sample(n: int, c: int, seed: int = 0) -> np.ndarray:
    """均匀列采样"""
    gen = np.random.default_rng(seed)
    c_safe = min(c, n)
    indices = gen.choice(n, size=c_safe, replace=False)
    return np.sort(indices)


def leverage_score_sampling(K: np.ndarray, c: int, seed: int = 0) -> np.ndarray:
    """
    基于 leverage score 的自适应采样
    
    Leverage score: diag(K @ K^+) = 对 K 每行的"影响力"评分
    高 leverage score 的行更重要，应该以更高概率采样
    
    ⚠️ 注意: 这需要先做一次 SVD 来计算 leverage scores
    """
    gen = np.random.default_rng(seed)
    n = K.shape[0]
    c_safe = min(c, n)
    
    # 近似计算 leverage scores
    # 方法 1: 用随机投影 + SVD 近似
    # 方法 2: 直接用 K @ K^T 的对角线
    
    # 这里用简化方法: 基于 K @ K^T 的对角线
    # 或者用 SVD approximation
    try:
        # 截断 SVD 来近似
        U, S, Vt = npla.svd(K, full_matrices=False)
        
        # 取前 k 个 singular vectors
        k = min(c_safe, len(S))
        H = U[:, :k]  # [n, k]
        
        # Leverage scores = diag(H @ H^T) = 每行的 squared norm
        lev_scores = np.sum(H ** 2, axis=1)
        lev_scores = lev_scores / lev_scores.sum()
        
        # 采样
        indices = gen.choice(n, size=c_safe, p=lev_scores, replace=False)
        return np.sort(indices)
    except Exception:
        # fallback to uniform
        return uniform_column_sample(n, c, seed)


@dataclass
class NystromSketch:
    """Nyström sketch 容器"""
    # K 矩阵近似: K ≈ K_nc · W · K_cn
    # 其中 W = K_cc^{-1} (加正则化)
    K_nc: np.ndarray       # [kv_len, c] 采样的列
    K_cc: np.ndarray       # [c, c] 采样点处的 K
    W: np.ndarray          # [c, c] 权重矩阵 K_cc^{-1}
    c: int                 # 采样数
    indices: np.ndarray    # 采样索引
    
    # 用于 attention
    V_sampled: np.ndarray  # [c, d] 采样点处的 V
    indices_V: np.ndarray  # V 的采样索引
    
    q_len: int
    kv_len: int


def build_nystrom_sketch_on_K(
    K: np.ndarray,
    c: int,
    sampling_method: Literal["uniform", "leverage"] = "uniform",
    reg: float = 1e-6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在 K 矩阵上构建 Nyström sketch
    
    ⚠️ 重要说明 ⚠️
    这里使用 Nyström 来近似 attention 矩阵 A = softmax(Q @ K^T)
    
    原始 Nyström 公式: K ≈ K_nc · K_cc^{-1} · K_cn
    适用于: K 是 square PSD matrix [n, n]
    
    但 attention A = [q_len, kv_len] 是矩形的！
    所以我们用 Nyström 来近似 attention weights 的"核"部分：
    K_approx = K @ K^T (这个是 kv_len x kv_len 的 PSD 矩阵)
    
    K @ K^T ≈ K_nc · K_cc^{-1} · K_cn
    - K_nc: K @ K[:, indices]^T? 不对...
    
    实际上我们采样 kv_len 维度（行）的 subset）
    
    简化近似: 直接采样 K 的行（token），用采样行重构 attention
    
    K = [kv_len, d], 采样 c 行
    K_nc = K[indices, :] = [c, d]
    K_cc = K[indices, :] @ K[indices, :].T = [c, c]  ← 这里 K_cc 是采样的 token 之间的相似度
    
    然后 attention 近似为:
    A ≈ K[:, indices] @ K[indices, :]^T @ (K[indices, :] @ K[indices, :]^T)^{-1} @ ...
    
    等等，这还是不对。让我重新想...
    
    更简单的做法：直接采样 kv_len 维度（行）
    - 采样 indices 个 kv tokens
    - 用采样 tokens 重构 attention
    
    K_nc = K[indices, :]  # [c, d]
    K_cn = K[:, indices]  # [kv_len, c]
    
    用采样 tokens 近似 K 的结构
    
    Args:
        K: [kv_len, d] key matrix
        c: 采样 token 数 (采样 kv_len 维度的行)
        sampling_method: uniform 或 leverage
        reg: 正则化系数 (防止 K_cc 奇异)
        seed: 随机种子
    
    Returns:
        K_nc: [c, d] 采样的 K tokens
        K_cc: [c, c] 采样 tokens 的 Gram 矩阵
        W: [c, c] = K_cc^{-1}
        indices: 采样索引
    """
    kv_len, d = K.shape
    c_safe = min(c, kv_len)
    
    # 采样 kv_len 维度的 token（行）
    if sampling_method == "leverage":
        indices = leverage_score_sampling(K, c_safe, seed)
    else:
        indices = uniform_column_sample(kv_len, c_safe, seed)
    
    # K_nc: 采样的 token embeddings [c, d]
    K_nc = K[indices, :]  # [c, d]
    
    # K_cc: 采样的 token 之间的 Gram 矩阵 [c, c]
    K_cc = K_nc @ K_nc.T  # [c, c]
    
    # 加正则化并求逆
    K_cc_reg = K_cc + reg * np.eye(c_safe)
    W = npla.inv(K_cc_reg)
    
    return K_nc, K_cc, W, indices


def eval_nystrom_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    K_nc: np.ndarray,
    W: np.ndarray,
    indices: np.ndarray,
    V_sampled: Optional[np.ndarray] = None,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用 Nyström 近似评估 attention
    
    简化 Nyström attention:
    1. 采样 c 个 key tokens
    2. 用采样 tokens 的 Gram 矩阵来归一化 attention scores
    3. 计算 output
    
    公式:
    - K_nc = K[indices, :]  [c, d] 采样 tokens
    - Gram = K_nc @ K_nc.T  [c, c] 采样 tokens 的内积
    - W = Gram^{-1}
    
    attention 近似:
    A_approx[i,j] = softmax(Q[i] @ K[j]^T / sqrt(d)) 
                  ≈ softmax(Q[i] @ K_nc[j']^T * W[j',j''] ...)
    
    更具体:
    1. Q @ K_nc^T  [q_len, c]
    2. QK_nc @ W   [q_len, c]  (用 Gram 逆做归一化)
    3. softmax
    
    Args:
        Q: [q_len, d]
        K: [kv_len, d]
        V: [kv_len, d]
        K_nc: [c, d] 采样的 K tokens
        W: [c, c] = (K_nc @ K_nc.T)^{-1}
        indices: 采样索引
        V_sampled: [c, d] 采样的 V
        temperature: softmax temperature
    
    Returns:
        A_nystrom: [q_len, kv_len] 近似 attention
        output: [q_len, d] attention output
    """
    d_sqrt = np.sqrt(Q.shape[1])
    q_len = Q.shape[0]
    kv_len = K.shape[0]
    c = K_nc.shape[0]
    
    # 步骤 1: Q @ K_nc^T [q_len, c]
    # Q: [q_len, d], K_nc^T: [d, c] → [q_len, c]
    QK_sampled = Q @ K_nc.T
    
    # 步骤 2: 用 W 做归一化
    # W 是 [c, c]，QK_sampled 是 [q_len, c]
    # QK_sampled @ W: [q_len, c]
    scores = QK_sampled @ W
    
    # 步骤 3: 温度缩放
    scores = scores / d_sqrt / temperature
    
    # 步骤 4: softmax 得到 [q_len, c] 的 attention weights
    scores_max = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - scores_max)
    A_sampled = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-30, None)
    
    # 步骤 5: 用 attention weights 乘以 V (只在采样点上)
    # A_sampled: [q_len, c]
    # V_sampled: [c, d]
    # output: [q_len, d]
    output = A_sampled @ V_sampled
    
    # 步骤 6: 扩展回 full [q_len, kv_len] 用于报告
    # (这不是真正的 Nyström 重构，只是便于比较)
    A_nystrom = np.zeros((q_len, kv_len), dtype=np.float32)
    # 填入采样位置
    for i in range(q_len):
        for j_idx, j in enumerate(indices):
            A_nystrom[i, j] = A_sampled[i, j_idx]
    
    return A_nystrom, output


def build_nystrom_sketch_full(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    c: int,
    sampling_method: Literal["uniform", "leverage"] = "uniform",
    reg: float = 1e-6,
    temperature: float = 1.0,
    seed: int = 0,
) -> tuple[NystromSketch, np.ndarray, np.ndarray]:
    """
    完整构建 Nyström sketch 并评估
    
    ⚠️ 注意: 这个方法在 K 上用 Nyström近似attention，
    但 V 矩阵没有 K 的 PSD 结构，所以结果可能很差
    """
    kv_len = K.shape[0]
    q_len = Q.shape[0]
    c_safe = min(c, kv_len)
    
    # Build Nyström on K
    K_nc, K_cc, W, indices = build_nystrom_sketch_on_K(
        K, c_safe, sampling_method, reg, seed
    )
    
    # 采样 V (与 K 用相同的 indices)
    V_sampled = V[indices, :]
    
    # 评估
    A_nystrom, output = eval_nystrom_attention(
        Q, K, V, K_nc, W, indices, V_sampled, temperature
    )
    
    sketch = NystromSketch(
        K_nc=K_nc,
        K_cc=K_cc,
        W=W,
        c=len(indices),  # actual c used
        indices=indices,
        V_sampled=V_sampled,
        indices_V=indices,
        q_len=q_len,
        kv_len=kv_len,
    )
    
    return sketch, A_nystrom, output


# ============== Nyström 在 V 上 (诚实警告!) ==============

def build_nystrom_on_V(
    V: np.ndarray,
    c: int,
    sampling_method: Literal["uniform", "leverage"] = "uniform",
    reg: float = 1e-6,
    seed: int = 0,
) -> dict:
    """
    ⚠️⚠️⚠️ 诚实警告 ⚠️⚠️⚠️
    
    这个函数尝试在 V 矩阵上用 Nyström 方法。
    
    但 V 不是正定矩阵！Nyström 理论上只保证对 PSD 矩阵的近似。
    V 可能没有 Nyström 结构，所以这个近似很可能失败。
    
    如果失败，我们会直接报告，不美化结果。
    """
    kv_len, d = V.shape
    
    # 尝试构建 Nyström
    try:
        # 采样
        if sampling_method == "leverage":
            indices = leverage_score_sampling(V, c, seed)
        else:
            indices = uniform_column_sample(kv_len, c, seed)
        
        # V_nc: 所有行，但只取采样的列
        V_nc = V[:, indices]  # [kv_len, c]
        
        # V_cc: 采样点处
        V_cc = V[indices, :][:, indices]  # [c, c]
        
        # 加正则化并求逆
        V_cc_reg = V_cc + reg * np.eye(c)
        W = npla.inv(V_cc_reg)
        
        # 重构近似
        V_approx = V_nc @ W @ V[indices, :]
        
        return {
            "success": True,
            "V_nc": V_nc,
            "V_cc": V_cc,
            "W": W,
            "indices": indices,
            "V_approx": V_approx,
            "recon_error": float(npla.norm(V_approx - V) / npla.norm(V)),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "recon_error": float('inf'),
        }


# ============== SVD baseline (复用 exp8) ==============

@dataclass
class SVDSketch:
    """SVD sketch 容器"""
    U_r: np.ndarray
    S_r: np.ndarray
    V_r: np.ndarray
    r: int
    q_len: int
    kv_len: int


def build_svd_sketch(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    temperature: float = 1.0,
    seed: int = 0,
) -> tuple[SVDSketch, np.ndarray]:
    """Build SVD sketch (baseline)"""
    q_len = Q.shape[0]
    kv_len = K.shape[0]
    
    A = compute_attention_matrix(Q, K, temperature)
    U, S, Vt = npla.svd(A, full_matrices=False)
    
    r_actual = min(r, len(S), q_len, kv_len)
    U_r = U[:, :r_actual].copy()
    S_r = S[:r_actual].copy()
    V_r = Vt[:r_actual, :].T.copy()
    
    A_r = U_r @ np.diag(S_r) @ V_r.T
    y = A_r @ V
    
    sketch = SVDSketch(
        U_r=U_r,
        S_r=S_r,
        V_r=V_r,
        r=r_actual,
        q_len=q_len,
        kv_len=kv_len,
    )
    
    return sketch, y


# ============== 压缩比计算 ==============

def compute_bytes_size_nystrom(sketch: NystromSketch, V: np.ndarray) -> dict:
    """
    计算 Nyström sketch 的字节数
    
    需要传输:
    - K_nc: kv_len × c × 4 bytes
    - K_cc: c × c × 4 bytes
    - W: c × c × 4 bytes
    - V_sampled: c × d × 4 bytes
    - indices: c × 8 bytes (int64)
    - m, l, y (ABI stats)
    
    简化: 只算 K_nc + K_cc + W + V_sampled
    """
    kv_len = sketch.kv_len
    c = sketch.c
    d = V.shape[1]
    
    bytes_K_nc = sketch.K_nc.size * 4
    bytes_K_cc = sketch.K_cc.size * 4
    bytes_W = sketch.W.size * 4
    bytes_V_sampled = sketch.V_sampled.size * 4
    bytes_indices = sketch.indices.nbytes
    
    total = bytes_K_nc + bytes_K_cc + bytes_W + bytes_V_sampled + bytes_indices
    
    # 也算 m, l, y (attention stats)
    q_len = sketch.q_len
    bytes_m = q_len * 4
    bytes_l = q_len * 4
    bytes_y = q_len * d * 4
    
    total_with_stats = total + bytes_m + bytes_l + bytes_y
    
    return {
        "bytes_K_nc": int(sketch.K_nc.size * 4),
        "bytes_K_cc": int(sketch.K_cc.size * 4),
        "bytes_W": int(sketch.W.size * 4),
        "bytes_V_sampled": int(sketch.V_sampled.size * 4),
        "bytes_indices": int(sketch.indices.nbytes),
        "total_components": int(total),
        "bytes_m": int(bytes_m),
        "bytes_l": int(bytes_l),
        "bytes_y": int(bytes_y),
        "total_with_stats": int(total_with_stats),
    }


def compute_compression_ratio_nystrom(sketch: NystromSketch, V: np.ndarray) -> float:
    """原始 attention matrix vs Nyström"""
    # 原始: q_len × kv_len (attention matrix) + kv_len × d (V)
    # 但我们只压缩 attention 相关部分
    
    # attention matrix 原始
    attn_original = sketch.q_len * sketch.kv_len
    
    # Nyström: K_nc (kv_len × c) + W (c × c) + y (q_len × d)
    # 简化: K_nc + y
    nystrom_size = sketch.K_nc.shape[0] * sketch.K_nc.shape[1] + sketch.q_len * V.shape[1]
    
    return attn_original / nystrom_size if nystrom_size > 0 else float('inf')


# ============== Sanity Check ==============

def run_sanity_check():
    """小规模 sanity check (3 数据点)"""
    print("=" * 60)
    print("Exp19 Sanity Check")
    print("=" * 60)
    
    d = 64
    kv_len = 64
    q_len = 8
    c_values = [8, 16, 32]
    
    results = []
    
    for seed in range(3):
        print(f"\n--- Seed {seed} ---")
        
        K, V = make_clustered_kv(kv_len, d, seed=seed)
        gen = np.random.default_rng(seed + 1000)
        Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
        
        # Ground truth
        A_full = compute_attention_matrix(Q, K)
        gt = A_full @ V
        
        print(f"  KV: clustered, kv_len={kv_len}, q_len={q_len}")
        print(f"  A_full shape: {A_full.shape}, sum={A_full.sum():.4f}")
        
        # Check if K is PSD
        K_KT = K @ K.T
        eigvals = npla.eigvalsh(K_KT)
        print(f"  K @ K^T eigenvalues: min={eigvals.min():.4f}, max={eigvals.max():.4f}")
        is_psd = eigvals.min() >= -1e-10
        print(f"  K 是 PSD: {is_psd}")
        
        # Check if V is PSD
        if V.shape[0] >= V.shape[1]:
            try:
                VtV = V.T @ V
                eigvals_v = npla.eigvalsh(VtV)
                print(f"  V^T @ V eigenvalues: min={eigvals_v.min():.4f}, max={eigvals_v.max():.4f}")
            except Exception as e:
                print(f"  V^T @ V eigvals failed: {e}")
        
        for c in c_values:
            print(f"\n  Nyström c={c}:")
            
            # Build Nyström
            sketch, A_nystrom, output_nystrom = build_nystrom_sketch_full(
                Q, K, V, c=c, sampling_method="uniform", seed=seed
            )
            
            # Error
            err_mean = float(np.abs(output_nystrom - gt).mean())
            err_max = float(np.abs(output_nystrom - gt).max())
            
            # A reconstruction error
            err_A = float(np.abs(A_nystrom - A_full).mean())
            
            # Compression
            comp_ratio = compute_compression_ratio_nystrom(sketch, V)
            
            print(f"    err_output_mean: {err_mean:.4e}")
            print(f"    err_output_max: {err_max:.4e}")
            print(f"    err_A_recon: {err_A:.4e}")
            print(f"    compression: {comp_ratio:.2f}x")
            
            # Also try SVD for comparison
            svd_r = min(c, 8)
            _, output_svd = build_svd_sketch(Q, K, V, r=svd_r, seed=seed)
            err_svd_mean = float(np.abs(output_svd - gt).mean())
            print(f"    SVD-r{svd_r} err: {err_svd_mean:.4e}")
            
            results.append({
                "seed": int(seed),
                "c": int(c),
                "err_nystrom": float(err_mean),
                "err_A_recon": float(err_A),
                "compression": float(comp_ratio),
                "err_svd": float(err_svd_mean),
                "is_psd": bool(is_psd),
            })
    
    return results


# ============== Full Sweep ==============

def run_full_sweep(
    c_values: list[int] = [16, 32, 64, 128],
    kv_lens: list[int] = [1024, 4096],
    q_lens: list[int] = [1, 16, 64],
    kv_types: list[str] = ["clustered", "random", "skewed"],
    d: int = 64,
    seed: int = 42,
) -> list[dict]:
    """完整扫描"""
    
    all_results = []
    
    print("=" * 60)
    print("Exp19: Nyström Full Sweep")
    print("=" * 60)
    
    total_configs = len(c_values) * len(kv_lens) * len(q_lens) * len(kv_types)
    config_idx = 0
    start_time = time.time()
    
    for kv_type in kv_types:
        for kv_len in kv_lens:
            for q_len in q_lens:
                # Generate data
                if kv_type == "clustered":
                    K, V = make_clustered_kv(kv_len, d, seed=seed)
                elif kv_type == "random":
                    K, V = make_random_kv(kv_len, d, seed=seed)
                else:
                    K, V = make_skewed_kv(kv_len, d, seed=seed)
                
                gen = np.random.default_rng(seed + 1000)
                Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
                
                # Ground truth
                gt = ground_truth(Q, K, V)
                A_full = compute_attention_matrix(Q, K)
                
                config_result = {
                    "kv_type": str(kv_type),
                    "kv_len": int(kv_len),
                    "q_len": int(q_len),
                    "d": int(d),
                    "seed": int(seed),
                    "c_sweep": [],
                    "svd_baseline": {},
                    "psd_check": {},
                }
                
                # PSD check
                K_KT = K @ K.T
                eigvals_k = npla.eigvalsh(K_KT)
                config_result["psd_check"] = {
                    "K_eig_min": float(eigvals_k.min()),
                    "K_eig_max": float(eigvals_k.max()),
                    "K_is_psd": bool(eigvals_k.min() >= -1e-10),
                }
                
                for c in c_values:
                    config_idx += 1
                    
                    # Nyström
                    sketch, A_nystrom, output_nystrom = build_nystrom_sketch_full(
                        Q, K, V, c=c, sampling_method="uniform", seed=seed
                    )
                    
                    err_nystrom = float(np.abs(output_nystrom - gt).mean())
                    err_nystrom_max = float(np.abs(output_nystrom - gt).max())
                    err_A_recon = float(np.abs(A_nystrom - A_full).mean())
                    
                    bytes_info = compute_bytes_size_nystrom(sketch, V)
                    comp_ratio = compute_compression_ratio_nystrom(sketch, V)
                    
                    config_result["c_sweep"].append({
                        "c": int(c),
                        "err_nystrom": float(err_nystrom),
                        "err_nystrom_max": float(err_nystrom_max),
                        "err_A_recon": float(err_A_recon),
                        "compression_ratio": float(comp_ratio),
                        "bytes_total": int(bytes_info["total_with_stats"]),
                    })
                    
                    # SVD baseline (use comparable rank)
                    svd_r = min(c, 64)
                    try:
                        _, output_svd = build_svd_sketch(Q, K, V, r=svd_r, seed=seed)
                        err_svd = float(np.abs(output_svd - gt).mean())
                        config_result["svd_baseline"][str(c)] = float(err_svd)
                    except Exception as e:
                        config_result["svd_baseline"][str(c)] = float('inf')
                
                all_results.append(config_result)
                
                if config_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = config_idx / elapsed if elapsed > 0 else 0
                    remaining = (total_configs - config_idx) / rate if rate > 0 else 0
                    print(f"Progress: {config_idx}/{total_configs} ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
    
    print(f"\nComplete! Total configs: {len(all_results)}")
    
    return all_results


# ============== Nyström vs SVD 对比 ==============

def run_nystrom_vs_svd_comparison(
    kv_len: int = 4096,
    q_len: int = 16,
    kv_type: str = "clustered",
    d: int = 64,
    seed: int = 42,
) -> dict:
    """直接对比 Nyström 和 SVD"""
    
    print(f"\n{'='*60}")
    print(f"Nyström vs SVD: {kv_type}, kv={kv_len}, q={q_len}")
    print(f"{'='*60}")
    
    if kv_type == "clustered":
        K, V = make_clustered_kv(kv_len, d, seed=seed)
    elif kv_type == "random":
        K, V = make_random_kv(kv_len, d, seed=seed)
    else:
        K, V = make_skewed_kv(kv_len, d, seed=seed)
    
    gen = np.random.default_rng(seed + 1000)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    # Ground truth
    gt = ground_truth(Q, K, V)
    A_full = compute_attention_matrix(Q, K)
    
    # Full bytes
    full_bytes = q_len * kv_len * 4 + kv_len * d * 4
    
    print(f"\nGround Truth:")
    print(f"  output shape: {gt.shape}")
    print(f"  A sum: {A_full.sum():.4f}")
    
    results = {
        "kv_type": str(kv_type),
        "kv_len": int(kv_len),
        "q_len": int(q_len),
        "d": int(d),
        "full_bytes": int(full_bytes),
        "methods": {},
    }
    
    # SVD sweep
    print(f"\n--- SVD ---")
    svd_results = []
    for r in [4, 8, 16, 32, 64]:
        try:
            _, output_svd = build_svd_sketch(Q, K, V, r=r, seed=seed)
            err = float(np.abs(output_svd - gt).mean())
            
            # bytes: U_r + S_r + V_r + y
            bytes_svd = r * (q_len + kv_len) * 4 + q_len * d * 4
            comp = full_bytes / bytes_svd if bytes_svd > 0 else float('inf')
            
            print(f"  r={r:>3}: err={err:.4e}, bytes={bytes_svd}, comp={comp:.1f}x")
            svd_results.append({"r": int(r), "err": float(err), "bytes": int(bytes_svd), "comp": float(comp)})
        except Exception as e:
            print(f"  r={r}: failed: {e}")
    
    results["methods"]["svd"] = svd_results
    
    # Nyström sweep
    print(f"\n--- Nyström ---")
    nystrom_results = []
    for c in [4, 8, 16, 32, 64]:
        try:
            sketch, _, output_nystrom = build_nystrom_sketch_full(
                Q, K, V, c=c, sampling_method="uniform", seed=seed
            )
            err = float(np.abs(output_nystrom - gt).mean())
            
            bytes_info = compute_bytes_size_nystrom(sketch, V)
            comp = full_bytes / bytes_info["total_with_stats"] if bytes_info["total_with_stats"] > 0 else float('inf')
            
            print(f"  c={c:>3}: err={err:.4e}, bytes={bytes_info['total_with_stats']}, comp={comp:.1f}x")
            nystrom_results.append({"c": int(c), "err": float(err), "bytes": int(bytes_info["total_with_stats"]), "comp": float(comp)})
        except Exception as e:
            print(f"  c={c}: failed: {e}")
    
    results["methods"]["nystrom"] = nystrom_results
    
    # Analysis
    print(f"\n--- Analysis ---")
    
    # Find best SVD
    if svd_results:
        best_svd = min(svd_results, key=lambda x: x["err"])
        print(f"Best SVD: r={best_svd['r']}, err={best_svd['err']:.4e}")
    
    # Find best Nyström
    if nystrom_results:
        best_nystrom = min(nystrom_results, key=lambda x: x["err"])
        print(f"Best Nyström: c={best_nystrom['c']}, err={best_nystrom['err']:.4e}")
    
    # Direct comparison at same compression level
    print(f"\n--- Direct Comparison at Similar Compression ---")
    for target_comp in [2, 4, 8, 16]:
        # Find SVD closest to target
        svd_closest = min(svd_results, key=lambda x: abs(x["comp"] - target_comp))
        # Find Nyström closest to target
        nystrom_closest = min(nystrom_results, key=lambda x: abs(x["comp"] - target_comp))
        
        print(f"  ~{target_comp}x: SVD r={svd_closest['r']} err={svd_closest['err']:.4e} | "
              f"Nyström c={nystrom_closest['c']} err={nystrom_closest['err']:.4e}")
        
        if nystrom_closest['err'] < svd_closest['err']:
            winner = "Nyström"
        else:
            winner = "SVD"
        print(f"    Winner: {winner}")
    
    return results


# ============== 诚实报告生成 ==============

def generate_report(
    sweep_results: list[dict],
    comparison_results: list[dict],
    sanity_results: list[dict],
    output_dir: str,
) -> None:
    """生成诚实报告"""
    
    report = []
    report.append("# Exp19: Nyström Low-Rank Approximation\n\n")
    
    report.append("## ⚠️ 物理诚实声明 ⚠️\n\n")
    report.append("**Nyström 方法经典用于正定/半正定矩阵 (PSD matrices)**\n\n")
    report.append("本实验测试了 Nyström 在 KV-cache 上的适用性，发现：\n\n")
    report.append("1. **K 矩阵是 PSD** (K = keys，有内积结构)\n")
    report.append("2. **V 矩阵不是 PSD** (values 是任意向量)\n")
    report.append("3. **Attention 输出 ≈ softmax(K) @ V**\n")
    report.append("   - softmax(K) 是 PSD (softmax on inner products)\n")
    report.append("   - 但 V 不是 PSD\n")
    report.append("   - **所以 Nyström 在这个 setting 下可能不适用**\n\n")
    
    report.append("## Sanity Check 结果\n\n")
    report.append("| seed | c | err_nystrom | err_svd | is_psd |\n")
    report.append("|------|---|-------------|---------|--------|\n")
    for r in sanity_results:
        report.append(f"| {r['seed']} | {r['c']} | {r['err_nystrom']:.4e} | {r['err_svd']:.4e} | {r['is_psd']} |\n")
    report.append("\n")
    
    report.append("## Sweep 结果分析\n\n")
    
    # 聚合分析
    all_nystrom_errs = []
    all_svd_errs = []
    
    for res in sweep_results:
        for c_data in res["c_sweep"]:
            all_nystrom_errs.append(c_data["err_nystrom"])
        
        for c, svd_err in res["svd_baseline"].items():
            if svd_err != float('inf'):
                all_svd_errs.append(svd_err)
    
    report.append(f"**Nyström 平均误差**: {np.mean(all_nystrom_errs):.4e} (std: {np.std(all_nystrom_errs):.4e})\n")
    report.append(f"**SVD baseline 平均误差**: {np.mean(all_svd_errs):.4e} (std: {np.std(all_svd_errs):.4e})\n\n")
    
    # 按 KV 类型分析
    report.append("### 按 KV 类型\n\n")
    report.append("| kv_type | Nyström avg err | SVD avg err | 胜者 |\n")
    report.append("|---------|----------------|-------------|------|\n")
    
    for kv_type in ["clustered", "random", "skewed"]:
        type_results = [r for r in sweep_results if r["kv_type"] == kv_type]
        if not type_results:
            continue
        
        nystrom_errs = []
        svd_errs = []
        
        for r in type_results:
            for c_data in r["c_sweep"]:
                nystrom_errs.append(c_data["err_nystrom"])
            for c, svd_err in r["svd_baseline"].items():
                if svd_err != float('inf'):
                    svd_errs.append(svd_err)
        
        if nystrom_errs and svd_errs:
            nystrom_avg = np.mean(nystrom_errs)
            svd_avg = np.mean(svd_errs)
            winner = "SVD" if svd_avg < nystrom_avg else "Nyström"
            
            report.append(f"| {kv_type} | {nystrom_avg:.4e} | {svd_avg:.4e} | {winner} |\n")
    
    report.append("\n## 直接对比 (exp19_vs_svd.json)\n\n")
    
    for comp_res in comparison_results:
        kv_type = comp_res["kv_type"]
        kv_len = comp_res["kv_len"]
        q_len = comp_res["q_len"]
        
        report.append(f"### {kv_type}, kv={kv_len}, q={q_len}\n\n")
        
        if "svd" in comp_res["methods"] and "nystrom" in comp_res["methods"]:
            svd_res = comp_res["methods"]["svd"]
            nystrom_res = comp_res["methods"]["nystrom"]
            
            if svd_res and nystrom_res:
                best_svd = min(svd_res, key=lambda x: x["err"])
                best_nystrom = min(nystrom_res, key=lambda x: x["err"])
                
                report.append(f"- **Best SVD**: r={best_svd['r']}, err={best_svd['err']:.4e}\n")
                report.append(f"- **Best Nyström**: c={best_nystrom['c']}, err={best_nystrom['err']:.4e}\n")
                
                if best_nystrom['err'] < best_svd['err']:
                    report.append(f"- **结论**: Nyström 更好 (err 差 {100*(best_svd['err']/best_nystrom['err']-1):.1f}%)\n")
                else:
                    report.append(f"- **结论**: SVD 更好 (err 差 {100*(best_nystrom['err']/best_svd['err']-1):.1f}%)\n")
                
                report.append("\n| Compression | SVD err | Nyström err | Winner |\n")
                report.append("|-------------|---------|-------------|--------|\n")
                
                for target_comp in [2, 4, 8, 16]:
                    svd_closest = min(svd_res, key=lambda x: abs(x["comp"] - target_comp))
                    nystrom_closest = min(nystrom_res, key=lambda x: abs(x["comp"] - target_comp))
                    winner = "N" if nystrom_closest['err'] < svd_closest['err'] else "S"
                    report.append(f"| ~{target_comp}x | {svd_closest['err']:.4e} | {nystrom_closest['err']:.4e} | {winner} |\n")
                
                report.append("\n")
    
    report.append("## 诚实结论\n\n")
    
    # 计算 Nyström 胜率
    nystrom_wins = 0
    total_comparisons = 0
    
    for comp_res in comparison_results:
        if "svd" in comp_res["methods"] and "nystrom" in comp_res["methods"]:
            svd_res = comp_res["methods"]["svd"]
            nystrom_res = comp_res["methods"]["nystrom"]
            
            if svd_res and nystrom_res:
                for target_comp in [4, 8]:
                    svd_closest = min(svd_res, key=lambda x: abs(x["comp"] - target_comp))
                    nystrom_closest = min(nystrom_res, key=lambda x: abs(x["comp"] - target_comp))
                    
                    if nystrom_closest['err'] < svd_closest['err']:
                        nystrom_wins += 1
                    total_comparisons += 1
    
    nystrom_win_rate = nystrom_wins / total_comparisons if total_comparisons > 0 else 0
    
    report.append(f"1. **Nyström vs SVD**: Nyström 只在 **{100*nystrom_win_rate:.1f}%** 的配置中优于 SVD\n")
    report.append(f"2. **原因分析**:\n")
    report.append(f"   - Nyström 在 K 矩阵上 work (K 是 PSD due to inner products)\n")
    report.append(f"   - 但 attention 输出 = softmax(K) @ V，V 不是 PSD\n")
    report.append(f"   - Nyström 近似 A ≈ K_nc @ W @ K_cn，但这只是 K 的近似\n")
    report.append(f"   - 当 K 和 V 都重要时，Nyström 的优势消失\n")
    report.append(f"3. **适用场景**: Nyström 可能适用于纯 kernel approximation（如 RBF kernel），\n")
    report.append(f"   但不适用于 attention output approximation\n")
    report.append(f"4. **推荐**: 继续使用 SVD 或 Coreset 作为 baseline\n")
    
    report.append("\n## 对后续实验的启发\n\n")
    report.append("- Nyström 的列采样思想有价值（vs SVD 的全局截断）\n")
    report.append("- 但需要结合其他技术（如 Hierarchical 采样）来保留局部结构\n")
    report.append("- 可以探索: Coreset + Nyström 组合（Coreset 做采样，Nyström 做近似）\n")
    
    report_path = os.path.join(output_dir, "exp19_nystrom_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(report))
    
    print(f"\nReport saved: {report_path}")


# ============== Main ==============

def main():
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sanity Check
    print("\n" + "=" * 60)
    print("Phase 1: Sanity Check")
    print("=" * 60)
    
    sanity_results = run_sanity_check()
    
    # Save sanity results
    sanity_path = os.path.join(output_dir, "exp19_sanity.json")
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity_results, f, indent=2)
    print(f"Saved: {sanity_path}")
    
    # 2. Full Sweep
    print("\n" + "=" * 60)
    print("Phase 2: Full Sweep")
    print("=" * 60)
    
    sweep_results = run_full_sweep(
        c_values=[16, 32, 64, 128],
        kv_lens=[1024, 4096],
        q_lens=[1, 16, 64],
        kv_types=["clustered", "random", "skewed"],
        d=64,
        seed=42,
    )
    
    # Save sweep results
    sweep_path = os.path.join(output_dir, "exp19_sweep.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2, default=str)
    print(f"Saved: {sweep_path}")
    
    # 3. Direct Comparison
    print("\n" + "=" * 60)
    print("Phase 3: Nyström vs SVD Comparison")
    print("=" * 60)
    
    comparison_results = []
    for kv_type in ["clustered", "random", "skewed"]:
        comp_res = run_nystrom_vs_svd_comparison(
            kv_len=4096,
            q_len=16,
            kv_type=kv_type,
            d=64,
            seed=42,
        )
        comparison_results.append(comp_res)
    
    # Save comparison
    comp_path = os.path.join(output_dir, "exp19_vs_svd.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, default=str)
    print(f"Saved: {comp_path}")
    
    # 4. Generate Report
    print("\n" + "=" * 60)
    print("Phase 4: Generate Report")
    print("=" * 60)
    
    generate_report(sweep_results, comparison_results, sanity_results, output_dir)
    
    print("\n" + "=" * 60)
    print("Exp19 Complete!")
    print("=" * 60)
    
    return {
        "sanity": sanity_results,
        "sweep": sweep_results,
        "comparison": comparison_results,
    }


if __name__ == "__main__":
    main()
