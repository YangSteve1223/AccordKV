"""
Discussion: Cross-Block / Cross-Layer KV Sharing for ACCORD-KV
================================================================

核心问题：
1. 跨 block / 跨 layer 的结构是否存在？
2. 能否通过共享 KV 信息来绕开单 block 下界？
3. 压缩比能否达到 128×？

Baseline: Serial Cascade (clustered 3.45, random 0.48, skewed 0.22)
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

from simulation.exp1_fidelity_vs_bandwidth import ground_truth


# ============== 配置 ==============

@dataclass
class ExperimentConfig:
    """实验配置"""
    n_blocks: int = 3
    kv_len_per_block: int = 1024  # 3 blocks × 1024 = 4096 total
    total_kv_len: int = 4096
    q_len: int = 64
    d: int = 128
    seed: int = 42


# ============== 数据生成 ==============

def make_block_clustered_kv(
    n_blocks: int,
    kv_len_per_block: int,
    d: int,
    n_clusters_per_block: int = 8,
    inter_block_correlation: float = 0.0,  # 相邻 block 之间的相关性
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    生成多个 block 的 KV 数据。
    
    Args:
        inter_block_correlation: 相邻 block 的相似度（0-1）
            - 0: 完全独立（每个 block 有自己的 cluster 中心）
            - 1: 完全相同（所有 block 共享同一个 cluster 中心）
    """
    blocks = []
    gen = np.random.default_rng(seed)
    
    # 全局 cluster 中心（用于跨 block 共享）
    global_centroids = gen.standard_normal((n_clusters_per_block, d)) * 2.0
    
    for block_idx in range(n_blocks):
        block_seed = seed + block_idx * 1000
        
        # 根据 inter_block_correlation 决定 cluster 中心
        if inter_block_correlation > 0.5:
            # 高相关性：使用部分全局中心 + 部分局部扰动
            centroids = global_centroids.copy()
            if inter_block_correlation < 1.0:
                noise = gen.standard_normal((n_clusters_per_block, d)) * (1 - inter_block_correlation) * 2.0
                centroids = centroids + noise
        else:
            # 低相关性：每个 block 有自己的中心
            centroids = gen.standard_normal((n_clusters_per_block, d)) * 2.0
        
        block_gen = np.random.default_rng(block_seed)
        assignments = block_gen.integers(0, n_clusters_per_block, size=kv_len_per_block)
        K = centroids[assignments] + block_gen.standard_normal((kv_len_per_block, d)) * 0.5
        
        # V = K @ W + noise (W 是 block 内共享的线性变换)
        W = block_gen.standard_normal((d, d)) * 0.3
        V = K @ W + block_gen.standard_normal((kv_len_per_block, d)) * 0.1
        
        blocks.append((K.astype(np.float32), V.astype(np.float32)))
    
    return blocks


def make_block_random_kv(
    n_blocks: int,
    kv_len_per_block: int,
    d: int,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成完全随机的多个 block"""
    blocks = []
    for block_idx in range(n_blocks):
        block_seed = seed + block_idx * 1000
        gen = np.random.default_rng(block_seed)
        K = gen.standard_normal((kv_len_per_block, d)).astype(np.float32)
        V = gen.standard_normal((kv_len_per_block, d)).astype(np.float32)
        blocks.append((K, V))
    return blocks


def make_block_skewed_kv(
    n_blocks: int,
    kv_len_per_block: int,
    d: int,
    n_outliers: int = 16,
    inter_block_similarity: float = 0.0,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成 skew 结构的多个 block"""
    blocks = []
    gen = np.random.default_rng(seed)
    
    # 跨 block 共享的 outliers
    shared_outlier_K = gen.standard_normal((n_outliers, d)) * 3.0
    shared_outlier_V = gen.standard_normal((n_outliers, d)) * 3.0
    
    for block_idx in range(n_blocks):
        block_seed = seed + block_idx * 1000
        block_gen = np.random.default_rng(block_seed)
        
        # 每个 block 有自己的 outliers（部分共享）
        n_local_outliers = max(0, n_outliers - int(n_outliers * inter_block_similarity))
        n_shared_outliers = int(n_outliers * inter_block_similarity)
        
        outlier_K = np.concatenate([
            shared_outlier_K[:n_shared_outliers],
            block_gen.standard_normal((n_local_outliers, d)) * 3.0
        ])
        outlier_V = np.concatenate([
            shared_outlier_V[:n_shared_outliers],
            block_gen.standard_normal((n_local_outliers, d)) * 3.0
        ])
        
        normal_K = block_gen.standard_normal((kv_len_per_block - n_outliers, d)) * 0.3
        normal_V = block_gen.standard_normal((kv_len_per_block - n_outliers, d)) * 0.3
        
        K = np.concatenate([outlier_K, normal_K])
        V = np.concatenate([outlier_V, normal_V])
        
        perm = block_gen.permutation(kv_len_per_block)
        blocks.append((K[perm].astype(np.float32), V[perm].astype(np.float32)))
    
    return blocks


# ============== 跨 block 结构验证 ==============

def verify_cross_block_similarity(blocks: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """
    验证跨 block 结构的相似性。
    
    返回:
        - V 矩阵之间的 cosine 相似度
        - Attention pattern 之间的相关性
        - Cluster 结构的相似性
    """
    n_blocks = len(blocks)
    results = {
        "v_cosine_similarity": [],  # 相邻 block 的 V cosine 相似度
        "attention_pattern_correlation": [],  # attention pattern 相关性
        "v_matrix_similarity_matrix": np.zeros((n_blocks, n_blocks)),
    }
    
    # 1. V 矩阵的 cosine 相似度
    for i in range(n_blocks):
        for j in range(n_blocks):
            V_i = blocks[i][1]
            V_j = blocks[j][1]
            
            # Flatten 后计算 cosine 相似度
            v_i_flat = V_i.flatten()
            v_j_flat = V_j.flatten()
            
            dot = np.dot(v_i_flat, v_j_flat)
            norm_i = npla.norm(v_i_flat)
            norm_j = npla.norm(v_j_flat)
            
            cosine = dot / (norm_i * norm_j + 1e-10)
            results["v_matrix_similarity_matrix"][i, j] = cosine
    
    # 只记录相邻 block 的相似度
    for i in range(n_blocks - 1):
        results["v_cosine_similarity"].append(
            float(results["v_matrix_similarity_matrix"][i, i + 1])
        )
    
    return results


def compute_attention_pattern_similarity(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    Q: np.ndarray,
) -> dict:
    """
    计算相邻 block 的 attention pattern 相似性。
    """
    n_blocks = len(blocks)
    correlations = []
    
    # 为每个 block 计算 attention pattern
    patterns = []
    for K, V in blocks:
        gt = ground_truth(Q, K, V)
        patterns.append(gt.mean(axis=0))  # [d]
    
    # 计算相邻 block 的相关性
    for i in range(n_blocks - 1):
        p_i = patterns[i]
        p_j = patterns[i + 1]
        
        # Pearson 相关系数
        corr = np.corrcoef(p_i, p_j)[0, 1]
        correlations.append(float(corr) if not np.isnan(corr) else 0.0)
    
    return {
        "adjacent_attention_correlation": correlations,
        "mean_attention_correlation": float(np.mean(correlations)) if correlations else 0.0,
    }


# ============== 方向 A: Layer-Similar KV 合并 ==============

def cross_layer_kv_merge(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    Q: np.ndarray,
    target_ratio: float = 0.25,  # 保留 25% 的 KV
    seed: int = 0,
) -> dict:
    """
    方向 A: Layer-Similar KV 合并
    
    假设：相邻 layer/block 的 KV 有相似结构，可以共享信息。
    
    方法：
    1. 将所有 block 的 V 堆叠
    2. 使用 SVD 找到跨 block 的共同模式
    3. 只存储低秩近似 + 每个 block 的残差
    """
    gen = np.random.default_rng(seed)
    
    # 堆叠所有 block 的 V
    all_V = np.stack([V for _, V in blocks], axis=0)  # [n_blocks, kv_len_per_block, d]
    n_blocks, kv_len, d = all_V.shape
    
    # 对 V 矩阵进行跨 block SVD
    # reshape: [n_blocks, kv_len, d] -> [n_blocks * kv_len, d]
    V_stacked = all_V.reshape(-1, d)
    
    # SVD
    U, S, Vt = npla.svd(V_stacked, full_matrices=False)
    
    # 保留 top-k 奇异值
    r = max(1, int(target_ratio * len(S)))
    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]
    
    # 重建
    V_reconstructed = U_r @ np.diag(S_r) @ Vt_r
    V_reconstructed = V_reconstructed.reshape(n_blocks, kv_len, d)
    
    # 计算误差
    errors = []
    for i, (K, V) in enumerate(blocks):
        gt = ground_truth(Q, K, V)
        recon = ground_truth(Q, K, V_reconstructed[i])
        err = float(np.abs(recon - gt).mean())
        errors.append(err)
    
    # 计算压缩比
    original_size = n_blocks * kv_len * d * 4  # float32
    compressed_size = U_r.size * 4 + S_r.size * 4 + Vt_r.size * 4
    compression_ratio = original_size / compressed_size
    
    return {
        "method": "Cross-Layer KV Merge (SVD)",
        "compression_ratio": compression_ratio,
        "mean_attention_err": float(np.mean(errors)),
        "block_errors": errors,
        "rank": r,
        "singular_values_kept": int(r),
    }


# ============== 方向 B: Cross-Block Attention 共享 ==============

def cross_block_attention_share(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    Q: np.ndarray,
    share_ratio: float = 0.5,
    seed: int = 0,
) -> dict:
    """
    方向 B: Cross-Block Attention 共享
    
    假设：相邻 block 的 attention pattern 相似，可以共享 attention 结果。
    
    方法：
    1. 计算第一个 block 的 attention（作为 anchor）
    2. 对后续 block，只存储相对于 anchor 的残差
    """
    gen = np.random.default_rng(seed)
    n_blocks = len(blocks)
    
    # Block 0 作为 anchor
    anchor_K, anchor_V = blocks[0]
    anchor_gt = ground_truth(Q, anchor_K, anchor_V)
    
    results = []
    total_original = 0
    total_compressed = 0
    
    for i, (K, V) in enumerate(blocks):
        gt = ground_truth(Q, K, V)
        
        if i == 0:
            # Anchor block: 完整存储
            err = float(np.abs(gt - gt).mean())
            compressed_size = K.size * 4 + V.size * 4
        else:
            # 残差编码
            residual = gt - anchor_gt  # 跨 block 的残差
            
            # 对残差进行阈值截断
            residual_magnitude = np.abs(residual).mean()
            threshold = residual_magnitude * (1 - share_ratio)
            residual_sparse = np.where(np.abs(residual) > threshold, residual, 0)
            
            # 统计稀疏度
            sparsity = 1.0 - (np.abs(residual_sparse) > 1e-10).mean()
            
            # 近似重建
            approx_gt = anchor_gt + residual_sparse
            err = float(np.abs(approx_gt - gt).mean())
            
            # 压缩大小：anchor + 稀疏残差
            compressed_size = (
                anchor_K.size * 4 + anchor_V.size * 4 +  # anchor
                residual_sparse.size * 4 * (1 - sparsity)  # 稀疏残差
            )
        
        original_size = K.size * 4 + V.size * 4
        total_original += original_size
        total_compressed += compressed_size
        
        results.append({
            "block_idx": i,
            "attention_err": err,
            "block_original_size": int(original_size),
            "block_compressed_size": int(compressed_size),
        })
    
    compression_ratio = total_original / total_compressed if total_compressed > 0 else float('inf')
    mean_err = float(np.mean([r["attention_err"] for r in results]))
    
    return {
        "method": "Cross-Block Attention Share (Anchor + Residual)",
        "compression_ratio": compression_ratio,
        "mean_attention_err": mean_err,
        "block_results": results,
    }


# ============== 方向 C: Hierarchical KV Cache ==============

def hierarchical_kv_cache(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    Q: np.ndarray,
    coarse_ratio: float = 0.125,  # 粗粒度保留 12.5%
    seed: int = 0,
) -> dict:
    """
    方向 C: Hierarchical KV Cache (两层结构)
    
    假设：存在天然的粗粒度/细粒度层次结构。
    
    方法：
    1. 第一层 (coarse): 对每个 block 做 coreset，选择最具代表性的 token
    2. 第二层 (fine): 只存储每个 block 的残差（与 coarse 的差异）
    """
    from simulation.exp10_kmean_normalized_sketch import (
        build_coreset_sketch,
        eval_coreset_sketch,
    )
    
    n_blocks = len(blocks)
    results = []
    total_original = 0
    total_compressed = 0
    
    for i, (K, V) in enumerate(blocks):
        gt = ground_truth(Q, K, V)
        kv_len = K.shape[0]
        r = max(2, int(kv_len * coarse_ratio))
        
        # Stage 1: Coreset (粗粒度)
        coreset_sketch = build_coreset_sketch(K, V, r=r, seed=seed + i)
        centroids_K = coreset_sketch.centroids_K
        centroids_V = coreset_sketch.centroids_V
        
        # 评估 coarse
        coarse_stats = eval_coreset_sketch(coreset_sketch, Q)
        coarse_out = coarse_stats.finalize().squeeze(0)
        coarse_err = float(np.abs(coarse_out - gt).mean())
        
        # Stage 2: 细粒度残差（量化）
        # 计算每个 token 相对于其最近 centroids 的残差
        dists = np.zeros((kv_len, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids_K[j]) ** 2, axis=1)
        nearest_centroid = dists.argmin(axis=1)
        
        # 每个 token 的残差
        residuals = V - centroids_V[nearest_centroid]  # [kv_len, d]
        
        # 量化残差
        residual_scale = np.abs(residuals).max()
        if residual_scale > 1e-10:
            residuals_quant = np.round(residuals / residual_scale * 127).clip(-128, 127).astype(np.int8)
        else:
            residuals_quant = np.zeros_like(residuals, dtype=np.int8)
        
        # 重建
        V_reconstructed = centroids_V[nearest_centroid] + residuals_quant.astype(np.float32) * (residual_scale / 127)
        
        # 评估
        recon_stats = eval_coreset_sketch(coreset_sketch, Q)  # 用 coarse centroids 评估
        # 实际上我们需要重新构建 sketch with V_reconstructed
        from simulation.exp10_kmean_normalized_sketch import build_coreset_sketch as rebuild_coreset
        recon_sketch = rebuild_coreset(K, V_reconstructed, r=r, seed=seed + i)
        recon_out = eval_coreset_sketch(recon_sketch, Q).finalize().squeeze(0)
        recon_err = float(np.abs(recon_out - gt).mean())
        
        # 计算压缩比
        # 原始: kv_len * d * 2 * 4 bytes
        # 压缩: r * d * 2 * 4 (centroids) + kv_len * d * 1 (quantized residuals) + scale
        original_size = kv_len * d * 2 * 4
        compressed_size = r * d * 2 * 4 + kv_len * d * 1 + 4
        
        total_original += original_size
        total_compressed += compressed_size
        
        results.append({
            "block_idx": i,
            "coarse_err": coarse_err,
            "recon_err": recon_err,
            "r": r,
            "original_size": int(original_size),
            "compressed_size": int(compressed_size),
        })
    
    compression_ratio = total_original / total_compressed if total_compressed > 0 else float('inf')
    mean_coarse_err = float(np.mean([r["coarse_err"] for r in results]))
    mean_recon_err = float(np.mean([r["recon_err"] for r in results]))
    
    return {
        "method": "Hierarchical KV Cache (Coreset + Quantized Residual)",
        "compression_ratio": compression_ratio,
        "mean_coarse_err": mean_coarse_err,
        "mean_recon_err": mean_recon_err,
        "block_results": results,
    }


# ============== 方向 D: Delta Encoding (bonus) ==============

def delta_encoding(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    Q: np.ndarray,
    seed: int = 0,
) -> dict:
    """
    方向 D: Delta Encoding
    
    假设：相邻 block 的 V 差分更稀疏。
    
    方法：
    V_i+1 - V_i 可能比 V_i+1 更稀疏
    """
    n_blocks = len(blocks)
    results = []
    total_original = 0
    total_compressed = 0
    
    for i, (K, V) in enumerate(blocks):
        gt = ground_truth(Q, K, V)
        kv_len = K.shape[0]
        original_size = kv_len * V.shape[1] * 4
        
        if i == 0:
            # 第一个 block：完整存储
            err = 0.0
            compressed_size = original_size
        else:
            prev_V = blocks[i - 1][1]
            
            # Delta = V - prev_V
            delta = V - prev_V
            
            # 稀疏化
            delta_magnitude = np.abs(delta).mean()
            threshold = delta_magnitude * 0.5
            delta_sparse = np.where(np.abs(delta) > threshold, delta, 0)
            sparsity = 1.0 - (np.abs(delta_sparse) > 1e-10).mean()
            
            # 压缩：存储稀疏 delta
            compressed_size = delta_sparse.size * 4 * (1 - sparsity) + 4  # +4 for threshold
            
            # 重建
            V_reconstructed = prev_V + delta_sparse
            
            # 评估（用原始 K）
            from simulation.exp10_kmean_normalized_sketch import (
                build_coreset_sketch,
                eval_coreset_sketch,
            )
            sketch = build_coreset_sketch(K, V_reconstructed, r=max(2, kv_len // 8), seed=seed + i)
            recon_out = eval_coreset_sketch(sketch, Q).finalize().squeeze(0)
            err = float(np.abs(recon_out - gt).mean())
        
        total_original += original_size
        total_compressed += compressed_size
        
        results.append({
            "block_idx": i,
            "attention_err": err,
        })
    
    compression_ratio = total_original / total_compressed if total_compressed > 0 else float('inf')
    mean_err = float(np.mean([r["attention_err"] for r in results]))
    
    return {
        "method": "Delta Encoding (V_i+1 - V_i)",
        "compression_ratio": compression_ratio,
        "mean_attention_err": mean_err,
        "block_results": results,
    }


# ============== Baseline: Serial Cascade ==============

def serial_cascade_baseline(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    Q: np.ndarray,
    coreset_ratio: float = 0.25,
    seed: int = 0,
) -> dict:
    """
    Serial Cascade Baseline (per-block)
    
    每个 block 独立应用 Serial Cascade。
    """
    from simulation.exp15_serial_fusion import (
        build_coreset_sketch,
        eval_coreset_sketch,
        quantize_nbit,
        dequantize_nbit,
    )
    
    n_blocks = len(blocks)
    results = []
    total_original = 0
    total_compressed = 0
    
    for i, (K, V) in enumerate(blocks):
        gt = ground_truth(Q, K, V)
        kv_len = K.shape[0]
        r = max(4, int(kv_len * coreset_ratio))
        
        # Coreset
        centroids_K, centroids_V, weights = build_coreset_sketch(K, V, r=r, seed=seed + i)
        
        # 评估
        out = eval_coreset_sketch(centroids_K, centroids_V, weights, Q, K.shape[1])
        err = float(np.abs(out - gt).mean())
        
        original_size = kv_len * K.shape[1] * 2 * 4
        compressed_size = r * K.shape[1] * 2 * 4 + r * 4  # centroids + weights
        
        total_original += original_size
        total_compressed += compressed_size
        
        results.append({
            "block_idx": i,
            "attention_err": err,
            "r": r,
        })
    
    compression_ratio = total_original / total_compressed if total_compressed > 0 else float('inf')
    mean_err = float(np.mean([r["attention_err"] for r in results]))
    
    return {
        "method": "Serial Cascade Baseline",
        "compression_ratio": compression_ratio,
        "mean_attention_err": mean_err,
        "block_results": results,
    }


# ============== 主实验 ==============

def run_cross_block_experiment(
    config: ExperimentConfig = ExperimentConfig(),
) -> dict:
    """运行跨 block 实验"""
    print("=" * 60)
    print("Cross-Block / Cross-Layer KV Sharing Exploration")
    print("=" * 60)
    print(f"Config: n_blocks={config.n_blocks}, kv_len/block={config.kv_len_per_block}, "
          f"total={config.total_kv_len}, d={config.d}, q_len={config.q_len}")
    print()
    
    # 生成 Q
    gen = np.random.default_rng(config.seed + 999)
    Q = (gen.standard_normal((config.q_len, config.d)) * 0.5).astype(np.float32)
    
    results = {
        "config": asdict(config),
        "cross_block_structure_verification": {},
        "methods": {},
        "baseline": {},
    }
    
    # 测试不同的 KV 类型
    kv_types_to_test = ["clustered", "random", "skewed"]
    
    for kv_type in kv_types_to_test:
        print(f"\n### KV Type: {kv_type.upper()} ###")
        
        # 生成 blocks
        if kv_type == "clustered":
            blocks = make_block_clustered_kv(
                config.n_blocks, config.kv_len_per_block, config.d,
                inter_block_correlation=0.3,  # 假设有 30% 相关性
                seed=config.seed,
            )
        elif kv_type == "random":
            blocks = make_block_random_kv(
                config.n_blocks, config.kv_len_per_block, config.d,
                seed=config.seed,
            )
        else:  # skewed
            blocks = make_block_skewed_kv(
                config.n_blocks, config.kv_len_per_block, config.d,
                inter_block_similarity=0.3,
                seed=config.seed,
            )
        
        # 1. 验证跨 block 结构
        print(f"\n[1] Verifying Cross-Block Structure...")
        similarity = verify_cross_block_similarity(blocks)
        attention_corr = compute_attention_pattern_similarity(blocks, Q)
        
        print(f"  V cosine similarity (adjacent): {similarity['v_cosine_similarity']}")
        print(f"  Mean V cosine: {np.mean(similarity['v_cosine_similarity']):.4f}")
        print(f"  Attention correlation (adjacent): {attention_corr['adjacent_attention_correlation']}")
        print(f"  Mean attention correlation: {attention_corr['mean_attention_correlation']:.4f}")
        
        results["cross_block_structure_verification"][kv_type] = {
            "v_cosine_adjacent": similarity["v_cosine_similarity"],
            "mean_v_cosine": float(np.mean(similarity["v_cosine_similarity"])),
            "attention_correlation_adjacent": attention_corr["adjacent_attention_correlation"],
            "mean_attention_correlation": attention_corr["mean_attention_correlation"],
        }
        
        # 2. 测试各个方向
        methods_results = {}
        
        # Direction A: Cross-Layer KV Merge
        print(f"\n[2] Direction A: Cross-Layer KV Merge...")
        try:
            method_a = cross_layer_kv_merge(blocks, Q, target_ratio=0.25, seed=config.seed)
            methods_results["A_cross_layer_kv_merge"] = method_a
            print(f"  Compression: {method_a['compression_ratio']:.1f}x, Err: {method_a['mean_attention_err']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            methods_results["A_cross_layer_kv_merge"] = {"error": str(e)}
        
        # Direction B: Cross-Block Attention Share
        print(f"\n[3] Direction B: Cross-Block Attention Share...")
        try:
            method_b = cross_block_attention_share(blocks, Q, share_ratio=0.5, seed=config.seed)
            methods_results["B_cross_block_attention"] = method_b
            print(f"  Compression: {method_b['compression_ratio']:.1f}x, Err: {method_b['mean_attention_err']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            methods_results["B_cross_block_attention"] = {"error": str(e)}
        
        # Direction C: Hierarchical KV Cache
        print(f"\n[4] Direction C: Hierarchical KV Cache...")
        try:
            method_c = hierarchical_kv_cache(blocks, Q, coarse_ratio=0.125, seed=config.seed)
            methods_results["C_hierarchical_kv"] = method_c
            print(f"  Compression: {method_c['compression_ratio']:.1f}x, "
                  f"Coarse Err: {method_c['mean_coarse_err']:.4f}, "
                  f"Recon Err: {method_c['mean_recon_err']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            methods_results["C_hierarchical_kv"] = {"error": str(e)}
        
        # Direction D: Delta Encoding
        print(f"\n[5] Direction D: Delta Encoding...")
        try:
            method_d = delta_encoding(blocks, Q, seed=config.seed)
            methods_results["D_delta_encoding"] = method_d
            print(f"  Compression: {method_d['compression_ratio']:.1f}x, Err: {method_d['mean_attention_err']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            methods_results["D_delta_encoding"] = {"error": str(e)}
        
        # Baseline: Serial Cascade
        print(f"\n[6] Baseline: Serial Cascade...")
        try:
            baseline = serial_cascade_baseline(blocks, Q, coreset_ratio=0.25, seed=config.seed)
            methods_results["baseline_serial_cascade"] = baseline
            print(f"  Compression: {baseline['compression_ratio']:.1f}x, Err: {baseline['mean_attention_err']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            methods_results["baseline_serial_cascade"] = {"error": str(e)}
        
        results["methods"][kv_type] = methods_results
    
    return results


def analyze_cross_block_structure(verification_data: dict) -> dict:
    """分析跨 block 结构的真实性"""
    analysis = {
        "v_similarity_verified": False,
        "attention_similarity_verified": False,
        "recommendation": None,
        "details": {},
    }
    
    all_v_cosine = []
    all_attention_corr = []
    
    for kv_type, data in verification_data.items():
        v_cosine = data.get("mean_v_cosine", 0)
        attention_corr = data.get("mean_attention_correlation", 0)
        all_v_cosine.append(v_cosine)
        all_attention_corr.append(attention_corr)
        
        analysis["details"][kv_type] = {
            "v_cosine": v_cosine,
            "attention_correlation": attention_corr,
            "structure_exists": v_cosine > 0.5 or attention_corr > 0.3,
        }
    
    # 平均相似度
    avg_v_cosine = np.mean(all_v_cosine)
    avg_attention_corr = np.mean(all_attention_corr)
    
    analysis["avg_v_cosine"] = float(avg_v_cosine)
    analysis["avg_attention_correlation"] = float(avg_attention_corr)
    
    # 判断结构是否存在
    # V cosine > 0.5 表示有中等以上相似度
    # Attention correlation > 0.3 表示有弱相关
    if avg_v_cosine > 0.3 or avg_attention_corr > 0.2:
        analysis["v_similarity_verified"] = True
        analysis["attention_similarity_verified"] = True
        analysis["recommendation"] = "Cross-block structure EXISTS - sharing is viable"
    else:
        analysis["recommendation"] = "Cross-block structure WEAK - sharing may not help"
    
    return analysis


def compare_with_baseline(
    methods_results: dict,
    baseline_err: float = 3.45,  # Serial Cascade clustered
) -> dict:
    """与 Serial Cascade baseline 对比"""
    comparison = {}
    
    for method_name, result in methods_results.items():
        if "error" in result:
            comparison[method_name] = {"status": "failed", "error": result["error"]}
            continue
        
        method_err = result.get("mean_attention_err", result.get("mean_recon_err", float('inf')))
        method_comp = result.get("compression_ratio", 0)
        
        # 计算相对于 baseline 的改进
        err_improvement = baseline_err - method_err
        err_ratio = method_err / baseline_err if baseline_err > 0 else float('inf')
        
        comparison[method_name] = {
            "method": result.get("method", method_name),
            "compression_ratio": method_comp,
            "attention_err": method_err,
            "baseline_err": baseline_err,
            "err_improvement": err_improvement,
            "err_ratio": err_ratio,
            "beats_baseline": method_err < baseline_err,
        }
    
    return comparison


def main():
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行实验
    config = ExperimentConfig(
        n_blocks=3,
        kv_len_per_block=1024,
        total_kv_len=4096,
        q_len=64,
        d=128,
        seed=42,
    )
    
    results = run_cross_block_experiment(config)
    
    # 分析跨 block 结构
    structure_analysis = analyze_cross_block_structure(
        results["cross_block_structure_verification"]
    )
    results["structure_analysis"] = structure_analysis
    
    print("\n" + "=" * 60)
    print("Cross-Block Structure Analysis")
    print("=" * 60)
    print(f"V cosine similarity (avg): {structure_analysis['avg_v_cosine']:.4f}")
    print(f"Attention correlation (avg): {structure_analysis['avg_attention_correlation']:.4f}")
    print(f"Structure exists: {structure_analysis['v_similarity_verified']}")
    print(f"Recommendation: {structure_analysis['recommendation']}")
    
    # 与 baseline 对比
    print("\n" + "=" * 60)
    print("Comparison with Serial Cascade Baseline (err=3.45)")
    print("=" * 60)
    
    for kv_type, methods in results["methods"].items():
        print(f"\n### {kv_type.upper()} ###")
        baseline_err = 3.45 if kv_type == "clustered" else (0.48 if kv_type == "random" else 0.22)
        comparison = compare_with_baseline(methods, baseline_err)
        
        for method_name, comp in comparison.items():
            if comp.get("status") == "failed":
                print(f"  {method_name}: FAILED - {comp['error']}")
            else:
                beats = "✅" if comp["beats_baseline"] else "❌"
                print(f"  {method_name}: comp={comp['compression_ratio']:.1f}x, "
                      f"err={comp['attention_err']:.4f}, "
                      f"ratio={comp['err_ratio']:.2f} {beats}")
    
    # 保存结果
    results_path = os.path.join(output_dir, "discussion_cross_block_data.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved data to {results_path}")
    
    return results


if __name__ == "__main__":
    main()
