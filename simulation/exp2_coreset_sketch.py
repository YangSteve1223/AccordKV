"""
Exp3: Coreset Sketch 实现 + E1 Bytes-Error Pareto 仿真 + 创新探索

目标：
1. 实现 Key-centroid coreset sketch（基于 Lloyd k-means）
2. E1 Pareto 仿真：Full KV vs Coreset vs Drop baseline
3. 主动探索创新点（至少 2 个方向）：
   - A: Attention-pattern-aware sketch（按 attention pattern 分配 centroid）
   - B: Hierarchical sketch（2层 k-means）
   - C: Coreset + INT4 quantization

核心数学：
- sketch 存储: centroids C_j ∈ R^d, weights w_j ∈ R
- eval: score_j = q · C_j / sqrt(d) + log(w_j)
- m = max(score), p = exp(score - m), l = Σp, y = Σ p * V_j

仿真矩阵：
- block_size ∈ {32, 64, 128}
- kv_len ∈ {1024, 4096, 16384}
- sketch r ∈ {4, 8, 16, 32}
- q_len ∈ {16, 64}
- d = 128
- 3 paths × 3 block_size × 3 kv_len × 4 r × 2 q_len = 72 组
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    ground_truth,
)

# ============== 类型定义 ==============


@dataclass
class CoresetSketch:
    """Coreset sketch 存储结构。
    
    centroids: [r, d] 的 key centroids
    values: [r, d] 的 value aggregates（每个 cluster 的 mean v）
    weights: [r] 的 cluster 权重（token 数 / block_size）
    assignments: [kv_len] 的 token → cluster 映射
    """
    centroids: np.ndarray  # [r, d]
    values: np.ndarray     # [r, d]
    weights: np.ndarray    # [r]
    assignments: np.ndarray  # [kv_len]
    
    def bytes_size(self) -> int:
        """估算 sketch 传输字节数（fp32）"""
        return (self.centroids.size + self.values.size + self.weights.size) * 4


@dataclass
class HierarchicalSketch:
    """2层 hierarchical sketch。
    
    顶层: super_centroids [s, d], super_weights [s]
    底层: sub_centroids [s, sub_r, d], sub_values [s, sub_r, d], sub_weights [s, sub_r]
    分配: super_assignments [kv_len], sub_within_super [kv_len]
    """
    super_centroids: np.ndarray      # [s, d]
    super_weights: np.ndarray        # [s]
    sub_centroids: np.ndarray        # [s, sub_r, d]
    sub_values: np.ndarray          # [s, sub_r, d]
    sub_weights: np.ndarray          # [s, sub_r]
    super_assignments: np.ndarray    # [kv_len]
    sub_within_super: np.ndarray     # [kv_len]
    
    def bytes_size(self) -> int:
        s = self.super_centroids.shape[0]
        sub_r = self.sub_centroids.shape[1]
        # 顶层: s*d + s
        # 底层: s*(sub_r*d + sub_r*d + sub_r) = s*sub_r*(2d+1)
        return (s * self.super_centroids.shape[1] + s + 
                s * sub_r * (self.sub_centroids.shape[2] * 2 + 1)) * 4


@dataclass 
class AttentionAwareSketch:
    """Attention-pattern-aware sketch。
    
    区别于均匀 k-means：
    - 用 calibration queries 计算 attention pattern
    - 高 attention 的 cluster 分配更多 centroid
    - 实现了 importance-weighted k-means
    """
    centroids: np.ndarray       # [r, d]
    values: np.ndarray          # [r, d] 
    weights: np.ndarray         # [r]
    assignments: np.ndarray     # [kv_len]
    attention_scores: np.ndarray  # [r], 每个 centroid 的平均 attention score
    calibration_used: bool = True


# ============== 工具函数 ==============


def kmeans_plusplus_init(K: np.ndarray, r: int, seed: int = 0) -> np.ndarray:
    """K-Means++ 初始化，比随机初始化好。
    
    选择概率与距离平方成正比，避免随机初始化陷入局部最优。
    """
    gen = np.random.default_rng(seed)
    n, d = K.shape
    
    # 选择第一个 centroid 随机
    idx = gen.integers(0, n)
    centroids = [K[idx].copy()]
    
    # 选择剩下的 r-1 个
    for _ in range(r - 1):
        # 计算每个点到最近 centroid 的距离
        dists = np.zeros(n)
        for c in centroids:
            dists += np.sum((K - c) ** 2, axis=1)
        
        # 概率与距离成正比
        probs = dists / dists.sum()
        idx = gen.choice(n, p=probs)
        centroids.append(K[idx].copy())
    
    return np.array(centroids)  # [r, d]


def lloyd_iteration(K: np.ndarray, V: np.ndarray, centroids: np.ndarray, 
                    block_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lloyd 算法一次迭代。
    
    Returns:
        new_centroids: [r, d]
        new_values: [r, d]（每个 cluster 的 mean v）
        weights: [r]（每个 cluster 的 token 比例）
    """
    r = centroids.shape[0]
    n, d = K.shape
    
    # E-step: 分配每个 token 到最近的 centroid
    dists = np.zeros((n, r))
    for j in range(r):
        dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
    assignments = dists.argmin(axis=1)  # [n]
    
    # M-step: 计算新的 centroids 和 weights
    new_centroids = np.zeros_like(centroids)
    new_values = np.zeros((r, d))
    weights = np.zeros(r)
    
    for j in range(r):
        mask = assignments == j
        count = mask.sum()
        if count > 0:
            new_centroids[j] = K[mask].mean(axis=0)
            new_values[j] = V[mask].mean(axis=0)
            weights[j] = count / n  # 归一化 weight
        else:
            # 空 cluster，保留原 centroid
            new_centroids[j] = centroids[j]
            new_values[j] = V[gen_in_range(0, r)]
            weights[j] = 1e-10
    
    return new_centroids, new_values, weights


# 辅助：获取随机数生成器（避免 lloyd_iteration 里用 gen）
_gen_counter = 0


def gen_in_range(seed: int, bound: int) -> int:
    """简单伪随机"""
    global _gen_counter
    _gen_counter += 1
    return (seed * 1103515245 + _gen_counter) % bound


# ============== 核心 Sketch 实现 ==============


def build_coreset_sketch(
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    block_size: int = 64,
    seed: int = 0,
    num_iters: int = 15,
) -> CoresetSketch:
    """Lloyd iterations k-means 构建 coreset sketch。
    
    Parameters
    ----------
    K: [kv_len, d] - key vectors
    V: [kv_len, d] - value vectors
    r: int - centroid 数（compression ratio = kv_len / r）
    block_size: int - 用于 weight 计算
    seed: int - 随机种子
    num_iters: int - Lloyd 迭代次数（10-20 次）
    
    Returns
    -------
    CoresetSketch 对象
    """
    n, d = K.shape
    gen = np.random.default_rng(seed)
    
    # K-Means++ 初始化
    centroids = kmeans_plusplus_init(K, r, seed)
    
    # Lloyd 迭代
    for _ in range(num_iters):
        # E-step: 分配
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
        assignments = dists.argmin(axis=1)
        
        # M-step: 更新
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
            else:
                new_centroids[j] = centroids[j]
                new_values[j] = V[gen.integers(0, n)]
                weights[j] = 1e-10
        
        centroids = new_centroids
        
        # 检查收敛
        centroid_shift = np.sum((centroids - new_centroids) ** 2)
        if centroid_shift < 1e-8:
            break
    
    # 最终分配
    dists = np.zeros((n, r))
    for j in range(r):
        dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
    final_assignments = dists.argmin(axis=1)
    
    # 重新计算最终 weights
    final_weights = np.zeros(r)
    for j in range(r):
        mask = final_assignments == j
        final_weights[j] = mask.sum() / n
    
    return CoresetSketch(
        centroids=centroids,
        values=new_values,
        weights=final_weights,
        assignments=final_assignments,
    )


def eval_coreset_sketch(
    sketch: CoresetSketch,
    Q: np.ndarray,
    d: int,
) -> NumpyAttnStats:
    """用 coreset sketch 评估 attention。
    
    score_j = q · c_j / sqrt(d) + log(w_j + 1e-12)
    m = max(score), p = exp(score - m), l = Σp, y = Σ p * v_j
    
    Parameters
    ----------
    sketch: CoresetSketch
    Q: [q_len, d]
    d: dimension
    
    Returns
    -------
    NumpyAttnStats (H=1)
    """
    q_len = Q.shape[0]
    r = sketch.centroids.shape[0]
    
    # score_j = q · c_j / sqrt(d) + log(w_j)
    scores = Q @ sketch.centroids.T / math.sqrt(d)  # [q_len, r]
    scores = scores + np.log(sketch.weights + 1e-12)  # [q_len, r]
    
    # softmax
    m = scores.max(axis=-1, keepdims=True)  # [q_len, 1]
    p = np.exp(scores - m)  # [q_len, r]
    l = p.sum(axis=-1, keepdims=True)  # [q_len, 1]
    
    # y = Σ p_j * v_j
    y = p @ sketch.values  # [q_len, d]
    
    return NumpyAttnStats(
        m=m[None, :, :],  # [1, q_len, 1]
        l=l[None, :, :],
        y=y[None, :, :],
    )


# ============== 探索 A: Attention-pattern-aware Sketch ==============


def build_attention_aware_sketch(
    K: np.ndarray,
    V: np.ndarray,
    Q_cal: np.ndarray,
    r: int,
    block_size: int = 64,
    seed: int = 0,
    num_iters: int = 15,
) -> AttentionAwareSketch:
    """Attention-pattern-aware sketch（改进版）。
    
    不同于均匀 k-means：
    1. 先用 calibration queries 计算 attention pattern
    2. 按 attention score 排序，高 attention 的区域分配更多 centroid
    3. 这本质是 importance-weighted k-means（但用更直接的 soft assignment）
    
    改进：不再分层，而是直接用 attention score 作为软权重参与 k-means 目标函数
    """
    n, d = K.shape
    gen = np.random.default_rng(seed)
    
    # Step 1: 计算每个 token 的 attention importance
    # attention_scores = Σ_j exp(Q_cal[j] · K[i] / sqrt(d))
    attn_raw = Q_cal @ K.T / math.sqrt(d)  # [cal_len, n]
    attn_max = attn_raw.max(axis=0, keepdims=True)  # [1, n]
    attn_soft = np.exp(attn_raw - attn_max)  # [cal_len, n]
    importance = attn_soft.sum(axis=0)  # [n]
    # 归一化
    importance = importance / (importance.sum() + 1e-10)
    
    # Step 2: 基于 importance 初始化 centroids
    # 先选最重要的 token 作为初始 centroids
    top_indices = np.argsort(-importance)[:r]
    centroids = K[top_indices].copy()
    
    # Step 3: 带 importance 权重的 k-means
    for _ in range(num_iters):
        # E-step: 计算带权重的距离
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
        
        # 软分配：考虑 importance
        # 距离近 AND importance 高的 token 更可能被分配
        # score = -dists + log(importance + 1e-10)
        scores = -dists + np.log(importance[:, None] + 1e-10) * 0.5
        assignments = scores.argmax(axis=1)  # [n]
        
        # M-step: 带 importance 加权的 centroid 更新
        new_centroids = np.zeros_like(centroids)
        new_values = np.zeros((r, d))
        weights = np.zeros(r)
        
        for j in range(r):
            mask = assignments == j
            if mask.sum() > 0:
                # 加权平均：importance 高的 token 对 centroid 影响更大
                w = importance[mask]
                w = w / (w.sum() + 1e-10)
                new_centroids[j] = np.average(K[mask], axis=0, weights=w)
                new_values[j] = np.average(V[mask], axis=0, weights=w)
                weights[j] = importance[mask].sum()
            else:
                new_centroids[j] = centroids[j]
                new_values[j] = V[gen.integers(0, n)]
                weights[j] = 1e-10
        
        centroids = new_centroids
        values = new_values
    
    # 最终分配
    dists = np.zeros((n, r))
    for j in range(r):
        dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
    final_assign = dists.argmin(axis=1)
    
    # 最终 weights
    final_weights = np.zeros(r)
    for j in range(r):
        mask = final_assign == j
        final_weights[j] = importance[mask].sum()
    final_weights = final_weights / (final_weights.sum() + 1e-10)
    
    return AttentionAwareSketch(
        centroids=centroids,
        values=values,
        weights=final_weights,
        assignments=final_assign,
        attention_scores=importance[final_assign],
        calibration_used=True,
    )


def eval_attention_aware_sketch(
    sketch: AttentionAwareSketch,
    Q: np.ndarray,
    d: int,
) -> NumpyAttnStats:
    """评估 attention-aware sketch"""
    return eval_coreset_sketch(
        CoresetSketch(
            centroids=sketch.centroids,
            values=sketch.values,
            weights=sketch.weights,
            assignments=sketch.assignments,
        ),
        Q, d
    )


# ============== 探索 B: Hierarchical Sketch ==============


def build_hierarchical_sketch(
    K: np.ndarray,
    V: np.ndarray,
    s: int = 4,
    sub_r: int = 4,
    block_size: int = 64,
    seed: int = 0,
    num_iters: int = 15,
) -> HierarchicalSketch:
    """2层 hierarchical sketch。
    
    顶层: s 个 super-cluster
    每 super-cluster 内: sub_r 个 sub-centroid
    总共: s * sub_r 个 centroid
    
    优点: 可扩展到 kv_len > 100k，每个 query 只访问相关 super-cluster
    """
    n, d = K.shape
    gen = np.random.default_rng(seed)
    
    # Step 1: 顶层 k-means
    super_centroids = kmeans_plusplus_init(K, s, seed)
    
    for _ in range(num_iters):
        dists = np.zeros((n, s))
        for j in range(s):
            dists[:, j] = np.sum((K - super_centroids[j]) ** 2, axis=1)
        super_assign = dists.argmin(axis=1)
        
        new_super = np.zeros_like(super_centroids)
        for j in range(s):
            mask = super_assign == j
            if mask.sum() > 0:
                new_super[j] = K[mask].mean(axis=0)
        super_centroids = new_super
    
    # 最终 super assignment
    super_dists = np.zeros((n, s))
    for j in range(s):
        super_dists[:, j] = np.sum((K - super_centroids[j]) ** 2, axis=1)
    super_assignments = super_dists.argmin(axis=1)
    
    # Step 2: 每 super-cluster 内做 sub k-means
    sub_centroids = np.zeros((s, sub_r, d))
    sub_values = np.zeros((s, sub_r, d))
    sub_weights = np.zeros((s, sub_r))
    
    for j in range(s):
        mask = super_assignments == j
        sub_K = K[mask]
        sub_V = V[mask]
        sub_n = len(sub_K)
        
        if sub_n == 0:
            # 空 cluster，随机初始化
            sub_centroids[j] = kmeans_plusplus_init(K, sub_r, seed + j + 100)
            sub_values[j] = K[gen.integers(0, n)]
            sub_weights[j] = 1e-10
            continue
        
        # sub k-means
        if sub_n >= sub_r:
            sub_init = kmeans_plusplus_init(sub_K, sub_r, seed + j + 100)
        else:
            # token 比 centroid 少，复制
            sub_init = np.broadcast_to(
                sub_K[gen.choice(sub_n, sub_r, replace=True)],
                (sub_r, d)
            ).copy()
        
        for _ in range(num_iters):
            dists = np.zeros((sub_n, sub_r))
            for k in range(sub_r):
                dists[:, k] = np.sum((sub_K - sub_init[k]) ** 2, axis=1)
            sub_assign = dists.argmin(axis=1)
            
            new_sub = np.zeros_like(sub_init)
            new_val = np.zeros((sub_r, d))
            new_w = np.zeros(sub_r)
            
            for k in range(sub_r):
                sub_mask = sub_assign == k
                cnt = sub_mask.sum()
                if cnt > 0:
                    new_sub[k] = sub_K[sub_mask].mean(axis=0)
                    new_val[k] = sub_V[sub_mask].mean(axis=0)
                    new_w[k] = cnt / n
                else:
                    new_sub[k] = sub_init[k]
            
            sub_init = new_sub
        
        sub_centroids[j] = sub_init
        sub_values[j] = new_val
        sub_weights[j] = new_w
    
    # 重新计算 sub assignment（用于记录）
    sub_within_super = np.zeros(n, dtype=np.int32)
    for j in range(s):
        mask = super_assignments == j
        sub_K = K[mask]
        if len(sub_K) > 0:
            dists = np.zeros((len(sub_K), sub_r))
            for k in range(sub_r):
                dists[:, k] = np.sum((sub_K - sub_centroids[j, k]) ** 2, axis=1)
            sub_within_super[mask] = dists.argmin(axis=1)
    
    # super_weights
    super_weights = np.zeros(s)
    for j in range(s):
        super_weights[j] = (super_assignments == j).sum() / n
    
    return HierarchicalSketch(
        super_centroids=super_centroids,
        super_weights=super_weights,
        sub_centroids=sub_centroids,
        sub_values=sub_values,
        sub_weights=sub_weights,
        super_assignments=super_assignments,
        sub_within_super=sub_within_super,
    )


def eval_hierarchical_sketch(
    sketch: HierarchicalSketch,
    Q: np.ndarray,
    d: int,
    top_k_super: int = 2,
) -> NumpyAttnStats:
    """评估 hierarchical sketch。
    
    每个 query 只访问 attention score 最高的 top_k_super 个 super-cluster。
    """
    q_len = Q.shape[0]
    s = sketch.super_centroids.shape[0]
    sub_r = sketch.sub_centroids.shape[1]
    
    # Step 1: 计算 query 对每个 super-cluster 的 attention
    super_scores = Q @ sketch.super_centroids.T / math.sqrt(d)  # [q_len, s]
    super_scores = super_scores + np.log(sketch.super_weights + 1e-12)
    
    # Step 2: 选择 top-k super-cluster
    top_k_idx = np.argsort(-super_scores, axis=-1)[:, :top_k_super]  # [q_len, top_k]
    
    # Step 3: 聚合 sub-centroids
    final_scores = np.zeros((q_len, s * sub_r))
    final_values = np.zeros((s * sub_r, d))
    
    for q_idx in range(q_len):
        for sk in range(top_k_super):
            super_idx = top_k_idx[q_idx, sk]
            offset = super_idx * sub_r
            for k in range(sub_r):
                final_values[offset + k] = sketch.sub_values[super_idx, k]
                # super score 加上 sub weight
                final_scores[q_idx, offset + k] = (
                    super_scores[q_idx, super_idx] + 
                    math.log(sketch.sub_weights[super_idx, k] + 1e-12)
                )
    
    # 补全其他 super-cluster（如果 token 不够）
    all_sub_centroids = sketch.sub_centroids.reshape(-1, d)
    all_sub_weights = sketch.sub_weights.flatten()
    
    # softmax
    m = final_scores.max(axis=-1, keepdims=True)
    p = np.exp(final_scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ final_values
    
    return NumpyAttnStats(
        m=m[None, :, :],
        l=l[None, :, :],
        y=y[None, :, :],
    )


# ============== 探索 C: Coreset + INT4 Quantization ==============


@dataclass
class QuantizedSketch:
    """INT4 量化后的 sketch"""
    centroids_int4: np.ndarray      # [r, d] int4 量化
    values_int4: np.ndarray          # [r, d] int4 量化
    weights: np.ndarray              # [r] fp32（权重用 fp32 保持精度）
    scales: np.ndarray               # [r, 2] 每个 centroid 的 scale (for K and V)
    
    def bytes_size(self) -> int:
        """INT4: 2 values per byte, weights fp32, scales fp32"""
        return (self.centroids_int4.size // 2 + 
                self.values_int4.size // 2 + 
                self.weights.size * 4 +
                self.scales.size * 4)


def quantize_sketch(
    sketch: CoresetSketch,
    n_bits: int = 4,
) -> QuantizedSketch:
    """将 sketch 量化到 INT4"""
    r, d = sketch.centroids.shape
    
    # 计算每个 centroid 的 scale
    c_scales = np.zeros((r, 2))
    for j in range(r):
        c_max = max(np.abs(sketch.centroids[j]).max(), np.abs(sketch.values[j]).max())
        c_scales[j, 0] = c_max / (2 ** (n_bits - 1) - 1)  # for K
        c_scales[j, 1] = c_max / (2 ** (n_bits - 1) - 1)  # for V
    
    # 量化
    centroids_int4 = np.zeros((r, d), dtype=np.int8)
    values_int4 = np.zeros((r, d), dtype=np.int8)
    
    for j in range(r):
        centroids_int4[j] = np.clip(
            np.round(sketch.centroids[j] / c_scales[j, 0]),
            -(2 ** (n_bits - 1)), 
            2 ** (n_bits - 1) - 1
        ).astype(np.int8)
        values_int4[j] = np.clip(
            np.round(sketch.values[j] / c_scales[j, 1]),
            -(2 ** (n_bits - 1)),
            2 ** (n_bits - 1) - 1
        ).astype(np.int8)
    
    return QuantizedSketch(
        centroids_int4=centroids_int4,
        values_int4=values_int4,
        weights=sketch.weights.copy(),
        scales=c_scales,
    )


def dequantize_sketch(q_sketch: QuantizedSketch) -> CoresetSketch:
    """反量化回 float"""
    r, d = q_sketch.centroids_int4.shape
    
    centroids = np.zeros((r, d), dtype=np.float32)
    values = np.zeros((r, d), dtype=np.float32)
    
    for j in range(r):
        centroids[j] = q_sketch.centroids_int4[j].astype(np.float32) * q_sketch.scales[j, 0]
        values[j] = q_sketch.values_int4[j].astype(np.float32) * q_sketch.scales[j, 1]
    
    return CoresetSketch(
        centroids=centroids,
        values=values,
        weights=q_sketch.weights,
        assignments=np.zeros(r, dtype=np.int32),  # 不需要
    )


# ============== 数据生成 ==============


def make_clustered_kv(
    num_blocks: int,
    block_size: int,
    d: int,
    num_clusters: int,
    cluster_std: float = 0.5,
    seed: int = 0,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """生成有聚类结构的 K/V 数据。
    
    K 有明显的聚类结构，这样 coreset 可以展示优势。
    V 跟 K 相关联（通过 cluster membership）。
    
    Returns
    -------
    kv_cache: {block_id: (K, V)}
    K_all: [kv_len, d]
    V_all: [kv_len, d]
    """
    kv_len = num_blocks * block_size
    gen = np.random.default_rng(seed)
    
    # 生成 cluster centers
    cluster_centers = gen.standard_normal((num_clusters, d)) * 2.0
    
    # 每个 token 分配到 cluster
    cluster_assign = np.zeros(kv_len, dtype=np.int32)
    tokens_per_cluster = kv_len // num_clusters
    for i in range(num_clusters):
        start = i * tokens_per_cluster
        end = start + tokens_per_cluster if i < num_clusters - 1 else kv_len
        cluster_assign[start:end] = i
    
    # 打乱分配
    cluster_assign = gen.choice(num_clusters, kv_len).astype(np.int32)
    
    # 生成 K 和 V
    K_all = np.zeros((kv_len, d), dtype=np.float32)
    V_all = np.zeros((kv_len, d), dtype=np.float32)
    
    for i in range(kv_len):
        c = cluster_assign[i]
        K_all[i] = cluster_centers[c] + gen.standard_normal(d) * cluster_std
        # V 跟 K 高度相关（但有噪声）
        V_all[i] = K_all[i] * 0.8 + gen.standard_normal(d) * 0.2
    
    # 打包成 block
    kv_cache = {}
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        kv_cache[b] = (K_all[start:end].copy(), V_all[start:end].copy())
    
    return kv_cache, K_all, V_all


def make_random_kv(
    num_blocks: int,
    block_size: int,
    d: int,
    seed: int = 0,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """生成随机 K/V（无聚类结构）。
    
    作为对照，验证 coreset 在无结构情况下不会比 uniform sampling 差太多。
    """
    kv_len = num_blocks * block_size
    gen = np.random.default_rng(seed)
    
    K_all = gen.standard_normal((kv_len, d)).astype(np.float32) * 0.5
    V_all = gen.standard_normal((kv_len, d)).astype(np.float32) * 0.5
    
    kv_cache = {}
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        kv_cache[b] = (K_all[start:end].copy(), V_all[start:end].copy())
    
    return kv_cache, K_all, V_all


# ============== Drop Baseline ==============


def build_drop_sketch(
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform random drop baseline。
    
    随机保留 r 个 token，模拟没有结构利用的情况。
    """
    n = len(K)
    gen = np.random.default_rng(seed)
    
    # 随机选择 r 个保留
    keep_idx = gen.choice(n, r, replace=False)
    keep_idx = np.sort(keep_idx)
    
    K_drop = K[keep_idx]
    V_drop = V[keep_idx]
    weights = np.ones(r) / r  # 均匀权重
    
    return K_drop, V_drop, weights


def eval_drop_sketch(
    K_drop: np.ndarray,
    V_drop: np.ndarray,
    weights: np.ndarray,
    Q: np.ndarray,
    d: int,
) -> NumpyAttnStats:
    """评估 drop baseline"""
    r = len(K_drop)
    q_len = Q.shape[0]
    
    # score = Q @ K_drop.T / sqrt(d) + log(weights)
    scores = Q @ K_drop.T / math.sqrt(d)
    scores = scores + np.log(weights + 1e-12)
    
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ V_drop
    
    return NumpyAttnStats(
        m=m[None, :, :],
        l=l[None, :, :],
        y=y[None, :, :],
    )


# ============== 主仿真逻辑 ==============


def run_pareto_single(
    kv_len: int,
    block_size: int,
    sketch_r: int,
    q_len: int,
    d: int = 128,
    seed: int = 0,
    clustered: bool = True,
    verbose: bool = True,
) -> dict:
    """单组配置仿真。
    
    Returns
    -------
    dict 包含三种路径的 bytes 和 error
    """
    num_blocks = kv_len // block_size
    
    # 生成数据
    if clustered:
        kv_cache, K_all, V_all = make_clustered_kv(
            num_blocks, block_size, d, 
            num_clusters=max(4, kv_len // 256),
            seed=seed
        )
    else:
        kv_cache, K_all, V_all = make_random_kv(
            num_blocks, block_size, d, seed=seed
        )
    
    # 生成 query
    Q = (np.random.default_rng(seed + 1000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    # Ground truth
    gt = ground_truth(Q, K_all, V_all)
    
    # ====== Path 1: Full KV ======
    bytes_full = 2 * kv_len * d * 4  # K + V, fp32
    
    # ====== Path 2: Coreset Sketch ======
    sketch = build_coreset_sketch(K_all, V_all, sketch_r, block_size, seed)
    stats_sketch = eval_coreset_sketch(sketch, Q, d)
    out_sketch = stats_sketch.finalize().squeeze(0)
    err_sketch = float(np.abs(out_sketch - gt).mean())
    bytes_sketch = sketch.bytes_size()
    
    # ====== Path 3: Drop Baseline ======
    K_drop, V_drop, weights_drop = build_drop_sketch(K_all, V_all, sketch_r, seed)
    stats_drop = eval_drop_sketch(K_drop, V_drop, weights_drop, Q, d)
    out_drop = stats_drop.finalize().squeeze(0)
    err_drop = float(np.abs(out_drop - gt).mean())
    bytes_drop = sketch_r * d * 2 * 4  # K_drop + V_drop
    
    # ====== 指标 ======
    compression_ratio_full = 1.0
    compression_ratio_sketch = bytes_sketch / bytes_full
    compression_ratio_drop = bytes_drop / bytes_full
    
    if verbose:
        print(
            f"  kv={kv_len:>5} bs={block_size:>3} r={sketch_r:>2} q={q_len:>2}  "
            f"sketch_err={err_sketch:.3e} drop_err={err_drop:.3e}  "
            f"ratio_sk={compression_ratio_sketch:.3f} dr={compression_ratio_drop:.3f}"
        )
    
    return {
        "kv_len": kv_len,
        "block_size": block_size,
        "sketch_r": sketch_r,
        "q_len": q_len,
        "d": d,
        "clustered": clustered,
        "bytes_full": bytes_full,
        "bytes_sketch": bytes_sketch,
        "bytes_drop": bytes_drop,
        "err_sketch": err_sketch,
        "err_drop": err_drop,
        "compression_sketch": compression_ratio_sketch,
        "compression_drop": compression_ratio_drop,
    }


def run_exploration_A(
    kv_len: int,
    block_size: int,
    sketch_r: int,
    q_len: int,
    d: int = 128,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """探索 A: Attention-pattern-aware sketch vs 均匀 k-means"""
    num_blocks = kv_len // block_size
    
    # 生成有聚类结构的数据
    kv_cache, K_all, V_all = make_clustered_kv(
        num_blocks, block_size, d,
        num_clusters=max(4, kv_len // 256),
        seed=seed
    )
    
    # 生成 query（用于 calibration 和 evaluation）
    Q_cal = (np.random.default_rng(seed + 1000).standard_normal((16, d)) * 0.5).astype(np.float32)
    Q_eval = (np.random.default_rng(seed + 2000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    # Ground truth
    gt = ground_truth(Q_eval, K_all, V_all)
    
    # === 均匀 k-means baseline ===
    sketch_uniform = build_coreset_sketch(K_all, V_all, sketch_r, block_size, seed)
    stats_uniform = eval_coreset_sketch(sketch_uniform, Q_eval, d)
    out_uniform = stats_uniform.finalize().squeeze(0)
    err_uniform = float(np.abs(out_uniform - gt).mean())
    bytes_uniform = sketch_uniform.bytes_size()
    
    # === Attention-aware sketch ===
    sketch_attn = build_attention_aware_sketch(K_all, V_all, Q_cal, sketch_r, block_size, seed)
    stats_attn = eval_attention_aware_sketch(sketch_attn, Q_eval, d)
    out_attn = stats_attn.finalize().squeeze(0)
    err_attn = float(np.abs(out_attn - gt).mean())
    bytes_attn = sketch_attn.centroids.size * 4 + sketch_attn.values.size * 4 + sketch_attn.weights.size * 4
    
    # === Drop baseline ===
    K_drop, V_drop, weights_drop = build_drop_sketch(K_all, V_all, sketch_r, seed)
    stats_drop = eval_drop_sketch(K_drop, V_drop, weights_drop, Q_eval, d)
    out_drop = stats_drop.finalize().squeeze(0)
    err_drop = float(np.abs(out_drop - gt).mean())
    
    improvement = (err_uniform - err_attn) / (err_uniform + 1e-10)
    
    if verbose:
        print(
            f"  Exploration A: kv={kv_len} r={sketch_r}  "
            f"uniform={err_uniform:.3e} attn={err_attn:.3e} drop={err_drop:.3e}  "
            f"improve={improvement*100:.1f}%"
        )
    
    return {
        "exploration": "A_attention_aware",
        "kv_len": kv_len,
        "sketch_r": sketch_r,
        "err_uniform": err_uniform,
        "err_attn": err_attn,
        "err_drop": err_drop,
        "improvement_pct": improvement * 100,
        "bytes_uniform": bytes_uniform,
        "bytes_attn": bytes_attn,
    }


def run_exploration_B(
    kv_len: int,
    block_size: int,
    s: int = 4,
    sub_r: int = 4,
    q_len: int = 64,
    d: int = 128,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """探索 B: Hierarchical sketch vs 单层 k-means"""
    num_blocks = kv_len // block_size
    
    # 生成数据
    kv_cache, K_all, V_all = make_clustered_kv(
        num_blocks, block_size, d,
        num_clusters=max(4, kv_len // 256),
        seed=seed
    )
    
    # Query
    Q = (np.random.default_rng(seed + 2000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K_all, V_all)
    
    total_r = s * sub_r  # 总共 16 个 centroid
    
    # === 单层 k-means (r = s * sub_r) ===
    sketch_single = build_coreset_sketch(K_all, V_all, total_r, block_size, seed)
    stats_single = eval_coreset_sketch(sketch_single, Q, d)
    out_single = stats_single.finalize().squeeze(0)
    err_single = float(np.abs(out_single - gt).mean())
    bytes_single = sketch_single.bytes_size()
    
    # === Hierarchical sketch ===
    sketch_hier = build_hierarchical_sketch(K_all, V_all, s, sub_r, block_size, seed)
    stats_hier = eval_hierarchical_sketch(sketch_hier, Q, d, top_k_super=2)
    out_hier = stats_hier.finalize().squeeze(0)
    err_hier = float(np.abs(out_hier - gt).mean())
    bytes_hier = sketch_hier.bytes_size()
    
    # === Drop baseline ===
    K_drop, V_drop, weights_drop = build_drop_sketch(K_all, V_all, total_r, seed)
    stats_drop = eval_drop_sketch(K_drop, V_drop, weights_drop, Q, d)
    out_drop = stats_drop.finalize().squeeze(0)
    err_drop = float(np.abs(out_drop - gt).mean())
    
    improvement = (err_single - err_hier) / (err_single + 1e-10)
    
    if verbose:
        print(
            f"  Exploration B: kv={kv_len} s={s} sub_r={sub_r}  "
            f"single={err_single:.3e} hier={err_hier:.3e} drop={err_drop:.3e}  "
            f"improve={improvement*100:.1f}%"
        )
    
    return {
        "exploration": "B_hierarchical",
        "kv_len": kv_len,
        "s": s,
        "sub_r": sub_r,
        "total_r": total_r,
        "err_single": err_single,
        "err_hier": err_hier,
        "err_drop": err_drop,
        "improvement_pct": improvement * 100,
        "bytes_single": bytes_single,
        "bytes_hier": bytes_hier,
    }


def run_exploration_C(
    kv_len: int,
    block_size: int,
    sketch_r: int,
    q_len: int,
    d: int = 128,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """探索 C: Coreset + INT4 quantization"""
    num_blocks = kv_len // block_size
    
    # 生成数据
    kv_cache, K_all, V_all = make_clustered_kv(
        num_blocks, block_size, d,
        num_clusters=max(4, kv_len // 256),
        seed=seed
    )
    
    # Query
    Q = (np.random.default_rng(seed + 2000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K_all, V_all)
    
    # === FP32 Coreset ===
    sketch_fp32 = build_coreset_sketch(K_all, V_all, sketch_r, block_size, seed)
    stats_fp32 = eval_coreset_sketch(sketch_fp32, Q, d)
    out_fp32 = stats_fp32.finalize().squeeze(0)
    err_fp32 = float(np.abs(out_fp32 - gt).mean())
    bytes_fp32 = sketch_fp32.bytes_size()
    
    # === INT4 Coreset ===
    sketch_int4 = quantize_sketch(sketch_fp32, n_bits=4)
    sketch_deq = dequantize_sketch(sketch_int4)
    stats_int4 = eval_coreset_sketch(sketch_deq, Q, d)
    out_int4 = stats_int4.finalize().squeeze(0)
    err_int4 = float(np.abs(out_int4 - gt).mean())
    bytes_int4 = sketch_int4.bytes_size()
    
    compression_gain = bytes_fp32 / bytes_int4
    error_increase = (err_int4 - err_fp32) / (err_fp32 + 1e-10)
    
    if verbose:
        print(
            f"  Exploration C: kv={kv_len} r={sketch_r}  "
            f"fp32={err_fp32:.3e} int4={err_int4:.3e}  "
            f"bytes_fp32={bytes_fp32} int4={bytes_int4} gain={compression_gain:.1f}x  "
            f"err_inc={error_increase*100:.1f}%"
        )
    
    return {
        "exploration": "C_int4_quantization",
        "kv_len": kv_len,
        "sketch_r": sketch_r,
        "err_fp32": err_fp32,
        "err_int4": err_int4,
        "bytes_fp32": bytes_fp32,
        "bytes_int4": bytes_int4,
        "compression_gain": compression_gain,
        "error_increase_pct": error_increase * 100,
    }


# ============== 主 Sweep ==============


def run_pareto_sweep(verbose: bool = True) -> dict:
    """运行完整 Pareto sweep"""
    results = []
    
    block_sizes = [32, 64, 128]
    kv_lens = [1024, 4096, 16384]
    sketch_rs = [4, 8, 16, 32]
    q_lens = [16, 64]
    d = 128
    
    if verbose:
        print("=" * 78)
        print("E1 Pareto Sweep: Full KV vs Coreset Sketch vs Drop Baseline")
        print("=" * 78)
    
    for block_size in block_sizes:
        for kv_len in kv_lens:
            if kv_len % block_size != 0:
                continue
            for sketch_r in sketch_rs:
                for q_len in q_lens:
                    # 跳过不合理的配置
                    if sketch_r >= kv_len // block_size:
                        continue
                    
                    # Clustered 数据
                    r = run_pareto_single(
                        kv_len, block_size, sketch_r, q_len, d, 
                        seed=0, clustered=True, verbose=verbose
                    )
                    r["data_type"] = "clustered"
                    results.append(r)
    
    return {"pareto": results}


def run_random_baseline_sweep(verbose: bool = True) -> dict:
    """K 随机时 coreset 是否退化到 drop 同级"""
    results = []
    
    block_sizes = [64]
    kv_lens = [1024, 4096, 16384]
    sketch_rs = [4, 8, 16, 32]
    q_lens = [16, 64]
    d = 128
    
    if verbose:
        print()
        print("=" * 78)
        print("Random K Baseline: Coreset vs Drop (无聚类结构)")
        print("=" * 78)
    
    for block_size in block_sizes:
        for kv_len in kv_lens:
            if kv_len % block_size != 0:
                continue
            for sketch_r in sketch_rs:
                for q_len in q_lens:
                    if sketch_r >= kv_len // block_size:
                        continue
                    
                    r = run_pareto_single(
                        kv_len, block_size, sketch_r, q_len, d,
                        seed=42, clustered=False, verbose=verbose
                    )
                    results.append(r)
    
    return {"random_baseline": results}


def run_exploration_sweeps(verbose: bool = True) -> dict:
    """运行所有探索方向"""
    results_A = []
    results_B = []
    results_C = []
    
    kv_lens = [1024, 4096, 16384]
    sketch_rs = [4, 8, 16, 32]
    d = 128
    
    if verbose:
        print()
        print("=" * 78)
        print("Exploration A: Attention-pattern-aware Sketch")
        print("=" * 78)
    
    for kv_len in kv_lens:
        for r in sketch_rs:
            if r >= kv_len // 64:
                continue
            try:
                ra = run_exploration_A(
                    kv_len, 64, r, 64, d, 
                    seed=0, verbose=verbose
                )
                results_A.append(ra)
            except Exception as e:
                if verbose:
                    print(f"  Exploration A failed for kv={kv_len} r={r}: {e}")
    
    if verbose:
        print()
        print("=" * 78)
        print("Exploration B: Hierarchical Sketch")
        print("=" * 78)
    
    for kv_len in kv_lens:
        for s, sub_r in [(4, 4), (4, 8), (8, 4)]:
            if s * sub_r >= kv_len // 64:
                continue
            try:
                rb = run_exploration_B(
                    kv_len, 64, s, sub_r, 64, d,
                    seed=0, verbose=verbose
                )
                results_B.append(rb)
            except Exception as e:
                if verbose:
                    print(f"  Exploration B failed for kv={kv_len} s={s} sub_r={sub_r}: {e}")
    
    if verbose:
        print()
        print("=" * 78)
        print("Exploration C: Coreset + INT4 Quantization")
        print("=" * 78)
    
    for kv_len in kv_lens:
        for r in sketch_rs:
            if r >= kv_len // 64:
                continue
            try:
                rc = run_exploration_C(
                    kv_len, 64, r, 64, d,
                    seed=0, verbose=verbose
                )
                results_C.append(rc)
            except Exception as e:
                if verbose:
                    print(f"  Exploration C failed for kv={kv_len} r={r}: {e}")
    
    return {
        "exploration_A": results_A,
        "exploration_B": results_B,
        "exploration_C": results_C,
    }


def summarize_results(all_results: dict) -> str:
    """生成 ASCII summary"""
    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("SUMMARY: Coreset Sketch E1 Pareto")
    lines.append("=" * 78)
    
    # Pareto 结果汇总
    pareto = all_results.get("pareto", [])
    if pareto:
        lines.append("\n--- E1 Pareto Results (Clustered Data) ---")
        lines.append(f"{'kv_len':>7} {'r':>3} {'sketch_err':>12} {'drop_err':>12} {'sk_err%':>8}")
        
        for r in pareto:
            sketch_err = r.get("err_sketch", 0)
            drop_err = r.get("err_drop", 0)
            ratio = sketch_err / (drop_err + 1e-10)
            lines.append(
                f"{r['kv_len']:>7} {r['sketch_r']:>3} "
                f"{sketch_err:>12.4e} {drop_err:>12.4e} {ratio:>8.2f}"
            )
        
        # 计算平均改进
        total_ratio = sum(r.get("err_sketch", 0) / (r.get("err_drop", 1) + 1e-10) 
                        for r in pareto) / len(pareto)
        lines.append(f"\nAverage sketch/drop error ratio: {total_ratio:.3f}")
    
    # 探索 A 汇总
    exp_A = all_results.get("exploration_A", [])
    if exp_A:
        lines.append("\n--- Exploration A: Attention-Pattern-Aware ---")
        lines.append(f"{'kv_len':>7} {'r':>3} {'uniform':>12} {'attn':>12} {'improve%':>8}")
        for r in exp_A:
            lines.append(
                f"{r['kv_len']:>7} {r['sketch_r']:>3} "
                f"{r['err_uniform']:>12.4e} {r['err_attn']:>12.4e} {r['improvement_pct']:>8.1f}%"
            )
        avg_improve = sum(r['improvement_pct'] for r in exp_A) / len(exp_A)
        lines.append(f"\nAverage improvement: {avg_improve:.1f}%")
    
    # 探索 B 汇总
    exp_B = all_results.get("exploration_B", [])
    if exp_B:
        lines.append("\n--- Exploration B: Hierarchical Sketch ---")
        lines.append(f"{'kv_len':>7} {'s':>3} {'sub_r':>5} {'single':>12} {'hier':>12} {'improve%':>8}")
        for r in exp_B:
            lines.append(
                f"{r['kv_len']:>7} {r['s']:>3} {r['sub_r']:>5} "
                f"{r['err_single']:>12.4e} {r['err_hier']:>12.4e} {r['improvement_pct']:>8.1f}%"
            )
        avg_improve = sum(r['improvement_pct'] for r in exp_B) / len(exp_B)
        lines.append(f"\nAverage improvement: {avg_improve:.1f}%")
    
    # 探索 C 汇总
    exp_C = all_results.get("exploration_C", [])
    if exp_C:
        lines.append("\n--- Exploration C: INT4 Quantization ---")
        lines.append(f"{'kv_len':>7} {'r':>3} {'fp32_err':>12} {'int4_err':>12} {'gain':>6} {'err_inc%':>8}")
        for r in exp_C:
            lines.append(
                f"{r['kv_len']:>7} {r['sketch_r']:>3} "
                f"{r['err_fp32']:>12.4e} {r['err_int4']:>12.4e} "
                f"{r['compression_gain']:>6.1f}x {r['error_increase_pct']:>8.1f}%"
            )
        avg_gain = sum(r['compression_gain'] for r in exp_C) / len(exp_C)
        lines.append(f"\nAverage compression gain: {avg_gain:.1f}x")
    
    # Random baseline 汇总
    random = all_results.get("random_baseline", [])
    if random:
        lines.append("\n--- Random K Baseline (无聚类结构) ---")
        lines.append(f"{'kv_len':>7} {'r':>3} {'sketch_err':>12} {'drop_err':>12} {'ratio':>8}")
        for r in random:
            sketch_err = r.get("err_sketch", 0)
            drop_err = r.get("err_drop", 0)
            ratio = sketch_err / (drop_err + 1e-10)
            lines.append(
                f"{r['kv_len']:>7} {r['sketch_r']:>3} "
                f"{sketch_err:>12.4e} {drop_err:>12.4e} {ratio:>8.3f}"
            )
        avg_ratio = sum(r.get("err_sketch", 0) / (r.get("err_drop", 1) + 1e-10) 
                       for r in random) / len(random)
        lines.append(f"\nAverage sketch/drop ratio (random K): {avg_ratio:.3f}")
        lines.append("(接近 1.0 说明 coreset 在无结构时退化到 drop)")
    
    return "\n".join(lines)


# ============== Main ==============


def main():
    print("ACCORD-KV Coreset Sketch Experiment")
    print("=" * 78)
    
    # 1. Pareto Sweep (72 组)
    pareto_results = run_pareto_sweep(verbose=True)
    
    # 2. Random Baseline Sweep
    random_results = run_random_baseline_sweep(verbose=True)
    
    # 3. 探索方向
    exploration_results = run_exploration_sweeps(verbose=True)
    
    # 合并所有结果
    all_results = {
        **pareto_results,
        **random_results,
        **exploration_results,
    }
    
    # 打印 summary
    summary = summarize_results(all_results)
    print(summary)
    
    # 保存结果
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp3_coreset_pareto.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
