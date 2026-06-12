"""
Exp21: Facility Location Problem for Token Selection
=====================================================

核心思路：FLP 替代 Coreset 优化 coverage 而非距离
- Coreset (k-means): 最小化总距离 → 偏向均匀分布
- FLP: 最大化 coverage → 偏向覆盖聚类

完整链路: FLP(k) → SVD(r=8) → INT4(b=4)

物理诚实边界: ratio 上限 = k / n ≤ coreset_ratio
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from numpy import linalg as npla

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    ground_truth,
)


# ============== 类型定义 ==============

@dataclass
class FLPSketch:
    """Facility Location sketch 存储结构。"""
    K_selected: np.ndarray  # [r, d] 选中的 K (facilities)
    V_selected: np.ndarray # [r, d] 对应的 V
    weights: np.ndarray    # [r] 权重
    selected_indices: np.ndarray  # [r] 原始索引
    greedy_gains: np.ndarray  # [r] 每步的边际收益

    def bytes_size(self) -> int:
        """估算 sketch 传输字节数（fp32）"""
        return (self.K_selected.size + self.V_selected.size + self.weights.size) * 4


# ============== Facility Location 核心算法 ==============

def compute_cosine_similarity(K: np.ndarray) -> np.ndarray:
    """计算 K 矩阵的 cosine similarity [n, n]"""
    norms = np.linalg.norm(K, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    K_norm = K / norms
    return K_norm @ K_norm.T


def facility_location_greedy(
    K: np.ndarray,
    k: int,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """贪心选择 facility（最大化 coverage）

    Args:
        K: [n, d] key 矩阵
        k: 选择数量
        seed: 随机种子
        verbose: 是否输出进度

    Returns:
        selected_indices: [k] 选中的 token 索引
        greedy_gains: [k] 每步的边际收益

    物理诚实:
        - ratio = k / n，由调用方控制
        - 贪心算法 O(n²k)，长序列可能慢
        - 使用 lazy evaluation 加速
    """
    n, d = K.shape
    k = min(k, n)

    if k == n:
        return np.arange(n), np.zeros(n)

    if verbose:
        print(f"    FLP: n={n}, k={k}, computing similarity matrix...")

    # Step 1: 计算 cosine similarity 矩阵
    t0 = time.time()
    S = compute_cosine_similarity(K)
    if verbose:
        print(f"    FLP: similarity matrix computed in {time.time()-t0:.2f}s")

    # Step 2: 贪心选择
    gen = np.random.default_rng(seed)

    # 初始随机选择第一个 facility
    first_idx = gen.integers(0, n)
    selected = [first_idx]
    selected_set = {first_idx}

    # 已选中的覆盖度：每个 client 被选中的 facility 覆盖的最大相似度
    max_coverage = S[:, first_idx].copy()  # [n]

    # 边际收益
    marginal_gains = [0.0]

    if verbose:
        print(f"    FLP: greedy selection started...")

    # Lazy evaluation: 维护每个 client 的上界（假设后续可选更优的）
    # upper_bound[i] = max_coverage[i] + slack[i]
    slack = np.zeros(n)
    upper_bound = max_coverage.copy()

    # 优先队列：(负上界, idx)，用最小堆实现最大堆
    # 只包含未被选中的
    heap = [(-upper_bound[i], i) for i in range(n) if i != first_idx]
    heapq.heapify(heap)

    for step in range(1, k):
        if not heap:
            # 没有更多候选
            break

        # Lazy evaluation: 弹出上界最大的
        while heap:
            neg_ub, idx = heapq.heappop(heap)
            if idx in selected_set:
                continue
            # 检查上界是否过期
            if -neg_ub <= upper_bound[idx] + 1e-10:
                # 上界有效，选择它
                break
            # 否则继续pop（相当于更新上界）
        else:
            # heap 为空
            break

        # 选择这个 facility
        selected.append(idx)
        selected_set.add(idx)
        marginal_gain = S[idx, :].max() - max_coverage.max()
        marginal_gains.append(float(marginal_gain))

        # 更新 coverage
        new_coverage = np.maximum(max_coverage, S[idx, :])
        coverage_increase = new_coverage - max_coverage
        max_coverage = new_coverage

        # 更新上界（lazy）
        upper_bound += coverage_increase

        # 重新 push 所有未选中的（其实可以优化，但为了正确性简化）
        for i in range(n):
            if i not in selected_set:
                # 只在第一次push，后面通过lazy evaluation跳过
                if i != idx:
                    heapq.heappush(heap, (-upper_bound[i], i))

        if verbose and step % 10 == 0:
            print(f"    FLP: step {step}/{k}, current coverage={max_coverage.mean():.4f}")

    return np.array(selected, dtype=np.int32), np.array(marginal_gains)


# 需要 import heapq
import heapq


def build_flp_sketch(
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    seed: int = 0,
    verbose: bool = False,
) -> FLPSketch:
    """构建 FLP sketch

    物理诚实:
        - r/n ratio 由调用方控制
        - O(n²r) 时间复杂度
    """
    n, d = K.shape
    r = min(r, n)

    if verbose:
        print(f"  Building FLP sketch: n={n}, r={r}")

    # 贪心选择
    t0 = time.time()
    selected_indices, greedy_gains = facility_location_greedy(
        K, r, seed=seed, verbose=verbose
    )
    if verbose:
        print(f"  FLP selection done in {time.time()-t0:.2f}s")

    # 提取选中的 K, V
    K_selected = K[selected_indices].copy()
    V_selected = V[selected_indices].copy()

    # 权重：基于 coverage（简单用均匀权重）
    weights = np.ones(r, dtype=np.float32) / r

    return FLPSketch(
        K_selected=K_selected,
        V_selected=V_selected,
        weights=weights,
        selected_indices=selected_indices,
        greedy_gains=greedy_gains,
    )


def eval_flp_sketch(
    sketch: FLPSketch,
    Q: np.ndarray,
    d: int,
) -> NumpyAttnStats:
    """用 FLP sketch 评估 attention"""
    r = sketch.K_selected.shape[0]
    q_len = Q.shape[0]

    # 计算 attention scores
    scores = Q @ sketch.K_selected.T / math.sqrt(d)  # [q_len, r]
    scores = scores + np.log(sketch.weights + 1e-12)

    # Softmax
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ sketch.V_selected

    return NumpyAttnStats(
        m=m[None, :, :],
        l=l[None, :, :],
        y=y[None, :, :],
    )


# ============== Coreset 实现（对比基线）==============

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建 Coreset sketch（带分配）"""
    n, d = K.shape
    centroids = kmeans_plusplus_init(K, r, seed)
    values = np.zeros((r, d))
    weights = np.zeros(r)
    assignments = np.zeros(n, dtype=np.int32)

    for _ in range(num_iters):
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
        new_assignments = dists.argmin(axis=1)

        new_centroids = np.zeros_like(centroids)
        new_values = np.zeros((r, d))
        new_weights = np.zeros(r)

        for j in range(r):
            mask = new_assignments == j
            count = mask.sum()
            if count > 0:
                new_centroids[j] = K[mask].mean(axis=0)
                new_values[j] = V[mask].mean(axis=0)
                new_weights[j] = count / n
            else:
                new_centroids[j] = centroids[j]
                new_values[j] = V[np.random.default_rng(seed).integers(0, n)]

        centroids = new_centroids
        values = new_values
        weights = new_weights
        assignments = new_assignments

    return centroids, values, weights, assignments


def eval_coreset_sketch(
    centroids: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    Q: np.ndarray,
    d: int,
) -> NumpyAttnStats:
    """评估 Coreset sketch"""
    r = centroids.shape[0]
    scores = Q @ centroids.T / math.sqrt(d)
    scores = scores + np.log(weights + 1e-12)

    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ values

    return NumpyAttnStats(
        m=m[None, :, :],
        l=l[None, :, :],
        y=y[None, :, :],
    )


# ============== SVD + INT4 量化 ==============

def svd_compress(V: np.ndarray, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD 降维压缩 V"""
    U, S, Vt = npla.svd(V, full_matrices=False)
    actual_r = min(r, len(S))
    return U[:, :actual_r], S[:actual_r], Vt[:actual_r, :]


def svd_reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """SVD 重建"""
    return U @ np.diag(S) @ Vt


def quantize_nbit(x: np.ndarray, n_bits: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """N-bit 量化"""
    r, d = x.shape
    max_val = 2 ** (n_bits - 1) - 1

    scales = np.zeros(r, dtype=np.float32)
    x_quant = np.zeros_like(x, dtype=np.int8)

    for j in range(r):
        abs_max = np.abs(x[j]).max()
        if abs_max < 1e-10:
            scales[j] = 1.0
        else:
            scales[j] = abs_max / max_val
            x_quant[j] = np.clip(
                np.round(x[j] / scales[j]),
                -max_val,
                max_val
            ).astype(np.int8)

    return x_quant, scales


def dequantize_nbit(x_quant: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """反量化"""
    r, d = x_quant.shape
    x_rec = np.zeros((r, d), dtype=np.float32)
    for j in range(r):
        x_rec[j] = x_quant[j].astype(np.float32) * scales[j]
    return x_rec


# ============== 串行融合链路 ==============

@dataclass
class FullPipelineSketch:
    """完整链路: FLP → SVD → INT4"""
    flp_sketch: FLPSketch
    U: np.ndarray       # SVD U
    S: np.ndarray       # SVD S
    V_quant: np.ndarray  # 量化后的 V
    V_scales: np.ndarray
    int4_bits: int

    def bytes_size(self) -> int:
        """估算传输字节数"""
        # FLP K: r * d * 4 bytes
        k_bytes = self.flp_sketch.K_selected.size * 4
        # SVD: U, S, V (V 是量化后的)
        v_quant_bytes = self.V_quant.size * (self.int4_bits / 8)
        svd_bytes = self.U.size * 4 + self.S.size * 4
        scales_bytes = self.V_scales.size * 4
        weights_bytes = self.flp_sketch.weights.size * 4
        return int(k_bytes + v_quant_bytes + svd_bytes + scales_bytes + weights_bytes)


def build_full_pipeline(
    K: np.ndarray,
    V: np.ndarray,
    r_flp: int,
    svd_r: int,
    int4_bits: int,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[FullPipelineSketch, dict]:
    """完整链路: FLP → SVD → INT4

    Returns:
        pipeline: 压缩后的 sketch
        info: 中间结果信息
    """
    # Stage 1: FLP
    if verbose:
        print(f"  Stage 1: FLP selection (r={r_flp})")
    flp_sketch = build_flp_sketch(K, V, r_flp, seed=seed, verbose=verbose)

    # Stage 2: SVD
    if verbose:
        print(f"  Stage 2: SVD compression (r={svd_r})")
    U, S, Vt = svd_compress(flp_sketch.V_selected, svd_r)
    V_svd = svd_reconstruct(U, S, Vt)
    svd_error = float(np.abs(V_svd - flp_sketch.V_selected).mean())

    # Stage 3: INT4
    if verbose:
        print(f"  Stage 3: INT{int4_bits} quantization")
    V_quant, V_scales = quantize_nbit(V_svd, int4_bits)
    V_dequant = dequantize_nbit(V_quant, V_scales)
    quant_error = float(np.abs(V_dequant - V_svd).mean())

    pipeline = FullPipelineSketch(
        flp_sketch=flp_sketch,
        U=U,
        S=S,
        V_quant=V_quant,
        V_scales=V_scales,
        int4_bits=int4_bits,
    )

    info = {
        "svd_error": svd_error,
        "quant_error": quant_error,
        "actual_svd_r": U.shape[1],
        "flp_r": r_flp,
        "svd_r": svd_r,
        "int4_bits": int4_bits,
    }

    return pipeline, info


def eval_full_pipeline(
    pipeline: FullPipelineSketch,
    Q: np.ndarray,
    d: int,
) -> Tuple[np.ndarray, float]:
    """评估完整链路

    Returns:
        output: [q_len, d] 输出
        error: 与 ground truth 的误差
    """
    # 重建 V
    V_rec = dequantize_nbit(pipeline.V_quant, pipeline.V_scales)

    # FLP attention
    r = pipeline.flp_sketch.K_selected.shape[0]
    scores = Q @ pipeline.flp_sketch.K_selected.T / math.sqrt(d)
    weights = pipeline.flp_sketch.weights + 1e-12

    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m + np.log(weights))
    l = p.sum(axis=-1, keepdims=True)
    y = p @ V_rec

    return y / np.clip(l, 1e-30, None)


# ============== 数据生成 ==============

def make_clustered_kv(
    kv_len: int,
    d: int,
    num_clusters: int = 8,
    cluster_std: float = 0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成聚类数据"""
    gen = np.random.default_rng(seed)
    cluster_centers = gen.standard_normal((num_clusters, d)) * 2.0
    assignments = gen.integers(0, num_clusters, size=kv_len)

    K = cluster_centers[assignments] + gen.standard_normal((kv_len, d)) * cluster_std
    V = K * 0.8 + gen.standard_normal((kv_len, d)) * 0.2

    return K.astype(np.float32), V.astype(np.float32)


def make_random_kv(
    kv_len: int,
    d: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成随机数据"""
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((kv_len, d)).astype(np.float32) * 0.5
    V = gen.standard_normal((kv_len, d)).astype(np.float32) * 0.5
    return K, V


def make_skewed_kv(
    kv_len: int,
    d: int,
    n_outliers: int = 16,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成偏斜数据（少量 outliers）"""
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
    """小规模 sanity check（3 个数据点）"""
    print("\n" + "=" * 60)
    print("EXP21: Sanity Check (3 data points)")
    print("=" * 60)

    d = 128
    seed = 42
    results = []

    configs = [
        {"kv_len": 256, "r": 16, "kv_type": "clustered", "q_len": 16},
        {"kv_len": 512, "r": 32, "kv_type": "random", "q_len": 32},
        {"kv_len": 1024, "r": 64, "kv_type": "skewed", "q_len": 32},
    ]

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/3] Config: {cfg}")
        kv_len = cfg["kv_len"]
        r = cfg["r"]
        kv_type = cfg["kv_type"]
        q_len = cfg["q_len"]

        # 生成数据
        if kv_type == "clustered":
            K, V = make_clustered_kv(kv_len, d, seed=seed)
        elif kv_type == "random":
            K, V = make_random_kv(kv_len, d, seed=seed)
        else:
            K, V = make_skewed_kv(kv_len, d, seed=seed)

        Q = np.random.default_rng(seed + 100).standard_normal((q_len, d)).astype(np.float32) * 0.5
        gt = ground_truth(Q, K, V)

        # Full KV
        bytes_full = 2 * kv_len * d * 4

        # FLP
        print(f"  Building FLP sketch...")
        flp_sketch = build_flp_sketch(K, V, r, seed=seed, verbose=False)
        stats_flp = eval_flp_sketch(flp_sketch, Q, d)
        out_flp = stats_flp.finalize().squeeze(0)
        err_flp = float(np.abs(out_flp - gt).mean())
        bytes_flp = flp_sketch.bytes_size()

        # Coreset (对比基线)
        print(f"  Building Coreset sketch...")
        centroids, values, weights, _ = build_coreset_sketch(K, V, r, seed=seed)
        stats_coreset = eval_coreset_sketch(centroids, values, weights, Q, d)
        out_coreset = stats_coreset.finalize().squeeze(0)
        err_coreset = float(np.abs(out_coreset - gt).mean())
        bytes_coreset = (centroids.size + values.size + weights.size) * 4

        # 计算 overlap
        flp_set = set(flp_sketch.selected_indices.tolist())
        
        # Coreset: 每个 cluster 的代表是 centroid 最近的点
        dists = np.zeros((kv_len, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
        coreset_assignments = dists.argmin(axis=1)
        coreset_reps = []
        for j in range(r):
            mask = coreset_assignments == j
            if mask.sum() > 0:
                cluster_idx = np.where(mask)[0]
                dist_to_centroid = np.sum((K[cluster_idx] - centroids[j]) ** 2, axis=1)
                rep = cluster_idx[dist_to_centroid.argmin()]
                coreset_reps.append(rep)
        coreset_set = set(coreset_reps)
        overlap = len(flp_set & coreset_set)
        overlap_ratio_flp = overlap / r
        overlap_ratio_coreset = overlap / len(coreset_reps) if coreset_reps else 0

        ratio_flp = bytes_flp / bytes_full
        ratio_coreset = bytes_coreset / bytes_full

        print(f"  Results:")
        print(f"    FLP: err={err_flp:.4e}, ratio={ratio_flp:.4f}, overlap={overlap}/{r}")
        print(f"    Coreset: err={err_coreset:.4e}, ratio={ratio_coreset:.4f}")

        # 诚实报告
        winner = "FLP" if err_flp < err_coreset else "Coreset"
        print(f"    Winner: {winner}")

        results.append({
            "config": cfg,
            "err_flp": err_flp,
            "err_coreset": err_coreset,
            "bytes_flp": bytes_flp,
            "bytes_coreset": bytes_coreset,
            "ratio_flp": ratio_flp,
            "ratio_coreset": ratio_coreset,
            "overlap_count": overlap,
            "overlap_ratio_flp": overlap_ratio_flp,
            "overlap_ratio_coreset": overlap_ratio_coreset,
            "winner": winner,
            "flp_selected_indices": flp_sketch.selected_indices.tolist(),
            "coreset_reps": coreset_reps,
        })

    # Summary
    print("\n" + "-" * 60)
    print("Sanity Check Summary:")
    print("-" * 60)
    flp_wins = sum(1 for r in results if r["winner"] == "FLP")
    print(f"FLP wins: {flp_wins}/3")
    print(f"Coreset wins: {3 - flp_wins}/3")

    avg_overlap = np.mean([r["overlap_ratio_flp"] for r in results])
    print(f"Average overlap ratio (FLP selected): {avg_overlap:.2%}")

    # 诚实说明
    print("\n[诚实说明]")
    print("- FLP 在 random 数据上理论上不应优于 Coreset（因为没有聚类结构）")
    print("- FLP 贪心算法 O(n²r)，长序列可能慢")
    print("- overlap 分析反映两种方法选出的代表性 token 差异")

    return results


# ============== 完整 Sweep ==============

def run_full_sweep(seed: int = 42):
    """完整 sweep"""
    print("\n" + "=" * 60)
    print("EXP21: Full Sweep")
    print("=" * 60)

    d = 128
    results = []
    start_time = time.time()

    # 配置
    kv_types = ["clustered", "random", "skewed"]
    kv_lens = [1024, 4096]
    q_lens = [16, 64]
    flp_ratios = [0.0625, 0.125, 0.25]  # 64/1024, 128/1024, 256/1024, etc.

    total_configs = len(kv_types) * len(kv_lens) * len(q_lens) * len(flp_ratios)

    config_idx = 0

    for kv_type in kv_types:
        for kv_len in kv_lens:
            for q_len in q_lens:
                # 生成数据
                if kv_type == "clustered":
                    K, V = make_clustered_kv(kv_len, d, seed=seed)
                elif kv_type == "random":
                    K, V = make_random_kv(kv_len, d, seed=seed)
                else:
                    K, V = make_skewed_kv(kv_len, d, seed=seed)

                Q = np.random.default_rng(seed + 100).standard_normal((q_len, d)).astype(np.float32) * 0.5
                gt = ground_truth(Q, K, V)
                bytes_full = 2 * kv_len * d * 4

                for flp_ratio in flp_ratios:
                    config_idx += 1
                    r = max(4, int(kv_len * flp_ratio))

                    # 物理诚实检查
                    actual_ratio = r / kv_len
                    if actual_ratio > 0.5:  # 超过 50% 压缩不合理
                        continue

                    # FLP
                    t0 = time.time()
                    flp_sketch = build_flp_sketch(K, V, r, seed=seed, verbose=False)
                    stats_flp = eval_flp_sketch(flp_sketch, Q, d)
                    out_flp = stats_flp.finalize().squeeze(0)
                    err_flp = float(np.abs(out_flp - gt).mean())
                    bytes_flp = flp_sketch.bytes_size()
                    flp_time = time.time() - t0

                    # Coreset
                    t0 = time.time()
                    centroids, values, weights, _ = build_coreset_sketch(K, V, r, seed=seed)
                    stats_coreset = eval_coreset_sketch(centroids, values, weights, Q, d)
                    out_coreset = stats_coreset.finalize().squeeze(0)
                    err_coreset = float(np.abs(out_coreset - gt).mean())
                    bytes_coreset = (centroids.size + values.size + weights.size) * 4
                    coreset_time = time.time() - t0

                    # 计算 overlap
                    flp_set = set(flp_sketch.selected_indices.tolist())
                    dists = np.zeros((kv_len, r))
                    for j in range(r):
                        dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
                    coreset_assignments = dists.argmin(axis=1)
                    coreset_reps = []
                    for j in range(r):
                        mask = coreset_assignments == j
                        if mask.sum() > 0:
                            cluster_idx = np.where(mask)[0]
                            dist_to_centroid = np.sum((K[cluster_idx] - centroids[j]) ** 2, axis=1)
                            rep = cluster_idx[dist_to_centroid.argmin()]
                            coreset_reps.append(rep)
                    coreset_set = set(coreset_reps)
                    overlap = len(flp_set & coreset_set)

                    ratio_flp = bytes_flp / bytes_full
                    ratio_coreset = bytes_coreset / bytes_full

                    winner = "FLP" if err_flp < err_coreset else "Coreset"
                    improvement = (err_coreset - err_flp) / (err_coreset + 1e-10)

                    results.append({
                        "kv_type": kv_type,
                        "kv_len": kv_len,
                        "q_len": q_len,
                        "r": r,
                        "flp_ratio": actual_ratio,
                        "err_flp": err_flp,
                        "err_coreset": err_coreset,
                        "improvement": improvement,
                        "bytes_flp": bytes_flp,
                        "bytes_coreset": bytes_coreset,
                        "ratio_flp": ratio_flp,
                        "ratio_coreset": ratio_coreset,
                        "time_flp": flp_time,
                        "time_coreset": coreset_time,
                        "overlap_count": overlap,
                        "overlap_ratio_flp": overlap / r,
                        "winner": winner,
                    })

                    if config_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = config_idx / elapsed
                        remaining = (total_configs - config_idx) / rate if rate > 0 else 0
                        print(f"  Progress: {config_idx}/{total_configs} ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

    # 按 kv_type 分组统计
    by_kv_type = {}
    for kv_type in kv_types:
        subset = [r for r in results if r["kv_type"] == kv_type]
        if subset:
            flp_wins = sum(1 for r in subset if r["winner"] == "FLP")
            by_kv_type[kv_type] = {
                "count": len(subset),
                "flp_wins": flp_wins,
                "avg_improvement": np.mean([r["improvement"] for r in subset]),
                "avg_overlap": np.mean([r["overlap_ratio_flp"] for r in subset]),
            }

    elapsed_total = time.time() - start_time

    print(f"\nSweep complete in {elapsed_total:.1f}s")
    print(f"Total configs: {len(results)}")

    return {
        "results": results,
        "by_kv_type": by_kv_type,
        "total_time": elapsed_total,
    }


def run_full_pipeline_sweep(seed: int = 42):
    """完整链路 FLP → SVD → INT4 sweep"""
    print("\n" + "=" * 60)
    print("EXP21: Full Pipeline Sweep (FLP → SVD → INT4)")
    print("=" * 60)

    d = 128
    results = []
    start_time = time.time()

    kv_types = ["clustered", "random", "skewed"]
    kv_lens = [1024, 4096]
    q_lens = [16, 64]
    flp_ratios = [0.0625, 0.125, 0.25]
    svd_r_values = [4, 8]
    int4_bits = [4, 8]

    total_configs = len(kv_types) * len(kv_lens) * len(q_lens) * len(flp_ratios) * len(svd_r_values) * len(int4_bits)

    config_idx = 0

    for kv_type in kv_types:
        for kv_len in kv_lens:
            for q_len in q_lens:
                if kv_type == "clustered":
                    K, V = make_clustered_kv(kv_len, d, seed=seed)
                elif kv_type == "random":
                    K, V = make_random_kv(kv_len, d, seed=seed)
                else:
                    K, V = make_skewed_kv(kv_len, d, seed=seed)

                Q = np.random.default_rng(seed + 100).standard_normal((q_len, d)).astype(np.float32) * 0.5
                gt = ground_truth(Q, K, V)
                bytes_full = 2 * kv_len * d * 4

                for flp_ratio in flp_ratios:
                    r = max(4, int(kv_len * flp_ratio))
                    if r / kv_len > 0.5:
                        continue

                    for svd_r in svd_r_values:
                        for bits in int4_bits:
                            config_idx += 1

                            t0 = time.time()
                            try:
                                pipeline, info = build_full_pipeline(
                                    K, V, r, svd_r, bits, seed=seed, verbose=False
                                )
                                out_pipeline = eval_full_pipeline(pipeline, Q, d)
                                err_pipeline = float(np.abs(out_pipeline - gt).mean())
                                bytes_pipeline = pipeline.bytes_size()

                                # 对比基线
                                _, values_baseline, weights_baseline, _ = build_coreset_sketch(K, V, r, seed=seed)
                                stats_baseline = eval_coreset_sketch(
                                    values_baseline * 0, values_baseline, weights_baseline, Q, d
                                )
                                # 简化：用 coreset + SVD + INT4 作为基线
                                coreset_cent, coreset_val, coreset_w, _ = build_coreset_sketch(K, V, r, seed=seed)
                                U_c, S_c, Vt_c = svd_compress(coreset_val, svd_r)
                                V_c_svd = svd_reconstruct(U_c, S_c, Vt_c)
                                V_c_quant, V_c_scales = quantize_nbit(V_c_svd, bits)
                                V_c_deq = dequantize_nbit(V_c_quant, V_c_scales)
                                stats_baseline = eval_coreset_sketch(coreset_cent, V_c_deq, coreset_w, Q, d)
                                out_baseline = stats_baseline.finalize().squeeze(0)
                                err_baseline = float(np.abs(out_baseline - gt).mean())

                                pipeline_time = time.time() - t0

                                ratio_pipeline = bytes_pipeline / bytes_full
                                ratio_baseline = (r * d * 4 + r * svd_r * 4 + r * d * (bits / 8) + r * 4 + r * 4) / bytes_full

                                winner = "FLP" if err_pipeline < err_baseline else "Baseline"

                                results.append({
                                    "kv_type": kv_type,
                                    "kv_len": kv_len,
                                    "q_len": q_len,
                                    "flp_r": r,
                                    "svd_r": svd_r,
                                    "int4_bits": bits,
                                    "err_pipeline": err_pipeline,
                                    "err_baseline": err_baseline,
                                    "bytes_pipeline": bytes_pipeline,
                                    "ratio_pipeline": ratio_pipeline,
                                    "ratio_baseline": ratio_baseline,
                                    "time_pipeline": pipeline_time,
                                    "winner": winner,
                                    "svd_error": info["svd_error"],
                                    "quant_error": info["quant_error"],
                                })

                            except Exception as e:
                                print(f"  Error at config {config_idx}: {e}")
                                continue

                            if config_idx % 20 == 0:
                                elapsed = time.time() - start_time
                                rate = config_idx / elapsed
                                remaining = (total_configs - config_idx) / rate if rate > 0 else 0
                                print(f"  Progress: {config_idx}/{total_configs} ({elapsed:.1f}s elapsed)")

    elapsed_total = time.time() - start_time

    print(f"\nPipeline Sweep complete in {elapsed_total:.1f}s")
    print(f"Total configs: {len(results)}")

    return {
        "results": results,
        "total_time": elapsed_total,
    }


# ============== 主函数 ==============

def main():
    print("=" * 60)
    print("EXP21: Facility Location Problem for Token Selection")
    print("=" * 60)
    print("\n核心假设：FLP 优化 coverage 而非距离，在 clustered 数据上理论上更优")
    print("诚实边界：ratio 上限 = k/n ≤ 0.5\n")

    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Sanity Check
    sanity_results = run_sanity_check()

    with open(os.path.join(output_dir, "exp21_sanity.json"), "w") as f:
        json.dump({"sanity": sanity_results}, f, indent=2)
    print(f"\nSaved: results/exp21_sanity.json")

    # 2. Full Sweep
    sweep_results = run_full_sweep(seed=42)

    with open(os.path.join(output_dir, "exp21_sweep.json"), "w") as f:
        json.dump(sweep_results, f, indent=2, default=str)
    print(f"Saved: results/exp21_sweep.json")

    # 3. Full Pipeline Sweep
    pipeline_results = run_full_pipeline_sweep(seed=42)

    with open(os.path.join(output_dir, "exp21_pipeline_sweep.json"), "w") as f:
        json.dump(pipeline_results, f, indent=2, default=str)
    print(f"Saved: results/exp21_pipeline_sweep.json")

    # 4. 生成报告
    report = generate_report(sanity_results, sweep_results, pipeline_results)

    with open(os.path.join(output_dir, "exp21_facility_location_report.md"), "w") as f:
        f.write(report)
    print(f"Saved: results/exp21_facility_location_report.md")

    print("\n" + "=" * 60)
    print("EXP21 Complete")
    print("=" * 60)

    return sanity_results, sweep_results, pipeline_results


def generate_report(sanity: list, sweep: dict, pipeline: dict) -> str:
    """生成诚实报告"""
    lines = []

    lines.append("# EXP21: Facility Location Problem Analysis Report")
    lines.append("")
    lines.append("## 核心假设")
    lines.append("- Coreset (k-means): 最小化总距离 → 偏向均匀分布")
    lines.append("- FLP: 最大化 coverage → 偏向覆盖聚类")
    lines.append("- 假设：在有聚类结构的数据上，FLP 应该优于 Coreset")
    lines.append("")
    lines.append("## 物理诚实边界")
    lines.append("- ratio 上限 = k / n ≤ 0.5")
    lines.append("- FLP 贪心算法 O(n²k)，长序列可能慢")
    lines.append("- 贪心算法有 (1-1/e) ≈ 0.632 近似比，但不能保证全局最优")
    lines.append("")
    lines.append("## Sanity Check 结果")
    lines.append("")

    # Sanity summary
    flp_wins = sum(1 for r in sanity if r["winner"] == "FLP")
    lines.append(f"| 配置 | FLP 误差 | Coreset 误差 | 胜者 |")
    lines.append(f"|------|---------|-------------|------|")
    for r in sanity:
        cfg = r["config"]
        lines.append(f"| {cfg['kv_type']} kv={cfg['kv_len']} r={cfg['r']} | {r['err_flp']:.4e} | {r['err_coreset']:.4e} | {r['winner']} |")
    lines.append("")
    lines.append(f"FLP 胜率: {flp_wins}/3")
    lines.append("")
    lines.append("## Full Sweep 结果")
    lines.append("")

    # By kv_type
    by_kv_type = sweep.get("by_kv_type", {})
    for kv_type, stats in by_kv_type.items():
        lines.append(f"### {kv_type}")
        lines.append(f"- 配置数: {stats['count']}")
        lines.append(f"- FLP 胜率: {stats['flp_wins']}/{stats['count']}")
        lines.append(f"- 平均改善: {stats['avg_improvement']*100:+.2f}%")
        lines.append(f"- 平均 overlap: {stats['avg_overlap']*100:.1f}%")
        lines.append("")

    lines.append("## 完整链路 (FLP → SVD → INT4)")
    lines.append("")

    # Pipeline 结果
    pipeline_results = pipeline.get("results", [])
    if pipeline_results:
        flp_pipeline_wins = sum(1 for r in pipeline_results if r["winner"] == "FLP")
        avg_err_flp = np.mean([r["err_pipeline"] for r in pipeline_results])
        avg_err_baseline = np.mean([r["err_baseline"] for r in pipeline_results])

        lines.append(f"| 配置 | FLP 链路误差 | 基线误差 | 胜者 |")
        lines.append(f"|------|-------------|----------|------|")
        for r in pipeline_results[:10]:  # 只显示前 10 个
            lines.append(f"| {r['kv_type']} kv={r['kv_len']} r={r['flp_r']} | {r['err_pipeline']:.4e} | {r['err_baseline']:.4e} | {r['winner']} |")
        lines.append("")
        lines.append(f"完整链路 FLP 胜率: {flp_pipeline_wins}/{len(pipeline_results)}")
        lines.append(f"平均 FLP 误差: {avg_err_flp:.4e}")
        lines.append(f"平均基线误差: {avg_err_baseline:.4e}")
        lines.append("")

    lines.append("## 诚实结论")
    lines.append("")

    # 计算统计数据
    sweep_results = sweep.get("results", [])

    # 按数据类型的胜率
    clustered_wins = 0
    clustered_total = 0
    random_wins = 0
    random_total = 0
    skewed_wins = 0
    skewed_total = 0

    for r in sweep_results:
        if r["kv_type"] == "clustered":
            clustered_total += 1
            if r["winner"] == "FLP":
                clustered_wins += 1
        elif r["kv_type"] == "random":
            random_total += 1
            if r["winner"] == "FLP":
                random_wins += 1
        else:
            skewed_total += 1
            if r["winner"] == "FLP":
                skewed_wins += 1

    lines.append("1. **在 clustered 数据上**:")
    if clustered_total > 0:
        lines.append(f"   - FLP 胜率: {clustered_wins}/{clustered_total} ({clustered_wins/clustered_total*100:.1f}%)")
    lines.append("   - 结论: " + ("**FLP 确实表现更好**" if clustered_wins > clustered_total * 0.5 else "**FLP 不一定优于 Coreset**"))
    lines.append("")
    lines.append("2. **在 random 数据上**:")
    if random_total > 0:
        lines.append(f"   - FLP 胜率: {random_wins}/{random_total} ({random_wins/random_total*100:.1f}%)")
    lines.append("   - 结论: FLP 在没有聚类结构的数据上理论上不应有优势")
    lines.append("")
    lines.append("3. **在 skewed 数据上**:")
    if skewed_total > 0:
        lines.append(f"   - FLP 胜率: {skewed_wins}/{skewed_total} ({skewed_wins/skewed_total*100:.1f}%)")
    lines.append("   - 结论: 需要具体分析 outliers 对 coverage 的影响")
    lines.append("")
    lines.append("## 已知局限")
    lines.append("")
    lines.append("- FLP 贪心算法在长序列 (n > 10000) 上可能很慢 (O(n²k))")
    lines.append("- Lazy evaluation 优化，但仍需计算完整相似度矩阵")
    lines.append("- 贪心近似比 (1-1/e) 不能保证每次都好")
    lines.append("- 均匀权重可能不是最优（可考虑加权 FLP）")
    lines.append("")
    lines.append("## 建议")
    lines.append("")
    lines.append("1. **短序列 (< 4096)**: FLP 可以尝试，效果取决于数据结构")
    lines.append("2. **长序列 (> 4096)**: 建议先用 random sampling 或 importance sampling 减少 n")
    lines.append("3. **混合方法**: 先 FLP 选 diverse set，再 k-means fine-tune")
    lines.append("4. **加权 FLP**: 考虑 token 的 attention weight 作为 facility 的初始收益")
    lines.append("")
    lines.append("---")
    lines.append("*报告生成时间: 由 exp21_facility_location.py 自动生成*")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
