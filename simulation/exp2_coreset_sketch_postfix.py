"""
Exp2 Postfix: Coreset Sketch with Bug Fixes
============================================

修复了审查发现的 2 个 bug：

Bug 1 (Line 247-251): 收敛条件在赋值后检查
  - 旧: centroids = new_centroids; centroid_shift = np.sum((centroids - new_centroids)**2)
  - 新: centroid_shift = np.sum((centroids - new_centroids)**2); centroids = new_centroids; if < 1e-8: break

Bug 2 (Line 530-535): K/V 用相同 scale，应分开
  - 旧: c_max = max(|centroids|.max, |values|.max); 统一用 c_max/7
  - 新: k_scale = |centroids|.max/7; v_scale = |values|.max/7; 分开存储

新增:
  - quantize_sketch_nbit(): INT1-8 通用量化
  - dequantize_sketch_nbit(): 反量化

运行内容:
  - 30 组 E1 Pareto sweep (seed=42)
  - 20 组 INT4 sweep (seed=42)
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Tuple

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
    """Coreset sketch 存储结构。"""
    centroids: np.ndarray  # [r, d]
    values: np.ndarray     # [r, d]
    weights: np.ndarray    # [r]
    assignments: np.ndarray  # [kv_len]

    def bytes_size(self) -> int:
        """估算 sketch 传输字节数（fp32）"""
        return (self.centroids.size + self.values.size + self.weights.size) * 4


# ============== 工具函数 ==============

def kmeans_plusplus_init(K: np.ndarray, r: int, seed: int = 0) -> np.ndarray:
    """K-Means++ 初始化。"""
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


_gen_counter = 0

def gen_in_range(seed: int, bound: int) -> int:
    global _gen_counter
    _gen_counter += 1
    return (seed * 1103515245 + _gen_counter) % bound


# ============== 核心 Sketch 实现（Bug 1 已修复）==============

def build_coreset_sketch(
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    block_size: int = 64,
    seed: int = 0,
    num_iters: int = 15,
) -> CoresetSketch:
    """Lloyd iterations k-means 构建 coreset sketch。
    
    Bug 1 Fix: 收敛检查移到 centroid 赋值之后，保证 shift 计算正确。
    """
    n, d = K.shape
    gen = np.random.default_rng(seed)

    centroids = kmeans_plusplus_init(K, r, seed)

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

        # ===== BUG 1 FIX: 先算 shift，后赋值，再检查 =====
        centroid_shift = np.sum((centroids - new_centroids) ** 2)
        centroids = new_centroids
        if centroid_shift < 1e-8:
            break

    # 最终分配
    dists = np.zeros((n, r))
    for j in range(r):
        dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
    final_assignments = dists.argmin(axis=1)

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
    """用 coreset sketch 评估 attention。"""
    q_len = Q.shape[0]
    r = sketch.centroids.shape[0]

    scores = Q @ sketch.centroids.T / math.sqrt(d)
    scores = scores + np.log(sketch.weights + 1e-12)

    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ sketch.values

    return NumpyAttnStats(
        m=m[None, :, :],
        l=l[None, :, :],
        y=y[None, :, :],
    )


# ============== 量化（Bug 2 已修复 + 推广到 n_bit）==============

@dataclass
class QuantizedSketch:
    """N-bit 量化后的 sketch（K/V scale 分开）。"""
    centroids_int: np.ndarray    # [r, d] int quantized
    values_int: np.ndarray        # [r, d] int quantized
    weights: np.ndarray           # [r] fp32
    scales: np.ndarray            # [r, 2] K-scale and V-scale
    n_bits: int

    def bytes_size(self) -> int:
        """压缩后字节数估算（INT4: 2/byte; INT8: 1/byte）。"""
        bits_total = self.n_bits * (self.centroids_int.size + self.values_int.size)
        bytes_kv = math.ceil(bits_total / 8)
        # weights + scales 都是 fp32
        return bytes_kv + self.weights.size * 4 + self.scales.size * 4


def quantize_sketch(
    sketch: CoresetSketch,
    n_bits: int = 4,
) -> QuantizedSketch:
    """将 sketch 量化到 n-bit（Bug 2 Fix: K/V scale 分开）。
    
    Bug 2 Fix:
      - 旧: c_max = max(|centroids|.max, |values|.max); K 和 V 共用同一 scale
      - 新: k_scale = |centroids|.max / level; v_scale = |values|.max / level
    """
    r, d = sketch.centroids.shape

    # scale 级别: 对称 INT，范围 [-7, 7] 对 INT4，通用化:
    # max_val = 2^(n-1) - 1
    max_val = 2 ** (n_bits - 1) - 1
    if max_val < 1:
        max_val = 1  # INT1 fallback

    # Bug 2 fix: 分开计算 K 和 V 的 scale
    c_scales = np.zeros((r, 2), dtype=np.float32)
    for j in range(r):
        c_scales[j, 0] = np.abs(sketch.centroids[j]).max() / max_val  # K scale
        c_scales[j, 1] = np.abs(sketch.values[j]).max() / max_val       # V scale
        # 避免 scale 为 0
        if c_scales[j, 0] < 1e-10:
            c_scales[j, 0] = 1e-10
        if c_scales[j, 1] < 1e-10:
            c_scales[j, 1] = 1e-10

    dtype = np.int8
    centroids_int = np.zeros((r, d), dtype=dtype)
    values_int = np.zeros((r, d), dtype=dtype)

    for j in range(r):
        centroids_int[j] = np.clip(
            np.round(sketch.centroids[j] / c_scales[j, 0]),
            -max_val,
            max_val
        ).astype(dtype)
        values_int[j] = np.clip(
            np.round(sketch.values[j] / c_scales[j, 1]),
            -max_val,
            max_val
        ).astype(dtype)

    return QuantizedSketch(
        centroids_int=centroids_int,
        values_int=values_int,
        weights=sketch.weights.copy(),
        scales=c_scales,
        n_bits=n_bits,
    )


def dequantize_sketch(q_sketch: QuantizedSketch) -> CoresetSketch:
    """反量化回 float。"""
    r, d = q_sketch.centroids_int.shape

    centroids = np.zeros((r, d), dtype=np.float32)
    values = np.zeros((r, d), dtype=np.float32)

    for j in range(r):
        centroids[j] = q_sketch.centroids_int[j].astype(np.float32) * q_sketch.scales[j, 0]
        values[j] = q_sketch.values_int[j].astype(np.float32) * q_sketch.scales[j, 1]

    return CoresetSketch(
        centroids=centroids,
        values=values,
        weights=q_sketch.weights,
        assignments=np.zeros(r, dtype=np.int32),
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
    """生成有聚类结构的 K/V 数据。"""
    kv_len = num_blocks * block_size
    gen = np.random.default_rng(seed)

    cluster_centers = gen.standard_normal((num_clusters, d)) * 2.0

    cluster_assign = np.zeros(kv_len, dtype=np.int32)
    tokens_per_cluster = kv_len // num_clusters
    for i in range(num_clusters):
        start = i * tokens_per_cluster
        end = start + tokens_per_cluster if i < num_clusters - 1 else kv_len
        cluster_assign[start:end] = i

    cluster_assign = gen.choice(num_clusters, kv_len).astype(np.int32)

    K_all = np.zeros((kv_len, d), dtype=np.float32)
    V_all = np.zeros((kv_len, d), dtype=np.float32)

    for i in range(kv_len):
        c = cluster_assign[i]
        K_all[i] = cluster_centers[c] + gen.standard_normal(d) * cluster_std
        V_all[i] = K_all[i] * 0.8 + gen.standard_normal(d) * 0.2

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
    """生成随机 K/V（无聚类结构）。"""
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
    """Uniform random drop baseline。"""
    n = len(K)
    gen = np.random.default_rng(seed)

    keep_idx = gen.choice(n, r, replace=False)
    keep_idx = np.sort(keep_idx)

    K_drop = K[keep_idx]
    V_drop = V[keep_idx]
    weights = np.ones(r) / r

    return K_drop, V_drop, weights


def eval_drop_sketch(
    K_drop: np.ndarray,
    V_drop: np.ndarray,
    weights: np.ndarray,
    Q: np.ndarray,
    d: int,
) -> NumpyAttnStats:
    """评估 drop baseline。"""
    r = len(K_drop)
    q_len = Q.shape[0]

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


# ============== 单组仿真 ==============

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
    """单组配置仿真。"""
    num_blocks = kv_len // block_size

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

    Q = (np.random.default_rng(seed + 1000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K_all, V_all)

    # Full KV
    bytes_full = 2 * kv_len * d * 4

    # Coreset Sketch
    sketch = build_coreset_sketch(K_all, V_all, sketch_r, block_size, seed)
    stats_sketch = eval_coreset_sketch(sketch, Q, d)
    out_sketch = stats_sketch.finalize().squeeze(0)
    err_sketch = float(np.abs(out_sketch - gt).mean())
    bytes_sketch = sketch.bytes_size()

    # Drop Baseline
    K_drop, V_drop, weights_drop = build_drop_sketch(K_all, V_all, sketch_r, seed)
    stats_drop = eval_drop_sketch(K_drop, V_drop, weights_drop, Q, d)
    out_drop = stats_drop.finalize().squeeze(0)
    err_drop = float(np.abs(out_drop - gt).mean())
    bytes_drop = sketch_r * d * 2 * 4

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


def run_int4_single(
    kv_len: int,
    block_size: int,
    sketch_r: int,
    q_len: int,
    d: int = 128,
    seed: int = 0,
    n_bits: int = 4,
    verbose: bool = True,
) -> dict:
    """单组 INT4/INTn 配置仿真。"""
    num_blocks = kv_len // block_size

    kv_cache, K_all, V_all = make_clustered_kv(
        num_blocks, block_size, d,
        num_clusters=max(4, kv_len // 256),
        seed=seed
    )

    Q = (np.random.default_rng(seed + 1000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K_all, V_all)

    # FP32 Coreset
    sketch_fp32 = build_coreset_sketch(K_all, V_all, sketch_r, block_size, seed)
    stats_fp32 = eval_coreset_sketch(sketch_fp32, Q, d)
    out_fp32 = stats_fp32.finalize().squeeze(0)
    err_fp32 = float(np.abs(out_fp32 - gt).mean())
    bytes_fp32 = sketch_fp32.bytes_size()

    # Quantized Coreset
    q_sketch = quantize_sketch(sketch_fp32, n_bits=n_bits)
    sketch_deq = dequantize_sketch(q_sketch)
    stats_intn = eval_coreset_sketch(sketch_deq, Q, d)
    out_intn = stats_intn.finalize().squeeze(0)
    err_intn = float(np.abs(out_intn - gt).mean())
    bytes_intn = q_sketch.bytes_size()

    compression_gain = bytes_fp32 / bytes_intn
    error_increase = (err_intn - err_fp32) / (err_fp32 + 1e-10)

    if verbose:
        print(
            f"  kv={kv_len:>5} r={sketch_r:>2} n_bits={n_bits}  "
            f"fp32={err_fp32:.3e} int{n_bits}={err_intn:.3e}  "
            f"gain={compression_gain:.1f}x err_inc={error_increase*100:.2f}%"
        )

    return {
        "kv_len": kv_len,
        "block_size": block_size,
        "sketch_r": sketch_r,
        "q_len": q_len,
        "d": d,
        "n_bits": n_bits,
        "err_fp32": err_fp32,
        "err_intn": err_intn,
        "bytes_fp32": bytes_fp32,
        "bytes_intn": bytes_intn,
        "compression_gain": compression_gain,
        "error_increase_pct": error_increase * 100,
        "error_increase_abs": err_intn - err_fp32,
    }


# ============== 主 Sweep ==============

def run_pareto_sweep(seed: int = 42, verbose: bool = True) -> list:
    """运行完整 Pareto sweep（30 组，seed=42）。"""
    results = []

    block_sizes = [32, 64, 128]
    kv_lens = [1024, 4096, 16384]
    sketch_rs = [4, 8, 16, 32]
    q_lens = [16, 64]
    d = 128

    if verbose:
        print("=" * 78)
        print(f"E1 Pareto Sweep (postfix, seed={seed}): Full KV vs Coreset vs Drop")
        print("=" * 78)

    for block_size in block_sizes:
        for kv_len in kv_lens:
            if kv_len % block_size != 0:
                continue
            for sketch_r in sketch_rs:
                if sketch_r >= kv_len // block_size:
                    continue
                for q_len in q_lens:
                    r = run_pareto_single(
                        kv_len, block_size, sketch_r, q_len, d,
                        seed=seed, clustered=True, verbose=verbose
                    )
                    results.append(r)

    return results


def run_int4_sweep(seed: int = 42, n_bits: int = 4, verbose: bool = True) -> list:
    """运行 INT4 sweep（20 组，seed=42）。"""
    results = []

    block_sizes = [32, 64, 128]
    kv_lens = [1024, 4096, 16384]
    sketch_rs = [4, 8, 16, 32]
    q_lens = [16, 64]
    d = 128

    if verbose:
        print()
        print("=" * 78)
        print(f"INT{n_bits} Sweep (postfix, seed={seed}): Coreset FP32 vs INT{n_bits}")
        print("=" * 78)

    count = 0
    for block_size in block_sizes:
        for kv_len in kv_lens:
            if kv_len % block_size != 0:
                continue
            for sketch_r in sketch_rs:
                if sketch_r >= kv_len // block_size:
                    continue
                for q_len in q_lens:
                    count += 1
                    if count > 20:
                        break
                    r = run_int4_single(
                        kv_len, block_size, sketch_r, q_len, d,
                        seed=seed, n_bits=n_bits, verbose=verbose
                    )
                    results.append(r)
            if count > 20:
                break

    return results


def summarize_postfix(pareto_results: list, int4_results: list) -> str:
    """生成修复后 summary。"""
    lines = []
    lines.append("\n" + "=" * 78)
    lines.append("SUMMARY: Coreset Sketch Postfix (Bug Fix Verification)")
    lines.append("=" * 78)

    # Pareto
    if pareto_results:
        sketch_errs = [r["err_sketch"] for r in pareto_results]
        drop_errs = [r["err_drop"] for r in pareto_results]
        ratios = [r["err_sketch"] / (r["err_drop"] + 1e-10) for r in pareto_results]
        avg_ratio = sum(ratios) / len(ratios)

        lines.append(f"\n--- E1 Pareto (seed=42, {len(pareto_results)} configs) ---")
        lines.append(f"Average sketch/drop ratio: {avg_ratio:.4f}")
        lines.append(f"Average sketch error: {sum(sketch_errs)/len(sketch_errs):.4e}")
        lines.append(f"Average drop error: {sum(drop_errs)/len(drop_errs):.4e}")
        lines.append(f"(与修复前 0.895 对比，应该一致)")

    # INT4
    if int4_results:
        gains = [r["compression_gain"] for r in int4_results]
        err_incs = [r["error_increase_pct"] for r in int4_results]
        err_abs = [r["error_increase_abs"] for r in int4_results]
        n_bits = int4_results[0]["n_bits"]

        lines.append(f"\n--- INT{n_bits} Quantization (seed=42, {len(int4_results)} configs) ---")
        lines.append(f"Average compression gain: {sum(gains)/len(gains):.2f}x")
        lines.append(f"Average error increase: {sum(err_incs)/len(err_incs):.2f}%")
        lines.append(f"Average absolute error increase: {sum(err_abs)/len(err_abs):.4e}")
        lines.append(f"预期: error increase < 0.3%（比修复前 0.43% 更低）")

        # 按 r 分组
        for r_val in [4, 8, 16, 32]:
            sub = [x for x in int4_results if x["sketch_r"] == r_val]
            if sub:
                avg_inc = sum(x["error_increase_pct"] for x in sub) / len(sub)
                avg_gain = sum(x["compression_gain"] for x in sub) / len(sub)
                lines.append(f"  r={r_val}: avg_gain={avg_gain:.2f}x avg_err_inc={avg_inc:.2f}%")

    return "\n".join(lines)


def main():
    print("ACCORD-KV Coreset Sketch Postfix: Bug Fix Verification")
    print("=" * 78)

    # 1. Pareto sweep (30 configs, seed=42)
    pareto_results = run_pareto_sweep(seed=42, verbose=True)

    # 2. INT4 sweep (20 configs, seed=42)
    int4_results = run_int4_sweep(seed=42, n_bits=4, verbose=True)

    # 3. Summary
    summary = summarize_postfix(pareto_results, int4_results)
    print(summary)

    # 4. Save results
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "coreset_postfix_pareto.json"), "w") as f:
        json.dump({"pareto": pareto_results}, f, indent=2)
    print(f"\nSaved: results/coreset_postfix_pareto.json ({len(pareto_results)} configs)")

    with open(os.path.join(output_dir, "coreset_postfix_int4.json"), "w") as f:
        json.dump({"int4_sweep": int4_results}, f, indent=2)
    print(f"Saved: results/coreset_postfix_int4.json ({len(int4_results)} configs)")

    return pareto_results, int4_results


if __name__ == "__main__":
    main()
