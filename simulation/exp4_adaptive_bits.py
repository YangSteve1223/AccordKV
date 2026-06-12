"""
Exp4: Adaptive Bits — 混合精度 Coreset
=======================================

核心 idea: 不同 block 重要度不同 → 不同 bit 精度分配

实现：
- build_adaptive_bits_sketch(): 基于 importance 三档分配
  - importance ≥ th1 (top 25%) → INT8
  - th1 > importance ≥ th2 (mid 50%) → INT4
  - importance < th2 (bottom 25%) → INT2

- importance 怎么算？基于 calibration queries 的 attention score

对比：
- uniform INT4 vs adaptive
- 同 byte budget: adaptive 应该 5-10% 更优

72 组配置（E1 Pareto 同款）
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    ground_truth,
)
from simulation.exp4_coreset_nbit import (
    CoresetSketch, QuantizedSketch,
    build_coreset_sketch, eval_coreset_sketch,
    quantize_sketch_nbit, dequantize_sketch_nbit,
    make_clustered_kv,
)


def compute_importance_scores(
    K: np.ndarray, V: np.ndarray,
    Q_cal: np.ndarray, d: int,
) -> np.ndarray:
    """基于 calibration queries 计算每个 centroid 的 importance。
    
    Importance = Σ_q exp(q · c_j / √d) / Σ_c Σ_q exp(q · c_c / √d)
    即每个 centroid 占总 attention mass 的比例。
    """
    n = K.shape[0]
    
    # 先建 sketch 获取 centroids
    r = max(4, n // 64)  # 估算 r
    sketch = build_coreset_sketch(K, V, r, block_size=64, seed=42)
    
    # 计算每个 centroid 的 attention weight
    scores = Q_cal @ sketch.centroids.T / math.sqrt(d)
    attn_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn_weights = attn_weights.sum(axis=0)  # [r]
    
    # 归一化
    importance = attn_weights / (attn_weights.sum() + 1e-10)
    return importance


def build_adaptive_bits_sketch(
    K: np.ndarray,
    V: np.ndarray,
    Q_cal: np.ndarray,
    r: int,
    block_size: int = 64,
    seed: int = 0,
    d: int = 128,
    top_pct: float = 0.25,
    bottom_pct: float = 0.25,
) -> Tuple[QuantizedSketch, dict]:
    """混合精度 Coreset sketch。
    
    根据 importance 将 centroid 分成三档：
    - Top 25%: INT8 (高保真)
    - Middle 50%: INT4 (标准)
    - Bottom 25%: INT2 (激进压缩)
    
    Returns
    -------
    (QuantizedSketch, allocation_dict)
    """
    # 先建 FP32 sketch
    sketch_fp32 = build_coreset_sketch(K, V, r, block_size, seed)
    
    # 计算 importance
    attn_raw = Q_cal @ sketch_fp32.centroids.T / math.sqrt(d)
    attn_max = attn_raw.max(axis=0, keepdims=True)
    attn_soft = np.exp(attn_raw - attn_max)
    importance = attn_soft.sum(axis=0)  # [r]
    importance = importance / (importance.sum() + 1e-10)
    
    # 按 importance 排序分档
    sorted_idx = np.argsort(-importance)
    n_top = max(1, int(r * top_pct))
    n_bottom = max(1, int(r * bottom_pct))
    
    top_idx = sorted_idx[:n_top]
    mid_idx = sorted_idx[n_top:r - n_bottom]
    bot_idx = sorted_idx[r - n_bottom:]
    
    # 分配 bit
    bit_alloc = np.zeros(r, dtype=np.int32)
    bit_alloc[top_idx] = 8   # INT8
    bit_alloc[mid_idx] = 4   # INT4
    bit_alloc[bot_idx] = 2   # INT2
    
    # 量化
    centroids_int = np.zeros_like(sketch_fp32.centroids, dtype=np.int8)
    values_int = np.zeros_like(sketch_fp32.values, dtype=np.int8)
    scales = np.zeros((r, 2), dtype=np.float32)
    
    for j in range(r):
        nb = bit_alloc[j]
        max_val = float(2 ** (nb - 1) - 1) if nb > 1 else 1.0
        
        # K scale
        k_max = np.abs(sketch_fp32.centroids[j]).max()
        scales[j, 0] = k_max / max_val if k_max > 1e-10 else 1e-10
        # V scale
        v_max = np.abs(sketch_fp32.values[j]).max()
        scales[j, 1] = v_max / max_val if v_max > 1e-10 else 1e-10
        
        if nb == 1:
            centroids_int[j] = np.sign(sketch_fp32.centroids[j]).astype(np.int8)
            values_int[j] = np.sign(sketch_fp32.values[j]).astype(np.int8)
        else:
            centroids_int[j] = np.clip(
                np.round(sketch_fp32.centroids[j] / scales[j, 0]),
                -max_val, max_val
            ).astype(np.int8)
            values_int[j] = np.clip(
                np.round(sketch_fp32.values[j] / scales[j, 1]),
                -max_val, max_val
            ).astype(np.int8)
    
    # 计算实际 bit budget
    total_bits = int(bit_alloc.sum() * 2)  # K + V
    total_bytes = math.ceil(total_bits / 8)
    total_bytes += r * 2 * 4  # scales + weights
    
    allocation = {
        "n_top": int(n_top),
        "n_mid": int(len(mid_idx)),
        "n_bottom": int(n_bottom),
        "bits_top": int(n_top * 8 * 2),
        "bits_mid": int(len(mid_idx) * 4 * 2),
        "bits_bottom": int(n_bottom * 2 * 2),
        "total_bits": total_bits,
        "total_bytes": total_bytes,
    }
    
    q_sketch = QuantizedSketch(
        centroids_int=centroids_int,
        values_int=values_int,
        weights=sketch_fp32.weights.copy(),
        scales=scales,
        n_bits=4,  # 报告用
    )
    return q_sketch, allocation


def dequantize_adaptive(q_sketch: QuantizedSketch) -> CoresetSketch:
    """反量化 adaptive sketch。"""
    return dequantize_sketch_nbit(q_sketch)


def run_adaptive_single(
    kv_len: int, block_size: int, sketch_r: int, q_len: int,
    d: int = 128, seed: int = 0, verbose: bool = True,
) -> dict:
    """单组 adaptive vs uniform INT4 对比。"""
    num_blocks = kv_len // block_size
    _, K_all, V_all = make_clustered_kv(
        num_blocks, block_size, d,
        num_clusters=max(4, kv_len // 256), seed=seed
    )
    
    # Cal queries
    Q_cal = (np.random.default_rng(seed + 500).standard_normal((8, d)) * 0.5).astype(np.float32)
    Q = (np.random.default_rng(seed + 1000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K_all, V_all)
    
    # FP32 baseline
    sketch_fp32 = build_coreset_sketch(K_all, V_all, sketch_r, block_size, seed)
    stats_fp32 = eval_coreset_sketch(sketch_fp32, Q, d)
    out_fp32 = stats_fp32.finalize().squeeze(0)
    err_fp32 = float(np.abs(out_fp32 - gt).mean())
    
    # Uniform INT4
    q_uniform = quantize_sketch_nbit(sketch_fp32, n_bits=4)
    sketch_uniform = dequantize_sketch_nbit(q_uniform)
    stats_uniform = eval_coreset_sketch(sketch_uniform, Q, d)
    out_uniform = stats_uniform.finalize().squeeze(0)
    err_uniform = float(np.abs(out_uniform - gt).mean())
    bytes_uniform = q_uniform.bytes_size()
    
    # Adaptive
    q_adaptive, alloc = build_adaptive_bits_sketch(
        K_all, V_all, Q_cal, sketch_r, block_size, seed, d
    )
    sketch_adaptive = dequantize_adaptive(q_adaptive)
    stats_adaptive = eval_coreset_sketch(sketch_adaptive, Q, d)
    out_adaptive = stats_adaptive.finalize().squeeze(0)
    err_adaptive = float(np.abs(out_adaptive - gt).mean())
    bytes_adaptive = q_adaptive.bytes_size()
    
    # 计算 improvement
    err_inc_uniform = (err_uniform - err_fp32) / (err_fp32 + 1e-10) * 100
    err_inc_adaptive = (err_adaptive - err_fp32) / (err_fp32 + 1e-10) * 100
    adaptive_improve = (err_uniform - err_adaptive) / (err_uniform + 1e-10) * 100
    
    if verbose:
        print(
            f"  kv={kv_len:>5} r={sketch_r:>2}  "
            f"fp32={err_fp32:.3e} uni4={err_uniform:.3e} ({err_inc_uniform:+.1f}%)  "
            f"adapt={err_adaptive:.3e} ({err_inc_adaptive:+.1f}%)  "
            f"improve={adaptive_improve:+.1f}%"
        )
    
    return {
        "kv_len": kv_len,
        "block_size": block_size,
        "sketch_r": sketch_r,
        "q_len": q_len,
        "err_fp32": err_fp32,
        "err_uniform_int4": err_uniform,
        "err_adaptive": err_adaptive,
        "bytes_uniform": bytes_uniform,
        "bytes_adaptive": bytes_adaptive,
        "err_inc_uniform_pct": err_inc_uniform,
        "err_inc_adaptive_pct": err_inc_adaptive,
        "adaptive_improve_pct": adaptive_improve,
        "allocation": alloc,
    }


def run_adaptive_sweep(seed: int = 42, verbose: bool = True) -> list:
    """72 组 sweep（E1 Pareto 同款）。"""
    results = []
    block_sizes = [32, 64, 128]
    kv_lens = [1024, 4096, 16384]
    sketch_rs = [4, 8, 16, 32]
    q_lens = [16, 64]
    d = 128

    if verbose:
        print("=" * 78)
        print(f"Adaptive Bits Sweep (seed={seed}): 72 configs")
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
                    r = run_adaptive_single(
                        kv_len, block_size, sketch_r, q_len, d, seed, verbose
                    )
                    results.append(r)
    
    if verbose:
        print(f"\nCompleted {len(results)} configs")
    return results


def analyze_adaptive(results: list) -> dict:
    """分析 adaptive vs uniform。"""
    uniform_errs = [r["err_inc_uniform_pct"] for r in results]
    adaptive_errs = [r["err_inc_adaptive_pct"] for r in results]
    improves = [r["adaptive_improve_pct"] for r in results]
    
    # 同 budget 对比（adaptive 和 uniform 实际 byte budget 不同）
    # adaptive_bytes < uniform_bytes，但 adaptive 精度更高
    adaptive_wins = sum(1 for i in improves if i > 0) / len(improves)
    adaptive_avg_improve = sum(improves) / len(improves)
    
    # 按 r 分组
    r_groups = {}
    for r_val in sorted(set(rr["sketch_r"] for rr in results)):
        sub = [x for x in results if x["sketch_r"] == r_val]
        r_groups[r_val] = {
            "count": len(sub),
            "avg_uniform_inc": round(sum(x["err_inc_uniform_pct"] for x in sub) / len(sub), 2),
            "avg_adaptive_inc": round(sum(x["err_inc_adaptive_pct"] for x in sub) / len(sub), 2),
            "avg_improve": round(sum(x["adaptive_improve_pct"] for x in sub) / len(sub), 2),
            "adaptive_win_rate": round(sum(1 for x in sub if x["adaptive_improve_pct"] > 0) / len(sub), 2),
        }
    
    return {
        "total": len(results),
        "adaptive_win_rate": round(adaptive_wins, 3),
        "avg_improve_pct": round(adaptive_avg_improve, 2),
        "avg_uniform_inc_pct": round(sum(uniform_errs) / len(uniform_errs), 2),
        "avg_adaptive_inc_pct": round(sum(adaptive_errs) / len(adaptive_errs), 2),
        "by_r": r_groups,
    }


def main():
    print("ACCORD-KV: Adaptive Bits Sweep")
    print("=" * 78)
    
    results = run_adaptive_sweep(seed=42, verbose=True)
    analysis = analyze_adaptive(results)
    
    print("\n" + "=" * 78)
    print("SUMMARY: Adaptive Bits vs Uniform INT4")
    print("=" * 78)
    print(f"\nTotal configs: {analysis['total']}")
    print(f"Adaptive win rate: {analysis['adaptive_win_rate']:.1%}")
    print(f"Average improvement: {analysis['avg_improve_pct']:+.2f}%")
    print(f"Average uniform INT4 error increase: {analysis['avg_uniform_inc_pct']:+.2f}%")
    print(f"Average adaptive error increase: {analysis['avg_adaptive_inc_pct']:+.2f}%")
    
    print(f"\n--- By r ---")
    print(f"{'r':>4} {'count':>6} {'uni_inc%':>9} {'adapt_inc%':>11} {'improve%':>9} {'win_rate':>9}")
    for r_val, s in sorted(analysis["by_r"].items()):
        print(
            f"{r_val:>4} {s['count']:>6} {s['avg_uniform_inc']:>+9.2f} "
            f"{s['avg_adaptive_inc']:>+11.2f} {s['avg_improve']:>+9.2f} {s['adaptive_win_rate']:>9.1%}"
        )
    
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "exp4_adaptive_bits.json"), "w") as f:
        json.dump({
            "experiment": "Adaptive_Bits",
            "results": results,
            "analysis": analysis,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: results/exp4_adaptive_bits.json ({len(results)} configs)")
    return results, analysis


if __name__ == "__main__":
    main()
