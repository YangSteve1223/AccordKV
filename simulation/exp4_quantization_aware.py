"""
Exp4: Quantization-Aware k-means (QA-kmeans)
============================================

核心 idea: 训练 k-means 时直接用 INT4 重建 loss，
让 centroids 天然 friendly 量化。

实现：
- k-means 收敛后，加 1-2 轮 QA refinement
- 每轮：quantize → dequantize → 用 quantized centroids 重新做 assignment + update
- 输出最终 centroids (量化友好)

对比：standard k-means + INT4 vs QA-kmeans + INT4
预期：QA-kmeans 比 standard + INT4 error 降低 10-20%
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
    make_clustered_kv, kmeans_plusplus_init,
)


def build_qa_kmeans_sketch(
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    block_size: int = 64,
    seed: int = 0,
    num_iters: int = 15,
    qa_rounds: int = 2,
    n_bits: int = 4,
) -> Tuple[CoresetSketch, QuantizedSketch]:
    """Quantization-aware k-means。
    
    流程：
    1. Standard k-means 收敛（15 轮）
    2. QA refinement（2 轮）：
       - quantize → dequantize → 用 quantized centroids 做 assignment + update
    3. 最终 quantize 输出
    """
    n, d = K.shape
    gen = np.random.default_rng(seed)
    
    # Step 1: Standard k-means++
    centroids = kmeans_plusplus_init(K, r, seed)
    
    for _ in range(num_iters):
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
        assignments = dists.argmin(axis=1)
        
        new_centroids = np.zeros_like(centroids)
        new_values = np.zeros((r, d))
        for j in range(r):
            mask = assignments == j
            if mask.sum() > 0:
                new_centroids[j] = K[mask].mean(axis=0)
                new_values[j] = V[mask].mean(axis=0)
            else:
                new_centroids[j] = centroids[j]
        
        centroid_shift = np.sum((centroids - new_centroids) ** 2)
        centroids = new_centroids
        if centroid_shift < 1e-8:
            break
    
    # 保存 standard centroids
    standard_centroids = centroids.copy()
    standard_values = new_values.copy()
    
    # Step 2: QA refinement
    for qa_round in range(qa_rounds):
        # 2a: quantize current centroids
        max_val = float(2 ** (n_bits - 1) - 1)
        scales = np.zeros((r, 2), dtype=np.float32)
        for j in range(r):
            k_max = np.abs(centroids[j]).max()
            v_max = np.abs(new_values[j]).max()
            scales[j, 0] = k_max / max_val if k_max > 1e-10 else 1e-10
            scales[j, 1] = v_max / max_val if v_max > 1e-10 else 1e-10
        
        c_int = np.zeros((r, d), dtype=np.int8)
        v_int = np.zeros((r, d), dtype=np.int8)
        for j in range(r):
            c_int[j] = np.clip(
                np.round(centroids[j] / scales[j, 0]),
                -max_val, max_val
            ).astype(np.int8)
            v_int[j] = np.clip(
                np.round(new_values[j] / scales[j, 1]),
                -max_val, max_val
            ).astype(np.int8)
        
        # 2b: dequantize
        q_centroids = np.zeros_like(centroids)
        q_values = np.zeros_like(new_values)
        for j in range(r):
            q_centroids[j] = c_int[j].astype(np.float32) * scales[j, 0]
            q_values[j] = v_int[j].astype(np.float32) * scales[j, 1]
        
        # 2c: 用 quantized centroids 做 assignment
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K - q_centroids[j]) ** 2, axis=1)
        assignments = dists.argmin(axis=1)
        
        # 2d: 更新（基于原始 K/V，用 quantized centroids 做空间划分）
        new_centroids = np.zeros_like(centroids)
        new_values = np.zeros((r, d))
        for j in range(r):
            mask = assignments == j
            if mask.sum() > 0:
                new_centroids[j] = K[mask].mean(axis=0)
                new_values[j] = V[mask].mean(axis=0)
            else:
                new_centroids[j] = q_centroids[j]
        
        # 用 refine 后的 centroids 继续
        centroids = new_centroids
    
    # 最终 weights
    dists = np.zeros((n, r))
    for j in range(r):
        dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
    final_assignments = dists.argmin(axis=1)
    final_weights = np.zeros(r)
    for j in range(r):
        mask = final_assignments == j
        final_weights[j] = mask.sum() / n
    
    sketch = CoresetSketch(
        centroids=centroids,
        values=new_values,
        weights=final_weights,
        assignments=final_assignments,
    )
    
    # 最终量化
    q_sketch = quantize_sketch_nbit(sketch, n_bits=n_bits)
    
    return sketch, q_sketch


def run_qa_single(
    kv_len: int, block_size: int, sketch_r: int, q_len: int,
    n_bits: int = 4, d: int = 128, seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Standard vs QA k-means + INT4 对比。"""
    num_blocks = kv_len // block_size
    _, K_all, V_all = make_clustered_kv(
        num_blocks, block_size, d,
        num_clusters=max(4, kv_len // 256), seed=seed
    )
    Q = (np.random.default_rng(seed + 1000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K_all, V_all)
    
    # Standard k-means + INT4
    sketch_std = build_coreset_sketch(K_all, V_all, sketch_r, block_size, seed)
    q_std = quantize_sketch_nbit(sketch_std, n_bits=n_bits)
    deq_std = dequantize_sketch_nbit(q_std)
    out_std = eval_coreset_sketch(deq_std, Q, d).finalize().squeeze(0)
    err_std = float(np.abs(out_std - gt).mean())
    
    # QA k-means + INT4
    sketch_qa, q_qa = build_qa_kmeans_sketch(
        K_all, V_all, sketch_r, block_size, seed,
        qa_rounds=2, n_bits=n_bits
    )
    deq_qa = dequantize_sketch_nbit(q_qa)
    out_qa = eval_coreset_sketch(deq_qa, Q, d).finalize().squeeze(0)
    err_qa = float(np.abs(out_qa - gt).mean())
    
    improvement = (err_std - err_qa) / (err_std + 1e-10) * 100
    bytes_std = q_std.bytes_size()
    bytes_qa = q_qa.bytes_size()
    
    if verbose:
        print(
            f"  kv={kv_len:>5} r={sketch_r:>2} nb={n_bits}  "
            f"std={err_std:.3e} qa={err_qa:.3e}  "
            f"improve={improvement:+.1f}%  "
            f"bytes_std={bytes_std} qa={bytes_qa}"
        )
    
    return {
        "kv_len": kv_len,
        "block_size": block_size,
        "sketch_r": sketch_r,
        "q_len": q_len,
        "n_bits": n_bits,
        "err_standard": err_std,
        "err_qa": err_qa,
        "improvement_pct": improvement,
        "bytes_standard": bytes_std,
        "bytes_qa": bytes_qa,
    }


def run_qa_sweep(seed: int = 42, verbose: bool = True) -> list:
    """30 组 sweep。"""
    results = []
    
    block_sizes = [32, 64, 128]
    kv_lens = [1024, 4096, 16384]
    sketch_rs = [4, 8, 16]
    q_lens = [16, 64]
    n_bits_list = [2, 4]
    
    if verbose:
        print("=" * 78)
        print(f"QA k-means Sweep (seed={seed}): ~30 configs")
        print("=" * 78)
    
    count = 0
    for block_size in block_sizes:
        for kv_len in kv_lens:
            if kv_len % block_size != 0:
                continue
            for sketch_r in sketch_rs:
                if sketch_r >= kv_len // block_size:
                    continue
                for n_bits in n_bits_list:
                    for q_len in q_lens:
                        count += 1
                        if count > 30:
                            break
                        try:
                            r = run_qa_single(
                                kv_len, block_size, sketch_r, q_len,
                                n_bits, seed=seed, verbose=verbose
                            )
                            results.append(r)
                        except Exception as e:
                            if verbose:
                                print(f"  ERROR: {e}")
                        if count > 30:
                            break
                    if count > 30:
                        break
    return results


def analyze_qa(results: list) -> dict:
    improvements = [r["improvement_pct"] for r in results]
    wins = sum(1 for i in improvements if i > 0)
    
    by_nb = {}
    for nb in sorted(set(r["n_bits"] for r in results)):
        sub = [x for x in results if x["n_bits"] == nb]
        by_nb[nb] = {
            "count": len(sub),
            "avg_improve": round(sum(x["improvement_pct"] for x in sub) / len(sub), 2),
            "win_rate": round(wins / len(results), 2) if sub else 0,
        }
    
    return {
        "total": len(results),
        "avg_improvement_pct": round(sum(improvements) / len(improvements), 2),
        "win_rate": round(wins / len(results), 2),
        "by_nbits": by_nb,
    }


def main():
    print("ACCORD-KV: Quantization-Aware k-means")
    print("=" * 78)
    
    results = run_qa_sweep(seed=42, verbose=True)
    analysis = analyze_qa(results)
    
    print("\n" + "=" * 78)
    print("SUMMARY: QA k-means vs Standard k-means")
    print("=" * 78)
    print(f"\nTotal configs: {analysis['total']}")
    print(f"Win rate: {analysis['win_rate']:.1%}")
    print(f"Average improvement: {analysis['avg_improvement_pct']:+.2f}%")
    
    print(f"\n--- By n_bits ---")
    for nb, s in analysis["by_nbits"].items():
        print(f"  INT{nb}: count={s['count']} avg_improve={s['avg_improve']:+.2f}%")
    
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "exp4_qa_kmeans.json"), "w") as f:
        json.dump({
            "experiment": "QA_kmeans",
            "results": results,
            "analysis": analysis,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: results/exp4_qa_kmeans.json ({len(results)} configs)")
    return results, analysis


if __name__ == "__main__":
    main()
