"""
Exp4: Coreset + INT1-8 极限压缩 Sweep
=====================================

核心 idea: Coreset+INT4 是 ACCORD 最 novel 的方向。
把极限推到 INT1-8 完整 sweep。

Pass criterion:
- INT3: 大多数 config error increase < 5%
- INT2: 大多数 config error increase < 15%
- INT1: error increase < 30%（可接受 fallback）
- Pareto front: 每个 (bytes, error) trade-off 的最优组合
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    ground_truth,
)


@dataclass
class CoresetSketch:
    centroids: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    assignments: np.ndarray

    def bytes_size(self) -> int:
        return (self.centroids.size + self.values.size + self.weights.size) * 4


@dataclass
class QuantizedSketch:
    centroids_int: np.ndarray
    values_int: np.ndarray
    weights: np.ndarray
    scales: np.ndarray   # [r, 2] K-scale and V-scale
    n_bits: int

    def bytes_size(self) -> int:
        bits_total = self.n_bits * (self.centroids_int.size + self.values_int.size)
        bytes_kv = math.ceil(bits_total / 8)
        return bytes_kv + self.weights.size * 4 + self.scales.size * 4


def kmeans_plusplus_init(K: np.ndarray, r: int, seed: int = 0) -> np.ndarray:
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
    K: np.ndarray, V: np.ndarray, r: int,
    block_size: int = 64, seed: int = 0,
) -> CoresetSketch:
    n, d = K.shape
    # 大 kv_len 减少迭代次数以加速
    num_iters = 10 if n > 8192 else 15
    gen = np.random.default_rng(seed)
    centroids = kmeans_plusplus_init(K, r, seed)

    for _ in range(num_iters):
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
        assignments = dists.argmin(axis=1)

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

        centroid_shift = np.sum((centroids - new_centroids) ** 2)
        centroids = new_centroids
        if centroid_shift < 1e-8:
            break

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


def eval_coreset_sketch(sketch: CoresetSketch, Q: np.ndarray, d: int) -> NumpyAttnStats:
    scores = Q @ sketch.centroids.T / math.sqrt(d)
    scores = scores + np.log(sketch.weights + 1e-12)
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ sketch.values
    return NumpyAttnStats(m=m[None, :, :], l=l[None, :, :], y=y[None, :, :])


def quantize_sketch_nbit(sketch: CoresetSketch, n_bits: int) -> QuantizedSketch:
    """Per-channel symmetric quantization to n_bits.
    
    K 和 V scale 分开（Bug 2 fix）。
    n_bits=1: bipolar {-1, +1} encoding
    n_bits>=2: symmetric INT with max_val = 2^(n-1) - 1
    """
    r, d = sketch.centroids.shape

    if n_bits == 1:
        c_scales = np.zeros((r, 2), dtype=np.float32)
        for j in range(r):
            c_scales[j, 0] = np.abs(sketch.centroids[j]).mean() + 1e-10
            c_scales[j, 1] = np.abs(sketch.values[j]).mean() + 1e-10

        centroids_int = np.zeros((r, d), dtype=np.int8)
        values_int = np.zeros((r, d), dtype=np.int8)
        for j in range(r):
            centroids_int[j] = np.sign(sketch.centroids[j]).astype(np.int8)
            values_int[j] = np.sign(sketch.values[j]).astype(np.int8)

        return QuantizedSketch(
            centroids_int=centroids_int,
            values_int=values_int,
            weights=sketch.weights.copy(),
            scales=c_scales,
            n_bits=1,
        )

    max_val = float(2 ** (n_bits - 1) - 1)
    if max_val < 1:
        max_val = 1

    c_scales = np.zeros((r, 2), dtype=np.float32)
    for j in range(r):
        k_max = np.abs(sketch.centroids[j]).max()
        v_max = np.abs(sketch.values[j]).max()
        c_scales[j, 0] = k_max / max_val if k_max > 1e-10 else 1e-10
        c_scales[j, 1] = v_max / max_val if v_max > 1e-10 else 1e-10

    centroids_int = np.zeros((r, d), dtype=np.int8)
    values_int = np.zeros((r, d), dtype=np.int8)

    for j in range(r):
        centroids_int[j] = np.clip(
            np.round(sketch.centroids[j] / c_scales[j, 0]),
            -max_val, max_val
        ).astype(np.int8)
        values_int[j] = np.clip(
            np.round(sketch.values[j] / c_scales[j, 1]),
            -max_val, max_val
        ).astype(np.int8)

    return QuantizedSketch(
        centroids_int=centroids_int,
        values_int=values_int,
        weights=sketch.weights.copy(),
        scales=c_scales,
        n_bits=n_bits,
    )


def dequantize_sketch_nbit(q_sketch: QuantizedSketch) -> CoresetSketch:
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


def make_clustered_kv(
    num_blocks: int, block_size: int, d: int,
    num_clusters: int, cluster_std: float = 0.5, seed: int = 0,
):
    kv_len = num_blocks * block_size
    gen = np.random.default_rng(seed)
    cluster_centers = gen.standard_normal((num_clusters, d)) * 2.0
    cluster_assign = gen.choice(num_clusters, kv_len).astype(np.int32)
    K_all = np.zeros((kv_len, d), dtype=np.float32)
    V_all = np.zeros((kv_len, d), dtype=np.float32)
    for i in range(kv_len):
        c = cluster_assign[i]
        K_all[i] = cluster_centers[c] + gen.standard_normal(d) * cluster_std
        V_all[i] = K_all[i] * 0.8 + gen.standard_normal(d) * 0.2
    return {}, K_all, V_all


def run_nbit_single(
    kv_len: int, block_size: int, sketch_r: int, q_len: int,
    n_bits: int, d: int = 128, seed: int = 0, verbose: bool = True,
) -> dict:
    num_blocks = kv_len // block_size
    _, K_all, V_all = make_clustered_kv(
        num_blocks, block_size, d,
        num_clusters=max(4, kv_len // 256), seed=seed
    )
    Q = (np.random.default_rng(seed + 1000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K_all, V_all)

    sketch_fp32 = build_coreset_sketch(K_all, V_all, sketch_r, block_size, seed)
    stats_fp32 = eval_coreset_sketch(sketch_fp32, Q, d)
    out_fp32 = stats_fp32.finalize().squeeze(0)
    err_fp32 = float(np.abs(out_fp32 - gt).mean())
    bytes_fp32 = sketch_fp32.bytes_size()

    q_sketch = quantize_sketch_nbit(sketch_fp32, n_bits=n_bits)
    sketch_deq = dequantize_sketch_nbit(q_sketch)
    stats_nbit = eval_coreset_sketch(sketch_deq, Q, d)
    out_nbit = stats_nbit.finalize().squeeze(0)
    err_nbit = float(np.abs(out_nbit - gt).mean())
    bytes_nbit = q_sketch.bytes_size()

    compression = bytes_fp32 / bytes_nbit
    err_inc_pct = (err_nbit - err_fp32) / (err_fp32 + 1e-10) * 100
    err_inc_abs = err_nbit - err_fp32
    bytes_per_centroid = bytes_nbit / sketch_r

    if verbose:
        print(
            f"  kv={kv_len:>5} bs={block_size:>3} r={sketch_r:>2} "
            f"nb={n_bits} fp32={err_fp32:.3e} int{n_bits}={err_nbit:.3e} "
            f"gain={compression:.1f}x inc={err_inc_pct:+.2f}%"
        )

    return {
        "kv_len": kv_len,
        "block_size": block_size,
        "sketch_r": sketch_r,
        "q_len": q_len,
        "n_bits": n_bits,
        "err_fp32": err_fp32,
        "err_nbit": err_nbit,
        "bytes_fp32": bytes_fp32,
        "bytes_nbit": bytes_nbit,
        "compression_gain": compression,
        "bytes_per_centroid": bytes_per_centroid,
        "error_increase_pct": err_inc_pct,
        "error_increase_abs": err_inc_abs,
    }


def run_nbit_sweep(seed: int = 42, verbose: bool = True) -> list:
    """完整 sweep。优化：大 kv_len 用更少迭代。"""
    results = []
    n_bits_list = [1, 2, 3, 4, 8]
    block_sizes = [32, 64, 128]
    kv_lens = [1024, 4096, 16384]
    sketch_rs = [4, 8, 16]  # 去掉 32（kv_len=16384 太慢）
    q_lens = [16, 64]
    d = 128

    if verbose:
        n_total = len(block_sizes) * len(kv_lens) * len(sketch_rs) * len(n_bits_list) * len(q_lens)
        print("=" * 78)
        print(f"INT1-8 Full Sweep (seed={seed}): {n_total} configs")
        print("=" * 78)

    total = len(block_sizes) * len(kv_lens) * len(sketch_rs) * len(n_bits_list) * len(q_lens)
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
                        try:
                            r = run_nbit_single(
                                kv_len, block_size, sketch_r, q_len,
                                n_bits, d, seed=seed, verbose=verbose
                            )
                            results.append(r)
                        except Exception as e:
                            if verbose:
                                print(f"  ERROR kv={kv_len} bs={block_size} r={sketch_r} nb={n_bits}: {e}")
                        if count % 50 == 0 and verbose:
                            print(f"  Progress: {count}/{total}")

    if verbose:
        print(f"\nCompleted {len(results)}/{total} configs")
    return results


def analyze_nbit_sweep(results: list) -> dict:
    n_bits_list = sorted(set(r["n_bits"] for r in results))

    nbit_stats = {}
    for nb in n_bits_list:
        sub = [r for r in results if r["n_bits"] == nb]
        err_incs = [r["error_increase_pct"] for r in sub]
        gains = [r["compression_gain"] for r in sub]
        err_abs = [r["error_increase_abs"] for r in sub]

        pass_5pct = sum(1 for e in err_incs if e < 5) / len(err_incs)
        pass_15pct = sum(1 for e in err_incs if e < 15) / len(err_incs)
        pass_30pct = sum(1 for e in err_incs if e < 30) / len(err_incs)

        nbit_stats[f"INT{nb}"] = {
            "count": len(sub),
            "avg_err_inc_pct": round(sum(err_incs) / len(err_incs), 3),
            "max_err_inc_pct": round(max(err_incs), 2),
            "min_err_inc_pct": round(min(err_incs), 2),
            "std_err_inc_pct": round(np.std(err_incs), 2),
            "avg_compression_gain": round(sum(gains) / len(gains), 2),
            "avg_err_abs": round(sum(err_abs) / len(err_abs), 6),
            "pass_5pct": round(pass_5pct, 3),
            "pass_15pct": round(pass_15pct, 3),
            "pass_30pct": round(pass_30pct, 3),
        }

    # Pareto front
    pareto_front = {}
    for kv_len in sorted(set(r["kv_len"] for r in results)):
        sub = [r for r in results if r["kv_len"] == kv_len]
        seen = {}
        for r in sub:
            key = (r["sketch_r"], r["n_bits"])
            if key not in seen or r["err_nbit"] < seen[key]["err_nbit"]:
                seen[key] = r
        candidates = list(seen.values())
        pareto = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if (other["bytes_nbit"] <= c["bytes_nbit"] and
                    other["err_nbit"] <= c["err_nbit"] and
                    (other["bytes_nbit"] < c["bytes_nbit"] or other["err_nbit"] < c["err_nbit"])):
                    dominated = True
                    break
            if not dominated:
                pareto.append({
                    "sketch_r": c["sketch_r"],
                    "n_bits": c["n_bits"],
                    "bytes_nbit": c["bytes_nbit"],
                    "err_nbit": c["err_nbit"],
                    "compression_gain": c["compression_gain"],
                })
        pareto_front[kv_len] = sorted(pareto, key=lambda x: x["bytes_nbit"])

    # Sweet spots
    sweet_spots = {}
    for kv_len in sorted(set(r["kv_len"] for r in results)):
        sub = [r for r in results if r["kv_len"] == kv_len and r["error_increase_pct"] < 15]
        if sub:
            best = max(sub, key=lambda x: x["compression_gain"])
            sweet_spots[kv_len] = {
                "sketch_r": int(best["sketch_r"]),
                "n_bits": int(best["n_bits"]),
                "compression_gain": round(best["compression_gain"], 2),
                "error_inc_pct": round(best["error_increase_pct"], 2),
            }

    return {
        "nbit_stats": nbit_stats,
        "pareto_front": {str(k): v for k, v in pareto_front.items()},
        "sweet_spots": sweet_spots,
    }


def main():
    print("ACCORD-KV: Coreset + INT1-8 Full Sweep")
    print("=" * 78)
    results = run_nbit_sweep(seed=42, verbose=True)
    analysis = analyze_nbit_sweep(results)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    stats = analysis["nbit_stats"]
    print(f"\n{'n_bits':>6} {'avg_inc%':>10} {'max_inc%':>10} {'pass<5%':>9} {'pass<15%':>9} {'gain':>7}")
    for nb in [1, 2, 3, 4, 8]:
        key = f"INT{nb}"
        if key in stats:
            s = stats[key]
            print(
                f"{key:>6} {s['avg_err_inc_pct']:>+10.2f} {s['max_err_inc_pct']:>+10.2f} "
                f"{s['pass_5pct']:>9.1%} {s['pass_15pct']:>9.1%} {s['avg_compression_gain']:>7.1f}x"
            )

    print("\n--- Sweet Spots (compression gain max, error < 15%) ---")
    for kv_len, spot in analysis["sweet_spots"].items():
        print(
            f"  kv_len={kv_len:>5}: r={spot['sketch_r']:>2} n_bits={spot['n_bits']} "
            f"gain={spot['compression_gain']:.1f}x err_inc={spot['error_inc_pct']:+.2f}%"
        )

    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "exp4_coreset_nbit_sweep.json"), "w") as f:
        json.dump({
            "experiment": "Coreset_INT1_8_Sweep",
            "total_configs": len(results),
            "results": results,
            "analysis": analysis,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: results/exp4_coreset_nbit_sweep.json ({len(results)} configs)")
    return results, analysis


if __name__ == "__main__":
    main()
