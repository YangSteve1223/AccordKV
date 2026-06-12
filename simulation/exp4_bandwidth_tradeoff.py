"""
Exp4: Compression × Bandwidth Trade-off
========================================

核心 idea: 不同 bandwidth 下最优压缩比不同。

- 低带宽 (1 Gbps): 应该用更激进压缩 (INT2, adaptive)
- 高带宽 (100 Gbps): FP32 都可以

跑 sweep 看 trade-off，输出 60 组数据。
"""

from __future__ import annotations

import json
import math
import os
import sys

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


def compute_e2e_latency(
    bytes_total: int, kv_len: int, bandwidth_gbps: float,
    rtt_ms: float, compute_cost_us: float = 50.0,
) -> float:
    """E2E latency = transfer + compute + network."""
    T_transfer = (bytes_total * 8) / (bandwidth_gbps * 1e9) * 1000
    T_compute = kv_len * compute_cost_us / 1000 * 0.001
    return T_transfer + T_compute + rtt_ms / 2


def run_bw_tradeoff_single(
    kv_len: int, block_size: int, sketch_r: int, q_len: int,
    n_bits: int, bandwidth_gbps: float, d: int = 128,
    seed: int = 0, verbose: bool = True,
) -> dict:
    num_blocks = kv_len // block_size
    _, K_all, V_all = make_clustered_kv(
        num_blocks, block_size, d,
        num_clusters=max(4, kv_len // 256), seed=seed
    )
    Q = (np.random.default_rng(seed + 1000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K_all, V_all)
    
    # FP32 sketch
    sketch_fp32 = build_coreset_sketch(K_all, V_all, sketch_r, block_size, seed)
    stats_fp32 = eval_coreset_sketch(sketch_fp32, Q, d)
    out_fp32 = stats_fp32.finalize().squeeze(0)
    err_fp32 = float(np.abs(out_fp32 - gt).mean())
    bytes_fp32 = sketch_fp32.bytes_size()
    
    # N-bit
    q_nbit = quantize_sketch_nbit(sketch_fp32, n_bits=n_bits)
    deq_nbit = dequantize_sketch_nbit(q_nbit)
    stats_nbit = eval_coreset_sketch(deq_nbit, Q, d)
    out_nbit = stats_nbit.finalize().squeeze(0)
    err_nbit = float(np.abs(out_nbit - gt).mean())
    bytes_nbit = q_nbit.bytes_size()
    
    # Latency
    lat_fp32 = compute_e2e_latency(bytes_fp32, kv_len, bandwidth_gbps, 5.0)
    lat_nbit = compute_e2e_latency(bytes_nbit, kv_len, bandwidth_gbps, 5.0)
    
    err_inc = (err_nbit - err_fp32) / (err_fp32 + 1e-10) * 100
    latency_reduction = (lat_fp32 - lat_nbit) / (lat_fp32 + 1e-10) * 100
    
    return {
        "kv_len": kv_len,
        "sketch_r": sketch_r,
        "n_bits": n_bits,
        "bandwidth_gbps": bandwidth_gbps,
        "bytes_nbit": bytes_nbit,
        "bytes_fp32": bytes_fp32,
        "err_fp32": err_fp32,
        "err_nbit": err_nbit,
        "err_inc_pct": err_inc,
        "lat_fp32_ms": round(lat_fp32, 6),
        "lat_nbit_ms": round(lat_nbit, 6),
        "latency_reduction_pct": round(latency_reduction, 2),
    }


def run_bw_tradeoff_sweep(seed: int = 42, verbose: bool = True) -> list:
    """60 组 sweep。"""
    results = []
    
    kv_lens = [1024, 4096, 16384]
    sketch_rs = [4, 8, 16]
    n_bits_list = [1, 2, 3, 4]
    bandwidths = [1.0, 10.0, 100.0, 1000.0]
    block_size = 64
    q_len = 64
    
    if verbose:
        print("=" * 78)
        print(f"BW Trade-off Sweep (seed={seed}): ~{len(kv_lens)*len(sketch_rs)*len(n_bits_list)*len(bandwidths)} configs")
        print("=" * 78)
    
    count = 0
    for kv_len in kv_lens:
        for sketch_r in sketch_rs:
            if sketch_r >= kv_len // block_size:
                continue
            for n_bits in n_bits_list:
                for bw in bandwidths:
                    count += 1
                    r = run_bw_tradeoff_single(
                        kv_len, block_size, sketch_r, q_len,
                        n_bits, bw, seed=seed, verbose=False
                    )
                    results.append(r)
                    if count % 20 == 0 and verbose:
                        print(f"  Progress: {count}")
    
    if verbose:
        print(f"\nCompleted {len(results)} configs")
    return results


def analyze_bw_tradeoff(results: list) -> dict:
    # 按 bandwidth 分组
    bw_stats = {}
    for bw in sorted(set(r["bandwidth_gbps"] for r in results)):
        sub = [x for x in results if x["bandwidth_gbps"] == bw]
        avg_lat_red = sum(x["latency_reduction_pct"] for x in sub) / len(sub)
        avg_err_inc = sum(x["err_inc_pct"] for x in sub) / len(sub)
        bw_stats[str(bw)] = {
            "count": len(sub),
            "avg_latency_reduction_pct": round(avg_lat_red, 2),
            "avg_err_inc_pct": round(avg_err_inc, 2),
        }
    
    # 最优策略推荐
    optimal_by_bw = {}
    for bw in sorted(set(r["bandwidth_gbps"] for r in results)):
        sub = [x for x in results if x["bandwidth_gbps"] == bw]
        # 在 error < 15% 约束下找最大 latency reduction
        feasible = [x for x in sub if x["err_inc_pct"] < 15]
        if feasible:
            best = max(feasible, key=lambda x: x["latency_reduction_pct"])
            optimal_by_bw[str(bw)] = {
                "sketch_r": int(best["sketch_r"]),
                "n_bits": int(best["n_bits"]),
                "latency_reduction": best["latency_reduction_pct"],
                "err_inc": best["err_inc_pct"],
            }
    
    return {
        "total": len(results),
        "by_bandwidth": bw_stats,
        "optimal_by_bandwidth": optimal_by_bw,
    }


def main():
    print("ACCORD-KV: Compression × Bandwidth Trade-off")
    print("=" * 78)
    
    results = run_bw_tradeoff_sweep(seed=42, verbose=True)
    analysis = analyze_bw_tradeoff(results)
    
    print("\n" + "=" * 78)
    print("SUMMARY: BW Trade-off")
    print("=" * 78)
    
    print("\n--- By Bandwidth ---")
    print(f"{'BW_Gbps':>8} {'configs':>7} {'avg_lat_red%':>12} {'avg_err_inc%':>12}")
    for bw, s in sorted(analysis["by_bandwidth"].items(), key=lambda x: float(x[0])):
        print(
            f"{bw:>8} {s['count']:>7} "
            f"{s['avg_latency_reduction_pct']:>+12.2f} {s['avg_err_inc_pct']:>+12.2f}%"
        )
    
    print("\n--- Optimal Strategy per BW (error < 15%) ---")
    for bw, opt in sorted(analysis["optimal_by_bandwidth"].items(), key=lambda x: float(x[0])):
        print(
            f"  {bw:>6} Gbps: r={opt['sketch_r']:>2} n_bits={opt['n_bits']}  "
            f"lat_red={opt['latency_reduction']:+.1f}% err_inc={opt['err_inc']:+.1f}%"
        )
    
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "exp4_bandwidth_tradeoff.json"), "w") as f:
        json.dump({
            "experiment": "BW_Tradeoff",
            "results": results,
            "analysis": analysis,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: results/exp4_bandwidth_tradeoff.json ({len(results)} configs)")
    return results, analysis


if __name__ == "__main__":
    main()
