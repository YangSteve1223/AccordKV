"""
Exp4 E2E: Coreset+INT4 接到 E5 PD Network Sim
=============================================

核心 idea: 把 Coreset+INT4 接到 E5 PD network sim 看真实提升。

修改 SKETCH_LOCAL contract 实现：
- 当前：传 sketch (centroids FP32 + weights FP32) — 大
- 改后：传 coreset+INT4 (centroids INT4 + weights INT4 + scale FP32) — 小 7.3x

Sweep:
- 4 strategies: ACCORD_FP32 / ACCORD_INT4 / ACCORD_INT2 / FULL_KV
- 4 bandwidths: 1/10/100/1000 Gbps
- 4 RTTs: 0.1/1/10/50 ms
- 总 configs = 4 × 4 × 4 = 64 组

Pass criterion:
- ACCORD_INT4 跟 ACCORD_FP32 误差差 < 5%
- ACCORD_INT4 比 FULL_KV TTFT 优势 > 4x
- ACCORD_INT2 误差 < 15%, TTFT 优势 > 5x
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    ground_truth,
)
from simulation.exp4_coreset_nbit import (
    CoresetSketch,
    QuantizedSketch,
    build_coreset_sketch,
    eval_coreset_sketch,
    quantize_sketch_nbit,
    dequantize_sketch_nbit,
    make_clustered_kv,
)


# ============== PD Simulation 核心 ==============

class PDContract:
    """PD (Prefill-Decode) contract 基类。"""
    def __init__(self, name: str, sketch_or_kv, n_bits: int = 32):
        self.name = name
        self.sketch_or_kv = sketch_or_kv
        self.n_bits = n_bits
    
    def bytes_on_wire(self) -> int:
        raise NotImplementedError
    
    def ttft_ms(
        self, kv_len: int, bandwidth_gbps: float,
        rtt_ms: float, q_len: int, d: int,
        num_shards: int = 4,
    ) -> float:
        """Time to First Token (ms)。"""
        raise NotImplementedError
    
    def eval_attention(self, Q: np.ndarray, d: int, gt: Optional[np.ndarray] = None) -> dict:
        """评估 attention fidelity。"""
        raise NotImplementedError


class FullKVContract(PDContract):
    """FULL_KV: 传完整 KV。"""
    def __init__(self, K: np.ndarray, V: np.ndarray):
        super().__init__("FULL_KV", (K, V), n_bits=32)
        self.K = K
        self.V = V
    
    def bytes_on_wire(self) -> int:
        return self.K.size * 4 + self.V.size * 4
    
    def ttft_ms(self, kv_len: int, bandwidth_gbps: float, rtt_ms: float,
                q_len: int, d: int, num_shards: int = 4) -> float:
        bytes_total = self.bytes_on_wire()
        T_transfer = (bytes_total * 8) / (bandwidth_gbps * 1e9) * 1000
        T_compute = kv_len * d * 2 * 1e-6  # 估算
        return T_transfer + T_compute + rtt_ms / 2
    
    def eval_attention(self, Q: np.ndarray, d: int, gt: Optional[np.ndarray] = None) -> dict:
        if gt is not None:
            out = ground_truth(Q, self.K, self.V)
            err = float(np.abs(out - gt).mean())
        else:
            err = 0.0
        return {"name": self.name, "error": err, "error_pct": 0.0}


class AccordFP32Contract(PDContract):
    """ACCORD_FP32: Coreset sketch FP32。"""
    def __init__(self, sketch: CoresetSketch):
        super().__init__("ACCORD_FP32", sketch, n_bits=32)
        self.sketch = sketch
    
    def bytes_on_wire(self) -> int:
        return self.sketch.bytes_size()
    
    def ttft_ms(self, kv_len: int, bandwidth_gbps: float, rtt_ms: float,
                q_len: int, d: int, num_shards: int = 4) -> float:
        bytes_total = self.bytes_on_wire()
        T_transfer = (bytes_total * 8) / (bandwidth_gbps * 1e9) * 1000
        r = self.sketch.centroids.shape[0]
        T_compute = r * d * 2 * 1e-6  # sketch compute
        return T_transfer + T_compute + rtt_ms / 2
    
    def eval_attention(self, Q: np.ndarray, d: int, gt: Optional[np.ndarray] = None) -> dict:
        stats = eval_coreset_sketch(self.sketch, Q, d)
        out = stats.finalize().squeeze(0)
        if gt is not None:
            err = float(np.abs(out - gt).mean())
            err_pct = err / (float(np.abs(gt).mean()) + 1e-10) * 100
        else:
            err = 0.0
            err_pct = 0.0
        return {"name": self.name, "error": err, "error_pct": err_pct}


class AccordINT4Contract(PDContract):
    """ACCORD_INT4: Coreset sketch INT4（Bug 2 fix 后）。"""
    def __init__(self, sketch_fp32: CoresetSketch, n_bits: int = 4):
        q_sketch = quantize_sketch_nbit(sketch_fp32, n_bits=n_bits)
        super().__init__(f"ACCORD_INT{n_bits}", q_sketch, n_bits=n_bits)
        self.sketch_fp32 = sketch_fp32
        self.q_sketch = q_sketch
        self.n_bits = n_bits
    
    def bytes_on_wire(self) -> int:
        return self.q_sketch.bytes_size()
    
    def ttft_ms(self, kv_len: int, bandwidth_gbps: float, rtt_ms: float,
                q_len: int, d: int, num_shards: int = 4) -> float:
        bytes_total = self.bytes_on_wire()
        T_transfer = (bytes_total * 8) / (bandwidth_gbps * 1e9) * 1000
        r = self.sketch_fp32.centroids.shape[0]
        T_compute = r * d * 2 * 1e-6
        return T_transfer + T_compute + rtt_ms / 2
    
    def eval_attention(self, Q: np.ndarray, d: int, gt: Optional[np.ndarray] = None) -> dict:
        sketch_deq = dequantize_sketch_nbit(self.q_sketch)
        stats = eval_coreset_sketch(sketch_deq, Q, d)
        out = stats.finalize().squeeze(0)
        fp32_stats = eval_coreset_sketch(self.sketch_fp32, Q, d)
        out_fp32 = fp32_stats.finalize().squeeze(0)
        fp32_err = float(np.abs(out_fp32 - gt).mean()) if gt is not None else 0.0
        int_err = float(np.abs(out - gt).mean()) if gt is not None else 0.0
        return {
            "name": self.name,
            "error": int_err,
            "error_pct": (int_err - fp32_err) / (fp32_err + 1e-10) * 100,
        }


# ============== E2E Sweep ==============

def run_e2e_single(
    kv_len: int, block_size: int, sketch_r: int, q_len: int,
    bandwidth_gbps: float, rtt_ms: float,
    strategy: str, seed: int = 0, d: int = 128,
    verbose: bool = True,
) -> dict:
    """单组 E2E 配置。"""
    num_blocks = kv_len // block_size
    _, K_all, V_all = make_clustered_kv(
        num_blocks, block_size, d,
        num_clusters=max(4, kv_len // 256), seed=seed
    )
    
    Q = (np.random.default_rng(seed + 1000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K_all, V_all)
    
    # 建立 contracts
    full_kv = FullKVContract(K_all, V_all)
    sketch_fp32 = build_coreset_sketch(K_all, V_all, sketch_r, block_size, seed)
    accord_fp32 = AccordFP32Contract(sketch_fp32)
    accord_int4 = AccordINT4Contract(sketch_fp32, n_bits=4)
    accord_int2 = AccordINT4Contract(sketch_fp32, n_bits=2)
    
    strategy_map = {
        "FULL_KV": full_kv,
        "ACCORD_FP32": accord_fp32,
        "ACCORD_INT4": accord_int4,
        "ACCORD_INT2": accord_int2,
    }
    
    contract = strategy_map[strategy]
    
    ttft = contract.ttft_ms(kv_len, bandwidth_gbps, rtt_ms, q_len, d)
    fidelity = contract.eval_attention(Q, d, gt)
    
    bytes_wire = contract.bytes_on_wire()
    
    if verbose:
        print(
            f"  strat={strategy:>12} bw={bandwidth_gbps:>5}Gbps rtt={rtt_ms:>4}ms  "
            f"ttft={ttft:.4f}ms err={fidelity['error']:.3e}"
        )
    
    return {
        "strategy": strategy,
        "kv_len": kv_len,
        "block_size": block_size,
        "sketch_r": sketch_r,
        "q_len": q_len,
        "bandwidth_gbps": bandwidth_gbps,
        "rtt_ms": rtt_ms,
        "ttft_ms": round(ttft, 6),
        "bytes_on_wire": bytes_wire,
        "error": fidelity["error"],
        "error_pct": fidelity["error_pct"],
    }


def run_e2e_sweep(seed: int = 42, verbose: bool = True) -> list:
    """64 组: 4 strategies × 4 bandwidths × 4 RTTs（固定 kv_len/block/r）。"""
    results = []
    
    strategies = ["FULL_KV", "ACCORD_FP32", "ACCORD_INT4", "ACCORD_INT2"]
    bandwidths = [1.0, 10.0, 100.0, 1000.0]
    rtts = [0.1, 1.0, 10.0, 50.0]
    
    kv_len = 4096
    block_size = 64
    sketch_r = 8
    q_len = 64
    
    if verbose:
        print("=" * 78)
        print(f"E2E PD Sim: Coreset+INT4 (4×4×4={len(strategies)*len(bandwidths)*len(rtts)} configs)")
        print(f"kv_len={kv_len} r={sketch_r} q_len={q_len}")
        print("=" * 78)
    
    total = len(strategies) * len(bandwidths) * len(rtts)
    count = 0
    
    for strategy in strategies:
        for bw in bandwidths:
            for rtt in rtts:
                count += 1
                r = run_e2e_single(
                    kv_len, block_size, sketch_r, q_len,
                    bw, rtt, strategy, seed, verbose=verbose
                )
                results.append(r)
    
    if verbose:
        print(f"\nCompleted {len(results)}/{total} configs")
    return results


def analyze_e2e(results: list) -> dict:
    """分析 E2E 结果。"""
    strategies = sorted(set(r["strategy"] for r in results))
    
    # 每个 strategy 的统计
    strat_stats = {}
    for s in strategies:
        sub = [r for r in results if r["strategy"] == s]
        ttfts = [r["ttft_ms"] for r in sub]
        errs = [r["error"] for r in sub]
        err_pcts = [r["error_pct"] for r in sub]
        strat_stats[s] = {
            "count": len(sub),
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 4),
            "avg_error": round(sum(errs) / len(errs), 6),
            "avg_error_pct": round(sum(err_pcts) / len(err_pcts), 2),
        }
    
    # INT4 vs FP32 对比
    int4_results = [r for r in results if r["strategy"] == "ACCORD_INT4"]
    fp32_results = [r for r in results if r["strategy"] == "ACCORD_FP32"]
    full_kv_results = [r for r in results if r["strategy"] == "FULL_KV"]
    
    int4_vs_fp32_err = []
    int4_vs_full_ttft = []
    for i4 in int4_results:
        fp32 = next((f for f in fp32_results if f["bandwidth_gbps"] == i4["bandwidth_gbps"] and f["rtt_ms"] == i4["rtt_ms"]), None)
        full = next((f for f in full_kv_results if f["bandwidth_gbps"] == i4["bandwidth_gbps"] and f["rtt_ms"] == i4["rtt_ms"]), None)
        if fp32:
            int4_vs_fp32_err.append(i4["error_pct"] - fp32["error_pct"])
        if full:
            ratio = full["ttft_ms"] / max(i4["ttft_ms"], 1e-9)
            int4_vs_full_ttft.append(ratio)
    
    # INT2 vs FP32 对比
    int2_results = [r for r in results if r["strategy"] == "ACCORD_INT2"]
    int2_vs_fp32_err = []
    int2_vs_full_ttft = []
    for i2 in int2_results:
        fp32 = next((f for f in fp32_results if f["bandwidth_gbps"] == i2["bandwidth_gbps"] and f["rtt_ms"] == i2["rtt_ms"]), None)
        full = next((f for f in full_kv_results if f["bandwidth_gbps"] == i2["bandwidth_gbps"] and f["rtt_ms"] == i2["rtt_ms"]), None)
        if fp32:
            int2_vs_fp32_err.append(i2["error_pct"] - fp32["error_pct"])
        if full:
            ratio = full["ttft_ms"] / max(i2["ttft_ms"], 1e-9)
            int2_vs_full_ttft.append(ratio)
    
    return {
        "total": len(results),
        "strat_stats": strat_stats,
        "int4_vs_fp32_avg_err_diff": round(sum(int4_vs_fp32_err) / len(int4_vs_fp32_err), 3) if int4_vs_fp32_err else 0,
        "int4_vs_full_avg_ttft_ratio": round(sum(int4_vs_full_ttft) / len(int4_vs_full_ttft), 2) if int4_vs_full_ttft else 0,
        "int2_vs_fp32_avg_err_diff": round(sum(int2_vs_fp32_err) / len(int2_vs_fp32_err), 3) if int2_vs_fp32_err else 0,
        "int2_vs_full_avg_ttft_ratio": round(sum(int2_vs_full_ttft) / len(int2_vs_full_ttft), 2) if int2_vs_full_ttft else 0,
    }


def main():
    print("ACCORD-KV: E2E PD Sim — Coreset+INT4")
    print("=" * 78)
    
    results = run_e2e_sweep(seed=42, verbose=True)
    analysis = analyze_e2e(results)
    
    print("\n" + "=" * 78)
    print("SUMMARY: E2E Coreset+INT4")
    print("=" * 78)
    
    print("\n--- By Strategy ---")
    print(f"{'Strategy':>14} {'avg_TTFT':>10} {'avg_Error':>12} {'err_vs_FP32':>12}")
    for s, st in analysis["strat_stats"].items():
        diff = st["avg_error_pct"]
        print(f"{s:>14} {st['avg_ttft_ms']:>10.4f} {st['avg_error']:>12.4e} {diff:>+12.2f}%")
    
    print(f"\nACCORD_INT4 vs FP32 error diff: {analysis['int4_vs_fp32_avg_err_diff']:+.3f}%")
    print(f"ACCORD_INT4 vs FULL_KV TTFT ratio: {analysis['int4_vs_full_avg_ttft_ratio']:.2f}x")
    print(f"ACCORD_INT2 vs FP32 error diff: {analysis['int2_vs_fp32_avg_err_diff']:+.3f}%")
    print(f"ACCORD_INT2 vs FULL_KV TTFT ratio: {analysis['int2_vs_full_avg_ttft_ratio']:.2f}x")
    
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "exp5_pd_coreset_int4.json"), "w") as f:
        json.dump({
            "experiment": "E2E_PD_Coreset_INT4",
            "results": results,
            "analysis": analysis,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: results/exp5_pd_coreset_int4.json ({len(results)} configs)")
    return results, analysis


if __name__ == "__main__":
    main()
