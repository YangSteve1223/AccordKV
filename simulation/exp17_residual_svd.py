"""
Exp17: Residual SVD — 频域分解的更聪明融合方式
================================================

核心假设：
- Serial Cascade = Coreset(25%) → SVD(r=8 on values) → INT4，对 clustered 数据 err=3.45 不理想
- 原因：SVD 只作用在 Coreset 压缩后的 values 上，忽略了完整的 V 结构

新思路：
- 对完整 V 做 SVD：V ≈ U_r @ S_r @ V_r^T
- 传输：Coreset sketch + SVD_residual components
- 评估时重建 V_approx = U_r @ S_r @ V_r^T

与 Serial Cascade 的区别：
- Serial Cascade: SVD on Coreset values (r x d) → 压缩量小但丢失 V 的高频
- Residual SVD: SVD on full V (kv_len x d) → 压缩量大但保留更多 V 结构
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
from numpy import linalg as npla

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import ground_truth


# ============== 核心组件 ==============

def kmeans_plusplus_init(K: np.ndarray, r: int, seed: int = 0) -> np.ndarray:
    """K-Means++ 初始化（向量化优化版）"""
    gen = np.random.default_rng(seed)
    n, d = K.shape
    idx = gen.integers(0, n)
    centroids = [K[idx].copy()]
    
    for _ in range(r - 1):
        C = np.array(centroids)
        K_sq = np.sum(K**2, axis=1, keepdims=True)
        C_sq = np.sum(C**2, axis=1)
        cross = K @ C.T
        all_dists = K_sq - 2 * cross + C_sq
        dists = np.min(all_dists, axis=1)
        dists = np.clip(dists, 0, None)
        if dists.sum() < 1e-10:
            dists = np.ones_like(dists) / n
        probs = dists / dists.sum()
        idx = gen.choice(n, p=probs)
        centroids.append(K[idx].copy())
    
    return np.array(centroids)


def build_coreset_sketch(
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    seed: int = 0,
    num_iters: int = 5,  # 减少迭代次数加速
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建 Coreset sketch（优化版）"""
    n, d = K.shape
    gen = np.random.default_rng(seed)
    
    centroids = kmeans_plusplus_init(K, r, seed)
    values = np.zeros((r, d))
    weights = np.zeros(r)
    
    for _ in range(num_iters):
        # Vectorized distance computation
        K_sq = np.sum(K**2, axis=1, keepdims=True)
        C_sq = np.sum(centroids**2, axis=1)
        cross = K @ centroids.T
        all_dists = K_sq - 2 * cross + C_sq
        assignments = np.argmin(all_dists, axis=1)
        
        new_centroids = np.zeros_like(centroids)
        new_values = np.zeros((r, d))
        new_weights = np.zeros(r)
        
        for j in range(r):
            mask = assignments == j
            count = mask.sum()
            if count > 0:
                new_centroids[j] = K[mask].mean(axis=0)
                new_values[j] = V[mask].mean(axis=0)
                new_weights[j] = count / n
            else:
                new_centroids[j] = centroids[j]
                new_values[j] = V[gen.integers(0, n)]
                new_weights[j] = 1e-10
        
        shift = np.sum((centroids - new_centroids) ** 2)
        centroids = new_centroids
        values = new_values
        weights = new_weights
        if shift < 1e-8:
            break
    
    # Final assignments
    K_sq = np.sum(K**2, axis=1, keepdims=True)
    C_sq = np.sum(centroids**2, axis=1)
    cross = K @ centroids.T
    all_dists = K_sq - 2 * cross + C_sq
    final_assignments = np.argmin(all_dists, axis=1)
    
    return centroids, values, weights, final_assignments


def compute_svd_on_v(V: np.ndarray, r: int) -> dict:
    """对完整 V 矩阵做 SVD（不同于 Coreset values）
    
    返回 SVD 分解和奇异值谱分析
    """
    U, S, Vt = npla.svd(V, full_matrices=False)
    
    actual_r = min(r, len(S))
    U_r = U[:, :actual_r]
    S_r = S[:actual_r]
    V_r = Vt[:actual_r, :]
    
    V_reconstructed = U_r @ np.diag(S_r) @ V_r
    
    singular_spectrum = {
        "all_singular_values": S.tolist()[:50],  # 只保存前50个
        "top10": S[:min(10, len(S))].tolist(),
        "explained_variance_ratio": (S**2 / (S**2).sum()).tolist()[:50],
        "cumulative_variance": np.cumsum(S**2 / (S**2).sum()).tolist()[:50],
        "effective_rank": float(np.exp(-np.sum((S/S.sum()) * np.log(S/S.sum() + 1e-30)))),
        "rank_r_coverage": float((S[:actual_r]**2).sum() / (S**2).sum()) if S.sum() > 0 else 0,
    }
    
    return {
        "U_r": U_r,
        "S_r": S_r,
        "V_r": V_r,
        "V_reconstructed": V_reconstructed,
        "singular_spectrum": singular_spectrum,
        "actual_r": actual_r,
        "reconstruction_error": float(np.linalg.norm(V - V_reconstructed) / np.linalg.norm(V)) if np.linalg.norm(V) > 0 else 0,
    }


def compute_svd_on_coreset_values(values: np.ndarray, r: int) -> dict:
    """对 Coreset values 做 SVD（Serial Cascade 方式）"""
    U, S, Vt = npla.svd(values, full_matrices=False)
    
    actual_r = min(r, len(S))
    U_r = U[:, :actual_r]
    S_r = S[:actual_r]
    V_r = Vt[:actual_r, :]
    
    V_reconstructed = U_r @ np.diag(S_r) @ V_r
    
    singular_spectrum = {
        "effective_rank": float(np.exp(-np.sum((S/S.sum()) * np.log(S/S.sum() + 1e-30)))),
        "rank_r_coverage": float((S[:actual_r]**2).sum() / (S**2).sum()) if S.sum() > 0 else 0,
    }
    
    return {
        "U_r": U_r,
        "S_r": S_r,
        "V_r": V_r,
        "V_reconstructed": V_reconstructed,
        "singular_spectrum": singular_spectrum,
        "actual_r": actual_r,
    }


def quantize_nbit(x: np.ndarray, n_bits: int = 4) -> tuple[np.ndarray, float]:
    """INT4/INT8 量化"""
    abs_max = np.abs(x).max()
    if abs_max < 1e-10:
        return x.astype(np.int8), 1.0
    
    max_val = 2 ** (n_bits - 1) - 1
    if max_val < 1:
        max_val = 1
    
    scale = abs_max / max_val
    x_quant = np.round(x / scale).clip(-max_val, max_val)
    return x_quant.astype(np.int8), scale


def dequantize_nbit(x_quant: np.ndarray, scale: float) -> np.ndarray:
    """反量化"""
    return x_quant.astype(np.float32) * scale


def eval_coreset_sketch_with_v(
    centroids: np.ndarray,
    V: np.ndarray,
    weights: np.ndarray,
    Q: np.ndarray,
    d: int,
) -> np.ndarray:
    """评估 Coreset sketch，使用给定的 V（可以是从 SVD 重建的完整 V）
    
    注意：这里 centroids 是 r x d，但 V 可能是 kv_len x d
    """
    r = centroids.shape[0]
    kv_len = V.shape[0]
    
    # 分配 tokens 到 centroids
    # 方式：用 V 乘以 centroids 的注意力权重
    # 但这需要 Q 和 K，这里只有 centroids
    
    # 简化方式：用 dot product 分配
    dists = Q @ centroids.T / np.sqrt(d)  # q_len x r
    scores = dists + np.log(weights + 1e-30)
    
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    
    # 使用 V 作为 weighted sum 的值
    # 但 p 是 q_len x r，V 是 kv_len x d
    # 需要把 V 按照 assignments 映射到 r 个 groups
    
    # 简化：用 centroids 近似分配，然后加权
    # 实际上这里的设计有问题...
    
    # 重新思考：Coreset sketch 的正确评估方式是：
    # y = softmax(Q @ K_c.T) @ V_c
    # 其中 K_c 是 centroids，V_c 是 values
    
    # 如果我们要用 SVD 重建的完整 V，需要重新设计评估方式
    
    # 方式 1：用 SVD 重建的 V，但用完整 K
    # y = softmax(Q @ K.T) @ V_svd
    # 但这需要存储完整 K，违背了压缩目的
    
    # 方式 2：用 SVD 重建 V，同时用 Coreset 分配
    # 这是 Serial Cascade 的做法
    
    # 让我简化：只比较在 Coreset values 上的 SVD
    # 因为这才是公平的比较
    
    raise NotImplementedError("需要重新设计评估方式")


def eval_with_coreset_and_svd_v(
    centroids: np.ndarray,
    V_svd: np.ndarray,
    assignments: np.ndarray,
    weights: np.ndarray,
    Q: np.ndarray,
    d: int,
) -> np.ndarray:
    """评估：用 SVD 重建的完整 V，但用 Coreset 分配方式
    
    思路：
    1. V_svd 是 kv_len x d（SVD 重建的）
    2. 用 Coreset 的 assignments 把 tokens 分配到 r groups
    3. 每个 group 的"代表 V"是 V_svd[该 group tokens] 的加权和
    """
    r = centroids.shape[0]
    kv_len = V_svd.shape[0]
    
    # Scores
    scores = Q @ centroids.T / np.sqrt(d)
    scores = scores + np.log(weights + 1e-30)
    
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)  # q_len x r
    
    # 为每个 group 计算加权 V
    # 方式：group j 的 tokens 是 assignments == j
    # 对应的 V 是 V_svd[assignments == j]
    
    V_grouped = np.zeros((r, d))
    for j in range(r):
        mask = assignments == j
        if mask.sum() > 0:
            # 使用 V_svd 中对应位置的 V，按注意力权重加权
            # 这里简化：用平均
            V_grouped[j] = V_svd[mask].mean(axis=0)
        else:
            V_grouped[j] = centroids[j]  # fallback
    
    y = p @ V_grouped
    
    return y


# ============== 数据生成 ==============

def make_clustered_kv(kv_len: int, d: int, n_clusters: int = 8, seed: int = 0):
    gen = np.random.default_rng(seed)
    centroids = gen.standard_normal((n_clusters, d)) * 2.0
    assignments = gen.integers(0, n_clusters, size=kv_len)
    K = centroids[assignments] + gen.standard_normal((kv_len, d)) * 0.5
    V = K @ (gen.standard_normal((d, d)) * 0.3) + gen.standard_normal((kv_len, d)) * 0.1
    return K.astype(np.float32), V.astype(np.float32)


def make_random_kv(kv_len: int, d: int, seed: int = 0):
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((kv_len, d)).astype(np.float32)
    V = gen.standard_normal((kv_len, d)).astype(np.float32)
    return K, V


def make_skewed_kv(kv_len: int, d: int, n_outliers: int = 16, seed: int = 0):
    gen = np.random.default_rng(seed)
    outlier_K = gen.standard_normal((n_outliers, d)) * 3.0
    outlier_V = gen.standard_normal((n_outliers, d)) * 3.0
    normal_K = gen.standard_normal((kv_len - n_outliers, d)) * 0.3
    normal_V = gen.standard_normal((kv_len - n_outliers, d)) * 0.3
    K = np.concatenate([outlier_K, normal_K])
    V = np.concatenate([outlier_V, normal_V])
    perm = gen.permutation(kv_len)
    return K[perm].astype(np.float32), V[perm].astype(np.float32)


# ============== 核心实验 ==============

@dataclass
class ResidualSVDResult:
    """单次实验结果"""
    kv_type: str
    kv_len: int
    q_len: int
    coreset_ratio: float
    svd_r: int
    int4_bits: int
    
    # 误差指标
    err_baseline_coreset: float  # 仅 Coreset
    err_serial_cascade: float    # Serial Cascade (SVD on values)
    err_residual_svd: float       # Residual SVD (SVD on full V)
    
    # SVD 分析
    svd_v_coverage: float         # 完整 V 的 SVD 覆盖
    svd_values_coverage: float    # Coreset values 的 SVD 覆盖
    
    # 压缩比
    compression_serial: float
    compression_residual: float
    
    # 字节数
    bytes_full: int
    bytes_serial: int
    bytes_residual: int


def run_residual_svd_experiment(
    kv_type: Literal["clustered", "random", "skewed"],
    kv_len: int,
    q_len: int,
    coreset_ratio: float,
    svd_r: int,
    int4_bits: int,
    d: int = 128,
    seed: int = 42,
    verbose: bool = False,
) -> ResidualSVDResult:
    """运行单次 Residual SVD 实验"""
    
    # 生成数据
    if kv_type == "clustered":
        K, V = make_clustered_kv(kv_len, d, seed=seed)
    elif kv_type == "random":
        K, V = make_random_kv(kv_len, d, seed=seed)
    else:
        K, V = make_skewed_kv(kv_len, d, seed=seed)
    
    gen = np.random.default_rng(seed + 1000)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    gt = ground_truth(Q, K, V)
    
    r_coreset = max(4, int(kv_len * coreset_ratio))
    
    # ===== Baseline: Coreset only =====
    centroids, values, weights, assignments = build_coreset_sketch(K, V, r_coreset, seed=seed)
    
    # 评估 baseline（直接用 Coreset values）
    scores = Q @ centroids.T / np.sqrt(d)
    scores = scores + np.log(weights + 1e-30)
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y_baseline = p @ values
    err_baseline = float(np.abs(y_baseline - gt).mean())
    
    # ===== Serial Cascade: Coreset → SVD on values → INT4 =====
    # SVD on Coreset values
    svd_values = compute_svd_on_coreset_values(values, svd_r)
    V_values_recon = svd_values["V_reconstructed"]
    
    # INT4 量化
    V_values_quant, scale_values = quantize_nbit(V_values_recon, int4_bits)
    V_values_final = dequantize_nbit(V_values_quant, scale_values)
    
    y_serial = p @ V_values_final
    err_serial = float(np.abs(y_serial - gt).mean())
    
    # ===== Residual SVD: Coreset → SVD on full V → INT4 =====
    # SVD on full V
    svd_v = compute_svd_on_v(V, svd_r)
    V_v_recon = svd_v["V_reconstructed"]
    
    # 用 Coreset 分配方式评估
    # 计算每个 group 的加权 V
    V_grouped = np.zeros((r_coreset, d))
    for j in range(r_coreset):
        mask = assignments == j
        if mask.sum() > 0:
            V_grouped[j] = V_v_recon[mask].mean(axis=0)
        else:
            V_grouped[j] = centroids[j]
    
    # INT4 量化
    V_v_quant, scale_v = quantize_nbit(V_grouped, int4_bits)
    V_v_final = dequantize_nbit(V_v_quant, scale_v)
    
    y_residual = p @ V_v_final
    err_residual = float(np.abs(y_residual - gt).mean())
    
    # 字节数
    bytes_full = kv_len * d * 2 * 4
    
    # Serial: Coreset(K) + SVD(values)
    bytes_serial = r_coreset * d * 2 * 4 + svd_values["U_r"].size * 4 + svd_values["S_r"].size * 4 + svd_values["V_r"].size * 4 + 4
    
    # Residual: Coreset(K) + SVD(V)
    bytes_residual = r_coreset * d * 2 * 4 + svd_v["U_r"].size * 4 + svd_v["S_r"].size * 4 + svd_v["V_r"].size * 4 + 4
    
    compression_serial = bytes_full / bytes_serial
    compression_residual = bytes_full / bytes_residual
    
    if verbose:
        print(f"  [{kv_type}] kv={kv_len} q={q_len} cr={coreset_ratio:.2f} r={svd_r} b={int4_bits}")
        print(f"    Baseline: {err_baseline:.4e}, Serial: {err_serial:.4e}, Residual: {err_residual:.4e}")
        print(f"    V coverage: {svd_v['singular_spectrum']['rank_r_coverage']:.4f}, Values coverage: {svd_values['singular_spectrum']['rank_r_coverage']:.4f}")
    
    return ResidualSVDResult(
        kv_type=kv_type,
        kv_len=kv_len,
        q_len=q_len,
        coreset_ratio=coreset_ratio,
        svd_r=svd_r,
        int4_bits=int4_bits,
        err_baseline_coreset=err_baseline,
        err_serial_cascade=err_serial,
        err_residual_svd=err_residual,
        svd_v_coverage=svd_v['singular_spectrum']['rank_r_coverage'],
        svd_values_coverage=svd_values['singular_spectrum']['rank_r_coverage'],
        compression_serial=compression_serial,
        compression_residual=compression_residual,
        bytes_full=bytes_full,
        bytes_serial=bytes_serial,
        bytes_residual=bytes_residual,
    )


# ============== 扫描 ==============

def run_full_sweep(seed: int = 42, verbose: bool = True) -> dict:
    """完整参数扫描"""
    
    configs = {
        "coreset_ratios": [0.10, 0.20, 0.25, 0.30, 0.50],
        "svd_r_values": [2, 4, 8],
        "int4_bits": [3, 4, 6, 8],
        "kv_types": ["clustered", "random", "skewed"],
        "kv_lens": [1024, 4096],
        "q_lens": [16, 64],
    }
    
    results = []
    singular_spectra = []
    
    total_configs = (
        len(configs["coreset_ratios"]) *
        len(configs["svd_r_values"]) *
        len(configs["int4_bits"]) *
        len(configs["kv_types"]) *
        len(configs["kv_lens"]) *
        len(configs["q_lens"])
    )
    
    config_idx = 0
    start_time = time.time()
    
    if verbose:
        print("=" * 70)
        print("Exp17: Residual SVD Sweep")
        print("=" * 70)
        print(f"Total configs: {total_configs}")
        print()
    
    for kv_type in configs["kv_types"]:
        for kv_len in configs["kv_lens"]:
            for q_len in configs["q_lens"]:
                for coreset_ratio in configs["coreset_ratios"]:
                    for svd_r in configs["svd_r_values"]:
                        for int4_bits in configs["int4_bits"]:
                            config_idx += 1
                            
                            try:
                                result = run_residual_svd_experiment(
                                    kv_type=kv_type,
                                    kv_len=kv_len,
                                    q_len=q_len,
                                    coreset_ratio=coreset_ratio,
                                    svd_r=svd_r,
                                    int4_bits=int4_bits,
                                    d=128,
                                    seed=seed,
                                    verbose=False,
                                )
                                
                                results.append(asdict(result))
                                
                                # 保存奇异值谱
                                singular_spectra.append({
                                    "kv_type": kv_type,
                                    "kv_len": kv_len,
                                    "q_len": q_len,
                                    "coreset_ratio": coreset_ratio,
                                    "svd_r": svd_r,
                                    "svd_v_coverage": result.svd_v_coverage,
                                    "svd_values_coverage": result.svd_values_coverage,
                                })
                                
                            except Exception as e:
                                print(f"Error at config {config_idx}: {e}")
                                continue
                            
                            if verbose and config_idx % 100 == 0:
                                elapsed = time.time() - start_time
                                rate = config_idx / elapsed
                                remaining = (total_configs - config_idx) / rate if rate > 0 else 0
                                print(f"Progress: {config_idx}/{total_configs} ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
    
    elapsed = time.time() - start_time
    
    # 分析
    analysis = analyze_results(results, singular_spectra)
    
    if verbose:
        print(f"\nSweep complete in {elapsed:.1f}s!")
        print(f"Total configs: {len(results)}")
    
    return {
        "results": results,
        "singular_spectra": singular_spectra,
        "configs": configs,
        "analysis": analysis,
        "elapsed_seconds": elapsed,
    }


def analyze_results(results: list, spectra: list) -> dict:
    """分析实验结果"""
    
    # ===== 按 KV 类型分析 =====
    by_kv_type = {}
    for kv_type in ["clustered", "random", "skewed"]:
        subset = [r for r in results if r["kv_type"] == kv_type]
        if subset:
            by_kv_type[kv_type] = {
                "mean_err_residual": float(np.mean([r["err_residual_svd"] for r in subset])),
                "mean_err_serial": float(np.mean([r["err_serial_cascade"] for r in subset])),
                "mean_err_baseline": float(np.mean([r["err_baseline_coreset"] for r in subset])),
                "residual_wins": sum(1 for r in subset if r["err_residual_svd"] < r["err_serial_cascade"]),
                "total": len(subset),
            }
    
    # ===== 按 SVD r 分析 =====
    by_svd_r = {}
    for r in [2, 4, 8]:
        subset = [r2 for r2 in results if r2["svd_r"] == r]
        if subset:
            by_svd_r[str(r)] = {
                "mean_err_residual": float(np.mean([r2["err_residual_svd"] for r2 in subset])),
                "mean_err_serial": float(np.mean([r2["err_serial_cascade"] for r2 in subset])),
                "avg_v_coverage": float(np.mean([s["svd_v_coverage"] for s in spectra if s["svd_r"] == r])),
                "avg_values_coverage": float(np.mean([s["svd_values_coverage"] for s in spectra if s["svd_r"] == r])),
            }
    
    # ===== 总体统计 =====
    residual_wins = sum(1 for r in results if r["err_residual_svd"] < r["err_serial_cascade"])
    total_comparison = len(results)
    win_rate = residual_wins / total_comparison if total_comparison > 0 else 0
    
    # ===== 诚实判决 =====
    honest_verdict = {
        "residual_svd_works": win_rate > 0.5,
        "win_rate": win_rate,
        "wins": residual_wins,
        "total": total_comparison,
        "reason": None,
    }
    
    avg_v_coverage = np.mean([s["svd_v_coverage"] for s in spectra])
    avg_values_coverage = np.mean([s["svd_values_coverage"] for s in spectra])
    
    if avg_v_coverage < 0.5:
        honest_verdict["reason"] = f"完整 V 的 SVD 覆盖仅 {avg_v_coverage:.1%}，说明 V 是 high-rank 的，小 r 不够用。"
    elif win_rate < 0.5:
        honest_verdict["reason"] = f"Residual SVD 在 {win_rate:.1%} 配置中优于 Serial Cascade，不够稳健。"
    else:
        honest_verdict["reason"] = f"Residual SVD 在 {win_rate:.1%} 配置中优于 Serial Cascade。"
    
    # ===== Pareto 前沿 =====
    pareto = compute_pareto_frontier(results)
    
    return {
        "by_kv_type": by_kv_type,
        "by_svd_r": by_svd_r,
        "singular_coverage": {
            "V": float(avg_v_coverage),
            "values": float(avg_values_coverage),
        },
        "honest_verdict": honest_verdict,
        "pareto_frontier": pareto,
        "residual_wins": residual_wins,
        "total_comparison": total_comparison,
    }


def compute_pareto_frontier(results: list) -> list:
    """计算 Pareto 前沿"""
    pareto = []
    
    for r in results:
        pareto.append({
            "kv_type": r["kv_type"],
            "kv_len": r["kv_len"],
            "q_len": r["q_len"],
            "compression": r["compression_residual"],
            "err": r["err_residual_svd"],
        })
    
    pareto.sort(key=lambda x: x["compression"], reverse=True)
    
    pareto_frontier = []
    min_err = float('inf')
    
    for p in pareto:
        if p["err"] < min_err:
            pareto_frontier.append(p)
            min_err = p["err"]
    
    return pareto_frontier


# ============== Sanity Check ==============

def run_sanity_check(seed: int = 42) -> list:
    """3 点 sanity check"""
    
    sanity_configs = [
        ("clustered", 1024, 16),
        ("random", 4096, 64),
        ("skewed", 1024, 16),
    ]
    
    results = []
    
    print("=" * 60)
    print("Exp17 Sanity Check (3 configs)")
    print("=" * 60)
    
    for kv_type, kv_len, q_len in sanity_configs:
        result = run_residual_svd_experiment(
            kv_type=kv_type,
            kv_len=kv_len,
            q_len=q_len,
            coreset_ratio=0.25,
            svd_r=4,
            int4_bits=4,
            d=128,
            seed=seed,
            verbose=True,
        )
        results.append(asdict(result))
    
    return results


# ============== Main ==============

def main():
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Sanity check
    print("\n" + "=" * 70)
    print("STEP 1: Sanity Check")
    print("=" * 70)
    sanity_results = run_sanity_check(seed=42)
    
    sanity_path = os.path.join(output_dir, "exp17_sanity.json")
    with open(sanity_path, "w") as f:
        json.dump({"sanity_check": sanity_results}, f, indent=2, default=str)
    print(f"Saved: {sanity_path}")
    
    # 检查 singular coverage
    avg_v_coverage = np.mean([r["svd_v_coverage"] for r in sanity_results])
    avg_values_coverage = np.mean([r["svd_values_coverage"] for r in sanity_results])
    print(f"\nAvg V coverage: {avg_v_coverage:.1%}")
    print(f"Avg Values coverage: {avg_values_coverage:.1%}")
    
    if avg_v_coverage < 0.5:
        print("⚠️ Warning: 完整 V 的 SVD coverage 很低，残差 SVD 可能不 work")
    
    # 2. Full sweep
    print("\n" + "=" * 70)
    print("STEP 2: Full Parameter Sweep")
    print("=" * 70)
    sweep_data = run_full_sweep(seed=42, verbose=True)
    
    # 3. Save results
    sweep_path = os.path.join(output_dir, "exp17_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump({
            "results": sweep_data["results"],
            "configs": sweep_data["configs"],
            "analysis": sweep_data["analysis"],
            "elapsed_seconds": sweep_data["elapsed_seconds"],
        }, f, indent=2, default=str)
    print(f"Saved: {sweep_path} ({len(sweep_data['results'])} configs)")
    
    # 4. Singular spectra
    spectra_path = os.path.join(output_dir, "exp17_singular_spectrum.json")
    with open(spectra_path, "w") as f:
        json.dump({"singular_spectra": sweep_data["singular_spectra"]}, f, indent=2, default=str)
    print(f"Saved: {spectra_path}")
    
    # 5. Pareto
    pareto_path = os.path.join(output_dir, "exp17_pareto.json")
    with open(pareto_path, "w") as f:
        json.dump({"pareto_frontier": sweep_data["analysis"]["pareto_frontier"]}, f, indent=2, default=str)
    print(f"Saved: {pareto_path}")
    
    # 6. Comparison
    comparison_path = os.path.join(output_dir, "exp17_vs_serial_cascade.json")
    comparison_data = [
        {
            "kv_type": r["kv_type"],
            "kv_len": r["kv_len"],
            "q_len": r["q_len"],
            "coreset_ratio": r["coreset_ratio"],
            "svd_r": r["svd_r"],
            "int4_bits": r["int4_bits"],
            "err_serial_cascade": r["err_serial_cascade"],
            "err_residual_svd": r["err_residual_svd"],
            "improvement": r["err_serial_cascade"] - r["err_residual_svd"],
        }
        for r in sweep_data["results"]
    ]
    with open(comparison_path, "w") as f:
        json.dump({"comparison": comparison_data}, f, indent=2, default=str)
    print(f"Saved: {comparison_path}")
    
    # 7. Generate report
    generate_report(sweep_data, output_dir)
    
    # 8. Print summary
    print("\n" + "=" * 70)
    print("EXP17 SUMMARY")
    print("=" * 70)
    
    verdict = sweep_data["analysis"]["honest_verdict"]
    print(f"\n🎯 诚实判决:")
    print(f"   Residual SVD {'✅ WORK' if verdict['residual_svd_works'] else '❌ NOT WORK'}")
    print(f"   Win rate: {verdict['win_rate']:.1%} ({verdict['wins']}/{verdict['total']})")
    print(f"   原因: {verdict['reason']}")
    
    print(f"\n📊 SVD Coverage:")
    cov = sweep_data["analysis"]["singular_coverage"]
    print(f"   V (full): {cov['V']:.1%}")
    print(f"   Values (coreset): {cov['values']:.1%}")
    
    print(f"\n📈 By KV Type:")
    for kv_type, stats in sweep_data["analysis"]["by_kv_type"].items():
        wins = stats["residual_wins"]
        total = stats["total"]
        print(f"   {kv_type}: err_res={stats['mean_err_residual']:.4e}, err_serial={stats['mean_err_serial']:.4e}, wins={wins}/{total}")
    
    print(f"\n✅ Complete!")
    
    return sweep_data


def generate_report(sweep_data: dict, output_dir: str) -> None:
    """生成完整报告"""
    
    report = []
    report.append("# Exp17: Residual SVD — 对完整 V 的 SVD\n\n")
    
    report.append("## 核心假设\n\n")
    report.append("Serial Cascade = Coreset → SVD(on Coreset values) → INT4\n")
    report.append("- SVD 只作用在 Coreset 压缩后的 values (r x d) 上\n")
    report.append("- 忽略了完整 V (kv_len x d) 的结构\n\n")
    
    report.append("## 新思路\n\n")
    report.append("Residual SVD = Coreset → SVD(on full V) → INT4\n")
    report.append("- SVD 作用在完整 V (kv_len x d) 上\n")
    report.append("- 然后用 Coreset 分配方式评估\n\n")
    
    report.append("## 诚实判决\n\n")
    verdict = sweep_data["analysis"]["honest_verdict"]
    report.append(f"**Residual SVD {'✅ WORK' if verdict['residual_svd_works'] else '❌ NOT WORK'}**\n\n")
    report.append(f"- Win rate: {verdict['win_rate']:.1%} ({verdict['wins']}/{verdict['total']})\n")
    report.append(f"- 原因: {verdict['reason']}\n\n")
    
    report.append("## SVD Coverage 分析\n\n")
    cov = sweep_data["analysis"]["singular_coverage"]
    report.append(f"| 对象 | Coverage (r=2,4,8 平均) |\n")
    report.append(f"|------|------------------------|\n")
    report.append(f"| 完整 V | {cov['V']:.1%} |\n")
    report.append(f"| Coreset values | {cov['values']:.1%} |\n")
    report.append("\n")
    
    if cov["V"] < 0.5:
        report.append("**关键发现**：完整 V 的 SVD coverage 很低，说明 V 是 high-rank 的。\n")
        report.append("小 r 的 SVD 无法捕捉 V 的主要变异，Residual SVD 在这里是无效的。\n\n")
    
    report.append("## 按 KV 类型分析\n\n")
    report.append("| KV Type | Residual err | Serial err | Wins |\n")
    report.append("|---------|--------------|------------|------|\n")
    for kv_type, stats in sweep_data["analysis"]["by_kv_type"].items():
        wins = stats["residual_wins"]
        total = stats["total"]
        report.append(f"| {kv_type} | {stats['mean_err_residual']:.4e} | {stats['mean_err_serial']:.4e} | {wins}/{total} |\n")
    report.append("\n")
    
    report.append("## 结论\n\n")
    if verdict["residual_svd_works"]:
        report.append("✅ **Residual SVD 比 Serial Cascade 更聪明**\n\n")
        report.append(f"- 在 {verdict['win_rate']:.1%} 配置中优于 Serial Cascade\n")
    else:
        report.append("❌ **Residual SVD 不 work，Serial Cascade 仍然更好**\n\n")
        report.append(f"- Win rate 仅 {verdict['win_rate']:.1%}\n")
        report.append(f"- 原因：完整 V 是 high-rank 的，小 r SVD 不够用\n")
        report.append("- **诚实失败发现**：对完整 V 做 SVD 不是解决 clustered 数据问题的正确方法\n")
    
    report_path = os.path.join(output_dir, "exp17_residual_svd_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(report))
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
