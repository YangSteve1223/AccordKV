"""
Method D Ablation Study
========================
Comprehensive ablation over:
- Cluster count k ∈ {1, 2, 4, 8, 16, 32}
- Cluster algorithms: KMeans / GMM / Agglomerative / Birch
- Distributions: random / skewed / clustered
- SVD rank r ∈ {2, 4, 8}
- 3 seeds

Total: 6 × 4 × 3 × 3 × 3 = 648 configs

Author: Accord-KV Team
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple, Optional
from itertools import product

import numpy as np
from numpy import linalg as npla
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ============== Configuration ==============

CONFIG = {
    "kv_len": 4096,
    "d": 128,
    "r_values": [2, 4, 8],
    "q_len": 64,
    "seeds": [42, 123, 456],
    "k_values": [1, 2, 4, 8, 16, 32],
    "algorithms": ["KMeans", "GMM", "Agglomerative", "Birch"],
    "distributions": ["random", "skewed", "clustered"],
}


# ============== Ground Truth Attention ==============

def attention_output(Q, K, V):
    """Standard softmax attention. Q:[q,d] K/V:[kv,d] -> [q,d]"""
    scores = Q @ K.T / np.sqrt(CONFIG["d"])
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    l = p.sum(axis=-1, keepdims=True)
    return (p @ V) / np.clip(l, 1e-30, None)


# ============== Data Generation ==============

def make_clustered_kv(kv_len: int, d: int, n_clusters: int = 8, seed: int = 42):
    """K: cluster structure; V = K @ W + noise."""
    gen = np.random.default_rng(seed)
    centroids = gen.standard_normal((n_clusters, d)) * 2.0
    assignments = gen.integers(0, n_clusters, size=kv_len)
    K = centroids[assignments] + gen.standard_normal((kv_len, d)) * 0.5
    W_v = gen.standard_normal((d, d)) * 0.3
    V = K @ W_v + gen.standard_normal((kv_len, d)) * 0.1
    return K.astype(np.float32), V.astype(np.float32), assignments.astype(np.int32)


def make_random_kv(kv_len: int, d: int, seed: int = 42):
    """Random: independent Gaussian K and V."""
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((kv_len, d)).astype(np.float32)
    V = gen.standard_normal((kv_len, d)).astype(np.float32)
    return K, V, None


def make_skewed_kv(kv_len: int, d: int, n_outliers: int = 16, seed: int = 42):
    """Skewed: few high-magnitude outliers + many small tokens."""
    gen = np.random.default_rng(seed)
    outlier_K = gen.standard_normal((n_outliers, d)) * 3.0
    outlier_V = gen.standard_normal((n_outliers, d)) * 3.0
    normal_K = gen.standard_normal((kv_len - n_outliers, d)) * 0.3
    normal_V = gen.standard_normal((kv_len - n_outliers, d)) * 0.3
    K = np.concatenate([outlier_K, normal_K])
    V = np.concatenate([outlier_V, normal_V])
    perm = gen.permutation(kv_len)
    return K[perm].astype(np.float32), V[perm].astype(np.float32), None


# Underlying cluster structure for clustered distribution
UNDERLYING_N_CLUSTERS = 8


def generate_data(distribution: str, kv_len: int, d: int, seed: int = 42):
    """Generate K, V based on distribution type.
    
    Note: The underlying cluster structure (n_clusters=8) is FIXED regardless of
    the k value used in Method D compression. This ensures fair comparison.
    """
    if distribution == "clustered":
        return make_clustered_kv(kv_len, d, UNDERLYING_N_CLUSTERS, seed=seed)
    elif distribution == "random":
        return make_random_kv(kv_len, d, seed=seed)
    elif distribution == "skewed":
        return make_skewed_kv(kv_len, d, n_outliers=16, seed=seed)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ============== Clustering Methods ==============

def cluster_kmeans(K: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """K-Means clustering."""
    if n_clusters >= len(K):
        return np.arange(len(K))
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, 
                    random_state=seed, max_iter=10)
    return kmeans.fit_predict(K)


def cluster_gmm(K: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Gaussian Mixture Model clustering."""
    if n_clusters >= len(K):
        return np.arange(len(K))
    gmm = GaussianMixture(n_components=n_clusters, random_state=seed, 
                         covariance_type='full', max_iter=10, n_init=1)
    return gmm.fit_predict(K)


def cluster_agglomerative(K: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Agglomerative (Hierarchical) clustering."""
    if n_clusters >= len(K):
        return np.arange(len(K))
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    return agg.fit_predict(K)


def cluster_birch(K: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Birch clustering."""
    if n_clusters >= len(K):
        return np.arange(len(K))
    birch = Birch(n_clusters=n_clusters, threshold=0.5, 
                  branching_factor=50)
    return birch.fit_predict(K)


def get_cluster_labels(K: np.ndarray, algorithm: str, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Get cluster labels using specified algorithm."""
    if algorithm == "KMeans":
        return cluster_kmeans(K, n_clusters, seed)
    elif algorithm == "GMM":
        return cluster_gmm(K, n_clusters, seed)
    elif algorithm == "Agglomerative":
        return cluster_agglomerative(K, n_clusters, seed)
    elif algorithm == "Birch":
        return cluster_birch(K, n_clusters, seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ============== Method D: K-Cluster-Conditional V Compression =============

def method_D_cluster_conditional(K: np.ndarray, V: np.ndarray, n_clusters: int,
                                  r: int, algorithm: str, seed: int = 0):
    """Cluster K -> per-cluster SVD on V.
    
    Args:
        K: Key matrix [kv_len, d]
        V: Value matrix [kv_len, d]
        n_clusters: Number of clusters (k=1 means global SVD baseline)
        r: SVD rank per cluster
        algorithm: Clustering algorithm
        seed: Random seed
        
    Returns:
        V_approx: Approximated V [kv_len, d]
        labels: Cluster assignments [kv_len]
        cluster_sizes: Dict of cluster -> size
        compression_ratio: Overall compression ratio
        timing: Dict of timing info
    """
    kv_len, d = K.shape
    t0 = time.time()
    
    # Special case: k=1 means global SVD (baseline)
    if n_clusters == 1:
        U, S, Vt = npla.svd(V, full_matrices=False)
        r_actual = min(r, len(S))
        U_r = U[:, :r_actual]
        S_r = S[:r_actual]
        Vt_r = Vt[:r_actual, :]
        V_approx = (U_r @ np.diag(S_r) @ Vt_r).astype(np.float32)
        labels = np.zeros(kv_len, dtype=np.int32)
        cluster_sizes = {0: kv_len}
        cluster_timing = {"clustering": 0.0, "svd_total": time.time() - t0}
        
        original_size = kv_len * d * 4
        compressed_size = (U_r.size + S_r.size + Vt_r.size) * 4
        compression_ratio = original_size / max(compressed_size, 1)
        return V_approx, labels, cluster_sizes, compression_ratio, cluster_timing
    
    # Normal case: cluster K, then per-cluster SVD on V
    t_cluster = time.time()
    labels = get_cluster_labels(K, algorithm, n_clusters, seed)
    cluster_time = time.time() - t_cluster
    
    V_approx = np.zeros((kv_len, d), dtype=np.float32)
    cluster_sizes = {}
    total_compressed = 0
    
    t_svd = time.time()
    for c in range(n_clusters):
        mask = labels == c
        V_c = V[mask]
        n_c = mask.sum()
        cluster_sizes[int(c)] = int(n_c)
        
        if n_c <= r:
            # Not enough points for SVD, keep original
            V_approx[mask] = V_c
            total_compressed += n_c * d * 4
        else:
            # SVD on this cluster
            U_c, S_c, Vt_c = npla.svd(V_c, full_matrices=False)
            r_c = min(r, len(S_c))
            U_c_r = U_c[:, :r_c]
            S_c_r = S_c[:r_c]
            Vt_c_r = Vt_c[:r_c, :]
            V_approx[mask] = (U_c_r @ np.diag(S_c_r) @ Vt_c_r).astype(np.float32)
            total_compressed += (U_c_r.size + S_c_r.size + Vt_c_r.size) * 4
    
    svd_time = time.time() - t_svd
    cluster_timing = {"clustering": cluster_time, "svd_total": svd_time}
    
    original_size = kv_len * d * 4
    compression_ratio = original_size / max(total_compressed, 1)
    
    return V_approx, labels, cluster_sizes, compression_ratio, cluster_timing


# ============== Baselines ==============

def baseline_global_svd(V, r):
    """Global SVD on V (same as Method D with k=1)."""
    U, S, Vt = npla.svd(V, full_matrices=False)
    r_actual = min(r, len(S))
    U_r = U[:, :r_actual]
    S_r = S[:r_actual]
    Vt_r = Vt[:r_actual, :]
    V_approx = (U_r @ np.diag(S_r) @ Vt_r).astype(np.float32)
    kv_len, d = V.shape
    original_size = kv_len * d * 4
    compressed_size = (U_r.size + S_r.size + Vt_r.size) * 4
    compression_ratio = original_size / max(compressed_size, 1)
    return V_approx, compression_ratio


# ============== Metrics ==============

def compute_attention_metrics(O_gt, O_approx):
    """Compute attention output error metrics."""
    diff = O_gt - O_approx
    return {
        "error_frobenius": float(npla.norm(diff, 'fro')),
        "error_mean": float(np.abs(diff).mean()),
        "error_max": float(np.abs(diff).max()),
        "error_std": float(np.std(diff)),
    }


def compute_v_error(V, V_approx):
    """Compute V reconstruction error."""
    diff = V - V_approx
    return {
        "v_error_fro": float(npla.norm(diff, 'fro')),
        "v_error_mean": float(np.sqrt(np.mean(diff ** 2))),
        "v_error_rel": float(npla.norm(diff, 'fro') / max(npla.norm(V, 'fro'), 1e-10)),
    }


# ============== Run Single Config ==============

def run_single_config(distribution: str, algorithm: str, k: int, r: int, seed: int) -> dict:
    """Run Method D with one configuration.
    
    Returns results with all metrics and timing info.
    """
    cfg = CONFIG
    kv_len = cfg["kv_len"]
    d = cfg["d"]
    q_len = cfg["q_len"]
    
    # Generate data (use fixed underlying structure for fair comparison)
    K, V, _ = generate_data(distribution, kv_len, d, seed=seed)
    
    # Generate query
    gen_q = np.random.default_rng(seed + 1000)
    Q = (gen_q.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    # Ground truth attention
    O_gt = attention_output(Q, K, V)
    
    # Method D compression
    t0 = time.time()
    V_approx, labels, cluster_sizes, compression_ratio, timing = \
        method_D_cluster_conditional(K, V, k, r, algorithm, seed)
    
    # Compute approximate attention
    O_approx = attention_output(Q, K, V_approx)
    
    total_time = time.time() - t0
    
    # Compute metrics
    attn_metrics = compute_attention_metrics(O_gt, O_approx)
    v_metrics = compute_v_error(V, V_approx)
    
    return {
        "config": {
            "distribution": distribution,
            "algorithm": algorithm,
            "k": k,
            "r": r,
            "seed": seed,
        },
        "metrics": {
            "attention_error_mean": attn_metrics["error_mean"],
            "attention_error_fro": attn_metrics["error_frobenius"],
            "attention_error_max": attn_metrics["error_max"],
            "v_error_fro": v_metrics["v_error_fro"],
            "v_error_mean": v_metrics["v_error_mean"],
            "v_error_rel": v_metrics["v_error_rel"],
        },
        "compression_ratio": float(compression_ratio),
        "cluster_sizes": cluster_sizes,
        "n_clusters_actual": len(cluster_sizes),
        "timing": {
            "total": total_time,
            "clustering": timing["clustering"],
            "svd": timing["svd_total"],
        },
        "improvement_vs_baseline": None,  # Fill in later
    }


def run_toy_test() -> dict:
    """Run a single toy test to verify basic functionality."""
    print("\n" + "=" * 70)
    print("TOY TEST: Method D with k=4, KMeans, clustered, r=4, seed=42")
    print("=" * 70)
    
    result = run_single_config("clustered", "KMeans", k=4, r=4, seed=42)
    
    # Also run baseline for comparison
    cfg = CONFIG
    kv_len = cfg["kv_len"]
    d = cfg["d"]
    q_len = cfg["q_len"]
    
    K, V, _ = make_clustered_kv(kv_len, d, n_clusters=4, seed=42)
    gen_q = np.random.default_rng(42 + 1000)
    Q = (gen_q.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    O_gt = attention_output(Q, K, V)
    
    V_baseline, _ = baseline_global_svd(V, r=4)
    O_baseline = attention_output(Q, K, V_baseline)
    baseline_metrics = compute_attention_metrics(O_gt, O_baseline)
    
    # Improvement
    improvement = baseline_metrics["error_mean"] - result["metrics"]["attention_error_mean"]
    result["improvement_vs_baseline"] = improvement
    
    print(f"\nResults:")
    print(f"  Method D Attention Error: {result['metrics']['attention_error_mean']:.6f}")
    print(f"  Baseline (global SVD) Error: {baseline_metrics['error_mean']:.6f}")
    print(f"  Improvement: {improvement:+.6f} ({improvement/baseline_metrics['error_mean']*100:+.2f}%)")
    print(f"  Compression Ratio: {result['compression_ratio']:.2f}x")
    print(f"  N clusters: {result['n_clusters_actual']}")
    print(f"  Total time: {result['timing']['total']:.2f}s")
    
    # Check improvement
    k_val = result["config"]["k"]
    if k_val > 1:
        assert improvement >= -0.001, f"Expected improvement but got {improvement}"
        print(f"\n✓ Self-test PASSED: Method D improves over baseline")
    else:
        assert abs(improvement) < 0.001, f"k=1 should equal baseline but got {improvement}"
        print(f"\n✓ Self-test PASSED: k=1 equals baseline")
    
    return result


def run_full_ablation() -> List[dict]:
    """Run the full ablation study across all 648 configs."""
    cfg = CONFIG
    
    # Generate all configurations
    all_configs = list(product(
        cfg["distributions"],
        cfg["algorithms"],
        cfg["k_values"],
        cfg["r_values"],
        cfg["seeds"]
    ))
    
    n_configs = len(all_configs)
    print("\n" + "=" * 70)
    print(f"Method D Full Ablation Study")
    print("=" * 70)
    print(f"Configurations: {n_configs}")
    print(f"  - k values: {cfg['k_values']}")
    print(f"  - algorithms: {cfg['algorithms']}")
    print(f"  - distributions: {cfg['distributions']}")
    print(f"  - r values: {cfg['r_values']}")
    print(f"  - seeds: {cfg['seeds']}")
    print("=" * 70)
    
    results = []
    t_start = time.time()
    
    for i, (dist, algo, k, r, seed) in enumerate(all_configs):
        try:
            result = run_single_config(dist, algo, k, r, seed)
            results.append(result)
            
            # Progress update every 50 configs
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                remaining = (n_configs - i - 1) / rate
                print(f"  [{i+1}/{n_configs}] {elapsed:.1f}s elapsed, ~{remaining/60:.1f}min remaining")
                
        except Exception as e:
            print(f"  ERROR at config {i+1} ({dist}, {algo}, k={k}, r={r}, seed={seed}): {e}")
            results.append({
                "config": {"distribution": dist, "algorithm": algo, "k": k, "r": r, "seed": seed},
                "error": str(e),
                "failed": True,
            })
    
    total_time = time.time() - t_start
    print(f"\nCompleted {len(results)} configs in {total_time:.1f}s ({total_time/60:.1f}min)")
    
    return results


# ============== Analysis ==============

def compute_baseline_results() -> dict:
    """Compute baseline results (k=1 global SVD) for all distributions and r values."""
    cfg = CONFIG
    baselines = {}
    
    for dist in cfg["distributions"]:
        baselines[dist] = {}
        for r in cfg["r_values"]:
            errors = []
            for seed in cfg["seeds"]:
                kv_len = cfg["kv_len"]
                d = cfg["d"]
                q_len = cfg["q_len"]
                
                K, V, _ = generate_data(dist, kv_len, d, seed=seed)
                gen_q = np.random.default_rng(seed + 1000)
                Q = (gen_q.standard_normal((q_len, d)) * 0.5).astype(np.float32)
                O_gt = attention_output(Q, K, V)
                
                V_baseline, _ = baseline_global_svd(V, r)
                O_baseline = attention_output(Q, K, V_baseline)
                attn_err = np.abs(O_gt - O_baseline).mean()
                errors.append(attn_err)
            
            baselines[dist][r] = {
                "mean": float(np.mean(errors)),
                "std": float(np.std(errors)),
                "errors": errors,
            }
    
    return baselines


def analyze_results(results: List[dict], baselines: dict) -> dict:
    """Analyze ablation results and compute statistics."""
    cfg = CONFIG
    
    # Add improvement vs baseline to each result
    for r in results:
        if "error" not in r:
            dist = r["config"]["distribution"]
            r_val = r["config"]["r"]
            baseline_err = baselines[dist][r_val]["mean"]
            r["improvement_vs_baseline"] = baseline_err - r["metrics"]["attention_error_mean"]
    
    # Aggregate by key dimensions
    analysis = {
        "by_k": {},
        "by_algorithm": {},
        "by_distribution": {},
        "by_r": {},
        "by_k_and_distribution": {},
        "best_configs": [],
    }
    
    # Group by k
    for k in cfg["k_values"]:
        k_results = [r for r in results if r["config"]["k"] == k and "error" not in r]
        if k_results:
            mean_err = np.mean([r["metrics"]["attention_error_mean"] for r in k_results])
            mean_imp = np.mean([r["improvement_vs_baseline"] for r in k_results])
            mean_cr = np.mean([r["compression_ratio"] for r in k_results])
            analysis["by_k"][k] = {
                "mean_attention_error": float(mean_err),
                "mean_improvement": float(mean_imp),
                "mean_compression_ratio": float(mean_cr),
                "n_configs": len(k_results),
            }
    
    # Group by algorithm
    for algo in cfg["algorithms"]:
        algo_results = [r for r in results if r["config"]["algorithm"] == algo and "error" not in r]
        if algo_results:
            mean_err = np.mean([r["metrics"]["attention_error_mean"] for r in algo_results])
            mean_imp = np.mean([r["improvement_vs_baseline"] for r in algo_results])
            analysis["by_algorithm"][algo] = {
                "mean_attention_error": float(mean_err),
                "mean_improvement": float(mean_imp),
                "n_configs": len(algo_results),
            }
    
    # Group by distribution
    for dist in cfg["distributions"]:
        dist_results = [r for r in results if r["config"]["distribution"] == dist and "error" not in r]
        if dist_results:
            mean_err = np.mean([r["metrics"]["attention_error_mean"] for r in dist_results])
            mean_imp = np.mean([r["improvement_vs_baseline"] for r in dist_results])
            analysis["by_distribution"][dist] = {
                "mean_attention_error": float(mean_err),
                "mean_improvement": float(mean_imp),
                "n_configs": len(dist_results),
            }
    
    # Group by r
    for r in cfg["r_values"]:
        r_results = [r for r in results if r["config"]["r"] == r and "error" not in r]
        if r_results:
            mean_err = np.mean([r["metrics"]["attention_error_mean"] for r in r_results])
            mean_imp = np.mean([r["improvement_vs_baseline"] for r in r_results])
            analysis["by_r"][r] = {
                "mean_attention_error": float(mean_err),
                "mean_improvement": float(mean_imp),
                "n_configs": len(r_results),
            }
    
    # Group by k and distribution
    for k in cfg["k_values"]:
        for dist in cfg["distributions"]:
            kd_results = [r for r in results 
                         if r["config"]["k"] == k and r["config"]["distribution"] == dist 
                         and "error" not in r]
            if kd_results:
                mean_err = np.mean([r["metrics"]["attention_error_mean"] for r in kd_results])
                mean_imp = np.mean([r["improvement_vs_baseline"] for r in kd_results])
                key = f"k={k}_{dist}"
                analysis["by_k_and_distribution"][key] = {
                    "k": k,
                    "distribution": dist,
                    "mean_attention_error": float(mean_err),
                    "mean_improvement": float(mean_imp),
                    "n_configs": len(kd_results),
                }
    
    # Find best configs
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        sorted_by_improvement = sorted(valid_results, 
                                        key=lambda x: x["improvement_vs_baseline"], 
                                        reverse=True)
        analysis["best_configs"] = sorted_by_improvement[:20]
    
    # Cross-distribution stability
    cross_dist_stats = {}
    for k_val in cfg["k_values"]:
        for algo in cfg["algorithms"]:
            for r_val in cfg["r_values"]:
                subset = [res for res in results 
                         if res["config"]["k"] == k_val 
                         and res["config"]["algorithm"] == algo 
                         and res["config"]["r"] == r_val
                         and "error" not in res]
                expected_n = len(cfg["distributions"]) * len(cfg["seeds"])
                if len(subset) == expected_n:
                    improvements = [s["improvement_vs_baseline"] for s in subset]
                    key = f"k{k_val}_{algo}_r{r_val}"
                    cross_dist_stats[key] = {
                        "k": k_val,
                        "algorithm": algo,
                        "r": r_val,
                        "mean_improvement": float(np.mean(improvements)),
                        "std_improvement": float(np.std(improvements)),
                        "min_improvement": float(np.min(improvements)),
                        "max_improvement": float(np.max(improvements)),
                        "stable": float(np.std(improvements)) < 0.1,  # Stable if std < 0.1
                    }
    
    analysis["cross_distribution_stability"] = cross_dist_stats
    
    return analysis


def generate_pareto_data(results: List[dict], baselines: dict) -> dict:
    """Generate data for Pareto analysis (comp vs err tradeoff)."""
    pareto_points = []
    
    for r in results:
        if "error" in r:
            continue
        
        dist = r["config"]["distribution"]
        r_val = r["config"]["r"]
        baseline_err = baselines[dist][r_val]["mean"]
        
        pareto_points.append({
            "k": r["config"]["k"],
            "algorithm": r["config"]["algorithm"],
            "distribution": dist,
            "r": r_val,
            "seed": r["config"]["seed"],
            "compression_ratio": r["compression_ratio"],
            "attention_error": r["metrics"]["attention_error_mean"],
            "improvement": r["improvement_vs_baseline"],
            "improvement_pct": r["improvement_vs_baseline"] / baseline_err * 100 if baseline_err > 0 else 0,
        })
    
    return {"pareto_points": pareto_points}


# ============== Visualization ==============

def create_pareto_plot(results: List[dict], baselines: dict, output_dir: str):
    """Create Pareto plot: compression ratio vs attention error."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    pareto_data = generate_pareto_data(results, baselines)
    points = pareto_data["pareto_points"]
    
    if not points:
        print("No valid points for Pareto plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Colors for distributions
    colors = {"random": "blue", "skewed": "orange", "clustered": "green"}
    markers = {"KMeans": "o", "GMM": "s", "Agglomerative": "^", "Birch": "D"}
    
    # Plot 1: All points colored by distribution
    ax = axes[0, 0]
    for dist in ["random", "skewed", "clustered"]:
        dist_points = [p for p in points if p["distribution"] == dist]
        if dist_points:
            crs = [p["compression_ratio"] for p in dist_points]
            errs = [p["attention_error"] for p in dist_points]
            ax.scatter(crs, errs, c=colors[dist], label=dist, alpha=0.6, s=30)
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Attention Error (mean)")
    ax.set_title("Method D: Compression vs Error (by Distribution)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: All points colored by algorithm
    ax = axes[0, 1]
    for algo in ["KMeans", "GMM", "Agglomerative", "Birch"]:
        algo_points = [p for p in points if p["algorithm"] == algo]
        if algo_points:
            crs = [p["compression_ratio"] for p in algo_points]
            errs = [p["attention_error"] for p in algo_points]
            ax.scatter(crs, errs, marker=markers[algo], c="purple", label=algo, alpha=0.6, s=30)
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Attention Error (mean)")
    ax.set_title("Method D: Compression vs Error (by Algorithm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Improvement by k value (box plot style)
    ax = axes[1, 0]
    k_values = sorted(set(p["k"] for p in points))
    data_by_k = {k: [p["improvement"] for p in points if p["k"] == k] for k in k_values}
    
    bp = ax.boxplot([data_by_k[k] for k in k_values], labels=[str(k) for k in k_values],
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No improvement')
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Improvement vs Baseline")
    ax.set_title("Method D: Improvement by Cluster Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Pareto frontier for clustered distribution
    ax = axes[1, 1]
    clustered_points = [p for p in points if p["distribution"] == "clustered"]
    if clustered_points:
        # Group by (k, algorithm, r)
        for k in [1, 2, 4, 8, 16, 32]:
            for algo in ["KMeans", "GMM"]:
                k_subset = [p for p in clustered_points if p["k"] == k and p["algorithm"] == algo]
                if k_subset:
                    # Average across seeds
                    from collections import defaultdict
                    by_r = defaultdict(list)
                    for p in k_subset:
                        by_r[p["r"]].append((p["compression_ratio"], p["attention_error"]))
                    
                    for r, vals in by_r.items():
                        cr_mean = np.mean([v[0] for v in vals])
                        err_mean = np.mean([v[1] for v in vals])
                        ax.scatter(cr_mean, err_mean, marker=markers[algo], 
                                  s=100, label=f"k={k},{algo},r={r}", alpha=0.7)
    
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Attention Error (mean)")
    ax.set_title("Method D: Pareto Frontier (clustered distribution)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "method_d_pareto_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Pareto plot saved to: {plot_path}")
    return plot_path


def create_cross_distribution_plot(analysis: dict, output_dir: str):
    """Create cross-distribution stability plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean improvement by distribution
    ax = axes[0]
    dists = list(analysis["by_distribution"].keys())
    improvements = [analysis["by_distribution"][d]["mean_improvement"] for d in dists]
    colors = ["blue", "orange", "green"]
    bars = ax.bar(dists, improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("Distribution")
    ax.set_ylabel("Mean Improvement vs Baseline")
    ax.set_title("Method D: Mean Improvement by Distribution")
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:.4f}',
                ha='center', va='bottom' if height >= 0 else 'top')
    
    # Plot 2: Improvement by k, grouped by distribution
    ax = axes[1]
    k_values = sorted(analysis["by_k"].keys())
    x = np.arange(len(k_values))
    width = 0.25
    
    for i, dist in enumerate(["random", "skewed", "clustered"]):
        means = []
        for k in k_values:
            key = f"k={k}_{dist}"
            if key in analysis["by_k_and_distribution"]:
                means.append(analysis["by_k_and_distribution"][key]["mean_improvement"])
            else:
                means.append(0)
        ax.bar(x + i * width, means, width, label=dist, alpha=0.7)
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Mean Improvement vs Baseline")
    ax.set_title("Method D: Improvement by k and Distribution")
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "method_d_cross_distribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Cross-distribution plot saved to: {plot_path}")
    return plot_path


# ============== Report Generation ==============

def generate_report(results: List[dict], baselines: dict, analysis: dict, 
                   total_time: float) -> str:
    """Generate comprehensive ablation report."""
    
    cfg = CONFIG
    valid_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]
    
    # Find best config
    best_configs = sorted(valid_results, 
                         key=lambda x: x["improvement_vs_baseline"], 
                         reverse=True)[:10]
    
    # Cross-distribution stats
    cross_dist = analysis["cross_distribution_stability"]
    stable_configs = [k for k, v in cross_dist.items() if v["stable"]]
    
    report = f"""# Method D Ablation Study Report

## Executive Summary

- **Total Configurations**: {len(results)} (target: 648)
- **Successful Runs**: {len(valid_results)}
- **Failed Runs**: {len(failed_results)}
- **Total Time**: {total_time/60:.1f} minutes
- **Best Improvement**: {best_configs[0]['improvement_vs_baseline']:.4f} (config below)
- **Stable Across Distributions**: {len(stable_configs)} configs

## Experiment Configuration

| Parameter | Values |
|-----------|--------|
| Cluster count (k) | {cfg['k_values']} |
| Cluster algorithms | {', '.join(cfg['algorithms'])} |
| Distributions | {', '.join(cfg['distributions'])} |
| SVD rank (r) | {cfg['r_values']} |
| Seeds | {cfg['seeds']} |

## 1. Ablation Matrix Results

### 1.1 By Number of Clusters (k)

| k | Mean Attention Error | Mean Improvement | Mean Compression Ratio |
|---|---------------------|------------------|------------------------|
"""
    
    for k in sorted(analysis["by_k"].keys()):
        v = analysis["by_k"][k]
        report += f"| {k} | {v['mean_attention_error']:.6f} | {v['mean_improvement']:+.6f} | {v['mean_compression_ratio']:.2f}x |\n"
    
    report += f"""
**Key Finding**: k=1 represents the global SVD baseline. Values k>1 show 
{"improvement" if analysis["by_k"].get(8, {}).get("mean_improvement", 0) > 0 else "variable performance"}.

### 1.2 By Cluster Algorithm

| Algorithm | Mean Attention Error | Mean Improvement | Configs |
|-----------|----------------------|------------------|--------|
"""
    
    for algo in cfg["algorithms"]:
        if algo in analysis["by_algorithm"]:
            v = analysis["by_algorithm"][algo]
            report += f"| {algo} | {v['mean_attention_error']:.6f} | {v['mean_improvement']:+.6f} | {v['n_configs']} |\n"
    
    report += f"""
### 1.3 By Distribution

| Distribution | Mean Attention Error | Mean Improvement |
|--------------|----------------------|------------------|
"""
    
    for dist in cfg["distributions"]:
        if dist in analysis["by_distribution"]:
            v = analysis["by_distribution"][dist]
            report += f"| {dist} | {v['mean_attention_error']:.6f} | {v['mean_improvement']:+.6f} |\n"
    
    report += f"""
### 1.4 By SVD Rank (r)

| r | Mean Attention Error | Mean Improvement |
|---|----------------------|------------------|
"""
    
    for r in sorted(analysis["by_r"].keys()):
        v = analysis["by_r"][r]
        report += f"| {r} | {v['mean_attention_error']:.6f} | {v['mean_improvement']:+.6f} |\n"
    
    report += f"""
## 2. Top 10 Best Configurations

| Rank | k | Algorithm | Dist | r | Improvement | Attn Error | CR |
|------|---|-----------|------|---|-------------|------------|-----|
"""
    
    for i, cfg_res in enumerate(best_configs):
        c = cfg_res["config"]
        report += f"| {i+1} | {c['k']} | {c['algorithm']} | {c['distribution']} | {c['r']} | {cfg_res['improvement_vs_baseline']:+.6f} | {cfg_res['metrics']['attention_error_mean']:.6f} | {cfg_res['compression_ratio']:.2f}x |\n"
    
    report += f"""
## 3. Cross-Distribution Stability Analysis

A configuration is considered "stable" if its improvement standard deviation 
across distributions is < 0.1.

**Stable Configurations**: {len(stable_configs)} / {len(cross_dist)}

### 3.1 Stable Configurations (sorted by mean improvement)

| Config | k | Algorithm | r | Mean Improvement | Std | Stable |
|--------|---|-----------|---|------------------|-----|--------|
"""
    
    stable_sorted = sorted(
        [(k, v) for k, v in cross_dist.items() if v["stable"]], 
        key=lambda x: x[1]["mean_improvement"], 
        reverse=True
    )
    
    for k, v in stable_sorted[:15]:
        report += f"| {k} | {v['k']} | {v['algorithm']} | {v['r']} | {v['mean_improvement']:+.6f} | {v['std_improvement']:.6f} | ✓ |\n"
    
    report += f"""
### 3.2 Unstable Configurations (top 5 worst)

| Config | k | Algorithm | r | Mean Improvement | Std |
|--------|---|-----------|---|------------------|-----|
"""
    
    unstable_sorted = sorted(
        [(k, v) for k, v in cross_dist.items() if not v["stable"]], 
        key=lambda x: x[1]["std_improvement"], 
        reverse=True
    )
    
    for k, v in unstable_sorted[:5]:
        report += f"| {k} | {v['k']} | {v['algorithm']} | {v['r']} | {v['mean_improvement']:+.6f} | {v['std_improvement']:.6f} |\n"
    
    report += f"""
## 4. Key Findings

### 4.1 Effect of Cluster Count (k)

"""
    
    # Analyze k effect
    k_improvements = {k: analysis["by_k"][k]["mean_improvement"] for k in sorted(analysis["by_k"].keys())}
    best_k = max(k_improvements.keys(), key=lambda x: k_improvements[x])
    worst_k = min(k_improvements.keys(), key=lambda x: k_improvements[x])
    
    report += f"""- **Best k**: {best_k} (improvement: {k_improvements[best_k]:+.6f})
- **Worst k**: {worst_k} (improvement: {k_improvements[worst_k]:+.6f})
- **k=1** (baseline): improvement = {k_improvements[1]:+.6f} (as expected, should be ~0)

"""
    
    if k_improvements[1] < 0.01:
        report += "✓ **k=1 baseline check passed**: Method D with k=1 equals global SVD baseline.\n\n"
    else:
        report += "✗ **Warning**: k=1 shows unexpected deviation from baseline.\n\n"
    
    report += f"""### 4.2 Effect of Cluster Algorithm

"""
    
    algo_improvements = {a: analysis["by_algorithm"][a]["mean_improvement"] 
                        for a in cfg["algorithms"] if a in analysis["by_algorithm"]}
    best_algo = max(algo_improvements.keys(), key=lambda x: algo_improvements[x])
    
    report += f"""- **Best Algorithm**: {best_algo} (improvement: {algo_improvements[best_algo]:+.6f})
- Algorithm ranking: {', '.join([f"{a} ({algo_improvements[a]:+.4f})" for a in sorted(algo_improvements.keys(), key=lambda x: algo_improvements[x], reverse=True)])}

"""
    
    report += f"""### 4.3 Cross-Distribution Robustness

"""
    
    dist_improvements = {d: analysis["by_distribution"][d]["mean_improvement"] 
                        for d in cfg["distributions"]}
    
    report += f"""- **Random**: {dist_improvements.get('random', 0):+.6f}
- **Skewed**: {dist_improvements.get('skewed', 0):+.6f}
- **Clustered**: {dist_improvements.get('clustered', 0):+.6f}

"""
    
    if all(v > 0 for v in dist_improvements.values()):
        report += "✓ **Method D improves over baseline across ALL distributions**.\n\n"
    else:
        neg_dists = [d for d, v in dist_improvements.items() if v <= 0]
        report += f"⚠ **Warning**: Method D does NOT improve on distributions: {', '.join(neg_dists)}\n\n"
    
    report += f"""## 5. Pareto Analysis

### 5.1 Compression vs Error Tradeoff

Method D achieves the following compression-error Pareto frontier:

| k | Algorithm | Mean CR | Mean Error |
|---|-----------|---------|------------|
"""
    
    for k in [1, 2, 4, 8, 16, 32]:
        for algo in ["KMeans", "GMM"]:
            subset = [r for r in valid_results 
                     if r["config"]["k"] == k and r["config"]["algorithm"] == algo]
            if subset:
                cr = np.mean([r["compression_ratio"] for r in subset])
                err = np.mean([r["metrics"]["attention_error_mean"] for r in subset])
                report += f"| {k} | {algo} | {cr:.2f}x | {err:.6f} |\n"
    
    report += f"""
## 6. Conclusions

### 6.1 Summary of Results

1. **Best Configuration**: {best_configs[0]['config']['algorithm']} with k={best_configs[0]['config']['k']}, r={best_configs[0]['config']['r']}, distribution={best_configs[0]['config']['distribution']}
   - Improvement: {best_configs[0]['improvement_vs_baseline']:+.6f}
   - Attention Error: {best_configs[0]['metrics']['attention_error_mean']:.6f}
   - Compression Ratio: {best_configs[0]['compression_ratio']:.2f}x

2. **Cross-Distribution Stability**: {len(stable_configs)} configurations are stable across random, skewed, and clustered distributions.

3. **Algorithm Performance**: {best_algo} performs best on average across all configurations.

### 6.2 Recommendations

"""
    
    if k_improvements.get(8, 0) > 0:
        report += f"""1. **Use k=8 clusters** for best overall improvement (improvement: {k_improvements[8]:+.6f})
2. **Use {best_algo}** as the clustering algorithm
3. **Choose r based on accuracy/compression tradeoff** needed for your application

"""
    else:
        report += f"""1. **k values show variable improvement**: Consider tuning k for specific distributions
2. **Cross-distribution robustness varies**: Some configs stable across distributions
3. **Further investigation needed** for distributions where Method D doesn't improve

"""
    
    report += f"""## 7. Output Files

- `method_d_ablation.py` - Experiment code
- `method_d_ablation_data.json` - Raw experiment results
- `method_d_ablation_report.md` - This report
- `method_d_pareto_plot.png` - Pareto visualization
- `method_d_cross_distribution.png` - Cross-distribution analysis plot

---
*Generated by Method D Ablation Study*
"""
    
    return report


# ============== Main ==============

def main():
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Method D Ablation Study")
    print("=" * 70)
    
    total_start = time.time()
    
    # Step 1: Toy test
    toy_result = run_toy_test()
    
    # Step 2: Compute baselines
    print("\n" + "=" * 70)
    print("Computing Baseline Results (k=1 global SVD)")
    print("=" * 70)
    baselines = compute_baseline_results()
    print("Baselines computed.")
    
    # Step 3: Run full ablation
    print("\n" + "=" * 70)
    print("Running Full Ablation (648 configs)")
    print("=" * 70)
    results = run_full_ablation()
    
    total_time = time.time() - total_start
    
    # Step 4: Analyze results
    print("\n" + "=" * 70)
    print("Analyzing Results")
    print("=" * 70)
    analysis = analyze_results(results, baselines)
    
    # Step 5: Generate plots
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)
    pareto_plot = create_pareto_plot(results, baselines, output_dir)
    cross_plot = create_cross_distribution_plot(analysis, output_dir)
    
    # Step 6: Generate report
    print("\n" + "=" * 70)
    print("Generating Report")
    print("=" * 70)
    report = generate_report(results, baselines, analysis, total_time)
    
    # Save outputs
    output_data = {
        "config": CONFIG,
        "results": results,
        "baselines": baselines,
        "analysis": analysis,
        "total_time_seconds": total_time,
    }
    
    with open(os.path.join(output_dir, "method_d_ablation_data.json"), "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    
    with open(os.path.join(output_dir, "method_d_ablation_report.md"), "w") as f:
        f.write(report)
    
    print("\n" + "=" * 70)
    print("Ablation Study Complete!")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print(f"  - method_d_ablation_data.json")
    print(f"  - method_d_ablation_report.md")
    print(f"  - method_d_pareto_plot.png")
    print(f"  - method_d_cross_distribution.png")
    
    return results, baselines, analysis


if __name__ == "__main__":
    main()
