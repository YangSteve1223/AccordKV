"""
Method D Proof: Per-Cluster SVD Reconstruction Error Bound
===========================================================

Main experiment: 864 configurations covering:
- k ∈ {1, 2, 4, 8, 16, 32} clusters
- cov_type ∈ {isotropic, diagonal, full-cov}
- noise_std ∈ {0.01, 0.05, 0.1, 0.2}
- r ∈ {2, 4, 8, 16}
- seeds ∈ {42, 43, 44}

Total: 6 × 3 × 4 × 4 × 3 = 864 configs

Key Result: ratio = global_SVD_err / per_cluster_SVD_err
- ratio > 1: per-cluster SVD is better
- ratio < 1: global SVD is better
"""

import numpy as np
from scipy import linalg
import json
from typing import Dict, List, Tuple, Any
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_clustered_v(
    n: int, 
    d: int, 
    k: int, 
    cov_type: str,
    noise_std: float,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic clustered V matrix.
    
    Model: V_c = U_c + S_c + N_c
    - U_c: cluster center (deterministic structure)
    - S_c: within-cluster variation (structured)
    - N_c: Gaussian noise
    
    Args:
        n: number of vectors
        d: dimension  
        k: number of clusters
        cov_type: 'isotropic', 'diagonal', 'full-cov'
        noise_std: noise level
        seed: random seed
        
    Returns:
        V: (n, d) clustered V matrix
        cluster_labels: (n,) cluster assignment
        cluster_centers: (k, d) cluster centers
    """
    np.random.seed(seed)
    
    # Allocate
    V = np.zeros((n, d))
    cluster_labels = np.zeros(n, dtype=int)
    cluster_centers = np.zeros((k, d))
    
    # Per-cluster sizes (approximately equal)
    base_size = n // k
    sizes = [base_size] * k
    for i in range(n % k):
        sizes[i] += 1
    
    # Generate cluster centers (orthogonal-ish in principal subspace)
    for c in range(k):
        angle = 2 * np.pi * c / k
        center = np.zeros(d)
        
        # First few dimensions carry most information
        center[0] = np.cos(angle) * 5.0
        center[1] = np.sin(angle) * 5.0
        
        # Smaller components
        center[2:4] = np.random.randn(2) * 2.0
        
        # Add random variation in remaining dimensions
        center[4:] = np.random.randn(d - 4) * 0.5
        
        cluster_centers[c] = center
    
    # Generate V for each cluster
    start_idx = 0
    for c in range(k):
        size = sizes[c]
        end_idx = start_idx + size
        cluster_labels[start_idx:end_idx] = c
        
        # Generate cluster-specific V
        V_c = generate_cluster_v(
            size, d, cluster_centers[c], cov_type, seed + c + 1000, noise_std
        )
        V[start_idx:end_idx] = V_c
        
        start_idx = end_idx
    
    return V, cluster_labels, cluster_centers


def generate_cluster_v(
    n_c: int,
    d: int,
    center: np.ndarray,
    cov_type: str,
    seed: int,
    noise_std: float
) -> np.ndarray:
    """Generate V matrix for a single cluster."""
    np.random.seed(seed)
    
    V_c = np.zeros((n_c, d))
    
    # Cluster center (signal)
    V_c += center
    
    # Within-cluster variation (structured noise / cluster-specific pattern)
    if cov_type == 'isotropic':
        # Spherical covariance
        variation = np.random.randn(n_c, d) * 2.0
    elif cov_type == 'diagonal':
        # Diagonal covariance with different scales per dimension
        scales = np.array([3.0, 2.5, 2.0, 1.5] + [0.5] * max(0, d - 4))
        variation = np.random.randn(n_c, d) * scales[:d]
    else:  # full-cov
        # Full covariance matrix (correlated dimensions)
        # Create a random positive definite matrix
        d_half = max(1, d // 2)
        A = np.random.randn(d, d_half)
        cov = A @ A.T + np.eye(d) * 0.1
        try:
            L = np.linalg.cholesky(cov)
        except:
            # Fallback to isotropic if Cholesky fails
            L = np.eye(d) * 2.0
        # L is (d, d), random samples (d, n_c), result is (d, n_c)
        variation = (L @ np.random.randn(d, n_c)).T
    
    V_c += variation
    
    # Add Gaussian noise
    if noise_std > 0:
        V_c += np.random.randn(n_c, d) * noise_std
    
    return V_c


def global_svd_reconstruction(V: np.ndarray, r: int) -> Tuple[np.ndarray, float]:
    """
    Global SVD reconstruction of V.
    
    Returns:
        V_approx: reconstructed V
        recon_error: Frobenius norm of reconstruction error
    """
    # Full SVD
    U, s, Vt = np.linalg.svd(V, full_matrices=False)
    
    # Keep top-r components
    r_actual = min(r, len(s))
    U_r = U[:, :r_actual]
    s_r = s[:r_actual]
    Vt_r = Vt[:r_actual, :]
    
    # Reconstruct
    V_approx = U_r @ np.diag(s_r) @ Vt_r
    
    # Error = sum of squares of discarded singular values
    recon_error = np.sqrt(np.sum(s[r_actual:] ** 2)) if r_actual < len(s) else 0.0
    
    return V_approx, recon_error


def per_cluster_svd_reconstruction(
    V: np.ndarray,
    cluster_labels: np.ndarray,
    k: int,
    r: int
) -> Tuple[np.ndarray, float]:
    """
    Per-cluster SVD reconstruction.
    
    For each cluster c:
        V_c = V[cluster_labels == c]
        V_c_approx = SVD_c(V_c) with rank r
        
    Returns:
        V_approx: reconstructed V (stacked)
        recon_error: sum of per-cluster reconstruction errors
    """
    n = V.shape[0]
    V_approx = np.zeros_like(V)
    total_error = 0.0
    
    for c in range(k):
        mask = cluster_labels == c
        n_c = np.sum(mask)
        if n_c == 0:
            continue
            
        V_c = V[mask]
        
        # Per-cluster SVD
        U_c, s_c, Vt_c = np.linalg.svd(V_c, full_matrices=False)
        
        # Number of singular values to keep for this cluster
        r_c = min(r, len(s_c))
        
        # Reconstruct
        V_c_approx = U_c[:, :r_c] @ np.diag(s_c[:r_c]) @ Vt_c[:r_c, :]
        V_approx[mask] = V_c_approx
        
        # Error for this cluster
        if r_c < len(s_c):
            error_c = np.sqrt(np.sum(s_c[r_c:] ** 2))
        else:
            error_c = 0.0
        total_error += error_c
    
    return V_approx, total_error


def compute_frobenius_error(V: np.ndarray, V_approx: np.ndarray) -> float:
    """Compute Frobenius norm of reconstruction error."""
    return np.linalg.norm(V - V_approx, 'fro')


def spectral_bound_theory(
    V: np.ndarray,
    cluster_labels: np.ndarray,
    k: int,
    r: int
) -> Dict[str, Any]:
    """
    Compute theoretical bounds based on spectral analysis.
    
    Key inequality:
        Σ_c σ_{r+1}(V_c) ≤ σ_{r+1}(V)
        
    This holds when V has block-diagonal structure.
    """
    # Global singular values
    s_global = np.linalg.svd(V, compute_uv=False)
    sigma_r1_global = s_global[r] if r < len(s_global) else 0.0
    
    # Per-cluster singular values
    sum_sigma_r1 = 0.0
    cluster_sigmas = []
    
    for c in range(k):
        mask = cluster_labels == c
        V_c = V[mask]
        
        if V_c.shape[0] < 2:
            continue
            
        s_c = np.linalg.svd(V_c, compute_uv=False)
        sigma_r1_c = s_c[r] if r < len(s_c) else 0.0
        sum_sigma_r1 += sigma_r1_c
        cluster_sigmas.append(float(sigma_r1_c))
    
    # Theoretical ratio
    ratio_bound = sum_sigma_r1 / (sigma_r1_global + 1e-10)
    
    return {
        'sigma_r1_global': float(sigma_r1_global),
        'sum_sigma_r1_per_cluster': float(sum_sigma_r1),
        'ratio_bound': float(ratio_bound),
        'cluster_sigmas': cluster_sigmas
    }


def run_single_experiment(
    n: int,
    d: int,
    k: int,
    cov_type: str,
    noise_std: float,
    r: int,
    seed: int
) -> Dict[str, Any]:
    """Run a single experimental configuration."""
    start_time = time.time()
    
    # Generate clustered V
    V, cluster_labels, cluster_centers = generate_clustered_v(
        n, d, k, cov_type, noise_std, seed
    )
    
    # Global SVD
    V_approx_global, theoretical_err_global = global_svd_reconstruction(V, r)
    actual_err_global = compute_frobenius_error(V, V_approx_global)
    
    # Per-cluster SVD
    V_approx_pc, theoretical_err_pc = per_cluster_svd_reconstruction(
        V, cluster_labels, k, r
    )
    actual_err_pc = compute_frobenius_error(V, V_approx_pc)
    
    # Theoretical bounds
    theory = spectral_bound_theory(V, cluster_labels, k, r)
    
    elapsed = time.time() - start_time
    
    # Compute ratio (per-cluster / global)
    # ratio > 1 means per-cluster is better (smaller error)
    ratio = actual_err_global / (actual_err_pc + 1e-10)
    
    return {
        'config': {
            'n': n, 'd': d, 'k': int(k), 'cov_type': cov_type,
            'noise_std': float(noise_std), 'r': int(r), 'seed': int(seed)
        },
        'global_svd': {
            'theoretical_error': float(theoretical_err_global),
            'actual_error': float(actual_err_global)
        },
        'per_cluster_svd': {
            'theoretical_error': float(theoretical_err_pc),
            'actual_error': float(actual_err_pc)
        },
        'ratio': float(ratio),
        'theory': theory,
        'runtime': float(elapsed)
    }


def run_full_experiment(
    n: int = 4096,
    d: int = 128,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the full 864-config experiment.
    
    Experimental design:
    - k ∈ {1, 2, 4, 8, 16, 32}
    - cov_type ∈ {isotropic, diagonal, full-cov}
    - noise_std ∈ {0.01, 0.05, 0.1, 0.2}
    - r ∈ {2, 4, 8, 16}
    - seeds ∈ {42, 43, 44}
    """
    # Configuration grid
    k_values = [1, 2, 4, 8, 16, 32]
    cov_types = ['isotropic', 'diagonal', 'full-cov']
    noise_stds = [0.01, 0.05, 0.1, 0.2]
    r_values = [2, 4, 8, 16]
    seeds = [42, 43, 44]
    
    # Generate all configurations
    all_configs = []
    for k in k_values:
        for cov_type in cov_types:
            for noise_std in noise_stds:
                for r in r_values:
                    for seed in seeds:
                        all_configs.append({
                            'n': n, 'd': d, 'k': k, 'cov_type': cov_type,
                            'noise_std': noise_std, 'r': r, 'seed': seed
                        })
    
    total_configs = len(all_configs)
    
    if verbose:
        print("=" * 60)
        print("Method D Proof: Full Experiment")
        print("=" * 60)
        print(f"Total configurations: {total_configs}")
        print(f"Parameters: k={k_values}, cov={cov_types}, noise={noise_stds}, r={r_values}, seeds={seeds}")
        print("-" * 60)
    
    results = []
    start_total = time.time()
    
    for i, config in enumerate(all_configs):
        result = run_single_experiment(**config)
        results.append(result)
        
        if verbose and ((i + 1) % 50 == 0 or i == 0):
            elapsed = time.time() - start_total
            rate = (i + 1) / elapsed
            remaining = (total_configs - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{total_configs} ({100*(i+1)/total_configs:.1f}%) | "
                  f"Rate: {rate:.1f} configs/s | ETA: {remaining:.0f}s")
    
    total_time = time.time() - start_total
    
    # Aggregate statistics
    ratios = [r['ratio'] for r in results]
    global_errors = [r['global_svd']['actual_error'] for r in results]
    pc_errors = [r['per_cluster_svd']['actual_error'] for r in results]
    
    # Compute statistics by k
    stats_by_k = {}
    for k in k_values:
        k_results = [r for r in results if r['config']['k'] == k]
        k_ratios = [r['ratio'] for r in k_results]
        stats_by_k[k] = {
            'count': len(k_ratios),
            'ratio_mean': float(np.mean(k_ratios)),
            'ratio_std': float(np.std(k_ratios)),
            'ratio_min': float(np.min(k_ratios)),
            'ratio_max': float(np.max(k_ratios)),
            'global_err_mean': float(np.mean([r['global_svd']['actual_error'] for r in k_results])),
            'pc_err_mean': float(np.mean([r['per_cluster_svd']['actual_error'] for r in k_results])),
        }
    
    # Compute statistics by cov_type
    stats_by_cov = {}
    for cov_type in cov_types:
        cov_results = [r for r in results if r['config']['cov_type'] == cov_type]
        cov_ratios = [r['ratio'] for r in cov_results]
        stats_by_cov[cov_type] = {
            'count': len(cov_ratios),
            'ratio_mean': float(np.mean(cov_ratios)),
            'ratio_std': float(np.std(cov_ratios)),
            'ratio_min': float(np.min(cov_ratios)),
            'ratio_max': float(np.max(cov_ratios)),
        }
    
    # Compute statistics by noise_std
    stats_by_noise = {}
    for noise_std in noise_stds:
        noise_results = [r for r in results if r['config']['noise_std'] == noise_std]
        noise_ratios = [r['ratio'] for r in noise_results]
        stats_by_noise[noise_std] = {
            'count': len(noise_ratios),
            'ratio_mean': float(np.mean(noise_ratios)),
            'ratio_std': float(np.std(noise_ratios)),
        }
    
    # Compute statistics by r
    stats_by_r = {}
    for r in r_values:
        r_results = [r for r in results if r['config']['r'] == r]
        r_ratios = [r['ratio'] for r in r_results]
        stats_by_r[r] = {
            'count': len(r_ratios),
            'ratio_mean': float(np.mean(r_ratios)),
            'ratio_std': float(np.std(r_ratios)),
        }
    
    summary = {
        'total_configs': total_configs,
        'total_time_seconds': float(total_time),
        'overall': {
            'ratio_mean': float(np.mean(ratios)),
            'ratio_std': float(np.std(ratios)),
            'ratio_min': float(np.min(ratios)),
            'ratio_max': float(np.max(ratios)),
            'global_err_mean': float(np.mean(global_errors)),
            'pc_err_mean': float(np.mean(pc_errors)),
            'win_rate': float(np.sum([r > 1.0 for r in ratios]) / len(ratios)),
        },
        'by_k': stats_by_k,
        'by_cov_type': stats_by_cov,
        'by_noise_std': stats_by_noise,
        'by_r': stats_by_r,
    }
    
    return {
        'summary': summary,
        'results': results
    }


def run_toy_test() -> Dict[str, Any]:
    """
    Run a single toy test configuration for validation.
    k=2, σ=0.05, r=4
    """
    print("\n" + "=" * 60)
    print("Toy Test: k=2, noise_std=0.05, r=4")
    print("=" * 60)
    
    result = run_single_experiment(
        n=4096, d=128, k=2, cov_type='diagonal',
        noise_std=0.05, r=4, seed=42
    )
    
    print(f"\nGlobal SVD error: {result['global_svd']['actual_error']:.4f}")
    print(f"Per-cluster SVD error: {result['per_cluster_svd']['actual_error']:.4f}")
    print(f"Ratio (global/pc): {result['ratio']:.4f}")
    print(f"Theory ratio bound: {result['theory']['ratio_bound']:.4f}")
    print(f"\nPer-cluster singular values (σ_{result['config']['r']+1}):")
    for i, sigma in enumerate(result['theory']['cluster_sigmas']):
        print(f"  Cluster {i}: {sigma:.4f}")
    print(f"Global σ_{result['config']['r']+1}: {result['theory']['sigma_r1_global']:.4f}")
    print(f"Sum per-cluster: {result['theory']['sum_sigma_r1_per_cluster']:.4f}")
    
    # Expected: ratio > 1 (per-cluster should be better for k=2)
    if result['ratio'] > 1.0:
        print("\n✓ PASS: Per-cluster SVD outperforms global SVD (ratio > 1)")
    else:
        print("\n✗ FAIL: Per-cluster SVD does NOT outperform global SVD (ratio <= 1)")
    
    return result


if __name__ == '__main__':
    # First, run toy test
    toy_result = run_toy_test()
    
    # Then run full experiment if requested
    if '--full' in sys.argv:
        print("\n" + "=" * 60)
        print("Starting Full Experiment (864 configs)")
        print("=" * 60)
        
        data = run_full_experiment(n=4096, d=128, verbose=True)
        
        # Save results
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'method_d_proof_data.json')
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        summary = data['summary']
        print(f"Total configs: {summary['total_configs']}")
        print(f"Total time: {summary['total_time_seconds']:.1f}s")
        print(f"\nOverall ratio (global/pc):")
        print(f"  Mean: {summary['overall']['ratio_mean']:.4f}")
        print(f"  Std:  {summary['overall']['ratio_std']:.4f}")
        print(f"  Min:  {summary['overall']['ratio_min']:.4f}")
        print(f"  Max:  {summary['overall']['ratio_max']:.4f}")
        print(f"  Win rate (ratio > 1): {summary['overall']['win_rate']:.1%}")
        
        print(f"\nResults saved to: {output_path}")
