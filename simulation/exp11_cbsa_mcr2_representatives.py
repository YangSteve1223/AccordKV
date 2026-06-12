"""
Exp11: CBSA-style MCR² Representative Token Selection
- Head-to-head comparison: CBSA (MCR²-optimized) vs k-means Coreset

Based on CBSA paper (NeurIPS 2025, arXiv:2509.16875):
- MCR² objective maximizes coding rate reduction to learn compact representations
- Representatives are optimized via gradient descent, not just geometric clustering
- Key formula: R(F, Π) = 0.5 * log det(I + α * F^T * F * Π) - γ * log det(I + β * F^T * F)

Key insight from paper:
- When m = n (all tokens), error should = 0
- On clustered data, CBSA should outperform k-means by 5-15%
- On random data, CBSA and k-means should perform similarly
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    ground_truth,
)

# ============== MCR² Objective (Numpy Implementation) ==============

class MCR2Objective:
    """
    MCR² (Maximal Coding Rate Reduction) objective for representative selection.
    
    Paper: Yu et al. "Learning Diverse and Discriminative Representations via 
           the Principle of Maximal Coding Rate Reduction" (NeurIPS 2020)
    
    Simplified for our case:
    - F = K (key vectors) [n, d]
    - We want to select m representative columns
    - Π = I_m (identity for representative weights)
    
    Coding rate: R(Z) = 0.5 * log det(I + (d/Nε²) * Z^T * Z)
    
    MCR² = R(R) - γ * R(F)
    where R(F) is the coding rate of full tokens and R(R) is coding rate of representatives
    """
    
    def __init__(
        self,
        F: np.ndarray,
        m: int,
        alpha: float = 1.0,
        gamma: float = 0.5,
        eps: float = 1e-8,
    ):
        """
        Parameters
        ----------
        F : [n, d] - key vectors (tokens)
        m : int - number of representatives to select
        alpha : float - expansion rate
        gamma : float - compression penalty
        eps : float - numerical stability
        """
        self.F = F.astype(np.float64)  # Use float64 for numerical stability
        self.n, self.d = F.shape
        self.m = m
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        
        # Compute covariance matrix of full tokens: C_full = F^T * F / n
        self.C_full = (F.T @ F) / self.n
    
    def coding_rate(self, M: np.ndarray, scale: float = 1.0) -> float:
        """
        Compute R(M) = 0.5 * log det(I + scale * M)
        
        Uses slogdet for numerical stability.
        """
        identity = np.eye(M.shape[0])
        mat = identity + scale * M
        sign, logdet = np.linalg.slogdet(mat)
        if sign <= 0:
            # Matrix not positive definite, return -inf
            return -1e10
        return 0.5 * logdet
    
    def mcr2(self, representatives: np.ndarray) -> float:
        """
        Compute MCR² objective for given representatives.
        
        representatives : [m, d] - current representative vectors
        
        Returns MCR² value (higher is better).
        """
        m, d = representatives.shape
        if m == 0:
            return -1e10
        
        # Compute covariance of representatives: C_r = R^T * R / m
        C_r = (representatives.T @ representatives) / m
        
        # Coding rate of representatives
        # Scale factor accounts for ratio of representative count to full count
        scale_r = self.alpha * m / self.n
        
        # Coding rate of full tokens
        scale_full = self.alpha
        
        R_r = self.coding_rate(C_r, scale_r)
        R_full = self.coding_rate(self.C_full, scale_full)
        
        # MCR² = R_r - γ * R_full
        mcr2_val = R_r - self.gamma * R_full
        return mcr2_val
    
    def mcr2_gradient(self, representatives: np.ndarray) -> np.ndarray:
        """
        Compute gradient of -MCR² with respect to representatives.
        
        For MCR² = 0.5 * log det(I + scale * R @ R^T) where R is [m, d]:
        
        Using the chain rule:
        d(log det(M))/dR = d(log det(M))/dM * dM/dR
        d(log det(M))/dM = M^{-T} (gradient w.r.t symmetric M)
        dM/dR = s * (dR @ R^T + R @ dR^T) (but for gradient wrt R elements)
        
        The element-wise gradient is:
        dMCR²/dR_{ij} = s * R_{ij} * (M^{-1})_{ii} (approximately)
        
        More precisely: gradient = s * R @ (I + s * R @ R^T)^{-T}
        """
        representatives = representatives.astype(np.float64)
        m = representatives.shape[0]
        
        if m == 0:
            return np.zeros((1, self.d), dtype=np.float32)
        
        # Scale factor
        scale = self.alpha * m / self.n
        
        # R @ R^T has shape [m, m]
        RRt = representatives @ representatives.T
        
        # Compute (I + scale * R @ R^T)
        M = np.eye(m, dtype=np.float64) + scale * RRt
        
        try:
            # Use Cholesky if possible (faster and more stable)
            L = np.linalg.cholesky(M + self.eps * np.eye(m, dtype=np.float64))
            # Solve L @ L.T @ X = I => X = L^{-T} @ L^{-1}
            # First solve L @ Y = I, then L.T @ X = Y
            Y = np.linalg.solve(L, np.eye(m, dtype=np.float64))
            M_inv = np.linalg.solve(L.T, Y)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M + self.eps * np.eye(m, dtype=np.float64))
        
        # Gradient: dMCR²/dR = scale * M^{-1} @ R
        grad = scale * (M_inv @ representatives)
        
        # Return negative for gradient descent on -MCR²
        return -grad.astype(np.float32)


# ============== CBSA Representative Selection ==============

@dataclass
class CBSARepresentatives:
    """CBSA representative tokens container."""
    representatives: np.ndarray  # [m, d] - learned representatives
    mcr2_history: list[float]     # Training loss curve
    final_mcr2: float


def initialize_representatives(
    K: np.ndarray,
    m: int,
    method: str = "kmeans++",
    seed: int = 0,
) -> np.ndarray:
    """
    Initialize m representative tokens from K.
    
    Methods:
    - "random": Random selection
    - "kmeans++": K-Means++ initialization (better)
    - "uniform": Uniform sampling
    """
    n, d = K.shape
    gen = np.random.default_rng(seed)
    
    if method == "random":
        idx = gen.choice(n, size=m, replace=False)
        return K[idx].copy().astype(np.float32)
    
    elif method == "uniform":
        # Uniform sampling
        idx = np.linspace(0, n - 1, m, dtype=int)
        return K[idx].copy().astype(np.float32)
    
    elif method == "kmeans++":
        # K-Means++ initialization
        # 1. Pick first centroid randomly
        idx0 = gen.integers(0, n)
        centroids = [K[idx0].copy()]
        
        # 2. Iteratively pick remaining centroids
        for _ in range(m - 1):
            dists = np.array([
                min(np.linalg.norm(k - c) ** 2 for c in centroids)
                for k in K
            ])
            probs = dists / (dists.sum() + 1e-10)
            idx = gen.choice(n, p=probs)
            centroids.append(K[idx].copy())
        
        return np.array(centroids).astype(np.float32)
    
    else:
        raise ValueError(f"Unknown init method: {method}")


def train_cbsa_representatives(
    K: np.ndarray,
    V: np.ndarray,
    m: int,
    num_steps: int = 30,  # Reduced from 50 for speed
    lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    seed: int = 0,
    verbose: bool = True,
    log_every: int = 10,  # Increased for less output
) -> CBSARepresentatives:
    """
    Train CBSA representatives using gradient ascent on MCR² objective.
    
    Uses Adam optimizer for stable training.
    
    Parameters
    ----------
    K : [n, d] - key vectors
    V : [n, d] - value vectors (used for evaluation only)
    m : int - number of representatives
    num_steps : int - training iterations
    lr : float - learning rate
    beta1, beta2 : Adam hyperparameters
    seed : int - random seed
    
    Returns
    -------
    CBSARepresentatives with trained representatives
    """
    n, d = K.shape
    
    # Initialize representatives
    representatives = initialize_representatives(K, m, method="kmeans++", seed=seed)
    representatives = representatives.astype(np.float64)
    
    # MCR² objective
    mcr2_obj = MCR2Objective(K, m, alpha=1.0, gamma=0.5)
    
    # Adam optimizer state
    m_mt = np.zeros_like(representatives)  # First moment
    v_mt = np.zeros_like(representatives)  # Second moment
    
    mcr2_history = []
    
    for step in range(num_steps):
        # Compute MCR² (higher is better)
        current_mcr2 = mcr2_obj.mcr2(representatives.astype(np.float64))
        mcr2_history.append(float(current_mcr2))
        
        # Compute gradient of -MCR² (we want to maximize MCR²)
        grad = mcr2_obj.mcr2_gradient(representatives.astype(np.float64))
        
        # Adam update
        t = step + 1
        m_mt = beta1 * m_mt + (1 - beta1) * grad
        v_mt = beta2 * v_mt + (1 - beta2) * (grad ** 2)
        
        # Bias correction
        m_hat = m_mt / (1 - beta1 ** t)
        v_hat = v_mt / (1 - beta2 ** t)
        
        # Update
        representatives = representatives - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Verbose logging
        if verbose and (step % log_every == 0 or step == num_steps - 1):
            loss = -current_mcr2
            print(f"    Step {step:>3}: MCR²={current_mcr2:.6f}, Loss={loss:.6f}")
    
    return CBSARepresentatives(
        representatives=representatives.astype(np.float32),
        mcr2_history=mcr2_history,
        final_mcr2=mcr2_history[-1] if mcr2_history else 0.0,
    )


def eval_cbsa_representatives(
    representatives: np.ndarray,
    V: np.ndarray,
    Q: np.ndarray,
    K: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> NumpyAttnStats:
    """
    Evaluate attention output using CBSA representatives.
    
    Uses weighted attention with representatives:
    F(q) = Σ_i w_i * v_i * exp(q · r_i) / Σ_i w_i * exp(q · r_i)
    
    where r_i are representatives and w_i are learned weights.
    """
    m, d = representatives.shape
    q_len = Q.shape[0]
    
    # If no weights provided, use uniform weights
    if weights is None:
        weights = np.ones(m) / m
    
    # Compute attention scores: [q_len, m]
    scores = Q @ representatives.T  # [q_len, m]
    scores = scores + np.log(weights + 1e-12)
    
    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - scores_max)  # [q_len, m]
    l = p.sum(axis=-1, keepdims=True)  # [q_len, 1]
    p_norm = p / np.clip(l, 1e-30, None)  # [q_len, m]
    
    # For output, we need to aggregate V values
    # Simple approach: use representative V as cluster centers
    # More sophisticated: compute weighted V per cluster
    if m <= K.shape[0]:
        # Map each representative to nearest K token for V
        dists = np.linalg.norm(K[:, None, :] - representatives[None, :, :], axis=-1)  # [n, m]
        assignments = dists.argmin(axis=1)  # [n]
        
        # Compute cluster-weighted V
        V_agg = np.zeros((m, d), dtype=np.float32)
        for j in range(m):
            mask = assignments == j
            if mask.sum() > 0:
                V_agg[j] = V[mask].mean(axis=0)
            else:
                V_agg[j] = V[dists[:, j].argmin()]
    else:
        V_agg = V[:m]
    
    # Compute output
    F = p_norm @ V_agg  # [q_len, d]
    
    # Package as NumpyAttnStats
    H = 1
    m_out = scores_max  # [q_len, 1]
    l_out = l  # [q_len, 1]
    y_out = F * l_out  # [q_len, d]
    
    return NumpyAttnStats(
        m=m_out[None, :, :],
        l=l_out[None, :, :],
        y=y_out[None, :, :],
    )


# ============== K-Means Coreset (Baseline) ==============

def build_kmeans_coreset(
    K: np.ndarray,
    V: np.ndarray,
    m: int,
    seed: int = 0,
    num_iters: int = 10,  # Reduced from 15 for speed
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build k-means coreset (ACCORD baseline).
    
    Returns centroids_K, centroids_V, weights
    """
    n, d = K.shape
    gen = np.random.default_rng(seed)
    
    # K-Means++ initialization
    idx = gen.integers(0, n)
    centroids_K = [K[idx].copy()]
    centroids_V = [V[idx].copy()]
    
    for _ in range(m - 1):
        dists = np.array([
            min(np.linalg.norm(k - c) ** 2 for c in centroids_K)
            for k in K
        ])
        probs = dists / (dists.sum() + 1e-10)
        idx = gen.choice(n, p=probs)
        centroids_K.append(K[idx].copy())
        centroids_V.append(V[idx].copy())
    
    centroids_K = np.array(centroids_K)
    centroids_V = np.array(centroids_V)
    
    # Lloyd iterations
    for _ in range(num_iters):
        # E-step
        dists = np.zeros((n, m))
        for j in range(m):
            dists[:, j] = np.sum((K - centroids_K[j]) ** 2, axis=1)
        assignments = dists.argmin(axis=1)
        
        # M-step
        new_K = np.zeros_like(centroids_K)
        new_V = np.zeros_like(centroids_V)
        for j in range(m):
            mask = assignments == j
            count = mask.sum()
            if count > 0:
                new_K[j] = K[mask].mean(axis=0)
                new_V[j] = V[mask].mean(axis=0)
            else:
                new_K[j] = centroids_K[j]
                new_V[j] = centroids_V[j]
        
        centroids_K = new_K
        centroids_V = new_V
    
    # Compute weights
    weights = np.zeros(m)
    dists_final = np.zeros((n, m))
    for j in range(m):
        dists_final[:, j] = np.sum((K - centroids_K[j]) ** 2, axis=1)
    final_assign = dists_final.argmin(axis=1)
    for j in range(m):
        weights[j] = (final_assign == j).sum() / n
    weights = weights / (weights.sum() + 1e-10)
    
    return centroids_K, centroids_V, weights


def eval_kmeans_coreset(
    centroids_K: np.ndarray,
    centroids_V: np.ndarray,
    weights: np.ndarray,
    Q: np.ndarray,
) -> NumpyAttnStats:
    """Evaluate k-means coreset."""
    m, d = centroids_K.shape
    q_len = Q.shape[0]
    
    # Scores
    scores = Q @ centroids_K.T  # [q_len, m]
    scores = scores + np.log(weights + 1e-12)
    
    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - scores_max)
    l = p.sum(axis=-1, keepdims=True)
    p_norm = p / np.clip(l, 1e-30, None)
    
    # Output
    F = p_norm @ centroids_V
    y_out = F * l
    
    return NumpyAttnStats(
        m=scores_max[None, :, :],
        l=l[None, :, :],
        y=y_out[None, :, :],
    )


# ============== Data Generation ==============

def make_clustered_kv(
    kv_len: int,
    d: int,
    n_clusters: int = 8,
    cluster_std: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clustered KV data (CBSA should excel here)."""
    gen = np.random.default_rng(seed)
    
    # Generate cluster centers (well-separated)
    centroids = []
    for _ in range(n_clusters):
        for _ in range(100):
            c = gen.standard_normal(d) * 2.0
            if all(np.linalg.norm(c - oc) > 3.0 for oc in centroids):
                centroids.append(c)
                break
        if len(centroids) <= len(centroids):
            break
    while len(centroids) < n_clusters:
        centroids.append(gen.standard_normal(d) * 2.0)
    centroids = np.array(centroids)
    
    # Assign tokens to clusters
    cluster_assign = gen.integers(0, n_clusters, size=kv_len)
    
    # Generate K
    K = np.zeros((kv_len, d), dtype=np.float32)
    for i in range(kv_len):
        K[i] = centroids[cluster_assign[i]] + gen.standard_normal(d) * cluster_std
    
    # V related to K (linear transform + noise)
    W = gen.standard_normal((d, d)) * 0.3
    V = K @ W + gen.standard_normal((kv_len, d)) * 0.1
    
    return K.astype(np.float32), V.astype(np.float32)


def make_random_kv(
    kv_len: int,
    d: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random KV (no structure - CBSA and k-means should be similar)."""
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((kv_len, d)).astype(np.float32) * 0.5
    V = gen.standard_normal((kv_len, d)).astype(np.float32) * 0.5
    return K, V


def make_skewed_kv(
    kv_len: int,
    d: int,
    n_outliers: int = 16,
    outlier_std: float = 3.0,
    normal_std: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate skewed KV (few outliers + many normal tokens)."""
    gen = np.random.default_rng(seed)
    
    # Outliers (spread widely)
    outlier_K = gen.standard_normal((n_outliers, d)) * outlier_std
    outlier_V = gen.standard_normal((n_outliers, d)) * outlier_std
    
    # Normal tokens (clustered near origin)
    normal_K = gen.standard_normal((kv_len - n_outliers, d)) * normal_std
    normal_V = gen.standard_normal((kv_len - n_outliers, d)) * normal_std
    
    # Concatenate and shuffle
    K = np.concatenate([outlier_K, normal_K], axis=0)
    V = np.concatenate([outlier_V, normal_V], axis=0)
    
    perm = gen.permutation(kv_len)
    return K[perm].astype(np.float32), V[perm].astype(np.float32)


# ============== Experiment Runners ==============

def run_mcr2_training_curve(
    kv_len: int = 2048,  # Reduced from 4096 for speed
    m: int = 32,
    d: int = 128,
    num_steps: int = 30,  # Reduced from 50 for speed
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Run MCR² training curve experiment.
    
    Verifies that -MCR² loss decreases over training.
    """
    # Generate data
    K, V = make_clustered_kv(kv_len, d, n_clusters=8, seed=seed)
    
    # Train CBSA representatives
    cbsa = train_cbsa_representatives(
        K, V, m,
        num_steps=num_steps,
        lr=1e-2,
        seed=seed,
        verbose=verbose,
    )
    
    return {
        "kv_len": kv_len,
        "m": m,
        "d": d,
        "num_steps": num_steps,
        "mcr2_history": cbsa.mcr2_history,
        "final_mcr2": cbsa.final_mcr2,
        "initial_mcr2": cbsa.mcr2_history[0] if cbsa.mcr2_history else 0.0,
    }


def run_physical_consistency_check(
    kv_len: int = 1024,
    d: int = 128,
    m_values: list[int] = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Physical consistency check:
    - When m = n (all tokens), error should = 0
    - Training should be stable
    """
    if m_values is None:
        m_values = [4, 16, 64, 256, 1024]
    
    K, V = make_clustered_kv(kv_len, d, n_clusters=8, seed=seed)
    Q = (np.random.default_rng(seed + 1000).standard_normal((64, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K, V)
    
    results = {"checks": []}
    
    for m in m_values:
        if m > kv_len:
            m = kv_len
        
        # Train CBSA
        cbsa = train_cbsa_representatives(
            K, V, m,
            num_steps=30,
            lr=1e-2,
            seed=seed,
            verbose=False,
        )
        
        # Evaluate
        out = eval_cbsa_representatives(cbsa.representatives, V, Q, K)
        out = out.finalize().squeeze(0)
        
        l1_err = float(np.abs(out - gt).mean())
        l2_err = float(np.sqrt(np.mean((out - gt) ** 2)))
        
        # Output error
        out_v = out @ np.eye(d)  # Simplified: just compare outputs
        gt_v = gt @ np.eye(d)
        output_err = float(np.sqrt(np.mean((out_v - gt_v) ** 2)))
        
        check = {
            "m": m,
            "l1_error": l1_err,
            "l2_error": l2_err,
            "output_error": output_err,
            "mcr2_final": cbsa.final_mcr2,
        }
        results["checks"].append(check)
        
        if verbose:
            is_full = m == kv_len
            expected = " (full tokens)" if is_full else ""
            print(f"    m={m:>4}: L1={l1_err:.2e}, L2={l2_err:.2e}, out_err={output_err:.2e}{expected}")
    
    return results


def run_coreset_vs_cbsa(
    kv_len: int,
    q_len: int,
    m: int,
    d: int,
    kv_type: str,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Head-to-head comparison: Coreset (k-means) vs CBSA (MCR²).
    
    Returns error metrics for both methods.
    """
    # Generate data based on type
    if kv_type == "clustered":
        K, V = make_clustered_kv(kv_len, d, n_clusters=8, seed=seed)
    elif kv_type == "random":
        K, V = make_random_kv(kv_len, d, seed=seed)
    else:  # skewed
        K, V = make_skewed_kv(kv_len, d, seed=seed)
    
    # Query
    Q = (np.random.default_rng(seed + 1000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    # Ground truth
    gt = ground_truth(Q, K, V)
    
    # === Coreset (k-means) ===
    centroids_K, centroids_V, weights = build_kmeans_coreset(K, V, m, seed=seed)
    coreset_stats = eval_kmeans_coreset(centroids_K, centroids_V, weights, Q)
    out_coreset = coreset_stats.finalize().squeeze(0)
    
    # === CBSA (MCR²) ===
    cbsa = train_cbsa_representatives(
        K, V, m,
        num_steps=30,  # Reduced from 50 for speed
        lr=1e-2,
        seed=seed,
        verbose=False,
    )
    cbsa_stats = eval_cbsa_representatives(cbsa.representatives, V, Q, K)
    out_cbsa = cbsa_stats.finalize().squeeze(0)
    
    # === Metrics ===
    # L1 error
    l1_coreset = float(np.abs(out_coreset - gt).mean())
    l1_cbsa = float(np.abs(out_cbsa - gt).mean())
    
    # L2 error
    l2_coreset = float(np.sqrt(np.mean((out_coreset - gt) ** 2)))
    l2_cbsa = float(np.sqrt(np.mean((out_cbsa - gt) ** 2)))
    
    # Output error (A_full * V vs A_sketch * V)
    out_v_coreset = out_coreset  # Already A * V form
    out_v_cbsa = out_cbsa
    output_err_coreset = float(np.sqrt(np.mean((out_v_coreset - gt) ** 2)))
    output_err_cbsa = float(np.sqrt(np.mean((out_v_cbsa - gt) ** 2)))
    
    # Relative improvement
    if l1_coreset > 0:
        l1_improve = (l1_coreset - l1_cbsa) / l1_coreset * 100
    else:
        l1_improve = 0.0
    
    result = {
        "kv_len": kv_len,
        "q_len": q_len,
        "m": m,
        "d": d,
        "kv_type": kv_type,
        "seed": seed,
        # Coreset metrics
        "coreset_l1": l1_coreset,
        "coreset_l2": l2_coreset,
        "coreset_output_err": output_err_coreset,
        # CBSA metrics
        "cbsa_l1": l1_cbsa,
        "cbsa_l2": l2_cbsa,
        "cbsa_output_err": output_err_cbsa,
        "cbsa_mcr2": cbsa.final_mcr2,
        # Improvement
        "l1_improvement_pct": l1_improve,
        "cbsa_wins": l1_cbsa < l1_coreset,
    }
    
    if verbose:
        winner = "CBSA" if result["cbsa_wins"] else "Coreset"
        print(
            f"  kv={kv_len:>5} m={m:>2} type={kv_type:>10}  "
            f"Coreset={l1_coreset:.4e}  CBSA={l1_cbsa:.4e}  "
            f"Δ={l1_improve:>+.1f}% ({winner})"
        )
    
    return result


def run_full_sweep(
    m_values: list[int] = None,
    kv_lens: list[int] = None,
    kv_types: list[str] = None,
    q_len: int = 64,
    d: int = 128,
    seed: int = 0,
    verbose: bool = True,
) -> list[dict]:
    """
    Run full sweep: 5 m × 3 kv_len × 3 kv_type = 45 configs.
    """
    if m_values is None:
        m_values = [4, 8, 16, 32, 64]
    if kv_lens is None:
        kv_lens = [1024, 4096]  # Reduced from [1024, 4096, 16384] for speed
    if kv_types is None:
        kv_types = ["clustered", "random", "skewed"]
    
    results = []
    
    if verbose:
        print("=" * 80)
        print("CBSA vs Coreset: Full Sweep (45 configs)")
        print("=" * 80)
    
    for kv_type in kv_types:
        if verbose:
            print(f"\n--- KV Type: {kv_type.upper()} ---")
        
        for kv_len in kv_lens:
            for m in m_values:
                # Skip if m >= kv_len (need at least some compression)
                if m >= kv_len:
                    continue
                
                try:
                    r = run_coreset_vs_cbsa(
                        kv_len=kv_len,
                        q_len=q_len,
                        m=m,
                        d=d,
                        kv_type=kv_type,
                        seed=seed,
                        verbose=verbose,
                    )
                    results.append(r)
                except Exception as e:
                    if verbose:
                        print(f"  Error for kv={kv_len} m={m} type={kv_type}: {e}")
    
    return results


# ============== Analysis ==============

def analyze_results(results: list[dict]) -> dict:
    """Analyze sweep results and compute aggregate statistics."""
    if not results:
        return {}
    
    analysis = {
        "total_configs": len(results),
        "by_kv_type": {},
        "by_m": {},
        "overall": {},
    }
    
    # Overall statistics
    cbsa_wins = sum(1 for r in results if r["cbsa_wins"])
    analysis["overall"]["cbsa_win_rate"] = cbsa_wins / len(results)
    
    all_improve = [r["l1_improvement_pct"] for r in results]
    analysis["overall"]["avg_improvement_pct"] = sum(all_improve) / len(all_improve)
    analysis["overall"]["max_improvement_pct"] = max(all_improve)
    analysis["overall"]["min_improvement_pct"] = min(all_improve)
    
    # By kv_type
    for kv_type in ["clustered", "random", "skewed"]:
        type_results = [r for r in results if r["kv_type"] == kv_type]
        if not type_results:
            continue
        
        type_wins = sum(1 for r in type_results if r["cbsa_wins"])
        type_improve = [r["l1_improvement_pct"] for r in type_results]
        
        analysis["by_kv_type"][kv_type] = {
            "n_configs": len(type_results),
            "cbsa_win_rate": type_wins / len(type_results),
            "avg_improvement_pct": sum(type_improve) / len(type_improve),
            "avg_coreset_l1": sum(r["coreset_l1"] for r in type_results) / len(type_results),
            "avg_cbsa_l1": sum(r["cbsa_l1"] for r in type_results) / len(type_results),
        }
    
    # By m
    for m in sorted(set(r["m"] for r in results)):
        m_results = [r for r in results if r["m"] == m]
        m_wins = sum(1 for r in m_results if r["cbsa_wins"])
        m_improve = [r["l1_improvement_pct"] for r in m_results]
        
        analysis["by_m"][m] = {
            "n_configs": len(m_results),
            "cbsa_win_rate": m_wins / len(m_results),
            "avg_improvement_pct": sum(m_improve) / len(m_results),
        }
    
    return analysis


def print_analysis(analysis: dict) -> None:
    """Print analysis results."""
    print()
    print("=" * 80)
    print("ANALYSIS: CBSA vs Coreset")
    print("=" * 80)
    
    print(f"\nTotal configs: {analysis['overall'].get('total_configs', len([]))}")
    print(f"CBSA win rate: {analysis['overall'].get('cbsa_win_rate', 0)*100:.1f}%")
    print(f"Avg improvement: {analysis['overall'].get('avg_improvement_pct', 0):+.2f}%")
    
    print("\n--- By KV Type ---")
    print(f"{'Type':<12} {'N':>4} {'CBSA Win%':>10} {'Avg Imp%':>10} {'Avg Coreset':>12} {'Avg CBSA':>12}")
    print("-" * 60)
    
    for kv_type, stats in analysis.get("by_kv_type", {}).items():
        print(
            f"{kv_type:<12} {stats['n_configs']:>4} "
            f"{stats['cbsa_win_rate']*100:>9.1f}% {stats['avg_improvement_pct']:>+9.2f}% "
            f"{stats['avg_coreset_l1']:>12.4e} {stats['avg_cbsa_l1']:>12.4e}"
        )
    
    print("\n--- By m (representatives) ---")
    print(f"{'m':>6} {'N':>4} {'CBSA Win%':>10} {'Avg Imp%':>10}")
    print("-" * 36)
    
    for m, stats in sorted(analysis.get("by_m", {}).items()):
        print(
            f"{m:>6} {stats['n_configs']:>4} "
            f"{stats['cbsa_win_rate']*100:>9.1f}% {stats['avg_improvement_pct']:>+9.2f}%"
        )


# ============== Main ==============

def main():
    print("=" * 80)
    print("Exp11: CBSA-style MCR² Representative Token Selection")
    print("Head-to-head: CBSA (MCR²-optimized) vs K-means Coreset")
    print("=" * 80)
    
    all_results = {}
    
    # 1. MCR² Training Curve (verify training stability)
    print("\n" + "=" * 80)
    print("1. MCR² Training Curve (verify -MCR² loss decreases)")
    print("=" * 80)
    training_result = run_mcr2_training_curve(
        kv_len=4096,
        m=32,
        d=128,
        num_steps=50,
        seed=0,
        verbose=True,
    )
    all_results["training_curve"] = training_result
    
    # Check training converged
    init_mcr2 = training_result["initial_mcr2"]
    final_mcr2 = training_result["final_mcr2"]
    print(f"\n  Training check: MCR² {init_mcr2:.4f} -> {final_mcr2:.4f}")
    print(f"  MCR² {'increased' if final_mcr2 > init_mcr2 else 'decreased'} (should increase for -loss)")
    
    # 2. Physical Consistency Check
    print("\n" + "=" * 80)
    print("2. Physical Consistency Check (m=full → error=0)")
    print("=" * 80)
    physical_check = run_physical_consistency_check(
        kv_len=1024,
        d=128,
        m_values=[4, 16, 64, 256, 1024],
        seed=0,
        verbose=True,
    )
    all_results["physical_check"] = physical_check
    
    # 3. Full Sweep (45 configs)
    print("\n" + "=" * 80)
    print("3. Full Sweep: Coreset vs CBSA")
    print("=" * 80)
    sweep_results = run_full_sweep(
        m_values=[4, 8, 16, 32, 64],
        kv_lens=[1024, 4096, 16384],
        kv_types=["clustered", "random", "skewed"],
        q_len=64,
        d=128,
        seed=0,
        verbose=True,
    )
    all_results["sweep_results"] = sweep_results
    
    # 4. Analysis
    print()
    analysis = analyze_results(sweep_results)
    all_results["analysis"] = analysis
    print_analysis(analysis)
    
    # Save results
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save MCR² training curve
    mcr2_path = os.path.join(output_dir, "exp11_mcr2_training.json")
    with open(mcr2_path, "w", encoding="utf-8") as f:
        json.dump(training_result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {mcr2_path}")
    
    # Save sweep results
    sweep_path = os.path.join(output_dir, "exp11_coreset_vs_cbsa.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {sweep_path}")
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    win_rate = analysis.get("overall", {}).get("cbsa_win_rate", 0) * 100
    avg_imp = analysis.get("overall", {}).get("avg_improvement_pct", 0)
    
    print(f"CBSA win rate: {win_rate:.1f}%")
    print(f"Average L1 improvement: {avg_imp:+.2f}%")
    
    # Per kv_type summary
    for kv_type, stats in analysis.get("by_kv_type", {}).items():
        winner = "CBSA" if stats["cbsa_win_rate"] > 0.5 else "Coreset"
        print(f"  {kv_type}: {winner} wins ({stats['cbsa_win_rate']*100:.1f}%), "
              f"avg Δ={stats['avg_improvement_pct']:+.2f}%")
    
    # Hypothesis check
    print()
    print("HYPOTHESIS CHECK:")
    clustered_stats = analysis.get("by_kv_type", {}).get("clustered", {})
    random_stats = analysis.get("by_kv_type", {}).get("random", {})
    
    if clustered_stats and random_stats:
        clustered_improve = clustered_stats["avg_improvement_pct"]
        random_improve = random_stats["avg_improvement_pct"]
        
        if clustered_improve > 0 and abs(clustered_improve) > abs(random_improve):
            print("  ✅ CBSA outperforms k-means more on clustered data (as expected)")
        else:
            print("  ⚠️  CBSA does NOT show expected advantage on clustered data")
        
        if abs(random_improve) < 5.0:
            print("  ✅ CBSA and k-means perform similarly on random data (as expected)")
        else:
            print("  ⚠️  CBSA shows unexpected difference on random data")
    
    return all_results


if __name__ == "__main__":
    main()
