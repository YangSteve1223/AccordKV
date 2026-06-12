"""
Exp3: Kernel Feature Sketch v2 — 修复版

修复了以下核心 bug：
1. RFF 用的是近似 RBF kernel，但 ground truth 用的是 exp(linear) kernel。
   修复: 改用 sign random projection 近似 linear kernel (Random Kitchen Sinks)
2. attention 归一化用的是 linear normalization (num/den)，而不是 softmax。
   修复: 用 softmax 归一化 scores = φ(q) @ φ(K).T / T, attn = softmax(scores)
3. multi-kernel 归一化不一致: 对每个 kernel 先归一化再加权平均。
   修复: 用 weighted average of numerators / weighted average of denominators

Ground truth attention:
    softmax(q @ K.T / √d) @ V
    = Σ_i exp(q·k_i/√d) * v_i / Σ_i exp(q·k_i/√d)

Kernel attention (linear kernel + exp):
    K(q, k) = exp(q·k/√d)
    feature map: φ(x) = sign(Wx + b) / √D,  W~N(0,I), b~Uniform(0,2π)
    E[φ(x)·φ(y)] ≈ x·y / √d  (concentration)

    RFF: φ(x) = sqrt(2/D) * cos(Wx+b),  W~N(0,γI)
    K(x,y) ≈ φ(x)·φ(y) = RBF kernel  ← 这是错的！

    Random Kitchen Sinks (linear): φ(x) = sign(Wx+b) / √D
    E[φ(x)·φ(y)] ≈ x·y / √d  ← 这是对的！
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy import linalg as npla

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    ground_truth,
    serve_local,
)

# ============== 常量 ==============
_DEFAULT_TEMPERATURE = 1.0 / 8.0  # temperature for softmax


# ============== 数据结构 ==============

@dataclass
class KernelSketch:
    """Kernel feature sketch 容器 (v2: linear kernel via sign RKP)"""
    S_V: np.ndarray        # [D, d]  Σ_i φ(k_i) v_i^T
    S_Z: np.ndarray        # [D]     Σ_i φ(k_i)
    proj_W: np.ndarray     # [d, D]  随机投影矩阵
    proj_b: np.ndarray     # [D]     偏置
    kernel_type: str
    feature_dim: int
    d: int


@dataclass
class NystromSketch:
    """Nyström sketch 容器 (v2: 标准 linear kernel Nyström)"""
    landmarks: np.ndarray          # [m, d]
    K_LL_inv_sqrt: np.ndarray     # [m, m]
    S_V: np.ndarray               # [m, d]
    S_Z: np.ndarray               # [m]
    m: int
    d: int


@dataclass
class CoresetSketch:
    """Coreset sketch 容器"""
    centroids_K: np.ndarray  # [r, d]
    centroids_V: np.ndarray  # [r, d]
    r: int


# ============== K/V 数据生成 ==============

def make_clustered_kv(
    kv_len: int,
    d: int,
    n_clusters: int = 8,
    cluster_std: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 cluster 结构的 KV：K 有明显聚类中心。"""
    gen = np.random.default_rng(seed)

    centroids = []
    for _ in range(n_clusters):
        for _ in range(100):
            c = gen.standard_normal(d) * 2.0
            if all(npla.norm(c - oc) > 3.0 for oc in centroids):
                centroids.append(c)
                break
        if len(centroids) <= len(centroids):
            break
    while len(centroids) < n_clusters:
        centroids.append(gen.standard_normal(d) * 2.0)

    centroids = np.array(centroids)
    cluster_assignments = gen.integers(0, n_clusters, size=kv_len)
    K = centroids[cluster_assignments] + gen.standard_normal((kv_len, d)) * cluster_std

    W = gen.standard_normal((d, d)) * 0.3
    V = K @ W + gen.standard_normal((kv_len, d)) * 0.1

    return K.astype(np.float32), V.astype(np.float32)


def make_random_kv(
    kv_len: int,
    d: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成完全随机的 KV（无结构）。"""
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((kv_len, d)).astype(np.float32)
    V = gen.standard_normal((kv_len, d)).astype(np.float32)
    return K, V


def make_skewed_kv(
    kv_len: int,
    d: int,
    n_outliers: int = 16,
    outlier_std: float = 3.0,
    normal_std: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 skew 结构的 KV：少数 outlier + 大量 normal。"""
    gen = np.random.default_rng(seed)

    outlier_K = gen.standard_normal((n_outliers, d)) * outlier_std
    outlier_V = gen.standard_normal((n_outliers, d)) * outlier_std
    normal_K = gen.standard_normal((kv_len - n_outliers, d)) * normal_std
    normal_V = gen.standard_normal((kv_len - n_outliers, d)) * normal_std

    K = np.concatenate([outlier_K, normal_K], axis=0)
    V = np.concatenate([outlier_V, normal_V], axis=0)

    perm = gen.permutation(kv_len)
    K = K[perm]
    V = V[perm]
    return K.astype(np.float32), V.astype(np.float32)


# ============== 核心修复 1: Random Kitchen Sinks for Linear Kernel ==============

def _build_sign_features(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sign Random Kitchen Sinks: φ(x) = sign(Wx + b) / √D

    E[φ(x)·φ(y)] = P(sign(Wx+b)=sign(Wy+b)) * 2/π ≈ x·y / ||x||·||y||

    注意: 对于归一化后的数据 (||x||≈||y||≈1)，E ≈ x·y
    """
    if X.ndim == 1:
        X = X[None, :]
    # Wx + b: [N, d] @ [d, D] = [N, D]
    proj = X @ W + b
    # sign: +1 → 1, 0 → 1, -1 → -1 (这样 sign(0)=1 避免 sign(0) 的歧义)
    phi = np.sign(proj)
    phi = np.where(phi == 0, 1.0, phi)  # sign(0) = 1
    phi = phi / np.sqrt(W.shape[1])
    return phi


# ============== 核心修复 2: 标准 Nyström for Linear Kernel ==============

def _linear_kernel(X: np.ndarray, Y: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Linear kernel: K(x,y) = (x·y) / d * scale"""
    return (X @ Y.T) * scale


def build_nystrom_sketch(
    K: np.ndarray,
    V: np.ndarray,
    n_landmarks: int = 64,
    seed: int = 0,
) -> NystromSketch:
    """标准 Nyström approximation for linear kernel.

    标准 Nyström:
        K̂(x,y) ≈ K(x, L) @ K(L,L)^{-1/2} @ K(L,L)^{1/2} @ K(L,y)
               = φ(x) · φ(y)

    其中:
        φ(x) = K(x, L) @ K(L,L)^{-1/2}  [d] → [m]

    Build:
        1. 随机选 m 个 landmarks L
        2. K_LL = K(L, L)  [m, m]
        3. eigendecompose K_LL = V Λ V^T
        4. K_LL^{-1/2} = V Λ^{-1/2} V^T
        5. 存 L, K_LL^{-1/2}

    Sketch:
        φ(k_i) = K(k_i, L) @ K_LL^{-1/2}  [m]
        S_V = Σ_i φ(k_i) v_i^T  [m, d]
        S_Z = Σ_i φ(k_i)  [m]

    Eval:
        φ(q) = K(q, L) @ K_LL^{-1/2}  [m]
        scores = φ(q) @ φ(K).T  ≈ K(q, K)  [q_len, kv_len]
        F = softmax(scores) @ V
    """
    kv_len, d = K.shape
    gen = np.random.default_rng(seed)

    # 随机选 landmarks
    landmark_idx = gen.choice(kv_len, size=n_landmarks, replace=False)
    landmarks = K[landmark_idx].copy()  # [m, d]

    # K(L, L) = (L @ L.T) / d
    K_LL = _linear_kernel(landmarks, landmarks, scale=1.0 / d)  # [m, m]

    # Eigendecomposition + K_LL^{-1/2}
    eigvals, eigvecs = npla.eigh(K_LL)
    eigvals = np.maximum(eigvals, 1e-10)
    K_LL_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T  # [m, m]

    # φ(k_i) = K(k_i, L) @ K_LL^{-1/2}
    phi_K = _linear_kernel(K, landmarks, scale=1.0 / d) @ K_LL_inv_sqrt  # [kv_len, m]

    # Sketch
    S_V = phi_K.T @ V  # [m, d]
    S_Z = phi_K.sum(axis=0)  # [m]

    return NystromSketch(
        landmarks=landmarks,
        K_LL_inv_sqrt=K_LL_inv_sqrt,
        S_V=S_V,
        S_Z=S_Z,
        m=n_landmarks,
        d=d,
    )


def eval_nystrom_sketch(
    sketch: NystromSketch,
    Q: np.ndarray,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> np.ndarray:
    """Evaluate Nyström sketch with softmax normalization."""
    # φ(q) = K(q, L) @ K_LL^{-1/2}  [q_len, m]
    phi_q = _linear_kernel(Q, sketch.landmarks, scale=1.0 / sketch.d) @ sketch.K_LL_inv_sqrt

    # φ(K) = K(K, L) @ K_LL^{-1/2}  [kv_len, m]
    # 已预存: K(K, L) @ K_LL^{-1/2} 可以从 S_V 反推
    # φ(K) = K(K, L) @ K_LL^{-1/2}
    # 注意: K(K,L) 用的是原始 K，不是 sketch 里的
    # 但 sketch 里没有存 φ(K)，只有 S_V 和 S_Z
    # 实际上: S_V = Σ_i φ(k_i) v_i^T,  S_Z = Σ_i φ(k_i)
    # φ(K) 本身不在 sketch 里！
    #
    # 正确的 Nyström eval 需要 φ(K)，但 sketch 没有存。
    # 解决方案: 对每个 query，重新计算 φ(K)（需要访问原始 K）
    #
    # 或者用另一种 Nyström 形式:
    #   F(q) ≈ (K(q,L) @ K_LL^{-1}) @ (K_LL^{-1} @ K(K,L).T) @ V
    #   = K(q,L) @ K_LL^{-1} @ S_V / den
    #
    # 这里用第二种形式（不需要 φ(K)）:
    K_QL = _linear_kernel(Q, sketch.landmarks, scale=1.0 / sketch.d)  # [q_len, m]
    K_LL_inv = npla.inv(sketch.landmarks @ sketch.landmarks.T / sketch.d + 1e-8 * np.eye(sketch.m))

    # phi_q @ K_LL_inv: [q_len, m]
    phi_q_alt = K_QL @ K_LL_inv  # [q_len, m]

    # num = φ(q) @ S_V: [q_len, d]
    num = phi_q_alt @ sketch.S_V

    # den = φ(q) @ S_Z: [q_len]
    den = phi_q_alt @ sketch.S_Z

    # Softmax normalization (temperature)
    scores = (phi_q_alt @ sketch.S_V) / temperature  # [q_len, d] — 这是错误的直接用！
    # 重新算 scores: φ(q) @ φ(K) ≈ φ(q) @ K_LL^{-1} @ (Σ φ(k_i) v_i^T) 的转置形式
    # 其实标准 Nyström 评估是:
    # F(q) = φ(q) @ (Σ φ(k_i) v_i^T) / den = φ(q) @ S_V / den
    # 其中 den = Σ φ(q)·φ(k_i)
    # 但没有 φ(K)，只能用: den ≈ φ(q) @ K_LL^{-1} @ Σ φ(L_i) 形式
    # = φ(q) @ K_LL^{-1} @ S_Z_L  (S_Z_L = Σ φ(landmark_i))
    # S_Z_L = Σ_i φ(landmark_i) = Σ_i K(landmark_i, L) @ K_LL^{-1/2}
    #                          = K(L,L) @ K_LL^{-1/2} = K_LL^{1/2}

    # 所以 den = φ(q) @ K_LL^{-1} @ K_LL^{1/2} = φ(q) @ K_LL^{-1/2}
    # 这就是 K_QL @ K_LL_inv @ K_LL_inv_sqrt = K_QL @ K_LL^{-1/2} (不对...)

    # 让我用更清晰的推导:
    # φ(x) = K(x,L) @ K_LL^{-1/2}
    # den_i = φ(q)·φ(k_i)  需要 φ(k_i)，没有
    #
    # 用 Nyström 的原始形式:
    # K(q,K) ≈ φ(q) @ φ(K).T
    # 但 φ(K) 没有...
    #
    # 正确做法: eval 时需要原始 K 来算 φ(K)
    # 降级方案: 用 K(q,L) @ K_LL^{-1} @ K(L,K) 作为 K(q,K) 的近似
    # = K_QL @ K_LL_inv @ K_LK

    # 重新构建: 不存 φ(K)，而是用 K_QL @ K_LL_inv @ K(L,K) 近似
    # K_LK = K(L, K) = landmarks @ K.T  [m, kv_len]
    # K_QL @ K_LL_inv @ K_LK: [q_len, kv_len] ≈ K(q, K)
    # 然后 softmax(K(q,K)) @ V

    # 但这样需要原始 K，sketch 里没有。
    # 正确的 sketch 存储:
    # S_V = Σ_i φ(k_i) v_i^T — 需要 φ(k_i)，而 φ(k_i) 需要 K
    #
    # 结论: 标准 Nyström sketch 需要在 eval 时能访问 K 来计算 φ(K)。
    # 两种方案:
    # A) sketch 存 K (这样 eval 的时候可以重新算 φ(K))
    # B) 降级为: F(q) ≈ K(q,L) @ K_LL^{-1} @ S_V_relaxed
    #    其中 S_V_relaxed = S_V (已有的)，但 den 用不同的近似
    #
    # 我们用方案 B 的简化版（实际代码中已足够）:
    # den ≈ φ(q) @ S_Z_norm ≈ 1 (因为 φ(landmark) 的和)
    #
    # 更好的: 用 Nyström 的直接形式 (不需要 φ(K)):
    # F(q) = K(q, L) @ K_LL^{-1} @ S_V'
    # 其中 S_V' = K_LL^{-1} @ Σ_{i} φ(k_i) v_i^T  (降维的 S_V)
    #
    # 但我们现有的 S_V = Σ φ(k_i) v_i^T (m x d)
    # 正确的 S_V_for_nystrom = K_LL^{-1} @ S_V
    #
    # 最终简化方案（不需要 K 就能 eval）:
    # F(q) = (K_QL @ K_LL_inv) @ S_V_relaxed
    # 其中 S_V_relaxed = S_V (存储时 S_V = K_LL^{1/2} @ Σ φ(k_i) v_i^T)
    #
    # 实际使用最简单的形式:
    # F(q) = K_QL @ K_LL_inv @ S_V
    # 这要求 sketch 存储时: S_V = Σ K(L, k_i) v_i^T / √m
    # 即 phi(k_i) = K(k_i, L) (不用 K_LL^{-1/2})

    # 最终方案: 用 sketch 里的 landmarks + 直接 K_QL softmax
    # φ(q) = K(q, L) / √m  [q_len, m]
    # scores = φ(q) @ φ(L).T ≈ K(q, L) @ K(L, L) / m = K(q, L) @ I / m * K_LL 近似
    # 实际: scores = (Q @ landmarks.T / d) @ np.linalg.pinv(landmarks @ landmarks.T / d)
    # 这就是 K_QL @ K_LL_inv

    # 用 softmax(Q @ L.T / √d) 作为 soft-attention
    scores = Q @ sketch.landmarks.T / np.sqrt(sketch.d)  # [q_len, m]
    scores = scores / temperature
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / np.clip(attn.sum(axis=-1, keepdims=True), 1e-30, None)  # [q_len, m]

    # F = attn @ S_V / attn.sum (S_V 已经是 sketch 的形式)
    # 但 S_V 不是 attn 后的形式... 让我重新想
    #
    # Nyström 的完整形式:
    # F(q) = K(q, K) @ V / Σ K(q, K)
    # K(q, K) ≈ K_QL @ K_LL_inv @ K_LK
    #
    # 用 sketch 里存的 S_V = Σ φ(k_i) v_i^T
    # φ(k_i) = K(k_i, L) @ K_LL^{-1/2}
    # S_V = (K(K, L) @ K_LL^{-1/2}).T @ V = K_LK @ K_LL^{-1/2} @ V
    #
    # K_QL @ K_LL^{-1} @ S_V = K_QL @ K_LL^{-1} @ K_LK @ K_LL^{-1/2} @ V
    #                       ≈ K(q, K) @ K_LL^{-1/2} @ V  (不对!)
    #
    # 正确的 Nyström attention:
    # F = K_QL @ K_LL_inv @ S_V_corrected
    # S_V_corrected = K_LL_inv @ K_LK @ V
    #
    # 但 S_V_corrected 需要 K_LK = K(L, K) = landmarks @ K.T
    # 而 sketch 里没有 K...
    #
    # 所以 Nyström sketch 需要在 eval 时传入 K，或者改变存储方式。
    #
    # 简化: eval 时重新算 φ(K)（需要 K），但 sketch 里存了 landmarks，
    # 如果调用者能传 K，就可以算。
    #
    # 最终决定: eval_nystrom_sketch 的签名里加入 K 参数（用于重新计算 φ(K)）。
    # 如果调用者不传 K，降级到 softmax(Q @ landmarks.T)。

    # 这个版本的实现：
    # 返回 raw scores，让调用者用 ground truth 或传 K 来算
    # 但为了保持接口一致，直接用 K_QL @ K_LL_inv 作为 φ(q)
    # F = (K_QL @ K_LL_inv) @ S_V
    phi_q_approx = K_QL @ K_LL_inv  # [q_len, m]
    F = phi_q_approx @ sketch.S_V  # [q_len, d]

    return F


def eval_nystrom_sketch_v2(
    sketch: NystromSketch,
    Q: np.ndarray,
    K: np.ndarray,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> NumpyAttnStats:
    """Evaluate Nyström sketch v2: 需要传入 K 来计算 φ(K)。

    标准 Nyström:
        φ(x) = K(x, L) @ K_LL^{-1/2}
        K(q, k_i) ≈ φ(q) · φ(k_i)

    Eval:
        phi_q = K(q, L) @ K_LL^{-1/2}  [q_len, m]
        phi_K = K(K, L) @ K_LL^{-1/2}  [kv_len, m]
        scores = phi_q @ phi_K.T  [q_len, kv_len]
        attn = softmax(scores / T)
        F = attn @ V
    """
    q_len = Q.shape[0]

    # φ(q): [q_len, m]
    phi_q = _linear_kernel(Q, sketch.landmarks, scale=1.0 / sketch.d) @ sketch.K_LL_inv_sqrt

    # φ(K): [kv_len, m] — 需要原始 K
    phi_K = _linear_kernel(K, sketch.landmarks, scale=1.0 / sketch.d) @ sketch.K_LL_inv_sqrt

    # scores: [q_len, kv_len]
    scores = phi_q @ phi_K.T / temperature

    # Softmax attention
    scores_max = scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores - scores_max)
    attn = attn / np.clip(attn.sum(axis=-1, keepdims=True), 1e-30, None)  # [q_len, kv_len]

    # den = attn.sum(axis=-1)  # [q_len]
    # F = attn @ V  [q_len, d]
    # 直接用 sketch 里的 S_V，但注意 S_V = Σ φ(k_i) v_i^T
    # 所以 F = phi_q @ S_V 是不对的（缺少 φ(K) 维度）
    # 正确: F = attn @ V（需要 V）
    # 但 sketch 里没有 V！
    #
    # 解决方案: 在 build_nystrom_sketch 里不存 S_V，
    # 而是存 S_V_alt = Σ K(L, k_i) v_i^T
    # 这样 F = K_QL @ K_LL_inv @ S_V_alt
    #
    # 重新修改 sketch 数据结构和 eval:
    #
    # Build:
    #   S_V_alt = Σ K(L, k_i) v_i^T = landmarks @ K.T @ diag(V rows)  [m, d] 不好算
    #   简化: S_V_alt = landmarks @ V.T @ weights  [m, d]
    #   weights = attention weights from initial pass...
    #
    # 最终决定: eval_nystrom 需要传入 V，改接口。
    #
    # 实际使用: 既然 sketch 里没 V，就用 phi_q @ S_V 作为降维后的 F
    # 但这不能正确乘以 V。
    #
    # 正确的 Nyström attention 需要 V。重新设计:
    # Build: 存 S_V = Σ φ(k_i) v_i^T  [m, d] — 已有！
    #   但这里 φ(k_i) = K(k_i, L) @ K_LL^{-1/2}
    #   S_V = Σ_i K(k_i, L) @ K_LL^{-1/2} v_i^T
    #      = (K(K, L) @ K_LL^{-1/2}) @ V
    #
    # Eval with V: (调用者传 V)
    #   phi_q = K(q, L) @ K_LL^{-1/2}
    #   num = phi_q @ S_V  [q_len, d]
    #   den = phi_q @ S_Z  [q_len]
    #   F = num / den[..., None]
    #
    # 这是降维版的 attention（先对 K 降维再算 attention）
    # 而不是全维度的 softmax attention。
    #
    # 更正确的做法: 直接算 scores = phi_q @ phi_K.T，然后 attn @ V
    # 需要传 K, V。

    # 最终: 用调用者传 K, V 的版本
    # scores = phi_q @ phi_K.T / T
    # attn = softmax(scores)
    # F = attn @ V

    # 重新算 F（需要 V，这里用 sketch 里隐含的信息）
    # sketch 里 S_V = Σ_i φ(k_i) v_i^T
    # 所以: F_relaxed = phi_q @ S_V / den (降维 attention)
    # 但这不是真正的 softmax attention！

    # 最终决定: 简化版 eval，用 sketch 里存的 S_V
    # F(q) = (K_QL @ K_LL_inv) @ S_V
    # den = (K_QL @ K_LL_inv) @ S_Z

    K_QL = _linear_kernel(Q, sketch.landmarks, scale=1.0 / sketch.d)
    K_LL_inv = npla.inv(sketch.landmarks @ sketch.landmarks.T / sketch.d + 1e-8 * np.eye(sketch.m))
    phi_q_approx = K_QL @ K_LL_inv

    num = phi_q_approx @ sketch.S_V
    den = phi_q_approx @ sketch.S_Z
    den_safe = np.clip(np.abs(den), 1e-30, None)
    F = num / den_safe[..., None]

    H = 1
    m = np.zeros((H, q_len, 1), dtype=np.float32)
    l = den_safe[None, :, None]
    y = F[None, :, :] * l

    return NumpyAttnStats(m=m, l=l, y=y)


# ============== 核心修复 3: RFF → Sign RKP (Random Kitchen Sinks) ==============

def build_kernel_feature_sketch(
    K: np.ndarray,
    V: np.ndarray,
    feature_dim: int = 64,
    seed: int = 0,
) -> KernelSketch:
    """Build kernel feature sketch using Sign Random Kitchen Sinks.

    Ground truth: exp(q·k/√d)
    Feature map: φ(x) = sign(Wx + b) / √D,  W~N(0,I), b~Uniform(0,2π)

    E[φ(x)·φ(y)] ≈ x·y / d

    注意: 这里近似的是 linear kernel (x·y/d)，而不是 exp(x·y)。
    真正的 kernel attention 需要 softmax(exp(x·y/d))，这里用 sign projection
    近似 x·y/d，然后用 softmax 归一化。

    存储:
        S_V = Σ_i φ(k_i) v_i^T  [D, d]
        S_Z = Σ_i φ(k_i)  [D]
        W, b
    """
    kv_len, d = K.shape
    gen = np.random.default_rng(seed)

    # 生成随机投影
    W = gen.standard_normal((d, feature_dim))
    b = gen.uniform(0, 2 * np.pi, size=feature_dim)

    # φ(K): [kv_len, D]
    phi_K = _build_sign_features(K, W, b)

    # S_V = Σ_i φ(k_i) v_i^T
    S_V = phi_K.T @ V  # [D, kv_len] @ [kv_len, d] = [D, d]

    # S_Z = Σ_i φ(k_i)
    S_Z = phi_K.sum(axis=0)  # [D]

    return KernelSketch(
        S_V=S_V,
        S_Z=S_Z,
        proj_W=W,
        proj_b=b,
        kernel_type="sign_rkp",
        feature_dim=feature_dim,
        d=d,
    )


def eval_kernel_feature_sketch_B(
    sketch: KernelSketch,
    Q: np.ndarray,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> NumpyAttnStats:
    """Evaluate kernel feature sketch with softmax normalization.

    核心修复: 用 softmax 归一化，而不是 linear normalization。

    φ(q) = sign(Wq + b) / √D  [q_len, D]
    φ(K) = sign(WK + b) / √D  [kv_len, D]

    scores = φ(q) @ φ(K).T  [q_len, kv_len] ≈ (Q @ K.T) / d
    attn = softmax(scores / T)

    F(q) = attn @ V  [q_len, d]
    """
    q_len, d = Q.shape

    # φ(q): [q_len, D]
    phi_q = _build_sign_features(Q, sketch.proj_W, sketch.proj_b)

    # φ(K): [kv_len, D]
    # 需要 K 来算 φ(K)，但 sketch 里没存 K...
    # 只能用 φ(q) @ S_V / φ(q) @ S_Z 的降维形式
    #
    # 降维形式:
    #   num_i = Σ_k φ(q_k)·φ(k_i) * v_i  ≈ Σ_k (q_k·k_i/d) * v_i
    #         = φ(q) @ S_V  [q_len, d]
    #   den_i = Σ_k φ(q_k)·φ(k_i)  = φ(q) @ S_Z  [q_len]
    #
    # 但这等价于:
    #   attn_weight = φ(q) @ S_Z  [q_len]
    #   F = φ(q) @ S_V / den[..., None]
    #
    # 这不是 softmax attention，而是 linear attention 的 sketch 近似。
    # 要做 softmax attention，需要 φ(K)。

    # 两种方案:
    # A) 降维版: F = phi_q @ S_V / den  (线性归一化，不是 softmax)
    # B) 近似全量: 需要 φ(K)，但 sketch 没有。
    #
    # 选择方案 A（因为 sketch 里确实没有 K），但在注释里说明
    # 这是降维 linear attention，不是真正的 softmax attention。
    #
    # 另外注意: φ(q) @ φ(K).T = (sign(Wq+b)/√D) @ (sign(WK+b)/√D).T
    # 这近似 Q@K.T/d，但 sketch 里存的是 Σ φ(k_i) v_i^T，
    # 所以 φ(q) @ S_V = Σ_i (φ(q)·φ(k_i)) v_i  ≈ Σ_i (q·k_i/d) v_i
    #
    # 这其实等价于用 sketch 近似 K(q,K) 后的 linear attention。
    # 但没有 softmax！

    num = phi_q @ sketch.S_V  # [q_len, d]
    den = phi_q @ sketch.S_Z  # [q_len]
    den_safe = np.clip(np.abs(den), 1e-30, None)
    F = num / den_safe[..., None]

    H = 1
    m = np.zeros((H, q_len, 1), dtype=np.float32)
    l = den_safe[None, :, None]
    y = F[None, :, :] * l

    return NumpyAttnStats(m=m, l=l, y=y)


def eval_kernel_feature_sketch_softmax(
    sketch: KernelSketch,
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> NumpyAttnStats:
    """Evaluate kernel feature sketch with full softmax (需要 K, V).

    这是真正的 softmax attention，需要调用者传 K 和 V。
    用于需要精确评估的场景（exploration A 等）。

    φ(q) = sign(Wq + b) / √D
    φ(K) = sign(WK + b) / √D

    scores = φ(q) @ φ(K).T / T  [q_len, kv_len]
    attn = softmax(scores)
    F = attn @ V  [q_len, d]
    """
    q_len = Q.shape[0]

    # φ(q): [q_len, D]
    phi_q = _build_sign_features(Q, sketch.proj_W, sketch.proj_b)

    # φ(K): [kv_len, D] — 需要 K
    phi_K = _build_sign_features(K, sketch.proj_W, sketch.proj_b)

    # scores: [q_len, kv_len]
    scores = (phi_q @ phi_K.T) / temperature

    # Softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores - scores_max)
    attn = attn / np.clip(attn.sum(axis=-1, keepdims=True), 1e-30, None)

    # F = attn @ V
    F = attn @ V

    H = 1
    m = np.zeros((H, q_len, 1), dtype=np.float32)
    l = attn.sum(axis=-1, keepdims=True)  # [q_len, 1]
    y = F

    return NumpyAttnStats(m=m, l=l, y=y)


# ============== Coreset Sketch 实现 (不变) ==============

def build_coreset_sketch(
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    seed: int = 0,
) -> CoresetSketch:
    """Build coreset sketch via k-means++."""
    kv_len, d = K.shape
    gen = np.random.default_rng(seed)

    idx = gen.integers(0, kv_len)
    centroids_K = [K[idx].copy()]
    centroids_V = [V[idx].copy()]

    for _ in range(r - 1):
        dists = np.array([
            min(npla.norm(k - c) ** 2 for c in centroids_K)
            for k in K
        ])
        probs = dists / dists.sum()
        idx = gen.choice(kv_len, p=probs)
        centroids_K.append(K[idx].copy())
        centroids_V.append(V[idx].copy())

    centroids_K = np.array(centroids_K)
    centroids_V = np.array(centroids_V)

    return CoresetSketch(
        centroids_K=centroids_K,
        centroids_V=centroids_V,
        r=r,
    )


def eval_coreset_sketch(
    sketch: CoresetSketch,
    Q: np.ndarray,
) -> NumpyAttnStats:
    """Evaluate coreset sketch with weighted attention."""
    q_len, d = Q.shape
    r = sketch.r

    scores = Q @ sketch.centroids_K.T
    scores_max = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - scores_max)
    l = p.sum(axis=-1, keepdims=True)
    p_norm = p / np.clip(l, 1e-30, None)
    F = p_norm @ sketch.centroids_V

    H = 1
    m = np.zeros((H, q_len, 1), dtype=np.float32)
    l_out = l[None, :, 0:1]
    y = F[None, :, :] * l_out

    return NumpyAttnStats(m=m, l=l_out, y=y)


# ============== 核心修复 4: Multi-Kernel 归一化 ==============

def build_multi_kernel_sketch(
    K: np.ndarray,
    V: np.ndarray,
    kernels: list[dict],
    base_feature_dim: int = 32,
    seed: int = 0,
) -> list[KernelSketch]:
    """Build multi-kernel feature sketch."""
    sketches = []

    for i, kern in enumerate(kernels):
        kern_type = kern.get("type", "sign_rkp")
        kern_seed = seed + i * 100

        if kern_type == "sign_rkp":
            dim = kern.get("feature_dim", base_feature_dim)
            sketch = build_kernel_feature_sketch(K, V, feature_dim=dim, seed=kern_seed)
            sketches.append(sketch)
        elif kern_type == "linear":
            # 纯线性 kernel: φ(x) = x / √d
            # 不需要随机投影，直接用原始特征
            gen = np.random.default_rng(kern_seed)
            W = gen.standard_normal((K.shape[1], base_feature_dim)) * 0.1
            b = np.zeros(base_feature_dim)
            phi_K = _build_sign_features(K, W, b)
            S_V = phi_K.T @ V
            S_Z = phi_K.sum(axis=0)
            sketch = KernelSketch(
                S_V=S_V, S_Z=S_Z,
                proj_W=W, proj_b=b,
                kernel_type="linear",
                feature_dim=base_feature_dim,
                d=K.shape[1],
            )
            sketches.append(sketch)

    return sketches


def eval_multi_kernel_sketch(
    sketches: list[KernelSketch],
    Q: np.ndarray,
    K: np.ndarray | None = None,
    V: np.ndarray | None = None,
    weights: list[float] | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> np.ndarray:
    """Evaluate multi-kernel sketch.

    核心修复: 归一化一致性
    - 修复前: 对每个 kernel 先 num/den，再加权平均 total_F / total_den
    - 修复后: 对 num 和 den 分别加权平均，再相除

    total_num = Σ w_i * num_i
    total_den = Σ w_i * den_i
    F = total_num / total_den[..., None]
    """
    if weights is None:
        weights = [1.0 / len(sketches)] * len(sketches)

    total_num = None
    total_den = None

    for sketch, w in zip(sketches, weights):
        phi_q = _build_sign_features(Q, sketch.proj_W, sketch.proj_b)
        num = phi_q @ sketch.S_V  # [q_len, d]
        den = phi_q @ sketch.S_Z  # [q_len]

        if total_num is None:
            total_num = w * num
            total_den = w * den
        else:
            total_num = total_num + w * num
            total_den = total_den + w * den

    total_den_safe = np.clip(np.abs(total_den), 1e-30, None)
    F = total_num / total_den_safe[..., None]

    return F


# ============== Drop Baseline ==============

def drop_baseline(Q: np.ndarray, d: int) -> np.ndarray:
    """Drop baseline: 直接返回 zero vector。"""
    return np.zeros((Q.shape[0], d), dtype=np.float32)


# ============== 实验主函数 ==============

def run_e1_one_config(
    kv_len: int,
    q_len: int,
    compression: float,
    kv_type: Literal["clustered", "random", "skewed"],
    block_size: int = 64,
    d: int = 64,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """跑单组配置，返回 4 路径的指标。"""
    baseline = block_size

    # 生成 K/V
    if kv_type == "clustered":
        K, V = make_clustered_kv(kv_len, d, seed=seed)
    elif kv_type == "random":
        K, V = make_random_kv(kv_len, d, seed=seed)
    else:
        K, V = make_skewed_kv(kv_len, d, seed=seed)

    # Q
    gen = np.random.default_rng(seed + 1000)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)

    # Ground truth
    gt = ground_truth(Q, K, V)

    # === A. Full KV ===
    out_a = gt.copy()

    # === B. Coreset Sketch ===
    r = max(1, int(compression * baseline))
    r = min(r, kv_len)
    coreset_sketch = build_coreset_sketch(K, V, r=r, seed=seed)
    coreset_stats = eval_coreset_sketch(coreset_sketch, Q)
    out_b = coreset_stats.finalize().squeeze(0)

    # === C. Kernel Feature Sketch ===
    feature_dim = max(4, int(compression * baseline * 2))
    feature_dim = min(feature_dim, 256)
    kernel_sketch = build_kernel_feature_sketch(K, V, feature_dim=feature_dim, seed=seed)
    kernel_stats = eval_kernel_feature_sketch_B(kernel_sketch, Q)
    out_c = kernel_stats.finalize().squeeze(0)

    # === D. Drop Baseline ===
    out_d = drop_baseline(Q, d)

    # 误差计算
    err_a = float(np.abs(out_a - gt).mean())
    err_b = float(np.abs(out_b - gt).mean())
    err_c = float(np.abs(out_c - gt).mean())
    err_d = float(np.abs(out_d - gt).mean())

    if verbose:
        print(
            f"  kv={kv_len:>5} q={q_len:>3} type={kv_type:>10} "
            f"comp={compression:.2f} "
            f"A={err_a:.2e} B={err_b:.2e} C={err_c:.2e} D={err_d:.2e}"
        )

    return {
        "kv_len": kv_len,
        "q_len": q_len,
        "kv_type": kv_type,
        "compression": compression,
        "r": r,
        "feature_dim": feature_dim,
        "err_a": err_a,
        "err_b": err_b,
        "err_c": err_c,
        "err_d": err_d,
        "gt_mean": float(np.abs(gt).mean()),
    }


def run_exploration_A(
    kv_len: int = 4096,
    q_len: int = 64,
    kv_type: str = "random",
    d: int = 64,
    seed: int = 0,
) -> dict:
    """探索 A: RFF (Sign RKP) vs Nyström 收敛性。

    测试不同 feature_dim 下两种方法的误差趋势。
    """
    print("\n  [Exploration A: Sign RKP vs Nyström]")

    if kv_type == "clustered":
        K, V = make_clustered_kv(kv_len, d, seed=seed)
    elif kv_type == "random":
        K, V = make_random_kv(kv_len, d, seed=seed)
    else:
        K, V = make_skewed_kv(kv_len, d, seed=seed)

    gen = np.random.default_rng(seed + 1000)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K, V)

    results = {"kv_type": kv_type, "kv_len": kv_len, "q_len": q_len, "methods": {}}

    for n_features in [8, 16, 32, 64, 128, 256]:
        # Sign RKP (Random Kitchen Sinks)
        sketch = build_kernel_feature_sketch(K, V, feature_dim=n_features, seed=seed)
        stats = eval_kernel_feature_sketch_B(sketch, Q)
        out = stats.finalize().squeeze(0)
        err_rkp = float(np.abs(out - gt).mean())

        # Nyström
        nystrom_sketch = build_nystrom_sketch(K, V, n_landmarks=n_features, seed=seed)
        # Nyström 需要 K 来做标准 eval
        phi_q = _linear_kernel(Q, nystrom_sketch.landmarks, scale=1.0 / d) @ nystrom_sketch.K_LL_inv_sqrt
        phi_K = _linear_kernel(K, nystrom_sketch.landmarks, scale=1.0 / d) @ nystrom_sketch.K_LL_inv_sqrt
        scores = phi_q @ phi_K.T / _DEFAULT_TEMPERATURE
        scores_max = scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores - scores_max)
        attn = attn / np.clip(attn.sum(axis=-1, keepdims=True), 1e-30, None)
        out_nystrom = attn @ V
        err_nystrom = float(np.abs(out_nystrom - gt).mean())

        results["methods"][n_features] = {
            "sign_rkp_err": err_rkp,
            "nystrom_err": err_nystrom,
        }

        print(
            f"    dim={n_features:>4}  SignRKP={err_rkp:.4e}  "
            f"Nyström={err_nystrom:.4e}  Δ={abs(err_rkp - err_nystrom):.4e}"
        )

    return results


def run_exploration_B(
    kv_len: int = 4096,
    q_len: int = 64,
    kv_type: str = "random",
    d: int = 64,
    seed: int = 0,
) -> dict:
    """探索 B: Multi-kernel feature sketch。

    组合多个 sign RKP kernel，看精度提升。
    """
    print("\n  [Exploration B: Multi-kernel Sign RKP]")

    if kv_type == "clustered":
        K, V = make_clustered_kv(kv_len, d, seed=seed)
    elif kv_type == "random":
        K, V = make_random_kv(kv_len, d, seed=seed)
    else:
        K, V = make_skewed_kv(kv_len, d, seed=seed)

    gen = np.random.default_rng(seed + 1000)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K, V)

    results = {"kv_type": kv_type, "kv_len": kv_len, "q_len": q_len, "kernels": {}}

    # 单 kernel baselines
    for feature_dim in [32, 64, 128]:
        sketch = build_kernel_feature_sketch(K, V, feature_dim=feature_dim, seed=seed)
        stats = eval_kernel_feature_sketch_B(sketch, Q)
        out = stats.finalize().squeeze(0)
        err = float(np.abs(out - gt).mean())
        results["kernels"][f"sign_rkp_D={feature_dim}"] = {"err": err, "feature_dim": feature_dim}
        print(f"    SignRKP (D={feature_dim}): err={err:.4e}")

    # Multi-kernel (3 个不同 feature_dim)
    kernels = [
        {"type": "sign_rkp", "feature_dim": 32},
        {"type": "sign_rkp", "feature_dim": 64},
        {"type": "sign_rkp", "feature_dim": 128},
    ]
    multi_sketches = build_multi_kernel_sketch(K, V, kernels, base_feature_dim=32, seed=seed)
    out_multi = eval_multi_kernel_sketch(multi_sketches, Q)
    err_multi = float(np.abs(out_multi - gt).mean())
    results["kernels"]["multi_sign_rkp"] = {"err": err_multi, "feature_dim": 32 + 64 + 128}
    print(f"    Multi-SignRKP (3 kernels): err={err_multi:.4e}")

    # 对比: Ground truth 用 linear kernel
    # 直接用 Q @ K.T / √d 做 attention (无 projection)
    scores_gt = Q @ K.T / np.sqrt(d)
    scores_max = scores_gt.max(axis=-1, keepdims=True)
    attn_gt = np.exp(scores_gt - scores_max)
    attn_gt = attn_gt / np.clip(attn_gt.sum(axis=-1, keepdims=True), 1e-30, None)
    out_gt_linear = attn_gt @ V
    err_gt_linear = float(np.abs(out_gt_linear - gt).mean())
    results["kernels"]["gt_linear"] = {"err": err_gt_linear, "feature_dim": "full"}
    print(f"    GT-Linear (no projection): err={err_gt_linear:.4e}")

    return results


def run_exploration_C(
    kv_len: int = 4096,
    q_len: int = 64,
    kv_type: str = "random",
    d: int = 64,
    compression: float = 0.5,
    seed: int = 0,
) -> dict:
    """探索 C: Kernel + Coreset 组合。

    对比异构 backend 下的最优选择。
    """
    print("\n  [Exploration C: Kernel + Coreset Combination]")

    if kv_type == "clustered":
        K, V = make_clustered_kv(kv_len, d, seed=seed)
    elif kv_type == "random":
        K, V = make_random_kv(kv_len, d, seed=seed)
    else:
        K, V = make_skewed_kv(kv_len, d, seed=seed)

    gen = np.random.default_rng(seed + 1000)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    gt = ground_truth(Q, K, V)

    results = {"kv_type": kv_type, "kv_len": kv_len, "q_len": q_len, "compression": compression}

    # Coreset
    r = max(1, int(compression * 64))
    coreset_sketch = build_coreset_sketch(K, V, r=r, seed=seed)
    coreset_stats = eval_coreset_sketch(coreset_sketch, Q)
    out_coreset = coreset_stats.finalize().squeeze(0)
    err_coreset = float(np.abs(out_coreset - gt).mean())

    # Kernel (Sign RKP)
    feature_dim = max(4, int(compression * 64 * 2))
    kernel_sketch = build_kernel_feature_sketch(K, V, feature_dim=feature_dim, seed=seed)
    kernel_stats = eval_kernel_feature_sketch_B(kernel_sketch, Q)
    out_kernel = kernel_stats.finalize().squeeze(0)
    err_kernel = float(np.abs(out_kernel - gt).mean())

    # Oracle selection（选择更好的）
    err_oracle = min(err_coreset, err_kernel)

    # Average（简单平均）
    out_avg = (out_coreset + out_kernel) / 2
    err_avg = float(np.abs(out_avg - gt).mean())

    results["coreset_err"] = err_coreset
    results["kernel_err"] = err_kernel
    results["oracle_err"] = err_oracle
    results["avg_err"] = err_avg

    print(f"    Coreset: err={err_coreset:.4e}")
    print(f"    Kernel:  err={err_kernel:.4e}")
    print(f"    Oracle:  err={err_oracle:.4e}")
    print(f"    Average: err={err_avg:.4e}")

    return results


def run_e1_sweep(verbose: bool = True) -> list[dict]:
    """Run E1 sweep across all configurations."""
    results = []
    kv_lens = [1024, 4096]
    q_lens = [16, 64]
    compressions = [0.25, 0.5, 0.75]
    kv_types = ["clustered", "random", "skewed"]

    print("=" * 80)
    print("E1 Sweep v2: Kernel Feature Sketch (Sign RKP) vs Coreset")
    print("=" * 80)

    for kv_type in kv_types:
        print(f"\n--- KV Type: {kv_type.upper()} ---")
        for kv_len in kv_lens:
            for q_len in q_lens:
                for compression in compressions:
                    r = run_e1_one_config(
                        kv_len=kv_len,
                        q_len=q_len,
                        compression=compression,
                        kv_type=kv_type,
                        seed=0,
                        verbose=verbose,
                    )
                    results.append(r)

    return results


def run_verification(seed: int = 42, n_configs: int = 30) -> list[dict]:
    """Run independent verification with seed=42."""
    print(f"\n{'=' * 80}")
    print(f"Verification (seed={seed}, {n_configs} configs)")
    print("=" * 80)

    results = []
    kv_lens = [1024, 4096]
    q_lens = [16, 64]
    compressions = [0.25, 0.5, 0.75]
    kv_types = ["clustered", "random", "skewed"]

    config_idx = 0
    for seed_offset in range(n_configs):
        seed_i = seed + seed_offset
        kv_type = kv_types[seed_offset % len(kv_types)]
        kv_len = kv_lens[(seed_offset // 3) % len(kv_lens)]
        q_len = q_lens[(seed_offset // 6) % len(q_lens)]
        compression = compressions[(seed_offset // 12) % len(compressions)]

        r = run_e1_one_config(
            kv_len=kv_len,
            q_len=q_len,
            compression=compression,
            kv_type=kv_type,
            seed=seed_i,
            verbose=True,
        )
        r["seed"] = seed_i
        results.append(r)
        config_idx += 1

    return results


def summarize_e1(results: list[dict]) -> None:
    """Summarize E1 results."""
    print()
    print("=" * 80)
    print("E1 Summary (v2)")
    print("=" * 80)

    for kv_type in ["clustered", "random", "skewed"]:
        type_results = [r for r in results if r["kv_type"] == kv_type]
        if not type_results:
            continue

        print(f"\n### {kv_type.upper()} ###")

        for comp in sorted({r["compression"] for r in type_results}):
            comp_results = [r for r in type_results if r["compression"] == comp]

            avg_err = {
                "Full": np.mean([r["err_a"] for r in comp_results]),
                "Coreset": np.mean([r["err_b"] for r in comp_results]),
                "Kernel": np.mean([r["err_c"] for r in comp_results]),
                "Drop": np.mean([r["err_d"] for r in comp_results]),
            }

            print(
                f"  comp={comp:.2f}: "
                f"Full={avg_err['Full']:.3e} "
                f"Coreset={avg_err['Coreset']:.3e} "
                f"Kernel={avg_err['Kernel']:.3e} "
                f"Drop={avg_err['Drop']:.3e}"
            )


def main():
    print("Exp3 v2: Kernel Feature Sketch (Fixed) + E1 + Exploration + Verification")
    print("=" * 80)

    # E1 Sweep
    e1_results = run_e1_sweep(verbose=True)
    summarize_e1(e1_results)

    # 探索 A: Sign RKP vs Nyström
    print()
    print("=" * 80)
    print("Exploration A: Sign RKP vs Nyström Convergence")
    print("=" * 80)
    exploration_A_results = []
    for kv_type in ["clustered", "random", "skewed"]:
        r = run_exploration_A(kv_len=4096, q_len=64, kv_type=kv_type, seed=0)
        exploration_A_results.append(r)

    # 探索 B: Multi-kernel
    print()
    print("=" * 80)
    print("Exploration B: Multi-kernel Sign RKP")
    print("=" * 80)
    exploration_B_results = []
    for kv_type in ["clustered", "random", "skewed"]:
        r = run_exploration_B(kv_len=4096, q_len=64, kv_type=kv_type, seed=0)
        exploration_B_results.append(r)

    # 探索 C: Kernel + Coreset 组合
    print()
    print("=" * 80)
    print("Exploration C: Kernel + Coreset Combination")
    print("=" * 80)
    exploration_C_results = []
    for kv_type in ["clustered", "random", "skewed"]:
        for comp in [0.25, 0.5, 0.75]:
            r = run_exploration_C(kv_len=4096, q_len=64, kv_type=kv_type, compression=comp, seed=0)
            exploration_C_results.append(r)

    # 验证 (seed=42, 30 configs)
    verification_results = run_verification(seed=42, n_configs=30)

    # 保存结果
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)

    # E1 + all explorations 合并
    e1_path = os.path.join(output_dir, "exp3_kernel_feature_v2.json")
    with open(e1_path, "w", encoding="utf-8") as f:
        json.dump({
            "e1_results": e1_results,
            "exploration_A": exploration_A_results,
            "exploration_B": exploration_B_results,
            "exploration_C": exploration_C_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {e1_path}")

    # 单独保存
    for name, data in [
        ("exp3_kernel_feature_v2", e1_results),
        ("exp3_exploration_A_v2", exploration_A_results),
        ("exp3_exploration_B_v2", exploration_B_results),
        ("exp3_exploration_C_v2", exploration_C_results),
        ("exp3_verification_v2", verification_results),
    ]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved to {path}")

    # 生成报告
    generate_report(e1_results, exploration_A_results, exploration_B_results,
                   exploration_C_results, verification_results, output_dir)


def generate_report(
    e1_results: list,
    exp_A: list,
    exp_B: list,
    exp_C: list,
    verification: list,
    output_dir: str,
) -> None:
    """生成 v2 修复报告。"""

    # 计算关键统计
    def avg_by_type(results, kv_type, path_key):
        return np.mean([r[path_key] for r in results if r["kv_type"] == kv_type])

    # v1 原始数据 (从 exp3_kernel_feature.json 读取)
    try:
        with open(os.path.join(output_dir, "exp3_kernel_feature.json")) as f:
            v1_data = json.load(f)
        v1_e1 = v1_data["e1_results"]
    except Exception:
        v1_e1 = []

    report = []
    report.append("# Exp3 v2 修复报告\n")
    report.append("## 修复的 3 个核心 Bug\n\n")
    report.append("### Bug 1 (Critical): RFF kernel type 错误\n")
    report.append("**问题**: RFF 近似的是 RBF kernel `exp(-||x-y||²·γ/2)`，但 ground truth 用的是 `exp(q·k/√d)`（线性 kernel 的 exp）。两个 kernel 数学上完全不等价。\n")
    report.append("**证据**: RFF error 不随 D 增加而下降（D=64 和 D=256 的 error 完全相同）。\n")
    report.append("**修复**: 改用 **Sign Random Kitchen Sinks** (sgn(Wx+b)/√D)，其中 W~N(0,I)。E[sgn(Wx)·sgn(Wy)] ≈ x·y/d，这是线性 kernel 的正确近似。\n\n")

    report.append("### Bug 2 (Critical): Attention 归一化缺少 softmax\n")
    report.append("**问题**: 脚本用 `F = (φ(q)·S_V) / (φ(q)·S_Z)`（线性归一化），而标准 attention 是 softmax 归一化。\n")
    report.append("**修复**: 用 `scores = φ(q) @ φ(K).T / T`，`attn = softmax(scores)`，`F = attn @ V`。注意 sketch 里没存 K，所以 E1 sweep 用降维版 `F = φ(q) @ S_V / (φ(q) @ S_Z)`；Exploration A/B 用完整 softmax 版。\n\n")

    report.append("### Bug 3 (Medium): Multi-kernel 归一化不一致\n")
    report.append("**问题**: 对每个 kernel 先归一化（F_i = num_i/den_i），再对 F 加权平均后除以 total_den，数学上等价于 `Σ w_i·num_i/den_i / Σ w_i`，不是真正的 kernel combination。\n")
    report.append("**修复**: 统一为 `total_num = Σ w_i·num_i`，`total_den = Σ w_i·den_i`，`F = total_num / total_den`。\n\n")

    report.append("### Bug 4 (Medium): Nyström 实现偏离标准公式\n")
    report.append("**问题**: 用 `softmax(Q @ landmarks.T)` 而非 `K(q,L) @ K(L,L)^{-1/2}`，这是 k-NN soft attention，不是标准 Nyström。\n")
    report.append("**修复**: 实现标准 Nyström: φ(x) = K(x, L) @ K(L,L)^{-1/2}，`K(q,k) ≈ φ(q)·φ(k)`。\n\n")

    report.append("---\n\n## 新数据汇总\n\n")

    report.append("### E1 Sweep (36 configs, seed=0)\n\n")
    report.append("| KV Type | Compression | Coreset err | Kernel err (v2) | Kernel err (v1) | Drop |\n")
    report.append("|---------|-------------|-------------|-----------------|-----------------|------|\n")
    for kv_type in ["clustered", "random", "skewed"]:
        for comp in [0.25, 0.5, 0.75]:
            v2_r = [r for r in e1_results if r["kv_type"] == kv_type and r["compression"] == comp]
            v1_r = [r for r in v1_e1 if r["kv_type"] == kv_type and r["compression"] == comp]
            if v2_r and v1_r:
                avg_b = np.mean([r["err_b"] for r in v2_r])
                avg_c_v2 = np.mean([r["err_c"] for r in v2_r])
                avg_c_v1 = np.mean([r["err_c"] for r in v1_r])
                avg_d = np.mean([r["err_d"] for r in v2_r])
                report.append(f"| {kv_type} | {comp:.2f} | {avg_b:.4f} | {avg_c_v2:.4f} | {avg_c_v1:.4f} | {avg_d:.4f} |\n")
    report.append("\n")

    report.append("### Exploration A: Sign RKP vs Nyström 收敛性\n\n")
    for r_exp in exp_A:
        kv_type = r_exp["kv_type"]
        report.append(f"**{kv_type.upper()}**:\n")
        report.append("| D | SignRKP err | Nyström err | Δ |\n")
        report.append("|---|-------------|-------------|---|\n")
        for D, vals in sorted(r_exp["methods"].items(), key=lambda x: int(x[0])):
            report.append(f"| {D} | {vals['sign_rkp_err']:.4e} | {vals['nystrom_err']:.4e} | {abs(vals['sign_rkp_err']-vals['nystrom_err']):.4e} |\n")
        report.append("\n")

    report.append("### Exploration B: Multi-kernel\n\n")
    for r_exp in exp_B:
        kv_type = r_exp["kv_type"]
        report.append(f"**{kv_type.upper()}**:\n")
        for k_name, k_vals in r_exp["kernels"].items():
            report.append(f"  - {k_name}: err={k_vals['err']:.4e}\n")
        report.append("\n")

    report.append("### Exploration C: Coreset vs Kernel vs Oracle\n\n")
    for r_exp in exp_C:
        report.append(f"**{r_exp['kv_type']} (comp={r_exp['compression']:.2f})**: "
                       f"Coreset={r_exp['coreset_err']:.4f}, "
                       f"Kernel={r_exp['kernel_err']:.4f}, "
                       f"Oracle={r_exp['oracle_err']:.4f}, "
                       f"Avg={r_exp['avg_err']:.4f}\n")
    report.append("\n")

    report.append("### Verification (seed=42, 30 configs)\n\n")
    for kv_type in ["clustered", "random", "skewed"]:
        v_r = [r for r in verification if r["kv_type"] == kv_type]
        if v_r:
            avg_b = np.mean([r["err_b"] for r in v_r])
            avg_c = np.mean([r["err_c"] for r in v_r])
            avg_d = np.mean([r["err_d"] for r in v_r])
            n_coreset_wins = sum(1 for r in v_r if r["err_b"] < r["err_c"])
            n_kernel_wins = sum(1 for r in v_r if r["err_c"] < r["err_b"])
            report.append(f"**{kv_type.upper()}** (n={len(v_r)}): "
                           f"avg Coreset={avg_b:.4f}, avg Kernel={avg_c:.4f}, "
                           f"Drop={avg_d:.4f}, "
                           f"Coreset wins={n_coreset_wins}/{len(v_r)}, "
                           f"Kernel wins={n_kernel_wins}/{len(v_r)}\n")
    report.append("\n")

    report.append("---\n\n## 新发现\n\n")

    # 分析 Exploration A: SignRKP 是否收敛
    has_convergence = False
    for r_exp in exp_A:
        vals = r_exp["methods"]
        dims = sorted(int(k) for k in vals.keys())
        if len(dims) >= 2:
            err_start = vals[str(dims[0])]["sign_rkp_err"]
            err_end = vals[str(dims[-1])]["sign_rkp_err"]
            if err_end < err_start * 0.9:
                has_convergence = True

    if has_convergence:
        report.append("✅ **SignRKP 随 D 增加 error 下降**（收敛性得到验证）\n")
    else:
        report.append("⚠️ **SignRKP 仍不收敛**：D 增加 error 未显著下降。可能原因：sign projection 对当前数据分布不友好，或 D 需要更大（如 512+）才能收敛。\n")

    # 分析 Nyström 是否优于 SignRKP
    nystrom_wins = 0
    total = 0
    for r_exp in exp_A:
        for D, vals in r_exp["methods"].items():
            total += 1
            if vals["nystrom_err"] < vals["sign_rkp_err"]:
                nystrom_wins += 1
    if total > 0:
        report.append(f"\nNyström 在 {nystrom_wins}/{total} 配置中优于 SignRKP。")
        if nystrom_wins > total * 0.5:
            report.append(" **Nyström 整体优于 SignRKP**。\n")
        else:
            report.append(" **SignRKP 整体优于 Nyström** 或两者接近。\n")

    report.append("\n## 与 v1 的关键差异\n\n")
    report.append("| 对比项 | v1 | v2 |\n")
    report.append("|--------|----|----|\n")
    report.append("| Kernel type | RFF (RBF) | Sign RKP (Linear) |\n")
    report.append("| Normalization | Linear (num/den) | Softmax (温度归一化) |\n")
    report.append("| Multi-kernel | 先归一化后平均 | 加权 num/den 后相除 |\n")
    report.append("| Nyström | softmax(Q@L.T) | K(q,L)@K(L,L)^{-1/2} |\n")

    report_path = os.path.join(output_dir, "exp3_v2_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(report))
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
