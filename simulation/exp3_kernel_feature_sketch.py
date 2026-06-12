"""
Exp3: Kernel Feature Sketch — Random Fourier Features 近似 softmax attention
+ 跟 Coreset 的 Pareto 对比 + 创新探索

核心思想：
    exp(q·k / sqrt(d)) ≈ φ(q)^T φ(k)

预存:
    S_V = Σ_i φ(k_i) v_i^T   # [feature_dim, d]
    S_Z = Σ_i φ(k_i)         # [feature_dim]

decode 时:
    F(q) ≈ φ(q)^T S_V / φ(q)^T S_Z

三种 K/V 数据结构：
    - Clustered: K 有明显 cluster 结构（coreset 优势）
    - Random: K 随机分布（kernel feature 优势）
    - Skewed: 少数 outlier + 大量 normal（两者都一般）

创新探索：
    A: RFF vs Nyström approximation
    B: Multi-kernel feature sketch (RBF σ=1 + σ=0.1 + polynomial)
    C: Kernel + Coreset 组合（heterogeneous backend）
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    ground_truth,
    serve_local,
)

# ============== 类型别名 ==============

@dataclass
class KernelSketch:
    """Kernel feature sketch 容器"""
    S_V: np.ndarray  # [feature_dim, d]
    S_Z: np.ndarray  # [feature_dim]
    rff_w: np.ndarray  # [d, feature_dim]
    rff_b: np.ndarray  # [feature_dim]
    kernel_type: str
    feature_dim: int
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
    """生成 cluster 结构的 KV：K 有明显聚类中心。
    
    先采样 n_clusters 个 centroid，再用 Gaussian noise 生成每个 token。
    V 跟 K 相关（模拟真实 KV cache 的语义关系）。
    """
    gen = np.random.default_rng(seed)
    
    # 采样聚类中心（中心要离得够远）
    centroids = []
    for _ in range(n_clusters):
        # rejection sampling: 新中心要跟已有中心距离足够远
        for _ in range(100):
            c = gen.standard_normal(d) * 2.0  # 中心 spread 在 ±6
            if all(np.linalg.norm(c - oc) > 3.0 for oc in centroids):
                centroids.append(c)
                break
        if len(centroids) <= len(centroids):
            break
    
    # 如果 rejection 失败，直接随机采样
    while len(centroids) < n_clusters:
        centroids.append(gen.standard_normal(d) * 2.0)
    
    centroids = np.array(centroids)  # [n_clusters, d]
    
    # 每个 token 分配 cluster，然后加 noise
    cluster_assignments = gen.integers(0, n_clusters, size=kv_len)
    K = centroids[cluster_assignments] + gen.standard_normal((kv_len, d)) * cluster_std
    
    # V 跟 K 相关（线性变换 + noise）
    # V = K @ W + noise，W 随机投影矩阵
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
    """生成 skew 结构的 KV：少数 outlier + 大量 normal。
    
    这模拟了 attention 中少数特殊 token（如 [SEP], [CLS]）跟大多数普通 token 的差异。
    """
    gen = np.random.default_rng(seed)
    
    # 少数 outlier token（spread 很广）
    outlier_K = gen.standard_normal((n_outliers, d)) * outlier_std
    outlier_V = gen.standard_normal((n_outliers, d)) * outlier_std
    
    # 大量 normal token（cluster 在原点附近）
    normal_K = gen.standard_normal((kv_len - n_outliers, d)) * normal_std
    normal_V = gen.standard_normal((kv_len - n_outliers, d)) * normal_std
    
    # 随机打乱顺序
    K = np.concatenate([outlier_K, normal_K], axis=0)
    V = np.concatenate([outlier_V, normal_V], axis=0)
    
    # 随机 shuffle
    perm = gen.permutation(kv_len)
    K = K[perm]
    V = V[perm]
    
    return K.astype(np.float32), V.astype(np.float32)


# ============== Kernel Feature Sketch 实现 ==============

def rbf_kernel_scale(d: int) -> float:
    """RBF kernel 的 scale 参数（常见的 gamma = 1/d）。"""
    return 1.0 / d


def build_rff_features(
    X: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """计算 Random Fourier Features。
    
    φ(x) = sqrt(2/D) * [cos(w_i^T x + b_i), sin(w_i^T x + b_i)]_{i=1..D}
    
    对于 RFF (cos only variant):
    φ(x) = sqrt(2/D) * cos(w^T x + b)
    
    Parameters
    ----------
    X : [N, d] 或 [d]
    w : [d, D]
    b : [D]
    
    Returns
    -------
    phi : [N, D] 或 [D]
    """
    if X.ndim == 1:
        X = X[None, :]
    
    # X @ w: [N, d] @ [d, D] = [N, D]
    projections = X @ w + b  # [N, D]
    phi = np.sqrt(2.0 / w.shape[1]) * np.cos(projections)
    return phi


def build_kernel_feature_sketch(
    K: np.ndarray,
    V: np.ndarray,
    feature_dim: int = 64,
    kernel_scale: float = 1.0,
    seed: int = 0,
) -> KernelSketch:
    """Build kernel feature sketch using Random Fourier Features.
    
    RBF kernel: K(x, y) = exp(-γ ||x - y||² / 2)
    对应的 RFF: φ(x) = sqrt(2/D) * cos(w^T x + b)，其中 w ~ N(0, γ I)
    
    Parameters
    ----------
    K : [kv_len, d]
    V : [kv_len, d]
    feature_dim : 特征维度（D）
    kernel_scale : γ 值（RBF kernel 的 bandwidth）
    seed : 随机种子
    
    Returns
    -------
    KernelSketch with S_V, S_Z, rff_w, rff_b
    """
    kv_len, d = K.shape
    
    gen = np.random.default_rng(seed)
    
    # RFF: w ~ N(0, γ I)，γ = kernel_scale / d
    # 避免数值爆炸：需要归一化 K 和 V
    gamma = kernel_scale / d
    
    # 归一化 K（重要！）
    K_norm = K / (np.linalg.norm(K, axis=-1, keepdims=True) + 1e-8)
    Q_norm = V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-8)  # V 也归一化
    
    w = gen.standard_normal((d, feature_dim)) * np.sqrt(gamma)
    b = gen.uniform(0, 2 * np.pi, size=feature_dim)
    
    # φ(k_i): [kv_len, feature_dim]
    phi_K = build_rff_features(K_norm, w, b)  # [kv_len, feature_dim]
    
    # S_V = Σ_i φ(k_i) v_i^T: [feature_dim, d]
    S_V = phi_K.T @ V  # [feature_dim, kv_len] @ [kv_len, d] = [feature_dim, d]
    
    # S_Z = Σ_i φ(k_i): [feature_dim]
    S_Z = phi_K.sum(axis=0)  # [feature_dim]
    
    return KernelSketch(
        S_V=S_V,
        S_Z=S_Z,
        rff_w=w,
        rff_b=b,
        kernel_type="rff",
        feature_dim=feature_dim,
        d=d,
    )


def build_nystrom_sketch(
    K: np.ndarray,
    V: np.ndarray,
    n_landmarks: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nyström approximation：用数据点做 landmark + SVD。
    
    K̂ ≈ K_{:, L} @ K_{L, L}^{-1} @ K_{L, :}
    
    这里简化：用 landmark 的 K 矩阵做近似，然后存 Σ s_i φ(l_i) v_i^T。
    
    Returns
    -------
    S_V: [n_landmarks, d]
    S_Z: [n_landmarks]
    landmarks: [n_landmarks, d] - 保存 landmark K 用于 query
    """
    kv_len, d = K.shape
    gen = np.random.default_rng(seed)
    
    # 随机采样 landmark
    landmark_idx = gen.choice(kv_len, size=n_landmarks, replace=False)
    landmarks = K[landmark_idx]  # [n_landmarks, d]
    
    # K_LL = landmarks @ landmarks.T
    K_LL = landmarks @ landmarks.T  # [n_landmarks, n_landmarks]
    
    # Nyström feature: φ(l_i) = K_{:, l_i} @ K_{L, L}^{-1/2}
    # 简化：直接用 K 矩阵近似 + normalize
    # 更精确的做法是 eigendecomposition
    try:
        # K_LL^{-1/2} via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(K_LL)
        eigvals = np.maximum(eigvals, 1e-10)
        K_LL_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    except np.linalg.LinAlgError:
        # fallback: pseudo inverse
        K_LL_inv_sqrt = np.linalg.pinv(K_LL)
    
    # φ(L) = K_LL^{1/2} (简化版本)
    phi_L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-10)))  # [n_landmarks, n_landmarks]
    
    # 对于每个数据点，计算其 feature
    # φ(k) = k^T @ φ(L) (近似)
    # 更简单：直接用 K 跟 landmark 的相似度作为 feature
    phi_L_full = landmarks  # [n_landmarks, d]
    
    # S_V = Σ_i φ(k_i) v_i^T
    # φ(k_i) = softmax(K @ K_L^T) (简化)
    K_KL = K @ phi_L_full.T  # [kv_len, n_landmarks]
    # row-wise softmax
    K_KL_max = K_KL.max(axis=-1, keepdims=True)
    phi_K = np.exp(K_KL - K_KL_max)
    phi_K = phi_K / (phi_K.sum(axis=-1, keepdims=True) + 1e-10)
    
    S_V = phi_K.T @ V  # [n_landmarks, d]
    S_Z = phi_K.sum(axis=0)  # [n_landmarks]
    
    return S_V, S_Z, landmarks


def eval_kernel_feature_sketch_B(
    sketch: KernelSketch,
    Q: np.ndarray,
) -> NumpyAttnStats:
    """Evaluate kernel feature sketch - Implementation B (m=0, l, y).
    
    φ(q) [Q_len, feature_dim]
    num = φ(q) @ S_V [Q_len, d]
    den = φ(q) @ S_Z [Q_len]
    F(q) = num / den[..., None]  # normalized output
    
    然后反推 (m=0, l=den, y=F(q)*l)
    """
    # 归一化 Q（与 build 时一致）
    Q_norm = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-8)
    phi_q = build_rff_features(Q_norm, sketch.rff_w, sketch.rff_b)  # [Q_len, feature_dim]
    
    # num = φ(q) @ S_V: [Q_len, d]
    num = phi_q @ sketch.S_V
    
    # den = φ(q) @ S_Z: [Q_len]
    den = phi_q @ sketch.S_Z
    
    # 防止除以 0 或极小值
    den_safe = np.clip(den, 1e-6, None)
    
    # F(q) = num / den[..., None]: [Q_len, d]
    F = num / den_safe[..., None]
    
    # Implementation B: m=0, l=den, y=F*l
    q_len, d = Q.shape
    H = 1
    
    m = np.zeros((H, q_len, 1), dtype=np.float32)
    l = den_safe[None, :, None]  # [1, Q_len, 1]
    y = F[None, :, :] * l  # [1, Q_len, d]
    
    return NumpyAttnStats(m=m, l=l, y=y)


def eval_nystrom_sketch(
    S_V: np.ndarray,
    S_Z: np.ndarray,
    landmarks: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    """Evaluate Nyström sketch。
    
    φ(q) = softmax(q @ landmarks^T)  [Q_len, n_landmarks]
    F(q) = φ(q) @ S_V / (φ(q) @ S_Z + eps)
    """
    Q_len, d = Q.shape
    n_landmarks = landmarks.shape[0]
    
    # φ(q) = softmax(q @ landmarks^T)
    Q_KL = Q @ landmarks.T  # [Q_len, n_landmarks]
    Q_KL_max = Q_KL.max(axis=-1, keepdims=True)
    phi_q = np.exp(Q_KL - Q_KL_max)
    phi_q = phi_q / (phi_q.sum(axis=-1, keepdims=True) + 1e-10)
    
    # num = φ(q) @ S_V: [Q_len, d]
    num = phi_q @ S_V
    
    # den = φ(q) @ S_Z: [Q_len]
    den = phi_q @ S_Z
    
    # F(q) = num / den[..., None]
    F = num / np.clip(den[..., None], 1e-30, None)
    
    return F


# ============== Coreset Sketch 实现 ==============

def build_coreset_sketch(
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    seed: int = 0,
) -> CoresetSketch:
    """Build coreset sketch via k-means++ (简化版).
    
    Parameters
    ----------
    K : [kv_len, d]
    V : [kv_len, d]
    r : centroids 数量
    seed : 随机种子
    """
    kv_len, d = K.shape
    gen = np.random.default_rng(seed)
    
    # k-means++ 初始化
    # 1. 随机选第一个 centroid
    idx = gen.integers(0, kv_len)
    centroids_K = [K[idx].copy()]
    centroids_V = [V[idx].copy()]
    
    # 2. 迭代选剩下 r-1 个
    for _ in range(r - 1):
        # 计算每个点到最近 centroid 的距离
        dists = np.array([
            min(np.linalg.norm(k - c) ** 2 for c in centroids_K)
            for k in K
        ])
        # 概率 proportional to distance²
        probs = dists / dists.sum()
        idx = gen.choice(kv_len, p=probs)
        centroids_K.append(K[idx].copy())
        centroids_V.append(V[idx].copy())
    
    centroids_K = np.array(centroids_K)  # [r, d]
    centroids_V = np.array(centroids_V)  # [r, d]
    
    return CoresetSketch(
        centroids_K=centroids_K,
        centroids_V=centroids_V,
        r=r,
    )


def eval_coreset_sketch(
    sketch: CoresetSketch,
    Q: np.ndarray,
) -> NumpyAttnStats:
    """Evaluate coreset sketch。
    
    用 weighted attention：
    F(q) = Σ_i w_i * v_i * exp(q · k_i) / Σ_i w_i * exp(q · k_i)
    
    这里 w_i = 1（简化），实际可以用 k-means count 作为 weight。
    """
    q_len, d = Q.shape
    r = sketch.r
    
    # scores: [q_len, r]
    scores = Q @ sketch.centroids_K.T
    
    # softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - scores_max)  # [q_len, r]
    l = p.sum(axis=-1, keepdims=True)  # [q_len, 1]
    p_norm = p / np.clip(l, 1e-30, None)  # [q_len, r]
    
    # F(q) = p_norm @ V_coreset: [q_len, d]
    F = p_norm @ sketch.centroids_V  # [q_len, d]
    
    # 包装成 NumpyAttnStats: m=0, l=l, y=F*l
    H = 1
    m = np.zeros((H, q_len, 1), dtype=np.float32)
    l_out = l[None, :, 0:1]  # [1, q_len, 1] - l is [q_len, 1]
    y = F[None, :, :] * l_out  # [1, q_len, d] * [1, q_len, 1] -> [1, q_len, d]
    
    return NumpyAttnStats(m=m, l=l_out, y=y)


# ============== Multi-Kernel Feature Sketch ==============

def build_multi_kernel_sketch(
    K: np.ndarray,
    V: np.ndarray,
    kernels: list[dict],
    base_feature_dim: int = 32,
    seed: int = 0,
) -> list[KernelSketch]:
    """Build multi-kernel feature sketch。
    
    kernels: [
        {"type": "rff", "scale": 1.0},
        {"type": "rff", "scale": 0.1},
    ]
    """
    sketches = []
    
    for i, kern in enumerate(kernels):
        kern_type = kern.get("type", "rff")
        kern_seed = seed + i * 100
        
        if kern_type == "rff":
            scale = kern.get("scale", 1.0)
            sketch = build_kernel_feature_sketch(
                K, V,
                feature_dim=base_feature_dim,
                kernel_scale=scale,
                seed=kern_seed,
            )
            sketches.append(sketch)
        
        elif kern_type == "linear":
            # Linear kernel: K(x, y) = x^T y
            # Feature map: φ(x) = x (normalized)
            K_norm = K / (np.linalg.norm(K, axis=-1, keepdims=True) + 1e-8)
            gen = np.random.default_rng(kern_seed)
            
            # 随机投影到 base_feature_dim
            W = gen.standard_normal((K.shape[1], base_feature_dim)) * 0.1
            phi_K = K_norm @ W
            S_V = phi_K.T @ V
            S_Z = phi_K.sum(axis=0)
            
            sketch = KernelSketch(
                S_V=S_V,
                S_Z=S_Z,
                rff_w=W,
                rff_b=np.zeros(base_feature_dim),
                kernel_type="linear",
                feature_dim=base_feature_dim,
                d=K.shape[1],
            )
            sketches.append(sketch)
    
    return sketches


def eval_multi_kernel_sketch(
    sketches: list[KernelSketch],
    Q: np.ndarray,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Evaluate multi-kernel sketch（加权组合）。
    
    需要对每个 sketch 单独归一化 Q。
    """
    if weights is None:
        weights = [1.0 / len(sketches)] * len(sketches)
    
    total_F = None
    total_den = None
    
    for sketch, w in zip(sketches, weights):
        # 归一化 Q
        Q_norm = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-8)
        phi_q = build_rff_features(Q_norm, sketch.rff_w, sketch.rff_b)
        num = phi_q @ sketch.S_V
        den = phi_q @ sketch.S_Z
        
        # 防止除以 0
        den_safe = np.clip(np.abs(den), 1e-6, None)
        F = num / den_safe[..., None]
        
        if total_F is None:
            total_F = w * F
            total_den = w * den_safe
        else:
            total_F += w * F
            total_den += w * den_safe
    
    return total_F / np.clip(total_den[..., None], 1e-6, None)


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
    """跑单组配置，返回 4 路径的指标。
    
    compression: sketch 压缩率（相对于 block_size 或 feature_dim）
    """
    num_blocks = kv_len // block_size
    baseline = block_size  # coreset 的基准
    
    # 生成 K/V
    if kv_type == "clustered":
        K, V = make_clustered_kv(kv_len, d, seed=seed)
    elif kv_type == "random":
        K, V = make_random_kv(kv_len, d, seed=seed)
    else:  # skewed
        K, V = make_skewed_kv(kv_len, d, seed=seed)
    
    # Q
    gen = np.random.default_rng(seed + 1000)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    # Ground truth
    gt = ground_truth(Q, K, V)  # [q_len, d]
    
    # === A. Full KV ===
    out_a = gt.copy()
    
    # === B. Coreset Sketch ===
    r = max(1, int(compression * baseline))
    r = min(r, kv_len)  # 不能超过 KV 长度
    coreset_sketch = build_coreset_sketch(K, V, r=r, seed=seed)
    coreset_stats = eval_coreset_sketch(coreset_sketch, Q)
    out_b = coreset_stats.finalize().squeeze(0)
    
    # === C. Kernel Feature Sketch ===
    feature_dim = max(4, int(compression * baseline * 2))  # kernel 用更多 feature
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
    """探索 A: RFF vs Nyström approximation。
    
    对比两种近似方法在不同 feature_dim 下的精度。
    """
    print("\n  [Exploration A: RFF vs Nyström]")
    
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
    
    for n_features in [16, 32, 64, 128]:
        # RFF
        rff_sketch = build_kernel_feature_sketch(K, V, feature_dim=n_features, seed=seed)
        rff_stats = eval_kernel_feature_sketch_B(rff_sketch, Q)
        out_rff = rff_stats.finalize().squeeze(0)
        err_rff = float(np.abs(out_rff - gt).mean())
        
        # Nyström
        S_V, S_Z, landmarks = build_nystrom_sketch(K, V, n_landmarks=n_features, seed=seed)
        out_nystrom = eval_nystrom_sketch(S_V, S_Z, landmarks, Q)
        err_nystrom = float(np.abs(out_nystrom - gt).mean())
        
        results["methods"][n_features] = {
            "rff_err": err_rff,
            "nystrom_err": err_nystrom,
        }
        
        print(
            f"    dim={n_features:>4}  RFF={err_rff:.4e}  Nyström={err_nystrom:.4e}  "
            f"Δ={abs(err_rff - err_nystrom):.4e}"
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
    
    组合多个 kernel（RBF σ=1 + σ=0.1 + poly）看精度提升。
    """
    print("\n  [Exploration B: Multi-kernel Feature Sketch]")
    
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
    for scale in [0.1, 1.0, 10.0]:
        sketch = build_kernel_feature_sketch(K, V, feature_dim=64, kernel_scale=scale, seed=seed)
        stats = eval_kernel_feature_sketch_B(sketch, Q)
        out = stats.finalize().squeeze(0)
        err = float(np.abs(out - gt).mean())
        results["kernels"][f"rff_scale={scale}"] = {"err": err, "feature_dim": 64}
        print(f"    RFF (scale={scale}): err={err:.4e}")
    
    # Multi-kernel
    kernels = [
        {"type": "rff", "scale": 0.1},
        {"type": "rff", "scale": 1.0},
        {"type": "rff", "scale": 10.0},
    ]
    multi_sketches = build_multi_kernel_sketch(K, V, kernels, base_feature_dim=32, seed=seed)
    out_multi = eval_multi_kernel_sketch(multi_sketches, Q)
    err_multi = float(np.abs(out_multi - gt).mean())
    results["kernels"]["multi_rff"] = {"err": err_multi, "feature_dim": 32 * 3}
    print(f"    Multi-RFF (3 kernels): err={err_multi:.4e}")
    
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
    
    同一个 block 同时有两种 sketch，用 validity 决定选哪个。
    这里简化：直接对比两者的误差，选择误差更小的。
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
    
    # Kernel
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
    kv_lens = [1024, 4096]  # 减少到 2 个
    q_lens = [16, 64]
    compressions = [0.25, 0.5, 0.75]
    kv_types = ["clustered", "random", "skewed"]
    
    print("=" * 80)
    print("E1 Sweep: Kernel Feature Sketch vs Coreset")
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


def summarize_e1(results: list[dict]) -> None:
    """Summarize E1 results."""
    print()
    print("=" * 80)
    print("E1 Summary")
    print("=" * 80)
    
    # 按 kv_type 分组
    for kv_type in ["clustered", "random", "skewed"]:
        type_results = [r for r in results if r["kv_type"] == kv_type]
        if not type_results:
            continue
        
        print(f"\n### {kv_type.upper()} ###")
        
        # 按 compression 分组
        for comp in sorted({r["compression"] for r in type_results}):
            comp_results = [r for r in type_results if r["compression"] == comp]
            
            # 计算每个路径的平均误差
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
    print("Exp3: Kernel Feature Sketch + E1 Pareto + Exploration")
    print("=" * 80)
    
    # E1 Sweep
    e1_results = run_e1_sweep(verbose=True)
    summarize_e1(e1_results)
    
    # 探索 A: RFF vs Nyström
    print()
    print("=" * 80)
    print("Exploration A: RFF vs Nyström")
    print("=" * 80)
    exploration_A_results = []
    for kv_type in ["clustered", "random", "skewed"]:
        r = run_exploration_A(kv_len=4096, q_len=64, kv_type=kv_type, seed=0)
        exploration_A_results.append(r)
    
    # 探索 B: Multi-kernel
    print()
    print("=" * 80)
    print("Exploration B: Multi-kernel Feature Sketch")
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
    
    # 保存结果
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # E1 结果
    e1_path = os.path.join(output_dir, "exp3_kernel_feature.json")
    with open(e1_path, "w", encoding="utf-8") as f:
        json.dump({
            "e1_results": e1_results,
            "exploration_A": exploration_A_results,
            "exploration_B": exploration_B_results,
            "exploration_C": exploration_C_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {e1_path}")
    
    # 分别保存探索结果
    for name, data in [
        ("exp3_exploration_A_rff_nystrom", exploration_A_results),
        ("exp3_exploration_B_multikernel", exploration_B_results),
        ("exp3_exploration_C_combination", exploration_C_results),
    ]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved to {path}")


if __name__ == "__main__":
    main()
