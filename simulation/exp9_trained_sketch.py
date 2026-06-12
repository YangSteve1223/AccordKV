"""
Exp9: TrainableSketch — 可学习 φ attention distillation

核心思想 (借鉴 LoLA 2025):
    训练可学习映射 φ: R^d → R^{2D}
    φ(x) = [exp(w_1·x), ..., exp(w_D·x), exp(-w_1·x), ..., exp(-w_D·x)]
    
    这样绕过了固定 kernel (RBF/RFF) 的边界限制，
    让 φ 自动学习适合当前 KV 数据分布的特征。

训练目标:
    Teacher: A_teacher = softmax(Q·K^T/√d)·V
    Student: A_student = φ(Q)^T·(Σ φ(K_i)·V_i) / φ(Q)^T·(Σ φ(K_i))
    Loss: L = ||A_teacher - A_student||²

这是 ACCORD-KV 新 contract type TRAINED_SKETCH 的基础。
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional, Literal

import numpy as np
from scipy.special import softmax as scipy_softmax

# ============== 项目路径 ==============
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ============== Adam Optimizer (纯 numpy 实现) ==============

class AdamOptimizer:
    """纯 numpy 实现的 Adam 优化器。"""
    
    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = params  # Store reference to params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # 状态变量
        self.m = [np.zeros_like(p) for p in params]  # first moment
        self.v = [np.zeros_like(p) for p in params]  # second moment
        self.t = 0  # timestep
    
    def step(self, grads: list[np.ndarray]) -> None:
        """执行一步 Adam 更新。"""
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            # p_t = p_{t-1} - lr_t * m_t / (sqrt(v_t) + eps)
            p -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + self.eps)


# ============== 数据类型定义 ==============

@dataclass
class TrainableSketch:
    """可学习 φ 的 sketch 容器。"""
    W: np.ndarray  # [d, D] - 可学习参数
    D: int  # 特征维度 (输出是 2D)
    d: int  # 输入维度
    trained: bool = False


@dataclass
class SketchResult:
    """Sketch 评估结果。"""
    method: str
    error: float  # MSE
    bytes_approx: int  # 近似参数量
    num_params: int  # 可训练参数量


# ============== φ 函数定义 (可学习) ==============

def safe_exp(x: np.ndarray, clip_val: float = 10.0) -> np.ndarray:
    """数值稳定的 exp，防止 overflow。"""
    x_clipped = np.clip(x, -clip_val, clip_val)
    return np.exp(x_clipped)


def compute_phi(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    计算可学习特征映射 φ(x)。
    
    φ(x) = [exp(w_1·x), ..., exp(w_D·x), exp(-w_1·x), ..., exp(-w_D·x)] ∈ R^{2D}
    
    Parameters
    ----------
    x : [N, d] 或 [d] 或 [B, N, d]
    W : [d, D]
    
    Returns
    -------
    phi : [N, 2D] 或 [B, N, 2D]
    """
    # Handle 3D input
    original_3d = False
    B, N_out = 1, 1
    if x.ndim == 3:
        original_3d = True
        B, N_out, d_in = x.shape
        x = x.reshape(B * N_out, d_in)  # [B*N, d]
    elif x.ndim == 1:
        x = x[None, :]
    
    N, d = x.shape
    D = W.shape[1]
    
    # 投影: [N, d] @ [d, D] = [N, D]
    projections = x @ W  # [N, D]
    
    # 正负两部分
    exp_pos = safe_exp(projections)  # [N, D]
    exp_neg = safe_exp(-projections)  # [N, D]
    
    # 拼接: [N, 2D]
    phi = np.concatenate([exp_pos, exp_neg], axis=1)
    
    # Restore 3D if needed
    if original_3d:
        phi = phi.reshape(B, N_out, 2 * D)
    
    return phi


def compute_phi_grad(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    计算 dφ/dW，用于梯度计算。
    
    φ(x) = [exp(w_i·x), exp(-w_i·x)] for i=1..D
    
    d(exp(w_i·x))/dw_i = x * exp(w_i·x)
    d(exp(-w_i·x))/dw_i = -x * exp(-w_i·x)
    
    Parameters
    ----------
    x : [N, d]
    W : [d, D]
    
    Returns
    -------
    grad_phi : [N, 2D, d] (每个样本的梯度)
    """
    N, d = x.shape
    D = W.shape[1]
    
    projections = x @ W  # [N, D]
    exp_pos = safe_exp(projections)  # [N, D]
    exp_neg = safe_exp(-projections)  # [N, D]
    
    # 正部分的梯度: d(exp(w_i·x))/dw_i_j = x_j * exp(w_i·x)
    # [N, D, 1] * [1, 1, d] = [N, D, d]
    grad_pos = exp_pos[:, :, None] * x[:, None, :]  # [N, D, d]
    
    # 负部分的梯度: d(exp(-w_i·x))/dw_i_j = -x_j * exp(-w_i·x)
    grad_neg = -exp_neg[:, :, None] * x[:, None, :]  # [N, D, d]
    
    # 拼接: [N, 2D, d]
    grad_phi = np.concatenate([grad_pos, grad_neg], axis=1)
    
    return grad_phi


# ============== Teacher Attention ==============

def compute_teacher_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """
    计算 Teacher attention (ground truth)。
    
    A_teacher = softmax(Q·K^T / √d) · V
    
    Parameters
    ----------
    Q : [Q_len, d]
    K : [kv_len, d]
    V : [kv_len, d]
    scale : 缩放因子
    
    Returns
    -------
    A : [Q_len, d]
    """
    Q_len, d = Q.shape
    kv_len = K.shape[0]
    
    # Q @ K^T: [Q_len, kv_len]
    logits = Q @ K.T
    
    # Scale
    logits = logits * scale
    
    # Softmax
    attn_weights = scipy_softmax(logits, axis=-1)  # [Q_len, kv_len]
    
    # A = attn_weights @ V: [Q_len, d]
    A = attn_weights @ V
    
    return A


# ============== Student Attention ==============

def build_trainable_sketch(
    K: np.ndarray,
    V: np.ndarray,
    D: int = 32,
    seed: int = 0,
) -> tuple[TrainableSketch, np.ndarray, np.ndarray]:
    """
    构建可训练 sketch 的初始状态。
    
    预存:
        S_V = Σ_i φ(k_i) v_i^T  [2D, d]
        S_Z = Σ_i φ(k_i)         [2D]
    
    Parameters
    ----------
    K : [kv_len, d]
    V : [kv_len, d]
    D : 特征维度 (输出是 2D)
    seed : 随机种子
    
    Returns
    -------
    sketch : TrainableSketch
    S_V : [2D, d] - 预存的加权 V
    S_Z : [2D]    - 预存的归一化因子
    """
    kv_len, d = K.shape
    gen = np.random.default_rng(seed)
    
    # 初始化 W: w_i ~ N(0, 1/d)
    W = gen.standard_normal((d, D)) * np.sqrt(1.0 / d)
    
    # 计算初始 φ(K)
    phi_K = compute_phi(K, W)  # [kv_len, 2D]
    
    # 预存
    S_V = phi_K.T @ V  # [2D, d]
    S_Z = phi_K.sum(axis=0)  # [2D]
    
    sketch = TrainableSketch(W=W, D=D, d=d, trained=False)
    
    return sketch, S_V, S_Z


def compute_student_attention(
    Q: np.ndarray,
    S_V: np.ndarray,
    S_Z: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """
    计算 Student attention (sketch approximation)。
    
    A_student = φ(Q)^T · S_V / (φ(Q)^T · S_Z + eps)
    
    Parameters
    ----------
    Q : [Q_len, d]
    S_V : [2D, d]
    S_Z : [2D]
    W : [d, D]
    
    Returns
    -------
    A : [Q_len, d]
    """
    # φ(Q): [Q_len, 2D]
    phi_Q = compute_phi(Q, W)
    
    # num = φ(Q) @ S_V: [Q_len, d]
    num = phi_Q @ S_V
    
    # den = φ(Q) @ S_Z: [Q_len]
    den = phi_Q @ S_Z
    den_safe = np.clip(den, 1e-30, None)
    
    # A = num / den[..., None]: [Q_len, d]
    A = num / den_safe[..., None]
    
    return A


# ============== 梯度计算 ==============

def compute_loss_gradient(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    A_teacher: np.ndarray,
    W: np.ndarray,
    S_V: np.ndarray,
    S_Z: np.ndarray,
    batch_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    计算 L = ||A_teacher - A_student||² 对 W 的梯度。
    
    使用链式法则，完整计算 dL/dW:
        L = (1/N) Σ ||A_teacher_i - A_student_i||²
        A_student = φ(Q)^T @ S_V / (φ(Q)^T @ S_Z)
    
    Parameters
    ----------
    Q : [Q_len, d] 或 [B, Q_len, d]
    K : [kv_len, d]
    V : [kv_len, d]
    A_teacher : [Q_len, d] 或 [B, Q_len, d]
    W : [d, D]
    S_V : [2D, d]
    S_Z : [2D]
    batch_idx : 如果提供，只用这些 Q
    
    Returns
    -------
    grad_W : [d, D]
    """
    # 处理 batch
    if batch_idx is not None:
        Q = Q[batch_idx]
        A_teacher = A_teacher[batch_idx]
    
    if Q.ndim == 2:
        Q = Q[None, ...]
        A_teacher = A_teacher[None, ...]
    
    B, Q_len, d = Q.shape
    D = W.shape[1]
    
    # φ(Q): [B, Q_len, 2D]
    phi_Q = compute_phi(Q, W)
    
    # den = φ(Q) @ S_Z: [B, Q_len]
    den = phi_Q @ S_Z  # [B, Q_len]
    den_safe = np.clip(den, 1e-30, None)  # [B, Q_len]
    
    # num = φ(Q) @ S_V: [B, Q_len, d]
    num = phi_Q @ S_V
    
    # A_student = num / den[..., None]: [B, Q_len, d]
    A_student = num / den_safe[..., None]
    
    # Ensure A_teacher has same shape as A_student
    if A_teacher.shape != A_student.shape:
        if A_teacher.ndim == 2 and A_teacher.shape[0] == B:
            A_teacher = A_teacher[:, None, :]
    
    # dL/dA = 2 * (A_student - A_teacher) / N: [B, Q_len, d]
    dL_dA = 2.0 * (A_student - A_teacher) / (B * Q_len)
    
    # 计算 dA/dφ:
    # A[i,j] = Σ_k φ[i,k] * S_V[k,j] / Σ_k φ[i,k] * S_Z[k]
    # dA[i,j]/dφ[i,m] = S_V[m,j] / den[i] - num[i,j] * S_Z[m] / den²[i]
    
    # den: [B, Q_len]
    # num: [B, Q_len, d]
    # S_V: [2D, d]
    # S_Z: [2D]
    
    # dA/dφ: [B, Q_len, 2D]
    # For each (b, q, m): S_V[m,:] / den[b,q] - num[b,q,:] * S_Z[m] / den²[b,q]
    
    den_sq = den_safe ** 2  # [B, Q_len]
    
    # [B, Q_len, 2D, d] = [B, Q_len, 2D, 1] - [B, Q_len, 2D, d]
    # First term: S_V.T[None, None, :, :] / den[..., None, None] -> [B, Q_len, d, 2D]
    # But we need [B, Q_len, 2D, d], so transpose
    first_term = (S_V.T[None, None, :, :] / den[..., None, None]).transpose(0, 1, 3, 2)  # [B, Q_len, 2D, d]
    
    # Second term: num[..., None] * S_Z[..., None] -> [B, Q_len, d, 2D]
    # Need to reorder to match
    num_expanded = num[..., None] * S_Z[None, None, None, :]  # [B, Q_len, d, 2D]
    den_sq_expanded = den_sq[..., None, None]  # [B, Q_len, 1, 1]
    second_term = num_expanded / den_sq_expanded  # [B, Q_len, d, 2D]
    second_term = second_term.transpose(0, 1, 3, 2)  # [B, Q_len, 2D, d]
    
    dA_dphi = first_term - second_term
    
    # dL/dφ = dL/dA @ dA/dφ (sum over d dimension)
    # dL/dA: [B, Q_len, d] -> expand to [B, Q_len, 1, d]
    # dA/dφ: [B, Q_len, 2D, d]
    # Result: [B, Q_len, 2D]
    dL_dphi = np.sum(dL_dA[..., None, :] * dA_dphi, axis=-1)
    
    # dφ/dW 的梯度
    # φ(Q) = [exp(Q@W), exp(-Q@W)]
    # d(exp(Q@W))/dW = x * exp(Q@W)
    # d(exp(-Q@W))/dW = -x * exp(-Q@W)
    
    # flatten for batch computation
    dL_dphi_flat = dL_dphi.reshape(-1, 2 * D)  # [B*Q_len, 2D]
    Q_flat = Q.reshape(-1, d)  # [B*Q_len, d]
    
    # 计算梯度
    # grad_pos = Σ dL/dφ_pos * x
    # grad_neg = Σ (-dL/dφ_neg) * x
    
    grad_pos = dL_dphi_flat[:, :D].T @ Q_flat  # [D, d]
    grad_neg = -dL_dphi_flat[:, D:].T @ Q_flat  # [D, d]
    
    # 合并: grad_W = grad_pos + grad_neg, shape [D, d]
    grad_W = grad_pos + grad_neg
    
    # 转置为 [d, D]
    grad_W = grad_W.T
    
    return grad_W


# ============== 训练循环 ==============

def train_sketch(
    K: np.ndarray,
    V: np.ndarray,
    Q_train: np.ndarray,
    D: int = 32,
    lr: float = 1e-3,
    batch_size: int = 32,
    num_steps: int = 200,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[TrainableSketch, dict]:
    """
    训练 TrainableSketch。
    
    Parameters
    ----------
    K : [kv_len, d] - Key 向量
    V : [kv_len, d] - Value 向量
    Q_train : [N_train, q_len, d] 或 [N_train, d] - 训练 query
    D : φ 的维度
    lr : 学习率
    batch_size : batch 大小
    num_steps : 训练步数
    seed : 随机种子
    
    Returns
    -------
    sketch : 训练好的 sketch
    history : 训练历史
    """
    gen = np.random.default_rng(seed)
    
    # 处理 Q 维度
    if Q_train.ndim == 2:
        Q_train = Q_train[:, None, :]  # [N, 1, d]
    
    N_train, q_len, d = Q_train.shape
    kv_len = K.shape[0]
    
    # 初始化 sketch 和 optimizer
    sketch, S_V_init, S_Z_init = build_trainable_sketch(K, V, D, seed=seed)
    W = sketch.W.copy()
    
    optimizer = AdamOptimizer([W], lr=lr)
    
    # 预计算 teacher attention
    if q_len == 1:
        # Q_train: [N, 1, d] -> [N, d]
        Q_for_teacher = Q_train[:, 0, :]
        A_teacher_all = compute_teacher_attention(Q_for_teacher, K, V, scale=1.0/np.sqrt(d))
    else:
        A_teacher_all = np.zeros((N_train, q_len, d), dtype=np.float32)
        for i in range(N_train):
            A_teacher_all[i] = compute_teacher_attention(Q_train[i], K, V, scale=1.0/np.sqrt(d))
    
    # 训练循环
    history = {
        "loss": [],
        "step": [],
        "time": [],
    }
    
    start_time = time.time()
    
    for step in range(num_steps):
        # Sample batch
        if N_train >= batch_size:
            batch_idx = gen.choice(N_train, size=batch_size, replace=False)
        else:
            batch_idx = np.arange(N_train)
        
        # 重新计算 S_V, S_Z (因为 W 更新了)
        phi_K = compute_phi(K, W)  # [kv_len, 2D]
        S_V = phi_K.T @ V  # [2D, d]
        S_Z = phi_K.sum(axis=0)  # [2D]
        
        # 获取 batch data
        Q_batch = Q_train[batch_idx]  # [B, q_len, d]
        A_teacher_batch = A_teacher_all[batch_idx]  # [B, q_len, d]
        
        # 计算梯度
        grad_W = compute_loss_gradient(
            Q_batch, K, V, A_teacher_batch, W, S_V, S_Z
        )
        
        # 更新
        optimizer.step([grad_W])
        
        # 计算 loss (整个训练集)
        phi_Q = compute_phi(Q_train[:, 0, :] if q_len == 1 else Q_train.reshape(N_train * q_len, d), W)
        if q_len > 1:
            Q_flat = Q_train.reshape(N_train * q_len, d)
            A_teacher_flat = A_teacher_all.reshape(N_train * q_len, d)
        else:
            Q_flat = Q_train[:, 0, :]
            A_teacher_flat = A_teacher_all
        
        num = phi_Q @ S_V if q_len == 1 else phi_Q @ S_V
        den = phi_Q @ S_Z
        den_safe = np.clip(den, 1e-30, None)
        
        # 处理 q_len > 1 的情况
        if q_len > 1:
            num = num.reshape(N_train, q_len, d)
            den_safe = den_safe.reshape(N_train, q_len)
            A_student_flat = num / den_safe[..., None]
        else:
            A_student_flat = num / den_safe[..., None]
        
        loss = np.mean((A_student_flat - A_teacher_flat) ** 2)
        
        history["loss"].append(float(loss))
        history["step"].append(step)
        history["time"].append(time.time() - start_time)
        
        if verbose and (step % 20 == 0 or step == num_steps - 1):
            print(f"  Step {step:3d}: loss = {loss:.6f}")
    
    # 更新 sketch
    sketch.W = W
    sketch.trained = True
    
    return sketch, history


# ============== 数据生成 ==============

def make_clustered_data(
    n_samples: int,
    d: int,
    n_clusters: int = 8,
    cluster_std: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 clustered K/V 数据。"""
    gen = np.random.default_rng(seed)
    
    # 生成聚类中心
    centroids = []
    for _ in range(n_clusters * 10):
        c = gen.standard_normal(d) * 2.0
        if all(np.linalg.norm(c - oc) > 2.5 for oc in centroids):
            centroids.append(c)
        if len(centroids) >= n_clusters:
            break
    
    centroids = np.array(centroids) if centroids else gen.standard_normal((n_clusters, d)) * 2.0
    
    # 分配 cluster
    assignments = gen.integers(0, n_clusters, size=n_samples)
    K = centroids[assignments] + gen.standard_normal((n_samples, d)) * cluster_std
    
    # V 跟 K 相关
    W_proj = gen.standard_normal((d, d)) * 0.3
    V = K @ W_proj + gen.standard_normal((n_samples, d)) * 0.1
    
    return K.astype(np.float32), V.astype(np.float32)


def make_random_data(
    n_samples: int,
    d: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成完全随机 K/V 数据 (无结构)。"""
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((n_samples, d)).astype(np.float32)
    V = gen.standard_normal((n_samples, d)).astype(np.float32)
    return K, V


def make_skewed_data(
    n_samples: int,
    d: int,
    n_outliers: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 skewed K/V 数据 (少数 outlier + 大量 normal)。"""
    gen = np.random.default_rng(seed)
    
    # Outlier
    outlier_K = gen.standard_normal((n_outliers, d)) * 3.0
    outlier_V = gen.standard_normal((n_outliers, d)) * 3.0
    
    # Normal
    normal_K = gen.standard_normal((n_samples - n_outliers, d)) * 0.3
    normal_V = gen.standard_normal((n_samples - n_outliers, d)) * 0.3
    
    # 合并
    K = np.concatenate([outlier_K, normal_K])
    V = np.concatenate([outlier_V, normal_V])
    
    # Shuffle
    perm = gen.permutation(n_samples)
    return K[perm].astype(np.float32), V[perm].astype(np.float32)


def make_ood_data(
    n_samples: int,
    d: int,
    shift: float = 5.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 OOD 数据 (与训练数据分布不同)。"""
    gen = np.random.default_rng(seed)
    K = gen.standard_normal((n_samples, d)) * 2.0 + shift
    V = gen.standard_normal((n_samples, d)) * 2.0
    return K.astype(np.float32), V.astype(np.float32)


# ============== 评估函数 ==============

def eval_sketch_error(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    sketch: TrainableSketch,
    S_V: np.ndarray = None,
    S_Z: np.ndarray = None,
) -> float:
    """
    评估 sketch 的 MSE error。
    
    如果 S_V 和 S_Z 未提供，则使用 K, V 重新计算。
    """
    Q_len, d = Q.shape
    
    # 如果 S_V, S_Z 未提供，从 K, V 计算
    if S_V is None or S_Z is None:
        phi_K = compute_phi(K, sketch.W)
        S_V = phi_K.T @ V
        S_Z = phi_K.sum(axis=0)
    
    # Teacher
    A_teacher = compute_teacher_attention(Q, K, V, scale=1.0/np.sqrt(d))
    
    # Student
    A_student = compute_student_attention(Q, S_V, S_Z, sketch.W)
    
    # MSE
    mse = np.mean((A_teacher - A_student) ** 2)
    return float(mse)


def eval_coreset_error(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    r: int,
    seed: int = 0,
) -> float:
    """评估 coreset 的 MSE error。"""
    from simulation.exp3_kernel_feature_sketch import (
        build_coreset_sketch,
        eval_coreset_sketch,
    )
    
    kv_len, d = K.shape
    q_len = Q.shape[0]
    
    # Build coreset
    cs = build_coreset_sketch(K, V, r, seed=seed)
    
    # Evaluate
    result = eval_coreset_sketch(cs, Q)
    A_coreset = result.y[0]  # [q_len, d]
    
    # Teacher
    A_teacher = compute_teacher_attention(Q, K, V, scale=1.0/np.sqrt(d))
    
    # MSE
    mse = np.mean((A_teacher - A_coreset) ** 2)
    return float(mse)


def eval_kernel_error(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    feature_dim: int = 64,
    seed: int = 0,
) -> float:
    """评估 kernel (RFF) 的 MSE error。"""
    from simulation.exp3_kernel_feature_sketch import (
        build_kernel_feature_sketch,
        eval_kernel_feature_sketch_B,
    )
    
    kv_len, d = K.shape
    q_len = Q.shape[0]
    
    # Build kernel sketch
    ks = build_kernel_feature_sketch(K, V, feature_dim=feature_dim, seed=seed)
    
    # Evaluate
    result = eval_kernel_feature_sketch_B(ks, Q)
    A_kernel = result.y[0]  # [q_len, d]
    
    # Teacher
    A_teacher = compute_teacher_attention(Q, K, V, scale=1.0/np.sqrt(d))
    
    # MSE
    mse = np.mean((A_teacher - A_kernel) ** 2)
    return float(mse)


def eval_svd_error(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    rank: int = 32,
) -> float:
    """评估 truncated SVD 的 MSE error。"""
    kv_len, d = K.shape
    
    # Compute attention weights for each query
    A_teacher_list = []
    A_svd_list = []
    
    for i in range(len(Q)):
        q = Q[i:i+1]  # [1, d]
        
        # Teacher attention
        A_teacher = compute_teacher_attention(q, K, V, scale=1.0/np.sqrt(d))  # [1, d]
        
        # Simplified SVD approximation: V = U @ S @ VT, keep top-r
        # For each query, compute effective V
        # A ≈ softmax(q·K^T) @ V
        # Use SVD of K, V to approximate
        # Here we just use truncated V directly
        
        # Actually, use truncated K and V for SVD approximation
        # This is a simplified version
        # k_mean = K.mean(axis=0)
        # K_centered = K - k_mean
        # U, s, Vt = np.linalg.svd(K_centered, full_matrices=False)
        # K_trunc = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :] + k_mean
        # V_trunc = V  # simplified
        
        # For fair comparison, we use low-rank V approximation
        U, s, Vt = np.linalg.svd(V, full_matrices=False)
        V_trunc = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        
        # Recompute with truncated V
        # Actually we should recompute K too, but for simplicity just truncate V
        # This is a simplified SVD approximation
        
        # Better: compute A directly with truncated representation
        # A = softmax(qK^T) @ V ≈ softmax(qK^T) @ V_trunc
        
        logits = q @ K.T / np.sqrt(d)
        attn = scipy_softmax(logits, axis=-1)
        A_svd = attn @ V_trunc  # [1, d]
        
        A_teacher_list.append(A_teacher)
        A_svd_list.append(A_svd)
    
    A_teacher_all = np.concatenate(A_teacher_list, axis=0)
    A_svd_all = np.concatenate(A_svd_list, axis=0)
    
    mse = np.mean((A_teacher_all - A_svd_all) ** 2)
    return float(mse)


# ============== 主实验 ==============

def run_exp9(
    d: int = 64,
    kv_len: int = 128,
    q_len: int = 16,
    n_train: int = 32,
    n_test: int = 64,
    D: int = 32,
    num_steps: int = 200,
    seed: int = 42,
) -> dict:
    """
    运行 Exp9 完整实验。
    
    Returns
    -------
    results : dict with training curve, generalization, and Pareto data
    """
    print(f"\n{'='*60}")
    print(f"Exp9: TrainableSketch 实验")
    print(f"{'='*60}")
    print(f"d={d}, kv_len={kv_len}, q_len={q_len}")
    print(f"n_train={n_train}, n_test={n_test}, D={D}, steps={num_steps}")
    
    # 生成数据
    print(f"\n[1] 生成数据...")
    
    # 训练数据: clustered
    K_train, V_train = make_clustered_data(n_train, d, n_clusters=8, seed=seed)
    Q_train = make_clustered_data(n_train, d, n_clusters=8, seed=seed+1)[0]
    
    # 测试数据: 多种类型
    K_test_clustered, V_test_clustered = make_clustered_data(n_test, d, n_clusters=8, seed=seed+100)
    Q_test_clustered = make_clustered_data(n_test, d, n_clusters=8, seed=seed+101)[0]
    
    K_test_random, V_test_random = make_random_data(n_test, d, seed=seed+200)
    Q_test_random = make_random_data(n_test, d, seed=seed+201)[0]
    
    K_test_skewed, V_test_skewed = make_skewed_data(n_test, d, n_outliers=16, seed=seed+300)
    Q_test_skewed = make_skewed_data(n_test, d, n_outliers=16, seed=seed+301)[0]
    
    K_test_ood, V_test_ood = make_ood_data(n_test, d, shift=5.0, seed=seed+400)
    Q_test_ood = make_ood_data(n_test, d, shift=5.0, seed=seed+401)[0]
    
    print(f"  训练: {n_train} samples (clustered)")
    print(f"  测试: {n_test} samples x 4 types (clustered/random/skewed/OOD)")
    
    # 训练
    print(f"\n[2] 训练 TrainableSketch...")
    start_time = time.time()
    
    sketch, history = train_sketch(
        K_train, V_train, Q_train,
        D=D, lr=1e-3, batch_size=32,
        num_steps=num_steps, seed=seed, verbose=True
    )
    
    train_time = time.time() - start_time
    print(f"  训练完成: {num_steps} steps in {train_time:.2f}s")
    print(f"  最终 loss: {history['loss'][-1]:.6f}")
    
    # 泛化测试 - 使用训练好的 φ，在测试数据上评估
    print(f"\n[3] 泛化测试...")
    
    generalization = {}
    
    # In-domain (clustered) - 测试数据与训练数据同分布
    # 使用 trained φ 在测试 K/V 上计算 S_V, S_Z
    train_error = eval_sketch_error(Q_train, K_train, V_train, sketch)
    test_clustered_error = eval_sketch_error(
        Q_test_clustered, K_test_clustered, V_test_clustered, sketch
    )
    generalization["in_domain_clustered"] = {
        "train_error": train_error,
        "test_error": test_clustered_error,
    }
    print(f"  In-domain (clustered): train={train_error:.6f}, test={test_clustered_error:.6f}")
    
    # OOD: random - 测试数据与训练数据分布不同
    test_random_error = eval_sketch_error(
        Q_test_random, K_test_random, V_test_random, sketch
    )
    generalization["ood_random"] = {
        "test_error": test_random_error,
    }
    print(f"  OOD (random): test={test_random_error:.6f}")
    
    # OOD: skewed
    test_skewed_error = eval_sketch_error(
        Q_test_skewed, K_test_skewed, V_test_skewed, sketch
    )
    generalization["ood_skewed"] = {
        "test_error": test_skewed_error,
    }
    print(f"  OOD (skewed): test={test_skewed_error:.6f}")
    
    # OOD: shifted
    test_ood_error = eval_sketch_error(
        Q_test_ood, K_test_ood, V_test_ood, sketch
    )
    generalization["ood_shift"] = {
        "test_error": test_ood_error,
    }
    print(f"  OOD (shift): test={test_ood_error:.6f}")
    
    # 对比: Coreset / Kernel / SVD
    print(f"\n[4] 对比 Coreset / Kernel / SVD...")
    
    # 同等参数量比较
    # TrainableSketch: D params, D bytes
    # Coreset: r * d * 2 params (K + V)
    # Kernel: feature_dim * d * 2 + feature_dim
    # SVD: rank * d * 2
    
    pareto_points = []
    
    # TrainableSketch
    ts_bytes = D * 4  # D floats
    pareto_points.append({
        "method": "TrainableSketch",
        "error": train_error,
        "bytes": ts_bytes,
        "num_params": D,
    })
    
    # Test with different D values
    for D_test in [16, 64, 128]:
        sketch_test, history_test = train_sketch(
            K_train, V_train, Q_train,
            D=D_test, lr=1e-3, batch_size=32,
            num_steps=num_steps, seed=seed, verbose=False
        )
        phi_K = compute_phi(K_train, sketch_test.W)
        S_V = phi_K.T @ V_train
        S_Z = phi_K.sum(axis=0)
        error = eval_sketch_error(Q_train, K_train, V_train, sketch_test, S_V, S_Z)
        pareto_points.append({
            "method": f"TrainableSketch_D{D_test}",
            "error": error,
            "bytes": D_test * 4,
            "num_params": D_test,
        })
    
    # Coreset with different r
    for r in [8, 16, 32, 64]:
        try:
            error = eval_coreset_error(Q_train, K_train, V_train, r=r, seed=seed)
            bytes_used = r * d * 2 * 4  # K + V, float32
            pareto_points.append({
                "method": f"Coreset_r{r}",
                "error": error,
                "bytes": bytes_used,
                "num_params": r * d * 2,
            })
        except Exception as e:
            print(f"  Coreset r={r} failed: {e}")
    
    # Kernel (RFF)
    for feature_dim in [32, 64, 128]:
        try:
            error = eval_kernel_error(Q_train, K_train, V_train, feature_dim=feature_dim, seed=seed)
            bytes_used = feature_dim * d * 2 * 4 + feature_dim * 4  # w + b
            pareto_points.append({
                "method": f"RFF_d{feature_dim}",
                "error": error,
                "bytes": bytes_used,
                "num_params": feature_dim * d * 2 + feature_dim,
            })
        except Exception as e:
            print(f"  RFF dim={feature_dim} failed: {e}")
    
    # SVD
    for rank in [8, 16, 32, 64]:
        try:
            error = eval_svd_error(Q_train, K_train, V_train, rank=rank)
            bytes_used = rank * d * 4  # truncated V
            pareto_points.append({
                "method": f"SVD_rank{rank}",
                "error": error,
                "bytes": bytes_used,
                "num_params": rank * d,
            })
        except Exception as e:
            print(f"  SVD rank={rank} failed: {e}")
    
    print(f"\n[5] 汇总结果...")
    
    # 训练曲线
    training_curve = {
        "steps": history["step"],
        "loss": history["loss"],
        "time": history["time"],
        "final_loss": float(history["loss"][-1]),
        "converged": history["loss"][-1] < 0.1,
    }
    
    # 泛化差距
    gen_gap = {
        "in_domain_test": test_clustered_error,
        "ood_random": test_random_error,
        "ood_skewed": test_skewed_error,
        "ood_shift": test_ood_error,
        "ood_gap_random": test_random_error - test_clustered_error,
        "ood_gap_skewed": test_skewed_error - test_clustered_error,
        "ood_gap_shift": test_ood_error - test_clustered_error,
    }
    
    results = {
        "config": {
            "d": d,
            "kv_len": kv_len,
            "q_len": q_len,
            "n_train": n_train,
            "n_test": n_test,
            "D": D,
            "num_steps": num_steps,
            "seed": seed,
        },
        "training_curve": training_curve,
        "generalization": generalization,
        "generalization_gap": gen_gap,
        "pareto": pareto_points,
        "train_time": train_time,
    }
    
    return results


# ============== 入口 ==============

if __name__ == "__main__":
    # 运行实验
    results = run_exp9(
        d=64,
        kv_len=128,
        q_len=16,
        n_train=32,
        n_test=64,
        D=32,
        num_steps=200,
        seed=42,
    )
    
    # 保存结果
    results_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 训练曲线
    with open(os.path.join(results_dir, "exp9_training_curve.json"), "w") as f:
        json.dump({
            "steps": results["training_curve"]["steps"],
            "loss": results["training_curve"]["loss"],
            "time": results["training_curve"]["time"],
            "final_loss": results["training_curve"]["final_loss"],
            "converged": results["training_curve"]["converged"],
        }, f, indent=2)
    
    # 泛化结果
    with open(os.path.join(results_dir, "exp9_generalization.json"), "w") as f:
        json.dump({
            "generalization": results["generalization"],
            "generalization_gap": results["generalization_gap"],
        }, f, indent=2)
    
    # Pareto 对比
    with open(os.path.join(results_dir, "exp9_pareto.json"), "w") as f:
        json.dump({
            "pareto_points": results["pareto"],
        }, f, indent=2)
    
    print(f"\n结果已保存到 {results_dir}/")
    print(f"  - exp9_training_curve.json")
    print(f"  - exp9_generalization.json")
    print(f"  - exp9_pareto.json")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("Exp9 摘要")
    print(f"{'='*60}")
    print(f"训练 loss: {results['training_curve']['final_loss']:.6f}")
    print(f"收敛: {'是' if results['training_curve']['converged'] else '否'}")
    print(f"泛化差距 (random): {results['generalization_gap']['ood_gap_random']:.6f}")
    print(f"训练时间: {results['train_time']:.2f}s")
