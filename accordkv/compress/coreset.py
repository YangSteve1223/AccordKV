"""
coreset — Uniform/weighted sampling based coreset compression.
"""
from __future__ import annotations

import numpy as np


def coreset_compress(K: np.ndarray, V: np.ndarray, r: int,
                     seed: int = 0) -> dict:
    """Coreset compression for KV cache.

    Args:
        K, V: [kv_len, d] key/value matrices
        r: number of centroids
        seed: random seed

    Returns:
        dict with centroids, weights for K and V
    """
    kv_len, d = K.shape
    gen = np.random.default_rng(seed)

    # Uniform sampling indices
    indices = gen.choice(kv_len, size=min(r, kv_len), replace=False)
    indices = np.sort(indices)

    K_centroids = K[indices]
    V_centroids = V[indices]
    weights = np.ones(len(indices), dtype=np.float32)

    return dict(
        K_centroids=K_centroids,  # [r, d]
        V_centroids=V_centroids,  # [r, d]
        indices=indices,          # [r]
        weights=weights,          # [r]
        r=len(indices),
    )


def coreset_attention(Q: np.ndarray, sketch: dict) -> np.ndarray:
    """Attention with coreset-compressed KV.

    Args:
        Q: [q_len, d] query
        sketch: output of coreset_compress

    Returns:
        attention output [q_len, d]
    """
    K = sketch["K_centroids"]
    V = sketch["V_centroids"]
    weights = sketch["weights"]

    scores = Q @ K.T  # [q_len, r]
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    # weight-normalized probability
    p = p * weights / (p @ weights[:, None] + 1e-30)
    return p @ V
