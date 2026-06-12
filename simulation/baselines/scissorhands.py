"""
Scissorhands: Consistent Concept Importance for KV Cache Compression (NeurIPS 2023)
====================================================================================

Reference: Scissorhands: Consistent Concept Importance for KV Cache Compression (NeurIPS 2023)
Paper: https://arxiv.org/abs/2311.09522

Key idea: Evict KV cache entries based on their cumulative PPL (perplexity) contribution.
Tokens that contribute less to the model's perplexity are evicted first.
In the streaming/inference setting, this means keeping tokens that are "harder to predict"
(i.e., have higher conditional perplexity when attended to).

Simplified numpy version:
- Compute PPL contribution of each token as the negative log-likelihood
  contribution under a simple language model prior
- Score = cumulative contribution to predicting subsequent tokens
- Keep tokens with highest PPL contribution (hardest to predict / most informative)
- Also retain recent tokens as they are critical for next-token prediction
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def compute_scissorhands_scores(K: np.ndarray, V: np.ndarray,
                                 Q: np.ndarray, decay: float = 0.95) -> np.ndarray:
    """
    Compute Scissorhands importance scores for each KV token.

    The score reflects each token's contribution to predicting future tokens:
    - Tokens that receive high attention when computing PPL(next_token) are important
    - We simulate this by looking at how much each token contributes to attention
      when queried by "future" query representations

    Simplified approach: Use token-level importance from:
    1. Attention reception weight (how much each key is attended to)
    2. Position-based recency bonus (recent tokens matter more for next-token)
    3. Magnitude-based importance (high-norm tokens more informative)

    Parameters
    ----------
    K : np.ndarray, shape [kv_len, d]
        Key vectors
    V : np.ndarray, shape [kv_len, d]
        Value vectors
    Q : np.ndarray, shape [q_len, d]
        Query vectors
    decay : float
        Decay factor for recency weighting (higher = more weight on recent)

    Returns
    -------
    scores : np.ndarray, shape [kv_len,]
        Scissorhands importance scores (higher = more important)
    """
    kv_len, d = K.shape

    # 1. Attention reception (how much does each key receive attention from queries?)
    scores_attn = Q @ K.T / np.sqrt(d)
    scores_attn -= scores_attn.max(axis=-1, keepdims=True)
    p_attn = np.exp(scores_attn)
    p_attn_sum = p_attn.sum(axis=-1, keepdims=True)
    p_attn = p_attn / np.clip(p_attn_sum, 1e-30, None)
    attn_reception = p_attn.sum(axis=0)  # [kv_len,]

    # 2. Recency bonus: exponential decay from end
    # Recent tokens (high index) get higher scores
    positions = np.arange(kv_len, dtype=np.float32)
    # Normalize to [0, 1]
    recency_scores = (positions / (kv_len - 1)) ** decay

    # 3. Token magnitude (informative tokens tend to have higher norms)
    norms = np.linalg.norm(V, axis=1)

    # Combined score: attention × recency × magnitude
    combined = attn_reception * recency_scores * (norms / (norms.mean() + 1e-30))

    return combined.astype(np.float32)


def scissorhands_compress(K: np.ndarray, V: np.ndarray, Q: np.ndarray,
                          budget: int, recent_ratio: float = 0.1,
                          decay: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scissorhands compression: keep tokens by PPL contribution + recency.

    Parameters
    ----------
    K : np.ndarray, shape [kv_len, d]
        Key vectors
    V : np.ndarray, shape [kv_len, d]
        Value vectors
    Q : np.ndarray, shape [q_len, d]
        Query vectors
    budget : int
        Number of tokens to keep
    recent_ratio : float
        Fraction of budget reserved for recent tokens
    decay : float
        Recency decay factor

    Returns
    -------
    K_sel, V_sel, indices : np.ndarray
        Selected K, V, and their indices
    """
    kv_len, d = K.shape

    if budget >= kv_len:
        return K.copy(), V.copy(), np.arange(kv_len)

    # Compute importance scores
    importance = compute_scissorhands_scores(K, V, Q, decay)

    # Reserve some budget for recent tokens (always keep the most recent)
    n_recent = max(1, int(budget * recent_ratio))
    recent_indices = np.arange(kv_len - n_recent, kv_len)

    # Mask recent from importance-based selection
    importance_masked = importance.copy()
    importance_masked[recent_indices] = -np.inf

    # Select top-k by importance from non-recent tokens
    remaining_budget = budget - n_recent
    if remaining_budget > 0:
        top_indices = np.argpartition(importance_masked, -remaining_budget)[-remaining_budget:]
        top_indices = top_indices[np.argsort(importance[top_indices])[::-1]]
    else:
        top_indices = np.array([], dtype=np.int64)

    # Combine and deduplicate
    selected_indices = np.sort(np.unique(np.concatenate([top_indices, recent_indices])))

    # Trim if over budget
    if len(selected_indices) > budget:
        # Prefer to trim from importance-based selection
        excess = len(selected_indices) - budget
        non_recent = np.setdiff1d(selected_indices, recent_indices)
        if len(non_recent) > excess:
            trim = non_recent[:excess]
            selected_indices = np.setdiff1d(selected_indices, trim)
        selected_indices = np.sort(selected_indices)

    K_sel = K[selected_indices].astype(np.float32)
    V_sel = V[selected_indices].astype(np.float32)
    return K_sel, V_sel, selected_indices


def scissorhands_attention(Q: np.ndarray, K_sel: np.ndarray, V_sel: np.ndarray) -> np.ndarray:
    """
    Standard softmax attention on Scissorhands-selected tokens.
    """
    q_len, d = Q.shape

    scores = Q @ K_sel.T / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p_sum = p.sum(axis=-1, keepdims=True)
    p = p / np.clip(p_sum, 1e-30, None)
    O = p @ V_sel  # [q_len, d]

    return O.astype(np.float32)


def scissorhands_full_compression(K: np.ndarray, V: np.ndarray, Q: np.ndarray,
                                   budget: int, recent_ratio: float = 0.1,
                                   decay: float = 0.95) -> Tuple[np.ndarray, dict]:
    """
    Full Scissorhands compression pipeline.

    Returns compressed attention output and metadata.
    """
    kv_len, d = K.shape

    K_sel, V_sel, indices = scissorhands_compress(K, V, Q, budget, recent_ratio, decay)
    O_approx = scissorhands_attention(Q, K_sel, V_sel)

    metadata = {
        "n_selected": len(indices),
        "n_recent": max(1, int(budget * recent_ratio)),
        "compression_ratio": kv_len / max(len(indices), 1),
        "decay": decay,
        "indices": indices.tolist(),
    }

    return O_approx, metadata
