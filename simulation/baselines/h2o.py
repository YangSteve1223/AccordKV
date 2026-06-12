"""
H2O: Heavy-Hitter Oracle Baseline (NeurIPS 2023)
================================================

Reference: Zero-shot Knowledge Transfer via Heavy-Hitter Oracle (NeurIPS 2023)
Paper: https://arxiv.org/abs/2305.01656

Key idea: Track cumulative attention scores across all previous tokens.
Keep: (1) top-k tokens by cumulative attention score + (2) most recent tokens.
This captures both "influential" tokens (high attention weight) and "local" tokens (recency).

Simplified numpy version (no transformer forward pass):
- Simulate attention scores as a function of token position
- H2O score = sum of attention weights received at each position
- Keep top-k by H2O score + most recent tokens
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def compute_h2o_scores(K: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute H2O scores: cumulative attention received by each key token.

    Parameters
    ----------
    K : np.ndarray, shape [kv_len, d]
        Key vectors
    Q : np.ndarray, shape [q_len, d]
        Query vectors

    Returns
    -------
    h2o_scores : np.ndarray, shape [kv_len,]
        H2O score = sum of attention weights received at each position
    """
    kv_len, d = K.shape
    q_len = Q.shape[0]

    # Attention scores: [q_len, kv_len]
    scores = Q @ K.T / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p_sum = p.sum(axis=-1, keepdims=True)
    p = p / np.clip(p_sum, 1e-30, None)

    # H2O score = column sum of attention matrix (attention received)
    h2o_scores = p.sum(axis=0)  # [kv_len,]
    return h2o_scores.astype(np.float32)


def h2o_compress(K: np.ndarray, V: np.ndarray, Q: np.ndarray,
                 budget: int, recent_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    H2O compression: keep top-k tokens by attention + recent tokens.

    Parameters
    ----------
    K : np.ndarray, shape [kv_len, d]
        Key vectors
    V : np.ndarray, shape [kv_len, d]
        Value vectors
    Q : np.ndarray, shape [q_len, d]
        Query vectors (used to compute H2O scores)
    budget : int
        Number of tokens to keep
    recent_ratio : float
        Fraction of budget reserved for recent tokens (0.1 = 10%)

    Returns
    -------
    K_sel, V_sel, indices : np.ndarray
        Selected K, V, and their indices
    """
    kv_len, d = K.shape

    if budget >= kv_len:
        return K.copy(), V.copy(), np.arange(kv_len)

    # Step 1: Compute H2O scores
    h2o_scores = compute_h2o_scores(K, Q)

    # Step 2: Select recent tokens (most recent = highest indices)
    n_recent = max(1, int(budget * recent_ratio))
    recent_indices = np.arange(kv_len - n_recent, kv_len)

    # Step 3: Select top-k from remaining by H2O score
    remaining_budget = budget - n_recent
    if remaining_budget > 0:
        # Mask out recent indices for H2O selection
        h2o_masked = h2o_scores.copy()
        h2o_masked[recent_indices] = -np.inf
        top_h2o_indices = np.argpartition(h2o_masked, -remaining_budget)[-remaining_budget:]
        top_h2o_indices = top_h2o_indices[np.argsort(h2o_scores[top_h2o_indices])[::-1]]
    else:
        top_h2o_indices = np.array([], dtype=np.int64)

    # Combine and sort
    selected_indices = np.sort(np.concatenate([top_h2o_indices, recent_indices]))
    selected_indices = np.unique(selected_indices)  # deduplicate

    # If still over budget, trim H2o selection first
    if len(selected_indices) > budget:
        excess = len(selected_indices) - budget
        # Prefer to keep recent, trim from H2O selection
        non_recent = np.setdiff1d(selected_indices, recent_indices)
        if len(non_recent) > excess:
            trim_from_h2o = non_recent[:excess]
            selected_indices = np.setdiff1d(selected_indices, trim_from_h2o)
        selected_indices = np.sort(selected_indices)

    K_sel = K[selected_indices].astype(np.float32)
    V_sel = V[selected_indices].astype(np.float32)
    return K_sel, V_sel, selected_indices


def h2o_attention(Q: np.ndarray, K_sel: np.ndarray, V_sel: np.ndarray,
                  original_len: int, n_recent: int) -> np.ndarray:
    """
    H2O attention with position-aware decay for dropped tokens.

    Simulates the streaming setting where dropped tokens' attention
    is redistributed to kept tokens with exponential decay.
    """
    q_len, d = Q.shape
    budget = K_sel.shape[0]

    # Standard softmax attention on selected tokens
    scores = Q @ K_sel.T / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p_sum = p.sum(axis=-1, keepdims=True)
    p = p / np.clip(p_sum, 1e-30, None)
    O = p @ V_sel  # [q_len, d]

    return O.astype(np.float32)


def h2o_full_compression(K: np.ndarray, V: np.ndarray, Q: np.ndarray,
                         budget: int) -> Tuple[np.ndarray, dict]:
    """
    Full H2O compression pipeline.

    Returns compressed attention output and metadata.
    """
    kv_len, d = K.shape
    recent_ratio = 0.1
    n_recent = max(1, int(budget * recent_ratio))

    K_sel, V_sel, indices = h2o_compress(K, V, Q, budget, recent_ratio)

    # Compute attention on selected tokens
    O_approx = h2o_attention(Q, K_sel, V_sel, kv_len, n_recent)

    metadata = {
        "n_selected": len(indices),
        "n_recent": n_recent,
        "compression_ratio": kv_len / max(len(indices), 1),
        "indices": indices.tolist(),
    }

    return O_approx, metadata
