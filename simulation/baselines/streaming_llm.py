"""
StreamingLLM: Streaming Language Model Baseline (ICLR 2024)
=============================================================

Reference: Efficient Streaming Language Models via Attention Sink (ICLR 2024)
Paper: https://arxiv.org/abs/2309.17453

Key idea: LLMs exhibit "attention sinks" - initial tokens (especially the first few)
that receive disproportionate attention regardless of their semantic content.
StreamingLLM keeps: (1) first 4 tokens as sinks + (2) most recent N tokens.
Everything in between is dropped.

Simplified numpy version:
- Keep first 4 tokens (attention sinks)
- Keep last (budget - 4) tokens (recency window)
- Drop everything in the middle
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def streaming_llm_compress(K: np.ndarray, V: np.ndarray,
                           budget: int, n_sinks: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    StreamingLLM compression: keep sinks + recent window.

    Parameters
    ----------
    K : np.ndarray, shape [kv_len, d]
        Key vectors
    V : np.ndarray, shape [kv_len, d]
        Value vectors
    budget : int
        Total tokens to keep
    n_sinks : int
        Number of initial tokens to keep as attention sinks (default: 4)

    Returns
    -------
    K_sel, V_sel, indices : np.ndarray
        Selected K, V, and their indices
    """
    kv_len, d = K.shape

    if budget >= kv_len:
        return K.copy(), V.copy(), np.arange(kv_len)

    # Number of sinks to keep (min of n_sinks and budget)
    n_sinks = min(n_sinks, budget)
    sink_indices = np.arange(n_sinks)

    # Number of recent tokens to keep
    n_recent = budget - n_sinks
    if n_recent > 0:
        recent_indices = np.arange(kv_len - n_recent, kv_len)
    else:
        recent_indices = np.array([], dtype=np.int64)

    # Combine: sinks + recent
    selected_indices = np.sort(np.concatenate([sink_indices, recent_indices]))

    K_sel = K[selected_indices].astype(np.float32)
    V_sel = V[selected_indices].astype(np.float32)
    return K_sel, V_sel, selected_indices


def streaming_llm_attention(Q: np.ndarray, K_sel: np.ndarray, V_sel: np.ndarray) -> np.ndarray:
    """
    Standard softmax attention on StreamingLLM-selected tokens.
    """
    q_len, d = Q.shape

    scores = Q @ K_sel.T / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p_sum = p.sum(axis=-1, keepdims=True)
    p = p / np.clip(p_sum, 1e-30, None)
    O = p @ V_sel  # [q_len, d]

    return O.astype(np.float32)


def streaming_llm_full_compression(K: np.ndarray, V: np.ndarray, Q: np.ndarray,
                                     budget: int, n_sinks: int = 4) -> Tuple[np.ndarray, dict]:
    """
    Full StreamingLLM compression pipeline.

    Returns compressed attention output and metadata.
    """
    kv_len, d = K.shape

    K_sel, V_sel, indices = streaming_llm_compress(K, V, budget, n_sinks)
    O_approx = streaming_llm_attention(Q, K_sel, V_sel)

    metadata = {
        "n_selected": len(indices),
        "n_sinks": min(n_sinks, budget),
        "n_recent": budget - min(n_sinks, budget),
        "compression_ratio": kv_len / max(len(indices), 1),
        "indices": indices.tolist(),
    }

    return O_approx, metadata
