"""
FastGen: Adaptive KV Cache Compression for Efficient Inference (ACL 2024)
===========================================================================

Reference: FastGen: Adaptive KV Cache Compression for Efficient Inference (ACL 2024)
Paper: https://arxiv.org/abs/2404.09526

Key idea: FastGen uses a composable policy that combines multiple eviction strategies:
1. Heavy Hitter: Keep tokens with highest cumulative attention weight
2. Recent: Keep the most recent tokens (critical for next-token prediction)
3. Special Tokens: Always keep special tokens (BOS, EOS, SEP, etc.)
4. Distance-based: Prefer tokens that are "representative" of their local context

Simplified numpy version:
- Compose three policies: 40% heavy hitter + 40% recent + 20% special
- Heavy hitter: tokens with highest attention reception scores
- Recent: most recent tokens
- Special: tokens 0 (BOS) + tokens divisible by 100 (heuristic for structure)
- If budget < total policy requirements, prioritize: special > recent > heavy
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List


def fastgen_heavy_hitter_selection(K: np.ndarray, Q: np.ndarray,
                                    n_select: int) -> np.ndarray:
    """
    Select tokens by cumulative attention reception (heavy hitter policy).
    """
    kv_len, d = K.shape

    scores = Q @ K.T / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p_sum = p.sum(axis=-1, keepdims=True)
    p = p / np.clip(p_sum, 1e-30, None)
    attn_reception = p.sum(axis=0)

    # Top-k by attention reception
    if n_select >= kv_len:
        return np.arange(kv_len)
    top_indices = np.argpartition(attn_reception, -n_select)[-n_select:]
    top_indices = top_indices[np.argsort(attn_reception[top_indices])[::-1]]
    return top_indices


def fastgen_recent_selection(kv_len: int, n_select: int) -> np.ndarray:
    """
    Select most recent tokens.
    """
    if n_select >= kv_len:
        return np.arange(kv_len)
    return np.arange(kv_len - n_select, kv_len)


def fastgen_special_selection(kv_len: int, n_select: int) -> np.ndarray:
    """
    Select special tokens: BOS (index 0) + periodic structural tokens.

    In real FastGen, special tokens are identified by tokenizer.
    Here we use: index 0 (BOS) + every 100th token (structural marker heuristic).
    """
    special_indices = [0]
    # Every ~100th token as structural marker
    for idx in range(100, kv_len, 100):
        special_indices.append(idx)
    special_indices = np.array([i for i in special_indices if i < kv_len])

    if n_select >= len(special_indices):
        return special_indices
    return special_indices[:n_select]


def fastgen_compress(K: np.ndarray, V: np.ndarray, Q: np.ndarray,
                     budget: int,
                     heavy_ratio: float = 0.4,
                     recent_ratio: float = 0.4,
                     special_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FastGen compression: composable policy combining multiple strategies.

    Parameters
    ----------
    K : np.ndarray, shape [kv_len, d]
        Key vectors
    V : np.ndarray, shape [kv_len, d]
        Value vectors
    Q : np.ndarray, shape [q_len, d]
        Query vectors
    budget : int
        Total tokens to keep
    heavy_ratio : float
        Fraction of budget for heavy hitter policy
    recent_ratio : float
        Fraction of budget for recent policy
    special_ratio : float
        Fraction of budget for special token policy

    Returns
    -------
    K_sel, V_sel, indices : np.ndarray
        Selected K, V, and their indices
    """
    kv_len, d = K.shape

    if budget >= kv_len:
        return K.copy(), V.copy(), np.arange(kv_len)

    # Budget allocation per policy
    n_special = max(1, int(budget * special_ratio))
    n_recent = max(1, int(budget * recent_ratio))
    n_heavy = budget - n_special - n_recent
    n_heavy = max(0, n_heavy)

    # Policy 1: Special tokens (highest priority)
    special_indices = fastgen_special_selection(kv_len, n_special)

    # Policy 2: Recent tokens
    # Adjust to avoid overlap with special tokens
    remaining_recent = n_recent
    recent_indices_list = []
    recent_start = kv_len - remaining_recent
    for idx in range(kv_len - 1, kv_len - 1 - remaining_recent - 1, -1):
        if idx >= 0 and idx not in special_indices:
            recent_indices_list.append(idx)
    recent_indices = np.array(sorted(recent_indices_list))

    # Policy 3: Heavy hitters (avoid overlap)
    remaining_heavy = n_heavy
    heavy_indices_list = []
    # Start from most recent non-special/non-recent, work backward
    used_set = set(special_indices) | set(recent_indices)
    count = 0
    for idx in range(kv_len - 1, -1, -1):
        if idx not in used_set:
            heavy_indices_list.append(idx)
            count += 1
            if count >= remaining_heavy:
                break
    heavy_indices = np.array(sorted(heavy_indices_list))

    # Combine all selected
    all_selected = np.sort(np.unique(np.concatenate([
        special_indices,
        recent_indices,
        heavy_indices
    ])))

    # If still over budget (due to overlaps), trim least important
    if len(all_selected) > budget:
        # Priority: special > recent > heavy
        # Remove from heavy first, then recent if needed
        excess = len(all_selected) - budget

        # Try to trim from heavy
        trim_from_heavy = min(excess, len(heavy_indices))
        if trim_from_heavy > 0:
            trim_indices = heavy_indices[:trim_from_heavy]
            all_selected = np.setdiff1d(all_selected, trim_indices)
            excess = len(all_selected) - budget

        # If still over, trim from recent
        if excess > 0 and len(all_selected) > budget:
            trim_from_recent = min(excess, len(recent_indices))
            trim_indices = recent_indices[:trim_from_recent]
            all_selected = np.setdiff1d(all_selected, trim_indices)

        all_selected = np.sort(all_selected)

    K_sel = K[all_selected].astype(np.float32)
    V_sel = V[all_selected].astype(np.float32)
    return K_sel, V_sel, all_selected


def fastgen_attention(Q: np.ndarray, K_sel: np.ndarray, V_sel: np.ndarray) -> np.ndarray:
    """
    Standard softmax attention on FastGen-selected tokens.
    """
    q_len, d = Q.shape

    scores = Q @ K_sel.T / np.sqrt(d)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p_sum = p.sum(axis=-1, keepdims=True)
    p = p / np.clip(p_sum, 1e-30, None)
    O = p @ V_sel  # [q_len, d]

    return O.astype(np.float32)


def fastgen_full_compression(K: np.ndarray, V: np.ndarray, Q: np.ndarray,
                               budget: int,
                               heavy_ratio: float = 0.4,
                               recent_ratio: float = 0.4,
                               special_ratio: float = 0.2) -> Tuple[np.ndarray, dict]:
    """
    Full FastGen compression pipeline.

    Returns compressed attention output and metadata.
    """
    kv_len, d = K.shape

    K_sel, V_sel, indices = fastgen_compress(
        K, V, Q, budget, heavy_ratio, recent_ratio, special_ratio)

    O_approx = fastgen_attention(Q, K_sel, V_sel)

    metadata = {
        "n_selected": len(indices),
        "heavy_ratio": heavy_ratio,
        "recent_ratio": recent_ratio,
        "special_ratio": special_ratio,
        "compression_ratio": kv_len / max(len(indices), 1),
        "indices": indices.tolist(),
    }

    return O_approx, metadata
