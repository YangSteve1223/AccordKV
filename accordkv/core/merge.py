"""
merge — FlashAttention online softmax stats merging.
"""
from __future__ import annotations

import torch

from accordkv.core.attn_stats import AttnStats, EPS


def merge_stats(a: AttnStats, b: AttnStats) -> AttnStats:
    """Merge two AttnStats segments."""
    if a.shape_tuple() != b.shape_tuple():
        raise ValueError(f"shape mismatch: {a.shape_tuple()} vs {b.shape_tuple()}")

    m_new = torch.maximum(a.m, b.m)
    alpha_a = torch.exp(a.m - m_new)
    alpha_b = torch.exp(b.m - m_new)

    # Override for empty+empty (m_new = -inf -> NaN path)
    override_mask = torch.isneginf(m_new)
    if override_mask.any():
        denom = a.l + b.l + EPS
        alpha_a = torch.where(override_mask, a.l / denom, alpha_a)
        alpha_b = torch.where(override_mask, b.l / denom, alpha_b)

    l_new = a.l * alpha_a + b.l * alpha_b
    y_new = a.y * alpha_a + b.y * alpha_b
    return AttnStats(m=m_new, l=l_new, y=y_new)


def merge_stats_list(stats_list: list[AttnStats]) -> AttnStats:
    if not stats_list:
        raise ValueError("stats_list is empty")
    if len(stats_list) == 1:
        return stats_list[0]
    result = stats_list[0]
    for s in stats_list[1:]:
        result = merge_stats(result, s)
    return result
