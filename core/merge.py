"""
merge_stats — FlashAttention online softmax 状态合并。

核心数学：
    给定两段独立的 attention stats（m1, l1, y1）和（m2, l2, y2），
    合并后等价于一次性跑完整段的 stats。

    m_new = max(m1, m2)
    α1 = exp(m1 - m_new), α2 = exp(m2 - m_new)
    l_new = l1 * α1 + l2 * α2
    y_new = y1 * α1 + y2 * α2

边界 case（review 发现）：
    当 a 和 b 都是 empty（m 全为 -inf）时：
        m_new = -inf
        a.m - m_new = -inf - (-inf) = NaN
        exp(NaN) = NaN
        l_new = 0 * NaN + 0 * NaN = NaN   ← 错
    修法：override 到 `l_a / (l_a + l_b + EPS)`，按 l 比例分（l=0 时分母 EPS，不会 NaN）。
    语义：当两边都没找到 KV block 时，合并后仍然是"空"状态（l=0, y=0, m=-inf）。

性质：
- associative:  merge(a, merge(b, c)) == merge(merge(a, b), c)
- commutative:  merge(a, b) == merge(b, a)
- 数值稳定:    exp(m_i - m_new) 总是 in [0, 1]（对非 -inf 的 max）

Phase 1 限制：
- 不做 num_heads 维度的并行化（每 head 独立调用一次）
- 不做 in-place（避免 autograd 图被破坏）
- 不做 dtype promotion（要求 a, b 的 dtype 一致 — 后续 Phase 加 cast）
"""

from __future__ import annotations

import torch

from core.attn_stats import AttnStats, EPS


def merge_stats(a: AttnStats, b: AttnStats) -> AttnStats:
    """合并两段 AttnStats。

    Parameters
    ----------
    a, b : AttnStats
        两段独立的 attention 统计。形状必须一致。

    Returns
    -------
    AttnStats
        合并后的 stats，与输入同形状。

    Examples
    --------
    >>> import torch
    >>> from core.attn_stats import AttnStats
    >>> a = AttnStats(
    ...     m=torch.tensor([[[1.0]]]),
    ...     l=torch.tensor([[[2.0]]]),
    ...     y=torch.tensor([[[3.0, 4.0]]]),
    ... )
    >>> b = AttnStats(
    ...     m=torch.tensor([[[2.0]]]),
    ...     l=torch.tensor([[[3.0]]]),
    ...     y=torch.tensor([[[5.0, 6.0]]]),
    ... )
    >>> m = merge_stats(a, b)
    >>> m.m
    tensor([[[2.]]])
    >>> # empty + empty 应该返回空状态（l=0, 不是 NaN）
    >>> ea = AttnStats.empty(1, 4, 8)
    >>> eb = AttnStats.empty(1, 4, 8)
    >>> em = merge_stats(ea, eb)
    >>> em.l[0, 0, 0].item()
    0.0
    """
    if a.shape_tuple() != b.shape_tuple():
        raise ValueError(
            f"shape mismatch: a={a.shape_tuple()}, b={b.shape_tuple()}"
        )
    if a.m.dtype != b.m.dtype:
        raise ValueError(
            f"dtype mismatch: a={a.m.dtype}, b={b.m.dtype} "
            "(add cast before merge)"
        )

    m_new = torch.maximum(a.m, b.m)

    # 默认公式: α_i = exp(m_i - m_new)
    alpha_a = torch.exp(a.m - m_new)
    alpha_b = torch.exp(b.m - m_new)

    # Override: 当 m_new = -inf（两边都是 empty）时
    #   -inf - (-inf) = NaN，exp(NaN) = NaN，会污染 l_new
    # 改用按 l 比例分（l=0 时分母 EPS 兜底，结果 0.5/0.5）
    override_mask = torch.isneginf(m_new)
    if override_mask.any():
        denom = a.l + b.l + EPS
        safe_a = a.l / denom
        safe_b = b.l / denom
        alpha_a = torch.where(override_mask, safe_a, alpha_a)
        alpha_b = torch.where(override_mask, safe_b, alpha_b)

    l_new = a.l * alpha_a + b.l * alpha_b
    y_new = a.y * alpha_a + b.y * alpha_b

    return AttnStats(m=m_new, l=l_new, y=y_new)


def merge_stats_list(stats_list: list[AttnStats]) -> AttnStats:
    """折叠合并 — 顺序无关（associative 保证）。

    空列表：返回 None（调用方需自行处理）。
    1 个：直接返回。
    ≥2：左折叠。
    """
    if not stats_list:
        raise ValueError("stats_list is empty")
    if len(stats_list) == 1:
        return stats_list[0]
    result = stats_list[0]
    for s in stats_list[1:]:
        result = merge_stats(result, s)
    return result
