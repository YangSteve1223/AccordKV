"""Numpy-based merge operations (no torch required)."""

from __future__ import annotations

import numpy as np


class NumpyAttnStats:
    """numpy version of (m, l, y) container.

    Shapes:
        m: [H, q_len, 1]
        l: [H, q_len, 1]
        y: [H, q_len, d]
    """
    __slots__ = ("m", "l", "y")

    def __init__(self, m: np.ndarray, l: np.ndarray, y: np.ndarray):
        self.m = m
        self.l = l
        self.y = y

    @classmethod
    def empty(cls, H: int, Ql: int, D: int, dtype=np.float32) -> "NumpyAttnStats":
        return cls(
            m=np.full((H, Ql, 1), -np.inf, dtype=dtype),
            l=np.zeros((H, Ql, 1), dtype=dtype),
            y=np.zeros((H, Ql, D), dtype=dtype),
        )

    def finalize(self) -> np.ndarray:
        return self.y / np.clip(self.l, 1e-30, None)

    def bytes_size(self) -> int:
        return (self.m.size + self.l.size + self.y.size) * self.m.itemsize


def numpy_merge_stats(a: NumpyAttnStats, b: NumpyAttnStats) -> NumpyAttnStats:
    """Merge two NumpyAttnStats segments."""
    if a.m.shape != b.m.shape:
        raise ValueError(f"shape mismatch: {a.m.shape} vs {b.m.shape}")

    m_new = np.maximum(a.m, b.m)
    alpha_a = np.exp(a.m - m_new)
    alpha_b = np.exp(b.m - m_new)

    # Override for empty+empty (m_new = -inf -> NaN path)
    override_mask = np.isneginf(m_new)
    if override_mask.any():
        denom = a.l + b.l + 1e-30
        alpha_a = np.where(override_mask, a.l / denom, alpha_a)
        alpha_b = np.where(override_mask, b.l / denom, alpha_b)

    l_new = a.l * alpha_a + b.l * alpha_b
    y_new = a.y * alpha_a + b.y * alpha_b
    return NumpyAttnStats(m_new, l_new, y_new)
