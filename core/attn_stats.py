"""
AttnStats — FlashAttention online softmax 的中间统计量。

这是 AVL 协议的网络化对象。把 attention 计算的"中间状态"当作 first-class
网络对象传输，是 AVL 跟其他分布式 attention 系统（DistCA / Infinite-LLM 等）
的本质差异。

形状约定（Phase 1）：
- m:  [num_heads, q_len, 1]   — log-sum-exp max
- l:  [num_heads, q_len, 1]   — sum-exp
- y:  [num_heads, q_len, d]   — un-normalized weighted sum of V

num_heads 维度恒为 1（Phase 1 不做 multi-head reshape — 留给 Phase 2 接真模型）。

数值稳定性：
- m 初值用 -inf（保证第一次 max(m1, m2) = m1 / m2）
- l 初值用 0
- finalize 用 l.clamp(min=eps) 避免除零
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


# 防止 finalize 时除零的最小分母
EPS = 1e-30


@dataclass
class AttnStats:
    """(m, l, y) — FlashAttention online softmax 状态。

    Examples
    --------
    >>> import torch
    >>> s = AttnStats(
    ...     m=torch.zeros(1, 4, 1),
    ...     l=torch.ones(1, 4, 1),
    ...     y=torch.ones(1, 4, 8),
    ... )
    >>> out = s.finalize()
    >>> out.shape
    torch.Size([1, 4, 8])
    """
    m: torch.Tensor  # [H, q_len, 1]
    l: torch.Tensor  # [H, q_len, 1]
    y: torch.Tensor  # [H, q_len, d]

    def __post_init__(self):
        H, Ql, _ = self.m.shape
        _, _, D = self.y.shape
        if self.l.shape != (H, Ql, 1):
            raise ValueError(
                f"l shape {tuple(self.l.shape)} != m shape {tuple(self.m.shape)}"
            )
        if self.y.shape != (H, Ql, D):
            raise ValueError(
                f"y shape {tuple(self.y.shape)} inconsistent with m {tuple(self.m.shape)}"
            )

    @classmethod
    def empty(
        cls,
        num_heads: int,
        q_len: int,
        d: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "AttnStats":
        """构造零状态（m = -inf, l = 0, y = 0）。

        用于 server 端没找到任何 KV block 时的 fallback。
        """
        return cls(
            m=torch.full((num_heads, q_len, 1), float("-inf"), device=device, dtype=dtype),
            l=torch.zeros((num_heads, q_len, 1), device=device, dtype=dtype),
            y=torch.zeros((num_heads, q_len, d), device=device, dtype=dtype),
        )

    def num_heads(self) -> int:
        return int(self.m.shape[0])

    def q_len(self) -> int:
        return int(self.m.shape[1])

    def d(self) -> int:
        return int(self.y.shape[-1])

    def bytes_size(self, dtype_bytes: int | None = None) -> int:
        """序列化字节数（用于带宽统计）。"""
        if dtype_bytes is None:
            dtype_bytes = self.m.element_size()
        return (self.m.numel() + self.l.numel() + self.y.numel()) * dtype_bytes

    def finalize(self) -> torch.Tensor:
        """从 (m, l, y) 还原 attention 输出: output = y / l。

        返回形状: [num_heads, q_len, d]
        """
        return self.y / self.l.clamp(min=EPS)

    def detach_clone(self) -> "AttnStats":
        """深拷贝 + 切断 autograd 图（用于跨进程传输前）。"""
        return AttnStats(
            m=self.m.detach().clone(),
            l=self.l.detach().clone(),
            y=self.y.detach().clone(),
        )

    def shape_tuple(self) -> Tuple[int, int, int]:
        """返回 (H, Ql, D) 形状三元组。"""
        return (self.num_heads(), self.q_len(), self.d())
