"""
AttnStats — FlashAttention online softmax intermediate statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

EPS = 1e-30


@dataclass
class AttnStats:
    m: torch.Tensor  # [H, q_len, 1] log-sum-exp max
    l: torch.Tensor  # [H, q_len, 1] sum-exp
    y: torch.Tensor  # [H, q_len, d] unnormalized weighted sum

    def __post_init__(self):
        H, Ql, _ = self.m.shape
        if self.l.shape != (H, Ql, 1):
            raise ValueError(f"l shape mismatch: {self.l.shape} != {(H, Ql, 1)}")
        if self.y.shape != (H, Ql, self.y.shape[-1]):
            raise ValueError(f"y shape mismatch: {self.y.shape}")

    @classmethod
    def empty(cls, num_heads: int, q_len: int, d: int,
              device="cpu", dtype=torch.float32) -> "AttnStats":
        return cls(
            m=torch.full((num_heads, q_len, 1), float("-inf"), device=device, dtype=dtype),
            l=torch.zeros((num_heads, q_len, 1), device=device, dtype=dtype),
            y=torch.zeros((num_heads, q_len, d), device=device, dtype=dtype),
        )

    def num_heads(self) -> int: return int(self.m.shape[0])
    def q_len(self) -> int: return int(self.m.shape[1])
    def d(self) -> int: return int(self.y.shape[-1])
    def shape_tuple(self) -> Tuple[int, int, int]: return (self.num_heads(), self.q_len(), self.d())

    def finalize(self) -> torch.Tensor:
        return self.y / self.l.clamp(min=EPS)

    def detach_clone(self) -> "AttnStats":
        return AttnStats(m=self.m.detach().clone(), l=self.l.detach().clone(), y=self.y.detach().clone())
