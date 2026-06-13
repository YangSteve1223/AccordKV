"""
ACR (Attention Computation Request) — AVL protocol request header.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import torch


class ContractType(str, Enum):
    EXACT = "exact"
    APPROX = "approx"
    BOUNDED = "bounded"


@dataclass(frozen=True)
class ACR:
    acr_id: str
    q_block_id: int
    q_tokens: torch.Tensor
    server_hints: List[str] = field(default_factory=list)
    block_ids: List[int] = field(default_factory=list)
    contract_type: ContractType = ContractType.EXACT
    deadline_ms: float = 50.0
    error_budget: float = 0.0
    prefer_local: bool = False
    max_rpc_hops: int = 2

    def __post_init__(self):
        if not self.acr_id:
            raise ValueError("acr_id must be non-empty")
        if self.q_tokens.ndim != 2:
            raise ValueError(f"q_tokens must be 2D [q_len, d], got {self.q_tokens.shape}")
        if self.deadline_ms <= 0:
            raise ValueError(f"deadline_ms must be > 0, got {self.deadline_ms}")

    @property
    def q_len(self) -> int:
        return int(self.q_tokens.shape[0])

    @property
    def d(self) -> int:
        return int(self.q_tokens.shape[-1])
