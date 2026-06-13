"""
ACR (Attention Computation Request) — AVL 协议的请求头。

Phase 1 最小字段集：
- 请求身份 (acr_id, q_block_id)
- Q 内容 (q_tokens)
- 路由 (server_hints, block_ids)
- 合约 (contract_type, deadline_ms, error_budget, prefer_local, max_rpc_hops)

设计原则：
- frozen=True: 协议对象一旦发出不可变（多 server 路由时安全）
- 不强制 q_tokens 的 dtype/device（由 caller 决定，server 内部转）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch


class ContractType(str, Enum):
    """合约类型 — 决定 server 端允许的近似程度。"""
    EXACT = "exact"        # 必须 exact attention（Phase 1 默认）
    APPROX = "approx"      # 容许 (m,l,y) 路径 / block-sparse 近似
    BOUNDED = "bounded"    # 数值误差必须 < error_budget


@dataclass(frozen=True)
class ACR:
    """Attention Computation Request。

    Examples
    --------
    >>> import torch
    >>> Q = torch.randn(64, 128)
    >>> acr = ACR(
    ...     acr_id="test-001",
    ...     q_block_id=0,
    ...     q_tokens=Q,
    ...     block_ids=[0, 1, 2, 3],
    ...     server_hints=["srv-A", "srv-B"],
    ...     contract_type=ContractType.EXACT,
    ...     deadline_ms=50.0,
    ... )
    >>> acr.q_block_id
    0
    """
    acr_id: str
    q_block_id: int
    q_tokens: torch.Tensor

    # 路由
    server_hints: List[str] = field(default_factory=list)
    block_ids: List[int] = field(default_factory=list)

    # 合约
    contract_type: ContractType = ContractType.EXACT
    deadline_ms: float = 50.0
    error_budget: float = 0.0

    # 路由策略
    prefer_local: bool = False
    max_rpc_hops: int = 2

    def __post_init__(self):
        # 早期校验：避免在 server 端才发现协议错误
        if not self.acr_id:
            raise ValueError("acr_id must be non-empty")
        if self.q_tokens.ndim != 2:
            raise ValueError(
                f"q_tokens must be 2D [q_len, d], got shape {tuple(self.q_tokens.shape)}"
            )
        if self.deadline_ms <= 0:
            raise ValueError(f"deadline_ms must be > 0, got {self.deadline_ms}")
        if self.contract_type == ContractType.BOUNDED and self.error_budget <= 0:
            raise ValueError(
                "BOUNDED contract requires error_budget > 0, "
                f"got {self.error_budget}"
            )

    @property
    def q_len(self) -> int:
        return int(self.q_tokens.shape[0])

    @property
    def d(self) -> int:
        return int(self.q_tokens.shape[-1])

    def is_local_only(self) -> bool:
        """是否强制走本地（用于 ExactLocal fallback 测试）。"""
        return self.prefer_local and not self.server_hints
