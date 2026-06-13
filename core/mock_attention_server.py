"""
MockAttentionServer — CPU attention server stub。

Phase 1 目标：
- 验证 (m,l,y) 路径的数学正确性
- 提供一个可调用的 .serve(ACR) -> AttnStats 接口
- 不涉及网络（同一进程内调用）

实现：朴素 attention（不接 SDPA / 不接 flash_attn），仅用于 Phase 1 趋势验证。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from core.acr import ACR
from core.attn_stats import AttnStats


# kv_cache: block_id -> (K, V)
# K, V 形状: [block_size, d]
KVCache = Dict[int, Tuple[torch.Tensor, torch.Tensor]]


class MockAttentionServer:
    """CPU attention server。

    Parameters
    ----------
    server_id : str
        server 唯一 ID（用于日志 / 路由追踪）
    kv_cache : KVCache
        本机持有的 KV 块。缺失块视为"不持有"。
    num_heads : int
        模拟 head 数（Phase 1 固定 1）
    """

    def __init__(
        self,
        server_id: str,
        kv_cache: KVCache | None = None,
        num_heads: int = 1,
    ):
        self.server_id = server_id
        self.kv_cache: KVCache = dict(kv_cache) if kv_cache else {}
        self.num_heads = num_heads
        # 计数器：用于 sanity check
        self.requests_served = 0
        self.requests_empty = 0

    def has_block(self, block_id: int) -> bool:
        return block_id in self.kv_cache

    def serve(self, acr: ACR) -> AttnStats:
        """处理一个 ACR，返回 (m, l, y)。

        流程：
        1. 从 kv_cache 收集 acr.block_ids 对应的 (K, V)
        2. 若全部缺失，返回 empty stats（m=-inf, l=0, y=0）
        3. 否则拼接 K, V，跑朴素 attention（FP32 累加）算 stats
        """
        self.requests_served += 1
        q_len = acr.q_len
        d = acr.d
        H = self.num_heads

        local_blocks = [bid for bid in acr.block_ids if bid in self.kv_cache]
        if not local_blocks:
            self.requests_empty += 1
            return AttnStats.empty(H, q_len, d, device=acr.q_tokens.device, dtype=acr.q_tokens.dtype)

        K_list = [self.kv_cache[bid][0] for bid in local_blocks]
        V_list = [self.kv_cache[bid][1] for bid in local_blocks]
        K = torch.cat(K_list, dim=0)  # [total_kv_len, d]
        V = torch.cat(V_list, dim=0)  # [total_kv_len, d]
        Q = acr.q_tokens             # [q_len, d]

        # 朴素 attention: scores = Q @ K^T
        scores = Q @ K.T             # [q_len, total_kv_len]

        # online softmax stats
        m = scores.max(dim=-1, keepdim=True).values  # [q_len, 1]
        p = torch.exp(scores - m)                      # [q_len, total_kv_len]
        l = p.sum(dim=-1, keepdim=True)               # [q_len, 1]
        y = p @ V                                      # [q_len, d]

        # 加 head 维度
        return AttnStats(
            m=m.unsqueeze(0),  # [1, q_len, 1]
            l=l.unsqueeze(0),
            y=y.unsqueeze(0),
        )

    def __repr__(self) -> str:
        n_blocks = len(self.kv_cache)
        return (
            f"MockAttentionServer(id={self.server_id!r}, blocks={n_blocks}, "
            f"served={self.requests_served}, empty={self.requests_empty})"
        )
