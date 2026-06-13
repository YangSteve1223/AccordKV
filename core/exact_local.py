"""
ExactLocal — 本地 attention baseline（无 stats 路径）。

作为 AVL 降级路径的最后一道防线：
- 远端 server 不可达 / deadline miss 时回退
- Phase 1 仿真中作为"无通信代价"的对照

Phase 2 计划：把 SpectrumKV SWS 接进来做 KV block 选择，
让 ExactLocal 不仅仅"用全部 KV"，而是"用 SWS 选出来的子集"。

接口跟 MockAttentionServer 完全一致 — 这样上层 caller 不感知
走的是 remote 还是 local。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from core.acr import ACR
from core.attn_stats import AttnStats
from core.mock_attention_server import KVCache


class ExactLocal:
    """本地 attention backend。

    Parameters
    ----------
    kv_cache : KVCache
        本地持有的 KV 块（一般 = 全部 KV）
    num_heads : int
        模拟 head 数（Phase 1 固定 1）

    Notes
    -----
    跟 MockAttentionServer 的区别：
    - 语义上是"本地"（不走网络）
    - Phase 2 计划：接入 SWS 做 block 选择（不是用全量 KV）
    - Phase 1 行为跟 server 完全一致（用全量 KV）
    """

    def __init__(
        self,
        kv_cache: KVCache | None = None,
        num_heads: int = 1,
    ):
        self.kv_cache: KVCache = dict(kv_cache) if kv_cache else {}
        self.num_heads = num_heads
        self.requests_served = 0
        self.requests_empty = 0

    def serve(self, acr: ACR) -> AttnStats:
        """本地直跑 attention（不经过网络）。

        跟 MockAttentionServer.serve 几乎相同 — Phase 1 用于对比。
        Phase 2 会在这里插入 SWS block 选择逻辑。
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
        K = torch.cat(K_list, dim=0)
        V = torch.cat(V_list, dim=0)
        Q = acr.q_tokens

        scores = Q @ K.T
        m = scores.max(dim=-1, keepdim=True).values
        p = torch.exp(scores - m)
        l = p.sum(dim=-1, keepdim=True)
        y = p @ V

        return AttnStats(
            m=m.unsqueeze(0),
            l=l.unsqueeze(0),
            y=y.unsqueeze(0),
        )

    def __repr__(self) -> str:
        n_blocks = len(self.kv_cache)
        return (
            f"ExactLocal(blocks={n_blocks}, "
            f"served={self.requests_served}, empty={self.requests_empty})"
        )
