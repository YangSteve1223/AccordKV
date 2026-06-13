"""
exact_local — Local attention baseline (no stats path).
"""
from __future__ import annotations

import torch

from accordkv.core.acr import ACR
from accordkv.core.attn_stats import AttnStats


class KVCache(dict):
    """KV cache: {block_id: (K, V)} where K,V shape [block_size, d]."""
    pass


class ExactLocal:
    def __init__(self, kv_cache: KVCache | None = None, num_heads: int = 1):
        self.kv_cache: KVCache = dict(kv_cache) if kv_cache else {}
        self.num_heads = num_heads

    def serve(self, acr: ACR) -> AttnStats:
        q_len, d = acr.q_len, acr.d
        H = self.num_heads
        local_blocks = [bid for bid in acr.block_ids if bid in self.kv_cache]

        if not local_blocks:
            return AttnStats.empty(H, q_len, d, device=acr.q_tokens.device, dtype=acr.q_tokens.dtype)

        K = torch.cat([self.kv_cache[bid][0] for bid in local_blocks], dim=0)
        V = torch.cat([self.kv_cache[bid][1] for bid in local_blocks], dim=0)
        Q = acr.q_tokens

        scores = Q @ K.T
        m = scores.max(dim=-1, keepdim=True).values
        p = torch.exp(scores - m)
        l = p.sum(dim=-1, keepdim=True)
        y = p @ V
        return AttnStats(m=m.unsqueeze(0), l=l.unsqueeze(0), y=y.unsqueeze(0))
