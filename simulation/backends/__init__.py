"""
ACCORD Backend 抽象层
====================

支持 4 种 backend 实现:
- FlashAttention2: SM80+, A100 兼容
- vLLM PagedAttention: SM80+, A100 兼容
- HPC-Ops: SM90+, H100/H20 only
- Triton: SM70+, 跨硬件
"""

from .flash_attn import FlashAttention2Backend
from .vllm import VllmPagedAttentionBackend
from .hpc_ops import HPCOpsBackend
from .triton import TritonBackend

__all__ = [
    'FlashAttention2Backend',
    'VllmPagedAttentionBackend',
    'HPCOpsBackend',
    'TritonBackend',
]
