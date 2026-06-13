"""
accordkv - ACCORD-KV: Attention Virtualization Layer for KV Cache Compression

Core exports:
- NumpyAttnStats from merge (numpy-based, no torch required)
- coreset_compress, svd_compress, int4_quantize from compress
- numpy_merge_stats from merge
"""

# numpy-based core (no torch required)
from accordkv.merge import NumpyAttnStats, numpy_merge_stats
from accordkv.compress.coreset import coreset_compress, coreset_attention
from accordkv.compress.svd import svd_compress, svd_decompress
from accordkv.compress.quantize import int4_quantize, int4_dequantize

__version__ = "0.1.0"

__all__ = [
    # numpy core
    "NumpyAttnStats",
    "numpy_merge_stats",
    # compress
    "coreset_compress",
    "coreset_attention",
    "svd_compress",
    "svd_decompress",
    "int4_quantize",
    "int4_dequantize",
]
