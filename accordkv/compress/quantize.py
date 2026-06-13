"""
quantize — INT4 quantization for KV cache compression.
"""
from __future__ import annotations

import numpy as np

GROUP_SIZE = 128


def int4_quantize(mat: np.ndarray, group_size: int = GROUP_SIZE) -> dict:
    """INT4 quantization with per-group scaling.

    Args:
        mat: [H, r, D] matrix to quantize
        group_size: quantization group size

    Returns:
        dict with quantized values, scales, original_dim
    """
    H, r, D_orig = mat.shape
    pad = (group_size - D_orig % group_size) % group_size
    if pad:
        mat = np.pad(mat, ((0, 0), (0, 0), (0, pad)))

    ng = D_orig // group_size
    g = mat.reshape(H, r, ng, group_size)
    scales = np.amax(np.abs(g), axis=-1, keepdims=True).clip(1e-8)
    q = np.round(g / scales).astype(np.int8)

    return dict(q=q, scales=scales, D_orig=D_orig, group_size=group_size)


def int4_dequantize(comp: dict) -> np.ndarray:
    """Dequantize INT4 compressed matrix.

    Args:
        comp: output of int4_quantize

    Returns:
        dequantized [H, r, D_orig]
    """
    q, scales = comp["q"], comp["scales"]
    D_orig = comp["D_orig"]
    H, r, ng, G = q.shape
    return (q.astype(float) * scales).reshape(H, r, ng * G)[:, :, :D_orig]
