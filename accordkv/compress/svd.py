"""
svd — SVD-based low-rank compression.
"""
from __future__ import annotations

import numpy as np


def svd_compress(kv: np.ndarray, rank: int = 8) -> dict:
    """SVD compression for KV cache.

    Args:
        kv: [H, T, D] or [T, D] KV matrix
        rank: target rank

    Returns:
        dict with U, S, Vh components
    """
    if kv.ndim == 2:
        U, S, Vh = np.linalg.svd(kv, full_matrices=False)
        return dict(
            U=U[:, :rank],
            S=S[:rank],
            Vh=Vh[:rank, :],
        )

    # Multi-head case [H, T, D]
    H, T, D = kv.shape
    U_list, S_list, Vh_list = [], [], []
    for h in range(H):
        Uh, Sh, Vhh = np.linalg.svd(kv[h], full_matrices=False)
        U_list.append(Uh[:, :rank])
        S_list.append(Sh[:rank])
        Vh_list.append(Vhh[:rank, :])

    return dict(
        U=np.stack(U_list),   # [H, T, r]
        S=np.stack(S_list),  # [H, r]
        Vh=np.stack(Vh_list),  # [H, r, D]
    )


def svd_decompress(comp: dict) -> np.ndarray:
    """Reconstruct KV from SVD compression.

    Args:
        comp: output of svd_compress

    Returns:
        reconstructed [H, T, D] or [T, D]
    """
    U, S, Vh = comp["U"], comp["S"], comp["Vh"]

    if U.ndim == 2:
        return np.einsum("tr, r, rd -> td", U, S, Vh)

    return np.einsum("htr, hr, rdh -> htd", U, S, Vh)
