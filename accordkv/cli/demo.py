#!/usr/bin/env python3
"""
accord-demo — CPU-friendly demonstration of ACCORD-KV compression.

Shows:
1. SVD compression + reconstruction
2. Coreset compression + attention
3. INT4 quantization
4. Merge commutativity (empty + empty)
"""
import numpy as np

from accordkv.merge import NumpyAttnStats, numpy_merge_stats
from accordkv.compress import svd_compress, svd_decompress, coreset_compress, coreset_attention, int4_quantize, int4_dequantize


def demo():
    print("=" * 60)
    print("ACCORD-KV CPU Demo")
    print("=" * 60)

    np.random.seed(42)
    d, T = 128, 512

    # 1. SVD compression
    print("\n[1] SVD Compression")
    V = np.random.randn(T, d).astype(np.float32)
    comp = svd_compress(V, rank=8)
    rec = svd_decompress(comp)
    err = np.abs(V - rec).mean() / (np.abs(V).mean() + 1e-8)
    print(f"    V shape: {V.shape}, rank: 8")
    print(f"    Reconstruction error: {err * 100:.4f}%")

    # 2. Coreset attention
    print("\n[2] Coreset Attention")
    K = np.random.randn(T, d).astype(np.float32)
    Q = np.random.randn(32, d).astype(np.float32)
    sketch = coreset_compress(K, V, r=32)
    out = coreset_attention(Q, sketch)
    print(f"    KV: {T}x{d}, Coreset: r=32")
    print(f"    Output shape: {out.shape}")

    # 3. INT4 quantization
    print("\n[3] INT4 Quantization")
    V_3d = np.random.randn(1, 8, 256).astype(np.float32)
    qcomp = int4_quantize(V_3d)
    rec_3d = int4_dequantize(qcomp)
    err_int4 = np.abs(V_3d - rec_3d).mean() / (np.abs(V_3d).mean() + 1e-8)
    print(f"    Original: {V_3d.shape}, Quantized: {qcomp['q'].shape}")
    print(f"    Reconstruction error: {err_int4 * 100:.4f}%")

    # 4. Merge commutativity
    print("\n[4] Merge Commutativity")
    ea = NumpyAttnStats.empty(1, 4, 8)
    eb = NumpyAttnStats.empty(1, 4, 8)
    m1 = numpy_merge_stats(ea, eb)
    m2 = numpy_merge_stats(eb, ea)
    l_match = np.allclose(m1.l, m2.l)
    print(f"    empty + empty: l={m1.l[0,0,0]:.4f}")
    print(f"    merge(a,b) == merge(b,a): {l_match}")

    # 5. Merge associativity
    print("\n[5] Merge Associativity")
    a = NumpyAttnStats(
        m=np.array([[[1.0]]]),
        l=np.array([[[2.0]]]),
        y=np.array([[[3.0, 4.0]]]),
    )
    b = NumpyAttnStats(
        m=np.array([[[2.0]]]),
        l=np.array([[[3.0]]]),
        y=np.array([[[5.0, 6.0]]]),
    )
    c = NumpyAttnStats(
        m=np.array([[[1.5]]]),
        l=np.array([[[1.5]]]),
        y=np.array([[[2.0, 2.5]]]),
    )
    m_ab_c = numpy_merge_stats(numpy_merge_stats(a, b), c)
    m_a_bc = numpy_merge_stats(a, numpy_merge_stats(b, c))
    assoc_ok = np.allclose(m_ab_c.m, m_a_bc.m) and np.allclose(m_ab_c.l, m_a_bc.l)
    print(f"    merge(merge(a,b),c) == merge(a,merge(b,c)): {assoc_ok}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
