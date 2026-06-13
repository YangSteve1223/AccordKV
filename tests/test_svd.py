"""Test for SVD compression."""
import numpy as np
import pytest

from accordkv.compress.svd import svd_compress, svd_decompress


class TestSVDCompress:
    def test_svd_2d(self):
        V = np.random.randn(256, 128).astype(np.float32)
        comp = svd_compress(V, rank=8)
        assert comp["U"].shape == (256, 8)
        assert comp["S"].shape == (8,)
        assert comp["Vh"].shape == (8, 128)

    def test_svd_multidim(self):
        V = np.random.randn(4, 512, 64).astype(np.float32)
        comp = svd_compress(V, rank=8)
        assert comp["U"].shape == (4, 512, 8)
        assert comp["S"].shape == (4, 8)
        assert comp["Vh"].shape == (4, 8, 64)

    def test_svd_reconstruction_error(self):
        # Low-rank matrix: rank=32 can capture most variance
        U = np.random.randn(128, 32).astype(np.float64)
        V = np.random.randn(32, 64).astype(np.float64)
        V_true = U @ V  # rank-32 matrix
        comp = svd_compress(V_true, rank=32)
        rec = svd_decompress(comp)
        rel_err = np.abs(V_true - rec).mean() / (np.abs(V_true).mean() + 1e-10)
        assert rel_err < 1e-6, f"High reconstruction error: {rel_err}"
