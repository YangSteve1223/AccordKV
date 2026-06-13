"""Tests for coreset compression."""
import numpy as np
import pytest

from accordkv.compress.coreset import coreset_compress, coreset_attention


class TestCoresetCompress:
    def test_coreset_basic(self, simple_kv, rng):
        K, V = simple_kv
        sketch = coreset_compress(K, V, r=32, seed=0)
        assert sketch["K_centroids"].shape == (32, 128)
        assert sketch["V_centroids"].shape == (32, 128)
        assert sketch["r"] == 32
        assert len(sketch["indices"]) == 32

    def test_coreset_r_larger_than_kvlen(self, simple_kv):
        K, V = simple_kv  # 256
        sketch = coreset_compress(K, V, r=512)  # more than 256
        assert sketch["r"] == 256  # capped

    def test_coreset_attention_shape(self, simple_kv, simple_q):
        K, V = simple_kv
        Q = simple_q  # 64, 128
        sketch = coreset_compress(K, V, r=32)
        out = coreset_attention(Q, sketch)
        assert out.shape == (64, 128)


class TestCoresetDeterminism:
    def test_same_seed_same_result(self, simple_kv):
        K, V = simple_kv
        s1 = coreset_compress(K, V, r=16, seed=123)
        s2 = coreset_compress(K, V, r=16, seed=123)
        assert np.array_equal(s1["indices"], s2["indices"])


class TestCoresetMultiHead:
    def test_multhead_kv(self, multihead_kv, rng):
        K = multihead_kv
        V = multihead_kv.copy()
        # Currently coreset only handles 2D, this tests the boundary
        with pytest.raises(ValueError):
            coreset_compress(K, V, r=8)
