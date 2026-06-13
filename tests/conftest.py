"""Shared pytest fixtures for accordkv tests."""
import pytest
import numpy as np


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_kv(rng):
    """Simple [T, d] KV matrices."""
    return (
        rng.standard_normal((256, 128)).astype(np.float32),
        rng.standard_normal((256, 128)).astype(np.float32),
    )


@pytest.fixture
def multihead_kv(rng):
    """Multi-head [H, T, D] KV matrices."""
    return rng.standard_normal((8, 512, 64)).astype(np.float32)


@pytest.fixture
def simple_q(rng):
    """Simple [q_len, d] query."""
    return rng.standard_normal((64, 128)).astype(np.float32)
