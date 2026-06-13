"""Tests for merge operations."""
import numpy as np
import pytest

from accordkv.merge import NumpyAttnStats, numpy_merge_stats


class TestMergeCommutativity:
    def test_merge_commutative_basic(self):
        a = NumpyAttnStats(
            m=np.array([[[1.0]], [[0.5]]]),
            l=np.array([[[2.0]], [[3.0]]]),
            y=np.array([[[3.0, 4.0]], [[5.0, 6.0]]]),
        )
        b = NumpyAttnStats(
            m=np.array([[[2.0]], [[1.0]]]),
            l=np.array([[[3.0]], [[2.0]]]),
            y=np.array([[[5.0, 6.0]], [[7.0, 8.0]]]),
        )
        m1 = numpy_merge_stats(a, b)
        m2 = numpy_merge_stats(b, a)
        assert np.allclose(m1.m, m2.m), "m should be commutative"
        assert np.allclose(m1.l, m2.l), "l should be commutative"
        assert np.allclose(m1.y, m2.y), "y should be commutative"


class TestMergeAssociativity:
    def test_merge_associative(self):
        a = NumpyAttnStats(m=np.array([[[1.0]]]), l=np.array([[[2.0]]]), y=np.array([[[3.0, 4.0]]]))
        b = NumpyAttnStats(m=np.array([[[2.0]]]), l=np.array([[[3.0]]]), y=np.array([[[5.0, 6.0]]]))
        c = NumpyAttnStats(m=np.array([[[1.5]]]), l=np.array([[[1.5]]]), y=np.array([[[2.0, 2.5]]]))
        m_ab_c = numpy_merge_stats(numpy_merge_stats(a, b), c)
        m_a_bc = numpy_merge_stats(a, numpy_merge_stats(b, c))
        assert np.allclose(m_ab_c.m, m_a_bc.m)
        assert np.allclose(m_ab_c.l, m_a_bc.l)


class TestMergeEmptyCases:
    def test_empty_plus_empty(self):
        ea = NumpyAttnStats.empty(1, 4, 8)
        eb = NumpyAttnStats.empty(1, 4, 8)
        m = numpy_merge_stats(ea, eb)
        assert np.isinf(m.m[0, 0, 0]) and m.m[0, 0, 0] < 0, "m should be -inf"
        assert m.l[0, 0, 0] == 0.0, "l should be 0, not NaN"

    def test_empty_plus_nonempty(self):
        # empty: H=1, Ql=2, D=4; nonempty: H=1, Ql=2, D=4 (same shapes)
        ea = NumpyAttnStats.empty(1, 2, 4)
        b = NumpyAttnStats(
            m=np.array([[[1.0], [0.5]]]),  # [1, 2, 1]
            l=np.array([[[2.0], [3.0]]]),  # [1, 2, 1]
            y=np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]),  # [1, 2, 4]
        )
        m = numpy_merge_stats(ea, b)
        # empty row should propagate from non-empty b
        assert np.allclose(m.m, b.m), "m should be from non-empty"


class TestMergeShapeMismatch:
    def test_shape_mismatch_raises(self):
        a = NumpyAttnStats.empty(1, 4, 8)
        b = NumpyAttnStats.empty(1, 8, 8)  # different q_len
        with pytest.raises(ValueError, match="shape mismatch"):
            numpy_merge_stats(a, b)
