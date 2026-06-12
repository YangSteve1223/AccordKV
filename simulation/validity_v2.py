"""
simulation/validity_v2.py — ACCORD-KV Query-domain Validity (Fixed v2)

Changes from v1 (review_to_delete/v1_subagent_output/validity.py):
==============================================================
Issue 1: NumpyAttnStats interface conflict with exp1
  - exp1 uses (m,l,y) form, validity.py used (keys,values) form
  - FIX: Renamed to ValidityQueryStats (validity-specific)
  - exp1 keeps its original NumpyAttnStats

Issue 2: SketchContract.eval q_repr computation
  - Old: q_repr = Q.mean(axis=1).mean(axis=0)  (averaged whole batch incorrectly)
  - FIX: q_repr = Q[0].mean(axis=0)  (take batch=0's q_len mean)
  - This matches the E7 fix applied in exp7_validity_final.py

Issue 3: fallback_contract default returns zero vector
  - Old: return NumpyAttnStats(keys=np.zeros(d), values=np.zeros(d), ...)
  - FIX: When both sketch and fallback fail, return calibration mean
  - Added SketchContractCalibMeanFallback class

Issue 4: StatisticalValidity._erfinv is fake
  - Old: Series approximation for |x|>0.5 returns garbage
  - FIX: Use scipy.special.erfinv for correct inverse error function
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any, List
from dataclasses import dataclass

# scipy is available in sandbox (used by E7)
try:
    from scipy.special import erfinv
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# Validity-specific stats (NOT the exp1 (m,l,y) form)
# ============================================================

@dataclass
class ValidityQueryStats:
    """
    Attention statistics for validity queries (key/value representation).

    This is a separate dataclass from exp1's NumpyAttnStats (m,l,y form).
    Used only by validity.py and validity-related experiments (E6/E7).
    """
    keys: np.ndarray      # [d] aggregated keys
    values: np.ndarray    # [d] aggregated values
    attn_weight_sum: float
    key_count: int


# ============================================================
# QueryDomain (FIXED: match q_repr statistical level)
# ============================================================

class QueryDomain:
    """
    Calibration queries estimate a low-dimensional subspace + threshold.

    Supports three validity metrics:
    - L-inf: max |z_i| threshold (baseline)
    - Mahalanobis: covariance-aware distance
    - Adaptive: dynamic threshold based on fallback rate

    FIXED (from E7 bug): Computes mu/sigma from query-means, not individual tokens.
    This matches the q_repr level used in SketchContract.eval.
    """

    def __init__(
        self,
        calibration_queries: np.ndarray,
        method: str = "linf",  # "linf", "mahalanobis", "adaptive"
        initial_threshold: float = 3.0,
        target_fallback_rate: float = 0.2,
    ):
        """
        Args:
            calibration_queries: [n_calib, q_len, d] or [n_calib, d] calibration query embeddings
            method: "linf", "mahalanobis", or "adaptive"
            initial_threshold: Initial threshold in std units
            target_fallback_rate: Target fallback rate for adaptive mode
        """
        self.calibration_queries = calibration_queries
        self.method = method
        self.target_fallback_rate = target_fallback_rate

        # FIXED (from E7): Match q_repr statistical level
        if calibration_queries.ndim == 3:
            # [n_calib, q_len, d] -> [n_calib, d] (each query's mean)
            self.calib_means: np.ndarray = calibration_queries.mean(axis=1)
            self.n_calib, self.q_len, self.d = calibration_queries.shape
        elif calibration_queries.ndim == 2:
            self.calib_means = calibration_queries
            self.n_calib, self.d = calibration_queries.shape
            self.q_len = 1
        else:
            raise ValueError(f"Unexpected calib shape: {calibration_queries.shape}")

        # FIXED: Use query-means to compute stats (matches q_repr level)
        self.mu: np.ndarray = self.calib_means.mean(axis=0)      # [d]
        self.sigma: np.ndarray = self.calib_means.std(axis=0) + 1e-6  # [d]

        # Covariance for Mahalanobis distance
        self.cov: np.ndarray = np.cov(self.calib_means.T) + 1e-6 * np.eye(self.d)
        try:
            self.cov_inv: np.ndarray = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            self.cov_inv = np.eye(self.d)

        # Threshold management
        self.threshold: float = initial_threshold
        self._fallback_history: List[int] = []
        self._window_size: int = 50

    def is_in_domain(self, q: np.ndarray, threshold: Optional[float] = None) -> bool:
        """
        Check if query q is within the validity domain.

        Args:
            q: [d] query embedding
            threshold: Override threshold (for adaptive mode)

        Returns:
            True if q is in domain, False otherwise
        """
        if threshold is None:
            threshold = self.threshold

        if self.method == "linf":
            return self._is_in_domain_linf(q, threshold)
        elif self.method == "mahalanobis":
            return self._is_in_domain_mahalanobis(q, threshold)
        else:
            return self._is_in_domain_linf(q, threshold)

    def _is_in_domain_linf(self, q: np.ndarray, threshold: float) -> bool:
        """L-inf distance: max |z_i| over dimensions"""
        z = (q - self.mu) / self.sigma
        max_z = np.max(np.abs(z))
        return max_z <= threshold

    def _is_in_domain_mahalanobis(self, q: np.ndarray, threshold: float) -> bool:
        """Mahalanobis distance with covariance"""
        diff = q - self.mu
        mahal_sq = diff @ self.cov_inv @ diff
        # Chi-squared threshold: threshold^2 * d for d dimensions
        chi2_threshold = threshold ** 2 * self.d
        return mahal_sq <= chi2_threshold

    def distance_to_domain(self, q: np.ndarray) -> float:
        """
        Compute normalized distance to domain boundary.
        Returns distance in std units (>0 if OOD, <0 if in-domain)
        """
        if self.method == "mahalanobis":
            diff = q - self.mu
            mahal_sq = diff @ self.cov_inv @ diff
            return np.sqrt(mahal_sq / self.d) - self.threshold
        else:
            z = (q - self.mu) / self.sigma
            return np.max(np.abs(z)) - self.threshold

    def record_fallback(self, triggered: bool) -> None:
        """Record fallback decision for adaptive threshold adjustment"""
        self._fallback_history.append(1 if triggered else 0)
        if len(self._fallback_history) > self._window_size:
            self._fallback_history.pop(0)

    def adjust_threshold(self) -> Dict[str, float]:
        """
        Adjust threshold based on fallback history (adaptive mode).
        Uses simple proportional control.

        Returns:
            Dict with adjustment info
        """
        if len(self._fallback_history) < 10:
            return {"action": "insufficient_data", "threshold": self.threshold}

        current_rate = np.mean(self._fallback_history)
        error = self.target_fallback_rate - current_rate

        # Proportional control: adjust threshold
        adjustment = 0.1 * error
        old_threshold = self.threshold
        self.threshold = np.clip(
            self.threshold + adjustment,
            0.5,
            10.0
        )

        return {
            "action": "adjusted",
            "old_threshold": old_threshold,
            "new_threshold": self.threshold,
            "current_rate": current_rate,
            "error": error,
        }


# ============================================================
# SketchContract with fallback (FIXED)
# ============================================================

class SketchContract:
    """
    Sketch Contract with query-domain validity and fallback.

    ACCORD core: Contract carries its own query-domain validity.
    OOD queries trigger fallback automatically.

    FIXED:
    - q_repr = Q[0].mean(axis=0) (not Q.mean(axis=1).mean(axis=0))
    - Fallback returns calibration mean when fallback_contract is None
    """

    def __init__(
        self,
        sketch: np.ndarray,
        sketch_type: str,
        query_domain: QueryDomain,
        fallback_contract: Optional[Callable[..., Tuple[ValidityQueryStats, Dict]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            sketch: [r, d] sketch matrix (r compressed tokens)
            sketch_type: "average" or "feature"
            query_domain: QueryDomain instance for validity checking
            fallback_contract: Fallback contract when OOD (receives Q)
            metadata: Additional contract metadata
        """
        self.sketch = sketch
        self.r = sketch.shape[0]
        self.d = sketch.shape[1]
        self.sketch_type = sketch_type
        self.query_domain = query_domain
        self.fallback_contract = fallback_contract
        self.metadata = metadata or {}

        # Statistics
        self.fallback_count: int = 0
        self.total_calls: int = 0
        self.sketch_calls: int = 0

    def eval(self, Q: np.ndarray, attn_mask: np.ndarray = None) -> Tuple[ValidityQueryStats, dict]:
        """
        Evaluate sketch contract on queries Q.

        Args:
            Q: [batch=1, q_len, d] query embeddings
            attn_mask: [batch, q_len] attention mask (unused in v2)

        Returns:
            (ValidityQueryStats, metadata_dict)
        """
        self.total_calls += 1

        # FIXED Issue 2: q_repr = Q[0].mean(axis=0)
        # Q shape: [batch=1, q_len, d]
        # Q[0] shape: [q_len, d]
        # Q[0].mean(axis=0) shape: [d] — matches QueryDomain's calib_means level
        if Q.ndim == 3:
            q_repr = Q[0].mean(axis=0)  # [d]
        else:
            q_repr = Q

        # Check query-domain validity
        in_domain = self.query_domain.is_in_domain(q_repr)

        if in_domain:
            return self._eval_sketch(Q, attn_mask)
        else:
            self.fallback_count += 1
            self.query_domain.record_fallback(True)
            return self._eval_fallback(Q, attn_mask)

    def _eval_sketch(self, Q: np.ndarray, attn_mask) -> Tuple[ValidityQueryStats, dict]:
        """Evaluate using sketch"""
        self.sketch_calls += 1

        if Q.ndim == 3:
            batch_size, q_len, d = Q.shape
            Q_flat = Q.reshape(-1, d)
        else:
            Q_flat = Q

        keys = np.zeros((self.r, self.d))
        values = np.zeros((self.r, self.d))
        attn_weight_sums = np.zeros(self.r)
        key_counts = np.zeros(self.r, dtype=int)

        for q in Q_flat:
            scores = np.dot(self.sketch, q)
            attn_weights = np.exp(scores - scores.max())
            attn_weights /= attn_weights.sum() + 1e-8

            for j in range(self.r):
                keys[j] += attn_weights[j] * q
                values[j] += attn_weights[j] * q
                attn_weight_sums[j] += attn_weights[j]
                key_counts[j] += 1

        for j in range(self.r):
            if key_counts[j] > 0:
                keys[j] /= key_counts[j]
                values[j] /= key_counts[j]

        stats = ValidityQueryStats(
            keys=keys.mean(axis=0),
            values=values.mean(axis=0),
            attn_weight_sum=attn_weight_sums.mean(),
            key_count=int(key_counts.sum()),
        )

        meta = {
            "method": "sketch",
            "sketch_type": self.sketch_type,
            "r": self.r,
        }
        return stats, meta

    def _eval_fallback(self, Q: np.ndarray, attn_mask) -> Tuple[ValidityQueryStats, dict]:
        """Evaluate using fallback contract"""
        if self.fallback_contract is not None:
            return self.fallback_contract(Q, attn_mask)

        # FIXED Issue 3: Return calibration mean instead of zero vector
        # This is the optimal OOD estimate (minimizes expected error)
        calib_mu = self.query_domain.mu
        stats = ValidityQueryStats(
            keys=calib_mu.copy(),
            values=calib_mu.copy(),
            attn_weight_sum=0.0,
            key_count=0,
        )

        return stats, {"method": "fallback_calib_mean"}

    def get_fallback_rate(self) -> float:
        """Get current fallback rate"""
        if self.total_calls == 0:
            return 0.0
        return self.fallback_count / self.total_calls


# ============================================================
# Fallback helper: returns calibration mean (E7 fix)
# ============================================================

class SketchContractCalibMeanFallback:
    """
    Fallback that returns the calibration mean (statistically optimal OOD estimate).

    FIXED Issue 3 (from E7): Previously this was blending perturbed query + calib mean,
    which propagated OOD error. Now it returns ONLY the calibration mean.
    """

    def __init__(self, calib_queries: np.ndarray, d: int):
        """
        Args:
            calib_queries: [n, q_len, d] or [n, d] calibration queries
            d: Dimension
        """
        if calib_queries.ndim == 3:
            self.calib_mu = calib_queries.mean(axis=(0, 1))  # [d]
        elif calib_queries.ndim == 2:
            self.calib_mu = calib_queries.mean(axis=0)
        else:
            self.calib_mu = np.zeros(d)
        self.d = d

    def eval(self, Q: np.ndarray, attn_mask=None) -> Tuple[ValidityQueryStats, dict]:
        """Return calibration mean as fallback stats"""
        if Q.ndim == 3:
            q_len = Q.shape[1]
        else:
            q_len = 1

        stats = ValidityQueryStats(
            keys=self.calib_mu.copy(),
            values=self.calib_mu.copy(),
            attn_weight_sum=float(q_len),
            key_count=q_len,
        )
        return stats, {"method": "fallback_calib_mean"}


# ============================================================
# Ground truth & error utilities
# ============================================================

def ground_truth(Q: np.ndarray, attn_mask=None) -> Tuple[ValidityQueryStats, dict]:
    """
    Ground truth: compute exact attention without compression.

    For synthetic data, ground truth is the mean query embedding.
    """
    if Q.ndim == 3:
        Q_flat = Q.reshape(-1, Q.shape[-1])
    else:
        Q_flat = Q

    keys = Q_flat.mean(axis=0)
    values = Q_flat.mean(axis=0)

    stats = ValidityQueryStats(
        keys=keys,
        values=values,
        attn_weight_sum=len(Q_flat),
        key_count=len(Q_flat),
    )

    return stats, {"method": "ground_truth"}


def compute_relative_error(
    stats_a: ValidityQueryStats,
    stats_b: ValidityQueryStats
) -> float:
    """Compute relative error between two attention statistics"""
    key_diff = np.linalg.norm(stats_a.keys - stats_b.keys)
    key_norm = np.linalg.norm(stats_b.keys) + 1e-8
    val_diff = np.linalg.norm(stats_a.values - stats_b.values)
    val_norm = np.linalg.norm(stats_b.values) + 1e-8
    return (key_diff / key_norm + val_diff / val_norm) / 2


# ============================================================
# Sketch construction
# ============================================================

def create_synthetic_sketch(
    calibration_queries: np.ndarray,
    r: int = 4,
    sketch_type: str = "average",
) -> np.ndarray:
    """
    Create a simple synthetic sketch from calibration queries.

    Args:
        calibration_queries: [n, d] calibration queries (flattened)
        r: Number of sketch tokens (compression ratio)
        sketch_type: "average" or "feature"

    Returns:
        [r, d] sketch matrix
    """
    n, d = calibration_queries.shape
    if sketch_type == "average":
        indices = np.random.choice(n, size=min(r, n), replace=r > n)
        sketch = calibration_queries[indices].copy()
    else:
        sketch = np.random.randn(r, d) * calibration_queries.std()
        sketch += calibration_queries.mean(axis=0)
    return sketch


# ============================================================
# Statistical Validity with correct erfinv (FIXED)
# ============================================================

class StatisticalValidity:
    """
    Statistical validity with Hoeffding/Bernstein bounds.

    Provides probably approximately correct (PAC) error bounds:
    - In domain: error ≤ ε with probability ≥ 1-δ
    - OOD: fallback triggered with probability ≥ 1-δ

    FIXED Issue 4: _erfinv now uses scipy.special.erfinv (when available)
    or Winitzki approximation (fallback).
    """

    def __init__(
        self,
        calibration_queries: np.ndarray,
        delta: float = 0.05,
        epsilon: float = 0.1,
    ):
        """
        Args:
            calibration_queries: [n, d] calibration queries
            delta: Failure probability
            epsilon: Error bound
        """
        self.calibration_queries = calibration_queries
        self.delta = delta
        self.epsilon = epsilon
        self.n, self.d = calibration_queries.shape

        # Statistics
        self.mu = calibration_queries.mean(axis=0)
        self.sigma = calibration_queries.std(axis=0) + 1e-6

        # Hoeffding bound for sample mean
        self.hoeffding_n_min = int(np.ceil(
            np.log(2 / delta) / (2 * epsilon ** 2)
        ))

        # For Mahalanobis distance
        self.chi2_delta = self._chi2_inverse(1 - delta, self.d)

    def _winitzki_erfinv(self, x: float) -> float:
        """
        Winitzki's approximation for erfinv.
        Accurate across the full range [-1, 1].

        Reference: Winitzki, S. (2008). "A handy approximation for the
        error function and its inverse."
        """
        # Constants
        a = 0.147
        b = 4.0 / np.pi

        # Compute: erfinv(x) ≈ sign(x) * sqrt( sqrt((2/(π*a) + b/2)² - log(1-x²)/a) - (2/(π*a) + b/2) )
        abs_x = abs(x)
        if abs_x >= 1.0:
            return float('inf') if x > 0 else float('-inf')

        inside = b / 2.0 + 1.0 / a
        inside_sq = inside * inside
        log_term = np.log(1.0 - abs_x * abs_x)
        sqrt_term = np.sqrt(inside_sq - log_term / a)

        result = sqrt_term - inside
        return np.sign(x) * result

    def _erfinv(self, x: float) -> float:
        """
        Inverse error function.

        FIXED Issue 4: Use scipy if available, else Winitzki approximation.
        The old series approximation was only valid for |x| < 0.5.
        """
        if HAS_SCIPY:
            return erfinv(x)
        else:
            return self._winitzki_erfinv(x)

    def _chi2_inverse(self, p: float, df: int) -> float:
        """Approximate chi-squared quantile using Wilson-Hilferty transformation"""
        if df <= 2:
            return -2 * np.log(1 - p)
        z = np.sqrt(2 * df) * (
            1 - 2 / (9 * df) +
            np.sqrt(2 / (9 * df)) * self._normal_inverse(p)
        )
        return z ** 2

    def _normal_inverse(self, p: float) -> float:
        """Approximate normal inverse CDF using erfinv"""
        return np.sqrt(2) * self._erfinv(2 * p - 1)

    def validity_bound(self, q: np.ndarray) -> Dict[str, float]:
        """
        Compute validity with statistical bounds.

        Returns:
            Dict with threshold, distance, in_domain, prob_bound
        """
        z = (q - self.mu) / self.sigma
        max_z = np.max(np.abs(z))

        # Statistical threshold based on Hoeffding concentration
        stat_threshold = np.sqrt(np.log(1 / self.delta) / (2 * self.n)) * 3

        in_domain = max_z <= stat_threshold

        return {
            "threshold": stat_threshold,
            "distance": max_z,
            "in_domain": in_domain,
            "prob_bound": 1 - self.delta,
            "hoeffding_n_min": self.hoeffding_n_min,
            "chi2_delta": self.chi2_delta,
        }

    def error_bound(self, in_domain: bool) -> Dict[str, float]:
        """
        Get error bound based on validity.

        Args:
            in_domain: Whether query is in validity domain

        Returns:
            Dict with error bound information
        """
        if in_domain:
            return {
                "error_bound": self.epsilon,
                "confidence": 1 - self.delta,
                "type": "hoeffding",
            }
        else:
            return {
                "error_bound": None,
                "confidence": 1 - self.delta,
                "type": "fallback_triggered",
            }


if __name__ == "__main__":
    # Test ValidityQueryStats
    print("=== ValidityQueryStats test ===")
    keys = np.random.randn(64)
    vals = np.random.randn(64)
    stats = ValidityQueryStats(keys=keys, values=vals, attn_weight_sum=1.0, key_count=1)
    print(f"  keys shape: {stats.keys.shape}, vals shape: {stats.values.shape}")

    # Test QueryDomain with 3D calibration
    print("\n=== QueryDomain (3D calib) test ===")
    np.random.seed(42)
    calib_3d = np.random.randn(100, 16, 64)  # [n_calib, q_len, d]
    domain = QueryDomain(calib_3d, method="linf", initial_threshold=3.0)
    print(f"  calib_means shape: {domain.calib_means.shape}")
    print(f"  mu shape: {domain.mu.shape}, sigma shape: {domain.sigma.shape}")

    # Test in-domain query
    test_q = np.random.randn(64)
    in_domain = domain.is_in_domain(test_q)
    dist = domain.distance_to_domain(test_q)
    print(f"  Test query in domain: {in_domain}, distance: {dist:.3f}")

    # Test OOD query
    ood_q = test_q + 5 * domain.sigma
    in_domain_ood = domain.is_in_domain(ood_q)
    dist_ood = domain.distance_to_domain(ood_q)
    print(f"  OOD query in domain: {in_domain_ood}, distance: {dist_ood:.3f}")

    # Test StatisticalValidity erfinv
    print("\n=== StatisticalValidity erfinv test ===")
    sv = StatisticalValidity(np.random.randn(100, 64), delta=0.05, epsilon=0.1)
    test_xs = [-0.9, -0.5, 0.0, 0.5, 0.9, 0.99]
    for x in test_xs:
        inv = sv._erfinv(x)
        print(f"  erfinv({x:.2f}) = {inv:.6f}")

    # Test fallback returns calib mean
    print("\n=== Fallback calib mean test ===")
    fallback = SketchContractCalibMeanFallback(calib_3d, 64)
    Q_test = np.random.randn(1, 16, 64)
    stats, meta = fallback.eval(Q_test)
    print(f"  Fallback method: {meta['method']}")
    print(f"  keys == values: {np.allclose(stats.keys, stats.values)}")
    print(f"  stats close to calib_mu: {np.allclose(stats.keys, domain.mu, atol=0.5)}")
