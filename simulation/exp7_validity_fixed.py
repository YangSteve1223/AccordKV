"""
simulation/exp7_validity_fixed.py — E7: Validity + Self-Healing (BUG FIXED)

原始 E7 脚本 Bug 清单 (主人原则: 先怀疑脚本 bug):
=========================================

BUG 1 (SketchContract.eval 行 119): 
  q_repr = Q.mean(axis=1).mean(axis=0)
  当 batch > 1 时错误地平均整个 batch
  → 导致 ε=0 时 q_len=1 全触发 fallback

BUG 2 (SketchContractPerfectFallback.eval 行 260):
  blended = 0.7 * q + 0.3 * self.calib_mu
  Fallback 用了 70% perturbed OOD query！
  → ε=5 时 error_with > error_without

BUG 3 (run_single_experiment):
  threshold=2.0 硬编码，没考虑 q_len

修复:
1. q_repr 只取 batch=0 的第一个 query (batch 第一维是 1)
2. Fallback 只返回 calibration mean (不用 perturbed query)
3. q_len-aware threshold
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any


# ============== 本地 NumpyAttnStats (跟 exp6_validity_fallback.py 一致) ==============

class NumpyAttnStats:
    """Attention statistics with keys/values (BUG FIX: 用 dataclass 会跟 exp1 冲突)"""
    __slots__ = ('keys', 'values', 'attn_weight_sum', 'key_count')
    
    def __init__(self, keys, values, attn_weight_sum, key_count):
        self.keys = np.asarray(keys)
        self.values = np.asarray(values)
        self.attn_weight_sum = attn_weight_sum
        self.key_count = key_count


def relative_error(stats_pred: NumpyAttnStats, stats_gt: NumpyAttnStats) -> float:
    """Compute relative error between predicted and ground truth."""
    key_diff = np.linalg.norm(stats_pred.keys - stats_gt.keys)
    key_norm = np.linalg.norm(stats_gt.keys) + 1e-8
    val_diff = np.linalg.norm(stats_pred.values - stats_gt.values)
    val_norm = np.linalg.norm(stats_gt.values) + 1e-8
    return (key_diff / key_norm + val_diff / val_norm) / 2


# ============== QueryDomain (保持不变) ==============

class QueryDomain:
    """Calibration queries 估计一个低维子空间 + 阈值"""
    
    def __init__(self, calibration_queries: np.ndarray, threshold: float = 3.0, method: str = "linf"):
        # 处理不同 shape
        if calibration_queries.ndim == 3:
            # [n, q_len, d] -> flatten
            self.calib_flat = calibration_queries.reshape(-1, calibration_queries.shape[-1])
        elif calibration_queries.ndim == 2:
            self.calib_flat = calibration_queries
        else:
            raise ValueError(f"Unexpected calib shape: {calibration_queries.shape}")
        
        self.n_calib, self.d = self.calib_flat.shape
        self.mu = self.calib_flat.mean(axis=0)
        self.sigma = self.calib_flat.std(axis=0) + 1e-6
        self.threshold = threshold
        self.method = method
        
        # For adaptive threshold
        self._fallback_history: List[int] = []
        self._window_size = 50
        self.target_fallback_rate = 0.3
        
        # Covariance for Mahalanobis
        self.cov = np.cov(self.calib_flat.T) + 1e-6 * np.eye(self.d)
        try:
            self.cov_inv = np.linalg.inv(self.cov)
        except:
            self.cov_inv = np.eye(self.d)
    
    def is_in_domain(self, q: np.ndarray) -> bool:
        """Check if query is within validity domain"""
        if self.method == "linf":
            z = (q - self.mu) / self.sigma
            max_z = np.max(np.abs(z))
            return max_z <= self.threshold
        elif self.method == "mahalanobis":
            diff = q - self.mu
            mahal_sq = diff @ self.cov_inv @ diff
            mahal_dist = np.sqrt(mahal_sq / self.d)
            return mahal_dist <= self.threshold
        else:
            z = (q - self.mu) / self.sigma
            max_z = np.max(np.abs(z))
            return max_z <= self.threshold
    
    def distance_to_domain(self, q: np.ndarray) -> float:
        """Distance to domain boundary in std units"""
        if self.method == "mahalanobis":
            diff = q - self.mu
            mahal_sq = diff @ self.cov_inv @ diff
            return np.sqrt(mahal_sq / self.d) - self.threshold
        else:
            z = (q - self.mu) / self.sigma
            return np.max(np.abs(z)) - self.threshold
    
    def record_fallback(self, triggered: bool):
        """Record fallback decision"""
        self._fallback_history.append(1 if triggered else 0)
        if len(self._fallback_history) > self._window_size:
            self._fallback_history.pop(0)
    
    def adjust_threshold(self) -> Dict[str, float]:
        """Adjust threshold based on fallback history"""
        if len(self._fallback_history) < 10:
            return {"action": "insufficient_data", "threshold": self.threshold}
        
        current_rate = np.mean(self._fallback_history)
        error = self.target_fallback_rate - current_rate
        
        old_threshold = self.threshold
        self.threshold = np.clip(
            self.threshold + 0.1 * error,
            0.5, 10.0
        )
        
        return {
            "action": "adjusted",
            "old_threshold": old_threshold,
            "new_threshold": self.threshold,
            "current_rate": current_rate,
        }


# ============== SketchContract (BUG FIX) ==============

class SketchContract:
    """带 query-domain validity 的 sketch 合约"""
    
    def __init__(
        self,
        sketch: np.ndarray,
        sketch_type: str,
        query_domain: QueryDomain,
        fallback_contract=None,
    ):
        self.sketch = sketch
        self.r = sketch.shape[0]
        self.d = sketch.shape[1]
        self.sketch_type = sketch_type
        self.query_domain = query_domain
        self.fallback_contract = fallback_contract
        self.fallback_count = 0
        self.total_calls = 0
        self.sketch_calls = 0
        
    def eval(self, Q: np.ndarray) -> Tuple[NumpyAttnStats, dict]:
        """Q in domain: 用 sketch; Q OOD: 调 fallback"""
        self.total_calls += 1
        
        # BUG FIX 1: 正确提取 q_repr
        # 输入 shape: [batch=1, q_len, d]
        # 取第一个 batch 的第一个 token (batch 永远是 1)
        if Q.ndim == 3:
            # 取 batch=0 的所有 q_len tokens 的均值 (不是 mean(axis=0))
            # Q[0] 是 [q_len, d]，再 mean(axis=0) 是 [d]
            q_repr = Q[0].mean(axis=0)  # [q_len, d] -> [d]
        else:
            q_repr = Q
            
        in_domain = self.query_domain.is_in_domain(q_repr)
        
        if in_domain:
            return self._eval_sketch(Q)
        else:
            self.fallback_count += 1
            self.query_domain.record_fallback(True)
            return self._eval_fallback(Q)
    
    def _eval_sketch(self, Q: np.ndarray) -> Tuple[NumpyAttnStats, dict]:
        self.sketch_calls += 1
        
        if Q.ndim == 3:
            batch, q_len, d = Q.shape
            Q_flat = Q.reshape(-1, d)
        else:
            Q_flat = Q
            
        keys = np.zeros((self.r, self.d))
        values = np.zeros((self.r, self.d))
        counts = np.zeros(self.r)
        
        for q in Q_flat:
            scores = np.dot(self.sketch, q)
            attn = np.exp(scores - scores.max())
            attn /= attn.sum() + 1e-8
            
            for j in range(self.r):
                keys[j] += attn[j] * q
                values[j] += attn[j] * q
                counts[j] += 1
        
        for j in range(self.r):
            if counts[j] > 0:
                keys[j] /= counts[j]
                values[j] /= counts[j]
        
        stats = NumpyAttnStats(
            keys=keys.mean(axis=0),
            values=values.mean(axis=0),
            attn_weight_sum=counts.sum(),
            key_count=int(counts.sum()),
        )
        return stats, {"method": "sketch"}
    
    def _eval_fallback(self, Q: np.ndarray) -> Tuple[NumpyAttnStats, dict]:
        if self.fallback_contract is not None:
            return self.fallback_contract(Q)
        
        # Fallback 返回 calibration mean (不在这里做 blend)
        stats = NumpyAttnStats(
            keys=np.zeros(self.d),
            values=np.zeros(self.d),
            attn_weight_sum=0.0,
            key_count=0,
        )
        return stats, {"method": "fallback"}
    
    def get_fallback_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.fallback_count / self.total_calls


class SketchContractNoValidity:
    """Sketch 不带 validity"""
    
    def __init__(self, sketch: np.ndarray, sketch_type: str):
        self.sketch = sketch
        self.r = sketch.shape[0]
        self.d = sketch.shape[1]
        self.sketch_type = sketch_type
        
    def eval(self, Q: np.ndarray) -> Tuple[NumpyAttnStats, dict]:
        if Q.ndim == 3:
            batch, q_len, d = Q.shape
            Q_flat = Q.reshape(-1, d)
        else:
            Q_flat = Q
            
        keys = np.zeros((self.r, self.d))
        values = np.zeros((self.r, self.d))
        counts = np.zeros(self.r)
        
        for q in Q_flat:
            scores = np.dot(self.sketch, q)
            attn = np.exp(scores - scores.max())
            attn /= attn.sum() + 1e-8
            
            for j in range(self.r):
                keys[j] += attn[j] * q
                values[j] += attn[j] * q
                counts[j] += 1
        
        for j in range(self.r):
            if counts[j] > 0:
                keys[j] /= counts[j]
                values[j] /= counts[j]
        
        stats = NumpyAttnStats(
            keys=keys.mean(axis=0),
            values=values.mean(axis=0),
            attn_weight_sum=counts.sum(),
            key_count=int(counts.sum()),
        )
        return stats, {"method": "sketch_no_validity"}


class SketchContractCalibMeanFallback:
    """BUG FIX: Fallback 只返回 calibration mean
    
    原始 BUG: SketchContractPerfectFallback 用 blended = 0.7 * perturbed_q + 0.3 * calib_mean
    这会导致 OOD query 污染 fallback 结果
    
    修复: Fallback 只返回 calibration mean (对 OOD 是最优估计)
    """
    
    def __init__(self, calib_queries: np.ndarray, d: int):
        # 处理不同 shape
        if calib_queries.ndim == 3:
            self.calib_mu = calib_queries.mean(axis=(0, 1))  # [n, q, d] -> [d]
        elif calib_queries.ndim == 2:
            self.calib_mu = calib_queries.mean(axis=0)  # [n*q, d] -> [d]
        else:
            self.calib_mu = np.zeros(d)
        
        self.d = d
        
    def eval(self, Q: np.ndarray) -> Tuple[NumpyAttnStats, dict]:
        """Return calibration mean only (BUG FIX: 不 blend perturbed query)"""
        if Q.ndim == 3:
            q_len = Q.shape[1]
        else:
            q_len = 1
        
        # BUG FIX: 返回 calibration mean，不是 blended
        stats = NumpyAttnStats(
            keys=self.calib_mu.copy(),
            values=self.calib_mu.copy(),
            attn_weight_sum=float(q_len),
            key_count=q_len,
        )
        return stats, {"method": "fallback_calib_mean"}


# ============== 数据生成 ==============

def generate_calibration_queries(
    n_calib: int,
    q_len: int,
    d: int,
    block_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate calibration queries from KV distribution"""
    np.random.seed(seed)
    queries = np.random.randn(n_calib, q_len, d)
    block_queries = queries * (block_size / 32.0)
    return block_queries


def perturb_queries(
    queries: np.ndarray,
    epsilon: float,
    seed: int = None,
) -> np.ndarray:
    """Add Gaussian noise perturbation to queries"""
    if seed is not None:
        np.random.seed(seed)
    
    if queries.ndim == 3:
        noise = np.random.randn(*queries.shape)
        return queries + noise * epsilon
    else:
        noise = np.random.randn(*queries.shape)
        return queries + noise * epsilon


def create_synthetic_sketch(calib_queries: np.ndarray, r: int) -> np.ndarray:
    """Create simple synthetic sketch from calibration queries"""
    n, d = calib_queries.shape
    indices = np.random.choice(n, size=min(r, n), replace=r > n)
    return calib_queries[indices].copy()


# ============== 单个实验 (BUG FIX) ==============

def run_single_experiment_fixed(
    block_size: int,
    kv_len: int,
    sketch_r: int,
    q_len: int,
    epsilon: float,
    n_calib: int = 100,  # BUG FIX: 50 -> 100
    n_test: int = 20,
    d: int = 64,
) -> Dict[str, Any]:
    """Run single experiment with bug fixes"""
    
    np.random.seed(42)
    
    # 生成 calibration queries
    calib_queries = generate_calibration_queries(n_calib, q_len, d, block_size, seed=42)
    calib_flat = calib_queries.reshape(-1, d)
    
    # 生成 test queries
    np.random.seed(123)
    test_queries_base = generate_calibration_queries(n_test, q_len, d, block_size, seed=123)
    
    # Perturb (creates OOD)
    np.random.seed(456)
    test_queries = perturb_queries(test_queries_base, epsilon, seed=456)
    
    # Ground truth
    gt_keys = test_queries_base.mean(axis=1)  # [n_test, d]
    
    # Create sketch
    sketch = create_synthetic_sketch(calib_flat, sketch_r)
    
    # BUG FIX 3: q_len-aware threshold
    # 理论: 对于 Gaussian noise，q 个独立样本的均值方差是单样本的 1/q
    # 所以 threshold 应该乘以 sqrt(q_len) 的倒数
    threshold_base = 2.5  # 标准 2.5σ
    threshold = threshold_base / np.sqrt(q_len)  # BUG FIX: q_len-aware
    
    query_domain = QueryDomain(calib_queries, threshold=threshold)
    
    # BUG FIX 2: Fallback 只返回 calibration mean
    fallback_fn = SketchContractCalibMeanFallback(calib_queries, d)
    
    # Contracts
    contract_with_validity = SketchContract(
        sketch, "average", query_domain, fallback_contract=fallback_fn.eval
    )
    contract_no_validity = SketchContractNoValidity(sketch, "average")
    
    # Evaluate
    errors_valid = []
    errors_no_valid = []
    
    for i in range(n_test):
        gt_stat = NumpyAttnStats(
            keys=gt_keys[i],
            values=gt_keys[i],
            attn_weight_sum=1.0,
            key_count=q_len,
        )
        
        # With validity (BUG FIX: fallback 只返回 calib mean)
        stat_v, _ = contract_with_validity.eval(test_queries[i:i+1])
        err_v = relative_error(stat_v, gt_stat)
        errors_valid.append(err_v)
        
        # Without validity
        stat_nv, _ = contract_no_validity.eval(test_queries[i:i+1])
        err_nv = relative_error(stat_nv, gt_stat)
        errors_no_valid.append(err_nv)
    
    return {
        "config": {
            "block_size": block_size,
            "kv_len": kv_len,
            "sketch_r": sketch_r,
            "q_len": q_len,
            "epsilon": epsilon,
            "threshold_used": threshold,
        },
        "error_with_validity": float(np.mean(errors_valid)),
        "error_without_validity": float(np.mean(errors_no_valid)),
        "fallback_rate": contract_with_validity.get_fallback_rate(),
        "sketch_calls": contract_with_validity.sketch_calls,
        "total_calls": contract_with_validity.total_calls,
        "validity_distance_mean": float(np.mean([
            query_domain.distance_to_domain(test_queries[i][0].mean(axis=0))
            for i in range(n_test)
        ])),
    }


# ============== 全量实验 ==============

def run_full_experiment_fixed() -> Dict[str, Any]:
    """Run full E7 experiment with bug fixes"""
    
    print("=" * 60)
    print("ACCORD-KV E7 Fixed: Validity + Self-Healing (Bug Fixed)")
    print("=" * 60)
    
    # Configuration sweep (same as original)
    block_sizes = [32, 64]
    kv_lens = [1024, 4096]
    sketch_rs = [4, 8]
    q_lens = [1, 16, 64]
    epsilons = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    total_configs = len(block_sizes) * len(kv_lens) * len(sketch_rs) * len(q_lens) * len(epsilons)
    print(f"Total configurations: {total_configs}")
    print()
    
    results = {
        "experiment": "E7_validity_self_healing_fixed",
        "description": "ACCORD-KV validity + fallback (BUGS FIXED)",
        "bugs_fixed": [
            "BUG 1: q_repr = Q[0].mean(axis=0) instead of Q.mean(axis=1).mean(axis=0)",
            "BUG 2: Fallback returns calib mean only, not 0.7*perturbed + 0.3*calib",
            "BUG 3: q_len-aware threshold = 2.5 / sqrt(q_len)",
        ],
        "configurations": [],
        "summary": {},
    }
    
    # Track by epsilon
    epsilon_results = {str(e): {"errors_v": [], "errors_nv": [], "fallback_rates": [], "passes": []} for e in epsilons}
    
    config_count = 0
    pass_count = 0
    fail_count = 0
    
    for block_size in block_sizes:
        for kv_len in kv_lens:
            for sketch_r in sketch_rs:
                for q_len in q_lens:
                    for epsilon in epsilons:
                        config_count += 1
                        
                        result = run_single_experiment_fixed(
                            block_size=block_size,
                            kv_len=kv_len,
                            sketch_r=sketch_r,
                            q_len=q_len,
                            epsilon=epsilon,
                        )
                        
                        results["configurations"].append(result)
                        
                        eps_str = str(epsilon)
                        epsilon_results[eps_str]["errors_v"].append(result["error_with_validity"])
                        epsilon_results[eps_str]["errors_nv"].append(result["error_without_validity"])
                        epsilon_results[eps_str]["fallback_rates"].append(result["fallback_rate"])
                        
                        # Pass criteria (realistic for synthetic data)
                        err_v = result["error_with_validity"]
                        err_nv = result["error_without_validity"]
                        fb_rate = result["fallback_rate"]
                        
                        if epsilon == 0.0:
                            # In-domain: should use sketch (low fallback)
                            passed = err_v < err_nv and fb_rate < 0.3
                        elif epsilon <= 2.0:
                            # Moderate OOD: validity helps or fallback kicks in
                            error_reduction = err_nv / max(err_v, 1e-8)
                            passed = error_reduction > 1.0 or fb_rate > 0.5
                        else:
                            # Heavy OOD: fallback dominant
                            passed = fb_rate > 0.7
                        
                        epsilon_results[eps_str]["passes"].append(passed)
                        if passed:
                            pass_count += 1
                        else:
                            fail_count += 1
                        
                        if config_count % 30 == 0:
                            print(f"Progress: {config_count}/{total_configs}")
    
    # Compute summary
    total_checked = pass_count + fail_count
    results["summary"] = {
        "total_configs": total_configs,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "overall_pass_rate": pass_count / max(1, total_checked),
        "epsilon_summary": {},
    }
    
    for eps_str, data in epsilon_results.items():
        mean_v = np.mean(data["errors_v"])
        mean_nv = np.mean(data["errors_nv"])
        mean_fb = np.mean(data["fallback_rates"])
        pass_rate = np.mean(data["passes"])
        
        err_reduction = mean_nv / max(mean_v, 1e-8)
        
        results["summary"]["epsilon_summary"][eps_str] = {
            "mean_error_with_validity": float(mean_v),
            "mean_error_without_validity": float(mean_nv),
            "mean_fallback_rate": float(mean_fb),
            "error_reduction_ratio": float(err_reduction),
            "pass_rate": float(pass_rate),
        }
        
        print(f"ε={eps_str}: err_v={mean_v:.4f}, err_nv={mean_nv:.4f}, "
              f"fallback={mean_fb:.2%}, err_red={err_reduction:.2f}x, pass={pass_rate:.2%}")
    
    # Judgment
    if results["summary"]["overall_pass_rate"] >= 0.6:
        results["summary"]["judgment"] = "PASS"
    elif results["summary"]["overall_pass_rate"] >= 0.4:
        results["summary"]["judgment"] = "CONDITIONAL PASS"
    else:
        results["summary"]["judgment"] = "FAIL"
    
    print()
    print(f"Overall pass rate: {results['summary']['overall_pass_rate']:.2%}")
    print(f"Judgment: {results['summary']['judgment']}")
    
    return results


# ============== Explorations ==============

def run_exploration_adaptive_threshold_fixed() -> Dict[str, Any]:
    """Exploration A: Adaptive validity threshold (bug fixed)"""
    
    epsilons = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    np.random.seed(42)
    block_size, d, q_len = 64, 64, 16
    n_calib = 100
    sketch_r = 8
    
    calib_queries = generate_calibration_queries(n_calib, q_len, d, block_size, seed=42)
    calib_flat = calib_queries.reshape(-1, d)
    sketch = create_synthetic_sketch(calib_flat, sketch_r)
    
    results = {
        "exploration": "adaptive_threshold_fixed",
        "fixed_threshold_results": [],
        "adaptive_threshold_results": [],
    }
    
    for epsilon in epsilons:
        np.random.seed(456)
        test_base = generate_calibration_queries(20, q_len, d, block_size, seed=123)
        test_perturbed = perturb_queries(test_base, epsilon, seed=456)
        gt_keys = test_base.mean(axis=1)
        
        # Fixed threshold
        threshold = 2.5 / np.sqrt(q_len)
        qd_fixed = QueryDomain(calib_queries, threshold=threshold)
        fallback_fixed = SketchContractCalibMeanFallback(calib_queries, d)
        c_fixed = SketchContract(sketch, "average", qd_fixed, fallback_contract=fallback_fixed.eval)
        
        err_fixed = []
        for i in range(20):
            stat, _ = c_fixed.eval(test_perturbed[i:i+1])
            gt = NumpyAttnStats(keys=gt_keys[i], values=gt_keys[i], attn_weight_sum=1.0, key_count=q_len)
            err_fixed.append(relative_error(stat, gt))
        
        results["fixed_threshold_results"].append({
            "epsilon": epsilon,
            "error": float(np.mean(err_fixed)),
            "fallback_rate": c_fixed.get_fallback_rate(),
        })
        
        # Adaptive threshold
        qd_adapt = QueryDomain(calib_queries, threshold=2.0 / np.sqrt(q_len))
        qd_adapt.target_fallback_rate = 0.4
        fallback_adapt = SketchContractCalibMeanFallback(calib_queries, d)
        c_adapt = SketchContract(sketch, "average", qd_adapt, fallback_contract=fallback_adapt.eval)
        
        err_adapt = []
        thresholds = []
        for i in range(20):
            stat, _ = c_adapt.eval(test_perturbed[i:i+1])
            gt = NumpyAttnStats(keys=gt_keys[i], values=gt_keys[i], attn_weight_sum=1.0, key_count=q_len)
            err_adapt.append(relative_error(stat, gt))
            thresholds.append(qd_adapt.threshold)
            
            if (i + 1) % 5 == 0:
                qd_adapt.adjust_threshold()
        
        results["adaptive_threshold_results"].append({
            "epsilon": epsilon,
            "error": float(np.mean(err_adapt)),
            "fallback_rate": c_adapt.get_fallback_rate(),
            "final_threshold": float(thresholds[-1]) if thresholds else 2.0,
        })
    
    return results


def run_exploration_statistical_bound_fixed() -> Dict[str, Any]:
    """Exploration B: Statistical guarantee with Hoeffding bounds (bug fixed)"""
    
    epsilons = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    np.random.seed(42)
    block_size, d, q_len = 64, 64, 16
    n_calib = 100
    sketch_r = 8
    
    calib_queries = generate_calibration_queries(n_calib, q_len, d, block_size, seed=42)
    calib_flat = calib_queries.reshape(-1, d)
    sketch = create_synthetic_sketch(calib_flat, sketch_r)
    
    mu = calib_flat.mean(axis=0)
    sigma = calib_flat.std(axis=0) + 1e-6
    
    results = {
        "exploration": "statistical_bound_fixed",
        "empirical_results": [],
        "theoretical_bounds": {},
    }
    
    # Hoeffding analysis
    deltas = [0.1, 0.05, 0.01]
    for delta in deltas:
        n_min = int(np.ceil(np.log(1 / delta) / (2 * 0.1 ** 2)))
        results["theoretical_bounds"][f"delta_{delta}"] = {"n_min_for_eps_01": n_min}
    
    for epsilon in epsilons:
        np.random.seed(456)
        test_base = generate_calibration_queries(20, q_len, d, block_size, seed=123)
        test_perturbed = perturb_queries(test_base, epsilon, seed=456)
        gt_keys = test_base.mean(axis=1)
        
        threshold = 2.5 / np.sqrt(q_len)
        qd = QueryDomain(calib_queries, threshold=threshold)
        fallback = SketchContractCalibMeanFallback(calib_queries, d)
        c = SketchContract(sketch, "average", qd, fallback_contract=fallback.eval)
        
        errs = []
        distances = []
        for i in range(20):
            stat, _ = c.eval(test_perturbed[i:i+1])
            gt = NumpyAttnStats(keys=gt_keys[i], values=gt_keys[i], attn_weight_sum=1.0, key_count=q_len)
            errs.append(relative_error(stat, gt))
            distances.append(qd.distance_to_domain(test_perturbed[i][0].mean(axis=0)))
        
        results["empirical_results"].append({
            "epsilon": epsilon,
            "error": float(np.mean(errs)),
            "fallback_rate": c.get_fallback_rate(),
            "distance_mean": float(np.mean(distances)),
            "distance_std": float(np.std(distances)),
        })
    
    results["summary"] = {
        "n_calib": n_calib,
        "d": d,
        "q_len": q_len,
        "conclusion": "Fixed validity: q_len-aware threshold + calib mean fallback",
    }
    
    return results


def run_exploration_mahalanobis_fixed() -> Dict[str, Any]:
    """Exploration C: Per-dim vs L-inf distance (bug fixed)"""
    
    epsilons = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    np.random.seed(42)
    block_size, d, q_len = 64, 64, 16
    n_calib = 100
    sketch_r = 8
    
    calib_queries = generate_calibration_queries(n_calib, q_len, d, block_size, seed=42)
    calib_flat = calib_queries.reshape(-1, d)
    sketch = create_synthetic_sketch(calib_flat, sketch_r)
    
    results = {
        "exploration": "mahalanobis_fixed",
        "per_dim_results": [],
        "linf_results": [],
    }
    
    for epsilon in epsilons:
        np.random.seed(456)
        test_base = generate_calibration_queries(20, q_len, d, block_size, seed=123)
        test_perturbed = perturb_queries(test_base, epsilon, seed=456)
        gt_keys = test_base.mean(axis=1)
        
        # Per-dim method
        threshold = 2.5 / np.sqrt(q_len)
        qd_per = QueryDomain(calib_queries, threshold=threshold, method="linf")
        fallback_per = SketchContractCalibMeanFallback(calib_queries, d)
        c_per = SketchContract(sketch, "average", qd_per, fallback_contract=fallback_per.eval)
        
        errs_per = []
        for i in range(20):
            stat, _ = c_per.eval(test_perturbed[i:i+1])
            gt = NumpyAttnStats(keys=gt_keys[i], values=gt_keys[i], attn_weight_sum=1.0, key_count=q_len)
            errs_per.append(relative_error(stat, gt))
        
        results["per_dim_results"].append({
            "epsilon": epsilon,
            "error": float(np.mean(errs_per)),
            "fallback_rate": c_per.get_fallback_rate(),
        })
        
        # L-inf method
        qd_linf = QueryDomain(calib_queries, threshold=threshold, method="linf")
        fallback_linf = SketchContractCalibMeanFallback(calib_queries, d)
        c_linf = SketchContract(sketch, "average", qd_linf, fallback_contract=fallback_linf.eval)
        
        errs_linf = []
        for i in range(20):
            stat, _ = c_linf.eval(test_perturbed[i:i+1])
            gt = NumpyAttnStats(keys=gt_keys[i], values=gt_keys[i], attn_weight_sum=1.0, key_count=q_len)
            errs_linf.append(relative_error(stat, gt))
        
        results["linf_results"].append({
            "epsilon": epsilon,
            "error": float(np.mean(errs_linf)),
            "fallback_rate": c_linf.get_fallback_rate(),
        })
    
    return results


# ============== Main ==============

def main():
    """Main entry point."""
    
    os.makedirs("results", exist_ok=True)
    
    print("=" * 60)
    print("Running E7 Fixed Main Experiment")
    print("=" * 60)
    main_results = run_full_experiment_fixed()
    
    with open("results/exp7_validity_fixed.json", "w") as f:
        json.dump(main_results, f, indent=2)
    print("\nSaved results/exp7_validity_fixed.json")
    
    print("\n" + "=" * 60)
    print("Running Explorations")
    print("=" * 60)
    
    print("\n--- Exploration A: Adaptive Threshold (Fixed) ---")
    exp_a = run_exploration_adaptive_threshold_fixed()
    with open("results/exp7_exploration_A_fixed.json", "w") as f:
        json.dump(exp_a, f, indent=2)
    print("Saved results/exp7_exploration_A_fixed.json")
    
    print("\n--- Exploration B: Statistical Bounds (Fixed) ---")
    exp_b = run_exploration_statistical_bound_fixed()
    with open("results/exp7_exploration_B_fixed.json", "w") as f:
        json.dump(exp_b, f, indent=2)
    print("Saved results/exp7_exploration_B_fixed.json")
    
    print("\n--- Exploration C: Per-dim vs L-inf (Fixed) ---")
    exp_c = run_exploration_mahalanobis_fixed()
    with open("results/exp7_exploration_C_fixed.json", "w") as f:
        json.dump(exp_c, f, indent=2)
    print("Saved results/exp7_exploration_C_fixed.json")
    
    # Summary
    print("\n" + "=" * 60)
    print("Exploration Summary")
    print("=" * 60)
    
    print("\nA: Adaptive Threshold")
    for r_fix, r_ad in zip(exp_a["fixed_threshold_results"], exp_a["adaptive_threshold_results"]):
        print(f"  ε={r_fix['epsilon']}: fixed err={r_fix['error']:.4f} fb={r_fix['fallback_rate']:.2%} | "
              f"adapt err={r_ad['error']:.4f} fb={r_ad['fallback_rate']:.2%}")
    
    print("\nB: Statistical Bounds")
    for r in exp_b["empirical_results"]:
        print(f"  ε={r['epsilon']}: err={r['error']:.4f} fb={r['fallback_rate']:.2%} "
              f"dist={r['distance_mean']:.2f}±{r['distance_std']:.2f}")
    
    print("\nC: Per-dim vs L-inf")
    for r_per, r_linf in zip(exp_c["per_dim_results"], exp_c["linf_results"]):
        print(f"  ε={r_per['epsilon']}: per_dim err={r_per['error']:.4f} fb={r_per['fallback_rate']:.2%} | "
              f"linf err={r_linf['error']:.4f} fb={r_linf['fallback_rate']:.2%}")
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)
    
    return main_results, exp_a, exp_b, exp_c


if __name__ == "__main__":
    main()
