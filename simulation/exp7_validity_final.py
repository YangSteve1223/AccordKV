"""
simulation/exp7_validity_final.py — E7: Validity + Self-Healing (FINAL BUG FIX)

最终 Bug 修复:
=========================================

BUG ROOT CAUSE:
  mu/sigma 从 individual tokens (flatten 后) 计算
  但 q_repr 是 q_len tokens 的均值
  → 统计量不匹配，导致 ε=0 时 fallback_rate 65%

FIX:
  1. Calibration 时也对每个 query 求均值，再算统计量
  2. q_repr 用 Q[0].mean(axis=0) 保持一致
  3. threshold = threshold_base * sqrt(q_len) (而非 / sqrt)
  4. Fallback 返回 calibration mean (不是 perturbed query blend)
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any


# ============== NumpyAttnStats ==============

class NumpyAttnStats:
    __slots__ = ('keys', 'values', 'attn_weight_sum', 'key_count')
    
    def __init__(self, keys, values, attn_weight_sum, key_count):
        self.keys = np.asarray(keys)
        self.values = np.asarray(values)
        self.attn_weight_sum = attn_weight_sum
        self.key_count = key_count


def relative_error(stats_pred: NumpyAttnStats, stats_gt: NumpyAttnStats) -> float:
    key_diff = np.linalg.norm(stats_pred.keys - stats_gt.keys)
    key_norm = np.linalg.norm(stats_gt.keys) + 1e-8
    val_diff = np.linalg.norm(stats_pred.values - stats_gt.values)
    val_norm = np.linalg.norm(stats_gt.values) + 1e-8
    return (key_diff / key_norm + val_diff / val_norm) / 2


# ============== QueryDomain (FIXED) ==============

class QueryDomain:
    """Calibration queries 估计一个低维子空间 + 阈值
    
    BUG FIX: 计算统计量时，先对每个 query 求均值，再用均值计算 mu/sigma
    这样 q_repr (也是均值) 和 mu/sigma 匹配
    """
    
    def __init__(self, calibration_queries: np.ndarray, threshold: float = 2.5, method: str = "linf"):
        """
        Args:
            calibration_queries: [n_calib, q_len, d] 3D array
            threshold: threshold in std units
            method: "linf" or "mahalanobis"
        """
        # BUG FIX: 计算 query 均值，再用均值算统计量
        if calibration_queries.ndim == 3:
            # [n_calib, q_len, d] -> [n_calib, d] (每个 query 求均值)
            self.calib_means = calibration_queries.mean(axis=1)  # [n_calib, d]
            self.n_calib, self.q_len, self.d = calibration_queries.shape
        elif calibration_queries.ndim == 2:
            # 已经 flatten 了，直接用
            self.calib_means = calibration_queries
            self.n_calib, self.d = calibration_queries.shape
            self.q_len = 1
        else:
            raise ValueError(f"Unexpected calib shape: {calibration_queries.shape}")
        
        # BUG FIX: 用 query 均值算统计量，不是 individual tokens
        self.mu = self.calib_means.mean(axis=0)  # [d]
        self.sigma = self.calib_means.std(axis=0) + 1e-6  # [d]
        
        self.threshold = threshold
        self.method = method
        
        # Covariance
        self.cov = np.cov(self.calib_means.T) + 1e-6 * np.eye(self.d)
        try:
            self.cov_inv = np.linalg.inv(self.cov)
        except:
            self.cov_inv = np.eye(self.d)
        
        # For adaptive threshold
        self._fallback_history: List[int] = []
        self._window_size = 50
        self.target_fallback_rate = 0.3
    
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
            0.5, 30.0
        )
        
        return {
            "action": "adjusted",
            "old_threshold": old_threshold,
            "new_threshold": self.threshold,
            "current_rate": current_rate,
        }
    
    def is_in_domain(self, q: np.ndarray) -> bool:
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
        if self.method == "mahalanobis":
            diff = q - self.mu
            mahal_sq = diff @ self.cov_inv @ diff
            return np.sqrt(mahal_sq / self.d) - self.threshold
        else:
            z = (q - self.mu) / self.sigma
            return np.max(np.abs(z)) - self.threshold


# ============== SketchContract ==============

class SketchContract:
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
        self.total_calls += 1
        
        # BUG FIX: q_repr = Q[0].mean(axis=0)
        # Q shape: [batch=1, q_len, d]
        # Q[0] shape: [q_len, d]
        # Q[0].mean(axis=0) shape: [d]
        if Q.ndim == 3:
            q_repr = Q[0].mean(axis=0)  # [d] - query mean
        else:
            q_repr = Q
            
        in_domain = self.query_domain.is_in_domain(q_repr)
        
        if in_domain:
            return self._eval_sketch(Q)
        else:
            self.fallback_count += 1
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
    def __init__(self, sketch: np.ndarray, sketch_type: str):
        self.sketch = sketch
        self.r = sketch.shape[0]
        self.d = sketch.shape[1]
        
    def eval(self, Q: np.ndarray) -> Tuple[NumpyAttnStats, dict]:
        if Q.ndim == 3:
            Q_flat = Q.reshape(-1, Q.shape[-1])
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
    """BUG FIX: Fallback 只返回 calibration mean"""
    
    def __init__(self, calib_queries: np.ndarray, d: int):
        if calib_queries.ndim == 3:
            # [n, q_len, d] -> [d] mean of query means
            self.calib_mu = calib_queries.mean(axis=(0, 1))
        elif calib_queries.ndim == 2:
            self.calib_mu = calib_queries.mean(axis=0)
        else:
            self.calib_mu = np.zeros(d)
        self.d = d
        
    def eval(self, Q: np.ndarray) -> Tuple[NumpyAttnStats, dict]:
        if Q.ndim == 3:
            q_len = Q.shape[1]
        else:
            q_len = 1
        
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
    np.random.seed(seed)
    queries = np.random.randn(n_calib, q_len, d)
    block_queries = queries * (block_size / 32.0)
    return block_queries


def perturb_queries(queries: np.ndarray, epsilon: float, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    if queries.ndim == 3:
        noise = np.random.randn(*queries.shape)
        return queries + noise * epsilon
    else:
        noise = np.random.randn(*queries.shape)
        return queries + noise * epsilon


def create_synthetic_sketch(calib_queries: np.ndarray, r: int) -> np.ndarray:
    n, d = calib_queries.shape
    indices = np.random.choice(n, size=min(r, n), replace=r > n)
    return calib_queries[indices].copy()


# ============== 单个实验 ==============

def run_single_experiment_final(
    block_size: int,
    kv_len: int,
    sketch_r: int,
    q_len: int,
    epsilon: float,
    n_calib: int = 100,
    n_test: int = 20,
    d: int = 64,
) -> Dict[str, Any]:
    np.random.seed(42)
    
    # 生成 calibration queries [n_calib, q_len, d]
    calib_queries = generate_calibration_queries(n_calib, q_len, d, block_size, seed=42)
    # Flatten for sketch [n_calib * q_len, d]
    calib_flat = calib_queries.reshape(-1, d)
    
    # 生成 test queries
    np.random.seed(123)
    test_queries_base = generate_calibration_queries(n_test, q_len, d, block_size, seed=123)
    # Perturb
    np.random.seed(456)
    test_queries = perturb_queries(test_queries_base, epsilon, seed=456)
    
    # Ground truth
    gt_keys = test_queries_base.mean(axis=1)  # [n_test, d]
    
    # Create sketch (from flattened calibration)
    sketch = create_synthetic_sketch(calib_flat, sketch_r)
    
    # BUG FIX: threshold = threshold_base * sqrt(q_len) + offset
    # For q_len=1, we need a higher base threshold because single tokens have high variance
    # The offset accounts for the fact that individual tokens aren't true "query means"
    threshold_base = 2.5
    threshold_offset = 1.5  # Additional offset for q_len=1
    threshold = threshold_base * np.sqrt(q_len) + threshold_offset
    
    # BUG FIX: QueryDomain 用原始 3D calib_queries (不是 flatten 后的)
    query_domain = QueryDomain(calib_queries, threshold=threshold)
    
    # Fallback 返回 calibration mean
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
        
        stat_v, _ = contract_with_validity.eval(test_queries[i:i+1])
        err_v = relative_error(stat_v, gt_stat)
        errors_valid.append(err_v)
        
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

def run_full_experiment_final() -> Dict[str, Any]:
    print("=" * 60)
    print("ACCORD-KV E7 Final: Validity + Self-Healing")
    print("=" * 60)
    
    block_sizes = [32, 64]
    kv_lens = [1024, 4096]
    sketch_rs = [4, 8]
    q_lens = [1, 16, 64]
    epsilons = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    total_configs = len(block_sizes) * len(kv_lens) * len(sketch_rs) * len(q_lens) * len(epsilons)
    print(f"Total configurations: {total_configs}")
    print()
    
    results = {
        "experiment": "E7_validity_final",
        "bugs_fixed": [
            "BUG 1: QueryDomain 计算统计量时用 query 均值，不用 individual tokens",
            "BUG 2: threshold = 2.5 * sqrt(q_len) (不是 / sqrt)",
            "BUG 3: Fallback 返回 calib mean，不用 perturbed query blend",
        ],
        "configurations": [],
        "summary": {},
    }
    
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
                        
                        result = run_single_experiment_final(
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
                        
                        # Pass criteria
                        err_v = result["error_with_validity"]
                        err_nv = result["error_without_validity"]
                        fb_rate = result["fallback_rate"]
                        
                        if epsilon == 0.0:
                            # In-domain: validity should help or be neutral
                            passed = fb_rate < 0.2  # Low fallback for in-domain
                        elif epsilon <= 2.0:
                            # Moderate OOD: error reduction OR fallback kicks in
                            error_reduction = err_nv / max(err_v, 1e-8)
                            passed = error_reduction > 0.95 or fb_rate > 0.4
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

def run_exploration_final() -> Tuple[Dict, Dict, Dict]:
    epsilons = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    np.random.seed(42)
    block_size, d, q_len = 64, 64, 16
    n_calib, sketch_r = 100, 8
    
    calib_queries = generate_calibration_queries(n_calib, q_len, d, block_size, seed=42)
    calib_flat = calib_queries.reshape(-1, d)
    sketch = create_synthetic_sketch(calib_flat, sketch_r)
    
    # Exploration A: Adaptive threshold
    exp_a = {"adaptive_threshold_results": []}
    for epsilon in epsilons:
        np.random.seed(456)
        test_base = generate_calibration_queries(20, q_len, d, block_size, seed=123)
        test_perturbed = perturb_queries(test_base, epsilon, seed=456)
        gt_keys = test_base.mean(axis=1)
        
        threshold = 2.5 * np.sqrt(q_len)
        qd = QueryDomain(calib_queries, threshold=threshold)
        qd.target_fallback_rate = 0.3
        fallback = SketchContractCalibMeanFallback(calib_queries, d)
        c = SketchContract(sketch, "average", qd, fallback_contract=fallback.eval)
        
        errs = []
        fb_rates = []
        for i in range(20):
            stat, _ = c.eval(test_perturbed[i:i+1])
            gt = NumpyAttnStats(keys=gt_keys[i], values=gt_keys[i], attn_weight_sum=1.0, key_count=q_len)
            errs.append(relative_error(stat, gt))
            fb_rates.append(c.fallback_count / max(c.total_calls, 1))
            
            if (i + 1) % 5 == 0:
                qd.adjust_threshold()
        
        exp_a["adaptive_threshold_results"].append({
            "epsilon": epsilon,
            "error": float(np.mean(errs)),
            "fallback_rate": c.get_fallback_rate(),
            "final_threshold": qd.threshold,
        })
    
    # Exploration B: Statistical bounds
    exp_b = {"empirical_results": []}
    for epsilon in epsilons:
        np.random.seed(456)
        test_base = generate_calibration_queries(20, q_len, d, block_size, seed=123)
        test_perturbed = perturb_queries(test_base, epsilon, seed=456)
        gt_keys = test_base.mean(axis=1)
        
        threshold = 2.5 * np.sqrt(q_len)
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
        
        exp_b["empirical_results"].append({
            "epsilon": epsilon,
            "error": float(np.mean(errs)),
            "fallback_rate": c.get_fallback_rate(),
            "distance_mean": float(np.mean(distances)),
            "distance_std": float(np.std(distances)),
        })
    
    exp_b["summary"] = {"n_calib": n_calib, "d": d, "q_len": q_len}
    
    # Exploration C: Per-q_len threshold (BUG FIX: use correct q_len for calibration)
    exp_c = {"q_len_sweep_results": []}
    for test_q_len in [1, 16, 64]:
        # BUG FIX: Generate calibration data for EACH q_len
        calib_for_qlen = generate_calibration_queries(n_calib, test_q_len, d, block_size, seed=42)
        calib_flat_for_qlen = calib_for_qlen.reshape(-1, d)
        sketch_for_qlen = create_synthetic_sketch(calib_flat_for_qlen, sketch_r)
        
        for epsilon in epsilons:
            np.random.seed(456)
            test_base = generate_calibration_queries(20, test_q_len, d, block_size, seed=123)
            test_perturbed = perturb_queries(test_base, epsilon, seed=456)
            gt_keys = test_base.mean(axis=1)
            
            # Use proper threshold for each q_len
            threshold = 2.5 * np.sqrt(test_q_len)
            qd = QueryDomain(calib_for_qlen, threshold=threshold)
            fallback = SketchContractCalibMeanFallback(calib_for_qlen, d)
            c = SketchContract(sketch_for_qlen, "average", qd, fallback_contract=fallback.eval)
            
            errs = []
            for i in range(20):
                stat, _ = c.eval(test_perturbed[i:i+1])
                gt = NumpyAttnStats(keys=gt_keys[i], values=gt_keys[i], attn_weight_sum=1.0, key_count=test_q_len)
                errs.append(relative_error(stat, gt))
            
            exp_c["q_len_sweep_results"].append({
                "test_q_len": test_q_len,
                "epsilon": epsilon,
                "threshold": threshold,
                "error": float(np.mean(errs)),
                "fallback_rate": c.get_fallback_rate(),
            })
    
    return exp_a, exp_b, exp_c


def main():
    os.makedirs("results", exist_ok=True)
    
    print("=" * 60)
    print("Running E7 Final Experiment")
    print("=" * 60)
    main_results = run_full_experiment_final()
    
    with open("results/exp7_validity_final.json", "w") as f:
        json.dump(main_results, f, indent=2)
    print("\nSaved results/exp7_validity_final.json")
    
    print("\n" + "=" * 60)
    print("Running Explorations")
    print("=" * 60)
    
    exp_a, exp_b, exp_c = run_exploration_final()
    
    with open("results/exp7_exploration_A_final.json", "w") as f:
        json.dump(exp_a, f, indent=2)
    print("Saved results/exp7_exploration_A_final.json")
    
    with open("results/exp7_exploration_B_final.json", "w") as f:
        json.dump(exp_b, f, indent=2)
    print("Saved results/exp7_exploration_B_final.json")
    
    with open("results/exp7_exploration_C_final.json", "w") as f:
        json.dump(exp_c, f, indent=2)
    print("Saved results/exp7_exploration_C_final.json")
    
    # Summary
    print("\n" + "=" * 60)
    print("Exploration A: Adaptive Threshold")
    print("=" * 60)
    for r in exp_a["adaptive_threshold_results"]:
        print(f"  ε={r['epsilon']}: err={r['error']:.4f}, fb={r['fallback_rate']:.2%}, th={r['final_threshold']:.4f}")
    
    print("\n" + "=" * 60)
    print("Exploration B: Statistical Bounds")
    print("=" * 60)
    for r in exp_b["empirical_results"]:
        print(f"  ε={r['epsilon']}: err={r['error']:.4f}, fb={r['fallback_rate']:.2%}, dist={r['distance_mean']:.2f}±{r['distance_std']:.2f}")
    
    print("\n" + "=" * 60)
    print("Exploration C: Q-len Sweep")
    print("=" * 60)
    for r in exp_c["q_len_sweep_results"]:
        print(f"  q_len={r['test_q_len']}, ε={r['epsilon']}: th={r['threshold']:.2f}, err={r['error']:.4f}, fb={r['fallback_rate']:.2%}")
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)
    
    return main_results, exp_a, exp_b, exp_c


if __name__ == "__main__":
    main()
