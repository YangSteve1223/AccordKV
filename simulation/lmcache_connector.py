"""
simulation/lmcache_connector.py — LMCache + ACCORD Integration Prototype (Skeleton)

Integration mode: ACCORD contract selector as LMCache KV connector backend.

This is a SKELETON for the LMCache integration PoC described in:
  docs/lmcache-integration-proposal.md

Key concepts:
- LMCache provides the memory pool (memory abstraction)
- ACCORD provides the contract selector (representation abstraction)
- Combined: memory pool with heterogeneous contract types + query validity

DISCLAIMER: This is a conceptual skeleton. Actual LMCache integration
requires running the real LMCache codebase. All interface definitions
are based on Junchen Jiang's 2026-04-28 blog post ("KV cache is
first-class semantic memory") and may not match LMCache's current API.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import warnings

# === torch 安全导入 + device fallback ===

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None


def _get_safe_device(requested_device: Optional[str] = None) -> str:
    """
    获取可用的计算设备，带 fallback。
    
    Args:
        requested_device: 用户请求的设备 (e.g., "cuda", "cpu")
        
    Returns:
        可用的设备字符串
    """
    if requested_device is not None:
        # 用户明确指定
        if requested_device == "cuda" and not _TORCH_AVAILABLE:
            warnings.warn(
                "CUDA requested but torch not available. Falling back to 'cpu'.",
                UserWarning
            )
            return "cpu"
        if requested_device == "cuda" and _TORCH_AVAILABLE and not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Falling back to 'cpu'.",
                UserWarning
            )
            return "cpu"
        return requested_device
    
    # 用户未指定，自动选择
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    
    warnings.warn(
        "No GPU available. Running on CPU.",
        UserWarning
    )
    return "cpu"


# ============================================================
# ACCORD Contract Types (for LMCache integration)
# ============================================================

class ContractType(Enum):
    """
    ACCORD contract types to extend LMCache's single "full KV" representation.
    
    Each contract type maps to a different memory pool representation:
    - EXACT_LOCAL: Full KV in GPU memory (LMCache default)
    - SKETCH_LOCAL: Coreset-compressed KV in GPU memory
    - REMOTE_EXACT: Full KV on remote server (LMCache server)
    - REHYDRATE: Not in memory, recompute on demand
    - DROP: Discard, no representation
    """
    EXACT_LOCAL = "exact_local"
    SKETCH_LOCAL = "sketch_local"
    REMOTE_EXACT = "remote_exact"
    REHYDRATE = "rehydrate"
    DROP = "drop"


@dataclass
class ACCORDContract:
    """
    ACCORD contract: representation + validity + metadata.
    
    Extends LMCache's "semantic context" with:
    - contract_type: Which representation (exact/sketch/remote/rehydrate/drop)
    - sketch_data: Compressed representation (for SKETCH_LOCAL)
    - query_domain: Statistical validity region
    - validity_threshold: OOD detection threshold
    - calibration_data: For computing fallback mean
    """
    contract_id: str
    contract_type: ContractType
    block_id: int
    
    # Representation data (varies by type)
    keys: Optional[np.ndarray] = None          # For EXACT_LOCAL
    values: Optional[np.ndarray] = None        # For EXACT_LOCAL
    sketch_data: Optional[np.ndarray] = None   # For SKETCH_LOCAL
    remote_handle: Optional[str] = None         # For REMOTE_EXACT
    
    # Query-domain validity
    query_domain_mu: Optional[np.ndarray] = None
    query_domain_sigma: Optional[np.ndarray] = None
    validity_threshold: float = 3.0
    is_valid: bool = True
    
    # Metadata
    calibration_bytes: int = 0
    representation_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    
    def total_bytes(self) -> int:
        """Total memory footprint of this contract."""
        total = self.representation_bytes + self.calibration_bytes
        return total
    
    def accuracy_proxy(self) -> float:
        """Estimate accuracy of this contract's representation."""
        if self.contract_type == ContractType.EXACT_LOCAL:
            return 1.0
        elif self.contract_type == ContractType.SKETCH_LOCAL:
            # Estimate based on compression ratio
            if self.sketch_data is not None and self.keys is not None:
                orig = self.keys.size + self.values.size
                compressed = self.sketch_data.size
                ratio = compressed / max(orig, 1)
                return 1.0 / (1.0 + ratio * 0.75)
            return 0.8
        elif self.contract_type == ContractType.REMOTE_EXACT:
            return 1.0
        elif self.contract_type == ContractType.REHYDRATE:
            return 0.9
        else:
            return 0.0


# ============================================================
# Query-domain validity (from validity_v2.py)
# ============================================================

class QueryDomainValidator:
    """
    Query-domain validity checker for ACCORD-LMCache integration.
    
    Implements OOD detection to decide whether to use the cached
    representation or fallback. This is the key ACCORD contribution
    that LMCache currently lacks.
    """
    
    def __init__(
        self,
        calibration_queries: np.ndarray,
        threshold: float = 3.0,
        method: str = "linf",
    ):
        """
        Args:
            calibration_queries: [n_calib, q_len, d] calibration embeddings
            threshold: Validity threshold in std units
            method: "linf" or "mahalanobis"
        """
        if calibration_queries.ndim == 3:
            self.calib_means = calibration_queries.mean(axis=1)
        else:
            self.calib_means = calibration_queries
            
        self.mu = self.calib_means.mean(axis=0)
        self.sigma = self.calib_means.std(axis=0) + 1e-6
        self.threshold = threshold
        self.method = method
        
        # Covariance for Mahalanobis
        self.cov = np.cov(self.calib_means.T) + 1e-6 * np.eye(self.mu.shape[0])
        try:
            self.cov_inv = np.linalg.inv(self.cov)
        except:
            self.cov_inv = np.eye(self.mu.shape[0])
    
    def is_valid(self, q_repr: np.ndarray, threshold: Optional[float] = None) -> bool:
        """Check if query representation is in-domain."""
        if threshold is None:
            threshold = self.threshold
            
        z = (q_repr - self.mu) / self.sigma
        max_z = np.max(np.abs(z))
        return max_z <= threshold
    
    def distance(self, q_repr: np.ndarray) -> float:
        """Compute validity distance (positive = OOD, negative = in-domain)."""
        z = (q_repr - self.mu) / self.sigma
        return float(np.max(np.abs(z)) - self.threshold)
    
    def fallback_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return fallback key/value (calibration mean)."""
        return self.mu.copy(), self.mu.copy()


# ============================================================
# ACCORD Contract Selector (LMCache integration point)
# ============================================================

class ACCORDContractSelector:
    """
    Selects ACCORD contract type based on query, network state, and policy.
    
    This is the core integration point between LMCache and ACCORD:
    - LMCache manages the memory pool (KV storage/retrieval)
    - ACCORDContractSelector decides which contract type to use
    
    Usage with LMCache:
    1. LMCache receives a query Q
    2. ACCORDContractSelector picks contract type based on Q
    3. LMCache retrieves/fetches the appropriate representation
    4. If OOD, fallback to calibration mean
    """
    
    # Contract selection thresholds (from policy_v2.py)
    HOTNESS_EXACT_LOCAL = 0.7
    HOTNESS_SKETCH_LOCAL = 0.5
    HOTNESS_REMOTE_EXACT = 0.2
    DROP_THRESHOLD = 0.1
    
    def __init__(
        self,
        policy: str = "fixed",
        deadline_class: str = "LOOSE",
        sketch_compression_ratio: float = 0.1,
        calibration_queries: Optional[np.ndarray] = None,
    ):
        """
        Args:
            policy: 'fixed' | 'deadline_aware' | 'streaming'
            deadline_class: For deadline_aware policy
            sketch_compression_ratio: r/n for sketch representation
            calibration_queries: For validity checking
        """
        self.policy = policy
        self.deadline_class = deadline_class
        self.sketch_compression_ratio = sketch_compression_ratio
        
        # Query-domain validity
        if calibration_queries is not None:
            self.validator = QueryDomainValidator(calibration_queries)
        else:
            self.validator = None
    
    def select_contract(
        self,
        block_id: int,
        query: np.ndarray,
        block_features: Dict[str, Any],
        network_state: Dict[str, Any],
        memory_budget: float,
        error_budget: float = 1e-3,
        observed_hotness: float = 0.0,
        phase: str = "decode",
    ) -> ContractType:
        """
        Select the best contract type for a block given current conditions.
        
        Args:
            block_id: Block identifier
            query: Query embedding [q_len, d]
            block_features: Block metadata (importance, clusterability, etc.)
            network_state: Network conditions (bandwidth, RTT)
            memory_budget: Available memory
            error_budget: Acceptable error
            observed_hotness: Observed access frequency [0, 1]
            phase: 'prefill' or 'decode'
            
        Returns:
            Selected ContractType
        """
        # Extract features
        importance = block_features.get("importance", 0.5)
        clusterability = block_features.get("clusterability", 0.5)
        bandwidth_gbps = network_state.get("bandwidth_gbps", 10.0)
        rtt_ms = network_state.get("rtt_ms", 5.0)
        
        # OOD validity check (ACCORD unique contribution)
        if self.validator is not None:
            if query.ndim == 2:
                q_repr = query.mean(axis=0)  # [d]
            else:
                q_repr = query
            is_valid = self.validator.is_valid(q_repr)
            
            if not is_valid:
                # OOD query: use fallback (REHYDRATE or SKETCH_LOCAL)
                # Returning REHYDRATE means recompute from scratch
                return ContractType.REHYDRATE
        
        # === Contract selection logic (from policy_v2.py) ===
        
        if self.policy == "deadline_aware":
            return self._select_deadline_aware(importance, deadline_class=self.deadline_class)
        
        elif self.policy == "streaming":
            return self._select_streaming(
                importance, clusterability, observed_hotness,
                phase, error_budget
            )
        
        else:  # fixed
            return self._select_fixed(importance, clusterability, error_budget)
    
    def _select_fixed(
        self,
        importance: float,
        clusterability: float,
        error_budget: float,
    ) -> ContractType:
        """Fixed policy selection."""
        if importance < self.DROP_THRESHOLD:
            return ContractType.DROP
        if importance > self.HOTNESS_EXACT_LOCAL:
            return ContractType.EXACT_LOCAL
        if clusterability > 0.5:
            estimated_error = block_features.get("estimated_error", error_budget)
            if estimated_error > 0.1 * error_budget:
                return ContractType.SKETCH_LOCAL
        if importance > self.HOTNESS_REMOTE_EXACT:
            return ContractType.REMOTE_EXACT
        return ContractType.REHYDRATE
    
    def _select_deadline_aware(
        self,
        importance: float,
        deadline_class: str,
    ) -> ContractType:
        """Deadline-aware policy selection."""
        pools = {
            "TIGHT": {ContractType.EXACT_LOCAL},
            "MODERATE": {ContractType.EXACT_LOCAL, ContractType.SKETCH_LOCAL},
            "LOOSE": {ContractType.EXACT_LOCAL, ContractType.SKETCH_LOCAL, ContractType.REMOTE_EXACT},
            "LAZY": {ContractType.EXACT_LOCAL, ContractType.SKETCH_LOCAL,
                    ContractType.REMOTE_EXACT, ContractType.REHYDRATE, ContractType.DROP},
        }
        allowed = pools.get(deadline_class, pools["LAZY"])
        
        if importance > 0.7 and ContractType.EXACT_LOCAL in allowed:
            return ContractType.EXACT_LOCAL
        elif importance > 0.5 and ContractType.SKETCH_LOCAL in allowed:
            return ContractType.SKETCH_LOCAL
        elif importance > 0.2 and ContractType.REMOTE_EXACT in allowed:
            return ContractType.REMOTE_EXACT
        elif importance > 0.1 and ContractType.REHYDRATE in allowed:
            return ContractType.REHYDRATE
        return next(iter(allowed))
    
    def _select_streaming(
        self,
        importance: float,
        clusterability: float,
        observed_hotness: float,
        phase: str,
        error_budget: float,
    ) -> ContractType:
        """Streaming/iterative policy selection."""
        if phase == "prefill":
            if importance > 0.8:
                return ContractType.EXACT_LOCAL
            return ContractType.REHYDRATE
        
        upgrade_to_sketch = observed_hotness > 0.4 or importance > 0.5
        upgrade_to_exact = observed_hotness > 0.7 or importance > 0.7
        
        if upgrade_to_exact:
            return ContractType.EXACT_LOCAL
        elif upgrade_to_sketch:
            return ContractType.SKETCH_LOCAL
        return ContractType.REHYDRATE


# ============================================================
# Mock LMCache integration
# ============================================================

class LMCacheACCORDConnector:
    """
    Mock LMCache KV connector with ACCORD contract support.
    
    This demonstrates how ACCORD would integrate with LMCache:
    - LMCache handles: memory pool storage, KV retrieval, RPC
    - ACCORD handles: contract selection, validity checking, fallback
    
    The actual implementation would subclass LMCache's VLLMCacheConnector
    or similar base class.
    """
    
    def __init__(
        self,
        model_name: str,
        contract_selector: ACCORDContractSelector,
        remote_server_url: Optional[str] = None,
        local_device: Optional[str] = None,
    ):
        """
        Args:
            model_name: Model name for LMCache
            contract_selector: ACCORD contract selector
            remote_server_url: LMCache server URL for remote contracts
            local_device: Device for local storage (None=自动检测)
        """
        self.model_name = model_name
        self.contract_selector = contract_selector
        self.remote_server_url = remote_server_url
        self.local_device = _get_safe_device(local_device)
        
        # Memory pool (what LMCache manages)
        self._memory_pool: Dict[int, ACCORDContract] = {}
        
        # Calibration data (for validity + fallback)
        self._calibration_queries: Optional[np.ndarray] = None
        
        # Statistics
        self.stats = {
            "total_lookups": 0,
            "hits": 0,
            "misses": 0,
            "ood_detections": 0,
            "fallback_uses": 0,
            "bytes_transferred": 0,
        }
    
    def load(
        self,
        block_id: int,
        kv_data: Dict[str, np.ndarray],
        calibration_queries: Optional[np.ndarray] = None,
        contract_type: ContractType = ContractType.EXACT_LOCAL,
        sketch_data: Optional[np.ndarray] = None,
    ) -> None:
        """
        Load a block into memory pool.
        
        This is called by LLM engine during prefill to store KV data
        with ACCORD contract metadata.
        """
        keys = kv_data.get("keys")
        values = kv_data.get("values")
        
        contract = ACCORDContract(
            contract_id=f"{self.model_name}_block_{block_id}",
            contract_type=contract_type,
            block_id=block_id,
            keys=keys,
            values=values,
            sketch_data=sketch_data,
            representation_bytes=keys.nbytes + values.nbytes if keys is not None else 0,
            calibration_bytes=calibration_queries.nbytes if calibration_queries is not None else 0,
        )
        
        self._memory_pool[block_id] = contract
        
        if calibration_queries is not None:
            self._calibration_queries = calibration_queries
    
    def lookup(
        self,
        block_id: int,
        query: np.ndarray,
        network_state: Dict[str, Any],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Lookup a block given a query.
        
        This is the ACCORD-LMCache integration hot path:
        1. ACCORDContractSelector picks contract type based on query
        2. OOD validity check (ACCORD unique contribution)
        3. Return appropriate representation or fallback
        
        Returns:
            (keys, values, metadata) or (None, None, metadata) if not found
        """
        self.stats["total_lookups"] += 1
        
        if block_id not in self._memory_pool:
            self.stats["misses"] += 1
            return None, None, {"hit": False, "reason": "not_in_pool"}
        
        contract = self._memory_pool[block_id]
        contract.hit_count += 1
        
        # Get block features for contract selection
        block_features = {
            "importance": contract.hit_count / max(self.stats["total_lookups"], 1),
            "clusterability": 0.5,  # Would be computed from attention patterns
            "estimated_error": 1e-3,
        }
        
        # ACCORD contract selection
        selected_type = self.contract_selector.select_contract(
            block_id=block_id,
            query=query,
            block_features=block_features,
            network_state=network_state,
            memory_budget=1e9,
        )
        
        # OOD validity check (the key ACCORD contribution)
        if self.contract_selector.validator is not None:
            q_repr = query.mean(axis=0) if query.ndim == 2 else query
            is_valid = self.contract_selector.validator.is_valid(q_repr)
            
            if not is_valid:
                self.stats["ood_detections"] += 1
                dist = self.contract_selector.validator.distance(q_repr)
                
                # OOD: return fallback stats
                fallback_keys, fallback_values = self.contract_selector.validator.fallback_stats()
                self.stats["fallback_uses"] += 1
                
                return (
                    fallback_keys[np.newaxis],  # Add seq dim
                    fallback_values[np.newaxis],
                    {
                        "hit": True,
                        "contract_type": "fallback_calib_mean",
                        "ood_distance": dist,
                        "is_ood": True,
                    }
                )
        
        # In-domain: return representation based on selected contract type
        if selected_type == ContractType.EXACT_LOCAL:
            self.stats["hits"] += 1
            return contract.keys, contract.values, {"hit": True, "contract_type": "exact_local"}
        
        elif selected_type == ContractType.SKETCH_LOCAL:
            # Would need sketch decompression (approximation)
            self.stats["hits"] += 1
            # For sketch, we'd decompress sketch_data to approximate K/V
            return contract.sketch_data, contract.sketch_data, {"hit": True, "contract_type": "sketch_local"}
        
        elif selected_type == ContractType.REMOTE_EXACT:
            # Would fetch from LMCache remote server
            self.stats["bytes_transferred"] += contract.remote_handle_size() if hasattr(contract, 'remote_handle_size') else 64
            return contract.keys, contract.values, {"hit": True, "contract_type": "remote_exact"}
        
        elif selected_type == ContractType.REHYDRATE:
            return None, None, {"hit": False, "reason": "rehydrate"}
        
        else:  # DROP
            return None, None, {"hit": False, "reason": "dropped"}
    
    def get_stats(self) -> Dict[str, Any]:
        """Return connector statistics."""
        total = self.stats["total_lookups"]
        return {
            **self.stats,
            "hit_rate": self.stats["hits"] / max(total, 1),
            "ood_rate": self.stats["ood_detections"] / max(total, 1),
            "fallback_rate": self.stats["fallback_uses"] / max(total, 1),
            "pool_size": len(self._memory_pool),
        }


# ============================================================
# Test / Demo
# ============================================================

def demo_lmcache_accord():
    """Demonstrate ACCORD-LMCache integration."""
    print("=== LMCache + ACCORD Integration Demo ===")
    
    # Mock calibration queries
    np.random.seed(42)
    calib = np.random.randn(100, 16, 64)
    
    # Create ACCORD contract selector
    selector = ACCORDContractSelector(
        policy="fixed",
        calibration_queries=calib,
    )
    
    # Create LMCache connector with ACCORD
    connector = LMCacheACCORDConnector(
        model_name="meta-llama/Llama-3-8b",
        contract_selector=selector,
    )
    
    # Load a block (simulate prefill)
    K = np.random.randn(512, 64).astype(np.float32)
    V = np.random.randn(512, 64).astype(np.float32)
    connector.load(block_id=0, kv_data={"keys": K, "values": V})
    
    # Query (in-domain)
    Q_in = np.random.randn(16, 64).astype(np.float32) * 0.5
    keys, vals, meta = connector.lookup(0, Q_in, network_state={"bandwidth_gbps": 10.0, "rtt_ms": 5.0})
    print(f"\nIn-domain query:")
    print(f"  Hit: {meta['hit']}, Type: {meta.get('contract_type', 'unknown')}")
    
    # Query (OOD — far from calibration distribution)
    Q_ood = np.random.randn(16, 64).astype(np.float32) * 5.0
    keys2, vals2, meta2 = connector.lookup(0, Q_ood, network_state={"bandwidth_gbps": 10.0, "rtt_ms": 5.0})
    print(f"\nOOD query:")
    print(f"  Hit: {meta2['hit']}, Type: {meta2.get('contract_type', 'unknown')}")
    print(f"  OOD distance: {meta2.get('ood_distance', 0):.3f}")
    
    # Statistics
    print(f"\nConnector stats: {connector.get_stats()}")


if __name__ == "__main__":
    demo_lmcache_accord()
