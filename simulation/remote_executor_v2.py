"""
simulation/remote_executor_v2.py — ACCORD-KV Remote Executor (Fixed v2)

Changes from v1 (review_to_delete/v1_subagent_output/remote_executor.py):
==============================================================
Issue 1: Random seed based on Q.tobytes() is unstable
  - Old: hash(Q.tobytes()[:8]) % 2**32 — float precision differences cause different bytes
  - FIX: hash(Q.tobytes()) + fixed salt, or hashlib.md5(Q.data.tobytes()).hexdigest()

Issue 2: Latency unit confusion (ms vs us)
  - Old: _recompute returns us converted to ms inconsistently
  - FIX: All latency in ms, with clear unit annotations in comments

Issue 3: HybridRemoteExecutor.execute_batch only handles REMOTE_EXACT and REHYDRATE
  - Old: EXACT_LOCAL, SKETCH_LOCAL, DROP branches missing
  - FIX: Added all 5 contract type branches

Issue 4: Class-level _cache shared across instances (multi-tenant bug)
  - Old: _cache = {} at class level, shared by all RemoteExactContract instances
  - FIX: Instance-level _query_cache (already present, but ensure isolation)
"""

import numpy as np
import hashlib
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass


# ============================================================
# Module-level constants
# ============================================================

DEFAULT_RTT_MS: float = 5.0          # Default round-trip time in milliseconds
DEFAULT_JITTER_MS: float = 1.0        # Maximum jitter in milliseconds
DEFAULT_BANDWIDTH_GBPS: float = 10.0  # Network bandwidth in Gbps
RECOMPUTE_COST_US_PER_TOKEN: float = 50.0  # Microseconds per token
CACHE_LOOKUP_OVERHEAD_MS: float = 0.1  # Cache hit lookup overhead in ms


# ============================================================
# Data types
# ============================================================

CacheKey = str
QueryKey = str  # Literal['EXACT_LOCAL', 'SKETCH_LOCAL', 'REMOTE_EXACT', 'REHYDRATE', 'DROP']
ContractType = str


# ============================================================
# RemoteStats
# ============================================================

@dataclass
class RemoteStats:
    """Statistics from remote contract execution."""
    attn_stats: Optional[np.ndarray]
    latency_ms: float        # All latencies in milliseconds
    bytes_transferred: int
    cache_hit: bool = False


# ============================================================
# RemoteExactContract (FIXED Issues 1, 2, 4)
# ============================================================

class RemoteExactContract:
    """
    Simulates a remote KV cache server.

    Provides exact KV retrieval with realistic network latency:
    - RPC latency = bytes / bandwidth + RTT/2 (all in ms)
    - Optional jitter for realistic simulation
    - Cache hit optimization (same query returns cached stats)

    FIXED:
    - Issue 1: Stable cache key using hashlib.md5 on raw data bytes
    - Issue 2: All latency in ms with explicit comments
    - Issue 4: Instance-level _query_cache (not class-level)
    """

    # Class-level cache statistics (shared read-only stats, not query cache)
    _total_cache_hits: int = 0
    _total_cache_misses: int = 0

    def __init__(
        self,
        K: np.ndarray,
        V: np.ndarray,
        RTT_ms: float = DEFAULT_RTT_MS,
        jitter_ms: float = DEFAULT_JITTER_MS,
        bandwidth_gbps: float = DEFAULT_BANDWIDTH_GBPS,
        cache_enabled: bool = True,
    ):
        """
        Initialize remote contract.

        Args:
            K: Key tensor (shape: [seq_len, d] or [H, seq_len, d])
            V: Value tensor (same shape as K)
            RTT_ms: Round-trip time in milliseconds
            jitter_ms: Maximum jitter to add to latency (milliseconds)
            bandwidth_gbps: Network bandwidth in Gbps
            cache_enabled: Whether to cache query results
        """
        self.K = K
        self.V = V
        self.RTT_ms = RTT_ms
        self.jitter_ms = jitter_ms
        self.bandwidth_gbps = bandwidth_gbps

        # Handle size (metadata for RPC) in bytes
        self.handle_bytes: int = 64

        # KV data size estimation (bytes)
        if K is not None:
            self.kv_bytes: int = K.nbytes + V.nbytes
        else:
            self.kv_bytes = 0

        # Per-instance query cache (FIXED Issue 4: not class-level)
        self.cache_enabled = cache_enabled
        self._query_cache: Dict[CacheKey, Tuple[np.ndarray, float]] = {}
        self._instance_cache_hits: int = 0
        self._instance_cache_misses: int = 0

    @property
    def seq_len(self) -> int:
        """Get sequence length from stored KV."""
        if self.K is None:
            return 0
        if len(self.K.shape) == 2:
            return self.K.shape[0]
        elif len(self.K.shape) == 3:
            return self.K.shape[1]
        return 0

    @property
    def d(self) -> int:
        """Get dimension from stored KV."""
        if self.K is None:
            return 0
        return self.K.shape[-1]

    @property
    def H(self) -> int:
        """Get number of heads (if multi-head)."""
        if self.K is None:
            return 0
        if len(self.K.shape) == 3:
            return self.K.shape[0]
        return 1

    def bytes_on_wire(self) -> int:
        """
        Return bytes transferred per RPC.

        For remote exact, we only transfer the handle (64 bytes).
        Actual KV stays on server.
        """
        return self.handle_bytes

    def _stable_cache_key(self, Q: np.ndarray) -> CacheKey:
        """
        Generate a stable cache key for query Q.

        FIXED Issue 1: Use hashlib.md5 on raw data bytes.
        This is stable across float formatting changes.

        Args:
            Q: Query tensor

        Returns:
            Cache key string
        """
        # FIXED: Use md5 hash of raw float32 bytes for stability
        Q_bytes = np.ascontiguousarray(Q, dtype=np.float32).data.tobytes()
        md5_hash = hashlib.md5(Q_bytes).hexdigest()
        shape_str = "_".join(str(s) for s in Q.shape)
        return f"Q_{md5_hash}_{shape_str}"

    def _compute_attention(self, Q: np.ndarray) -> np.ndarray:
        """
        Compute attention stats locally (simulating server computation).

        FIXED Issue 1: Use stable seed derived from query content.

        Args:
            Q: Query tensor (shape: [q_len, d] or [H, q_len, d])

        Returns:
            Attention statistics
        """
        if self.K is None or self.V is None:
            h = self.H
            q_len = Q.shape[-2] if len(Q.shape) == 3 else Q.shape[-2]
            d = self.d
            return np.zeros((h, q_len, d), dtype=np.float32)

        H, seq_len, d = self.H, self.seq_len, self.d

        if len(Q.shape) == 2:
            Q = Q[np.newaxis, :, :]

        q_len = Q.shape[1]

        # FIXED Issue 1: Stable seed from query content
        Q_bytes = np.ascontiguousarray(Q, dtype=np.float32).data.tobytes()
        seed = int(hashlib.md5(Q_bytes).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed)

        stats = rng.randn(H, q_len, d).astype(np.float32) * 0.1 + 0.5

        return stats

    def eval(self, Q: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate remote contract for query Q.

        Simulates fetching attention stats from remote KV cache.

        FIXED Issue 2: All latency in milliseconds.

        Args:
            Q: Query tensor (shape: [q_len, d] or [H, q_len, d])

        Returns:
            Tuple of (attention_stats, latency_ms)
        """
        if len(Q.shape) == 2:
            q_len = Q.shape[0]
        else:
            q_len = Q.shape[1]

        # Check cache (FIXED Issue 4: instance-level cache)
        cache_key = self._stable_cache_key(Q)
        if self.cache_enabled and cache_key in self._query_cache:
            self._instance_cache_hits += 1
            RemoteExactContract._total_cache_hits += 1
            cached_stats, cached_latency = self._query_cache[cache_key]
            return cached_stats, cached_latency + CACHE_LOOKUP_OVERHEAD_MS

        # FIXED Issue 2: Compute transfer latency in milliseconds
        bytes_tx = self.bytes_on_wire()

        # T_transfer_ms = bytes / (bandwidth_gbps * 1e9) * 1000
        T_transfer_ms: float = (bytes_tx * 8) / (self.bandwidth_gbps * 1e9) * 1000

        # Add jitter for realism (milliseconds)
        if self.jitter_ms > 0:
            jitter = np.random.uniform(-self.jitter_ms, self.jitter_ms)
        else:
            jitter = 0.0

        # Total latency: transfer + one-way RTT + jitter (milliseconds)
        T_latency_ms: float = T_transfer_ms + self.RTT_ms / 2 + jitter
        T_latency_ms = max(0.1, T_latency_ms)

        # Compute attention (simulating server work)
        stats = self._compute_attention(Q)

        # Cache result (FIXED Issue 4: instance-level)
        if self.cache_enabled:
            self._query_cache[cache_key] = (stats, T_latency_ms)

        self._instance_cache_misses += 1
        RemoteExactContract._total_cache_misses += 1

        return stats, T_latency_ms

    def eval_batch(self, Q_batch: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        """
        Evaluate for a batch of queries.

        Args:
            Q_batch: List of query tensors

        Returns:
            List of (stats, latency_ms) tuples
        """
        return [(stats, latency) for stats, latency in
                (self.eval(Q) for Q in Q_batch)]

    def reset_cache(self) -> None:
        """Reset instance-level cache."""
        self._query_cache.clear()
        self._instance_cache_hits = 0
        self._instance_cache_misses = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get instance-level cache statistics."""
        total = self._instance_cache_hits + self._instance_cache_misses
        hit_rate = self._instance_cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._instance_cache_hits,
            "misses": self._instance_cache_misses,
            "total": total,
            "hit_rate": hit_rate,
            "cache_size": len(self._query_cache),
        }

    @classmethod
    def get_global_cache_stats(cls) -> Dict[str, Any]:
        """Get class-level aggregate cache statistics."""
        total = cls._total_cache_hits + cls._total_cache_misses
        hit_rate = cls._total_cache_hits / total if total > 0 else 0.0
        return {
            "hits": cls._total_cache_hits,
            "misses": cls._total_cache_misses,
            "total": total,
            "hit_rate": hit_rate,
        }


# ============================================================
# HybridRemoteExecutor (FIXED Issue 3)
# ============================================================

class HybridRemoteExecutor:
    """
    Executes mixed contract types in a single request.

    Supports:
    - Parallel remote fetches
    - Batched remote requests
    - Fallback to recompute
    - EXACT_LOCAL, SKETCH_LOCAL, REMOTE_EXACT, REHYDRATE, DROP

    FIXED Issue 3: Added branches for EXACT_LOCAL, SKETCH_LOCAL, DROP.
    """

    def __init__(
        self,
        remote_contracts: Dict[int, RemoteExactContract],
        local_exact_contracts: Optional[Dict[int, "LocalExactContract"]] = None,
        local_sketch_contracts: Optional[Dict[int, "SketchLocalContract"]] = None,
        recompute_cost_per_token_us: float = RECOMPUTE_COST_US_PER_TOKEN,
    ):
        """
        Initialize hybrid executor.

        Args:
            remote_contracts: Dict of block_id -> RemoteExactContract
            local_exact_contracts: Dict of block_id -> LocalExactContract
            local_sketch_contracts: Dict of block_id -> SketchLocalContract
            recompute_cost_per_token_us: Cost to recompute a single token (microseconds)
        """
        self.remote_contracts: Dict[int, RemoteExactContract] = remote_contracts
        self.local_exact_contracts: Dict[int, Any] = local_exact_contracts or {}
        self.local_sketch_contracts: Dict[int, Any] = local_sketch_contracts or {}
        self.recompute_cost_per_token_us = recompute_cost_per_token_us

    def execute_batch(
        self,
        block_ids: List[int],
        Q: np.ndarray,
        contracts: List[ContractType],
    ) -> Tuple[Dict[int, np.ndarray], float]:
        """
        Execute batch of contracts.

        FIXED Issue 3: Now handles all 5 contract types:
        - EXACT_LOCAL: Use local exact contract
        - SKETCH_LOCAL: Use local sketch contract
        - REMOTE_EXACT: Fetch from remote
        - REHYDRATE: Recompute from scratch
        - DROP: Return zeros

        FIXED Issue 2: All latency in milliseconds.

        Args:
            block_ids: List of block IDs
            Q: Query tensor
            contracts: List of contract types for each block

        Returns:
            Tuple of (block_id -> stats dict, total_latency_ms)
        """
        results: Dict[int, np.ndarray] = {}
        total_latency_ms: float = 0.0

        for block_id, contract in zip(block_ids, contracts):
            if contract == "EXACT_LOCAL":
                # FIXED Issue 3: EXACT_LOCAL branch
                if block_id in self.local_exact_contracts:
                    stats, latency = self.local_exact_contracts[block_id].eval(Q)
                    results[block_id] = stats
                    total_latency_ms += latency
                else:
                    stats = self._recompute(block_id, Q)
                    latency = self._estimate_recompute_latency_ms(block_id)
                    results[block_id] = stats
                    total_latency_ms += latency

            elif contract == "SKETCH_LOCAL":
                # FIXED Issue 3: SKETCH_LOCAL branch
                if block_id in self.local_sketch_contracts:
                    stats, latency = self.local_sketch_contracts[block_id].eval(Q)
                    results[block_id] = stats
                    total_latency_ms += latency
                else:
                    stats = self._recompute(block_id, Q)
                    latency = self._estimate_recompute_latency_ms(block_id)
                    results[block_id] = stats
                    total_latency_ms += latency

            elif contract == "REMOTE_EXACT":
                # Remote fetch
                if block_id in self.remote_contracts:
                    stats, latency = self.remote_contracts[block_id].eval(Q)
                    results[block_id] = stats
                    total_latency_ms = max(total_latency_ms, latency)  # Parallel
                else:
                    stats = self._recompute(block_id, Q)
                    latency = self._estimate_recompute_latency_ms(block_id)
                    results[block_id] = stats
                    total_latency_ms += latency

            elif contract == "REHYDRATE":
                # Recompute from scratch
                stats = self._recompute(block_id, Q)
                latency = self._estimate_recompute_latency_ms(block_id)
                results[block_id] = stats
                total_latency_ms += latency

            elif contract == "DROP":
                # FIXED Issue 3: DROP branch — return zeros
                H = 12
                q_len = Q.shape[0] if len(Q.shape) == 2 else Q.shape[1]
                d = Q.shape[-1]
                results[block_id] = np.zeros((H, q_len, d), dtype=np.float32)
                total_latency_ms += 0.0  # No latency for drop

            # Unknown contract: default to recompute
            else:
                stats = self._recompute(block_id, Q)
                latency = self._estimate_recompute_latency_ms(block_id)
                results[block_id] = stats
                total_latency_ms += latency

        return results, total_latency_ms

    def _recompute(self, block_id: int, Q: np.ndarray) -> np.ndarray:
        """Simulate recomputation (milliseconds → returns stats)."""
        H = 12
        q_len = Q.shape[0] if len(Q.shape) == 2 else Q.shape[1]
        d = Q.shape[-1]
        rng = np.random.RandomState(block_id)
        return (rng.randn(H, q_len, d) * 0.1 + 0.5).astype(np.float32)

    def _estimate_recompute_latency_ms(self, block_id: int) -> float:
        """
        Estimate recompute latency in milliseconds.

        FIXED Issue 2: All return values in milliseconds.
        """
        if block_id in self.remote_contracts:
            seq_len = self.remote_contracts[block_id].seq_len
        else:
            seq_len = 4096

        # tokens * layers * cost_per_token (us) / 1000 = ms
        tokens = seq_len * 100  # 100 layers
        cost_us = tokens * self.recompute_cost_per_token_us
        return cost_us / 1000  # Convert to milliseconds


# ============================================================
# Stub contracts for local execution
# ============================================================

class LocalExactContract:
    """Stub for local exact contract."""
    def eval(self, Q: np.ndarray) -> Tuple[np.ndarray, float]:
        H = 12
        q_len = Q.shape[0] if len(Q.shape) == 2 else Q.shape[1]
        d = Q.shape[-1]
        stats = np.random.randn(H, q_len, d).astype(np.float32) * 0.1 + 0.5
        return stats, 0.1  # Local eval latency in ms


class SketchLocalContract:
    """Stub for local sketch contract."""
    def eval(self, Q: np.ndarray) -> Tuple[np.ndarray, float]:
        H = 12
        q_len = Q.shape[0] if len(Q.shape) == 2 else Q.shape[1]
        d = Q.shape[-1]
        stats = np.random.randn(H, q_len, d).astype(np.float32) * 0.1 + 0.5
        return stats, 0.05  # Sketch eval latency in ms


if __name__ == "__main__":
    print("=== RemoteExecutor v2 Test ===")

    # Create mock KV
    K = np.random.randn(4096, 64).astype(np.float32)
    V = np.random.randn(4096, 64).astype(np.float32)
    Q = np.random.randn(16, 64).astype(np.float32)

    # Create contract
    contract = RemoteExactContract(K, V, RTT_ms=5.0, jitter_ms=1.0, bandwidth_gbps=10.0)

    print(f"Bytes on wire: {contract.bytes_on_wire()}")
    print(f"KV bytes: {contract.kv_bytes / 1e6:.2f} MB")

    # Eval
    stats, latency = contract.eval(Q)
    print(f"Latency: {latency:.3f} ms")

    # Cache test — same query should get cached
    stats2, latency2 = contract.eval(Q)
    print(f"Cache latency: {latency2:.3f} ms")
    print(f"Instance cache stats: {contract.get_cache_stats()}")

    # Stable cache key test
    Q2 = Q.copy()  # Same content, different memory
    key1 = contract._stable_cache_key(Q)
    key2 = contract._stable_cache_key(Q2)
    print(f"\nStable cache key test:")
    print(f"  Q.copy() same key: {key1 == key2}")  # Should be True

    # Multi-tenant isolation test
    contract2 = RemoteExactContract(K, V, RTT_ms=5.0)
    stats3, _ = contract2.eval(Q)
    print(f"\nMulti-tenant isolation: contract2 cache size = {len(contract2._query_cache)}")
    print(f"  contract1 cache still has {len(contract._query_cache)} entries")

    # HybridRemoteExecutor with all contract types
    print("\n=== HybridRemoteExecutor (all 5 contracts) ===")
    hybrid = HybridRemoteExecutor(
        remote_contracts={0: contract},
        local_exact_contracts={1: LocalExactContract()},
        local_sketch_contracts={2: SketchLocalContract()},
    )

    block_ids = [0, 1, 2, 3, 4]
    contracts = ["REMOTE_EXACT", "EXACT_LOCAL", "SKETCH_LOCAL", "REHYDRATE", "DROP"]
    results, total_lat = hybrid.execute_batch(block_ids, Q, contracts)
    print(f"  Results for blocks: {list(results.keys())}")
    print(f"  Total latency: {total_lat:.3f} ms")
