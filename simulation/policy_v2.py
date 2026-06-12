"""
simulation/policy_v2.py — ACCORD-KV Policy Selector (Fixed v2)

Changes from v1 (review_to_delete/v1_subagent_output/policy.py):
==============================================================
Issue 1: Type hints missing
  - FIX: Added complete type hints to all functions

Issue 2: Docstrings missing
  - FIX: Added detailed docstrings to choose_contract, choose_contract_deadline_aware,
    choose_contract_streaming

Issue 3: Magic numbers 0.7/0.5/0.2
  - FIX: Extracted to module-level constants:
      HOTNESS_EXACT_LOCAL = 0.7
      HOTNESS_SKETCH_LOCAL = 0.5
      HOTNESS_REMOTE_EXACT = 0.2

Issue 4: compute_clusterability uses string block_type
  - FIX: Added BlockType enum
      class BlockType(Enum):
          SHARED = "shared"
          LAYER_NORM = "layer_norm"
          FFN = "ffn"
          ATTENTION = "attention"
          DEFAULT = "default"
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Set
from enum import Enum


# ============================================================
# Module-level constants (FIXED Issue 3)
# ============================================================

# Importance thresholds for contract selection
HOTNESS_EXACT_LOCAL: float = 0.7   # importance > 0.7 → EXACT_LOCAL
HOTNESS_SKETCH_LOCAL: float = 0.5  # importance > 0.5 → SKETCH_LOCAL candidate
HOTNESS_REMOTE_EXACT: float = 0.2  # importance > 0.2 → consider REMOTE_EXACT
DROP_THRESHOLD: float = 0.1         # importance < 0.1 → DROP

# Clusterability thresholds
CLUSTERABILITY_SKETCH: float = 0.5  # clusterability > 0.5 → SKETCH_LOCAL candidate

# Streaming thresholds
STREAMING_UPGRADE_SKETCH: float = 0.4  # observed_hotness > 0.4 → upgrade to sketch
STREAMING_UPGRADE_EXACT: float = 0.7  # observed_hotness > 0.7 → upgrade to exact
STREAMING_PREFILL_EXACT: float = 0.8  # importance > 0.8 → EXACT_LOCAL in prefill

# Deadline-aware importance thresholds
DEADLINE_IMPORTANCE_EXACT: float = 0.7
DEADLINE_IMPORTANCE_SKETCH: float = 0.5
DEADLINE_IMPORTANCE_REMOTE: float = 0.2
DEADLINE_IMPORTANCE_REHYDRATE: float = 0.1


# ============================================================
# BlockType enum (FIXED Issue 4)
# ============================================================

class BlockType(Enum):
    """Enumeration of block types for clusterability scoring."""
    SHARED = "shared"
    LAYER_NORM = "layer_norm"
    FFN = "ffn"
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    DEFAULT = "default"


# ============================================================
# Contract type alias
# ============================================================

Contract = str  # Literal['EXACT_LOCAL', 'SKETCH_LOCAL', 'REMOTE_EXACT', 'REHYDRATE', 'DROP']


# ============================================================
# Feature type hints
# ============================================================

BlockFeatures = Dict[str, Any]
NetworkState = Dict[str, Any]
ObservedAccessPattern = Dict[int, int]


# ============================================================
# Feature computation functions
# ============================================================

def compute_block_importance(
    block_features: BlockFeatures,
    history_q: Optional[np.ndarray],
    history_attn: Optional[np.ndarray],
) -> float:
    """
    Compute importance score for a block based on:
    - Historical query-key overlap (semantic relevance)
    - Attention weight distribution
    - Recency (exponential decay)

    Returns score in [0, 1].
    """
    if history_q is None or len(history_q) == 0:
        freq = block_features.get("access_frequency", 0.0)
        recency = block_features.get("recency_score", 0.5)
        semantic = block_features.get("semantic_relevance", 0.5)
        return float(np.clip(0.3 * freq + 0.4 * recency + 0.3 * semantic, 0.0, 1.0))

    attn_mean = float(np.mean(history_attn)) if len(history_attn) > 0 else 0.0
    attn_std = float(np.std(history_attn)) if len(history_attn) > 0 else 0.0
    attn_max = float(np.max(history_attn)) if len(history_attn) > 0 else 0.0
    recency = block_features.get("recency_score", 0.5)

    qk_overlap = float(np.dot(history_q[-1], history_q[-2])) if len(history_q) >= 2 else 0.5
    qk_overlap = (qk_overlap + 1) / 2

    importance = (
        0.25 * attn_mean +
        0.25 * attn_max +
        0.20 * attn_std +
        0.15 * qk_overlap +
        0.15 * recency
    )

    return float(np.clip(importance, 0.0, 1.0))


def compute_clusterability(block_features: BlockFeatures) -> float:
    """
    Compute how well this block's attention patterns cluster with others.
    High clusterability = good candidate for SKETCH_LOCAL (shared compression).

    FIXED Issue 4: Uses BlockType enum instead of raw strings.

    Returns score in [0, 1].
    """
    block_type_str = block_features.get("block_type", "default")

    # Map string to enum (defensive: handle both enum and string)
    if isinstance(block_type_str, BlockType):
        block_type = block_type_str
    else:
        block_type = BlockType(block_type_str)

    type_scores: Dict[BlockType, float] = {
        BlockType.SHARED: 0.9,
        BlockType.LAYER_NORM: 0.8,
        BlockType.FFN: 0.6,
        BlockType.ATTENTION: 0.7,
        BlockType.EMBEDDING: 0.5,
        BlockType.DEFAULT: 0.5,
    }

    base_score = type_scores.get(block_type, 0.5)

    pattern_consistency = block_features.get("pattern_consistency", 0.5)
    entropy = block_features.get("attention_entropy", 0.5)

    clusterability = base_score * (0.6 + 0.4 * pattern_consistency) * (1.0 - 0.3 * entropy)

    return float(np.clip(clusterability, 0.0, 1.0))


def estimate_remote_cost(
    block_features: BlockFeatures,
    network_state: NetworkState,
    memory_budget: float,
) -> Tuple[float, float]:
    """
    Estimate cost of REMOTE_EXACT contract.

    Returns (remote_cost_ms, raw_cost_ms) in milliseconds.
    Latency model: T_rpc = bytes / bandwidth + RTT/2
    """
    block_size_kv: int = block_features.get("block_size_kv", 4096)
    bandwidth_gbps: float = network_state.get("bandwidth_gbps", 10.0)
    rtt_ms: float = network_state.get("rtt_ms", 5.0)

    # Bytes on wire: 64 byte handle + KV data (2 bytes per token)
    total_bytes = 64 + block_size_kv * 2

    T_rpc: float = (total_bytes * 8) / (bandwidth_gbps * 1e9) * 1000
    T_rpc += rtt_ms / 2  # one-way latency

    tokens_per_block: int = block_features.get("tokens_in_block", block_size_kv)
    compute_cost_us: float = block_features.get("compute_cost_us", 50.0)
    num_layers: int = block_features.get("num_layers", 100)
    T_raw: float = tokens_per_block * compute_cost_us * num_layers / 1000

    return T_rpc, T_raw


# ============================================================
# Contract selection functions
# ============================================================

def choose_contract(
    block_features: BlockFeatures,
    network_state: NetworkState,
    memory_budget: float,
    error_budget: float = 1e-3,
    hot_threshold: float = HOTNESS_EXACT_LOCAL,
    cold_threshold: float = HOTNESS_REMOTE_EXACT,
    drop_threshold: float = DROP_THRESHOLD,
) -> Contract:
    """
    Choose the optimal contract for a block.

    Decision rules (priority order):
    1. importance < drop_threshold → DROP
    2. importance > hot_threshold → EXACT_LOCAL
    3. clusterability > 0.5 AND error > 0.1*error_budget → SKETCH_LOCAL
    4. remote_cost < raw_cost AND importance > cold_threshold → REMOTE_EXACT
    5. otherwise → REHYDRATE

    FIXED Issue 1: Added complete type hints.
    FIXED Issue 2: Added detailed docstring.
    FIXED Issue 3: Uses module-level constants (HOTNESS_EXACT_LOCAL etc.).

    Args:
        block_features: Dict with block metadata and historical stats
        network_state: Dict with bandwidth, RTT, etc.
        memory_budget: Available memory in bytes
        error_budget: Acceptable approximation error (default 1e-3)
        hot_threshold: Importance threshold for EXACT_LOCAL (default 0.7)
        cold_threshold: Importance threshold for REMOTE_EXACT (default 0.2)
        drop_threshold: Importance threshold for DROP (default 0.1)

    Returns:
        Contract type: 'EXACT_LOCAL' | 'SKETCH_LOCAL' | 'REMOTE_EXACT' | 'REHYDRATE' | 'DROP'
    """
    importance: float = compute_block_importance(
        block_features,
        block_features.get("history_q"),
        block_features.get("history_attn"),
    )

    clusterability: float = compute_clusterability(block_features)

    remote_cost: float
    raw_cost: float
    remote_cost, raw_cost = estimate_remote_cost(block_features, network_state, memory_budget)

    # Decision logic (priority order)

    # 1. DROP if importance is extremely low
    if importance < drop_threshold:
        return "DROP"

    # 2. EXACT_LOCAL if importance is high (hot block)
    if importance > hot_threshold:
        return "EXACT_LOCAL"

    # 3. SKETCH_LOCAL if clusterable and error tolerance allows
    error_estimate: float = block_features.get("estimated_error", error_budget)
    if clusterability > CLUSTERABILITY_SKETCH and error_estimate > 0.1 * error_budget:
        return "SKETCH_LOCAL"

    # 4. REMOTE_EXACT if remote is cheaper than recompute and block is moderately important
    if remote_cost < raw_cost and importance > cold_threshold:
        return "REMOTE_EXACT"

    # 5. REHYDRATE (recompute) for everything else
    return "REHYDRATE"


def choose_contract_deadline_aware(
    block_features: BlockFeatures,
    network_state: NetworkState,
    memory_budget: float,
    deadline_class: str,  # 'TIGHT' | 'MODERATE' | 'LOOSE' | 'LAZY'
    error_budget: float = 1e-3,
) -> Contract:
    """
    Deadline-aware contract selection.

    Deadline classes constrain the available contract pool:
    - TIGHT   (RTT < 1ms):   Only EXACT_LOCAL (must be preloaded)
    - MODERATE (RTT 1-10ms): EXACT_LOCAL + SKETCH_LOCAL
    - LOOSE   (RTT > 10ms):  EXACT_LOCAL + SKETCH_LOCAL + REMOTE_EXACT
    - LAZY:                   All contracts including REHYDRATE

    Decision: Choose highest-importance contract within allowed pool.

    FIXED Issue 1: Added complete type hints.
    FIXED Issue 2: Added detailed docstring.

    Args:
        block_features: Block feature dict
        network_state: Network state dict
        memory_budget: Available memory in bytes
        deadline_class: One of 'TIGHT', 'MODERATE', 'LOOSE', 'LAZY'
        error_budget: Acceptable error

    Returns:
        Contract type constrained by deadline class
    """
    importance: float = compute_block_importance(
        block_features,
        block_features.get("history_q"),
        block_features.get("history_attn"),
    )

    # Deadline-constrained contract pools
    contract_pools: Dict[str, Set[Contract]] = {
        "TIGHT": {"EXACT_LOCAL"},
        "MODERATE": {"EXACT_LOCAL", "SKETCH_LOCAL"},
        "LOOSE": {"EXACT_LOCAL", "SKETCH_LOCAL", "REMOTE_EXACT"},
        "LAZY": {"EXACT_LOCAL", "SKETCH_LOCAL", "REMOTE_EXACT", "REHYDRATE", "DROP"},
    }

    allowed: Set[Contract] = contract_pools.get(deadline_class, contract_pools["LAZY"])

    # Apply same decision logic but restricted to allowed contracts
    if importance > DEADLINE_IMPORTANCE_EXACT:
        if "EXACT_LOCAL" in allowed:
            return "EXACT_LOCAL"
    elif importance > DEADLINE_IMPORTANCE_SKETCH:
        if "SKETCH_LOCAL" in allowed:
            return "SKETCH_LOCAL"
    elif importance > DEADLINE_IMPORTANCE_REMOTE:
        if "REMOTE_EXACT" in allowed:
            return "REMOTE_EXACT"
    elif importance > DEADLINE_IMPORTANCE_REHYDRATE:
        if "REHYDRATE" in allowed:
            return "REHYDRATE"

    # Fallback: pick first available
    return next(iter(allowed))


def choose_contract_streaming(
    block_features: BlockFeatures,
    network_state: NetworkState,
    memory_budget: float,
    phase: str,  # 'prefill' | 'decode'
    observed_access_pattern: ObservedAccessPattern,
    current_contract: Contract,
    error_budget: float = 1e-3,
) -> Contract:
    """
    Streaming/iterative contract selection.

    Similar to OS page fault → swap → prefetch mechanism:
    - Prefill: Unknown hot blocks, use REHYDRATE initially
    - Decode: Observe access patterns, upgrade hot blocks progressively
    - Upgrade path: REHYDRATE → SKETCH_LOCAL → EXACT_LOCAL

    FIXED Issue 1: Added complete type hints.
    FIXED Issue 2: Added detailed docstring.
    FIXED Issue 3: Uses module-level constants.

    Args:
        block_features: Block feature dict
        network_state: Network state dict
        memory_budget: Available memory
        phase: 'prefill' or 'decode'
        observed_access_pattern: block_id -> access_count
        current_contract: Current contract for this block
        error_budget: Acceptable error

    Returns:
        Next contract (may be same, upgraded, or downgraded)
    """
    block_id: int = block_features.get("block_id", -1)
    importance: float = compute_block_importance(
        block_features,
        block_features.get("history_q"),
        block_features.get("history_attn"),
    )

    # Track observed hotness (normalized to [0, 1])
    access_count: int = observed_access_pattern.get(block_id, 0)
    observed_hotness: float = min(access_count / 10.0, 1.0)

    if phase == "prefill":
        # Prefill: default to REHYDRATE (don't commit memory)
        if importance > STREAMING_PREFILL_EXACT:
            return "EXACT_LOCAL"
        return "REHYDRATE"

    # Decode phase: upgrade based on observed access
    clusterability: float = compute_clusterability(block_features)

    # Upgrade threshold tracking (FIXED Issue 3: use constants)
    upgrade_to_sketch: bool = (
        observed_hotness > STREAMING_UPGRADE_SKETCH or importance > HOTNESS_SKETCH_LOCAL
    )
    upgrade_to_exact: bool = (
        observed_hotness > STREAMING_UPGRADE_EXACT or importance > HOTNESS_EXACT_LOCAL
    )

    # Contract upgrade path
    contract_rank: Dict[Contract, int] = {
        "DROP": 0,
        "REHYDRATE": 1,
        "SKETCH_LOCAL": 2,
        "REMOTE_EXACT": 2,
        "EXACT_LOCAL": 3,
    }

    current_rank: int = contract_rank.get(current_contract, 0)

    # Determine target rank
    if upgrade_to_exact and current_rank < 3:
        target_rank: int = 3
    elif upgrade_to_sketch and current_rank < 2:
        target_rank = 2
    else:
        target_rank = current_rank

    # Map rank back to contract
    rank_to_contract: Dict[int, Contract] = {
        0: "DROP",
        1: "REHYDRATE",
        2: "SKETCH_LOCAL",
        3: "EXACT_LOCAL",
    }

    return rank_to_contract[target_rank]


# ============================================================
# Batch processing
# ============================================================

def choose_contracts_batch(
    blocks: List[BlockFeatures],
    network_state: NetworkState,
    memory_budget: float,
    error_budget: float = 1e-3,
    policy: str = "fixed",  # 'fixed' | 'deadline_aware' | 'streaming'
    deadline_class: str = "LOOSE",
    phase: str = "decode",
    observed_access_pattern: Optional[ObservedAccessPattern] = None,
) -> List[Contract]:
    """
    Choose contracts for multiple blocks.

    Args:
        blocks: List of block feature dicts
        network_state: Network conditions
        memory_budget: Available memory
        error_budget: Error tolerance
        policy: 'fixed' | 'deadline_aware' | 'streaming'
        deadline_class: For deadline_aware policy
        phase: For streaming policy ('prefill' | 'decode')
        observed_access_pattern: For streaming policy (block_id -> count)

    Returns:
        List of contract choices (same length as blocks)
    """
    if observed_access_pattern is None:
        observed_access_pattern = {}

    contracts: List[Contract] = []
    for i, block in enumerate(blocks):
        if "block_id" not in block:
            block["block_id"] = i

        current_contract: Contract = block.get("current_contract", "REHYDRATE")
        block["current_contract"] = current_contract

        if policy == "deadline_aware":
            contract = choose_contract_deadline_aware(
                block, network_state, memory_budget, deadline_class, error_budget
            )
        elif policy == "streaming":
            contract = choose_contract_streaming(
                block, network_state, memory_budget, phase,
                observed_access_pattern, current_contract, error_budget
            )
        else:
            contract = choose_contract(block, network_state, memory_budget, error_budget)

        contracts.append(contract)

    return contracts


if __name__ == "__main__":
    # Quick test
    print("=== Policy v2 Test ===")

    # Mock block features
    block: BlockFeatures = {
        "block_id": 0,
        "block_type": BlockType.ATTENTION,  # FIXED: use enum
        "access_frequency": 0.8,
        "recency_score": 0.9,
        "semantic_relevance": 0.7,
        "pattern_consistency": 0.7,
        "attention_entropy": 0.3,
        "block_size_kv": 4096,
        "tokens_in_block": 4096,
        "num_layers": 100,
    }

    network: NetworkState = {"bandwidth_gbps": 10.0, "rtt_ms": 5.0}

    print("\n=== Fixed Policy ===")
    c = choose_contract(block, network, 1e9)
    print(f"  Contract: {c}")
    print(f"  Importance: {compute_block_importance(block, None, None):.3f}")
    print(f"  Clusterability: {compute_clusterability(block):.3f}")

    print("\n=== Deadline-Aware Policy ===")
    for deadline in ["TIGHT", "MODERATE", "LOOSE", "LAZY"]:
        contract = choose_contract_deadline_aware(block, network, 1e9, deadline)
        print(f"  {deadline}: {contract}")

    print("\n=== Streaming Policy ===")
    access_pattern: ObservedAccessPattern = {0: 8, 1: 2, 2: 0}
    for phase in ["prefill", "decode"]:
        print(f"  Phase {phase}:")
        for current in ["REHYDRATE", "SKETCH_LOCAL"]:
            block["current_contract"] = current
            contract = choose_contract_streaming(
                block, network, 1e9, phase, access_pattern, current
            )
            print(f"    {current} → {contract}")

    print("\n=== BlockType enum test ===")
    for bt in BlockType:
        block_test = {"block_type": bt, "pattern_consistency": 0.5, "attention_entropy": 0.5}
        print(f"  {bt.name}: clusterability={compute_clusterability(block_test):.2f}")
