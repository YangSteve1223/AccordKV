"""exp1 — Fidelity vs Bandwidth baseline."""
import sys
from pathlib import Path

# Link to original simulation exp1
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "simulation"))

# Re-export from simulation/exp1_fidelity_vs_bandwidth
from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    numpy_merge_stats,
    numpy_merge_stats_list,
    ground_truth,
    make_kv_cache,
    serve_local,
    bytes_kv_path,
    bytes_stats_path,
)

__all__ = [
    "NumpyAttnStats",
    "numpy_merge_stats",
    "numpy_merge_stats_list",
    "ground_truth",
    "make_kv_cache",
    "serve_local",
    "bytes_kv_path",
    "bytes_stats_path",
]
