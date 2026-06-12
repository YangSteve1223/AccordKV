"""
ACCORD-KV Baselines Package
============================

Implements 4 post-hoc KV compression baselines:
- H2O (Heavy-Hitter Oracle, NeurIPS 2023)
- StreamingLLM (ICLR 2024)
- Scissorhands (NeurIPS 2023)
- FastGen (ACL 2024)

Run:
    python -m simulation.baselines.run_baselines
"""

from simulation.baselines.h2o import h2o_full_compression
from simulation.baselines.streaming_llm import streaming_llm_full_compression
from simulation.baselines.scissorhands import scissorhands_full_compression
from simulation.baselines.fastgen import fastgen_full_compression

__all__ = [
    "h2o_full_compression",
    "streaming_llm_full_compression",
    "scissorhands_full_compression",
    "fastgen_full_compression",
]
