# ACCORD-KV

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.XXXXX-red.svg)](https://arxiv.org/)

**ACCORD-KV: KV Cache compression via attention-weighted coreset + SVD + INT4, up to 50.8x memory compression**

---

## Quick Start

```bash
# 1. Install dependencies (CPU only, no GPU required)
pip install -r requirements.txt

# 2. Run a simulation demo
python -m simulation.backend_demo

# 3. View results
python simulation/fig2_plot.py
```

## Key Results

| Model        | Compression | K_rel   | V_rel     |
|--------------|-------------|---------|-----------|
| Mistral-7B (r=8)  | 28.3x       | 0.22-0.31 | bottleneck |
| Gemma-2-9B (r=8)  | **50.8x**   | 0.15-0.21 | good       |
| Gemma-2-9B (r=32) | 12.7x       | -       | -          |

- **42 GPU experiments**, 100% success rate on Mistral-7B-Instruct-v0.3 and Gemma-2-9B-it
- Key tensors: 0.9413 cumulative variance at rank 8; Value tensors: 0.5995 (asymmetric compression sensitivity)
- V matrix is the primary bottleneck; K reconstruction achieves rel_err ~0.15-0.24

## Architecture

ACCORD-KV reorganizes distributed KV cache through **three stages**:

```
core/  ->  simulation/  ->  cli/
  |           |              |
  |           |              +--- User-facing entry points
  |           |
  |           +--- End-to-end simulation + baseline comparisons
  |
  +--- Core data structures + mathematical primitives
        |-- acr.py          -- ACR (Attention Computation Request) protocol header
        |-- merge.py        -- FlashAttention (m,l,y) online-softmax state merge
        |-- attn_stats.py   -- AttnStats dataclass (m, l, y per head)
        +-- exact_local.py  -- Full-precision local serving
```

**Compression pipeline** (simulation/ -> gpu/):

1. **Coreset selection**: Attention-weighted k-center clustering selects representative tokens
2. **SVD decomposition**: Per-head rank-r truncation on K and V matrices
3. **INT4 quantization**: Quantize low-rank components to 4-bit integers
4. **Serial cascade**: Chain ExactLocal -> SketchLocal -> Drop contracts for progressive refinement

**Five Attention Contracts** govern block-level policy:

| Contract      | Precision              | Source      | Use Case                   |
|---------------|------------------------|-------------|----------------------------|
| `ExactLocal`  | FP16                   | Local GPU   | High-importance recent tokens |
| `SketchLocal` | Coreset + SVD + INT4  | Local       | Standard decode phase      |
| `RemoteExact` | FP16                   | Remote fetch | OOD fallback               |
| `Rehydrate`   | Upgrade compressed     | -           | Precision upgrade path     |
| `Drop`        | -                      | -           | Low-value blocks           |

## Repository Structure

```
AccordKV/
|-- core/                          # Core data structures & math primitives
|   |-- acr.py                     # ACR (Attention Computation Request) header
|   |-- merge.py                   # FlashAttention (m,l,y) online-softmax merge
|   |-- attn_stats.py              # AttnStats: m (max), l (sum), y (weighted sum)
|   |-- exact_local.py             # Full-precision local serving
|   +-- mock_attention_server.py   # Mock server for testing
|
|-- simulation/                    # CPU-only simulation (no GPU required)
|   |-- accord_backend.py          # Backend abstraction (FlashAttention/vLLM/Triton)
|   |-- anytime_theory.py          # Theoretical framework
|   |-- exp14_svd_mly_wire.py      # (m,l,y) wire format experiments
|   |-- exp24_cluster_aware.py     # Cluster-conditional V SVD
|   |-- backends/                  # Backend implementations
|   |   |-- flash_attn.py
|   |   |-- vllm.py
|   |   +-- triton.py
|   |-- baselines/                 # Baseline methods
|   |   |-- streaming_llm.py
|   |   |-- h2o.py
|   |   +-- scissorhands.py
|   |-- gpu/                       # GPU wire format abstractions
|   |   |-- gpu_wire_format.py
|   |   +-- gpu_svd_compress.py
|   |-- results/                   # Simulation experiment results
|   |   |-- baselines_data.json
|   |   +-- baselines_report.md
|   +-- fig*.py                    # Figure generation scripts
|
|-- gpu/                           # GPU experiment scripts (requires GPU)
|   |-- gpu_svd_compress_v8.py    # Core compression library (v8)
|   |-- gpu_all_exp_v5.py         # 42-config comprehensive experiment
|   |-- gpu_model_loader.py        # Model loading utilities
|   |-- exp_ppl_direct.py          # PPL downstream evaluation
|   +-- scripts/                   # SSH/helper scripts
|
|-- results/                       # GPU experiment results
|   |-- all_exp_v5_summary.json   # 42-config comprehensive results
|   |-- head_analysis.json        # 256 heads per-layer analysis
|   |-- m_*.json / g_*.json       # Per-config results (Mistral / Gemma)
|   +-- exp_v5.log                # Full experiment log
|
|-- docs/                          # Documentation
|   +-- ARCHITECTURE.md           # Detailed architecture documentation
|
|-- ACCORD_KV_paper.tex            # arXiv paper source
|-- ACCORD_KV_paper.pdf            # Compiled paper
|-- requirements.txt               # Python dependencies
|-- README.md                      # This file
+-- LICENSE                        # MIT License
```

## Citation

```bibtex
@article{accordkv2025,
  title={{ACCORD-KV}: Attention Contract Oriented Decentralized Reorganization of Distributed KV Cache},
  author={Yang Pengju},
  year={2025}
}
```

## Related Papers

| Paper        | arXiv      | Description |
|--------------|------------|-------------|
| **SpectrumKV** | [2606.08635](https://arxiv.org/abs/2606.08635) | Spectrum-aware KV cache compression via low-rank SVD on the attention score matrix |
| StreamingLLM | [2309.17453](https://arxiv.org/abs/2309.17453) | Efficient streaming language models with attention sink |
| H2O          | [2310.16744](https://arxiv.org/abs/2310.16744) | Heavy-Hitter Oracle for KV cache compression |
| KVQuant      | [2402.10066](https://arxiv.org/abs/2402.10066) | KV cache quantization with per-token and per-head calibration |
| Splitwise    | [2307.12504](https://arxiv.org/abs/2307.12504) | Disaggregating prefill and decoding phases for LLM serving |
