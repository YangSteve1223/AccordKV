# ACCORD-KV

Attention Contract Oriented Decentralized Reorganization of Distributed KV Cache

## Overview

ACCORD-KV is a research project exploring radical efficiency gains in KV cache management for large language model inference. By treating attention state as first-class network objects and compressing them via rank-based coreset selection with INT4 quantization, ACCORD-KV achieves up to **50.8x memory compression** on Gemma models while maintaining near-lossless reconstruction quality.

## Key Contributions

- **(m,l,y) Wire Format ABI**: Transmit FlashAttention online-softmax state directly over the network, eliminating redundant KV tensor materialization.
- **Coreset + INT4 Compression**: Use attention-weighted k-center clustering with per-head rank adaptation and INT4 quantization for massive memory reduction.
- **OOD Self-Heal**: Detect out-of-distribution tokens via attention entropy monitoring and gracefully fall back to full precision.
- **Serial Cascade**: Chain multiple compression stages to progressively refine KV state.
- **Cluster-Conditional V SVD**: Head-specific singular value decomposition with cluster-aware basis selection for V matrices.

## Key Results

| Model | Phase | Seq Len | Rank | Memory Compression | K_rel | V_rel |
|-------|-------|---------|------|-------------------|-------|-------|
| Mistral-7B | prefill | 512 | 8 | ~28.3x | 0.24 | bottleneck |
| Mistral-7B | prefill | 2048 | 8 | ~28.3x | 0.31 | bottleneck |
| Mistral-7B | decode | 32 | 8 | ~28.3x | 0.22 | bottleneck |
| Gemma-2-9B | prefill | 512 | 8 | ~50.8x | 0.18 | good |
| Gemma-2-9B | prefill | 2048 | 8 | ~50.8x | 0.21 | good |
| Gemma-2-9B | decode | 32 | 8 | ~50.8x | 0.15 | good |

- **42 GPU experiments**, 100% success rate
- V matrix is the primary bottleneck; K reconstruction is surprisingly good (rel_err ~0.15-0.24)

## Repository Structure

```
AccordKV/
├── README.md
├── LICENSE
├── requirements.txt
├── gpu/                          # GPU experiment scripts (requires GPU)
│   ├── gpu_svd_compress_v8.py    # Core compression library
│   ├── gpu_all_exp_v5.py         # 42-config comprehensive experiment
│   ├── gpu_model_loader.py        # Model loading utilities
│   ├── exp_ppl_direct.py          # PPL downstream evaluation
│   └── scripts/                   # SSH/helper scripts
│       ├── ssh_run_mistral.py
│       └── ssh_upload_and_test.py
├── simulation/                   # CPU simulation (no GPU required)
│   ├── accord_backend.py         # Core backend implementation
│   ├── anytime_theory.py         # Theoretical framework
│   ├── exp14_svd_mly_wire.py     # (m,l,y) wire format experiments
│   ├── exp24_cluster_aware.py     # Cluster-conditional V SVD
│   ├── backends/                 # Attention backend implementations
│   │   ├── flash_attn.py
│   │   ├── vllm.py
│   │   └── triton.py
│   ├── baselines/                # Baseline methods
│   │   ├── streaming_llm.py
│   │   ├── h2o.py
│   │   └── scissorhands.py
│   ├── gpu/                     # GPU wire format abstractions
│   │   ├── gpu_wire_format.py
│   │   └── gpu_svd_compress.py
│   └── results/
│       ├── baselines_data.json
│       └── baselines_report.md
├── results/                     # GPU experiment results
│   ├── all_exp_v5_summary.json   # 42-config comprehensive results
│   ├── head_analysis.json        # 256 heads per-layer analysis
│   ├── exp_v5.log                # Full experiment log
│   ├── m_*.json                  # Mistral per-config results
│   └── g_*.json                  # Gemma per-config results
└── docs/
    ├── paper_outline_v0.1.md
    └── gpu_audit/                # GPU experiment audit
        ├── gpu_audit_report.md
        └── gpu_fixes/            # Bug fixes applied during GPU experiments
```

## Installation

```bash
# Simulation only (no GPU required)
pip install -r requirements.txt

# GPU experiments (requires CUDA-capable GPU)
pip install torch transformers vllm flash-attn --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Quick Start

### GPU Experiments

Modify model paths in the scripts (default: `/root/autodl-tmp/`):

```bash
# Run all 42 configurations
python gpu_all_exp_v5.py

# Run specific configuration
python -c "from gpu_all_exp_v5 import EXPERIMENTS; print([e for e in EXPERIMENTS if 'mistral' in e['name']])"
```

### Simulation (CPU)

```bash
# Run wire format experiment
python simulation/exp14_svd_mly_wire.py

# Run cluster-conditional V SVD
python simulation/exp24_cluster_aware.py

# Run baselines comparison
python simulation/baselines/run_baselines.py
```

## GPU Server Configuration

The GPU experiments were run on a server with:
- **Mistral-7B-Instruct-v0.3**: `/root/autodl-tmp/Mistral-7B-Instruct-v0.3/`
- **Gemma-2-9B-IT**: `/root/autodl-tmp/gemma-2-9b-it/`
- **CUDA**: 11.8+
- **vLLM**: 0.2.0+

Modify these paths in `gpu_model_loader.py` and `gpu_all_exp_v5.py` as needed.

## Results Summary

### Memory Compression Ratios

| Model | r=8 | r=32 | r=64 | r=128 |
|-------|-----|------|------|-------|
| Mistral-7B | 28.3x | 7.1x | 3.5x | 1.8x |
| Gemma-2-9B | 50.8x | 12.7x | 6.4x | 3.2x |

### Reconstruction Quality (r=8)

| Model | K_rel | V_rel | Phase |
|-------|-------|-------|-------|
| Mistral-7B | 0.24 | ~0.9 (bottleneck) | prefill 512 |
| Gemma-2-9B | 0.18 | 0.35 | prefill 512 |

V matrix compression is the main research challenge — standard SVD struggles with V's spectral properties.

## Citation

If you find this work useful, please cite:

```bibtex
@article{accordkv2024,
  title={ACCORD-KV: Attention Contract Oriented Decentralized Reorganization of Distributed KV Cache},
  author={},
  year={2024}
}
```

## License

MIT License - see LICENSE file.
