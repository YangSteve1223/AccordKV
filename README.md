# ACCORD-KV: Attention Contract-Based KV Cache Reorganization

Large language models (LLMs) suffer from the Prefill-Decode (PD) disaggregation bottleneck, where the Key-Value (KV) cache grows explosively during the prefill phase and causes significant memory and bandwidth overhead. To address this, we propose **Attention Contract**, a novel KV cache reorganization framework that formulates compression as a communication contract between prefill and decode nodes.

## Attention Contract Framework

Attention Contract defines five fundamental contract types:
1. **Budget Contract**: Target compression ratio with quality bounds
2. **Quality Contract**: Target quality with minimal bandwidth
3. **Hybrid Contract**: Joint bandwidth-quality Pareto frontier
4. **Adaptive Contract**: Dynamic adjustment during inference
5. **Serial Contract**: Streaming decode with incremental contracts

Our key insight reveals a **V-bottleneck** phenomenon: Key (K) cache shows rapid decay in importance (cumvar > 0.94 at rank 8), while Value (V) cache retains high residual variance (cumvar ≈ 0.60), indicating V requires higher compression rank than K.

## Key Contributions

- **Attention Contract ABI**: Standardized wire format `(m, l, γ)` with backward compatibility across 31,775× model configurations
- **Coreset + INT4 Quantization**: 7.3× compression with only 0.16% relative error
- **Cluster-conditional SVD**: 11.6–12.2× better quality than H2O, StreamingLLM, Scissorhands, and FastGen baselines
- **Serial Cascade**: 128–255× decode speedup at 0.22% error rate
- **V-bottleneck Analysis**: K cumvar 0.94 vs V cumvar 0.60 at rank 8, guiding rank allocation

## Repository Structure

```
ACCORD-KV/
├── gpu/                    # GPU experiments
│   ├── gpu_svd_compress_v8.py   # Core SVD compression
│   ├── exp_ppl_direct.py            # Downstream PPL evaluation
│   └── all_exp_v5.py             # Full benchmark suite
├── simulation/             # CPU simulation framework
│   ├── backends/           # Backend implementations
│   ├── baselines/          # H2O, StreamingLLM, Scissorhands, FastGen
│   └── exp*.py             # Individual experiments
├── results/                # Experiment data (JSON)
└── docs/                   # Design documents
```

## Quick Start

```bash
# Full GPU experiment pipeline
cd gpu && python all_exp_v5.py

# Downstream PPL evaluation
python exp_ppl_direct.py

# CPU simulation
cd simulation && python exp10_kmean_normalized_sketch.py
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.40+
- NVIDIA GPU with ≥24GB VRAM (for GPU experiments)
- numpy, scipy, scikit-learn

## Citation

```
@article{accordkv2025,
  title={ACCORD-KV: Attention Contract-Based KV Cache Reorganization},
  author={Yang et al.},
  journal={arXiv:2606.08635},
  year={2025}
}
```

📄 **Paper**: https://arxiv.org/abs/2606.08635
