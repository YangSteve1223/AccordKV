# ACCORD-KV Architecture

This document describes the overall architecture of ACCORD-KV, the key modules, and the experiment workflow.

## Overall Architecture

ACCORD-KV follows a **three-layer pipeline**:

```
core/  -->  simulation/  -->  cli/ (future)
  |           |              |
  |           |              +-- User-facing entry points
  |           |
  |           +-- End-to-end simulation + baseline comparisons
  |
  +-- Core data structures + mathematical primitives
        +-- Protocol layer (ACR contracts)
        +-- Merge layer (FlashAttention state merge)
        +-- Attention layer (full-precision serving)
```

- **core/**: Language-agnostic mathematical primitives. No PyTorch/TensorFlow dependency. Defines the ACR protocol, AttnStats merge, and exact attention contract.
- **simulation/**: Full Python simulation using numpy. Reproduces end-to-end behavior, runs baseline comparisons, and generates figures. No GPU required.
- **gpu/**: Real GPU implementations of the compression pipeline. Depends on PyTorch, transformers, vLLM.
- **cli/**: (Future) Command-line interface and user-facing tools.

## Core Modules

### core/acr.py -- ACR Protocol

`ACR` (Attention Computation Request) is a frozen dataclass that acts as the **request header** for the AVL protocol. It carries:

| Field          | Type          | Description                          |
|----------------|---------------|--------------------------------------|
| `acr_id`       | str           | Unique request identifier            |
| `q_block_id`   | int           | Query block ID                       |
| `q_tokens`      | torch.Tensor  | Query token embeddings               |
| `server_hints`  | List[str]     | Preferred server list                |
| `block_ids`     | List[int]     | KV block IDs to access               |
| `contract_type` | ContractType  | EXACT / APPROX / BOUNDED             |
| `deadline_ms`   | float         | Latency budget                       |
| `error_budget`  | float         | Max relative error (BOUNDED)         |

`ContractType` enum governs the server-side approximation contract:
- `EXACT`: Full precision required
- `APPROX`: (m,l,y) sketch path allowed
- `BOUNDED`: Relative error must stay within `error_budget`

### core/merge.py -- AttnStats Merge

`merge_stats(a, b)` performs **numerically stable fusion** of two FlashAttention online-softmax state tuples (m, l, y):

```
m_new = max(m1, m2)
a1 = exp(m1 - m_new), a2 = exp(m2 - m_new)
l_new = l1 * a1 + l2 * a2
y_new = y1 * a1 + y2 * a2
```

Properties:
- **Associative**: merge(a, merge(b, c)) == merge(merge(a, b), c)
- **Commutative**: merge(a, b) == merge(b, a)
- **Numerically stable**: exp argument always in [0, 1]

Boundary case: both inputs empty (m = -inf) -> result stays empty (l=0, y=0, m=-inf).

### core/attn_stats.py -- AttnStats Dataclass

`AttnStats` holds per-head attention statistics:
- `m`: Per-head max of attention scores (shape: [num_heads])
- `l`: Per-head sum of exp scores (shape: [num_heads])
- `y`: Per-head weighted value sum (shape: [num_heads, head_dim])

Together (m, l, y) is sufficient to compute attention output without materializing full K/V tensors.

### core/exact_local.py -- Full-Precision Serving

Implements the `ExactLocal` contract: full FP16 precision serving from local GPU memory. Serves as the ground truth baseline for all other contracts.

## Simulation Modules

### simulation/accord_backend.py -- Backend Abstraction

Abstract base class `AccordBackend` defines the uniform interface for KV cache operations across different hardware backends:

```python
class AccordBackend(ABC):
    @abstractmethod
    def encode_kv(self, K, V, block_meta) -> bytes: ...
    @abstractmethod
    def decode_kv(self, wire: bytes) -> Tuple[np.ndarray, np.ndarray]: ...
    @abstractmethod
    def attention(self, Q, block_ids) -> np.ndarray: ...
```

Implementations:
- `FlashAttentionBackend`: Simulated FlashAttention 2 behavior (numpy)
- `vLLMBackend`: vLLM PagedAttention integration
- `TritonBackend`: Triton kernel integration

### simulation/exp14_svd_mly_wire.py -- (m,l,y) Wire Format

Tests three SVD encoding strategies for the (m,l,y) wire format:

- **Method A**: SVD as post-processing - (m,l) preserve FlashAttention semantics, y_svd is compressed output
- **Method B**: SVD as kernel approximation - (m,l) based on A_r, y = A_r @ V
- **Method C**: Dual-layer structure - first layer guarantees mathematical correctness, second layer provides compression

Key invariant: merge((m1,l1,y1), (m2,l2,y2)) == merge((m1,l1,y1_svd), (m2,l2,y2_svd))

### simulation/exp24_cluster_aware.py -- Cluster-Conditional V SVD

Five strategies for V matrix compression (post-SVD cluster boundary protection):

- **Baseline**: Serial Cascade (Coreset + SVD r=8 + INT4)
- **Method A**: Cluster Boundary Residual Correction
- **Method B**: Attention-Output Rescaling
- **Method C**: K-aware V compression
- **Method D**: Hybrid (C + B) - best overall

Discovery: V matrices are already low-rank (rank at 90%% = 7-8, condition number = 73-387). The real problem is attention interaction error after V compression.

### simulation/baselines/ -- Baseline Methods

Reference implementations of prior KV cache compression work:

| File              | Paper            | Method                                        |
|-------------------|------------------|-----------------------------------------------|
| `streaming_llm.py`  | StreamingLLM (2023) | Keep first 4 + recent tokens               |
| `h2o.py`           | H2O (2023)       | Heavy-hitter token selection by gradient      |
| `scissorhands.py`  | Scissorhands (2024) | Importance-based token retention          |
| `fastgen.py`       | FastGen (2024)   | Profile-guided table lookup                  |

## GPU Modules

### gpu/gpu_svd_compress_v8.py -- Core Compression Library

The production compression pipeline. Implements:

1. **Coreset selection**: Attention-weighted k-center clustering with K-Means++ initialization
2. **Per-head SVD**: Rank-r truncation with cluster-conditional basis selection for V
3. **INT4 quantization**: Per-head scale factors, zero-point offset

Key finding: K captures 0.9413 cumulative variance at rank 8; V captures only 0.5995 - asymmetric compression sensitivity.

### gpu/gpu_all_exp_v5.py -- 42-Config Experiment Suite

Comprehensive ablation study covering:

- **Models**: Mistral-7B-Instruct-v0.3, Gemma-2-9B-it
- **Phases**: prefill (512, 2048 tokens), decode (32 tokens)
- **Ranks**: 8, 32, 64, 128

Total: 2 models x 3 phases x 7 configs = 42 configurations.

### gpu/gpu_model_loader.py -- Model Loading

Utilities for loading models and extracting KV cache tensors. Supports HuggingFace transformers and vLLM model formats.

## Experiment Workflow

### Running Experiments

```bash
# CPU simulation (no GPU required)
python simulation/exp14_svd_mly_wire.py          # Wire format experiments
python simulation/exp24_cluster_aware.py         # Cluster-aware V SVD
python simulation/baselines/run_baselines.py    # Baseline comparisons

# GPU experiments (requires GPU)
python gpu/gpu_all_exp_v5.py                     # 42-config suite
```

### Generating Figures

```bash
# After running experiments
python simulation/fig2_plot.py   # Key results figures
python simulation/fig3_plot.py   # Ablation analysis
```

### Results Flow

```
exp*.py scripts
    |
    v
simulation/results/   (simulation)
results/*.json        (GPU experiments)
    |
    v
fig*.py plot scripts
    |
    v
docs/gpu_audit/gpu_audit_report.md
```

### Key Results Files

| File                              | Description                           |
|-----------------------------------|---------------------------------------|
| `results/all_exp_v5_summary.json` | Aggregated 42-config results          |
| `results/head_analysis.json`       | Per-head compression analysis (256 heads) |
| `simulation/results/baselines_data.json` | Baseline method comparisons    |
| `simulation/results/baselines_report.md` | Baseline evaluation report     |

## Attention Contract System

The five contracts form a **tiered dispatch pipeline**:

```
Incoming KV Block
       |
       v
  [ExactLocal?]
  yes -> Serve FP16 from local GPU
   no |
       v
  [SketchLocal?]
  yes -> Coreset + SVD + INT4
   no |
       v
  [RemoteExact?]
  yes -> Fetch from remote storage
   no |
       v
  [Rehydrate?]
  yes -> Upgrade compressed to FP16
   no |
       v
     Drop
```

The **Serial Cascade** scheduler dispatches blocks through this pipeline, achieving 128-255x speedup at 0.22%% relative error.
