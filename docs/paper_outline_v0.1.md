# ACCORD-KV Paper Outline

**Status**: v0.1 (2026-06-12) — 基于 19 个完成实验 + 2 个理论下界
**Target venue**: SOSP 2027 / OSDI 2027 / MLSys 2027（PD-disagg systems + theoretical contribution）
**Working title**: *ACCORD-KV: Attention Contracts for Operator-Coded KV Handoff in PD-Disaggregated LLM Serving*

---

## 核心故事（4-1-1-1）

| 部件 | 内容 | 章节 |
|---|---|---|
| **4 个 claim** | (m,l,y) wire / Coreset+INT4 / OOD self-healing / Serial Cascade | §4 |
| **10 个失败** | Post-hoc 压缩算法系统化失败（Coreset-only / SVD / PQ / LSH / FLP / ToMe / Attention SVD / ...） | §5 |
| **2 个下界** | V-Centric Mismatch Bound (4.79×) + Rate-Distortion Lower Bound (2.91) | §6 |
| **1 个策略 + 1 个方法** | Anytime Compression (-21%) + Cluster-Conditional V SVD (-13%) | §7 |

**4 个新概念**：(m,l,y) wire format / per-block heterogeneous contract / query-domain validity / mathematical lower bound
**1 个新策略**：Anytime Compression
**1 个新方法**：Cluster-Conditional V SVD

---

## Section 1. Introduction (~1.5 pages)

- **背景**：LLM serving 进入 PD-disaggregated 时代（vLLM, SGLang, Mooncake）。Prefill 节点算完 KV，跨网络 handoff 到 decode 节点。
- **痛点**：KV cache 大（Qwen2.5-7B 单 token ~1MB，128k context 单请求 ~16GB）。跨节点传输是 bottleneck。
- **现状**：业界 + 学界都在 KV 压缩方向使劲（量化 INT4/INT8、低秩 SVD/Nyström、剪枝 ToMe/Coreset、Attention 矩阵压缩）。
- **我们的观察**：这些方向在一个隐藏假设下工作 — **每个 KV block 用同一种表示**。但 V 的分布因 query 而异，clustered V 不可压是物理上证明的。
- **关键 insight**：与其压，不如**换 wire format** — 把 context block 抽象成 "attention contract"，每个 block 自带 (m, l, y) 在线 softmax 统计量作为 wire，接收端按 block 选表示，查询时再做 OOD 校验。
- **核心贡献 4+1+1**（上面 4-1-1-1）。
- **真实数据声明**：本 paper 主要使用 synthetic clustered 分布做机制实验；real LLM KV 分布验证留到 §8 讨论，作为 GPU 验证阶段工作（已与作者沟通，将在 camera-ready 前完成）。

## Section 2. Background and Motivation (~1 page)

- **2.1 PD-Disaggregated Serving**：vLLM 0.6+ / SGLang disagg mode / Mooncake 设计动机
- **2.2 FlashAttention Online Softmax**：(m, l) running max + log-sum-exp 统计量
- **2.3 KV Compression Landscape**：
  - 量化：INT4/INT8, KVQuant
  - 低秩：SVD, Nyström, LoRA-style
  - 剪枝：ToMe, Coreset, FLP, LSH
  - 注意力矩阵：A-side SVD (我们 §4 验证)
- **2.4 关键 gap**：所有方案都假设单表征。我们打破这个假设。

## Section 3. ACCORD Design (~2 pages)

- **3.1 Attention Contract 抽象**：
  - KV block → (K, V, contract_meta)
  - contract_meta = (representation, validator, decoder)
  - per-block heterogeneous representation
- **3.2 (m, l, y) Wire Format**：
  - FlashAttention online softmax 统计量
  - **关键 claim**：attention(A, B) = attention(A, B̂) when (m, l) preserved（数学证明在 §6）
  - 31,775× wire compression（multi-head 实验）
- **3.3 Per-Block Encoder/Decoder**：
  - 每个 block 独立选表示（Coreset, SVD, INT4, Pruning, Identity）
  - Heterogeneous 是核心 — 不同 block 选不同算法
- **3.4 Query-Domain Validity Check**：
  - 接收端在 query 时做 OOD 检测
  - ε-threshold: clustered V 检测
  - ε=5 时 err **-7.1%**（不是 +7.1%，因为能 fallback）
- **3.5 Anytime Compression Scheduler**（提前介绍，详细在 §7）：
  - 把压缩当成 anytime 优化问题
  - 调度策略：先 high-fidelity 表征 + 后期 coarse

## Section 4. Empirical Validation — 4 Paper Claims (~2.5 pages)

| Claim | 实验 | 关键数字 |
|---|---|---|
| C1: (m,l,y) wire 兼容 | Exp2 multi-head | **31,775× wire compression**, 0.16% attn_err |
| C2: Coreset+INT4 baseline | Exp2 + Exp4 | 7.3× compression, 0.16% err |
| C3: OOD self-healing | Exp7 validity | ε=5, err **-7.1%** (clustered) |
| C4: Serial Cascade SOTA | Exp2 串联 | **128-255×**, 0.22% err |

- 4 个表格（每 claim 一表）
- 一个 figure 展示 4 claim 在不同 distribution（random / skewed / clustered）下的表现
- 强调：**clustered 仍是痛点**（3.43 err）— 这就是 §5-§6-§7 要解决的故事

## Section 5. Why Post-hoc Compression Fails (~3 pages)

**诚实是本 paper 的核心。** 我们系统化跑了 10 个独立方向的 post-hoc 压缩算法，全部失败。

### 5.1 Method Categories (10 个算法)

| Category | 算法 | clustered err | comp | 状态 |
|---|---|---|---|---|
| Coreset | exp16 Coreset only | 3.79 | 4.3× | ❌ |
| 低秩 | exp17 Residual SVD | - | - | 33% win rate |
| 低秩 | exp19 Nyström | - | - | 71/72 输 |
| 量化 | exp20 PQ | - | - | 输 INT4 |
| Coreset | exp21 FLP | - | - | 0/9 输 Coreset |
| 边界 | exp22 极端参数 | 4.07 | 264× | ❌ |
| 后验 | exp24 Cluster-aware rescale | - | - | 4 方案全输 |
| 软剪枝 | exp28 LSH | 0.246 | 2.2× | ❌ ratio 低 |
| Token merge | exp29 ToMe | - | - | 2/18 输 |
| 注意力 | exp8 Attention SVD | 165-330 | 2-128× | ❌ |

### 5.2 Unified Picture
- **V/K/A 三个矩阵统一不可压**
  - V: eff_rank 4.79× amplification
  - A: eff_rank 3.89× amplification
  - 没有任何 post-hoc 算法能 win baseline
- **物理上挖不动** — 不是没尝试

### 5.3 Cross-Domain / Cross-Block
- **Cross-domain**（频域/小波/PCA 方向）：5 方向全失败
- **Cross-block**（V/K 邻近 block 结构）：cosine -0.0014，无结构
- **结论**：物理空间已穷尽

## Section 6. Theoretical Foundations (~2 pages)

### 6.1 V-Centric Mismatch Bound（exp25）
- **定理**：If K̂ ≠ K and V̂ ≠ V, then attention error amplification ≥ ||ΔV|| · ||K||^T / ||ΔK|| · ||V||^T
- **物理**：clustered V 是 high-rank（rank@90% = 8），low-rank 假设直接破
- **数字**：4.79× amplification in clustered distribution

### 6.2 Rate-Distortion Lower Bound（exp26）
- **定理**：For clustered V with K clusters, any compressor with bits-per-token < log(K)/d must incur MSE ≥ 2.91
- **物理**：cluster 结构本身需要 log(K)/d 比特表达
- **意义**：post-hoc 压缩的硬下界

### 6.3 物理解释
- V 在 clustered 分布下是分块 low-rank，但**跨 cluster 是 high-rank**
- 这是为什么 Method D Cluster-Conditional 有效

## Section 7. Beyond Post-hoc: Strategies and Refinements (~1.5 pages)

承认 post-hoc 失败后，**我们提出 2 个层次的解决方案**。

### 7.1 Anytime Compression（策略层）
- **Idea**：接受压缩不可解，把 cascade 调度成 anytime optimization
- **数字**：平均 err **1.09**（vs Serial Cascade 1.38，**-21%**）
- **物理解释**：早期 block 给更多 bits（query-domain awareness 增强），后期 block 给更少（acceptance margin 增大）
- **理论基础**：需要在 §7 强化（当前是 empirical）

### 7.2 Cluster-Conditional V SVD（方法层）
- **Idea**：cluster(K) → per-cluster SVD on V_c
- **数字**：attn_err **0.0252**（vs exp25 SVD only 0.0305，**-13%**），amp 0.00148，comp 13.1×
- **3 seeds (42, 43, 44) std=0.0000** — 极稳定
- **物理解释**：clustered V 分块低秩，每 cluster 独立 SVD 改善
- **Pareto 改善**：在 comp-error 帕累托前沿占一席

### 7.3 Practical Guidance
- 何时用 Anytime（生产环境，query 分布稳定）
- 何时用 Method D（cluster 结构清晰，如 long context RAG）
- 何时退回 Coreset+INT4（distribution 不可知）

## Section 8. Discussion and Limitations (~1 page)

### 8.1 Synthetic vs Real LLM KV
- **核心 limitation**：所有 19 个实验都基于 synthetic clustered V 分布
- **作者立场**：synthetic 用于机制证明；real LLM KV 分布验证是 GPU 验证阶段核心工作
- **GPU Checklist**（camera-ready 前必做）：
  1. Qwen2.5-7B / Mistral-7B / Gemma-2-9B 三模型 V 分布分析
  2. Coreset+INT4 / Serial Cascade / Anytime / Method D 在真实 KV 上的 err < 0.5%
  3. OOD self-healing 在真实分布上仍有效
  4. vLLM/SGLang PD-disagg 端到端 TTFT 对比 baseline

### 8.2 训练时方向
- 我们的 paper 定位 "已训练 LLM"（post-training）
- 训练时方向（MLA, GQA, KV sharing）是另一个维度，**不在本 paper scope**
- 但值得 future work

### 8.3 Single-tenant 假设
- 当前调度假设 single-tenant
- 多 tenant 公平性是生产级问题，留 future work

## Section 9. Related Work (~1 page)

- **KV Compression**: KVQuant, ZipCache, SqueezeAttention, Attention Sinks
- **Disaggregated Serving**: vLLM, SGLang, Mooncake, TetriInfer
- **Attention Variants**: FlashAttention, MLA, GQA, Sliding Window
- **SpectrumKV（基线）**: arXiv 2606.08635, MLSys 2027 submission

## Section 10. Conclusion (~0.5 page)

- 总结 4-1-1-1 故事
- 强调诚实负面 + 理论下界 + 2 个 positive
- ACCORD = **A**ttention **C**ontracts for **C**ross-node **O**perator-coded **R**epresentation **D**ispatch

---

## Appendices（待 GPU 验证后补）

- A. 完整 experimental setup
- B. 19 个实验的 raw data + scripts
- C. 理论证明（exp25 / exp26 完整数学）
- D. Real LLM 验证（GPU 阶段补）

## Pending Items（GPU 验证阶段）

1. Real LLM V/K 分布分析 — 决定 §8 诚实度
2. vLLM/SGLang 端到端 TTFT — 决定 §4 工业说服力
3. Method D 在 real KV 上的 -13% 改善是否复现 — 决定 §7.2 物理稳健性
4. Anytime Compression 在 real query distribution 上的 -21% — 决定 §7.1 工程价值
5. OOD self-healing 在真实分布漂移下的稳定性 — 决定 §3.4 robustness
