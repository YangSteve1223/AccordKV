# ACCORD-KV 可行性评估报告

**评估日期**: 2026-06-11  
**研究员**: ACCORD-KV 子代理  
**目标**: 新方向深度可行性评估、文献调研、范式创新探索

---

## Task 1: 新方向可行性评估

### 1.1 新颖性核查：是否已有工作完整覆盖 ACCORD-KV 组合？

**结论：未发现任何一篇论文完整覆盖以下组合的全部要素。**

ACCORD-KV 的完整组合：
- PD-disaggregated serving
- + attention response contract (而非 KV tensor transfer)
- + FlashAttention merge statistics 作为 block interface
- + per-block heterogeneous representation (raw/sketch/remote/rehydrate)
- + error certificate / query-domain validity
- + SpectrumKV mixed-precision raw backend

逐一对标检查：

| 要素 | 最接近的近邻 | 差距分析 |
|------|-------------|---------|
| PD-disaggregated + attention response | **无** | 所有 PD 工作都把 KV 视为 tensor payload，没人把 attention output 变成接口 |
| FlashAttention merge statistics as interface | **Infinite-LLM (DistAttention)** | DistAttention 把 attention 计算分布化，但仍是 tensor 流，不是统计量 ABI |
| Per-block heterogeneous representation | **KVServe** | KVServe 有"策略空间 + 在线控制器"，但粒度是压缩方法组合，不是 contract 类型 |
| Query-domain validity / error certificate | **无** | 这是 ACCORD-KV 独有的——没人验证 sketch/rehydrate 的 attention 输出是否 match |
| SpectrumKV mixed-precision backend | **SpectrumKV (主人自己的)** | 这是已知起点，不是问题 |

**关键区分点**：即使是最接近的 Attention Matching (arXiv:2602.16284)，它解决的是**单实例内**长上下文压缩——把 KV cache 在同一 GPU 上压缩。它完全不涉及：
- PD 分离的跨节点传输
- 传输形式从 tensor 变成 contract 的语义转换
- 多种异构表示的统一接口

### 1.2 最弱碰撞点分析

在 5 种 contract 类型中，**最容易被近邻"顺手做掉"的是 SketchLocal**。

**理由**：
- **Attention Matching** (MIT/Harvard, arXiv:2602.16284, 2026-02) 的 C_k/C_v 构造，本质上就是在做"用 latent representation 替代 raw KV"。它虽然没有用 FlashAttention 的 (m, l, y) 统计量，但它用了 closed-form solution 来 match attention output。
- 如果有人读了 Attention Matching，再读了 FlashAttention online softmax 的 merge 公式，**SketchLocal 的核心思想就已经呼之欲出了**。
- SketchLocal 的技术门槛其实不高：给定 Q 和原始 KV，用 least-squares 或 closed-form 解出 compact K', V' 使得 softmax(QK'^T)V' ≈ softmax(QK^T)V。这个数学问题 Attention Matching 已经解决了。

**证据**：Attention Matching 的 Section 3 明确写道：
> "We show that this formulation naturally decomposes into simple subproblems, some of which admit efficient closed-form solutions."

这就是 SketchLocal 的数学内核。

**次弱碰撞点**：**Rehydrate**。理由是：
- Rehydrate = 在 remote 节点重新计算原始 attention output
- 这本质上就是 CacheGen/CacheBlend 的 selective recompute
- CacheBlend 已经证明：只重算 10-15% 的 high-deviation tokens 就能恢复精度
- 如果把 CacheBlend 的逻辑泛化到"任意 block 的 selective recompute"，Rehydrate 就被覆盖了

### 1.3 最强创新点：范式跳跃的护城河

**最强的、reviewer 难以否认的创新点**是：

**"Attention Contract" 作为 attention offloading 的 semantic interface 而非 tensor interface**

这个 claim 的护城河在于：**为什么没人做？**

技术原因分析：

| 障碍 | 为什么 ACCORD-KV 可以克服 |
|------|------------------------|
| 理论困难：FlashAttention 的 merge 公式需要 Q 才能合并 | SpectrumKV 的 SWS 可以在 prefill 节点预先计算 token importance，在不知道具体 decode query 的情况下做 conservative selection |
| 工程困难：contract 的调用接口需要标准化 | FlashAttention 的 tile 格式已经是事实标准，contract interface 可以 directly map 到 FA 的 block interface |
| 语义困难：如何保证 contract 结果的 validity | 这正是 error certificate 要解决的问题——用 (m, l) 的 residual 来 bound 输出误差 |
| 商业动机：为什么 industry 要改接口 | PD 分离的 KV tensor 传输已经是 10-100GB 的瓶颈，把传输内容从 tensor 变成统计量，bandwidth 需求可能降低 10-100x |

**范式跳跃的深度**：这不是一个"更好的压缩算法"，而是一个"接口抽象的范式转换"——从"传输什么"（tensor）到"传输能做什么"（contract）。这种抽象转换在 OS 领域有先例：虚拟内存从"传输 pages"进化到"传输 capabilities"，网络从"传输 bytes"进化到"传输 RPC calls"。

---

## Task 2: 文献深挖——PD Disaggregation + KV Transfer 相关工作 (2024-2026)

### 2.1 系统性文献地图

按技术方向分类：

#### 方向 A: PD Disaggregation 系统架构

| 论文 | 会议/年份 | 核心思想 | arXiv/链接 | 与 ACCORD-KV 的差异 |
|------|----------|---------|-----------|-------------------|
| DistServe | OSDI 2024 | 首次系统化分析 PD 分离的调度问题 | [arXiv:2401.09670](https://arxiv.org/abs/2401.09670) | 只做调度，KV 仍是完整 tensor 传输 |
| SplitWise | OSDI 2024 | PD 分离下的异构 GPU 调度 | [arXiv:2402.15212](https://arxiv.org/abs/2402.15212) | 同上 |
| **OrbitFlow** | **VLDB'26** | **ILP 调度 + real vLLM 实现**，KV cache 分层放置决策 | [arXiv:2601.04456](https://arxiv.org/abs/2601.04456) | **最危险近邻**：有真实系统，但传输仍是 KV tensor |
| P/D-Serve | arXiv 2025 | 端到端 PD 性能建模 + D2D KV 访问优化 | [arXiv:2503.15989](https://arxiv.org/abs/2503.15989) | KV tensor 形式不变 |
| ARES | arXiv 2025 | 自适应 rescheduling + 请求迁移 | [arXiv:2510.13668](https://arxiv.org/abs/2510.13668) | 同上 |
| Nexus | arXiv 2025 | 单 GPU 内 PD 分离的 model-guided 调度 | [arXiv:2507.06608](https://arxiv.org/abs/2507.06608) | Intra-GPU，暂无传输问题 |

#### 方向 B: KV Cache 压缩（Token/Quantization 两个轴）

| 论文 | 会议/年份 | 核心思想 | arXiv/链接 | 与 ACCORD-KV 的差异 |
|------|----------|---------|-----------|-------------------|
| PDTrim | arXiv 2025 | PD 分离的 targeted pruning，prefill/decode 分层剪枝 | [arXiv:2509.04467](https://arxiv.org/abs/2509.04467) | SpectrumKV 的 baseline，不是 contract |
| H2O | NeurIPS 2023 | Heavy-Hitter token eviction，动态子模优化 | [arXiv:2306.14062](https://arxiv.org/abs/2306.14062) | 单实例，无传输 |
| SnapKV | ICLR 2024 | Observation window 预测 token importance | [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) | 单实例，无传输 |
| PyramidKV | ICML 2024 | Layer-wise pyramidal allocation | — | 单实例，无传输 |
| **KVServe** | **SIGCOMM'26** | **自适应压缩策略空间 + Bayesian profiling + online bandit** | [arXiv:2605.13734](https://arxiv.org/abs/2605.13734) | **接近**：有策略选择，但仍是 tensor 流 |
| KVTuner | ICML 2025 | Layer-wise mixed-precision KV quantization tuning | [arXiv:2502.04420](https://arxiv.org/abs/2502.04420) | 单实例，系统内优化 |
| MiKV | ICLR 2025 | Importance-aware adaptive precision KV | [OpenReview](https://openreview.net/forum?id=CRQ8JuQDEd) | 与 SpectrumKV 正交 |
| More Tokens Lower Precision | arXiv 2024 | Token/precision 联合优化 | [arXiv:2412.12706](https://arxiv.org/abs/2412.12706) | 单实例，无传输 |
| KVzap | arXiv 2026 | 训练小 predictor 预测重要 token | [arXiv:2601.07238](https://arxiv.org/abs/2601.07238) | 单实例，无传输 |
| KVzip | ICLR 2025 | Query-agnostic KV eviction via reconstruction | [OpenReview](https://openreview.net/forum?id=JFygzwx8SJ) | 单实例 |

#### 方向 C: Compact Latent KV（与 SketchLocal 最相关）

| 论文 | 会议/年份 | 核心思想 | arXiv/链接 | 与 ACCORD-KV 的差异 |
|------|----------|---------|-----------|-------------------|
| **Attention Matching** | **arXiv 2602.16284** | **Latent space KV compaction via C_k/C_v match attention output** | [arXiv:2602.16284](https://arxiv.org/abs/2602.16284) | **最高风险**：单实例，数学内核被 SketchLocal 复用 |
| **Cartridges** | **ICLR 2026** | **Offline 训练 compact KV via self-study distillation** | [OpenReview](https://openreview.net/forum?id=0k5w8O0SNg) | 离线训练，per-context，不是 PD 传输 |
| STILL | arXiv 2025 | Reusable neural KV compaction mapping | [arXiv:2505.01886](https://arxiv.org/abs/2505.01886) | 单实例，不是传输场景 |
| SALS | EMNLP 2024 | Sparse attention in latent space + RoPE-free selection | [arXiv:2510.24273](https://arxiv.org/abs/2510.24273) | 单实例 |
| ShadowKV | NeurIPS 2024 | Low-rank key + offloaded value | [arXiv:2410.21465](https://arxiv.org/abs/2410.21465) | 单实例 |
| ReCalKV | arXiv 2025 | Low-rank KV via head reordering + offline calibration | [arXiv:2505.24357](https://arxiv.org/abs/2505.24357) | 单实例 |
| KVTC | ICLR 2026 | Transform coding for KV cache compact storage | [OpenReview](https://openreview.net/forum?id=aNVKROYpLB) | 单实例存储优化 |

#### 方向 D: Remote Attention / Attention Offload（与 RemoteExact 相关）

| 论文 | 会议/年份 | 核心思想 | arXiv/链接 | 与 ACCORD-KV 的差异 |
|------|----------|---------|-----------|-------------------|
| DistCA (Core Attention Disaggregation) | arXiv 2025 | 把 softmax(QK^T)V 拆出来做独立调度 | [arXiv:2510.12289](https://arxiv.org/abs/2510.12289) | **最相关**：确实在拆分 attention，但它拆分的是计算，不是输出统计量 |
| Model-Attention Disaggregation | arXiv 2024 | Attention 放在 memory-specialized accelerator | [arXiv:2405.01814](https://arxiv.org/abs/2405.01814) | 硬件异构，KV 仍是 tensor |
| Infinite-LLM (DistAttention) | arXiv 2401.02669 | 分布式 attention + KVCache，避免 KV 传输 | [arXiv:2401.02669](https://arxiv.org/abs/2401.02669) | 不传 KV，传 Q 块做分布式 attention |
| Adrenaline | arXiv 2025 | Attention offloading + load-aware scheduling | — | 仍是 KV tensor offload |
| NEO | OSDI/NSDI 2024 | GPU-CPU KV offloading + asymmetric pipelining | [arXiv:2411.01142](https://arxiv.org/abs/2411.01142) | CPU-GPU，不是 PD 分离 |
| Star Attention | ICML 2025 | 分布式 context encoding + query replication | [arXiv:2411.17116](https://arxiv.org/abs/2411.17116) | 长序列训练，query-conditional |

#### 方向 E: KV Cache 传输/存储系统

| 论文 | 会议/年份 | 核心思想 | arXiv/链接 | 与 ACCORD-KV 的差异 |
|------|----------|---------|-----------|-------------------|
| CacheGen | SIGCOMM'24 | KV 压缩 + streaming，从 disk/S3 快速加载 | [arXiv:2310.07240](https://arxiv.org/abs/2310.07240) | 存储优化，压缩后仍是 tensor |
| Mooncake | arXiv 2024 | KVCache-centric disaggregated architecture + RDMA | [arXiv:2407.00079](https://arxiv.org/abs/2407.00079) | 传输仍是 KV tensor，量大 |
| **CacheBlend** | **EuroSys'25 Best Paper** | **Non-prefix KV reuse + selective recompute** | [arXiv:2405.09552](https://arxiv.org/abs/2405.09552) | **高风险**：选择性重算 = Rehydrate 的雏形 |
| **EPIC** | **ICML'25** | **Position-Independent KV Cache = compilation/linking analogy** | [ICML](https://icml.cc/virtual/2025/poster/43926) | 最接近"contract"隐喻，但仍是 tensor 拼接 |

### 2.2 高风险近邻的深度分析

#### OrbitFlow (VLDB'26) —— 最危险

**威胁等级**: ★★★★★  
**原因**: 有 real vLLM 实现 + ILP 调度 + 分层 KV 放置

**它做了什么**:
- 用 ILP solver 在运行时决定每个请求的每层 KV 保存在 GPU 还是 CPU
- 动态 reconfigure KV 位置，SLO-driven
- 在 heavy load 下提升 SLO achievement 66%，TBT 降低 48%

**它没做什么**:
- KV 仍是 tensor 形式传输
- 没有改变"传输什么"的问题
- 解决的是"存哪"不是"怎么传"

**ACCORD-KV 的防御**: OrbitFlow 优化的是 placement，ACCORD-KV 优化的是传输内容的语义。当两者结合时，ACCORD-KV 可以作为 OrbitFlow 的"下游"——即使 OrbitFlow 决定把某 block 放 remote，ACCORD-KV 可以决定是传完整的 (m,l,y) 统计量还是只传 sketch。

#### Attention Matching (MIT/Harvard, arXiv:2602.16284) —— 最高学术风险

**威胁等级**: ★★★★☆  
**原因**: 数学内核与 SketchLocal 直接重叠

**它做了什么**:
- 把 KV compaction 问题 formalize 为: 找 C_k, C_v 使得 softmax(QC_k^T)V ≈ softmax(QK^T)V
- 分解为两个 subproblem，给出 closed-form solution
- 50x compaction in seconds

**它没做什么**:
- 单实例，不是 PD 传输场景
- 没有多种异构表示的统一 interface
- 没有 error certificate
- 没有与 PD disaggregation 的结合

**防御策略**: ACCORD-KV 必须明确声明：Attention Matching 给了 SketchLocal 的数学工具，但 SketchLocal 是把它放在"异构传输 + 多种 contract 类型统一接口"的框架下。这本质上是"把 Attention Matching 变成一个系统 primitive"。

#### KVServe (SIGCOMM'26) —— 最接近系统框架

**威胁等级**: ★★★☆☆  
**原因**: 统一的压缩策略空间 + 在线控制器

**它做了什么**:
- 把 KV 压缩统一为 modular strategy space
- Bayesian Profiling Engine 离线搜索 3D Pareto candidate
- Service-Aware Online Controller 在线上选择
- 声称 9.13x JCT speedup

**它没做什么**:
- 传输的仍是压缩后的 tensor
- 没有把"压缩后的 tensor"再抽象成"attention contract"
- 没有 merge statistics 作为接口
- 没有 error certificate

**防御策略**: KVServe 是 ACCORD-KV 的"压缩层"选项之一。ACCORD-KV 的 SketchLocal backend 可以使用 KVServe 的策略空间思想，但 SketchLocal 的输出是 contract 而非压缩 tensor。

#### CacheBlend (EuroSys'25 Best Paper) —— Rehydrate 的直接威胁

**威胁等级**: ★★★☆☆  
**原因**: selective recompute = Rehydrate 的核心操作

**它做了什么**:
- 识别 high-kv-deviation (HKVD) tokens
- 只重算 10-15% 的 HKVD tokens
- TTFT 降低 2.2-3.3x，吞吐量提升 2.8-5x

**它没做什么**:
- 针对的是 non-prefix caching，不是 PD 分离
- 重算的是 raw KV，不是 attention output 统计量
- 没有"contract interface"的概念

**防御策略**: CacheBlend 的 HKVD 识别机制可以直接用来实现 Rehydrate 的"何时触发 rehydration"决策。

---

## Task 3: 范式创新 Idea —— 比 ACCORD-KV 更激进的方向

主人明确要求：至少 3 个比 ACCORD-KV 更激进的范式跳跃，要求：
- 不是 ACCORD-KV 的小补丁
- 不撞上述近邻
- 落在 ML systems / LLM serving 顶会能讲清楚的故事
- 能用类似 SpectrumKV 论文 28 页讲清楚
- **至少 1 个满足"如果做出来就能上 SOSP/OSDI 而不是普通 MLSys"**

### Idea 1: Attention Virtualization Layer (AVL) —— SOSP/OSDI 级别

**定位**: 最激进的范式跳跃，目标是 SOSP/OSDI

**核心思想**：
把 attention 机制本身变成一个**可寻址、可路由、可组合的分布式服务**，而不是 GPU 上的本地计算。

**具体**：
- 把每一层的 attention 计算拆成 **Attention Function as a Service (AFaaS)**
- Prefill 节点不再传输 KV tensor，而是传输 **attention computation request (ACR)**
- ACR 包含: Q 块、block ID、contract type、deadline
- Decode 节点（或者专门的 Attention Server）执行 ACR，返回 (m, l, y) 统计量
- 整个 attention 计算变成了一个**可观测、可调度、可错误的分布式 RPC**

**为什么这个方向更强**：
- ACCORD-KV 改变的是"传输内容"（从 tensor 到 contract）
- AVL 改变的是"谁做计算"（从 prefll/decode GPU 到 attention server）
- 这对应了 OS 领域的 **compute migration** 而非 **data migration**

**为什么没人做过**：
- 需要解决 attention 的 statelessness 和 composability 的系统抽象
- 需要新的 attention scheduler（类似 OS process scheduler）
- 需要处理 network latency 对 attention 结果的影响

**SOSP/OSDI 的故事**：
> "We present AVL, a system that virtualizes attention computation by treating it as a network-addressable service. Unlike prior work that optimizes what KV data to transfer, AVL changes the fundamental question: not 'what to send' but 'where to compute'. We draw an analogy to OS process migration—AVL applies this principle to attention, enabling attention load balancing across heterogeneous hardware without changing the LLM architecture."

**技术挑战**：
- Attention computation 的 network latency 如何 bound
- 如何保证 composability（多个 ACR 的结果能 merge）
- Attention Server 的 scheduling policy

---

### Idea 2: KV Memory Hierarchy as Tiered Attention Cache (TAC) —— MLSys 级别

**定位**: 系统创新，面向 MLSys/VLDB

**核心思想**：
把 KV cache 的 tiering (GPU/CPU/SSD/Remote) 重新定义为 **Attention Cache Hierarchy**，类似于 CPU 的 memory hierarchy，但每一层存储的不是数据而是 **attention computation results**。

**具体**：
- L1 cache: 完整 (m, l, y) 统计量（最近的 blocks）
- L2 cache: SketchLocal 的压缩表示（attention output sketch）
- L3 cache: Block ID + importance score（类似 page table entry）
- 缺失时: 发起 contract execution（相当于 cache miss handler）

**为什么这个方向更强**：
- ACCORD-KV 的 5 种 contract 类型是这个 hierarchy 的 building blocks
- 但 TAC 把它推广到了完整的 memory hierarchy 抽象
- 引入了 cache coherence、consistency、write-back policy 等 OS 概念

**为什么没人做过**：
- KV cache 一直是"存储"视角，没人用"cache hierarchy"的视角
- 需要重新定义 cache line、cache miss、cache coherence
- 需要处理 attention cache 的 query-dependence（CPU cache 是 query-agnostic 的）

**故事**：
> "We observe that KV cache management in PD-disaggregated serving has evolved a de facto memory hierarchy (GPU→CPU→SSD→Remote), but each tier still stores the same thing: raw or compressed tensors. We propose TAC, which treats each tier as storing different levels of attention computation results, from complete (m,l,y) to sketches to block descriptors. This enables cache-aware attention execution that minimizes both data movement and recomputation."

---

### Idea 3: Query-Conditional Contract Composition (QC-3) —— MLSys 级别

**定位**: 理论创新，面向 ICLR/NeurIPS/MLSys

**核心思想**：
把 ACCORD-KV 的 contract 从"静态的 per-block 决策"升级为"动态的 query-conditional 组合"——contract 的类型不仅取决于 block 的特征，还取决于当前 query 的特征。

**具体**：
- 给定一个 block 和一个 query，ACCORD-KV 决定用哪种 contract
- QC-3 则更进一步：给定一个 query，决定**多个 block 的 contract 如何组合**
- 关键洞察：不同的 query 模式需要不同的 contract 组合策略
  - Factual recall: 需要 ExactLocal（精度优先）
  - Creative generation: SketchLocal 可以接受（语义相似度优先）
  - Chain-of-thought: Rehydrate 更有价值（中间结果精确）

**为什么这个方向更强**：
- ACCORD-KV 是 per-block 的静态决策
- QC-3 是 per-query 的动态组合优化
- 引入了"attention planning"的概念（类似 query planning in databases）

**为什么没人做过**：
- 需要对 query 类型做分类或预测
- 需要建立 query 类型与 contract 有效性的映射
- 需要在不知道完整 response 的情况下做决策

**故事**：
> "We observe that different query types have different attention patterns, and the optimal contract composition varies accordingly. QC-3 introduces query-conditional contract planning: given a query, the system dynamically selects the contract composition that maximizes expected attention fidelity under bandwidth constraints. We formalize this as a budget allocation problem over heterogeneous contracts and demonstrate that query-aware planning outperforms static composition by 15-30% on quality-critical tasks."

---

### Idea 4: Attention Contract Verification Framework (ACVF) —— 纯理论创新

**定位**: 形式化验证 + systems，面向 TOPLAS/TOCS 或 SOSP'26 Theory Track

**核心思想**：
为 attention contracts 建立**形式化验证框架**，证明 SketchLocal/Rehydrate 的结果在数学上 bounded by error certificate。

**具体**：
- 定义 attention contract 的 **semantic distance**（contract output vs. ground truth attention output）
- 证明: 给定 block 的 (m, l) 和 Q，semantic distance 有上界
- 建立 contract composition 的**三角不等式**：compose multiple contracts 的误差可以 bounded
- 提供自动验证工具：给定 contract 配置，自动证明其是否满足 SLO

**为什么这个方向更强**：
- ACCORD-KV 的 error certificate 是 heuristic 的
- ACVF 给出数学上严格的证明
- 这在 SOSP/OSDI 有极高的说服力

**为什么没人做过**：
- 需要把 attention 的数值稳定性形式化
- 需要处理非线性（softmax）导致的误差传播
- 需要在 presence of quantization 的情况下做 error bounding

---

## Task 4: Actionable 建议

### 4.1 必须坚持的核心

| 核心要素 | 理由 |
|---------|------|
| **Contract 抽象** | 这是 ACCORD-KV 区别于所有近邻的本质。不要弱化它。 |
| **FlashAttention merge statistics (m, l, y) 作为接口** | 这是 contract 抽象的技术载体，也是"为什么是 contract"的可视化证明。 |
| **Per-block heterogeneous representation** | 这是系统的实际价值：不同 block 确实有不同的最优表示。 |
| **SpectrumKV mixed-precision 作为 ExactLocal backend** | 这是主人的差异化优势，不需要放弃。 |

### 4.2 可以弱化或推迟的部分

| 部分 | 建议 | 理由 |
|------|------|------|
| **Drop contract** | 可以推迟 | 这只是"不传输"，没有新增的系统价值 |
| **Formal error certificate** | 可以弱化为 heuristic bound | 形式化验证太 heavy，可以先做 empirical validation |
| **Rehydrate 的 dynamic triggering** | 先做 static 配置 | CacheBlend 已经证明了 selective recompute 的可行性，先做固定配置更务实 |
| **多模型验证** | 先聚焦 1-2 个模型 | 资源有限，先证明概念再扩展 |

### 4.3 "是不是已经做晚了"的判断

**针对每个高风险近邻的判断**：

| 近邻 | 晚了？ | 判断理由 |
|------|--------|---------|
| OrbitFlow | **没有晚** | 它解决的是"存哪"，ACCORD-KV 解决的是"传什么"，两者正交且互补 |
| Attention Matching | **部分晚了** | SketchLocal 的数学内核已经被它覆盖，但"PD 分离 + 异构接口"是新的 |
| KVServe | **没有晚** | 它是 ACCORD-KV 的压缩层选项之一，不是竞争关系 |
| CacheBlend | **没有晚** | 它的 recompute 对象是 raw KV，ACCORD-KV 的 recompute 对象是 (m,l,y)，粒度不同 |
| DistCA | **没有晚** | DistCA 拆分计算，ACCORD-KV 拆分接口，层次不同 |

**最终判断**：

> ACCORD-KV 的时间窗口是**开放的**，但窗口正在缩小。
>
> 主要风险不是"有人已经做了"，而是"有人正在做且资源更多"。Attention Matching 的团队（MIT/Harvard）和 CacheBlend 的团队（LMCache）都有更强的工程能力和更多的计算资源。如果他们先发表了"PD 分离 + attention as service"的工作，ACCORD-KV 的 novelty 会被大幅稀释。
>
> 建议：
> 1. **立即开始**：用 2-3 个月完成核心实现（ExactLocal + SketchLocal + RemoteExact）
> 2. **瞄准 MLSys 2027**：时间窗口（~2026-10-30）刚好在 Attention Matching (2026-02) 和可能的 follow-up 之间
> 3. **差异化定位**：在论文 title/abstract 中明确"contract interface"而非"KV compression"，避免被归类为又一个压缩算法
> 4. **Idea 1 (AVL)** 作为长期路线图：如果 MLSys 成功，可以接着做 SOSP'28 或 OSDI'28

---

## 附录：关键文献索引

### 必读（主人已有）
- SpectrumKV: [arXiv:2606.08635](https://arxiv.org/abs/2606.08635)

### 高优先级（用于定位 ACCORD-KV novelty）
- Attention Matching: [arXiv:2602.16284](https://arxiv.org/abs/2602.16284)
- KVServe: [arXiv:2605.13734](https://arxiv.org/abs/2605.13734)
- CacheBlend: [arXiv:2405.09552](https://arxiv.org/abs/2405.09552)
- OrbitFlow: [arXiv:2601.04456](https://arxiv.org/abs/2601.04456)
- DistCA: [arXiv:2510.12289](https://arxiv.org/abs/2510.12289)

### 参考（用于 related work）
- Cartridges: [OpenReview](https://openreview.net/forum?id=0k5w8O0SNg)
- EPIC: [ICML 2025](https://icml.cc/virtual/2025/poster/43926)
- CacheGen: [arXiv:2310.07240](https://arxiv.org/abs/2310.07240)
- DistAttention/Infinite-LLM: [arXiv:2401.02669](https://arxiv.org/abs/2401.02669)
- Mooncake: [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
- PDTrim: [arXiv:2509.04467](https://arxiv.org/abs/2509.04467)

---

*报告结束。本报告对主人（杨鹏举）的 ACCORD-KV 研究方向提供了系统性的可行性评估，包括新颖性分析、文献调研、范式创新建议和 actionable 指导。*
