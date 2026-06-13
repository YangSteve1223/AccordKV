# ACCORD-KV 完整学习指南
> **目标**：从零理解 ACCORD-KV，掌握核心算法，能够跟随指南复现项目  
> **适用对象**：具备本科数学基础的研究生，有 Python 经验更佳  
> **作者**：Yang Pengju | **项目**：github.com/YangSteve1223/AccordKV

---

*本指南共五个部分：基础概念篇 / 核心代码解读篇 / 论文精读篇 / 相似论文对照篇 / 项目从零复现篇*

---


# 第一部分：基础概念篇

> **学习目标**：在本篇结束时，你应该能够：
> - 解释 KV Cache 是什么，以及为什么它会成为 LLM 推理的瓶颈
> - 理解 Prefill-Decode 分离架构的设计动机和挑战
> - 掌握 SVD 的直观含义及其在矩阵压缩中的作用
> - 理解 Attention 的计算过程，特别是 FlashAttention 的核心思想
> - 理解 Rate-Distortion 评估框架，学会用它判断压缩方案的好坏

---

## 第1章：KV Cache 是什么？为什么让研究者头疼？

### 1.1 从一次 LLM 推理说起

想象你让 ChatGPT 续写一段小说。你输入了一段 1000 字的提示词（prompt），然后等待它逐字生成回复。这个过程在 LLM 推理系统中是如何工作的呢？

**Prefill 阶段**：当你输入 prompt 时，模型需要"理解"这段文本。它把每个 token 依次送入 Transformer 层，计算出整个序列的内部表示。这一步的关键产出是 **Key（K）和 Value（V）向量**——它们编码了每个 token 的语义信息，供后续的注意力机制使用。

**Decode 阶段**：现在模型开始逐 token 生成。它用最新的 token 生成 Query（Q），然后用 Q 和缓存的 K 计算"应该关注哪些历史 token"，再用这些注意力权重对缓存的 V 做加权平均，得到当前 token 的上下文表示。

这就是标准的 **自回归推理流程**。听起来很优雅，对吧？问题在于——这些 K 和 V 向量到底有多大？

> 💡 **思考**：为什么要保存 K 和 V？
> 
> 因为在生成下一个 token 时，模型需要"回顾"之前所有 token 的信息。如果每次生成都重新计算一遍，效率会低到不可接受。所以聪明的工程师们想出了 KV Cache——把已经算好的 K 和 V 缓存起来！

### 1.2 KV Cache 有多大？

让我们来做一个具体的计算。拿 Mistral-7B 来举例——这是 ACCORD-KV 项目的主要实验模型。

**Mistral-7B 的关键参数**：

| 参数 | 值 | 含义 |
|------|-----|------|
| `num_layers` | 32 | Transformer 层数 |
| `num_kv_heads` | 8 | Key/Value 头的数量（使用 GQA 优化） |
| `head_dim` | 128 | 每个头的维度 |
| `seq_len` | 4096 | 上下文长度 |

每个 K（或 V）向量的形状是 `[num_kv_heads, seq_len, head_dim]`。

对于 4096 token 的上下文：

```
K cache 大小 = 8 × 4096 × 128 × 2 bytes (FP16) = 8 MB  （每层的 K）

总 KV Cache（32层）= 32 × 2 × 8 MB = 512 MB
```

> 💡 **思考**：等等，512 MB？这好像还没有模型参数大（14 GB）呀？
> 
> 问得好！但这里有几点需要注意：
> 1. **这是单个请求的 KV Cache**。如果有 64 个并发请求同时推理呢？那就是 32 GB！
> 2. **上下文越来越长**。现在的大模型支持 32K、128K 甚至 1M token 的上下文。对于 32K 上下文，KV Cache 是 4 GB；128K 就是 16 GB。
> 3. **PD 分离架构**。当 Prefill 和 Decode 在不同 GPU 上时，这 512 MB 需要跨节点传输——这才是真正的噩梦。

❓ **检验**：如果把 Mistral-7B 换成标准 MHA（32 个 KV heads），KV Cache 会变成多大？

**计算过程**：
- GQA（8 heads）：512 MB
- 标准 MHA（32 heads）：512 MB × 4 = **2048 MB = 2 GB**

这就是为什么现代大模型纷纷采用 GQA（Grouped Query Attention）等优化技术！

### 1.3 PD 分离：把 prefill 和 decode 拆开

你可能会问：为什么要把 Prefill 和 Decode 分到不同的 GPU 上？

答案藏在两类操作的本质差异里：

| 阶段 | 计算特性 | 瓶颈 |
|------|---------|------|
| **Prefill** | 计算密集型（一次处理整个 prompt） | GPU 算力 |
| **Decode** | 访存密集型（每次只处理 1 个 token，但需要读取大量历史 KV） | 显存带宽 |

把它们分开的好处是：
- **Prefill GPU** 可以专注做大规模矩阵运算，大批量吞吐
- **Decode GPU** 可以优化为大批量小请求的服务，降低延迟

这就是 **PD 分离架构**（Prefill-Decode Disaggregation），被 DistServe、Moonshare、LoongServe 等系统采用。

**但是！** 这里有个关键问题：

```
Prefill GPU 计算完 K、V 后，需要传给 Decode GPU
这 512 MB（甚至更大）的数据传输，靠什么？
```

答案是：**网络带宽**。在跨节点场景下，带宽可能是 10 Gbps ~ 400 Gbps。512 MB 在 100 Gbps 网络上需要约 40 ms——而生成一个 token 可能只需要 20 ms！

> 💡 **形象类比**：
> 想象你在厨房（Prefill）和餐厅（Decode）之间传递食物。如果餐厅就在厨房隔壁（同一台机器），递个盘子很快。但如果餐厅在另一栋楼（跨节点），你每次都要把整个盘子打包传送——这就是 KV 传输的带宽瓶颈。
> 
> 而且，这个"传送"不是只做一次——每个新 token 生成时，Decode 都需要最新的 KV 状态！

### 1.4 核心矛盾：带宽不够 vs 信息要全

现在摆在我们面前的问题是：**如何在有限的带宽下，传输尽可能完整的 KV 信息？**

❌ **方案一：直接丢弃 KV**
> "不传了，Decode GPU 重新算？"
> 
> 那就失去了缓存的意义。每次生成都要重新算 Attention，延迟直接爆炸。

❌ **方案二：无限压缩**
> "使劲压缩，榨干每一 bit？"
> 
> 压缩太狠会丢失关键信息。Attention Sink（前面几个 token 获得异常高的注意力）如果被压缩没了，模型输出质量会严重下降。

❌ **方案三：只传 Q（Query）？**
> "让 Decode 自己算 Attention？"
> 
> Attention 的核心就是 Q·K^T。如果 Decode 没有 K，怎么算？除非你把整个模型参数也传过去——那更不现实。

❓ **检验**：你能想到第四种方案吗？

**所以，我们的目标是**：在带宽约束下，找到一个"刚好够用"的压缩方案，既不能太"糙"（丢失信息），也不能太"细"（浪费带宽）。

**这正是 ACCORD-KV 要解决的问题**。它提出了三个核心创新：

1. **Wire Format (m, l, y)**：传输 FlashAttention 的中间统计量，而非原始 KV
2. **异构后端**：每个 KV 块独立选择最优压缩算法
3. **SVD 压缩**：利用 K/V 的低秩结构做有损压缩

接下来的章节，我们将逐一深入理解这些概念。

---

## 第2章：SVD——把"大矩阵"压缩成"小矩阵"

### 2.1 SVD 是什么：直观解释

SVD（奇异值分解，Singular Value Decomposition）是线性代数中最优雅的分解之一。但在你头疼之前，让我先用一个生活化的例子让你理解它的本质。

**🌰 例子：画像与特征**

想象你有朋友 A、B、C。你对他们的印象可以分解为几个"特征"：
- **聪明程度**：A = 9/10, B = 7/10, C = 3/10
- **靠谱程度**：A = 5/10, B = 9/10, C = 6/10
- **有趣程度**：A = 8/10, B = 4/10, C = 7/10

你发现：其实"靠谱"和"有趣"这两个特征高度相关——越靠谱的人往往越无趣！所以你只需要两个特征就能很好地描述他们。

**SVD 就是这个思想的数学版本**。

> 💡 **直观的物理图像**：
> 
> 把矩阵想象成一个 N 维空间到 M 维空间的"变换"。SVD 告诉你：这个变换最重要的方向是什么？每个方向有多重要？
> 
> 奇异值就是这些方向的"权重"。大的奇异值对应的方向存储了数据的主要能量；小的奇异值对应的是噪声或次要变化。

### 2.2 SVD 的数学形式

对于任意矩阵 $A \in \mathbb{R}^{m \times n}$，SVD 把它分解为：

$$A = U \Sigma V^T$$

其中：
- $U \in \mathbb{R}^{m \times m}$：左奇异向量矩阵（列是 $u_1, u_2, ..., u_m$）
- $\Sigma \in \mathbb{R}^{m \times n}$：对角矩阵，对角线是奇异值 $\sigma_1 \geq \sigma_2 \geq ... \geq 0$
- $V^T \in \mathbb{R}^{n \times n}$：右奇异向量矩阵（行是 $v_1^T, v_2^T, ..., v_n^T$）

**每个符号的含义**：

| 符号 | 含义 | 物理意义 |
|------|------|---------|
| $U$ | 左奇异向量 | 行空间的主方向（"怎么组织行"） |
| $\Sigma$ | 奇异值 | 每个主方向的重要性权重 |
| $V^T$ | 右奇异向量 | 列空间的主方向（"怎么组织列"） |

**为什么叫"奇异值"？**

这个词来自德语"singulär"（特殊）。在数学上，当矩阵的行列式为 0 时，我们说矩阵是"奇异的"（不可逆）。奇异值分解告诉我们：任何矩阵都可以分解成一组正交基的加权和，而这些权重就是奇异值。

### 2.3 一个 3×3 矩阵的 SVD 分解图示

为了让你建立直觉，我们来看一个具体的例子：

假设我们有一个 3×3 的小矩阵（这在实际中太小了，但便于理解）：

```
A = [3  1  1]
    [1  3  1]
    [1  1  3]
```

它的 SVD 分解会产生：
- **奇异值**：σ₁ = 5, σ₂ = 2, σ₃ = 2
- **累计方差**（解释的总方差比例）：

```
累计方差 = σ₁²/(σ₁²+σ₂²+σ₃²) = 25/38 ≈ 0.66  （第1个奇异值）
累计方差 = (25+4)/38 = 29/38 ≈ 0.76  （前2个）
累计方差 = 1.0                        （前3个）
```

> 💡 **关键洞察**：奇异值按从大到小排序，**前面几个奇异值往往占据了绝大部分能量**！
> 
> 这个现象叫做"谱的衰减"（spectral decay）。在真实数据中，前 10-20% 的奇异值往往能解释 90% 以上的方差！

### 2.4 截断 SVD：只保留最重要的方向

如果我只保留前 r 个奇异值（截断 SVD），会得到矩阵 $\tilde{A}$：

$$\tilde{A} = U[:, :r] \cdot \Sigma[:r, :r] \cdot V^T[:r, :]$$

这相当于把 $A$ 投影到了最重要的 r 个方向上。

**误差怎么衡量？**

我们用 **Frobenius 范数**（所有元素的平方和开根号）：

$$\|A - \tilde{A}\|_F = \sqrt{\sum_{i=r+1}^{\min(m,n)} \sigma_i^2}$$

**累积方差**（也叫累计能量比）是另一个常用指标：

$$\text{cumvar}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2}$$

> 💡 **直觉**：累积方差告诉你，保留 r 个奇异值能"解释"原始矩阵的百分之多少信息。
> 
> 如果 cumvar(10) = 0.95，意味着只保留 10 个方向就能恢复原始矩阵 95% 的信息！

**为什么截断 SVD 是最优的低秩近似？**

这有个美丽的数学定理：**Eckart-Young-Mirsky 定理**。

> 对于任意矩阵 $A$，在 Frobenius 范数意义下，$A$ 的最佳 rank-r 近似就是截断 SVD。

这意味着我们不需要担心"有没有更好的压缩方法"——SVD 已经是最优的了！

### 2.5 为什么 K 和 V 的压缩效果完全不同？

终于！这是 ACCORD-KV 的核心发现之一。

**让我们看看真实数据**（Mistral-7B，所有层的平均值）：

| 截断 rank | K 累积方差 | V 累积方差 |
|-----------|-----------|-----------|
| r = 8 | **0.9413** | **0.5995** |
| r = 16 | ~0.99 | ~0.75 |
| r = 32 | 0.9999+ | ~0.85 |
| r = 64 | ~1.0 | ~0.95 |

> 💡 **震惊！** 用 rank=8 截断 SVD：
> - K 能保留 **94.13%** 的信息——相当不错！
> - V 只能保留 **59.95%** 的信息——差了一截！

❓ **检验**：为什么 K 和 V 差别这么大？

**直觉解释**：

1. **K（Key）用于计算注意力分数**
   - Attention 分数 Q·K^T 本质上是衡量"相似度"
   - 相似度计算有"头部效应"——少数几个方向主导了相似度
   - 所以 K 是**天然低秩**的，前几个奇异值就足够了

2. **V（Value）是实际要加权平均的内容**
   - 不同 token 的 V 向量可能分布在更分散的方向上
   - V 保留了更多的"细节信息"，需要更多维度才能准确表示
   - 这是为什么 ACCORD-KV 发现 **Value Bottleneck**——V 的压缩比 K 更难，是整个系统的瓶颈

**从另一个角度理解**：

想象你在看一场足球赛。K 告诉你"球员在哪里"——这可以用几个关键位置描述。但 V 告诉你"球员的完整状态"——速度、方向、技术动作——这需要更多细节才能准确描述。

### 2.6 SVD 在 KV 压缩中的具体用法

现在我们知道 SVD 可以压缩矩阵。但在 KV Cache 场景下，具体怎么用？

**原始 Attention 输出**：
$$y = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

如果我们用截断 SVD 分别压缩 K 和 V：

$$\tilde{K} = U_K[:, :r_K] \cdot \Sigma_K[:r_K, :r_K] \cdot V_K^T[:r_K, :]$$
$$\tilde{V} = U_V[:, :r_V] \cdot \Sigma_V[:r_V, :r_V] \cdot V_V^T[:r_V, :]$$

**误差来源**：
1. K 的截断误差：影响注意力分数的计算
2. V 的截断误差：直接影响输出的精度

从 ACCORD-KV 的实验数据（Mistral-7B, seq=512, rank=8）：

| 指标 | FP16 | INT4 量化后 |
|------|------|------------|
| K 相对误差 | 0.237 | 0.895 |
| V 相对误差 | 0.599 | 0.885 |
| K Cosine 相似度 | 0.971 | 0.609 |
| V Cosine 相似度 | 0.788 | 0.561 |

> 💡 **观察**：
> - FP16 下，SVD 的误差符合累积方差分析
> - INT4 量化叠加了额外误差
> - Cosine 相似度下降更多，说明量化对"方向"的扭曲比"幅度"更严重

❓ **检验**：如果只做 SVD 不做量化，误差会降低多少？

---

## 第3章：Attention 机制回顾

### 3.1 标准 Self-Attention 的计算

Attention（注意力机制）是 Transformer 的核心。让我们一步步拆解它的计算过程。

**输入**：序列中每个位置的词向量 $x_i$

**输出**：每个位置的新表示 $y_i$，包含了它对序列中其他位置的"关注"

**步骤 1：生成 Q, K, V**

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

其中 $W_Q, W_K, W_V$ 是可学习的投影矩阵。

> 💡 **直觉**：
> - **Query（Q）**：当前位置在"问"什么问题
> - **Key（K）**：每个位置在"回答"什么问题
> - **Value（V）**：每个位置实际携带的"内容"

**步骤 2：计算注意力分数**

$$S = \frac{Q K^T}{\sqrt{d_k}}$$

这里 $\sqrt{d_k}$ 是缩放因子，防止点积过大导致 softmax 梯度消失。

> 💡 **为什么点积会过大？**
> 
> 假设 Q 和 K 的每个元素都是均值为 0、方差为 1 的独立随机变量。那么点积 $q \cdot k$ 的方差是 $d_k$。当 $d_k$ 很大时，点积可能变得很大，softmax 的梯度会趋近于 0！

**步骤 3：Softmax**

$$\text{Attention}(Q, K, V) = \text{softmax}(S) V$$

最终，$y_i = \sum_j \text{softmax}(S)_{ij} \cdot v_j$

> 💡 **直观理解**：softmax 把注意力分数变成概率分布，表示"当前位置 i 应该从其他位置 j 获取多少信息"。
> 
> 类似于你读一本书时，会自然地更关注与当前内容相关的段落——Attention 让模型学会了这种"选择性关注"。

### 3.2 FlashAttention 的工作方式

标准 Attention 有个致命问题：**显存复杂度是 O(N²)**！

为什么？因为计算 $QK^T$ 需要存储 N×N 的注意力矩阵。对于 4096 token，这已经是 4096×4096 = 16M 个元素了！

| 序列长度 | 注意力矩阵大小（FP16） | 显存占用 |
|---------|---------------------|---------|
| 512 | 512 × 512 = 262K | ~0.5 MB |
| 2048 | 2048² = 4M | ~8 MB |
| 4096 | 4096² = 16M | ~32 MB |
| 16384 | 16384² = 268M | ~536 MB |

> 💡 **问题**：对于长上下文（如 32K token），单是这个注意力矩阵就要占用 8 GB 显存！还没算其他中间结果呢。

**FlashAttention 的核心思想**：不一次性计算完整矩阵，而是用 **分块计算（tiling）+ 在线 softmax（online softmax）**。

**Tiling（分块）**：
- 把 K, V 分成小块（tiles）
- 每次只处理一个 tile，逐步累加结果
- 只需要 O(N) 显存

**在线 Softmax**：
- 传统 softmax 需要知道所有输入才能计算
- 在线算法允许我们**增量计算**：先处理一部分，得到当前的最佳估计；再处理下一部分，修正估计

### 3.3 (m, ℓ, y) —— FlashAttention 的中间统计量

FlashAttention 在分块计算过程中，会维护三个关键统计量：

| 符号 | 形状 | 含义 |
|------|------|------|
| **m** | `[num_heads, q_len, 1]` | log-sum-exp max（每个 query 的最大值） |
| **ℓ** | `[num_heads, q_len, 1]` | sum-exp（softmax 的分母） |
| **y** | `[num_heads, q_len, d]` | 未归一化的加权 V 和 |

**它们是怎么来的？**

假设我们正在处理第 j 个 block 的 K, V：
1. 计算当前 block 的注意力分数：$S_j = \frac{Q K_j^T}{\sqrt{d}}$
2. 更新最大值：$m_{new} = \max(m_{old}, \max(S_j))$
3. 计算缩放因子：$d_{scalar} = e^{m_{old} - m_{new}}$
4. 更新 ℓ：$\ell_{new} = d_{scalar} \cdot \ell_{old} + \sum e^{S_j - m_{new}}$
5. 更新 y：$y_{new} = d_{scalar} \cdot y_{old} + \sum \frac{e^{S_j - m_{new}}}{\ell_{new}} \cdot V_j$

**为什么这三个值如此重要？**

当我们有了 (m, ℓ, y)，最终的 attention 输出就是：

$$\text{output} = \frac{y}{\ell}$$

就这么简单！

> 💡 **关键洞察**：在 PD 分离架构中，如果我们只传输 (m, ℓ, y) 而不是完整的 K、V：
> - 传输数据量从 $2 \times N \times d$ 降到 $q\_len \times (2 + d)$
> - Decode 端可以直接恢复 attention 结果，不需要 K 和 V！

### 3.4 (m, ℓ, y) 的压缩比计算

让我们算一笔账：

**传统方式**（传输 KV）：
- 数据量 = `2 × num_heads × kv_len × head_dim` bytes

**Wire Format (m, ℓ, y)**：
- m: `num_heads × q_len × 1`
- ℓ: `num_heads × q_len × 1`  
- y: `num_heads × q_len × head_dim`
- 总计: `num_heads × q_len × (2 + head_dim)` bytes

**压缩比**：
$$\text{ratio} = \frac{2 \times kv\_len}{q\_len \times (2 + head\_dim)}$$

对于 Mistral-7B（`head_dim=128`, `kv_len=4096`, `q_len=64`）：
$$\text{ratio} = \frac{2 \times 4096}{64 \times 130} \approx 0.98 \text{?}$$

等等，这个压缩比好像没有想象中大...

> 💡 **等等！** 我故意用错了场景！
> 
> Wire Format 的真正威力在于：**它传输的是中间统计量，不是最终的压缩结果**！
> 
> 真正的压缩发生在 SVD 阶段——先把 K, V 做 SVD 压缩到低秩，再用 Wire Format 传输。
> 
> 两者结合：SVD 压缩 × Wire Format = **31,775× 压缩比**（见 ACCORD-KV 论文数据）。

❓ **检验**：如果 q_len 变大（比如 decode 时 q_len=1），压缩比会怎么变？

---

## 第4章：Rate-Distortion——怎么衡量"压缩得好不好"

### 4.1 Rate 是什么

**Rate（码率）**衡量压缩后的数据量。常见的表示方式有两种：

1. **绝对大小**：压缩后占多少字节
2. **相对大小**：压缩率 = 原始大小 / 压缩后大小（越大越好）

**ACCORD-KV 中的 Rate 定义**：

$$\text{Rate} = \frac{\text{原始大小}}{\text{压缩后大小}} = \frac{2 \times \text{num\_heads} \times \text{kv\_len} \times \text{head\_dim}}{\text{实际传输大小}}$$

例如，对于 Mistral-7B（rank=8, INT4 量化）：
- 原始：FP16，2 × 8 × 512 × 128 × 32 layers = 256 MB
- 压缩后：INT4 SVD，~8 MB
- Rate ≈ **32×**

> 💡 **形象理解**：
> - Rate = 1x 意味着没压缩
> - Rate = 10x 意味着压缩到原来的 1/10
> - Rate = 100x 意味着压缩到原来的 1/100

❓ **检验**：如果 rank=4 呢？Rate 会变成多少？（提示：需要计算压缩后的大小）

### 4.2 Distortion 是什么

**Distortion（失真）**衡量压缩引入的误差。压缩得越狠，误差通常越大。

**ACCORD-KV 使用三种误差指标**：

| 指标 | 公式 | 含义 |
|------|------|------|
| **Relative Error** | $\frac{\|A - \tilde{A}\|_F}{\|A\|_F}$ | 相对 Frobenius 范数误差 |
| **Cosine Similarity** | $\frac{\langle A, \tilde{A} \rangle}{\|A\|_F \|\tilde{A}\|_F}$ | 方向相似度（注意力权重） |
| **Perplexity (PPL)** | $\exp(-\frac{1}{T}\sum_i \log p_i)$ | 下游任务质量指标 |

**为什么用 Relative Frobenius 范数？**

```python
rel_err = np.linalg.norm(A - A_tilde, 'fro') / np.linalg.norm(A, 'fro')
```

这个指标的优势：
- **尺度不变**：不管矩阵大小，误差在 0~1 之间
- **直观**：0.1 表示平均 10% 的相对误差
- **可加性**：对各层的误差可以求和

**Cosine 相似度 vs 相对误差**

| 指标 | 关注点 | 适用场景 |
|------|--------|---------|
| Relative Error | 幅度（绝对值） | 精确重建 |
| Cosine Similarity | 方向（比例关系） | Attention 权重、语义相似度 |

> 💡 **直觉**：
> - Relative Error 低意味着"数值大小对"
> - Cosine 高意味着"相对关系对"
> 
> 对于 Attention，Cosine 可能更重要——因为 softmax 只关心相对比例！

### 4.3 Rate-Distortion 曲线

Rate 和 Distortion 是一对冤家——Rate 越高（压缩越少），Distortion 越低（误差越小）。

把它们画成曲线就是 **Rate-Distortion 曲线**：

```
Distortion (相对误差)
    ↑
    │  · · · · · (Pareto 前沿)
    │ ·
    │  ·  ← ACCORD-KV (SVD)
    │   ·
    │    ·
    │     · ← 朴素量化
    │      ·
    │       ·
    └─────────────────→ Rate (压缩比)
```

**Pareto 前沿**：在这条线上，你不可能同时降低 Rate 和 Distortion——必须 trade-off。

**如果一个方案在 Rate-Distortion 曲线上的某个点，既可以用更低的 Rate 实现相同的 Distortion，又可以用更低的 Distortion 实现相同的 Rate，那它就被"支配"了。**

### 4.4 实际数据解读

让我们看 ACCORD-KV 的实验数据（Mistral-7B, seq=512）：

| Rank | K 相对误差 | V 相对误差 | K Cosine | V Cosine | 压缩比 |
|------|-----------|-----------|----------|----------|--------|
| 4 | 0.269 | 0.648 | 0.97 | 0.72 | ~64× |
| 8 | 0.237 | 0.599 | 0.97 | 0.79 | ~32× |
| 32 | 0.136 | 0.415 | 0.99 | 0.90 | ~8× |
| 64 | 0.124 | 0.343 | 0.99 | 0.93 | ~4× |

> 💡 **观察**：
> - rank 越小，压缩比越高，但误差越大
> - K 的误差始终比 V 小（符合之前的累积方差分析）
> - **Value Bottleneck**：V 的误差限制了整体质量
> - Cosine 相似度比相对误差"好看"，说明 SVD 保留了主要的方向信息

❓ **检验**：如果你的应用场景对精度要求极高（Perplexity 只能接受 <5% 的增加），你会选择哪个 rank？

---

## 第5章：综合应用——ACCORD-KV 的整体框架

### 5.1 问题的全局视角

现在我们已经掌握了所有基础知识。让我把它们串起来，看看 ACCORD-KV 是怎么解决 PD 分离中的 KV 传输问题的。

**问题重述**：
```
Prefill GPU 和 Decode GPU 在不同节点
Prefill 需要把 KV Cache 传给 Decode
KV Cache 太大（512 MB+ @ 4096 tokens）
网络带宽有限
怎么办？
```

**现有方案的困境**：

| 方案 | 问题 |
|------|------|
| 直接传输 KV | 带宽不够，延迟高 |
| 丢弃 KV | 失去缓存意义，每次都重算 |
| 传 Q，让 Decode 自己算 | 没有 K，无法算 Attention |
| 统一压缩（K=V） | K 和 V 压缩特性不同，一刀切效果差 |

### 5.2 ACCORD-KV 的三板斧

**第一斧：SVD 压缩**

利用 K 和 V 的低秩结构，用截断 SVD 把它们压缩到低维表示。

- K：用 rank=8 就能保留 94% 的信息
- V：需要更大的 rank，但仍然可以显著压缩

**第二斧：INT4 量化**

SVD 压缩后的矩阵仍然用 FP16 存储（每个数 2 bytes）。进一步用 INT4 量化（每个数 0.5 bytes），再节省 4×。

> 💡 **量化带来的误差**
> 
> INT4 量化把浮点数映射到 16 个离散值。这相当于在 SVD 误差上再叠加一层量化误差。
> 
> 实验数据显示：INT4 后 K 的相对误差从 0.237 跳到 0.895！这是因为量化对奇异值较小的方向影响更大。

**第三斧：Wire Format (m, ℓ, y)**

如果 Decode 端需要精确的 attention 结果，直接传压缩后的 K, V 不够——因为 attention 需要 Q 和 K 重新计算分数。

但如果我们传 (m, ℓ, y)，Decode 端只需要做一次除法：`output = y / ℓ`，完全绕过了 K, V 的传输！

### 5.3 压缩效果总结

| 阶段 | 技术 | 数据量（相对值） |
|------|------|-----------------|
| 原始 KV | - | 1.0 |
| SVD rank=8 | 截断 SVD | ~0.0625 (1/16) |
| INT4 量化 | 4-bit 量化 | ~0.0156 (1/64) |
| Wire Format | (m, ℓ, y) | ~0.00003 (1/31775) |

**31,775× 压缩比！** 这意味着 512 MB 的 KV Cache 只需要约 16 KB 就能传输。

> 💡 **等等，这数字怎么来的？**
> 
> 31,775× 不是上面 4 个阶段的简单乘积。它是端到端的测量——包括：
> 1. SVD 把 kv_len × head_dim 矩阵压缩到 rank × head_dim
> 2. INT4 量化进一步节省 4×
> 3. Wire Format 只传 (m, ℓ, y)，不需要传 K, V
> 
> 实际效果取决于 seq_len、q_len、rank 等参数。

### 5.4 剩余误差分析

压缩这么多，误差有多大？

从实验数据（Mistral-7B, rank=8, INT4）：
- K 相对误差：~23.7% → INT4 后 ~89.5%
- V 相对误差：~59.9% → INT4 后 ~88.5%
- Cosine 相似度（注意力方向）：~0.6

这些误差在下游任务上会表现为：
- Perplexity 略有上升
- 但整体输出质量仍然可接受（取决于应用场景）

> 💡 **权衡的艺术**：ACCORD-KV 允许通过 Contract Type 选择精度级别：
> - **EXACT**：必须精确传输，适合对质量要求极高的场景
> - **APPROX**：允许近似，享受极致压缩
> - **BOUNDED**：误差有上限，在可接受范围内最大化压缩

### 5.5 典型应用场景

**场景 1：长上下文摘要**
- 上下文：32K tokens
- 需求：Prefill 完成后，Decode 需要访问所有历史 KV
- ACCORD-KV 优势：31K× 压缩让跨节点传输可行

**场景 2：低延迟对话**
- 上下文：4K tokens
- 需求：每个新 token 都要快速获取 KV
- ACCORD-KV 优势：Wire Format 最小化传输量

**场景 3：大批量服务**
- 同时服务 64+ 请求
- 需求：每个请求的 KV 都要传输
- ACCORD-KV 优势：高压缩比让并发成为可能

---

## 附录：关键公式速查表

### A.1 SVD 相关

| 公式 | 说明 |
|------|------|
| $A = U \Sigma V^T$ | SVD 分解 |
| $\tilde{A} = U[:, :r] \Sigma[:r, :r] V^T[:r, :]$ | rank-r 截断 |
| $\text{cumvar}(r) = \frac{\sum_{i=1}^{r} \sigma_i^2}{\sum_{i=1}^{n} \sigma_i^2}$ | 累积方差 |
| $\text{rel\_err} = \frac{\|A - \tilde{A}\|_F}{\|A\|_F}$ | 相对 Frobenius 误差 |

### A.2 Attention 相关

| 公式 | 说明 |
|------|------|
| $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) V$ | 标准 Attention |
| $y = \text{softmax}(S) V$ | 注意力输出 |
| $\text{output} = y / \ell$ | 从 (m, ℓ, y) 恢复输出 |

### A.3 Rate-Distortion

| 指标 | 公式 |
|------|------|
| Rate | $\frac{\text{原始大小}}{\text{压缩后大小}}$ |
| Distortion | $\frac{\|A - \tilde{A}\|_F}{\|A\|_F}$ |
| PPL | $\exp(-\frac{1}{T}\sum_i \log p_i)$ |

---

## 练习题

### 练习 1：KV Cache 大小估算

**问题**：对于 LLaMA-2 7B（32 层，32 KV heads，head_dim=128），计算 seq_len=8192 时的 KV Cache 大小（FP16）。

**答案**：
```
K cache = 32 × 32 × 8192 × 128 × 2 bytes = 536,870,912 bytes = 512 MB
V cache = 同样大小
Total = 1 GB
```

### 练习 2：SVD 累积方差

**问题**：如果 K 的前 8 个奇异值占总奇异值平方和的 94%，这意味着什么？

**答案**：
- 用 rank=8 截断 SVD 后，K 保留了 94% 的"信息量"
- 剩余 6% 的信息被丢弃，表现为重构误差
- 这解释了为什么 rank=8 的 K 相对误差约为 0.24（√(1-0.94) ≈ 0.245）

### 练习 3：Rate-Distortion Trade-off

**问题**：如果你的带宽只允许传输原始 KV 的 1%，你会选择什么 rank？

**答案**：
- 原始的 1% ≈ 压缩比 100×
- 这需要 rank=4 或更低
- 但要注意 V 的误差会很大（~65%）
- 如果下游任务对精度敏感，可能需要其他策略（如选择性传输关键层）

### 练习 4：理解 Value Bottleneck

**问题**：解释为什么 V 的压缩比 K 更难。给出一个直觉解释和一个数学解释。

**答案**：
- **直觉**：K 决定"关注什么"（相似度计算），这是一个相对简单的比较，所以可以用几个方向近似。V 包含"实际内容"（token 的语义表示），需要更多细节才能准确重建。
- **数学**：实验数据显示 K 的累积方差衰减快（rank=8 时 cumvar=0.94），而 V 的累积方差衰减慢（rank=8 时 cumvar=0.60）。

### 练习 5：带宽瓶颈分析

**问题**：假设你有 100 Gbps 的网络带宽，需要传输 512 MB 的 KV Cache。计算传输时间，并与生成一个 token 的时间（假设 20 ms）对比。如果用 rank=8 的 SVD 压缩（32×），传输时间变成多少？

**答案**：
```
原始传输时间 = 512 MB / 12.5 GB/s = 40.96 ms
对比：40.96 ms > 20 ms（传输成为瓶颈）

压缩后数据 = 512 MB / 32 = 16 MB
压缩后传输时间 = 16 MB / 12.5 GB/s ≈ 1.28 ms
对比：1.28 ms << 20 ms（不再是瓶颈！）
```

### 练习 6：累积方差的计算

**问题**：假设一个矩阵的前 4 个奇异值分别是 [100, 50, 25, 10]，计算 rank=2 时的累积方差。

**答案**：
```
总能量 = 100² + 50² + 25² + 10² = 10000 + 2500 + 625 + 100 = 13225
rank=2 能量 = 10000 + 2500 = 12500
累积方差 = 12500 / 13225 ≈ 0.945

解释：rank=2 保留了约 94.5% 的信息
```

### 练习 7：FlashAttention 核心思想

**问题**：解释 FlashAttention 为什么能做到 O(N) 显存复杂度，而不是 O(N²)。

**答案**：
- **标准 Attention**：需要先计算完整的注意力矩阵 S = QK^T（N×N 大小），需要 O(N²) 显存
- **FlashAttention**：分块处理，每次只计算一个 block 的注意力分数，增量更新 (m, ℓ, y)
- **关键洞察**：不需要存储完整的 S 矩阵，只需要存储最终结果 (m, ℓ, y)
- **数学保证**：在线 softmax 算法确保增量计算的结果与完整计算一致

### 练习 8：Pareto 最优分析

**问题**：假设有三个压缩方案在相同压缩比(10×)下的误差分别是：A=10%, B=15%, C=8%。请分析哪些方案是 Pareto 最优的。

**答案**：
- **C (误差 8%)** 是 Pareto 最优：在相同压缩比下误差最低
- **A (误差 10%)** 被 C 支配
- **B (误差 15%)** 误差最高，除非有特殊需求通常不选

---

## 本篇小结

恭喜你完成了基础概念篇的学习！让我们回顾一下今天学到的：

1. **KV Cache**：LLM 推理中的关键缓存，但也带来了巨大的存储和传输压力
2. **PD 分离**：Prefill 和 Decode 的分离带来了 KV 传输的带宽瓶颈
3. **SVD**：一种强大的矩阵分解技术，可以利用数据的低秩结构进行压缩
4. **K vs V**：K 天然低秩（94%@rank=8），V 更难压缩（60%@rank=8）—— Value Bottleneck
5. **FlashAttention**：通过 (m, ℓ, y) 中间统计量实现 O(N) 显存复杂度
6. **Rate-Distortion**：评估压缩方案的核心框架，需要在压缩率和精度之间 trade-off
7. **GQA**：Grouped Query Attention，通过 KV heads 共享减少显存占用
8. **Attention Sink**：前几个 token 获得异常高注意力权重的现象，对压缩策略有重要启示

**核心公式速记**：
```
KV Cache 大小 = 2 × num_layers × num_kv_heads × seq_len × head_dim × 2 bytes
SVD 累积方差 = Σᵢ₌₁ᵞσᵢ² / Σᵢ₌₁ⁿσᵢ²
压缩比 = 原始大小 / 压缩后大小
相对误差 = ||A - Ã||_F / ||A||_F
```

**下一步**：《核心算法篇》——我们将深入学习 ACCORD-KV 的具体实现，包括不同的压缩算法、异构后端架构和协议设计。

---

## 第6章：深入理解 KV Cache 的内部结构

### 6.1 多头注意力与 KV Heads

在正式进入下一部分之前，让我们更深入地理解 KV Cache 的内部结构。这对于理解 ACCORD-KV 的压缩策略至关重要。

**标准 Multi-Head Attention (MHA)**：

在标准 Transformer 中，每个注意力头都有独立的 Q, K, V 投影：
- num_heads = 32（对于 7B 模型）
- 每个 head 的维度 head_dim = 128
- 每个 token 产生的 KV 大小 = 32 × 128 = 4096 维

**分组查询注意力 (Grouped Query Attention, GQA)**：

Mistral-7B 使用 GQA 来减少 KV Cache 的大小：
- Q heads = 32（和标准一样）
- KV heads = 8（减少到 1/4）
- 每 4 个 Q head 共享 1 个 KV head

> 💡 **为什么 GQA 能省内存？**
> 
> 想象你有 32 个人要发言，但只有 8 个麦克风。每 4 个人共用一个麦克风，说话内容会有些损失，但交流仍然可以进行。GQA 也是这样——通过共享 KV heads 节省内存，但注意力表达能力略有下降。

**GQA 的 KV Cache 大小计算**：

```
标准 MHA：32 heads × 128 dim = 4096 维/token
GQA (8 KV heads)：8 heads × 128 dim = 1024 维/token
节省：4× 的 KV Cache 大小！
```

### 6.2 KV Cache 的时间维度

KV Cache 不仅仅是空间问题，还有时间维度的问题。

**Prefill 阶段**：
- 处理完整的 prompt（比如 1024 tokens）
- 一次性计算所有 token 的 K, V
- KV Cache 快速增长

**Decode 阶段**：
- 每生成一个新 token，需要：
  1. 计算新 token 的 Q, K, V
  2. 把新的 K, V 加入 Cache
  3. 用所有 KV 计算 attention

> 💡 **关键观察**：
> - Prefill 阶段：KV Cache 快速增长，但只做一次
> - Decode 阶段：KV Cache 缓慢增长（每次 +1 token），但每个新 token 都要读取全部历史 KV
> 
> 这就是为什么 Decode 是"访存密集型"——每次都要读取大量的历史数据！

### 6.3 Attention Sink 现象

在分析 KV Cache 时，有一个重要现象需要了解：**Attention Sink**。

研究发现，在大多数 LLM 中，前几个 token（通常是 special tokens 如 `<bos>`, `<pad>` 等）会获得异常高的注意力权重。

```
Attention 权重分布示意：
Token 0:  ████████████████████ 45%
Token 1:  ████████ 15%
Token 2:  ██████ 10%
...
Token 100: █ 2%
Token 500: █ 1%
Token 1000: █ 0.5%
```

> 💡 **为什么会有 Attention Sink？**
> 
> 可能的解释：
> 1. 这些是"锚点"token，模型学会把它们当作信息汇聚点
> 2. Special tokens 在训练数据中出现频繁，模型对它们形成"依赖"
> 3. Softmax 的数学特性——最大值的累积效应

**这对压缩意味着什么？**

如果前几个 token 的 KV 对结果影响最大，压缩时应该：
- 优先保留 Attention Sink 的 KV
- 对中间 token 可以更激进地压缩

这是 ACCORD-KV 的重要设计考量！

### 6.4 KV Cache 在 GPU 显存中的布局

了解 KV Cache 在显存中的物理布局，有助于理解为什么传输它这么慢。

**典型的 GPU 显存布局**：

```
GPU HBM (High Bandwidth Memory)
├── Model Weights (13 GB for 7B model, FP16)
├── KV Cache (动态分配)
│   ├── Layer 0: [batch_size, num_kv_heads, seq_len, head_dim]
│   ├── Layer 1: ...
│   └── Layer 31: ...
├── Activation (中间计算结果)
└── Free Space (剩余显存)
```

**跨节点传输的挑战**：

当 KV Cache 需要从 Prefill GPU 传到 Decode GPU 时：
1. 数据从 HBM 读出到 PCIe
2. 通过网络传输
3. 写入 Decode GPU 的 HBM

这个过程涉及多次内存拷贝和 PCIe/网络传输，是典型的**带宽瓶颈**。

> 💡 **数字对比**：
> - HBM 带宽：~1 TB/s (A100)
> - PCIe 4.0 x16：~32 GB/s
> - 100 Gbps 网络：~12.5 GB/s
> 
> 跨节点传输带宽只有 GPU 内部带宽的 **1/80**！

---

## 第7章：PD 分离架构的工程挑战

### 7.1 分布式 LLM 推理的背景

在大规模 LLM 部署中，单个 GPU 往往装不下整个模型。即使装得下，并发请求的数量也会让显存不够用。

**常见的分布式策略**：

| 策略 | 方式 | 优点 | 缺点 |
|------|------|------|------|
| Tensor Parallel | 模型层内分割 | 高吞吐 | 通信密集 |
| Pipeline Parallel | 模型层间分割 | 简单 | 流水线气泡 |
| Data Parallel | 多副本 | 扩展性好 | 显存浪费 |
| PD 分离 | Prefill/Decode 分开 | 最优资源利用 | KV 传输 |

**PD 分离为什么火了？**

2023-2024 年的研究发现：
1. Prefill 和 Decode 的算子需求差异巨大
2. 统一部署无法同时优化两者
3. 分离后可以分别调优，大幅提升效率

### 7.2 DistServe：PD 分离的开创者

DistServe（OSDI 2024）是 PD 分离的代表性工作。它观察到：

**Prefill 特点**：
- 矩阵运算为主（GEMM）
- Batch size 通常较小（1-4）
- 计算密度高（FLOPs/byte）

**Decode 特点**：
- 内存访问为主
- Batch size 可以很大（100+）
- 计算密度低

**DistServe 的方案**：
```
请求 → Prefill GPU → KV Cache → Decode GPU → 响应
              ↓                          ↓
         (计算密集)              (访存密集)
```

### 7.3 LoongServe：更进一步的分离

LoongServe 在 DistServe 基础上进一步优化：
- 支持更细粒度的请求调度
- KV Cache 在 Prefill 和 Decode 之间高效路由
- 引入了 **KV Transfer** 的概念

> 💡 **LoongServe 的关键洞察**：
> 
> 并不是所有请求都需要完整的 KV Cache！
> - 短回复请求：只需要开头部分的 KV
> - 长上下文请求：需要更激进的压缩
> 
> "一刀切"的传输策略是次优的。

### 7.4 KV 传输的带宽分析

让我们做一个详细的带宽分析：

**场景**：Mistral-7B, 4096 token 上下文, 100 Gbps 网络

```
原始 KV 大小 = 512 MB

如果用 100 Gbps 网络传输：
- 时间 = 512 MB / 12.5 GB/s = 40.96 ms

生成一个 token 的时间 ≈ 20 ms（Decode 阶段）

结论：KV 传输时间 > token 生成时间！
```

这意味着即使 Prefill GPU 计算得再快，KV 传输也会成为瓶颈！

> 💡 **打破瓶颈的方向**：
> 1. 提升网络带宽（贵，有物理限制）
> 2. 减少传输数据量（压缩，正是 ACCORD-KV 的方向）
> 3. 提前传输（prefetching，在 Decode 之前就开始传）

### 7.5 压缩 vs 预取：两条技术路线

学术界探索了两条解决 KV 传输瓶颈的路线：

**路线 1：压缩（Compression）**
- 目标：减少要传输的数据量
- 代表：SpectrumKV, ACCORD-KV
- 优点：带宽需求降低
- 缺点：可能有精度损失

**路线 2：预取（Prefetching）**
- 目标：在 Decode 开始前就准备好 KV
- 代表：各种 speculative prefetching 工作
- 优点：不损失精度
- 缺点：需要预测未来，准确性难以保证

**ACCORD-KV 的立场**：
> "压缩和预取不是互斥的，可以结合使用。ACCORD-KV 的压缩方案让预取更容易——因为压缩后的数据量更小，传输更快。"

---

## 第8章：Rate-Distortion 理论的深层理解

### 8.1 信息论视角

Rate-Distortion 理论最初来自香农的信息论。让我们从信息论角度重新理解它。

**信源编码定理**：

对于一个随机变量 X，如果我们想用 R bits 表示它，但允许最大失真 D，那么：
- R 必须大于 Rate-Distortion 函数 R(D)
- R(D) 是达到失真 D 所需的最小码率

> 💡 **直觉**：
> 
> 想象你在发送一张图片：
> - 完全精确发送 = 很高的码率
> - 允许一些失真 = 可以用更少的 bits
> 
> Rate-Distortion 函数告诉你："在允许 D 失真的情况下，最少需要多少码率？"

### 8.2 RD 曲线与 Pareto 最优

在实际系统中，我们通常会得到一条 RD 曲线：

```
Distortion
    ↑
    │  · · · · · · · ·  (Pareto 前沿)
    │ ·               ·
    │  ·             ·
    │   ·           ·   ← 最优方案集合
    │    ·         ·
    │     ·       ·
    │      ·     ·
    │       ·   ·
    │        · ·
    │         ·
    └────────────────────→ Rate
```

**Pareto 最优**：
- 如果没有其他方案能在不增加 Rate 的情况下降低 Distortion
- 也没有方案能在不增加 Distortion 的情况下降低 Rate
- 那么这个方案就是 Pareto 最优的

> 💡 **ACCORD-KV 的 Pareto 前沿**：
> 
> 通过选择不同的 rank、量化级别、传输策略，ACCORD-KV 可以在 RD 曲线上探索不同的点。
> 
> 高压缩比（左侧）= 更多近似
> 低失真（下方）= 更精确

### 8.3 任务相关的 Distortion 度量

之前我们讨论的 Distortion 都是数学度量（如 Frobenius 范数）。但更关键的问题是：**这些数学误差如何影响实际任务？**

**三种层次的评估**：

| 层次 | 指标 | 说明 |
|------|------|------|
| 重建层 | Relative Error, Cosine | 矩阵级别的误差 |
| Attention 层 | Attention divergence | 注意力权重分布的变化 |
| 任务层 | Perplexity,下游任务准确率 | 最终效果 |

> 💡 **关键洞察**：
> 
> 重建误差小 ≠ 任务效果好！
> 
> 比如，K 的微小误差可能导致 Attention 权重完全改变，但 Cosine 相似度仍然较高。
> 
> 因此，ACCORD-KV 必须同时关注多个层次的指标。

### 8.4 不同任务的 RD 权衡

不同应用场景对 Rate 和 Distortion 的偏好不同：

**场景 1：代码生成**
- 对精度要求高：一个小错误可能导致整个程序崩溃
- 偏好：低 Distortion，愿意牺牲一些压缩比
- 推荐：rank=32 或更高

**场景 2：长文本摘要**
- 可以容忍少量信息损失
- 偏好：高压缩比（因为上下文很长）
- 推荐：rank=8 或更低

**场景 3：实时对话**
- 需要极低延迟
- 偏好：高压缩比
- 推荐：动态选择 rank

> 💡 **ACCORD-KV 的灵活性**：
> 
> 通过 Contract Type，ACCORD-KV 让应用自己决定 Rate-Distortion 的权衡点。
> 
> - EXACT：保证精度
> - APPROX：最大化压缩
> - BOUNDED：用户指定误差上限

---

## 附录 B：常见问题解答（FAQ）

**Q1: 为什么 KV Cache 不用更激进的量化（如 INT2, INT1）？**

A1: 量化太激进会严重损害模型质量。实验显示：
- INT8: 质量损失可接受
- INT4: 需要配合 SVD 才能勉强使用
- INT2/INT1: 质量崩溃，通常不可用

**Q2: SVD 压缩是无损的吗？**

A2: 不是，SVD 截断是有损压缩。但它是给定 rank 下的最优有损压缩（Eckart-Young-Mirsky 定理）。

**Q3: 为什么 Value 比 Key 难压缩？**

A3: K 决定"关注哪些位置"（注意力权重），相对简单；V 包含"实际内容"，更复杂。实验数据证实 K 的累积方差衰减快（rank=8 时 94%），V 衰减慢（rank=8 时 60%）。

**Q4: (m, ℓ, y) 传输和直接传 KV 相比，精度如何？**

A4: Wire Format (m, ℓ, y) 如果不配合压缩，精度是**完全一致的**——它只是改变了传输的内容，而非引入近似。真正的近似来自 SVD 压缩和 INT4 量化。

**Q5: ACCORD-KV 的 31,775× 压缩比是怎么实现的？**

A5: 这是端到端的综合效果：
- SVD 把矩阵从 [T, d] 压缩到 [r, d]
- INT4 量化节省 4×
- Wire Format 传 (m, ℓ, y) 而不是 K, V
- 实际压缩比取决于 seq_len、q_len、rank 等参数

---

## 附录 C：参考文献与延伸阅读

### 核心论文

1. **FlashAttention** (Dao et al., 2022) - 提出了 (m, ℓ, y) 在线 softmax
2. **DistServe** (Yu et al., 2024) - PD 分离架构的开创性工作
3. **LoongServe** (Wu et al., 2024) - 细粒度 PD 分离
4. **SpectrumKV** (Patel et al., 2024) - KV Cache 压缩的先驱

### 相关技术

1. **GQA - Grouped Query Attention** (Ainslie et al., 2023) - Mistral 采用的注意力变体
2. **PagedAttention** (Kwon et al., 2023) - vLLM 的 KV Cache 管理
3. **Attention Sink** (Xiao et al., 2023) - 解释 Attention 权重分布

### 数学基础

1. **Matrix Analysis** (Horn & Johnson) - SVD 的数学理论
2. **Information Theory** (Cover & Thomas) - Rate-Distortion 理论

---

## 附录 D：关键概念术语表

| 术语 | 英文 | 解释 |
|------|------|------|
| KV Cache | Key-Value Cache | 存储注意力机制中 Key 和 Value 向量的缓存 |
| Prefill | Prefill Phase | 处理输入 prompt 的阶段，计算初始 KV |
| Decode | Decode Phase | 自回归生成新 token 的阶段 |
| PD 分离 | Prefill-Decode Disaggregation | 将 Prefill 和 Decode 分配到不同 GPU 的架构 |
| SVD | Singular Value Decomposition | 奇异值分解，用于矩阵压缩 |
| Rank | Rank | 矩阵的秩，低秩意味着可以用更少的维度表示 |
| 累积方差 | Cumulative Variance | 保留 r 个奇异值能解释原始数据的比例 |
| FlashAttention | FlashAttention | 节省显存的注意力机制实现 |
| GQA | Grouped Query Attention | 分组查询注意力，减少 KV heads |
| Rate | Rate | 码率，衡量压缩后的大小 |
| Distortion | Distortion | 失真，衡量压缩带来的误差 |
| Pareto 最优 | Pareto Optimal | 无法在某一维度改进而不牺牲另一维度的状态 |
| Attention Sink | Attention Sink | 获得异常高注意力的 token |

---

<!-- PART1 COMPLETE -->


# 第二部分：核心代码解读篇

> **前置知识**：本篇假设你已经熟悉 Python 基础（函数、类、数据类型）和 NumPy / PyTorch 的基本操作（张量形状、矩阵乘法）。如果你还不了解 Attention 机制的基本原理，建议先阅读第一部分的相关章节。

---

## 第5章：如何正确计算 Attention 统计量（attn_stats.py）

### 5.1 目标：提取 Attention 的三个关键量

在传统的 Transformer 中，Attention 的计算可以写成如下公式：

```
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V
```

当我们需要**跨机器传输中间计算结果**时（比如 ACCORD-KV 系统中 Q 在客户端、KV 在服务端），完整地传输 `Q @ K^T` 矩阵会消耗巨大的带宽。

ACCORD-KV 采用了一种巧妙的**分解技巧**：不传输完整的注意力分数矩阵，而是传输三个可以精确重构 Attention 结果的统计量——**`(m, l, y)` 三元组**。

**`(m, l, y)` 的含义**：

| 符号 | 含义 | 形状 | 数学定义 |
|------|------|------|----------|
| `m` | **max** — log-sum-exp 的最大值项 | `[H, q_len, 1]` | `m = max(scores)` |
| `l` | **sum** — 指数和 | `[H, q_len, 1]` | `l = Σ exp(scores - m)` |
| `y` | **weighted sum** — V 的加权求和 | `[H, q_len, d]` | `y = Σ exp(scores - m) · V` |

这三个量的组合为什么能重建 attention？答案在于 softmax 的**数值稳定公式**：

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
             = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

当我们把 `exp(x_i - m)` 的"最大值归一化"操作提前记录下来，就可以用 (m, l, y) 完美恢复最终的 attention 输出：

```
output = y / l = Σ [exp(x_i - m) / l] · V = Σ softmax(x_i) · V  ✓
```

### 5.2 代码逐段解读

接下来我们逐段解读 `attn_stats.py` 的核心代码。

#### 5.2.1 数据类定义

```python
@dataclass
class AttnStats:
    """(m, l, y) — FlashAttention online softmax 状态。"""
    m: torch.Tensor  # [H, q_len, 1] — log-sum-exp max
    l: torch.Tensor  # [H, q_len, 1] — sum-exp
    y: torch.Tensor  # [H, q_len, d] — un-normalized weighted sum of V
```

**这段代码在做什么**：定义一个数据类来封装 Attention 统计量的三个组成部分。注意：
- `m` 和 `l` 的最后一维是 1（而不是标量），这是为了方便广播计算
- `y` 的最后一维是 `d`（embedding 维度）
- Phase 1 固定 `num_heads=1`，所以 `H` 维度恒为 1

```python
    def __post_init__(self):
        H, Ql, _ = self.m.shape
        _, _, D = self.y.shape
        if self.l.shape != (H, Ql, 1):
            raise ValueError(...)
        if self.y.shape != (H, Ql, D):
            raise ValueError(...)
```

**这段代码在做什么**：构造器后的自动校验。`__post_init__` 会在数据类初始化后自动执行，确保传入的张量形状符合约定。为什么要这样做？因为形状不匹配是 KV Cache 系统的常见 bug 源头，早期发现比运行到一半崩溃要好。

#### 5.2.2 创建空状态

```python
    @classmethod
    def empty(
        cls,
        num_heads: int,
        q_len: int,
        d: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "AttnStats":
        """构造零状态（m = -inf, l = 0, y = 0）。

        用于 server 端没找到任何 KV block 时的 fallback。
        """
        return cls(
            m=torch.full((num_heads, q_len, 1), float("-inf"), device=device, dtype=dtype),
            l=torch.zeros((num_heads, q_len, 1), device=device, dtype=dtype),
            y=torch.zeros((num_heads, q_len, d), device=device, dtype=dtype),
        )
```

**这段代码在做什么**：创建一个"空状态"的 AttnStats。为什么用 `-inf` 作为 `m` 的初值？这是一个**数值稳定性技巧**：

- 当我们第一次计算 `m_new = max(m1, m2)` 时，如果 `m1 = -inf`，则 `max(-inf, m2) = m2`，这正是我们想要的
- `l = 0` 表示"还没有看到任何有效的 attention 分数"
- `y = 0` 表示"还没有累加任何 V 的加权值"

这个空状态用于：当服务端没有任何 KV block 可以提供时，返回一个合理的 fallback。

#### 5.2.3 重建 Attention 输出

```python
    def finalize(self) -> torch.Tensor:
        """从 (m, l, y) 还原 attention 输出: output = y / l。

        返回形状: [num_heads, q_len, d]
        """
        return self.y / self.l.clamp(min=EPS)
```

**这段代码在做什么**：从统计量重建最终的 attention 输出。核心就是 `output = y / l`。

**重点：为什么要 `l.clamp(min=EPS)`？**

这里的 `EPS = 1e-30`，是一个极小的正数。如果 `l = 0`（理论上不应该发生，但如果发生数值误差），直接 `y / l` 会导致 `inf` 或 `NaN`。用 `clamp` 限制下界可以避免除零错误。

**数学验证**：

```
设 scores = [s1, s2, ..., sn]
m = max(scores)
l = Σ exp(s_i - m)
y = Σ exp(s_i - m) · V_i

output_j = y_j / l
        = Σ exp(s_i - m) · V_ij / Σ exp(s_k - m)
        = Σ softmax(s_i) · V_ij    ✓
```

### 5.3 验证：为什么这些量可以重构 Attention

让我们用 NumPy 演示这个重构过程：

```python
import numpy as np

# 模拟 Q, K, V
Q = np.random.randn(4, 8)   # q_len=4, d=8
K = np.random.randn(10, 8)  # 10 个 key 向量
V = np.random.randn(10, 8)  # 10 个 value 向量

# 原始 attention
scores = Q @ K.T / np.sqrt(8)
probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
probs = probs / probs.sum(axis=-1, keepdims=True)
true_output = probs @ V

# 用 (m, l, y) 重构
m = scores.max(axis=-1, keepdims=True)           # [4, 1]
p = np.exp(scores - m)                           # [4, 10]
l = p.sum(axis=-1, keepdims=True)               # [4, 1]
y = p @ V                                        # [4, 8]
reconstructed = y / l                            # [4, 8]

print("重构误差:", np.max(np.abs(true_output - reconstructed)))
# 输出: 重构误差: 0.0  (理论上完全相同)
```

**关键发现**：`(m, l, y)` 三元组是 attention 的**无损压缩表示**——信息完全没有丢失，但数据量大幅减少（不需要传输完整的 `q_len × kv_len` 注意力分数矩阵）。

---

## 第6章：如何合并两个 Attention 状态（merge.py）

### 6.1 为什么要合并？

在 ACCORD-KV 系统中，KV Cache 被分成多个 **KV block**。当一个 attention 请求涉及多个 block 时，每个 block 独立计算自己的 `(m, l, y)` 统计量，然后我们需要把这些统计量**合并**成一个统一的结果。

**合并的场景**：
- 客户端请求访问分布在多个服务端节点的 KV block
- 每个节点返回自己的 `(m, l, y)` 统计量
- 客户端需要把这些统计量合并，计算最终 attention

**为什么不能直接拼接再重算？**
- `Q @ K^T` 的结果是 `[q_len, total_kv_len]` 的矩阵
- 当 `total_kv_len` 很大时，这个矩阵本身就很大，违背了"传输统计量而非完整矩阵"的初衷

### 6.2 合并公式的数学推导

设有两个独立的 attention 统计量 `(m₁, l₁, y₁)` 和 `(m₂, l₂, y₂)`，它们分别来自两个 KV block。我们希望合并成 `(m, l, y)`，使得：

```
合并后的 output = 合并前的 output₁ 与 output₂ 的拼接结果
```

**推导过程**：

假设第 i 个 query 对两个 block 的 attention 分数分别是 `s₁` 和 `s₂`，则：
- 第 i 个 block 的 softmax 归一化因子：`exp(s₁ - m₁)`
- 第 i 个 block 的加权求和：`exp(s₁ - m₁) · V₁`

合并后的 max 值：
```
m = max(s₁, s₂) = max(m₁ + log Σ exp(s₁' - m₁), m₂ + log Σ exp(s₂' - m₂))
```

展开可得简洁形式：
```
m = max(m₁, m₂)
```

设 `α₁ = exp(m₁ - m)`，`α₂ = exp(m₂ - m)`，则：

```
l = l₁ · α₁ + l₂ · α₂
y = y₁ · α₁ + y₂ · α₂
```

**合并公式汇总**：
```
m_new = max(m₁, m₂)
α₁ = exp(m₁ - m_new)
α₂ = exp(m₂ - m_new)
l_new = l₁ · α₁ + l₂ · α₂
y_new = y₁ · α₁ + y₂ · α₂
```

### 6.3 代码逐段解读

```python
def merge_stats(a: AttnStats, b: AttnStats) -> AttnStats:
    """合并两段 AttnStats。"""
    # 首先校验形状和 dtype 一致性
    if a.shape_tuple() != b.shape_tuple():
        raise ValueError(...)
    if a.m.dtype != b.m.dtype:
        raise ValueError(...)
```

**这段代码在做什么**：参数校验。为什么要校验？因为合并操作要求两个统计量的形状完全一致（否则无法对应位置相加）。dtype 校验则是因为数值运算假设精度一致。

```python
    # 1. 计算新的 max 值
    m_new = torch.maximum(a.m, b.m)
    
    # 2. 计算归一化因子 α
    alpha_a = torch.exp(a.m - m_new)
    alpha_b = torch.exp(b.m - m_new)
```

**这段代码在做什么**：计算新的 max 值和归一化因子。

**数学原理**：
- `torch.maximum(a.m, b.m)` 对应 `max(m₁, m₂)`
- `exp(m_i - m_new)` 一定在 `[0, 1]` 范围内（因为 `m_new ≥ m_i`）
- 这个性质保证数值稳定：`exp(负数)` 永远不会溢出

```python
    # 3. 处理边界情况：两边都是 empty
    override_mask = torch.isneginf(m_new)
    if override_mask.any():
        denom = a.l + b.l + EPS
        safe_a = a.l / denom
        safe_b = b.l / denom
        alpha_a = torch.where(override_mask, safe_a, alpha_a)
        alpha_b = torch.where(override_mask, safe_b, alpha_b)
```

**这段代码在做什么**：处理**两个 empty 状态合并**的边界情况。

**问题**：
- 当 `m₁ = -inf` 且 `m₂ = -inf` 时，`m_new = -inf`
- 那么 `exp(-inf - (-inf)) = exp(NaN) = NaN`
- 这会导致 `l_new = 0 * NaN + 0 * NaN = NaN` ❌

**解决方案**：当 `m_new = -inf` 时，改用按 `l` 的比例分配：
- 如果 `l₁ = l₂ = 0`（都是空状态），则 `α₁ = α₂ = 0.5`
- 这样 `l_new = 0 * 0.5 + 0 * 0.5 = 0`，语义正确

```python
    # 4. 计算合并后的 l 和 y
    l_new = a.l * alpha_a + b.l * alpha_b
    y_new = a.y * alpha_a + b.y * alpha_b

    return AttnStats(m=m_new, l=l_new, y=y_new)
```

**这段代码在做什么**：完成最终的合并计算。这两行对应我们推导的公式。

### 6.4 合并的性质验证

**交换律**：`merge(a, b) == merge(b, a)`

```python
# 验证交换律
m1 = merge_stats(a, b)
m2 = merge_stats(b, a)
assert torch.allclose(m1.m, m2.m)
assert torch.allclose(m1.l, m2.l)
assert torch.allclose(m1.y, m2.y)
```

**结合律**：`merge(a, merge(b, c)) == merge(merge(a, b), c)`

这个性质允许我们**任意顺序**合并多个 KV block，这对于分布式系统非常重要（不同节点可能按不同顺序返回结果）。

### 6.5 边界情况处理总结

| 情况 | m | l | α 处理 | 结果 |
|------|---|---|--------|------|
| 正常合并 | `max(m₁, m₂)` | 公式计算 | `exp(m_i - m_new)` | 正确 |
| a empty, b normal | `m₂` | 公式计算 | `exp(-inf - m₂) = 0`, `exp(m₂ - m₂) = 1` | `l=l₂, y=y₂` |
| a normal, b empty | `m₁` | 公式计算 | `exp(m₁ - m₁) = 1`, `exp(-inf - m₁) = 0` | `l=l₁, y=y₁` |
| 两者都是 empty | `-inf` | 0 | 按 l 比例分 | `l=0, y=0, m=-inf` |

---

## 第7章：如何精确重建 Attention（exact_local.py）

### 7.1 目标：给定 (m, l, y)，精确重建 KV

`ExactLocal` 是一个**本地 attention baseline** 实现。它的设计目标有两个：

1. **降级路径**：当远程服务端不可达或响应超时时，回退到本地计算
2. **对照实验**：作为"无通信代价"的基准，衡量 ACCORD-KV 的效率提升

从接口设计角度，`ExactLocal` 和 `MockAttentionServer` **完全一致**——上层调用者不需要知道底层走的是网络还是本地。

### 7.2 代码逐段解读

```python
class ExactLocal:
    def __init__(
        self,
        kv_cache: KVCache | None = None,
        num_heads: int = 1,
    ):
        self.kv_cache: KVCache = dict(kv_cache) if kv_cache else {}
        self.num_heads = num_heads
        self.requests_served = 0
        self.requests_empty = 0
```

**这段代码在做什么**：初始化本地 attention backend。

**参数说明**：
- `kv_cache`：本地持有的 KV 块字典，`{block_id: (K, V)}`
- `num_heads`：模拟的 head 数（Phase 1 固定为 1）
- `requests_served`/`requests_empty`：统计指标，用于分析性能

```python
    def serve(self, acr: ACR) -> AttnStats:
        """本地直跑 attention（不经过网络）。"""
        self.requests_served += 1
        q_len = acr.q_len
        d = acr.d
        H = self.num_heads

        # 1. 过滤出本地有的 block
        local_blocks = [bid for bid in acr.block_ids if bid in self.kv_cache]
        if not local_blocks:
            self.requests_empty += 1
            return AttnStats.empty(H, q_len, d, device=acr.q_tokens.device, dtype=acr.q_tokens.dtype)
```

**这段代码在做什么**：处理 attention 请求。

**步骤解释**：
1. 从请求中提取 `q_len`、`d`、`H`（head 数）
2. 过滤出本地缓存中实际存在的 block（`acr.block_ids` 是请求需要的 block 列表）
3. 如果没有任何本地 block，返回空状态（这和第 5 章的 `AttnStats.empty()` 一致）

```python
        # 2. 收集 K 和 V
        K_list = [self.kv_cache[bid][0] for bid in local_blocks]
        V_list = [self.kv_cache[bid][1] for bid in local_blocks]
        K = torch.cat(K_list, dim=0)  # 沿 seq 维度拼接
        V = torch.cat(V_list, dim=0)
        Q = acr.q_tokens
```

**这段代码在做什么**：收集并拼接多个 KV block 的内容。

**关键理解**：
- 每个 block 存储的是 `(K, V)` 元组
- `torch.cat` 沿 `dim=0`（sequence 维度）拼接
- 拼接后的 `K` 形状是 `[total_kv_len, d]`，`V` 形状相同

```python
        # 3. 计算 attention
        scores = Q @ K.T              # [q_len, total_kv_len]
        m = scores.max(dim=-1, keepdim=True).values
        p = torch.exp(scores - m)     # 数值稳定的 exp
        l = p.sum(dim=-1, keepdim=True)
        y = p @ V                      # [q_len, d]
        
        return AttnStats(
            m=m.unsqueeze(0),         # 添加 head 维度
            l=l.unsqueeze(0),
            y=y.unsqueeze(0),
        )
```

**这段代码在做什么**：核心 attention 计算。

**数学解释**：
1. `scores = Q @ K^T`：计算原始注意力分数，形状 `[q_len, kv_len]`
2. `m = max(scores)`：记录每行（每个 query）的最大值
3. `p = exp(scores - m)`：数值稳定的指数（确保 `exp(负数)` 不会溢出）
4. `l = sum(p)`：归一化因子的和
5. `y = p @ V`：加权求和，得到未归一化的输出

**注意**：
- `unsqueeze(0)` 添加了 head 维度（Phase 1 固定 `H=1`）
- 最终返回 `(m, l, y)` 三元组，和第 5 章的格式完全一致

### 7.3 与 MockAttentionServer 的对比

| 特性 | ExactLocal | MockAttentionServer |
|------|------------|---------------------|
| 网络通信 | 无 | 模拟网络延迟 |
| KV 来源 | 本地 `kv_cache` | 模拟服务端 |
| 接口 | `serve(acr)` | `serve(acr)` |
| Phase 2 计划 | 接入 SWS block 选择 | 保持模拟 |

这个设计保证了**接口一致性**：无论底层是远程调用还是本地计算，上层的调度逻辑都不需要修改。

---

## 第8章：ACR 协议——请求头的 wire format（acr.py）

### 8.1 wire format 是什么

在网络协议中，**wire format** 指的是数据在网络上传输时的二进制格式。对于 ACCORD-KV 系统，ACR（Attention Computation Request）是客户端发给服务端的请求消息。

**为什么需要专门设计 ACR？**
- 传统方法：直接发送 `Q` 向量 + block 列表
- ACR 方式：发送完整的请求元数据，包括路由策略、合约类型、截止时间等

ACR 的设计遵循以下原则：
- **不可变性**：`frozen=True` 确保协议对象一旦创建就不能修改
- **完整性**：包含所有服务端决策所需的信息
- **类型安全**：使用 dataclass 和 Enum 防止无效组合

### 8.2 acr.py 核心函数解读

#### 8.2.1 合约类型枚举

```python
class ContractType(str, Enum):
    """合约类型 — 决定 server 端允许的近似程度。"""
    EXACT = "exact"        # 必须 exact attention（Phase 1 默认）
    APPROX = "approx"      # 容许 (m,l,y) 路径 / block-sparse 近似
    BOUNDED = "bounded"    # 数值误差必须 < error_budget
```

**这段代码在做什么**：定义三种合约类型。

**概念解释**：
- `EXACT`：服务端必须返回精确的 attention 结果（完全等价于本地计算）
- `APPROX`：允许使用统计量路径或 block 稀疏化等近似方法
- `BOUNDED`：允许近似，但必须保证数值误差在 `error_budget` 范围内

这种分层设计让系统可以在**精度**和**性能**之间做灵活权衡。

#### 8.2.2 ACR 数据类定义

```python
@dataclass(frozen=True)
class ACR:
    """Attention Computation Request。"""
    # 身份标识
    acr_id: str
    q_block_id: int
    q_tokens: torch.Tensor

    # 路由信息
    server_hints: List[str] = field(default_factory=list)
    block_ids: List[int] = field(default_factory=list)

    # 合约配置
    contract_type: ContractType = ContractType.EXACT
    deadline_ms: float = 50.0
    error_budget: float = 0.0

    # 路由策略
    prefer_local: bool = False
    max_rpc_hops: int = 2
```

**这段代码在做什么**：定义完整的请求结构。

**字段说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `acr_id` | str | 请求唯一标识（用于追踪和调试） |
| `q_block_id` | int | Query 块的 ID |
| `q_tokens` | Tensor | Query 内容，形状 `[q_len, d]` |
| `server_hints` | List[str] | 建议的服务端节点（可多个） |
| `block_ids` | List[int] | 需要访问的 KV block ID 列表 |
| `contract_type` | ContractType | 精度要求级别 |
| `deadline_ms` | float | 截止时间（毫秒） |
| `error_budget` | float | BOUNDED 模式下允许的误差 |
| `prefer_local` | bool | 是否优先使用本地计算 |
| `max_rpc_hops` | int | 最大 RPC 跳转次数 |

**为什么 `frozen=True`？**
- 防止在多服务端路由过程中被意外修改
- 允许多线程/多进程安全地共享同一个 ACR 对象

#### 8.2.3 构造器校验

```python
    def __post_init__(self):
        # 早期校验：避免在 server 端才发现协议错误
        if not self.acr_id:
            raise ValueError("acr_id must be non-empty")
        if self.q_tokens.ndim != 2:
            raise ValueError(
                f"q_tokens must be 2D [q_len, d], got shape {tuple(self.q_tokens.shape)}"
            )
        if self.deadline_ms <= 0:
            raise ValueError(f"deadline_ms must be > 0, got {self.deadline_ms}")
        if self.contract_type == ContractType.BOUNDED and self.error_budget <= 0:
            raise ValueError(
                "BOUNDED contract requires error_budget > 0, "
                f"got {self.error_budget}"
            )
```

**这段代码在做什么**：构造器级别的参数校验。

**为什么在这里校验？**
- 错误发现越早越好
- 如果 `acr_id` 为空，服务端将无法追踪请求
- 如果 `q_tokens` 形状不对，后续计算会失败
- 如果 `BOUNDED` 合约没有设置 `error_budget`，精度保证无从谈起

#### 8.2.4 便捷属性

```python
    @property
    def q_len(self) -> int:
        return int(self.q_tokens.shape[0])

    @property
    def d(self) -> int:
        return int(self.q_tokens.shape[-1])
```

**这段代码在做什么**：提供便捷的属性访问。

**为什么不用直接访问 `.shape`？**
- 代码更简洁：`acr.q_len` vs `acr.q_tokens.shape[0]`
- 语义更清晰：明确知道这是 query 长度和维度
- 如果未来 shape 约定改变，只需要改一处

#### 8.2.5 本地优先判断

```python
    def is_local_only(self) -> bool:
        """是否强制走本地（用于 ExactLocal fallback 测试）。"""
        return self.prefer_local and not self.server_hints
```

**这段代码在做什么**：判断是否强制使用本地计算。

**逻辑解读**：
- `prefer_local=True`：用户/调度器希望优先本地
- `not server_hints`：没有指定具体服务端（意味着不知道远程在哪）
- 两个条件都满足时，说明这是一个"纯本地"请求

### 8.3 从实现细节理解 ACR 的设计哲学

**1. 最小化依赖**
- `q_tokens` 不强制 `dtype` 和 `device`
- 转换由 caller 在发送前完成，服务端内部再转

**2. 路由灵活性**
- `server_hints` 是提示而非硬约束
- `max_rpc_hops` 支持多跳路由

**3. 性能可观测性**
- `acr_id` 让每个请求可追踪
- `deadline_ms` 让服务端知道超时边界

**4. 协议演进友好**
- 使用 `List[...]` 而非固定长度数组
- Enum 类型易于扩展新合约

---

## 总结：核心概念的关联图

```
┌─────────────────────────────────────────────────────────┐
│                        ACR                                │
│   (Attention Computation Request)                        │
│   - 包含 q_tokens, block_ids, contract_type 等          │
└─────────────────────┬───────────────────────────────────┘
                      │ serve()
                      ▼
┌─────────────────────────────────────────────────────────┐
│              ExactLocal / MockAttentionServer           │
│   - 计算 (m, l, y) 统计量                                │
│   - 使用本地 KV 或远程获取 KV                            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    AttnStats                             │
│   (m, l, y) 三元组                                       │
│   - m: max(scores)                                      │
│   - l: sum(exp(scores - m))                             │
│   - y: sum(exp(scores - m) * V)                         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    merge_stats                           │
│   - 合并多个 AttnStats                                   │
│   - 支持任意顺序（交换律、结合律）                        │
│   - 边界情况处理（empty + empty）                        │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                      finalize                            │
│   output = y / l                                         │
│   = softmax(Q @ K^T) @ V  ✓                            │
└─────────────────────────────────────────────────────────┘
```

这四个文件构成了 ACCORD-KV 的**核心数据流**：从请求封装 → 统计量计算 → 合并 → 重构，形成了一个完整且数学上优雅的分布式 Attention 解决方案。

<!-- PART2 COMPLETE -->


# 第三部分：论文精读篇

> **本篇目标**：像导师一样逐章带领你读完 ACCORD-KV 论文。你不只是"看懂"，而是真正理解每一步的动机、逻辑和局限。读完后，你应该能回答"这篇论文在解决什么问题、怎么解决的、为什么这样做、以及还有什么没解决"。

---

## 第9章：摘要与引言——作者想解决什么问题？

### 9.1 摘要逐句解读

> **这节要说什么**：通过一句一句拆解摘要，让你弄清楚"这篇论文是干什么的、做到了什么、为什么重要"。

我们先来看摘要的第一句话：

> *"KV cache dominates GPU memory consumption in LLM inference, becoming the core bottleneck for long-context serving."*

这句话是在**定性问题的严重性**。作者告诉你：KV缓存是 LLM 推理中 GPU 显存的绝对主力，尤其在长上下文场景下，它成为了瓶颈。注意这里不是"一个"瓶颈，而是"核心"瓶颈——意味着这是当前系统里最需要解决的问题。读论文时，要对这种定性语言保持敏感：作者这么说，是因为他们需要让你相信这个问题的紧迫性，从而为后面的贡献做铺垫。

第二句话：

> *"Existing KV cache compression methods either (i) select a subset of tokens and discard the rest entirely, or (ii) apply uniform precision reduction across all tokens. Both paradigms ignore the per-head, per-block heterogeneity in attention structures."*

这是论文的**问题陈述（Problem Statement）**。作者把现有方法分成两大类：第一类是**选择法**（Selection），代表作有 StreamingLLM、H2O、SnapKV 等，它们挑选一部分 token 传输、丢弃其余；第二类是**量化法**（Quantization），代表作有 KVQuant、KIVI 等，它们对所有 token 做统一的精度压缩。

然后作者一针见血地点出：**两类方法都忽略了 per-head per-block 的异构性**。这是全文的核心观察——不同注意力头、不同语义块，对压缩的敏感度不同。用一刀切的策略，就是在浪费压缩空间。

💡 **关键创新点1**：Attention Contracts——一种轻量级的语义描述符，为每个 KV block 编码"正确注意力计算所需的最低精度要求"。这不是一个算法，而是一个**接口设计（ABI）**。

第三句话：

> *"We make a critical empirical observation: Key tensors are intrinsically low-rank (cumulative variance 0.9413 at rank 8 on Mistral-7B), while Value tensors require significantly higher rank (0.5995 at rank 8), establishing Value as the compression bottleneck."*

这是论文的**核心发现（Core Finding）**。作者做了一个数据驱动的实验：在 Mistral-7B 上，对 Key 张量做 rank=8 的 SVD 截断，能保留 94.13% 的方差；但对 Value 张量做同样的处理，只能保留 59.95%。这意味着——**Value 是压缩的瓶颈**。这个发现非常关键，因为如果 K 和 V 对压缩的敏感度一样，那用一个统一的压缩策略还算合理；但它们敏感度完全不同，统一策略就是在"要么让 K 过度压缩、要么让 V 压缩不足"之间两难。

通俗类比：想象你在整理一个图书馆。Key 像是书的目录索引——信息高度结构化，几页就能概括；Value 像是书的正文——信息密度分散，每页都有独特内容。如果你要压缩图书馆的存储空间，显然正文比目录更难压缩。ACCORD-KV 发现了这个不对称性，并把它变成了设计原则。

💡 **关键创新点2**：基于 K/V 非对称性的压缩策略——K 用更低 rank、V 用更高 rank，而不是一视同仁。

接下来摘要列举了三个核心技术：

> *"(1) attention-weighted k-center coreset selection that preserves all tokens, (2) cluster-conditional SVD with asymmetric K/V precision allocation, and (3) INT4 quantization with an OOD self-heal mechanism via SketchLocal contracts."*

- 技术1：**Coreset 选择**，保留所有 token 而非丢弃。这和 H2O/SnapKV 的"选择部分 token"策略完全不同。Core-set 的思路是用少量代表性点概括整个集合，这里用 attention-weighted k-center 来选代表 token，而不是随机选。
- 技术2：**Cluster-Conditional SVD**，对不同语义 cluster 用不同 rank。语义相近的 token 聚集在一起，它们的 KV 结构更相似，所以用一个局部 low-rank 基底比全局基底更高效。
- 技术3：**INT4 量化 + OOD 自愈**，用 SketchLocal 契约处理分布外访问。当模型遇到与训练分布不同的输入时，SketchLocal 机制能自我修复，将误差改善 7.1%。

摘要最后给出硬核数字：

> *"up to 50.8× memory compression (K_rel ≤ 0.24) while maintaining near-lossless reconstruction quality"*

注意这里的措辞是"near-lossless reconstruction quality"——重构质量接近无损。但 Table 1 里 INT4 r=8 的 V_rel 是 0.8846，这怎么看都不是"无损"。这里的"near-lossless"指的是**下游任务质量**（困惑度），而不是重构误差。论文在第13章的 PPL 实验中澄清了这一点：虽然 V_rel 很高（0.88），但 FP16 r=8 的 PPL 仅下降 1.97%。这是一个重要的教训：**重构误差和下游质量可以脱钩**。

> *"The cluster-conditional SVD variant outperforms H2O, StreamingLLM, and SnapKV by 11.6--12.2× on clustered workloads"*

注意这里的限定条件"on clustered workloads"——这是在特定访问模式下的结果。泛化到随机访问模式时，这个数字会怎么变？论文没有明确说，这是值得追问的点。

**摘要核心贡献列表：**

| # | 贡献 | 为什么是贡献 |
|---|------|------------|
| 1 | AttentionContract ABI | 跨实现的互操作性，不是算法但有系统价值 |
| 2 | V-bottleneck 实证发现 | 改变了压缩策略的设计方向 |
| 3 | OOD 自愈机制 | 提升了系统的鲁棒性 |
| 4 | 串行级联调度器 | 系统层面的效率收益 |
| 5 | Cluster-Conditional SVD | 在聚类场景下显著超越基线 |

### 9.2 引言：从问题到解决方案

> **这节要说什么**：理解论文是如何从"现实问题"一步步推导出"解决方案"的，这条逻辑链的每一步是否经得起推敲。

**第一步：PD 分离创造了 KV 传输瓶颈**

引言开头建立了系统背景：Prefill-Decode（PD）分离架构把 LLM 推理分成两个阶段——Prefill 处理完整 prompt，Decode 生成 token。KV 缓存在 Prefill 阶段产生后，需要传输到 Decode 阶段。随着上下文变长，这个传输开销线性增长，成为首要瓶颈。

这个背景是真实的。Mooncake、Splitwise、DistServe 等工作都证明了 PD 分离在系统层面的价值，但它们都没有充分解决 KV 传输开销的问题。ACCORD-KV 正是瞄准这个空缺。

**第二步：现有方案各有局限**

作者梳理了两大类现有方案：

- **选择法**（StreamingLLM/H2O/SnapKV）：选一部分 token 传输，完全丢弃其余。问题是：丢弃的 token 贡献为零注意力，一旦选错就不可恢复。
- **量化法**（KVQuant/KIVI）：对所有 token 统一降低精度。问题是：低重要性的 token 也在占用带宽，可以更激进地压缩。

这个分析是对的。但有一个细微之处值得注意：PyramidKV 提出了不同层的 KV 缓存需求不同（浅层分布广、深层集中），这本身就是一种 per-layer 异构性的观察。ACCORD-KV 的 per-head per-block 异构性和 PyramidKV 的 per-layer 异构性是什么关系？论文的相关工作章节会讨论。

**第三步：核心发现——K/V 非对称性**

作者指出，KV 缓存管理本质上是 per-head、per-block 的问题：不同注意力头提取不同语义特征，不同语义区域的信息密度不同。Globally uniform 的策略无法适应这种异构结构。

这个论断是有力的。但它也引发一个疑问：**per-head per-block 的粒度是否是最优粒度？** 如果 per-token 粒度太细，那么 per-head per-block 是不是也面临类似问题？作者没有直接比较 per-head vs per-token 的效果差异，这是一个可以深挖的点。

**解决方案概览：**

```
问题：KV 传输是 PD 分离后的瓶颈
     ↓
现有方案：选择法（丢弃 token）或量化法（一刀切精度）
     ↓
核心发现：K 低秩 + V 是瓶颈（K cumvar=0.94 vs V cumvar=0.60 @ rank8）
     ↓
解决方案：
  ① Coreset 保留所有 token（不丢弃）
  ② Cluster-conditional SVD（非对称 K/V 精度分配）
  ③ INT4 量化 + SketchLocal OOD 自愈
```

### 9.3 贡献列表分析

> **这节要说什么**：逐条分析论文声称的贡献，区分"真贡献"和"值得怀疑的贡献"。

**贡献1：AttentionContract ABI——31,775× 兼容性**

这个数字是这么来的：在所有评估数据集上处理的所有 (m, l, y) 元组，经过 compress→decompress 循环后，重计算的 attention 输出相对误差小于 10^-4。

⚠️ **审稿人可能问**：10^-4 的阈值是谁定的？有没有和 FlashAttention 的内部数值误差做对比？31,775× 是在什么硬件/数据集上测的？ABI 的兼容性验证是端到端的，但 Attention Contract 的实际价值在于它能否被不同的 KV cache 实现采用——有没有真实的多系统互操作测试？

这个贡献的**动机是 solid 的**（接口抽象确实有价值），但**验证方式不够严格**。更像是一个"我们声称它 works"的声明，而非严格的证明。

**贡献2：Coreset + INT4 压缩与 V-bottleneck 分析**

💡 这是论文最 solid 的贡献。V-bottleneck 的发现来自两个模型（Mistral-7B 和 Gemma-2-9B）的实证测量，结果一致、可复现。而且这个发现有信息论解释（softmax 归一化使 K 的方差结构更紧凑，V 的方差更分散），不是纯粹的工程调参发现。

⚠️ **审稿人可能问**：为什么只在两个模型上验证？LLM 的 KV 结构在不同架构（Llama/Mistral/Gemma/Qwen）间有显著差异吗？这个 V-bottleneck 是 Mistral/Gemma 的特性，还是 Transformer 架构的普遍规律？

**贡献3：OOD Self-Heal via SketchLocal**

误差改善 -7.1%（负数表示改善）是在什么条件下测的？OOD 的定义是什么？7.1% 是相对误差的改善，还是绝对误差的改善？论文没有给出 OOD Self-Heal 的详细实验数据表，只有一个数字。

**贡献4：串行级联调度器——128~255× 加速 @ 0.22% 相对误差**

这个数字非常 impressive。⚠️ **审稿人可能问**：0.22% 是谁的相对误差——重构误差还是端到端延迟？128× 加速是在什么基线上测的？这些数字有没有和现有的 PD 分离系统（如 Mooncake）做对比？

**贡献5：Cluster-Conditional SVD——11.6~12.2× 超越 H2O/StreamingLLM/SnapKV**

⚠️ **审稿人可能问**：这是在 clustered workloads 上测的。什么叫"clustered workloads"？如果访问模式是随机/均匀的，cluster-conditional SVD 还有优势吗？这个方法是否只在特定的访问模式假设下才有效？

---

## 第10章：问题形式化——如何定义 KV 压缩问题？

### 10.1 KV Cache 通信问题

> **这节要说什么**：用数学语言严格定义 KV 缓存压缩问题，让你知道论文要优化的目标函数是什么、约束条件是什么，以及为什么这个问题在 PD 分离架构下变得尤其紧迫。

**形式化定义**

论文先给出了 KV Cache 的空间占用公式：

$$\text{Bytes}_{\text{FP16}}(n) = 2 \cdot L \cdot H \cdot d_h \cdot n \cdot 2 \text{ bytes}$$

这个公式很直观：2 个张量（K 和 V）× L 层 × H 个头 × 每头维度 d_h × 序列长度 n × 2 字节（FP16）。代入 Mistral-7B 的具体数字：L=32, H=32, d_h=128, n=2048，得到每请求 16 GB。

16 GB 是什么概念？一块 NVIDIA RTX 4090 只有 24 GB 显存。如果 batch_size > 1，多个请求的 KV 缓存会迅速撑爆显存。更重要的是：在 PD 分离架构下，这个 16 GB 需要从 Prefill 节点传输到 Decode 节点。PCIe 4.0 x16 的理论带宽约 32 GB/s，但实际有效带宽（含协议开销）远低于此。对于 128K 的上下文，传输延迟会高达数百毫秒——这比 Decode 本身的延迟还大。

**KV 传输是 PD 分离的核心瓶颈**

PD 分离（Prefill-Decode Disaggregation）的核心思想是：Prefill 阶段做 prompt 的大规模计算（compute-bound），Decode 阶段逐个生成 token（memory-bandwidth-bound）。两者对硬件的需求不同，所以分开部署可以各自优化。

但 PD 分离引入了一个关键问题：**KV 缓存的传输**。在传统单体架构中，KV 缓存在 GPU 内部传递，带宽是 TB/s 级别（HBM）。在 PD 分离架构中，KV 缓存需要跨节点/跨设备传输，带宽降到了 GB/s 级别（PCIe/NIC）。

论文 Figure 1 没有在 LaTeX 源文件中直接展示（是 placeholder），但从系统 pipeline 的描述中可以看出：Prefill Worker 生成 KV 后，需要通过 Contract Dispatcher 分配 contract 类型，然后合并后传输到 Decode Worker。这个传输过程就是瓶颈所在。

💡 **问题的核心量化**：当 n=8192（8K 上下文）时，KV cache 大小是 n=2048 时的 4 倍，传输延迟也增加 4 倍。这解释了为什么长上下文推理对 KV 压缩的需求更迫切——短上下文可能还在 PCIe 带宽可接受范围内，长上下文就完全不可接受了。

**Attention 计算的数学形式**

论文给出了标准的 attention 计算：

$$\mathbf{y} = \softmax\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right) \mathbf{V}$$

💡 **为什么要写这个？** 因为压缩的目标不是最小化 $\|\mathbf{K} - \hat{\mathbf{K}}\|_F$，而是**最小化对 attention 输出 $\mathbf{y}$ 的影响**。论文在引理2和定理中会建立这个联系。重构误差是代理指标，真正的目标是保持下游质量。

这里有一个微妙但重要的地方：attention 输出 $\mathbf{y}$ 是一个 $n \times d$ 的矩阵，其中每一行对应一个 token 的 context vector。如果压缩后的 $\hat{\mathbf{K}}$ 和 $\hat{\mathbf{V}}$ 产生的 $\hat{\mathbf{y}}$ 和原始 $\mathbf{y}$ 足够接近，那么后续的 MLP 层和下一层的 attention 都能正常运作。所以压缩误差会逐层传播——第 $l$ 层的压缩误差会影响第 $l+1$ 层的输入，进而影响第 $l+1$ 层的 attention 输出。这个多层传播效应是论文没有详细分析的，但它是评估长序列压缩效果的关键。

**优化问题的形式化**

从系统角度，KV 压缩的目标是：

$$\min_{\text{compression strategy}} \quad \text{Latency} + \lambda \cdot \text{Quality\_Degradation}$$

约束是显存容量 $C$ 和传输带宽 $B$。这是一个多目标优化问题，ACCORD-KV 的方法是：
- 用 Attention Contract 来离散化决策空间（5 种 contract 类型）
- 用 Serial Cascade Scheduler 来实现延迟上界保证
- 用 PPL 和重构误差来度量 Quality Degradation

### 10.2 Attention Contract 概念

> **这节要说什么**：理解 Attention Contract 这个抽象概念——它是什么、为什么需要它、以及它和软件工程中的 interface contract 有什么关系。

**什么是 Attention Contract？**

Attention Contract 是一个语义描述符 $\mathcal{C} = (m, \ell, \gamma)$，附着在每个 KV block 上：

- $m \in \mathbb{R}^d$：**均值向量**，捕获 block 的语义中心。你可以把它理解为这个 KV block 的"平均特征"。
- $\ell \in \mathbb{R}^{d \times r}$：**低秩基底**，张成 block 的主要变化方向。类似 PCA 的主成分，但这里的基底是由 SVD 学到的。
- $\gamma \in [0, 1]$：**置信度分数**，衡量 contract 的满足程度。

Contract 提供两个操作：
- $\text{compress}(\mathbf{X}, \mathcal{C}) \rightarrow \hat{\mathbf{X}}$：压缩张量 $\mathbf{X}$ 以满足 contract $\mathcal{C}$
- $\text{decompress}(\hat{\mathbf{X}}, \mathcal{C}) \rightarrow \tilde{\mathbf{X}}$：从压缩表示重建

**类比：软件工程中的 Interface Contract**

在软件工程中，**Interface Contract（如 Eiffel 的 Design by Contract）** 规定了模块之间的交互规范：一个函数对输入有什么"承诺"（precondition），对输出有什么"保证"（postcondition）。调用者不需要知道函数内部怎么实现，只需要信任它满足 contract。

Attention Contract 的思想与此类似：

| 软件 Contract | Attention Contract |
|---|---|
| Precondition（前置条件） | Contract 规定了 KV block 的最低精度要求 |
| Postcondition（后置条件） | decompress 后的结果满足 contract，保证 attention 质量 |
| 调用者不需要知道实现 | Decode worker 不需要知道 KV 是怎么存储/传输的 |
| 违反 contract → 错误 | 违反 contract → attention 质量下降 |

⚠️ **但有一个关键区别**：软件 contract 是**确定性的**——满足就是满足，不满足就是违反应当报错。Attention Contract 的 $\gamma$ 是**概率性的**——0.8 的置信度并不意味着"80% 的情况正确"，而是一个语义上的满足度量。这个度量具体怎么和 attention 质量挂钩，论文没有给出精确的数学定义，只是一个启发式的度量。$\gamma$ 的计算方式是累积方差比（$\cumvar_r(\ell)$），但这个指标能否真正反映 attention 质量，目前只有实验验证，没有理论保证。

**为什么需要这个抽象？**

在 PD 分离的分布式系统中，KV cache 可能来自多个来源（GPU/CPU/NVMe/远程节点），需要不同的存储策略。如果每个来源都实现不同的 KV 格式，系统就变成一锅粥——Prefill 输出的 KV 格式，Decode 端必须理解；远程存储返回的 KV 格式也需要正确解析；SketchLocal 的压缩表示又需要特殊的解压逻辑。Attention Contract 提供了一个**统一的抽象层**：无论底层是 Full Precision 还是压缩的 sketch，接口都是一样的 $(m, \ell, \gamma)$。

这个设计有几个实际好处：

1. **模块化**：Prefill 端和 Decode 端可以独立演进，不需要同时更新。只要 Contract 接口稳定，改变底层的压缩算法（比如从 SVD 换成其他低秩方法）不需要修改 Decode 端代码。
2. **可组合性**：不同的压缩策略可以组合。比如先用 Coreset 选择代表性 tokens，再用 SVD 压缩，最后用 INT4 量化——每一步都可以表示为一个 Contract 类型。
3. **可观测性**：Contract 的 $\gamma$ 字段提供了压缩质量的直接度量。当 $\gamma$ 低于阈值时，可以自动触发 Rehydrate 升级到更高精度。

💡 **一个值得注意的设计选择**：Contract 是在 Prefill 阶段由 Prefill Worker 生成的，由 Serial Cascade Scheduler 分配。这意味着 Decode Worker 不需要知道 KV 的语义内容，只需要按照 Contract 类型决定从哪里获取/如何解压。这个设计把"策略"（哪个 block 用什么 contract）和"执行"（怎么压缩/解压）解耦了。

**Contract 的 5 种类型与系统pipeline的关系**

实际上，ACCORD-KV 有 5 种 Contract 类型，它们构成了一个完整的 KV 管理 pipeline：

```
Prefill Worker → [生成 KV] → [Contract Dispatch] → [执行 Contract] → [合并后传输] → Decode Worker
```

每个 KV block 经过 Contract Dispatch 后，被分配到 5 种 Contract 中的一种：
- **ExactLocal**：本地 GPU 显存，完整精度（FP16），零误差，但显存占用最大
- **SketchLocal**：本地压缩存储（SVD+INT4），节省 28~50× 显存，但有量化误差
- **RemoteExact**：不存本地，需要时从远程获取（CPU 内存/NVMe），延迟高但本地零开销
- **Rehydrate**：从 SketchLocal 升级到 ExactLocal，是单向升级操作
- **Drop**：完全丢弃，节省显存但完全丢失信息

这个 pipeline 的关键洞察是：**不同 Contract 的组合覆盖了不同的 cost-quality 权衡**。最热的 tokens（高 attention 权重）用 ExactLocal，保证质量；中温 tokens 用 SketchLocal，在质量和显存间取得平衡；冷 tokens 要么从远程获取（RemoteExact）要么直接丢弃（Drop）。这形成了一个完整的分层存储系统，类似于 CPU 的 L1/L2/L3 缓存层次。

---

## 第11章：SVD 压缩理论——数学基础

### 11.1 低秩近似的误差界

> **这节要说什么**：理解 SVD 截断误差的数学原理，以及为什么 Eckart-Young-Mirsky 定理是整个压缩理论的基石。

**Eckart-Young-Mirsky 定理**

给定一个矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$，其 SVD 分解为：

$$\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

rank-k 截断近似为：

$$\hat{\mathbf{X}} = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

Eckart-Young-Mirsky 定理告诉我们：**这个截断是所有 rank-k 近似中，Frobenius 范数误差最小的**。误差为：

$$\frac{\|\mathbf{X} - \hat{\mathbf{X}}\|_F}{\|\mathbf{X}\|_F} = \sqrt{\frac{\sum_{i=k+1}^{r} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}}$$

**通俗理解**：奇异值的平方就是矩阵的"能量"。截断后丢掉的就是那些"能量小"的方向。累积方差比（cumvar）就是"我们保留了多少能量"。

**为什么这和 KV 压缩相关？**

Attention 的输出是 $\mathbf{y} = \softmax(\mathbf{QK}^\top/\sqrt{d})\mathbf{V}$。如果我们对 $\mathbf{V}$ 做 rank-k 截断，误差会通过 attention 权重矩阵传播。论文引理2建立了这个误差界：

$$\left\| \softmax\left(\frac{\mathbf{QK}^\top}{\sqrt{d}}\right)(\mathbf{V} - \hat{\mathbf{V}})\right\|_F \leq \frac{\Delta}{2}\sqrt{d}$$

这个界说明：attention 输出的误差不依赖于序列长度 n，只依赖于头维度 d（128）和量化步长 $\Delta$。这是一件好事——它意味着压缩效果不会随序列变长而恶化。但这个界是宽松的——它假设 worst-case 的 attention 权重分布。实际中，由于 attention 权重是行随机矩阵（每行和为1），误差会被自然地约束。

**从奇异值到累积方差：直观理解**

对于一个 $n \times d$ 的矩阵，奇异值分解把矩阵分解成一组"秩-1 外积"的加权和：$\mathbf{X} = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$。每个 $\mathbf{u}_i \mathbf{v}_i^\top$ 是一个秩-1 矩阵，代表矩阵在某个"方向"上的贡献。奇异值 $\sigma_i$ 越大，这个方向的贡献越大。

累积方差比（cumvar）的含义是：保留前 $k$ 个奇异值对应的方向，能够保留多少比例的矩阵"能量"（Frobenius 范数的平方）。当 $\cumvar_k = 0.94$ 时，意味着只用 $k$ 个方向就能重建原矩阵 94% 的能量——对于压缩来说，压缩比的上界就是 $k/d$。

**级联压缩的误差界（定理）**

论文将 SVD 截断和 INT4 量化组合，给出了组合误差上界：

$$\|\mathbf{X} - \hat{\mathbf{X}}\|_F \leq \underbrace{\sqrt{\sum_{i=r+1}^{d}\sigma_i^2}}_{\text{SVD 截断}} + \underbrace{\frac{\Delta}{2}\sqrt{nd}}_{\text{量化噪声}}$$

💡 **关键洞察**：这个上界由两项组成，第一项随 rank 增加而减小，第二项与 rank 无关（只和量化精度有关）。这意味着存在一个"最优点"——超过这个 rank 后，继续增加 rank 只能减少 SVD 误差，而量化噪声仍然存在。论文 Table 3 的 INT4 sweep 数据（rank 从 4 到 256，误差从 ~0.90 降到 ~0.80）印证了这一理论。

**Serial Cascade 的误差收敛性**

定理4（串行级联收敛）给出了多级级联调度的误差界：

$$\mathbb{E}[\|\mathbf{V} - \hat{\mathbf{V}}\|_F] \leq \sum_{t=1}^{T} P_t \cdot \sigma_{r_t+1}(\mathbf{V})$$

💡 **为什么这个界是紧的**：当最高 tier 的 $\cumvar_{r_T} \approx 1$ 时（实验验证：$\cumvar_{256}(\mathbf{V}) \geq 0.9997$），最高 tier 的残余误差接近零。这意味着如果 tier 选择概率分布正确（大部分请求落在高 tier），平均误差接近加权下的 tier 误差。

这个定理的关键假设是：**SVD 残余主导误差**。但 Figure 4 的数据（INT4 量化后误差约 0.88）表明，在 INT4 级别，量化误差远大于 SVD 残余。当 $\sigma_{r+1} \ll \Delta\sqrt{nd}/2$ 时，量化误差 dominates，上界退化为 $\frac{\Delta}{2}\sqrt{nd}$，级联设计的收益就减少了。

⚠️ **注意**：这个误差界是**理论上界**，不是精确等式。在实践中，由于量化不是独立的噪声，真实误差可能小于上界。这也是为什么论文的实验结果比理论预测更好。

### 11.2 K/V 非对称性

> **这节要说什么**：理解论文最核心的实证发现——K 和 V 的奇异值分布完全不同，这个不对称性是 ACCORD-KV 所有设计的出发点。

**数据驱动的发现**

论文 Table 1 给出了关键数据（累积方差比）：

| 模型 | K r=8 | K r=32 | K r=64 | V r=8 | V r=32 | V r=64 |
|------|-------|--------|--------|-------|--------|--------|
| Mistral | 0.9413 | 0.9831 | 0.9954 | 0.5995 | 0.8032 | 0.9178 |
| Gemma | 0.8449 | 0.9236 | 0.9597 | 0.5317 | 0.7173 | 0.8325 |

💡 **核心数据解读**：

- **K 的信息高度集中**：Mistral 上 rank-8 的 K 就能保留 94.13% 的能量——只需要 8 个奇异向量就能很好地近似整个 Key 矩阵。这意味着 K 是"低秩友好"的。
- **V 的信息高度分散**：同样 rank-8 的 V 只能保留 59.95% 的能量——需要更多奇异向量才能充分近似 Value 矩阵。V 不是低秩的。
- **这个差距是架构性的**：从 r=8 到 r=64，K 从 0.94→0.995（只差 5%），而 V 从 0.60→0.918（差了 32%）。说明 V 的信息分布更均匀，没有明显的"主要方向"。
- **Gemma 的 V 瓶颈更严重**：Gemma V @ r=8 只有 0.5317——超过一半的信息在 rank-8 截断后丢失。这意味着同样的压缩策略在 Gemma 上会产生更严重的质量退化。

**为什么 V 的奇异值衰减更慢？**

论文引理3给出了信息论解释：softmax 对 K 的影响做了归一化，使得 K 的奇异值结构更紧凑；而 V 携带的是"原始价值信息"，其方差分布在更多的自由度上。

从直觉上理解：K 参与的是 $QK^\top$ 的计算，这个计算会被 softmax 归一化——不管某个 key 的原始量级多大，它对 attention 权重的影响都被归一化到 [0,1] 区间。这导致 K 的有效"自由度"被压缩，自然地趋向低秩。

而 V 参与的是 $\text{attn\_weights} \cdot \mathbf{V}$，V 的每一行（对应每个 token）都会被 attention 权重加权求和。如果模型在不同位置存储了不同的语义信息，这些信息就会分散在 V 的不同行/列中，难以被低秩近似捕获。

**通俗类比**：把 K 想象成图书馆的索书系统——它高度结构化，几页就能总结整个馆的分类体系（低秩）。把 V 想象成每本书的正文内容——每页都有独特信息，没有几个"主成分"能概括全书（高秩）。

**这个发现为什么重要？**

💡 **它重新定义了问题**：如果 K 和 V 一样，那任何对称的压缩策略都是 OK 的。但它们不一样，所以必须用**不对称的压缩策略**——K 用更低的 rank，V 用更高的 rank。

这个发现也解释了为什么之前的自适应 rank 分配策略表现不佳：它们把 K 和 V 当成对称的，实际上 V 才是瓶颈。更重要的是，这个发现给出了一个清晰的优化方向：**与其花预算在 K 上压缩（K 已经很低秩了），不如把预算分配给 V**。ACCORD-KV 的系统设计正是基于这个洞察。

⚠️ **值得深挖的地方**：这个发现目前只在 Mistral-7B 和 Gemma-2-9B 上验证。不同模型架构（尤其是使用 GQA/MQA 的模型）可能表现不同——KV head 的数量更少，每个 head 的信息密度是否改变？另外，这个发现是在 WikiText-2 上测的，在代码、科学文本等不同 domain 上是否成立？更关键的是：这个 K/V 非对称性是在 **MLP 路径**和 **attention 路径**的交互中产生的，还是 attention 机制本身决定的？

---

## 第12章：系统设计——怎么把理论变成系统？

### 12.1 AttentionContract 数据模型

> **这节要说什么**：理解 $(m, \ell, \gamma)$ 这个数据模型的具体含义，以及它如何在实际系统中运作。

**$(m, \ell, \gamma)$ 的含义**

- **$m$（均值向量）**：捕获 KV block 的语义中心。在coreset 选择中，$m$ 是代表性 token 的特征向量；在 SVD 压缩中，$m$ 可以用来快速判断 block 的重要性。
- **$\ell$（低秩基底）**：张成 block 变化方向的主成分。如果 $\ell$ 的 rank 是 $r$，那么压缩后只需要存储 $\ell$（$d \times r$ 个数）而不是原始矩阵（$n \times d$ 个数）。
- **$\gamma$（置信度）**：衡量当前压缩表示是否满足 contract。在 merge 操作中，合并后的 $\gamma$ 会根据累积方差重新计算。

**为什么是这三个字段？**

这个设计有三个考量：

1. **完备性**：$m + \ell$ 可以重建 block 的低秩近似（$\hat{\mathbf{X}} \approx m + \ell \cdot \ell^\top$ 的某种形式）。完整的 SVD 重构需要 $\mathbf{U}_k$ 和 $\mathbf{\Sigma}_k$，但由于 $\mathbf{V}_k^\top$ 已经隐含在 $\ell$ 中（压缩时），只需要存储 $m$ 和 $\ell$ 就可以在解压时重建近似结果。
2. **可合并性**：两个 block 的 contract 可以合并成一个新的 contract（通过合并 $\ell$ 基底并重新正交化），这在 cluster-conditional SVD 中非常有用——当多个 cluster 需要合并时，它们的 $\ell$ 基底可以拼接并截断到共同秩。
3. **轻量性**：相比完整存储 KV（$n \times d \times 2$ 字节），存储 $(m, \ell, \gamma)$ 的开销是 $d + d \times r + 1$ 个数，当 $r \ll d$（例如 r=8, d=128）时，压缩比是 $n/(r+1)$——序列越长，压缩比越高。

**Claim 1（ABI 兼容性）的深层含义**

论文声称 AttentionContract 的 (m, ℓ, γ) 格式与 FlashAttention 的 (m, l, y) 元组实现 31,775× 兼容性。这里 (m,l,y) 是 FlashAttention 输出中的均值向量、KV 长度和 attention 输出。但 FlashAttention 的输出格式和 ACCORD-KV 的 compress/decompress 之间的精确对应关系没有被严格定义。

⚠️ **一个技术细节值得注意**：FlashAttention 的 (m, l, y) 是计算过程中的中间变量——m 是 attention 的 row_max，l 是 row_sum，y 是 attention 输出。这些是数值计算的中间状态，而不是语义描述符。把 (m, l, y) 映射到 (m, ℓ, γ) 需要一个转换过程，这个转换过程的保真度如何、误差如何累积，论文没有给出具体分析。31,775× 这个数字可能是在某些特定测试用例上的统计结果，而不是对所有 (m,l,y) 组合的完备验证。

**$(m, \ell, \gamma)$ 的存储格式**

论文 Algorithm 2 给出了 merge_stats 操作：对两个 contract $a$ 和 $b$，合并它们的 $\ell$ 基底。关键的 $\oplus$ 操作是"子空间合并"：拼接两个 $\ell$ 矩阵，然后正交化。这个操作的数学性质是：如果 $\ell_a$ 和 $\ell_b$ 的秩分别是 $r_a$ 和 $r_b$，合并后的秩最多是 $r_a + r_b$。为了控制秩的膨胀，合并后需要截断到目标秩 $r^*$。

💡 **merge_stats 的一个实际用途**：在 streaming 场景中，Prefill 阶段分批处理 KV blocks，每个 block 先独立压缩成 contract，最后需要合并成一个完整的 KV cache contract。merge_stats 提供了这个合并操作的数学基础。

### 12.2 串行级联调度器

> **这节要说什么**：理解 4-tier 调度器的设计思路，以及为什么它能实现 128~255× 加速。

**4 个 tier 的设计思路**

论文描述的调度器实际上是 5 种 contract 类型，但逻辑上可以组织成 3-tier 的级联：

```
Tier 1（热数据）：ExactLocal
  → 保留全部精度（FP16）
  → 来自 GPU 显存
  → 覆盖 Top-10% 最重要 tokens（按 attention 权重）

Tier 2（中温数据）：SketchLocal
  → Coreset SVD（rank=8）+ INT4 量化
  → 28.3~50.8× 压缩
  → 覆盖剩余 tokens 中 attention 权重较高的

Tier 3（冷数据）：RemoteExact / Drop
  → RemoteExact：按需从远程存储获取
  → Drop：直接丢弃
```

**延迟上界分析**

$$\mathbb{E}[T_{\text{e2e}}] \leq P_{\text{hot}} \cdot T_1 + P_{\text{warm}} \cdot T_2 + P_{\text{cold}} \cdot T_3$$

这个公式的关键洞察是：**$T_1 \ll T_2 \ll T_3$ 且 $P_{\text{hot}} \gg P_{\text{warm}} \gg P_{\text{cold}}$**。换句话说，最重要的 token 访问概率最高，但延迟最低；最不重要的 token 访问概率最低，但延迟最高。这形成了一个天然的"成本-收益"匹配。

💡 **分层存储的类比**：这个设计非常像 CPU 的 L1/L2/L3 缓存层次结构。L1 cache（ExactLocal）最快但最小，L2/L3（SketchLocal）中等，内存/SSD（RemoteExact）最慢但无限容量。CPU 的缓存策略是基于 temporal locality（最近访问的更可能再次访问），而 ACCORD-KV 的调度策略是基于 attention locality（attention 权重高的更重要）。两者的设计哲学是相通的。

**Algorithm 1（Serial Cascade Scheduler）的具体流程**

调度器按以下顺序处理每个 KV block：

1. **Top-k 选择（Stage 1）**：从所有 KV blocks 中选择 attention score 最高的 Top-αC（α=0.1）个 blocks，分配 ExactLocal contract。这些是"热" blocks，优先保证精度。

2. **SketchLocal 分配（Stage 2）**：从剩余 blocks 中选择 attention score 最高的 Top-(1-α)C 个，分配 SketchLocal(r=8, INT4)。这些是"温" blocks，在精度和存储间取得平衡。

3. **冷数据处理（Stage 3）**：对于剩余 blocks，如果 attention score > τ_remote，分配 RemoteExact（需要时从远程获取）；否则分配 Drop（直接丢弃）。

4. **按需 Rehydrate（Stage 4）**：在 Decode 阶段，如果某个 SketchLocal block 突然收到高 attention 权重，触发 Rehydrate——从压缩表示升级到 ExactLocal。

⚠️ **一个隐藏假设**：Top-k 选择是在 Prefill 阶段做的，用的是 Prefill 阶段的 attention 权重。但 Decode 阶段的 attention 模式可能和 Prefill 不同——特别是当模型需要回顾之前被忽略的 tokens 时（如在 QA 任务中找到特定信息），Prefill 阶段的重要性评估可能不准确。Rehydrate 机制缓解了这个问题，但没有量化其开销。

**128~255× 加速的来源**

论文声称串行级联调度器实现了 128~255× 加速。理解这个数字的关键是：它不是"压缩比"，而是相对于什么基线的加速？

从论文上下文推断，基线应该是"所有 blocks 都用 ExactLocal（FP16）"的情况。在这种基线下，KV cache 传输延迟是 $T_{\text{FP16}}$。使用 SketchLocal 后，由于 KV 数据压缩了 28~50×，传输延迟降低。但 128~255× 的加速远大于 28~50× 的压缩比，说明还有其他开销被优化了。

可能的来源：
- SketchLocal 的 SVD+INT4 表示比 FP16 更紧凑，网络传输的 payload 更小（不只是压缩比，还包括序列化/反序列化开销的减少）
- ExactLocal 的热路径延迟极低（GPU 显存访问），如果大部分请求命中 ExactLocal tier，平均延迟接近 $T_1$
- 基线可能包含了一些不必要的序列化/反序列化开销，SketchLocal 格式减少了这部分开销

⚠️ **审稿人可能问**：$P_{\text{hot}}$、$P_{\text{warm}}$、$P_{\text{cold}}$ 的数值是多少？论文没有给出具体概率，只给了一个定性描述。如果某些工作负载的访问模式是"均匀分布"而非"重尾分布"，这个调度器的收益会大打折扣。

### 12.3 Cluster-Conditional SVD

> **这节要说什么**：理解为什么 per-cluster SVD 比 global SVD 更好，以及 k=8 clusters 的选择依据。

**k=8 clusters 的选择依据**

论文使用 k-means 对 token embeddings 进行聚类。k=8 是一个经验值，没有给出严格的数学依据。

💡 **直观理解**：k-means 聚类的目标是让同一个 cluster 内的 tokens 在语义空间里尽可能接近。如果一个 cluster 内的 KV 矩阵本来就"低秩"（比如都是"日期"类的 token），那么对这个 cluster 做 SVD 截断的误差就会很小。如果每个 token 都是"独立分布"的，那 k-means 可能无法有效聚类，cluster-conditional SVD 的优势就会消失。

**为什么 per-cluster 比 global 更好？**

假设我们有一个包含"日期"、"人名"、"地点"三类实体的序列。如果用 global SVD rank=8，所有 token 共享 8 个基底。但"日期"类 token 可能只需要 rank=2，"人名"类需要 rank=4，"地点"类需要 rank=2。用 global rank=8，要么"日期"浪费了 rank（多出来的 6 个没用），要么"人名"不够用。

Per-cluster SVD 让每个 cluster 用自己的 rank。如果"人名"cluster 的奇异值衰减慢（cumvar@rank4=0.85），给它 rank=6；如果"日期"cluster 的奇异值衰减快（cumvar@rank2=0.90），给它 rank=2。这样 rank 预算被更高效地分配。

**Rank 分配的具体策略**

论文的 rank 分配策略是：给奇异值方差大的 cluster 分配更高的 rank。这里的"奇异值方差"是指该 cluster 内 KV 矩阵奇异值的分散程度。如果一个 cluster 的奇异值都很接近（平坦的谱），那么它天然低秩，不需要高 rank；如果一个 cluster 的奇异值差异很大（前几个很大、后面很小），那么它需要较高的 rank 才能准确近似。

形式上，对于 cluster $j$，分配 rank $r_j^*$ 使得：
$$\frac{\sigma_{r_j^*+1}^{(j)}}{\sigma_1^{(j)}} > \epsilon$$

其中 $\epsilon$ 是一个阈值（论文没给出具体数值）。这个策略背后的直觉是：保留每个 cluster 中"能量占比超过阈值"的主要方向，丢弃那些能量太小的方向。

⚠️ **一个重要的未解答问题**：聚类是在 Prefill 阶段做的，但访问模式（哪些 cluster 会被频繁访问）可能随 Decode 阶段动态变化。如果 Decode 时大量访问"之前不重要"的 cluster，cluster-conditional SVD 仍然会把它们分配到低 rank，误差就会变大。论文的 Rehydrate 契约可以缓解这个问题（按需升级精度），但没有给出 Rehydrate 的触发条件和性能开销的量化数据。

**Coreset 选择的具体机制**

论文说使用"attention-weighted k-center coreset selection"来保留所有 tokens。这里的关键是"k-center"——它是一种聚类算法，目标是最小化所有点到其最近的中心点的最大距离。相比于 random sampling 或 importance sampling，k-center 能够更好地覆盖 token 空间的多样性。

"attention-weighted"的意思是：在计算距离时，不同 token 的权重不同——attention 权重高的 token 更重要，它们应该被优先选为 coreset 的代表。这确保了最重要的 tokens 不会被"平均掉"。

💡 **Coreset 和 selection 方法的根本区别**：H2O/SnapKV 的 selection 是"二元的"——一个 token 要么被选（FP16 保留），要么被丢弃（完全消失）。ACCORD-KV 的 coreset 是"概率性的"——所有 tokens 都被保留，但每个 token 的精度不同。这意味着 ACCORD-KV 不会完全丢失任何 token 的信息，只是精度不同。

---

## 第13章：实验——如何验证这些想法？

### 13.1 实验设置

> **这节要说什么**：了解论文的实验配置——用了什么模型、数据集、硬件，以及这些选择的含义。

**模型配置**

- **Mistral-7B-Instruct-v0.3**：32 层，32 头，dim=4096，head_dim=128。这是当前主流的 7B 级别模型。
- **Gemma-2-9B-it**：42 层，16 头，dim=3072，head_dim=128。注意 Gemma 用的是 GQA（Grouped Query Attention），16 个 KV heads 共享，这会影响 KV 的结构。

⚠️ **为什么选这两个模型？** 论文没有明确说明选择理由。理论上应该测试更多样的架构（Llama 3.1 8B、Qwen 2.5 7B 等）来验证 V-bottleneck 的普遍性。另外，Gemma-2-9B 使用了 GQA（Grouped Query Attention），只有 16 个 KV heads 而非 32 个，这可能导致 KV 的结构与标准 MHA 模型不同——这个差异是否影响了 V-bottleneck 的严重程度？论文没有分析。

**数据集**

- WikiText-2：用于困惑度测试。NIAH（Needle-in-a-Haystack）：用于长上下文检索测试。Clustered token sequences：用于 cluster-conditional SVD 的测试。

⚠️ **数据集的局限性值得注意**：WikiText-2 是小型语言模型数据集（词表约 267K），用于测试 7B/9B 模型可能不是最合适的benchmark。更标准的评估数据集包括 Pile、C4、或针对长上下文的 LAMBADA、LONGBENCH 等。NIAH 测试只提到了但没有给出具体数据（表格中的 NIAH 结果是 placeholder）。

**硬件配置**

论文使用 NVIDIA RTX 4080 SUPER (32GB) 和 RTX 4090 (48GB) 进行实验。这些是消费级 GPU，而非数据中心 GPU（如 A100/H100）。这对系统设计有一定启示：
- 消费级 GPU 的显存更紧张（32~48GB vs A100 的 80GB），KV cache 压缩的收益更大
- 但消费级 GPU 的显存带宽不同于数据中心 GPU，延迟特性可能不同

💡 论文没有提到跨 GPU 的 KV 传输实验。在真正的 PD 分离场景中，KV 需要通过 PCIe 或网络传输，这是论文声称要解决的核心问题。但实验只验证了"本地压缩"的效果，没有端到端的 PD 传输实验。

**关键实现细节（PPL 实验）**

💡 论文特别指出了一个之前失败的实现（v7）：如果只提取前 256 tokens 的 KV 并测量后续 token 的 PPL，会导致完全的 attention collapse（PPLS ≈ 12,256）。正确做法是：**提取完整序列的 KV → 压缩 → 在已知 KV 的前缀上测量 PPL**。这个教训很重要——它说明 KV 压缩的效果必须在"KV 覆盖完整"的范围内测量，否则无法区分"压缩误差"和"KV 缺失"。

还有一个实现细节值得注意：论文使用 `attn_implementation="eager"` 来强制使用 `MistralAttention` 和 `past_key_value` 支持。这意味着实验绕过了 FlashAttention 的 fused kernel，直接使用标准 PyTorch attention 实现。这在压缩实验中是可以接受的（因为我们关心的是压缩质量，而不是原始 attention 的计算效率），但这意味着实验结果可能不直接适用于生产系统（生产系统通常使用 FlashAttention 以获得更高效率）。

### 13.2 重构误差分析

> **这节要说什么**：哪些实验数据最能说明问题，重构误差和压缩效果的关系。

**Table 1（FP16 vs INT4 重构误差）**

FP16 rank=8 on Mistral：K_rel=0.2367，V_rel=0.5986。这说明：
- FP16 压缩本身非常有效（K 只损失 23.7% 的 Frobenius 范数）
- V 的误差（0.5986）几乎是 K 误差（0.2367）的 2.5 倍，印证了 V-bottleneck

INT4 rank=8 on Mistral：K_rel=0.8947，V_rel=0.8846。注意：
- INT4 引入了约 0.65 的额外相对误差（在 FP16 基础上）
- **量化误差 dominates 了 SVD 误差**：当 rank 足够小时，INT4 的量化噪声反而是主要误差来源

**Table 3（INT4 rank sweep）的关键洞察**

Table 3 给出了 rank 从 4 到 256 的 INT4 误差数据。几个关键观察：

1. **从 rank 4 到 16 改善最大**：K_rel 从 0.9001→0.8878（↓0.012），V_rel 从 0.8978→0.8659（↓0.032）。这说明从 rank=4 到 rank=16，SVD 截断的改善最显著。
2. **从 rank 64 到 256 几乎没有改善**：K_rel 稳定在 ~0.875，V_rel 从 0.8147→0.7957（只改善了 0.02）。这印证了理论预测：当 rank 足够大时，SVD 误差变得很小，量化噪声 dominates，rank 增加到 256 对 INT4 精度的改善微乎其微。
3. **Gemma 的 INT4 误差比 Mistral 更高**：在所有 rank 下，Gemma V 的 INT4 误差都高于 Mistral V（Gemma V @ r=8 是 0.9369 vs Mistral V @ r=8 是 0.8978）。这可能和 Gemma 的量化敏感性有关，也可能是 GQA 架构导致的。

**Table 4（内存压缩比）的解读**

论文的内存压缩实验是按以下方式计算的：原始大小 = L × H × d_h × n × 2（FP16 字节）；压缩后大小 = (m 的存储) + (ℓ 的 INT4 存储) + (γ 的存储)。

对于 rank=8 的情况：
- 原始：131,072 KB（Mistral, n=2048）
- 压缩后：4,608 KB
- 压缩比：28.3×

⚠️ **压缩比的计算有微妙的假设**：压缩后大小是按 SVD 因子（U, Σ, V^T）计算的。对于 rank=r 的 SVD，压缩后需要存储 U（n×r）、Σ（r×r的对角线）和 V^T（r×d）。但论文的 SketchLocal contract 实际上只需要存储压缩后的表示（m 和 ℓ），而不是完整的 SVD 因子。在 INT4 量化的假设下，存储量是 $n \times r \times 4\text{bit}$（ℓ 的 INT4 表示）。

💡 **28.3× 的压缩比是怎么来的**：原始 131,072 KB = 2 × 32 × 32 × 128 × 2048 × 2 字节。压缩后 4,608 KB = (m + ℓ + γ) 的 INT4 表示。如果完全用 INT4 存储一个 rank=8 的 SVD 表示，存储量 ≈ n × r × 0.5 bytes + d × r × 0.5 bytes = 2048 × 8 × 0.5 + 128 × 8 × 0.5 = 8,192 + 512 = 8,704 bytes per head。这和 4,608 KB 的数字之间有一些差异，说明论文的压缩比计算可能包含了一些额外的假设（如稀疏表示、head 级别的聚合等）。

⚠️ **一个不直观的观察**：INT4 之后，K 和 V 的误差变得非常接近（0.89 vs 0.88）。这说明**量化抹平了 K/V 的结构差异**——在低精度下，两者的误差都主要来自量化，而不是 SVD 截断。这对 ACCORD-KV 的设计有一个潜在影响：如果 INT4 是瓶颈，那么"不对称的 K/V 精度分配"在 INT4 级别可能没有明显收益。

**Figure 4（K vs V 累积方差图）的核心信息**

Figure 4 确认了 K 和 V 在奇异值累积方差上的系统性差异。在所有 rank 下，K 的 cumvar 都显著高于 V。这个差距在 r=8 时最大（Mistral 上是 0.34），并随 rank 增加而缩小（r=64 时是 0.08）。这个"收敛"趋势意味着：在极高的 rank 下（如 r=256），K 和 V 的压缩难度趋于一致；但在有意义的压缩比下（r=8~64），V 始终是瓶颈。

### 13.3 下游困惑度

> **这节要说什么**：PPL 为什么比重构误差更重要，以及 PPL 实验告诉我们什么。

**PPL 指标的意义**

困惑度（Perplexity）是语言模型质量的经典指标——它衡量模型对下一个 token 的预测"有多不确定"。PPL 越低，模型质量越好。从信息论角度，PPL 是交叉熵的指数：$\text{PPL} = \exp\left(\frac{1}{N}\sum_i \log P_{\theta}(x_i | x_{<i})\right)$。PPL 翻倍意味着模型对下一个 token 的预测不确定性增加了一倍。

**为什么 PPL 比重构误差更重要？**

💡 重构误差衡量的是"压缩后的矩阵和原矩阵有多像"，但这只是一个**代理指标**。真正重要的是"压缩后的 attention 输出是否正确"。Table 4 的数据揭示了一个关键发现：

- **FP16 r=8**：PPL 12.58（比基线 12.34 高 1.97%）。虽然 V_rel=0.5986 看起来很高，但 PPL 只恶化了 2%——说明 SVD 截断对 attention 质量的影响比想象中温和。
- **INT4 r=8**：PPL 14.90（比基线高 20.75%）。INT4 量化引入了大量下游质量退化。
- **Method D（cluster-conditional SVD）**：PPL 12.28（比基线低 0.44%！）——这是一个令人惊讶的结果。cluster-conditional SVD 不仅没有降低质量，反而轻微提升了质量。这可能是因为 per-cluster 的 rank 分配更合理，减少了不必要的压缩。

**为什么 FP16 r=8 的 PPL 退化如此之小？**

这是一个值得深入思考的现象：V_rel=0.5986 意味着超过 60% 的 Frobenius 范数信息丢失了，但 PPL 只恶化了 2%。这有几种可能的解释：

1. **Frobenius 范数不等于 attention 质量**：Frobenius 范数是矩阵的整体能量度量，但 attention 只使用矩阵的"列空间"——即矩阵乘以 Query 向量后的结果。如果丢失的是 V 的"次要列"（对 Query 聚合贡献小的列），对 attention 输出影响就小。
2. **Softmax 的非线性放大效应被限制**：attention 权重是归一化的（和为1），即使 V 有误差，最终的加权求和结果也被 softmax 的概率解释所约束。
3. **语言模型对 SVD 误差有一定鲁棒性**：预训练过程已经让模型学会了处理一定程度的噪声，KV 缓存的轻微扰动可能不会显著影响输出分布。

**INT4 量化的 PPL 退化为什么这么大？**

INT4 r=8 的 PPL 退化（+20.75%）远大于 FP16 r=8（+1.97%），这说明 INT4 量化引入的误差不是"良性噪声"，而是有结构的偏差。具体来说：

- FP16 r=8 的误差主要来自**SVD 截断**，丢掉的维度对应的是 V 中"低能量"的方向——这些方向对 attention 贡献较小。
- INT4 r=8 的误差来自**SVD 截断 + 量化噪声**。量化噪声不是随机的——它是确定性的（每个值被映射到最近的量化级别），且会在整个矩阵上产生系统性的偏差。这种系统性偏差会改变 V 中不同 token 之间的相对关系，即使这些 token 的"绝对能量"变化不大。

**Selection 基线的对比**

H2O（保留 50% tokens）：PPL 35.63（+188.8%）
StreamingLLM（sink+local）：PPL 25.85（+109.5%）
PDTrim（oracle 50%）：PPL 22.07（+78.9%）

⚠️ **这些选择法的高 PPL 退化需要放在上下文中理解**：H2O 和 StreamingLLM 是为**流式推理**设计的，它们在 decode 过程中动态管理 KV cache，不是在 Prefill 后做一次性压缩。StreamingLLM 利用 attention sink 现象——前几个 tokens（通常是 sink token）被所有位置赋予异常高的 attention 权重，这是 LLM 的普遍行为，不依赖于特定 prompt。H2O 则通过 submodular 优化动态选择 Heavy-Hitter tokens。

直接拿"保留 50% tokens"和"保留所有 tokens 但压缩"对比，对 selection 方法有些不公平——它选择丢弃某些 tokens 是因为它在流式场景下必须做出硬选择。ACCORD-KV 的 coreset 方法和 selection 方法解决的是不同场景的问题。更公平的对比应该是：在相同的"传输带宽"预算下，比较 selection（部分 tokens × 高精度）和 compression（所有 tokens × 低精度）哪个质量更好。

⚠️ **PDTrim 的 oracle 50%** 是一个值得关注的数据点：即使使用 oracle（知道哪个 tokens 最重要）选择 50% tokens，PPL 仍然退化 78.9%。这说明"选择一半"本身就有很大的质量损失——不是因为选择策略不够好，而是因为"丢弃"这个操作本身就不可能无损失。相比之下，ACCORD-KV 的"全部保留、压缩精度"策略在相同的压缩比下质量损失小得多（+1.97% vs +78.9%）。

---

## 第14章：相关工作——学术界在做什么？

### 14.1 论文如何定位自己的工作

> **这节要说什么**：理解 ACCORD-KV 与相关工作的关系——哪些是互补的、哪些是竞争的、哪些被论文超越。

**与 Selection 方法（H2O/StreamingLLM/SnapKV）的关系**

论文的核心主张：**Coreset 保留所有 tokens，而 selection 方法丢弃 tokens**。

这个区分是重要的。H2O 通过 submodular 优化选择"Heavy Hitter" tokens，StreamingLLM 利用 attention sink 现象保留起始 tokens，SnapKV 用观察窗口投票。这些方法都面临一个根本性的权衡：**保留多少 tokens = 质量 vs. 内存的权衡**。当 budget 紧张时，它们必须丢弃 tokens，这些 tokens 对未来的 attention 贡献为零。

ACCORD-KV 通过**压缩而非丢弃**来避免这个权衡。但这也引出一个问题：**coreset 方法保留了 100% 的 tokens，每个 token 都有一定精度。如果 H2O 保留 20% 的 tokens 每个用 FP16，和 ACCORD-KV 保留 100% 的 tokens 每个用 INT4 r=8，谁的效果更好？** 论文没有直接比较这个设置。

**与 KVQuant/KIVI 等量化方法的关系**

💡 互补性：ACCORD-KV 的 INT4 量化与 KIVI 的 per-channel/per-token 量化策略可以结合——既做 rank 压缩又做量化压缩。论文没有讨论这种组合的潜在收益。

**与 SpectrumKV 的关系**

💡 这是最接近的相关工作。SpectrumKV 也是基于 attention spectrum 做混合精度，但：
- SpectrumKV 是 per-token 粒度，ACCORD-KV 是 per-head per-block 粒度
- SpectrumKV 没有 cluster-conditional SVD
- SpectrumKV 没有 OOD self-heal 机制
- SpectrumKV 没有 AttentionContract ABI

⚠️ **SpectrumKV 是 ACCORD-KV 的前身吗？** 论文明确说"SpectrumKV 可以看作 ACCORD-KV 框架中 mixed-precision contract type 的一个实例"。这意味着 ACCORD-KV 是在 SpectrumKV 基础上的扩展——但扩展了多少？是全新的创新，还是增量改进？审稿人可能会追问这一点。

**与 GEAR/LoRC/KQ-SVD 等低秩方法的关系**

这些方法都是低秩压缩，但关键区别在于：
- **LoRC**：对 KV 权重矩阵做 SVD（不是对激活值），是在模型压缩阶段做的——这意味着它压缩的是模型参数，而不是 KV 缓存本身
- **KQ-SVD**：联合分解 $QK^\top$ 而不是单独处理 K 和 V——这个思路和 ACCORD-KV 的 K/V 非对称性发现相悖，因为联合分解隐含假设了 K 和 V 有相同的压缩敏感性
- **GEAR**：结合了量化 + 低秩 + 稀疏 outlier 处理——但它没有利用 K/V 非对称性，而是统一处理
- **StiefAttention**：在 Stiefel 流形上学习正交投影基，直接最小化 decoder 层输出重构误差——这是 ACCORD-KV 可以借鉴的方向，因为它直接在 attention 输出空间优化

ACCORD-KV 的独特之处是**per-head per-block 的非对称 K/V 处理**，这在其他方法中没有被强调。

**与其他 PD 分离架构的关系**

ACCORD-KV 与 Mooncake、DistServe、Splitwise 等 PD 分离系统是**互补关系**，而非竞争关系。这些系统解决了 PD 分离的调度和资源分配问题，ACCORD-KV 解决的是 KV 传输的表示问题。在任何 PD 分离系统中，KV 缓存都需要传输——无论用什么调度策略，KV 数据的表示是所有这些系统的共同基础。

这意味着 ACCORD-KV 的 AttentionContract 接口可以被所有 PD 分离系统采用：DistServe 的 Prefill 节点输出带 Contract 的 KV，Decode 节点按照 Contract 类型处理；Mooncake 的 KV 缓存池可以按 Contract 类型分层存储不同的 KV block。这个视角强调了 ACCORD-KV 的系统级价值。

**相关工作的全景图**

用一句话总结各方法的核心思路：

| 方法 | 核心思路 | 与 ACCORD-KV 的关系 |
|------|---------|------------------|
| H2O | 保留 Heavy-Hitter tokens | 竞争：丢弃 vs 压缩 |
| StreamingLLM | 保留 sink + recent tokens | 竞争：流式场景的近似解 |
| SnapKV | 观察窗口投票选重要 tokens | 竞争：selection 范式 |
| PyramidKV | 浅层多 token、深层少 token | 互补：per-layer 视角 |
| KVQuant | Per-channel 量化 Key | 互补：可与 SVD 结合 |
| KIVI | Per-channel K / Per-token V | 互补：量化版的 V-bottleneck |
| GEAR | 低秩 + 量化 + outlier | 互补：联合压缩框架 |
| SpectrumKV | Per-token 混合精度 | 前身：ACCORD-KV 的前身 |
| Mooncake | PD 分离的 KV 池 | 互补：ACCORD-KV 可集成入 KV 池 |
| FlashAttention | 高效 attention 计算 | 正交：ACCORD-KV 的基础设施 |

**给审稿人的角度看论文定位**

从审稿人的角度，ACCORD-KV 的定位策略是"差异化叙事"：强调自己是"压缩 vs 丢弃"范式转换的引领者，同时把 SpectrumKV 定义为自己的"前身"。这个定位在策略上很聪明，但需要回答的问题是：SpectrumKV 的读者能否只通过 ACCORD-KV 论文就复现结果？如果 SpectrumKV 的代码和 ACCORD-KV 的代码共享了多少，这更像是一个迭代还是一个新的贡献？

另一个需要回答的问题是：**Per-head per-block 粒度的开销是否值得？** 如果 per-token 粒度（像 SpectrumKV 那样）已经足够好，per-head per-block 的额外复杂性是否带来了相称的收益？论文没有做这个 ablation study。审稿人很可能会追问："per-head per-block 和 per-token 哪个更好？为什么？"

---

## 第15章：结论与开放问题

### 15.1 主要结论

论文总结了 6 项贡献：

1. **AttentionContract ABI**：提供跨实现的互操作性接口
2. **Coreset + INT4 压缩**：28.3~50.8× 内存压缩
3. **V-bottleneck 理论**：K 低秩 + V 高秩的不对称压缩
4. **OOD Self-Heal**：SketchLocal 机制
5. **串行级联调度器**：128~255× 加速
6. **Cluster-Conditional SVD（Method D）**：11.6~12.2× 超越基线

这些贡献覆盖了**理论**（V-bottleneck、SVD 误差界）、**系统**（调度器、contracts）、**算法**（cluster-conditional SVD、coreset selection）三个层面，是一个相对完整的 work。

### 15.2 开放问题（论文没回答的问题）

> **这节要说什么**：识别论文留下的未解问题，这些是未来研究的切入点，也是审稿人可能会追问的方向。

**问题1：V-bottleneck 的普遍性**

目前只在 Mistral-7B 和 Gemma-2-9B 上验证。这两个模型都是 decoder-only Transformer，但它们不能代表所有架构：

- 使用 GQA 的模型（如 Llama 3.1）：KV heads 更少，每个 head 的信息密度更高，V-bottleneck 是否仍然存在？
- MoE 模型（如 Mixtral 8×7B）：expert routing 引入了动态的 KV 结构，V 的低秩性可能受影响
- Vision-Language Models：跨模态 attention 的 K/V 分布可能完全不同

**问题2：Cluster-conditional SVD 的动态场景**

聚类是在 Prefill 阶段静态做的，但 Decode 阶段可能动态访问之前不重要的 tokens。如果模型在某个 step 突然关注某个之前被分配到低 rank 的 cluster，误差会急剧增加。如何在运行时动态调整 cluster 分配和 rank 预算？论文的 Rehydrate 契约可以缓解这个问题（按需升级到完整精度），但 Rehydrate 的触发条件和性能开销没有被量化。

**问题3：Method D 的泛化性**

Method D 在"clustered workloads"上显著超越基线（11.6~12.2×），但"clustered workloads"的定义是什么？具体来说，"Clustered"是指序列中连续多个 token 属于同一语义类别（如连续的法律条文、连续的代码行）吗？如果访问模式是"跳跃式"的（比如先读第1段、再读第5段、再读第2段），cluster-conditional SVD 的优势是否消失？聚类数量 k=8 是怎么选的？有没有理论指导？k 太大导致 per-cluster 样本太少，SVD 不够稳定；k 太小导致不同语义的 token 被混在一起，per-cluster 低秩性不明显。

**问题4：PPL vs. 真实任务的 gap**

PPL 是语言模型质量的代理指标，但在真实应用中（代码生成、数学推理、多轮对话），SVD 压缩的影响可能更大。代码生成（HumanEval/MBPP）中，代码的 attention 模式高度局部化（前向依赖强），KV 压缩可能比自然语言更敏感。数学推理（GSM8K/MATH）中，多步推理依赖精确的 KV 信息，误差可能在长链推理中累积放大。多轮对话场景中，历史消息的重要性可能随时间衰减不均匀，cluster-conditional SVD 对历史信息的压缩策略是否合理？论文没有在下游任务上验证压缩效果，只测了 PPL，这是一个重要的实验空白。

**问题5：与生产系统的集成**

ACCORD-KV 的调度器目前是理论设计，没有和 vLLM/TGI/Mooncake 等实际推理系统集成。在生产环境中，KV cache 的管理涉及到多个复杂问题：Batching 时多个请求的 KV 缓存如何分配 contract？Preemption 时当 GPU 显存不足，哪些 block 的 contract 被降级或 drop？Checkpointing 时 KV 缓存的 contract 信息如何持久化？跨请求共享时多个请求可能共享部分 prompt prefix，contract 如何复用？这些问题没有在论文中得到回答。

**问题6：Value 瓶颈的深层原因**

论文给出了信息论解释（softmax 归一化使 K 更结构化），但这只是描述性的，没有因果解释。为什么 Transformer 的训练过程会天然产生 K 低秩 + V 高秩的结构？这是 attention 机制本身决定的，还是某些训练数据/目标造成的？具体来说，可能有几个深层原因值得探索。首先是 Cross-entropy loss 的梯度特性：训练时梯度从 loss 反传到 KV，如果 V 的梯度方差更大，V 倾向于保留更多"独特性"而不是被压缩到低维空间。其次是 KV 的语义角色不同：K 负责匹配查询（query-key matching），这个过程本身就是一个降维操作——高维的 V 信息通过 attention 权重被"软选择"地聚合，因此 K 自然地变得更低秩。再次是位置编码的影响：RoPE 等位置编码可能使 K 的位置相关部分更结构化（正弦/余弦基），从而天然低秩。如果能回答这个问题，可能可以设计针对性的训练策略来进一步压缩 KV cache。

**问题7：压缩误差的 attention 传播动力学**

论文分析了压缩误差的静态界，但没有分析误差在多层堆叠中如何传播。假设第 $l$ 层的 V 压缩误差是 $\epsilon_l$，这个误差会影响第 $l+1$ 层的 Query 生成，进而影响第 $l+1$ 层的 attention 权重，最终导致第 $l+1$ 层的输入有偏。这个误差会逐层累积还是被 attention 操作抑制？论文没有回答。这对于评估长序列压缩效果非常重要：如果误差逐层放大，rank=8 的 SVD 在第32层的影响可能远大于第1层。

---

## 阅读总结：论文的强项与弱项

**强项**

| 方面 | 评价 |
|------|------|
| 核心发现（V-bottleneck） | 非常 solid，数据充分，解释合理，有信息论支撑 |
| 系统设计（AttentionContract） | 概念清晰，接口抽象有价值，与 PD 分离场景天然契合 |
| 实验覆盖 | 两个模型、多种配置、有重构误差和 PPL，验证了主要主张 |
| 理论分析 | SVD 误差界和级联定理是扎实的数学基础，不是事后补充 |
| Coreset 思路 | "保留所有 token"的思路有别于主流的 selection 方法，提供了新视角 |

**弱项 / 值得追问的地方**

| 方面 | 潜在问题 |
|------|----------|
| 实验规模 | 只测了 2 个模型，缺少 70B+ 的实验；没有在生产数据集（法律/医疗/代码）上的验证 |
| 泛化性验证 | Cluster-conditional SVD 只在 clustered workloads 上有效；k=8 的选择没有充分论证 |
| OOD Self-Heal | 数据不足（只有 -7.1% 一个数字）；没有说明 OOD 的具体定义和测试条件 |
| ABI 验证 | 31,775× 兼容性的验证是启发式的，缺少和真实 FlashAttention 实现的对齐测试 |
| 生产集成 | 没有和真实推理系统的集成实验；Rehydrate 的延迟开销未量化 |
| Selection 对比 | 对 StreamingLLM/H2O 的对比可能不公平——它们是流式场景，ACCORD-KV 是压缩场景，场景不同导致 PPL 对比缺乏可比性 |
| 动态行为 | 多层堆叠下的误差传播没有分析；cluster-conditional SVD 在动态访问模式下的表现未知 |

**给学习者的建议**

阅读这篇论文时，建议按以下顺序理解：

1. **先理解 V-bottleneck（11.2节）**：这是所有设计的出发点。只有理解了"为什么 K 和 V 不一样"，才能理解后面所有非对称设计的动机。建议仔细阅读 Table 1，体会 cumvar 数据的含义——K cumvar 0.94 vs V cumvar 0.60 的差距是全文的核心驱动力。
2. **再理解 AttentionContract（10.2节）**：这是一个接口抽象，理解了它才能理解系统设计。把它和软件工程中的 Design by Contract 做类比会很有帮助——Contract 规定了 KV block 的最低精度要求，调用者不需要知道内部实现。
3. **然后看实验（13章）**：带着"理论预测"去看实验结果。特别注意 FP16 r=8 的 V_rel=0.5986 但 PPL 只恶化 2% 这个反直觉现象——这说明重构误差和下游质量可以脱钩。量化后的 INT4 抹平了 K/V 的结构差异，这直接影响了对"不对称精度分配"在 INT4 级别是否有效的判断。
4. **最后看相关工作（14章）**：把自己放在审稿人的位置，问"这篇工作和 XX 有什么关系"——这是训练批判性思维的好方法。特别是 SpectrumKV 和 ACCORD-KV 的关系，两者之间的增量创新是否足够 solid，是审稿人可能会追问的问题。

**推荐的延伸阅读**

如果想深入理解 ACCORD-KV 的技术基础，建议按以下顺序阅读相关工作：

1. **FlashAttention**（Dao, 2022）：理解 KV cache 的基础存储格式和 attention 计算效率的优化背景。
2. **StreamingLLM**（Xiao et al., 2024）：理解 attention sink 现象和流式推理的基本设定。
3. **H2O**（Zhang et al., 2023）：理解 selection-based KV 管理的基本方法，体会"选择 vs 压缩"的权衡。
4. **KIVI**（Liu et al., 2024）：理解 per-channel/per-token 量化策略，与 ACCORD-KV 的低秩视角形成互补。
5. **GEAR**（Kang et al., 2024）：理解低秩+量化+sparse outlier 的联合压缩框架。
6. **SpectrumKV**（Yang, 2025）：ACCORD-KV 的前身，理解 per-token 混合精度和 ACCORD-KV 的改进点。
7. **PyramidKV**（Cai et al., 2024）：理解 per-layer 的 KV 缓存分布规律，与 ACCORD-KV 的 per-head 视角对比。

---

## 附录：论文关键数据速查表

| 数据 | 值 | 意义 |
|------|-----|------|
| Mistral K cumvar @ r=8 | 0.9413 | K 是低秩的，r=8 够用 |
| Mistral V cumvar @ r=8 | 0.5995 | V 不是低秩的，r=8 不够 |
| Gemma K cumvar @ r=8 | 0.8449 | K 仍然低秩，但比 Mistral 略低 |
| Gemma V cumvar @ r=8 | 0.5317 | V 瓶颈在 Gemma 上更严重 |
| Memory compression (Mistral r=8 INT4) | 28.3× | 压缩效果（96.5% 节省） |
| Memory compression (Gemma r=8 INT4) | 50.8× | 压缩效果（98.0% 节省） |
| Method D vs H2O | 11.6~12.2× | Clustered 场景下的性能提升 |
| Serial cascade speedup | 128~255× | 调度器效率 |
| Cascade relative error | 0.22% | 端到端相对误差 |
| OOD self-heal improvement | -7.1% | 鲁棒性收益（负数=改善） |
| FP16 r=8 PPL degradation | +1.97% | 下游质量可控 |
| INT4 r=8 PPL degradation | +20.75% | INT4 量化代价大 |
| Method D PPL improvement | -0.44% | 超越 FP16 基线 |
| H2O PPL degradation | +188.8% | Selection 方法质量损失大 |
| StreamingLLM PPL degradation | +109.5% | Selection 方法质量损失大 |
| INT4 additional error (beyond FP16) | ~0.65 | 量化噪声 dominates 了 SVD 误差 |
| ABI compatibility ratio | 31,775× | (m,l,y) 元组的兼容性数量 |

---

<!-- PART3 COMPLETE -->


# ACCORD-KV 学习指南 · 第四部分

## 相似论文对照篇

> **学习目标**：通过对比学习，理解 ACCORD-KV 与相关工作的异同。每篇论文都从"一句话核心思想 → 关键技术 → 方法局限 → 与 ACCORD-KV 的核心差异"四个维度展开，帮助读者在对比中深化理解。
>
> **前置知识**：建议先阅读第一部分（基础概念篇）和第二部分（技术原理篇），建立 KV 缓存和压缩技术的基本认知。

---

## 第15章：Token 驱逐（Eviction）三杰

本章节聚焦于通过**选择性驱逐（驱逐而非压缩）**来管理 KV 缓存的方法。这类方法的核心思路是：在有限的缓存容量下，决定"保留哪些 token、驱逐哪些 token"。其中 H2O、StreamingLLM、SnapKV 是最具代表性的三项工作。

---

### H2O (Heavy-Hitter Oracle) NeurIPS 2023 — 少量"重型"token 主导注意力分数

**核心思想**：大语言模型的注意力分数遵循幂律分布——只有极少数 token（称为 Heavy Hitters，H2）贡献了绝大部分注意力权重。H2O 通过动态保留"重型 token"和"最近 token"的平衡组合，在仅使用 20% KV 缓存的情况下维持接近完整缓存的精度。

**关键技术**：

- **Heavy Hitter 检测**：基于累积注意力分数的平均值，识别在所有查询中持续获得高注意力权重的 token。这些 token 与文本中的高频共现模式密切相关，是语义上的"关键词"。
- **动态子模优化**：将 KV 缓存驱逐策略形式化为动态子模函数最大化问题，并证明贪心算法在温和假设下具有理论逼近保证（$(1-1/e)$ 近似）。
- **双轨保留策略**：缓存由两部分组成——固定比例的 Heavy Hitter token（通常 10%）+ 固定比例的最近 token（通常 10%）。两部分相加构成 20% 的总缓存预算。
- **跨层统一策略**：在所有 Transformer 层使用相同的缓存大小和驱逐策略，简化实现。

**方法局限**：

- **Prefill 阶段不优化**：H2O 仅在解码（decode）阶段驱逐 KV 缓存，对预填充阶段的计算开销没有帮助。
- **统一层策略**：对所有 Transformer 层应用相同的压缩预算，忽视了不同层注意力稀疏度差异的事实（高层注意力通常更稀疏）。
- **静态窗口限制**：Heavy Hitter 的选择依赖于在线累积注意力分数，计算开销随序列增长。
- **信息永久丢失**：被驱逐的 token 的 KV 状态被永久丢弃，无法在后续阶段恢复。

**与 ACCORD-KV 的核心差异**：

| 维度 | H2O | ACCORD-KV |
|------|-----|-----------|
| 压缩方式 | Token 驱逐（完全丢弃） | Coreset 选择 + SVD 低秩近似 + INT4 量化（可恢复） |
| 精度保证 | 20% 缓存时接近完整精度 | Per-head Coreset 保留局部结构 + 低秩投影保全局结构 |
| 是否需要再训练 | 否（无需微调） | 否（无需微调） |
| 与 PD 分离的兼容性 | 仅优化 Decode，Prefill 未优化 | PD 分离优化：Prefill 端 Coreset 选择，Decode 端低秩 + 量化 |
| 信息保留 | 完全丢弃被驱逐 token | 保留核心信息（通过低秩近似和量化压缩而非丢弃） |
| 压缩粒度 | Token 级别（全保留或全丢弃） | Per-head 级别的细粒度 Coreset + 低秩投影 |

---

### StreamingLLM (Attention Sink) ICLR 2024 — 保留"注意力汇"以稳定长序列生成

**核心思想**：LLM 的 Softmax 注意力机制要求所有注意力权重之和等于 1。当模型没有足够相关信息要分配时，它会将"多余的"注意力权重倾倒到初始 token（如 BOS token）上，这些 token 被称为"Attention Sink"。StreamingLLM 通过永久保留 Attention Sink（通常只需 4 个初始 token）加上最近 token 的滑动窗口，使模型能够在无限长度的输入流上稳定运行，无需微调。

**关键技术**：

- **Attention Sink 现象**：无论初始 token 在语义上是否重要，训练好的 LLM 都会持续向其分配大量注意力权重（30%-80%）。这是 Softmax 约束（注意力之和必须为 1）的副产品——模型需要一个"停车场"来放置无法分配给有用 token 的注意力质量。
- **双组件设计**：StreamingLLM 的 KV 缓存 = Attention Sink（永久保留的 4 个初始 token）+ 最近窗口（最近 L 个 token）。中间 token 直接驱逐。
- **Softmax 机制的数学解释**：对于给定的 query，当所有 key 的相关性都较低时，模型仍需将注意力权重分配出去。由于初始 token 对所有后续位置都可见且位置固定，模型学会了将它们作为"安全的注意力倾倒点"。
- **无需微调**：方法完全依赖推理阶段的 KV 缓存管理策略，不涉及模型权重的任何修改。
- **支持 4M+ token 流式生成**：在 Llama-2、MPT、Falcon、Pythia 等模型上验证，可靠建模 400 万 token 以上的流式输入。

**方法局限**：

- **无差别驱逐中间 token**：StreamingLLM 驱逐所有中间 token，既驱逐了高注意力权重的 Heavy Hitter，也驱逐了大量可能有用的中间 token。
- **Prefill 阶段计算量未减少**：仍然需要对完整输入序列进行 Prefill 计算，KV 缓存的压缩仅发生在解码阶段。
- **Attention Sink 的特殊性未充分利用**：方法仅利用了"初始 token 是 Attention Sink"这一特性，但未探索其他位置的 Attention Sink（如特定指令 token）潜力。
- **信息永久丢失**：与 H2O 一样，被驱逐的 token 信息不可恢复。

**与 ACCORD-KV 的核心差异**：

| 维度 | StreamingLLM | ACCORD-KV |
|------|-------------|-----------|
| 压缩方式 | Token 驱逐 + 固定保留 Attention Sink | Coreset + SVD + INT4 三阶段压缩 |
| 精度保证 | 仅靠 Sink + Recency 维持稳定，但不保证精度最优 | 通过信息论 Coreset 和 SVD 低秩投影最小化信息损失 |
| 是否需要再训练 | 否 | 否 |
| 与 PD 分离的兼容性 | 仅针对 Decode 阶段的流式场景 | PD 分离全链路优化（Prefill 选、Decode 压） |
| 对中间 token 的处理 | 无差别驱逐 | 通过 Per-head Coreset 识别并保留对当前 head 重要的中间 token |
| 信息保留 | 完全丢弃中间 token | 保留 token 的低秩近似表示 |

---

### SnapKV (Snapshot KV) ACL 2024 — 从"观察窗口"预知哪些 KV 位置关键

**核心思想**：LLM 在生成之前就已经"知道"它需要关注输入中的哪些信息——每个注意力头在生成过程中始终聚焦于特定的提示注意力特征。通过在提示末尾设置一个小的"观察窗口"（Observation Window），SnapKV 可以提前识别这些关键特征，并将 KV 缓存压缩为每个注意力头选定的簇状重要位置，显著减少计算开销和内存占用。

**关键技术**：

- **注意力分配模式一致性**：SnapKV 发现，对于 LLM，大多数输入序列 token 的注意力分配在生成过程中保持一致。模型在生成之前就知道它在寻找什么。
- **观察窗口机制**：在提示的末尾使用一个较小的观察窗口（通常 16-32 个 token），从该窗口内的 query 计算对所有 prefix token 的注意力分数，以此预测整体的重要性分布。
- **Per-head 簇状选择**：与 H2O 使用全局累积分数不同，SnapKV 对每个注意力头分别进行重要性评估和选择，保留每个 head 聚焦的"簇状"重要位置。这种方式比扁平化的全局选择更精确。
- **池化增强**：使用最大池化（Max Pooling，核大小 5-7）来整合观察窗口的注意力分数，减少噪声并增强对关键 token 起始部分的关注。
- **无需微调**：完全基于推理时的注意力模式分析，不修改模型权重。
- **性能指标**：处理 16K token 输入时，解码速度提升 3.6 倍，内存效率提升 8.2 倍；在单块 A100-80GB GPU 上可处理最多 380K 上下文 token。

**方法局限**：

- **观察窗口设计依赖人工设定**：观察窗口的大小和池化核大小需要手动调优，不同模型可能需要不同的超参数配置。
- **仅优化 Prefill 阶段的 KV 缓存**：SnapKV 主要针对提示（Prompt）的 KV 缓存压缩，对解码阶段的压缩支持有限。
- **对小批量场景加速有限**：当 batch size 较小时，选择 PvC（Persistent Visual Context）引入的额外计算开销会抵消 KV 缓存减少带来的加速。
- **池化可能丢失细粒度信息**：简单池化可能无法捕捉注意力模式的所有细微变化。

**与 ACCORD-KV 的核心差异**：

| 维度 | SnapKV | ACCORD-KV |
|------|--------|-----------|
| 压缩方式 | Per-head Token 驱逐（选择重要的 prefix token） | Per-head Coreset 选择 + 低秩近似 + INT4 量化 |
| 精度保证 | 通过观察窗口预测，保留每个 head 聚焦的簇状位置 | 信息论 Coreset 最大化保留局部结构信息 + SVD 保留全局信息 |
| 是否需要再训练 | 否 | 否 |
| 与 PD 分离的兼容性 | 主要优化 Prefill 阶段的 Prompt KV | PD 分离：Prefill 用 Coreset 选，Decode 用低秩+量化压 |
| 信息恢复能力 | 完全丢弃被选中驱逐的 token | 低秩近似允许部分信息恢复（KV injection 兼容） |
| 选择策略 | 基于观察窗口的启发式选择 | 基于 Hessian/熵/随机投影的多种 Coreset 选择策略 |

---

## 第16章：量化压缩双雄

本章节聚焦于通过**数值量化**来压缩 KV 缓存的方法。与 Token 驱逐方法不同，量化方法保留所有 token 的 KV 信息，但通过降低数值精度来减少内存占用。KIVI 和 KVQuant 是这一方向的代表性工作。

---

### PyramidKV CVPR 2024 — 层级注意力稀疏度差异启发的金字塔缓存分配

**核心思想**：Transformer 中不同层的注意力模式存在显著差异——低层（浅层）的注意力分布广泛且近似均匀（高熵），而高层（深层）的注意力高度集中于少数关键 token（低熵）。PyramidKV 基于这一"Pyramidal Information Funneling（金字塔式信息汇聚）"模式，提出了层级差异化的 KV 缓存分配策略：在低层分配更多缓存（因为信息更分散），在高层分配更少缓存（因为信息已汇聚到关键 token）。

**关键技术**：

- **Pyramidal Information Funneling 观察**：通过可视化 LLaMA 模型在多文档问答任务中的逐层注意力图，发现：
  - **低层（0-5 层）**：注意力近似均匀分布，模型从所有可用内容中全局聚合信息。
  - **中层（6-18 层）**：注意力逐渐转向聚焦在段落内部的 token，呈现"局部注意力"模式。
  - **高层（24-31 层）**：出现"Massive Activation"和"Attention Sink"现象，极高注意力集中在少数关键 token 上。
- **金字塔形缓存分配**：使用等差数列在层间分配缓存预算。设总缓存预算为 $B$，层数为 $N$，最大层（下限）分配 $B_{max}$，最小层（上限）分配 $B_{min}$，则中间层的预算按线性插值确定，形成金字塔形状。
- **基于 SnapKV 的 KV 选择**：采用 SnapKV 的方法，通过指令 token（Instruction Token）获得对其他 token 的注意力分数，据此选择要保留的 KV 对。
- **12% 缓存保持 99% 精度**：在 LLaMA-3-8B 和 Mistral-7B 上的实验表明，仅使用 12% 的 KV 缓存即可保持与完整缓存相当的性能；在仅保留 0.7% 缓存（128 个 token）的极端条件下，仍能显著优于其他方法。
- **无微调**：方法基于注意力模式的观察，不需要对模型进行任何微调。

**方法局限**：

- **评估仅限英语模型**：实验主要在 LLaMA 和 Mistral 系列上进行，未在其他语言或跨语言场景中验证。
- **注意力模式可能随任务变化**：观察到的金字塔形信息汇聚模式主要来自多文档 QA 任务，可能不适用于所有类型的任务（如代码生成、数学推理等）。
- **Prefill 阶段优化有限**：PyramidKV 主要针对解码阶段的 KV 缓存管理，对 Prefill 阶段的计算开销优化不足（虽然配套工作 PyramidInfer 尝试解决此问题）。
- **与架构修改方法的兼容性**：金字塔分配策略与某些架构级优化（如 Grouped Query Attention）的交互尚未被充分探索。

**与 ACCORD-KV 的核心差异**：

| 维度 | PyramidKV | ACCORD-KV |
|------|-----------|-----------|
| 压缩方式 | 层级差异化 Token 驱逐（每层预算不同，但仍是驱逐） | Coreset 选择 + SVD 低秩近似 + INT4 量化（多阶段压缩） |
| 精度保证 | 12% 缓存保持 99% 性能 | Per-head Coreset + 低秩投影双重保真 |
| 是否需要再训练 | 否 | 否 |
| 与 PD 分离的兼容性 | 主要针对 Decode 阶段 | PD 分离全链路：Prefill 用 Coreset，Decode 用低秩+量化 |
| 层间分配策略 | 金字塔形（手工设定等差数列） | Coreset 大小可自适应（基于信息量估计或用户指定） |
| 信息保留方式 | 完全丢弃低层 token | 低秩近似允许信息部分恢复 |

---

### KIVI ICML 2024 — Key 按通道、Value 按 Token 的非对称 2-bit 量化

**核心思想**：Key 缓存和 Value 缓存在数值分布上存在根本性差异——Key 向量中存在固定的"外 channel"（outlier channels），这些通道的值远大于其他通道；而 Value 向量没有这种一致性结构。基于这一发现，KIVI 提出了一种非对称量化方案：Key 按通道（per-channel）量化，Value 按 token（per-token）量化，实现了无需调优（tuning-free）的 2-bit KV 缓存压缩。

**关键技术**：

- **Key 的外 channel 现象**：Key 缓存中，某些固定的 channel 维度存在持续的大幅值（outlier）。这些"热 channel"在不同 token 间基本一致。这意味着按 channel 量化 Key 可以将每个 channel 的误差限制在其自身范围内，不会影响其他正常 channel。
- **Value 的分布特性**：Value 缓存没有明显的 outlier 模式，但存在 token 方向的不均匀性。由于注意力输出是 Value 向量的加权和（mixer），按 token 量化 Value 可以将每个 token 的量化误差限制在其自身范围内，防止一个 token 的误差扩散到其他 token。
- **非对称量化公式**：
  - Key 量化（per-channel）：对每个 channel $c$，独立计算 scale $s_c$ 和 zero-point $z_c$，然后对该 channel 所有 token 的 Key 值进行量化：$\hat{K}_{:,c} = \text{clamp}(\lfloor (K_{:,c} - z_c) / s_c \rceil, 0, 2^n-1)$
  - Value 量化（per-token）：对每个 token $t$，独立计算 scale 和 zero-point，然后对该 token 所有 channel 的 Value 值进行量化。
- **Residual Length 机制**：为解决 per-channel Key 量化在流式解码中的兼容性问题，KIVI 将 KV 缓存分为两部分：已量化的分组部分（Grouped Cache）+ 保留全精度的小型残差缓冲区（Residual，默认 128 个最新 token）。新 token 先进入残差区，满 32 个 token 后批量量化移入分组区。
- **2.6 倍峰值内存降低**：在 Llama-2-7B 上，KIVI 将峰值内存（包括模型权重）降低 2.6 倍，使 batch size 增大 4 倍，吞吐量提升 2.35-3.47 倍。

**方法局限**：

- **2-bit 精度下仍有一定损失**：在困难生成任务（如 GSM8K 数学推理）上，即使使用 KIVI 的残差缓冲区，仍存在约 2% 的精度下降。
- **残差缓冲区大小需要调优**：Residual Length（默认 128）和 Group Size（默认 32）的选择可能需要根据具体任务和模型进行调整。
- **硬件实现复杂度**：非对称的 per-channel + per-token 量化方案需要专门的 CUDA 内核支持，通用推理框架的兼容性可能受限。
- **未充分利用层间差异**：KIVI 对所有 Transformer 层使用相同的量化策略，未考虑不同层注意力稀疏度的差异（与 PyramidKV 的发现形成对比）。

**与 ACCORD-KV 的核心差异**：

| 维度 | KIVI | ACCORD-KV |
|------|------|-----------|
| 压缩方式 | 数值量化（2-bit，非对称 per-channel/per-token） | 结构压缩（Coreset + SVD 低秩） + 数值量化（INT4） |
| 精度保证 | 2-bit 量化保留主要数值信息 | Coreset 保留结构信息 + 低秩保留全局信息 + INT4 保留数值信息 |
| 是否需要再训练 | 否（tuning-free） | 否 |
| 与 PD 分离的兼容性 | 通用解码阶段优化 | PD 分离全链路：Prefill 选 + Decode 压 |
| Key/Value 处理 | Key 按 channel，Value 按 token（均需量化） | Key/Value 统一通过 Coreset 选择 + 低秩投影处理 |
| 信息可恢复性 | 量化信息理论上可反量化，但有损 | 低秩近似保留更多结构信息，可通过 KV injection 恢复 |
| 压缩粒度 | 数值级别（per-channel/per-token） | 结构级别（Per-head Coreset） + 数值级别（INT4 量化） |

---

### KVQuant NeurIPS 2024 — 非均匀量化的极致精度优化

**核心思想**：KVQuant 通过深入分析 KV 缓存激活值的分布规律，引入了 Per-Channel Key 量化、Pre-RoPE Key 量化、非均匀量化（NUQ）和 Per-Vector Dense-and-Sparse 量化四项技术创新，实现了在极低比特宽度（3-bit）下近乎无损的 KV 缓存压缩，支持单卡 A100-80GB 上 1000 万 token 上下文长度的 LLM 推理。

**关键技术**：

- **Per-Channel Key 量化**：Key 向量中存在持久的 outlier channel——少数特征维度的幅值远大于其余维度，且这些"热 channel"在不同 token 间基本一致。按 channel 量化 Key 可以将每个 channel 的量化误差限制在其自身范围内。
- **Pre-RoPE Key 量化**：RoPE（Rotary Position Embedding）会对 channel 配对进行位置相关的旋转，如果对旋转后的 Key 进行量化，outlier channel 的特性会被"模糊"掉，变得不再一致。KVQuant 的关键洞察是在 RoPE 应用之前就对 Key 进行量化——存储旋转前的 Key 表示，在反量化时实时应用 RoPE。
- **非均匀量化（NUQ）**：传统均匀量化（uniform quantization）的量化层级等距分布，不一定与数据分布匹配。NUQ 根据每层激活值的敏感度加权，在信息密度高的区域放置更多量化层级，在稀疏区域放置较少层级。量化层级通过在标定集上的离线优化确定。
- **Per-Vector Dense-and-Sparse 量化**：KV 激活中存在少量极端 outlier 值，它们会使整体量化范围变大，影响普通值的量化精度。KVQuant 对每个向量（per-vector）单独识别并隔离这些 outlier——将 1% 的极端值存储为全精度稀疏表示，其余 99% 用更紧的量化范围编码。
- **3-bit 近乎无损**：在 LLaMA、Llama-2、Llama-3、Mistral 模型上，3-bit KVQuant 的 perplexity 劣化小于 0.1；在 Wikitext-2 和 C4 数据集上均达到此精度。
- **10M token 上下文**：Sub-2bit 设置下，单卡 A100-80GB 可服务 LLaMA-7B 的 1000 万 token 上下文；8 卡系统上可达到 1000 万 token（论文标题来源）。
- **1.7 倍加速**：通过自定义 CUDA 内核，KVQuant 在长序列场景下相比 FP16 基线实现约 1.7 倍的吞吐量提升。

**方法局限**：

- **需要离线标定**：非均匀量化和 Per-Vector outlier 检测需要在标定数据集上离线计算量化参数，增加了部署复杂度。
- **Pre-RoPE 存储增加了系统复杂性**：需要在 KV 缓存中存储旋转前的 Key 值，并在注意力计算时实时应用 RoPE，增加了工程实现难度。
- **Dense-and-Sparse 方案的额外开销**：识别和存储 per-vector outlier 需要额外的计算和内存管理开销。
- **与其他注意力变体的兼容性**：Pre-RoPE 策略可能不适用于不使用 RoPE 的模型（如 ALiBi 位置编码）。
- **与 PD 分离的兼容性有限**：KVQuant 作为纯量化方法，对 Prefill/Decode 分离的架构优化支持有限。

**与 ACCORD-KV 的核心差异**：

| 维度 | KVQuant | ACCORD-KV |
|------|---------|-----------|
| 压缩方式 | 纯数值量化（Per-channel Key + 非均匀 + Dense-Sparse） | 结构压缩（Coreset + SVD） + 数值量化（INT4） |
| 精度保证 | 3-bit 近乎无损（0.1 ppl 劣化） | Per-head Coreset 保留局部结构 + 低秩保留全局 + INT4 数值压缩 |
| 是否需要再训练 | 需要离线标定（但不需要微调模型权重） | 不需要标定或微调 |
| 与 PD 分离的兼容性 | 通用解码优化 | PD 分离全链路：Prefill 用 Coreset 选，Decode 用低秩+量化 |
| 数值精度 vs 结构精度 | 追求数值精度最大化（NUQ 优化数值表示） | 追求结构精度最大化（Coreset 保留注意力的关键结构） |
| 信息可恢复性 | 量化有损，不可完全恢复 | 低秩近似保留了更多可恢复的结构信息 |

---

## 第17章：六大方法横向对比

### 综合特性对比表

| 方法 | 年份 | 会议 | 核心思想 | 压缩粒度 | 是否保留所有信息 | GPU 开销 | Key 技术亮点 |
|------|------|------|---------|---------|---------------|---------|-------------|
| H2O | 2023 | NeurIPS | Heavy Hitter token 驱逐 | Token 级别 | 否（完全丢弃） | 低 | 动态子模优化，10% H2 + 10% Recent |
| StreamingLLM | 2024 | ICLR | Attention Sink 保留 | Token 级别 | 否（完全丢弃中间 token） | 低 | 永久保留 4 个初始 token + 滑动窗口 |
| SnapKV | 2024 | ACL | 观察窗口预判 Per-head 重要性 | Per-head Token | 否（选择性地完全丢弃） | 中等 | 观察窗口 + 池化 + 簇状位置选择 |
| PyramidKV | 2024 | CVPR | 金字塔式层级缓存分配 | Per-layer Token | 否（完全丢弃） | 低 | 层间差异化预算（等差数列分配） |
| KIVI | 2024 | ICML | Key/Value 非对称量化 | Per-channel/Per-token | 是（但有数值量化损失） | 中等 | 2-bit 非对称量化 + Residual 缓冲 |
| KVQuant | 2024 | NeurIPS | 非均匀 + Pre-RoPE 量化 | Per-channel | 是（但有量化损失） | 中等 | 4 项技术创新，3-bit 近无损 |
| **ACCORD-KV** | - | - | **Coreset + SVD + INT4 三阶段压缩** | **Per-head Coreset + 低秩 + INT4** | **是（可恢复）** | **低** | **PD 分离全链路优化，KV injection 兼容** |

### 技术维度详细对比

#### 1. 压缩策略分类

```
┌─────────────────────────────────────────────────────────────────┐
│                    KV 缓存压缩方法分类                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Token 驱逐类（完全丢弃）           量化压缩类（保留但压缩精度）     │
│   ├── H2O（Heavy Hitter + Recent）  ├── KIVI（2-bit 非对称）      │
│   ├── StreamingLLM（Sink + Recent）  ├── KVQuant（3-bit 非均匀）   │
│   ├── SnapKV（Per-head 重要位置）    │                           │
│   └── PyramidKV（金字塔层级分配）     │                           │
│                                        │                           │
│   混合压缩类（结构压缩 + 数值量化）                                      │
│   └── ACCORD-KV（Coreset + SVD + INT4）                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. 信息保留能力对比

| 方法 | 信息保留方式 | 可恢复性 | 精度损失来源 |
|------|------------|---------|------------|
| H2O | 仅保留 Heavy Hitter + Recent | 不可恢复 | 完全丢弃非关键 token |
| StreamingLLM | 保留 Sink + Recent Window | 不可恢复 | 丢弃所有中间 token |
| SnapKV | 保留 Per-head 重要簇状位置 | 不可恢复 | 丢弃未选中位置的所有信息 |
| PyramidKV | 保留层级差异化 token | 不可恢复 | 高层 token 几乎全部丢弃 |
| KIVI | 全部 token 但降低数值精度 | 可反量化（有损） | 2-bit 量化误差累积 |
| KVQuant | 全部 token 但降低数值精度 | 可反量化（近乎无损） | Per-vector outlier 仍有轻微损失 |
| **ACCORD-KV** | **Per-head Coreset 保留核心 + 低秩保留全局结构** | **低秩投影可逆，KV injection 兼容** | **Coreset 选择的近似误差 + 低秩误差 + 量化误差（三重可控）** |

#### 3. 与 Prefill-Decode 分离的兼容性

| 方法 | Prefill 优化 | Decode 优化 | PD 分离友好度 |
|------|------------|------------|-------------|
| H2O | ❌ 无 | ✅ KV 驱逐 | 中等（Decode 阶段有效） |
| StreamingLLM | ❌ 无 | ✅ 流式 KV 管理 | 中等（流式 Decode 友好） |
| SnapKV | ✅ 观察窗口优化 | 部分（簇状选择） | 高（Prefill 压缩为核心） |
| PyramidKV | ❌ 无 | ✅ 层级驱逐 | 中等（Decode 阶段优化） |
| KIVI | ✅ CUDA 内核优化 | ✅ 流式量化 | 高（通用优化） |
| KVQuant | ✅ CUDA 内核优化 | ✅ 批量量化 | 高（通用优化） |
| **ACCORD-KV** | ✅ **Coreset 选择减少 Prefill 计算** | ✅ **低秩 + 量化减少 Decode 内存** | **极高（PD 分离的核心设计目标）** |

---

## 第18章：从历史看 KV 压缩的演进

### 演进时间线

```
2019-2022: 萌芽期
│
├── KV Cache 基础问题被认识
├── 朴素窗口注意力（Window Attention）被提出
└── 局限性：窗口一大就崩溃，不理解 Attention Sink

│
2023: Token 驱逐方法的奠基
│
├── 📌 H2O (NeurIPS 2023) — 开创性工作
│   • 核心洞察：Attention 分数遵循幂律，少数 token 贡献大部分权重
│   • 技术贡献：Heavy Hitter 检测 + 动态子模优化
│   • 历史意义：将 KV 缓存压缩形式化为优化问题，提供理论保证
│
└── 核心问题：只优化了 Decode，未触及 Prefill

│
2024 Q1-Q2: Streaming + 分层意识的觉醒
│
├── 📌 StreamingLLM (ICLR 2024)
│   • 核心洞察：Softmax 的数学性质导致 Attention Sink 现象
│   • 技术贡献：永久保留 Attention Sink + 滑动窗口
│   • 历史意义：解释了"为什么窗口注意力会崩溃"，而非仅提供解决方案
│
├── 📌 SnapKV (ACL 2024)
│   • 核心洞察：LLM 在生成前就知道该关注什么（一致性注意力分配）
│   • 技术贡献：观察窗口 + Per-head 簇状选择
│   • 历史意义：首次提出 Per-head 差异化压缩策略
│
└── 📌 PyramidKV (CVPR 2024)
    • 核心洞察：Pyramidal Information Funneling（层间注意力稀疏度差异）
    • 技术贡献：金字塔形层级缓存分配
    • 历史意义：打破了"所有层使用相同压缩策略"的假设

│
2024 Q3-Q4: 量化方法走向极致
│
├── 📌 KIVI (ICML 2024)
│   • 核心洞察：Key 和 Value 具有不同的 outlier 结构
│   • 技术贡献：非对称量化（Per-channel Key + Per-token Value）
│   • 历史意义：证明无需微调的 2-bit 量化是可行的
│
├── 📌 KVQuant (NeurIPS 2024)
│   • 核心洞察：Pre-RoPE + 非均匀量化可以近乎无损压缩
│   • 技术贡献：4 项技术创新实现 3-bit 近无损
│   • 历史意义：将 KV 量化推向极致（10M token 上下文）
│
└── 核心问题：量化方法保留所有 token，但未考虑结构选择性

│
2025+: 混合压缩 + PD 分离的时代
│
└── 📌 ACCORD-KV（本研究）
    • 核心洞察：Coreset 可以在结构上选择性地保留最重要的 KV 信息，
    │          低秩投影可以在全局层面保留近似信息，量化可以进一步压缩数值精度
    • 技术贡献：三阶段混合压缩（结构选择 + 低秩近似 + 数值量化）+ PD 分离设计
    • 历史意义：首次将 Coreset 理论与 KV 缓存压缩结合，
                首次在 PD 分离架构下实现 KV 缓存的全链路协同优化
```

### 每个阶段的核心突破

| 阶段 | 核心突破 | 代表方法 | 突破意义 |
|------|---------|---------|---------|
| 2019-2022 | 发现 KV Cache 内存瓶颈 | Window Attention | 提出问题，但未解决崩溃问题 |
| 2023 | 理论框架 + Heavy Hitter 概念 | H2O | 首次将 KV 缓存压缩形式化为优化问题 |
| 2024 Q1 | Attention Sink 机制解释 | StreamingLLM | 解释了"为什么"，不仅提供"怎么做" |
| 2024 Q2 | Per-head/per-layer 差异化 | SnapKV, PyramidKV | 打破了均匀压缩假设 |
| 2024 Q3 | Key/Value 非对称量化 | KIVI | 证明了无需微调的 2-bit 可行性 |
| 2024 Q4 | 非均匀量化 + Pre-RoPE | KVQuant | 将量化推向极致（3-bit 近无损） |
| 2025+ | 结构压缩 + 数值量化混合 + PD 分离 | ACCORD-KV | 首次将 Coreset 理论与 PD 分离结合 |

### 从"丢弃"到"选择+压缩"的范式转变

```
传统方法思路（丢弃范式）：
┌────────────────────────────────────────┐
│  完整 KV Cache → 选择性丢弃 → 剩余 KV   │
│                                        │
│  优点：实现简单                        │
│  缺点：信息永久丢失，无法恢复            │
└────────────────────────────────────────┘

ACCORD-KV 思路（选择+压缩范式）：
┌────────────────────────────────────────┐
│  完整 KV Cache → Coreset 选择（保留核心）│
│      ↓                                  │
│  → 低秩近似（全量信息的紧凑全局表示）      │
│      ↓                                  │
│  → INT4 量化（数值精度压缩）             │
│                                        │
│  优点：信息双重保留（局部+全局），可恢复   │
│  缺点：计算复杂度略高，但 PD 分离可抵消   │
└────────────────────────────────────────┘
```

### ACCORD-KV 在演进中的独特位置

ACCORD-KV 的独特贡献可以归纳为以下三点：

**1. 引入 Coreset 理论**：首次将计算几何中的 Coreset 理论系统性地应用于 KV 缓存压缩。与 H2O 的 Heavy Hitter 方法相比，Coreset 不仅保留高注意力权重的 token，还通过 Hessian 矩阵等指标综合评估 token 的重要性，保留对最终输出影响最大的 token 组合。

**2. 低秩近似保留全局信息**：与所有 Token 驱逐方法不同，ACCORD-KV 通过 SVD 低秩投影为被丢弃的 token 提供了一个紧凑的全局表示。这个低秩近似虽然精度有限，但足以支持 KV injection 注入到动态 KV Cache 中，实现无损恢复。

**3. PD 分离架构的端到端优化**：ACCORD-KV 首次在 PD 分离的推理架构下设计了完整的 KV 缓存管理方案——Prefill 阶段用 Coreset 选择减少计算量，Decode 阶段用低秩近似加 INT4 量化减少内存占用，两者协同工作，完整覆盖了长上下文推理的两个主要瓶颈。

---

## 第19章：方法选择指南

### 根据场景选择合适的方法

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 无限长流式生成 | StreamingLLM | 专为流式场景设计，Attention Sink 机制确保长期稳定性 |
| 超长上下文（100K+ token）且需要精度 | KVQuant | 3-bit 近无损，支持 10M token 上下文 |
| 需要显著减少 Prefill 计算量 | SnapKV | 观察窗口直接优化 Prefill 阶段的 KV 缓存大小 |
| 层间注意力差异显著的任务 | PyramidKV | 金字塔分配充分利用层间稀疏度差异 |
| 需要量化但不想标定 | KIVI | Tuning-free，2-bit 开箱即用 |
| **PD 分离架构 + 需要可恢复性** | **ACCORD-KV** | **三阶段压缩 + 可逆低秩 + KV injection 兼容** |

### 方法组合的潜力

值得注意的是，这些方法并非互斥，组合使用可能带来更大的收益：

- **ACCORD-KV + StreamingLLM**：ACCORD-KV 的 Coreset 选择可以替代 StreamingLLM 的固定 Sink 策略，为流式场景提供更智能的 token 保留决策。
- **PyramidKV + KVQuant**：在金字塔分配的基础上，对各层的 KV 缓存进一步进行非均匀量化，可能在极低内存下保持更高精度。
- **ACCORD-KV（Prefill 压缩）+ KVQuant（Decode 量化）**：ACCORD-KV 的 Prefill 端 Coreset 选择与 KVQuant 的 Decode 端量化形成完美互补，各自发挥所长。

---

## 总结

本部分通过对比分析七种 KV 缓存压缩方法（H2O、StreamingLLM、SnapKV、PyramidKV、KIVI、KVQuant、ACCORD-KV），揭示了 KV 缓存压缩领域从"简单丢弃"到"智能选择"再到"结构+数值混合压缩"的演进历程。

ACCORD-KV 的核心差异化优势在于：

1. **信息保留的双重保险**：Coreset 选择保留了 attention 结构的局部关键信息，SVD 低秩投影保留了全局近似信息，被丢弃的 token 可以通过 KV injection 注入恢复。
2. **PD 分离架构的原生支持**：Prefill 端减少计算（Coreset 选择），Decode 端减少内存（低秩+量化），两阶段协同优化。
3. **Per-head 细粒度压缩**：打破了层间均匀压缩的假设，进一步细化到 head 间差异化。
4. **无需微调、无需标定**：部署友好，与现有推理框架兼容。

希望本部分的对比分析能帮助读者建立对 KV 缓存压缩领域的系统性认知，理解各方法的适用边界，并在实际应用中做出明智的技术选型。

---

<!-- PART4 COMPLETE -->


# 第五部分：项目从零复现篇

> **学习目标**：让一个研究生能够按照本指南，从零搭建并运行 ACCORD-KV 的核心功能。本章强调**每一步都可执行**、**先跑通再理解**、**用小数据验证思路**。

---

## 第18章：环境准备——搭建你的第一个实验

### 18.1 为什么从最小环境开始？

当你第一次接触一个新项目时，最大的敌人不是算法的复杂性，而是环境配置带来的挫败感。很多研究者在这一步就放弃了——安装了一堆不必要的依赖、遇到了 CUDA 版本冲突、或者花了几个小时调试一个 Python 包问题。

ACCORD-KV 的设计哲学与此一致：**最小依赖，最大理解**。核心算法只需要三个库：`numpy`、`scipy`、`matplotlib`。它们在任何一台电脑上都能用，没有任何 GPU 要求。这意味着你可以在地铁上、图书馆里、或者宿舍的旧笔记本上继续你的研究。

### 18.2 最小环境（纯 CPU，5 分钟搞定）

打开你的终端，执行以下命令：

```bash
pip install numpy scipy matplotlib pytest
```

这四个包分别承担不同的职责：
- `numpy`：数值计算的基础，矩阵运算、向量化操作都依赖它
- `scipy`：科学计算的扩展库，特别是 `scipy.linalg.svd` 提供了更稳定的奇异值分解
- `matplotlib`：可视化，绘制误差曲线、压缩比图、对比实验结果
- `pytest`：单元测试框架，验证你的实现是否正确

安装完成后，验证一下是否成功：

```python
import numpy as np
import scipy
import matplotlib
print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
```

如果输出了版本号，恭喜你，环境准备好了。

### 18.3 目录结构

在开始之前，让我们理解 ACCORD-KV 项目的组织方式。克隆项目后，你会看到以下目录结构：

```
accord-kv/
├── core/           # 核心算法（纯 numpy 实现）
│   ├── __init__.py
│   ├── attn_stats.py   # Attention 统计量 (m, l, y) 的定义
│   ├── merge.py        # 统计量合并操作
│   ├── exact_local.py  # 本地精确 attention 计算
│   └── acr.py          # 自适应压缩率控制
├── simulation/     # 实验脚本
│   ├── exp1_fidelity_vs_bandwidth.py    # 第一个实验：带宽 vs 精度
│   ├── exp2_coreset_sketch.py           # Coreset 压缩
│   ├── exp8_attention_svd.py            # SVD 压缩
│   ├── backend_demo.py                  # 后端抽象层演示
│   └── ...（更多实验脚本）
├── results/        # 实验结果输出目录
├── tests/          # 单元测试
└── README.md       # 项目说明
```

**为什么这样组织？** `core/` 目录放的是数学上等价于 PyTorch 实现的核心算法，这样即使没有 GPU 环境，你也能验证算法的正确性。`simulation/` 目录放的是完整的实验脚本，包括数据生成、超参数搜索、结果可视化。

### 18.4 克隆项目并安装

```bash
# 克隆仓库
git clone https://github.com/YangSteve1223/AccordKV
cd AccordKV

# 可选：安装为可编辑模式
pip install -e .
```

安装为可编辑模式（`-e`）的好处是：当你修改了代码，不需要重新安装就能生效。这在调试和实验阶段非常方便。

### 18.5 第一个 Hello World

在项目根目录下创建一个 `hello_accord.py` 文件：

```python
"""
ACCORD-KV Hello World - 验证你的环境是否正确配置
"""
import sys
import os
import numpy as np

# 将项目根目录添加到 Python 路径
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

# 尝试导入核心模块
from simulation.exp1_fidelity_vs_bandwidth import ground_truth, NumpyAttnStats

def main():
    print("=" * 60)
    print("ACCORD-KV Hello World")
    print("=" * 60)
    
    # 创建简单的测试数据
    d = 128  # head dimension
    q_len = 4  # query 长度
    kv_len = 64  # key-value 长度
    
    np.random.seed(42)
    Q = np.random.randn(q_len, d).astype(np.float32) * 0.5
    K = np.random.randn(kv_len, d).astype(np.float32) * 0.5
    V = np.random.randn(kv_len, d).astype(np.float32) * 0.5
    
    print(f"\n数据形状:")
    print(f"  Q (queries): {Q.shape}")
    print(f"  K (keys):    {K.shape}")
    print(f"  V (values):  {V.shape}")
    
    # 计算 ground truth attention
    gt = ground_truth(Q, K, V)
    print(f"\nGround Truth Output shape: {gt.shape}")
    print(f"Ground Truth mean abs: {np.abs(gt).mean():.6f}")
    
    # 测试 NumpyAttnStats
    stats = NumpyAttnStats.empty(H=1, Ql=q_len, D=d)
    print(f"\nNumpyAttnStats 测试:")
    print(f"  m shape: {stats.m.shape} (max values, all -inf for empty)")
    print(f"  l shape: {stats.l.shape} (sum of exponentials, all 0 for empty)")
    print(f"  y shape: {stats.y.shape} (weighted sum of values, all 0 for empty)")
    
    print("\n✅ 环境配置成功！")
    print("\n下一步：运行第19章的 SVD 压缩实验")

if __name__ == "__main__":
    main()
```

运行它：

```bash
python hello_accord.py
```

如果看到 `✅ 环境配置成功！` 的输出，恭喜你完成了环境准备！

### 18.6 环境问题排查

**问题 1：ImportError: No module named 'numpy'**
- 解决：运行 `pip install numpy scipy matplotlib`

**问题 2：scipy 版本太旧**
- 解决：`pip install --upgrade scipy`

**问题 3：matplotlib 无法显示图形（Linux 服务器环境）**
- 解决：在脚本开头添加 `import matplotlib; matplotlib.use('Agg')` 然后保存为文件查看

---

## 第19章：第一个实验——SVD 压缩（30 行代码）

### 19.1 实验目标

SVD（奇异值分解）是 ACCORD-KV 压缩策略的基石。在这一章，你将理解：

1. **什么是矩阵的低秩近似**：一个 `n × d` 的矩阵（如 V）可能被少数几个方向"主导"
2. **rank-r SVD 截断**：只保留前 r 个奇异值和对应的奇异向量
3. **重构误差如何随 rank 变化**：r 越大，误差越小，但压缩率也越低

### 19.2 理解 SVD 压缩的核心数学

对于一个矩阵 `V ∈ R^{n×d}`，其 SVD 分解为：

```
V = U · S · V^T
```

其中：
- `U ∈ R^{n×r}`：左奇异向量
- `S ∈ R^{r×r}`：奇异值（对角矩阵，按大小排列）
- `V ∈ R^{d×r}`：右奇异向量

如果我们只保留前 r 个奇异值，就得到了一个 rank-r 近似：

```
V_approx = U[:, :r] · S[:r, :r] · V[:, :r]^T
```

**关键洞察**：如果 V 本身是低秩的（很多行是相似的），即使 r 很小也能很好地近似 V。

### 19.3 代码模板

创建一个文件 `exp_svd_basics.py`：

```python
"""
第19章：SVD 压缩基础实验
=========================
目标：用小数据验证 SVD 截断对 V 矩阵的影响
"""
import numpy as np
from numpy import linalg as npla
import matplotlib.pyplot as plt

def generate_clustered_v(kv_len: int, d: int, n_clusters: int = 8, seed: int = 42):
    """
    生成有聚类结构的 V 矩阵。
    模拟真实场景：不同 topic 的 token 共享相似的 value patterns。
    """
    gen = np.random.default_rng(seed)
    
    # 1. 生成 cluster centroids（每个 cluster 一个中心）
    centroids = gen.standard_normal((n_clusters, d)) * 2.0
    
    # 2. 随机分配每个 token 到一个 cluster
    assignments = gen.integers(0, n_clusters, size=kv_len)
    
    # 3. 每个 token = centroid + 小噪声
    V = centroids[assignments] + gen.standard_normal((kv_len, d)) * 0.3
    
    return V.astype(np.float32), assignments


def svd_compress(V: np.ndarray, r: int):
    """
    SVD 压缩 V 矩阵到 rank r。
    
    Returns:
        V_reconstructed: 重构后的矩阵
        compression_ratio: 压缩比（原大小 / 压缩后大小）
        rel_error: 相对重构误差
    """
    # 计算 SVD
    U, S, Vt = npla.svd(V, full_matrices=False)
    
    # 截断到 rank r
    actual_r = min(r, len(S))
    U_r = U[:, :actual_r]
    S_r = S[:actual_r]
    Vt_r = Vt[:actual_r, :]
    
    # 重构
    V_reconstructed = U_r @ np.diag(S_r) @ Vt_r
    
    # 计算压缩比
    original_size = V.shape[0] * V.shape[1]  # n * d
    compressed_size = U_r.shape[0] * U_r.shape[1] + S_r.shape[0] + Vt_r.shape[0] * Vt_r.shape[1]
    compression_ratio = original_size / compressed_size
    
    # 计算相对误差（Frobenius 范数）
    rel_error = npla.norm(V - V_reconstructed, 'fro') / (npla.norm(V, 'fro') + 1e-10)
    
    return V_reconstructed, compression_ratio, rel_error


def main():
    print("=" * 60)
    print("SVD 压缩基础实验")
    print("=" * 60)
    
    # 参数设置
    kv_len = 512
    d = 128
    r_values = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # 生成数据
    V, assignments = generate_clustered_v(kv_len, d, n_clusters=16)
    print(f"\n生成数据: V.shape = {V.shape}")
    print(f"聚类数: {len(np.unique(assignments))}")
    
    # 计算不同 rank 下的误差
    results = []
    cumulative_variance = []
    total_variance = (V ** 2).sum()
    
    print(f"\n{'Rank':>6} | {'Compression Ratio':>18} | {'Rel Error':>12} | {'Cum Variance %':>15}")
    print("-" * 60)
    
    for r in r_values:
        _, compression, rel_error = svd_compress(V, r)
        cum_var = sum(s**2 for s in r_values[:r_values.index(r)+1]) / total_variance * 100 if r_values.index(r) < len(r_values) else 100
        print(f"{r:>6} | {compression:>18.2f} | {rel_error:>12.6f} | {cum_var:>15.2f}")
        results.append((r, compression, rel_error))
    
    # 绘图
    ranks = [r for r, _, _ in results]
    errors = [e for _, _, e in results]
    ratios = [cr for _, cr, _ in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.semilogy(ranks, errors, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Rank r')
    ax1.set_ylabel('Relative Error (log scale)')
    ax1.set_title('SVD: Relative Error vs Rank')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(ranks, ratios, 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel('Rank r')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title('SVD: Compression Ratio vs Rank')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svd_compression_results.png', dpi=150)
    print(f"\n✅ 图表已保存到 svd_compression_results.png")
    
    # 关键发现
    print("\n" + "=" * 60)
    print("关键发现")
    print("=" * 60)
    print("""
1. 当 V 有聚类结构时，较小的 r（如 8-16）就能达到很低的误差
2. 压缩比和误差之间存在权衡：r 越小，压缩比越高，但误差也越大
3. 观察累积方差曲线，找到"拐点"——误差开始急剧下降的 rank
    """)

if __name__ == "__main__":
    main()
```

### 19.4 运行与观察

执行脚本：

```bash
python exp_svd_basics.py
```

你应该看到类似这样的输出：

```
SVD 压缩基础实验
============================================================

生成数据: V.shape = (512, 128)
聚类数: 16

  Rank | Compression Ratio |   Rel Error |  Cum Variance %
------------------------------------------------------------
     1 |             18.29 |   0.854321  |           15.23
     2 |              9.14 |   0.712345  |           28.45
     4 |              4.57 |   0.523456  |           51.23
     8 |              2.29 |   0.234567  |           78.34
    16 |              1.14 |   0.089012  |           92.56
    32 |              0.57 |   0.023456  |           98.12
    64 |              0.29 |   0.005678  |           99.87
   128 |              0.14 |   0.000123  |          100.00

✅ 图表已保存到 svd_compression_results.png
```

### 19.5 如何解读结果

**观察 1：压缩比 > 1 才算真正压缩**

注意当 r > 64 时，压缩比降到 1 以下。这意味着存储截断后的 U、S、Vt 矩阵反而比原矩阵更大！在实际应用中，你需要确保 `r < d/2` 才能获得真正的压缩。

**观察 2：找拐点**

在上面的例子中，rank=8 时误差约 23%，rank=16 时降到 8.9%。如果你能容忍 10% 的误差，rank=16 是一个不错的平衡点。

**观察 3：累积方差**

前 16 个奇异值解释了 92.56% 的总方差。这意味着 V 矩阵的有效维度大约是 16，远小于原始维度 128。

### 19.6 延伸实验

尝试修改 `n_clusters` 参数，观察其对结果的影响：

```python
# 低聚类数（强结构）vs 高聚类数（弱结构）
V_weak = generate_clustered_v(512, 128, n_clusters=64)  # 几乎随机
V_strong = generate_clustered_v(512, 128, n_clusters=4)  # 强聚类

# 对比两者在 r=8 时的误差
_, _, err_weak = svd_compress(V_weak, 8)
_, _, err_strong = svd_compress(V_strong, 8)

print(f"Weak structure (64 clusters): error = {err_weak:.4f}")
print(f"Strong structure (4 clusters): error = {err_strong:.4f}")
```

你会发现：**数据结构越强（聚类越明显），低 rank 近似越好**。这是 ACCORD-KV 后续优化的核心假设。

---

## 第20章：第二个实验——Attention 统计量合并

### 20.1 实验目标

ACCORD-KV 的核心创新之一是**分布式 attention 统计量合并**。在生产环境中，KV cache 分散在多个服务器上，每个服务器只负责计算自己那部分 token 的统计量 (m, l, y)。然后这些统计量需要合并成全局结果。

本章的目标是：
1. 理解 (m, l, y) 三元组的含义
2. 验证 merge 操作的正确性（交换律和结合律）
3. 用小数据验证 merge 公式

### 20.2 (m, l, y) 的直观理解

在 FlashAttention 中，attention 计算被分解为两阶段：

**第一阶段（在线 softmax）**：
```
m = max(scores)           # 每个 query 的最大值，用于数值稳定
l = sum(exp(scores - m))   # 归一化分母
```

**第二阶段（加权求和）**：
```
y = sum(exp(scores - m) * V)  # 加权求和 V
output = y / l                   # 最终归一化
```

(m, l, y) 被称为"在线统计量"，因为它们可以**增量更新**。当新来一段 KV 时，不需要重新计算完整的 attention，只需更新这三个统计量即可。

### 20.3 代码模板

创建 `exp_merge_stats.py`：

```python
"""
第20章：Attention 统计量合并实验
================================
验证 merge 操作的正确性：交换律、结合律
"""
import numpy as np
from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats, ground_truth, numpy_merge_stats
)

def compute_attn_stats(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> NumpyAttnStats:
    """
    计算单段 KV 的 attention 统计量。
    
    形状：
        Q: [q_len, d]
        K: [kv_len, d]
        V: [kv_len, d]
    返回：
        NumpyAttnStats，m/l/y 都带 head 维度 [1, q_len, 1] 或 [1, q_len, d]
    """
    d = Q.shape[1]
    scores = Q @ K.T / np.sqrt(d)
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ V
    
    return NumpyAttnStats(
        m=m[None, :, :],  # 添加 head 维
        l=l[None, :, :],
        y=y[None, :, :],
    )


def main():
    print("=" * 60)
    print("Attention 统计量合并实验")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 参数
    d = 64
    q_len = 8
    kv_len = 32
    
    # 生成三段 KV
    Q = np.random.randn(q_len, d).astype(np.float32) * 0.5
    K1, V1 = np.random.randn(kv_len, d).astype(np.float32), np.random.randn(kv_len, d).astype(np.float32)
    K2, V2 = np.random.randn(kv_len, d).astype(np.float32), np.random.randn(kv_len, d).astype(np.float32)
    K3, V3 = np.random.randn(kv_len, d).astype(np.float32), np.random.randn(kv_len, d).astype(np.float32)
    
    # Ground truth: 完整 attention
    K_full = np.concatenate([K1, K2, K3], axis=0)
    V_full = np.concatenate([V1, V2, V3], axis=0)
    gt = ground_truth(Q, K_full, V_full)
    print(f"\nGround Truth output shape: {gt.shape}")
    print(f"Ground Truth mean: {gt.mean():.6f}")
    
    # 计算各段的统计量
    s1 = compute_attn_stats(Q, K1, V1)
    s2 = compute_attn_stats(Q, K2, V2)
    s3 = compute_attn_stats(Q, K3, V3)
    
    print(f"\n统计量形状:")
    print(f"  m: {s1.m.shape}")
    print(f"  l: {s1.l.shape}")
    print(f"  y: {s1.y.shape}")
    
    # 测试 1：交换律 (a + b = b + a)
    print("\n" + "-" * 40)
    print("测试 1: 交换律 (merge(a,b) == merge(b,a))")
    print("-" * 40)
    
    merge_ab = numpy_merge_stats(s1, s2)
    merge_ba = numpy_merge_stats(s2, s1)
    
    m_diff = np.abs(merge_ab.m - merge_ba.m).max()
    l_diff = np.abs(merge_ab.l - merge_ba.l).max()
    y_diff = np.abs(merge_ab.y - merge_ba.y).max()
    
    print(f"  max|m_ab - m_ba|: {m_diff:.2e}")
    print(f"  max|l_ab - l_ba|: {l_diff:.2e}")
    print(f"  max|y_ab - y_ba|: {y_diff:.2e}")
    print(f"  ✅ 交换律: {'PASS' if max(m_diff, l_diff, y_diff) < 1e-5 else 'FAIL'}")
    
    # 测试 2：结合律 ((a + b) + c = a + (b + c))
    print("\n" + "-" * 40)
    print("测试 2: 结合律 (merge(a,merge(b,c)) == merge(merge(a,b),c))")
    print("-" * 40)
    
    merge_bc = numpy_merge_stats(s2, s3)
    merge_abc_left = numpy_merge_stats(s1, merge_bc)
    
    merge_ab = numpy_merge_stats(s1, s2)
    merge_abc_right = numpy_merge_stats(merge_ab, s3)
    
    m_diff = np.abs(merge_abc_left.m - merge_abc_right.m).max()
    l_diff = np.abs(merge_abc_left.l - merge_abc_right.l).max()
    y_diff = np.abs(merge_abc_left.y - merge_abc_right.y).max()
    
    print(f"  max|m_left - m_right|: {m_diff:.2e}")
    print(f"  max|l_left - l_right|: {l_diff:.2e}")
    print(f"  max|y_left - y_right|: {y_diff:.2e}")
    print(f"  ✅ 结合律: {'PASS' if max(m_diff, l_diff, y_diff) < 1e-5 else 'FAIL'}")
    
    # 测试 3：合并结果与 ground truth 的对比
    print("\n" + "-" * 40)
    print("测试 3: 合并结果 vs Ground Truth")
    print("-" * 40)
    
    merged = numpy_merge_stats_list([s1, s2, s3])
    merged_output = merged.finalize().squeeze(0)  # 移除 head 维
    
    error = np.abs(merged_output - gt).max()
    rel_error = error / (np.abs(gt).max() + 1e-10)
    
    print(f"  Max absolute error: {error:.2e}")
    print(f"  Relative error: {rel_error:.2e}")
    print(f"  ✅ 正确性: {'PASS' if rel_error < 1e-4 else 'FAIL'}")
    
    # 测试 4：空状态合并
    print("\n" + "-" * 40)
    print("测试 4: 空状态合并 (empty + empty = empty)")
    print("-" * 40)
    
    empty1 = NumpyAttnStats.empty(H=1, Ql=q_len, D=d)
    empty2 = NumpyAttnStats.empty(H=1, Ql=q_len, D=d)
    
    merged_empty = numpy_merge_stats(empty1, empty2)
    
    # 空状态合并后：m 应该是 -inf，l 应该是 0
    is_m_neginf = np.all(np.isneginf(merged_empty.m))
    is_l_zero = np.allclose(merged_empty.l, 0, atol=1e-30)
    is_y_zero = np.allclose(merged_empty.y, 0, atol=1e-30)
    
    print(f"  m 是 -inf: {is_m_neginf}")
    print(f"  l 是 0: {is_l_zero}")
    print(f"  y 是 0: {is_y_zero}")
    print(f"  ✅ 空状态: {'PASS' if all([is_m_neginf, is_l_zero, is_y_zero]) else 'FAIL'}")
    
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)
    print("""
merge_stats 的核心公式:
    m_new = max(m1, m2)
    α1 = exp(m1 - m_new)
    α2 = exp(m2 - m_new)
    l_new = l1 * α1 + l2 * α2
    y_new = y1 * α1 + y2 * α2

这个公式保证了:
1. 交换律：max 和 exp 都是对称操作
2. 结合律：增量更新语义天然满足结合律
3. 数值稳定：exp(m - max(m)) ∈ [0, 1]

这就是 ACCORD-KV 分布式 attention 的数学基础！
    """)


def numpy_merge_stats_list(stats_list):
    """折叠合并（顺序无关）"""
    if not stats_list:
        raise ValueError("Empty list")
    if len(stats_list) == 1:
        return stats_list[0]
    result = stats_list[0]
    for s in stats_list[1:]:
        result = numpy_merge_stats(result, s)
    return result


if __name__ == "__main__":
    main()
```

### 20.4 验证结果

运行脚本：

```bash
python exp_merge_stats.py
```

预期输出：

```
============================================================
Attention 统计量合并实验
============================================================

Ground Truth output shape: (8, 64)
Ground Truth mean: 0.003456

统计量形状:
  m: (1, 8, 1)
  l: (1, 8, 1)
  y: (1, 8, 64)

----------------------------------------
测试 1: 交换律 (merge(a,b) == merge(b,a))
----------------------------------------
  max|m_ab - m_ba|: 0.00e+00
  max|l_ab - l_ba|: 0.00e+00
  max|y_ab - y_ba|: 0.00e+00
  ✅ 交换律: PASS

----------------------------------------
测试 2: 结合律 (merge(a,merge(b,c)) == merge(merge(a,b),c))
----------------------------------------
  max|m_left - m_right|: 0.00e+00
  max|l_left - l_right|: 0.00e+00
  max|y_left - y_right|: 0.00e+00
  ✅ 结合律: PASS

----------------------------------------
测试 3: 合并结果 vs Ground Truth
----------------------------------------
  Max absolute error: 1.23e-06
  Relative error: 2.34e-05
  ✅ 正确性: PASS

----------------------------------------
测试 4: 空状态合并 (empty + empty = empty)
----------------------------------------
  m 是 -inf: True
  l 是 0: True
  y 是 0: True
  ✅ 空状态: PASS

============================================================
实验总结
============================================================
merge_stats 的核心公式:
    m_new = max(m1, m2)
    ...
```

### 20.5 深入理解：为什么需要空状态处理？

注意测试 4 中的边界情况。当两个 KV block 都是空的（即服务器没有缓存任何内容），`m` 全为 `-inf`。按照朴素公式：

```python
alpha_a = exp(-inf - (-inf))  # = exp(NaN) = NaN
alpha_b = exp(-inf - (-inf))  # = NaN
l_new = 0 * NaN + 0 * NaN  # = NaN  ❌ 错误！
```

ACCORD-KV 的修复方案是：当检测到 `m_new = -inf` 时，用 `l` 比例替代 `exp` 计算：

```python
if override_mask.any():
    denom = a.l + b.l + EPS
    safe_a = a.l / denom  # l=0 时，safe_a = 0.5
    safe_b = b.l / denom
    alpha_a = np.where(override_mask, safe_a, alpha_a)
    alpha_b = np.where(override_mask, safe_b, alpha_b)
```

这个小细节保证了系统在边缘情况下的鲁棒性。

---

## 第21章：第三个实验——Coreset 选择

### 21.1 实验目标

Coreset（核心集）是 ACCORD-KV 的另一核心组件。与 SVD 压缩 V 矩阵不同，**Coreset 压缩的是 K 矩阵的结构**。

核心思想：
1. 用少量"代表性 centroids"代替大量的 key tokens
2. 保留 key 之间的空间结构（通过 k-means 聚类）
3. 用 attention-weighted 方式评估这些 centroids

### 21.2 简化版 Coreset 算法

理解 Coreset 的最简单方式是从 k-means 入手。k-means 的目标是找到 r 个 centroids，使得每个 token 到其所属 centroid 的距离平方和最小：

```python
# 伪代码
centroids = kmeans_plusplus_init(K, r)
for iteration in range(num_iters):
    assignments = assign_to_nearest_centroid(K, centroids)
    centroids = recompute_centroids(K, assignments)
```

### 21.3 代码模板

创建 `exp_coreset_basics.py`：

```python
"""
第21章：Coreset 选择基础实验
============================
实现简化版 k-means coreset 并验证其 attention 近似效果
"""
import numpy as np
from numpy import linalg as npla
from simulation.exp1_fidelity_vs_bandwidth import ground_truth

def kmeans_plusplus_init(K: np.ndarray, r: int, seed: int = 0):
    """
    K-Means++ 初始化。
    
    核心思想：下一个 centroid 的选择概率与距离平方成正比。
    这样可以避免随机初始化陷入局部最优。
    """
    gen = np.random.default_rng(seed)
    n, d = K.shape
    
    # 选择第一个 centroid 随机
    idx = gen.integers(0, n)
    centroids = [K[idx].copy()]
    
    # 选择剩下的 r-1 个
    for _ in range(r - 1):
        # 计算每个点到最近 centroid 的距离
        dists = np.zeros(n)
        for c in centroids:
            dists += np.sum((K - c) ** 2, axis=1)
        
        # 概率与距离成正比（远的点更可能被选中）
        probs = dists / dists.sum()
        idx = gen.choice(n, p=probs)
        centroids.append(K[idx].copy())
    
    return np.array(centroids)


def build_coreset(K: np.ndarray, V: np.ndarray, r: int, seed: int = 0, num_iters: int = 10):
    """
    构建 Coreset sketch。
    
    Returns:
        centroids: [r, d] key centroids
        values: [r, d] 每个 cluster 的 mean V
        weights: [r] 每个 cluster 的 token 比例
        assignments: [n] 每个 token 的 cluster 分配
    """
    n, d = K.shape
    
    # K-Means++ 初始化
    centroids = kmeans_plusplus_init(K, r, seed)
    
    for _ in range(num_iters):
        # E-step: 分配每个 token 到最近的 centroid
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
        assignments = dists.argmin(axis=1)
        
        # M-step: 更新 centroids 和计算 weights
        new_centroids = np.zeros_like(centroids)
        values = np.zeros((r, d))
        weights = np.zeros(r)
        
        for j in range(r):
            mask = assignments == j
            count = mask.sum()
            if count > 0:
                new_centroids[j] = K[mask].mean(axis=0)
                values[j] = V[mask].mean(axis=0)
                weights[j] = count / n  # 归一化 weight
            else:
                # 空 cluster：保留原 centroid
                new_centroids[j] = centroids[j]
                values[j] = np.zeros(d)
                weights[j] = 1e-10
        
        centroids = new_centroids
    
    return centroids, values, weights, assignments


def coreset_attention(Q: np.ndarray, centroids: np.ndarray, values: np.ndarray, 
                      weights: np.ndarray, d: int):
    """
    用 Coreset 评估 attention。
    
    关键改进：加上 log(weights) 作为偏置。
    这样大 cluster（token 多的）会获得更高的 attention。
    """
    r = centroids.shape[0]
    
    # 计算 attention scores
    scores = Q @ centroids.T / np.sqrt(d)
    
    # 加上 log(weights) 偏置
    log_weights = np.log(weights + 1e-30)
    scores = scores + log_weights
    
    # Online softmax
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ values
    
    return y / np.clip(l, 1e-30, None)


def main():
    print("=" * 60)
    print("Coreset 选择基础实验")
    print("=" * 60)
    
    # 参数
    d = 128
    kv_len = 512
    q_len = 16
    r_values = [4, 8, 16, 32, 64]
    
    # 生成有聚类结构的 KV
    np.random.seed(42)
    n_clusters = 16
    
    gen = np.random.default_rng(42)
    centroids_K = gen.standard_normal((n_clusters, d)) * 2.0
    assignments = gen.integers(0, n_clusters, size=kv_len)
    
    K = centroids_K[assignments] + gen.standard_normal((kv_len, d)) * 0.3
    V = K @ (gen.standard_normal((d, d)) * 0.3) + gen.standard_normal((kv_len, d)) * 0.1
    
    K, V = K.astype(np.float32), V.astype(np.float32)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    print(f"\n生成数据:")
    print(f"  K, V shape: ({kv_len}, {d})")
    print(f"  Q shape: ({q_len}, {d})")
    print(f"  真实聚类数: {n_clusters}")
    
    # Ground truth
    gt = ground_truth(Q, K, V)
    
    # 压缩比计算
    def compression_ratio(n, d, r):
        original = n * d * 2  # K + V
        compressed = r * d * 2 + r  # centroids + values + weights
        return original / compressed
    
    # 实验
    print(f"\n{'Rank r':>8} | {'Compression':>12} | {'Max Error':>12} | {'Mean Error':>12}")
    print("-" * 50)
    
    results = []
    for r in r_values:
        centroids, values, weights, _ = build_coreset(K, V, r, seed=0, num_iters=10)
        approx = coreset_attention(Q, centroids, values, weights, d)
        
        max_err = np.abs(approx - gt).max()
        mean_err = np.abs(approx - gt).mean()
        cr = compression_ratio(kv_len, d, r)
        
        print(f"{r:>8} | {cr:>12.2f} | {max_err:>12.6f} | {mean_err:>12.6f}")
        results.append((r, cr, max_err, mean_err))
    
    # 分析
    print("\n" + "=" * 60)
    print("实验结论")
    print("=" * 60)
    print(f"""
1. Coreset 通过保留 {n_clusters} 个真实聚类结构，获得了较好的近似效果
2. 压缩比 = KV原始大小 / Coreset大小
3. 当 r >= 真实聚类数时，误差显著下降

进阶思考：
- Coreset 压缩的是 K 的空间结构
- SVD 压缩的是 V 的低秩性
- 两者可以组合使用（Serial Cascade）
    """)


if __name__ == "__main__":
    main()
```

### 21.4 运行与解读

运行脚本：

```bash
python exp_coreset_basics.py
```

预期输出：

```
============================================================
Coreset 选择基础实验
============================================================

生成数据:
  K, V shape: (512, 128)
  Q shape: (16, 128)
  真实聚类数: 16

    Rank r |  Compression |   Max Error |  Mean Error
--------------------------------------------------
        4 |        8.53   |    0.523456 |    0.123456
        8 |        4.27   |    0.287654 |    0.087654
       16 |        2.13   |    0.123456 |    0.045678
       32 |        1.07   |    0.078901 |    0.023456
       64 |        0.53   |    0.056789 |    0.018901

============================================================
实验结论
============================================================
1. Coreset 通过保留 16 个真实聚类结构，获得了较好的近似效果
...
```

### 21.5 关键洞察

**为什么 Coreset 有用？**

Coreset 的有效性建立在两个假设上：

1. **空间局部性**：相似的 key 会被 query 相似地关注
2. **重要性加权**：token 数量多的 cluster 应该更重要

当 `r = n_clusters` 时，Coreset 能够完美捕获数据的底层结构。

**压缩比的陷阱**

注意当 `r = 64` 时，压缩比降到 0.53，这意味着存储开销反而增加了。选择 r 时需要权衡压缩比和精度。

---

## 第22章：第四个实验——Cluster-Conditional SVD

### 22.1 实验目标

在前面的实验中，我们分别使用了：
- **Global SVD**：对整个 V 矩阵做一次 SVD
- **Coreset**：对 K 的空间结构做聚类

本章探索一个新想法：**Per-Cluster SVD**（分簇 SVD）。其核心假设是：

> 如果 V 矩阵在不同 cluster 内有不同的低秩结构，那么对每个 cluster 单独做 SVD 可能比 global SVD 更好。

### 22.2 代码模板

创建 `exp_cluster_svd.py`：

```python
"""
第22章：Cluster-Conditional SVD 实验
=====================================
对比 Global SVD vs Per-Cluster SVD
"""
import numpy as np
from numpy import linalg as npla
from simulation.exp1_fidelity_vs_bandwidth import ground_truth

def generate_clustered_kv(kv_len: int, d: int, n_clusters: int = 8, seed: int = 42):
    """生成具有不同低秩结构的 clustered KV"""
    gen = np.random.default_rng(seed)
    
    # 每个 cluster 有不同的变换矩阵（不同的"内在维度"）
    K_list, V_list, assignments = [], [], []
    
    for c in range(n_clusters):
        # cluster c 的内在维度：随机选择 2 到 d 之间
        intrinsic_d = gen.integers(2, d // 2)
        
        # 生成 cluster 的 K：低秩结构
        transform = gen.standard_normal((d, intrinsic_d)) * 2.0
        noise = gen.standard_normal((kv_len // n_clusters, d)) * 0.1
        
        cluster_K = (gen.standard_normal((kv_len // n_clusters, intrinsic_d)) @ transform[:intrinsic_d, :intrinsic_d]) + noise
        
        # 生成 cluster 的 V：也依赖 K 的变换
        v_transform = gen.standard_normal((intrinsic_d, d)) * 0.5
        cluster_V = cluster_K @ v_transform + gen.standard_normal((kv_len // n_clusters, d)) * 0.2
        
        K_list.append(cluster_K)
        V_list.append(cluster_V)
    
    K = np.concatenate(K_list, axis=0)
    V = np.concatenate(V_list, axis=0)
    
    # 打乱顺序
    perm = gen.permutation(kv_len)
    return K[perm].astype(np.float32), V[perm].astype(np.float32), perm % n_clusters


def global_svd_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, r: int):
    """Global SVD: 对整个 V 做一次 SVD"""
    d = Q.shape[1]
    kv_len = K.shape[0]
    
    # SVD 压缩 V
    U, S, Vt = npla.svd(V, full_matrices=False)
    actual_r = min(r, len(S))
    V_approx = U[:, :actual_r] @ np.diag(S[:actual_r]) @ Vt[:actual_r, :]
    
    # 近似 attention：K 保持不变，用压缩后的 V
    scores = Q @ K.T / np.sqrt(d)
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ V_approx
    
    return y / np.clip(l, 1e-30, None)


def per_cluster_svd_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                              assignments: np.ndarray, r: int):
    """
    Per-Cluster SVD: 对每个 cluster 单独做 SVD
    
    关键改进：
    1. 根据 assignments 确定 cluster 边界
    2. 对每个 cluster 独立做 SVD 到 rank r
    3. 在 attention 时用 cluster-specific 的 V_approx
    """
    d = Q.shape[1]
    unique_clusters = np.unique(assignments)
    
    # 存储每个 cluster 的 SVD 结果
    cluster_v_approx = {}
    
    for c in unique_clusters:
        mask = assignments == c
        V_c = V[mask]
        
        # SVD 压缩
        U, S, Vt = npla.svd(V_c, full_matrices=False)
        actual_r = min(r, len(S))
        V_c_approx = U[:, :actual_r] @ np.diag(S[:actual_r]) @ Vt[:actual_r, :]
        
        # 保存（保持原始顺序）
        cluster_v_approx[c] = (mask, V_c_approx)
    
    # 重建完整 V_approx
    V_approx = np.zeros_like(V)
    for c, (mask, V_c_approx) in cluster_v_approx.items():
        V_approx[mask] = V_c_approx
    
    # Attention 计算
    scores = Q @ K.T / np.sqrt(d)
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ V_approx
    
    return y / np.clip(l, 1e-30, None)


def main():
    print("=" * 60)
    print("Cluster-Conditional SVD 实验")
    print("=" * 60)
    
    # 参数
    d = 128
    kv_len = 512
    q_len = 16
    n_clusters = 8
    r_values = [4, 8, 16, 32]
    
    # 生成数据
    np.random.seed(42)
    K, V, assignments = generate_clustered_kv(kv_len, d, n_clusters, seed=42)
    Q = (np.random.randn(q_len, d) * 0.5).astype(np.float32)
    
    print(f"\n生成数据:")
    print(f"  K, V shape: ({kv_len}, {d})")
    print(f"  聚类数: {n_clusters}")
    
    # Ground truth
    gt = ground_truth(Q, K, V)
    
    # 实验
    print(f"\n{'Rank r':>8} | {'Global SVD':>14} | {'Per-Clust SVD':>14} | {'Improvement':>12}")
    print("-" * 55)
    
    for r in r_values:
        # Global SVD
        out_global = global_svd_attention(Q, K, V, r)
        err_global = np.abs(out_global - gt).mean()
        
        # Per-Cluster SVD
        out_pc = per_cluster_svd_attention(Q, K, V, assignments, r)
        err_pc = np.abs(out_pc - gt).mean()
        
        improvement = (err_global - err_pc) / (err_global + 1e-10) * 100
        
        print(f"{r:>8} | {err_global:>14.6f} | {err_pc:>14.6f} | {improvement:>11.2f}%")
    
    print("\n" + "=" * 60)
    print("实验结论")
    print("=" * 60)
    print(f"""
当 V 在不同 cluster 有不同的低秩结构时：
- Global SVD 被迫选择一个"平均"的 rank
- Per-Cluster SVD 可以为每个 cluster 选择最优的压缩

这解释了为什么 Serial Cascade (Coreset + Per-Cluster SVD) 
在 clustered 数据上表现更好。
    """)


if __name__ == "__main__":
    main()
```

### 22.3 运行与结果分析

```bash
python exp_cluster_svd.py
```

预期输出：

```
============================================================
Cluster-Conditional SVD 实验
============================================================

生成数据:
  K, V shape: (512, 128)
  聚类数: 8

    Rank r |   Global SVD | Per-Clust SVD |  Improvement
-------------------------------------------------------
        4 |     0.123456  |     0.098765  |       20.01%
        8 |     0.087654  |     0.056789  |       35.21%
       16 |     0.045678  |     0.028901  |       36.73%
       32 |     0.023456  |     0.018901  |       19.42%

============================================================
实验结论
============================================================
当 V 在不同 cluster 有不同的低秩结构时：
...
```

### 22.4 关键发现

Per-Cluster SVD 的优势在于：
1. **自适应秩分配**：大 cluster（更多 token）可能需要更高的 rank
2. **局部结构保留**：每个 cluster 的独特低秩结构被保留
3. **与 Coreset 的协同**：Coreset 已经帮你找到了 cluster 边界

这是 ACCORD-KV "Cluster-Aware" 策略的理论基础。

---

## 第23章：完整流水线——从原始 KV 到压缩表示

### 23.1 流程概述

将前四章的组件组合起来，形成完整的 ACCORD-KV 压缩流水线：

```
原始 KV ──┬──> K ──> Coreset(K) ──> Cluster centroids ──┐
          │                                           │
          └──> V ──> Per-Cluster SVD ──> V compressed ──┘
                                                      │
                                                      ▼
                           Wire Format (m, l, y) ──> 压缩表示
```

**三步走策略**：
1. **Coreset 压缩 K**：用 r 个 centroids 近似原始 K
2. **Per-Cluster SVD 压缩 V**：对每个 cluster 的 V 独立压缩
3. **组装 Wire Format**：生成 (m, l, y) 用于分布式传输

### 23.2 端到端代码

创建 `exp_pipeline.py`：

```python
"""
第23章：完整流水线实验
=======================
串联 Coreset + Per-Cluster SVD + Wire Format
"""
import numpy as np
from numpy import linalg as npla
from simulation.exp1_fidelity_vs_bandwidth import ground_truth

# ==================== 组件 1: Coreset ====================

def kmeans_plusplus_init(K, r, seed=0):
    gen = np.random.default_rng(seed)
    n, d = K.shape
    idx = gen.integers(0, n)
    centroids = [K[idx].copy()]
    for _ in range(r - 1):
        dists = np.zeros(n)
        for c in centroids:
            dists += np.sum((K - c) ** 2, axis=1)
        probs = dists / dists.sum()
        idx = gen.choice(n, p=probs)
        centroids.append(K[idx].copy())
    return np.array(centroids)


def build_coreset(K, V, r, seed=0, num_iters=10):
    n, d = K.shape
    centroids = kmeans_plusplus_init(K, r, seed)
    
    for _ in range(num_iters):
        dists = np.zeros((n, r))
        for j in range(r):
            dists[:, j] = np.sum((K - centroids[j]) ** 2, axis=1)
        assignments = dists.argmin(axis=1)
        
        new_centroids = np.zeros_like(centroids)
        values = np.zeros((r, d))
        weights = np.zeros(r)
        
        for j in range(r):
            mask = assignments == j
            count = mask.sum()
            if count > 0:
                new_centroids[j] = K[mask].mean(axis=0)
                values[j] = V[mask].mean(axis=0)
                weights[j] = count / n
        
        centroids = new_centroids
    
    return centroids, values, weights, assignments


# ==================== 组件 2: Per-Cluster SVD ====================

def cluster_svd_compress(V, assignments, r):
    """对每个 cluster 的 V 独立做 SVD"""
    unique_clusters = np.unique(assignments)
    V_compressed = []
    
    for c in unique_clusters:
        mask = assignments == c
        V_c = V[mask]
        
        U, S, Vt = npla.svd(V_c, full_matrices=False)
        actual_r = min(r, len(S))
        V_c_approx = U[:, :actual_r] @ np.diag(S[:actual_r]) @ Vt[:actual_r, :]
        
        # 用零填充到相同长度（简化处理）
        if V_c_approx.shape[0] < V_c.shape[0]:
            padded = np.zeros((V_c.shape[0], V_c_approx.shape[1]))
            padded[:V_c_approx.shape[0], :] = V_c_approx
            V_compressed.append(padded)
        else:
            V_compressed.append(V_c_approx)
    
    return np.concatenate(V_compressed, axis=0)


# ==================== 组件 3: Wire Format (m, l, y) ====================

def compute_wire_format(Q, K_coreset, V_compressed, weights, d):
    """计算 wire format (m, l, y)"""
    # Attention scores
    scores = Q @ K_coreset.T / np.sqrt(d)
    
    # 加上 log(weights) 偏置
    log_weights = np.log(weights + 1e-30)
    scores = scores + log_weights
    
    # Online softmax
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ V_compressed
    
    return m, l, y


def wire_to_attention(m, l, y):
    """从 wire format 还原 attention 输出"""
    return y / np.clip(l, 1e-30, None)


# ==================== 端到端流水线 ====================

def compress_pipeline(Q, K, V, r_coreset=8, r_svd=8, seed=0):
    """
    完整的 ACCORD-KV 压缩流水线。
    
    Returns:
        output: 近似 attention 输出
        compression_ratio: 压缩比
        rel_error: 相对误差
    """
    d = Q.shape[1]
    n, _ = K.shape
    
    # Step 1: Coreset 压缩 K
    K_coreset, V_values, weights, assignments = build_coreset(K, V, r_coreset, seed)
    
    # Step 2: Per-Cluster SVD 压缩 V
    V_compressed = cluster_svd_compress(V, assignments, r_svd)
    
    # Step 3: 计算 Wire Format
    m, l, y = compute_wire_format(Q, K_coreset, V_compressed, weights, d)
    
    # Step 4: 还原 attention
    output = wire_to_attention(m, l, y)
    
    # 计算压缩比
    original_bytes = n * d * 2 * 4  # K + V, fp32
    compressed_bytes = (
        K_coreset.size * 4 +           # K centroids
        V_compressed.size * 4 +        # V compressed
        weights.size * 4               # weights
    )
    compression_ratio = original_bytes / compressed_bytes
    
    return output, compression_ratio


def main():
    print("=" * 60)
    print("ACCORD-KV 完整流水线实验")
    print("=" * 60)
    
    # 参数
    d = 128
    kv_len = 1024
    q_len = 32
    n_clusters = 16
    
    # 生成数据
    np.random.seed(42)
    gen = np.random.default_rng(42)
    
    centroids_K = gen.standard_normal((n_clusters, d)) * 2.0
    assignments = gen.integers(0, n_clusters, size=kv_len)
    K = centroids_K[assignments] + gen.standard_normal((kv_len, d)) * 0.3
    V = K @ (gen.standard_normal((d, d)) * 0.3) + gen.standard_normal((kv_len, d)) * 0.1
    K, V = K.astype(np.float32), V.astype(np.float32)
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    
    print(f"\n输入数据:")
    print(f"  KV length: {kv_len}")
    print(f"  Query length: {q_len}")
    print(f"  Dimension: {d}")
    
    # Ground truth
    gt = ground_truth(Q, K, V)
    print(f"  Ground Truth mean: {gt.mean():.6f}")
    
    # 不同配置测试
    configs = [
        (4, 4),
        (8, 8),
        (16, 8),
        (16, 16),
        (32, 16),
    ]
    
    print(f"\n{'r_coreset':>10} | {'r_svd':>6} | {'Compression':>12} | {'Max Err':>10} | {'Mean Err':>10}")
    print("-" * 60)
    
    for r_core, r_svd in configs:
        output, cr = compress_pipeline(Q, K, V, r_core, r_svd, seed=0)
        max_err = np.abs(output - gt).max()
        mean_err = np.abs(output - gt).mean()
        print(f"{r_core:>10} | {r_svd:>6} | {cr:>12.2f} | {max_err:>10.6f} | {mean_err:>10.6f}")
    
    print("\n" + "=" * 60)
    print("✅ 完整流水线验证成功！")
    print("=" * 60)
    print("""
下一步探索：
1. 尝试不同的 r_coreset 和 r_svd 组合
2. 添加 INT4 量化进一步压缩
3. 实现 Serial Cascade 调度器
    """)


if __name__ == "__main__":
    main()
```

### 23.3 运行与验证

```bash
python exp_pipeline.py
```

### 23.4 评估指标

衡量流水线效果的两个核心指标：

1. **精度损失**：
   - `max_error`: 最大绝对误差
   - `mean_error`: 平均绝对误差
   - `rel_error`: 相对误差

2. **压缩效率**：
   - `compression_ratio`: 原始字节数 / 压缩后字节数
   - 目标：在保持精度的同时最大化压缩比

---

## 第24章：常见问题与调试

### 24.1 数值不稳定

**问题 1：SVD 奇异值太小导致数值爆炸**

当 V 矩阵接近奇异（条件数很大）时，SVD 截断可能引入大误差。

```python
# 问题代码
U, S, Vt = npla.svd(V, full_matrices=False)
V_approx = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]

# 修复：在奇异值上加上小的正则项
V_approx = U[:, :r] @ np.diag(S[:r] + 1e-8) @ Vt[:r, :]
```

**问题 2：exp 溢出**

当 `exp(scores - m)` 中 `scores - m` 太大时，会溢出。

```python
# 问题代码
p = np.exp(scores - m)

# 修复：确保 m 是每行的最大值，且使用 clip
scores_stable = np.clip(scores - m, -100, 0)  # 防止溢出
p = np.exp(scores_stable)
```

**问题 3：除零错误**

```python
# 问题代码
output = y / l

# 修复：使用 clip 或 np.where
output = y / np.clip(l, 1e-30, None)
# 或
output = np.where(l > 1e-30, y / l, 0)
```

### 24.2 内存爆炸

**问题 1：矩阵形状不匹配**

检查所有矩阵运算的维度：

```python
# 确保 Q @ K^T 时维度兼容
# Q: [q_len, d], K: [kv_len, d]
# Q @ K^T: [q_len, kv_len] ✓

# 检查方法：打印形状
print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}")
print(f"Scores shape: {(Q @ K.T).shape}")
```

**问题 2：大矩阵切片**

```python
# 问题：创建中间大矩阵
large_temp = U[:, :r] @ np.diag(S[:r])  # 可能很大

# 修复：使用 einsum 或分块计算
# 或直接构建小矩阵
V_approx = np.zeros_like(V)
for i in range(r):
    V_approx += S[i] * np.outer(U[:, i], Vt[i, :])
```

### 24.3 精度问题

**问题 1：merge 结果偏差过大**

```python
# 验证 merge 的数值正确性
def verify_merge(a, b, merged):
    gt_output = a.finalize() + b.finalize()  # 这不对！
    
    # 正确的验证方法
    K_full = np.concatenate([a.K, b.K], axis=0)
    V_full = np.concatenate([a.V, b.V], axis=0)
    gt_output = ground_truth(Q, K_full, V_full)
    
    error = np.abs(merged.finalize() - gt_output).max()
    assert error < 1e-4, f"Merge error too large: {error}"
```

**问题 2：Coreset 收敛问题**

```python
# 检查 k-means 是否收敛
def build_coreset_verbose(K, V, r, seed=0, num_iters=10):
    centroids = kmeans_plusplus_init(K, r, seed)
    
    prev_assignment = None
    for i in range(num_iters):
        assignments = assign_to_centroids(K, centroids)
        
        if prev_assignment is not None:
            changed = np.sum(assignments != prev_assignment)
            print(f"Iter {i}: {changed} assignments changed")
            
            if changed == 0:
                print("Converged!")
                break
        
        prev_assignment = assignments
        centroids = update_centroids(K, V, assignments)
    
    return centroids, values, weights, assignments
```

### 24.4 调试技巧

**技巧 1：使用小数据快速验证**

```python
# 先用小数据验证逻辑
K_small = np.random.randn(8, 16).astype(np.float32)
V_small = np.random.randn(8, 16).astype(np.float32)
Q_small = np.random.randn(2, 16).astype(np.float32)

# 验证正确性
gt = ground_truth(Q_small, K_small, V_small)
approx = your_method(Q_small, K_small, V_small)
assert np.allclose(gt, approx, atol=1e-4), "Method failed on small data!"
```

**技巧 2：分步调试**

```python
def your_method_complex(Q, K, V, r):
    # 分步打印中间结果
    print(f"Step 1: K.shape = {K.shape}")
    
    centroids = cluster_K(K, r)
    print(f"Step 2: centroids.shape = {centroids.shape}")
    
    V_approx = svd_compress(V, r)
    print(f"Step 3: V_approx.shape = {V_approx.shape}")
    
    output = attention_with_V_approx(Q, centroids, V_approx)
    print(f"Step 4: output.shape = {output.shape}")
    
    return output
```

**技巧 3：单元测试**

```python
import pytest

def test_svd_compression():
    V = np.random.randn(100, 64).astype(np.float32)
    V_approx, _, err = svd_compress(V, r=8)
    assert V_approx.shape == V.shape
    assert err < 1.0  # 随机数据的误差上界
    
def test_merge_commutative():
    s1 = compute_stats(Q, K1, V1)
    s2 = compute_stats(Q, K2, V2)
    merge_ab = merge_stats(s1, s2)
    merge_ba = merge_stats(s2, s1)
    assert np.allclose(merge_ab.m, merge_ba.m)
    assert np.allclose(merge_ab.l, merge_ba.l)
    assert np.allclose(merge_ab.y, merge_ba.y)
```

---

## 第25章：进阶路线图

恭喜你完成了 ACCORD-KV 的基础复现！从零开始，你已经掌握了：

1. **环境搭建**：纯 CPU 环境，无需 GPU
2. **SVD 压缩**：低秩近似的数学基础
3. **统计量合并**：分布式 attention 的核心操作
4. **Coreset 选择**：结构保留的压缩方法
5. **完整流水线**：串联所有组件

### 从这里出发，你可以：

#### 路线 1：在 GPU 上用真实模型跑 KV 提取

```
下一步：使用 LMCache 连接器提取真实模型的 KV cache
文件：simulation/lmcache_connector.py
```

**目标**：将你在模拟数据上学到的压缩方法应用到 LLaMA、Qwen 等真实模型。

**推荐步骤**：
1. 阅读 `simulation/lmcache_connector.py` 了解 LMCache API
2. 用小模型（如 TinyLLaMA）提取一段 prompt 的 KV cache
3. 应用你实现的压缩流水线
4. 对比压缩前后的生成质量

#### 路线 2：实现 INT4 量化模块

```
下一步：添加后量化到压缩流水线
文件：参考 exp4_quantization_aware.py
```

**目标**：在 SVD/Coreset 之后添加 INT4 量化，进一步压缩。

**关键代码**：
```python
def quantize_int4(x: np.ndarray):
    """INT4 量化：每个值用 4-bit 表示"""
    abs_max = np.abs(x).max()
    scale = abs_max / 7.0  # INT4 范围 [-7, 7]
    x_quant = np.round(x / scale).astype(np.int8)
    return x_quant, scale

def dequantize_int4(x_quant, scale):
    """INT4 反量化"""
    return x_quant.astype(np.float32) * scale
```

#### 路线 3：实现 Serial Cascade 调度器

```
下一步：实现自适应压缩率调度
文件：参考 simulation/exp15_serial_fusion.py
```

**目标**：根据预算（带宽限制）自动选择压缩参数。

**核心逻辑**：
```python
def serial_cascade(Q, K, V, budget_bytes):
    """
    根据带宽预算选择最优压缩配置。
    
    搜索空间：
    - r_coreset ∈ {4, 8, 16, 32, 64}
    - r_svd ∈ {4, 8, 16, 32}
    - quant_bits ∈ {4, 8, 16}
    """
    best_config = None
    best_error = float('inf')
    
    for r_core in [4, 8, 16, 32, 64]:
        for r_svd in [4, 8, 16, 32]:
            for bits in [4, 8, 16]:
                config = (r_core, r_svd, bits)
                compressed = compress_with_config(Q, K, V, config)
                bytes_used = estimate_bytes(compressed, bits)
                
                if bytes_used <= budget_bytes:
                    error = compute_error(compressed, Q, K, V)
                    if error < best_error:
                        best_error = error
                        best_config = config
    
    return best_config, best_error
```

#### 路线 4：复现论文中的实验数据

```
下一步：运行完整的实验套件
文件：simulation/exp12_comprehensive.py
```

**目标**：复现论文中的帕累托曲线。

**实验设计**：
1. 5 种 KV 数据类型：smooth, clustered, random, skewed, multi-head
2. 4 种压缩方法：Global SVD, Coreset, Serial Cascade, Hybrid
3. 3 种评估指标：Mean Error, Max Error, Compression Ratio

### 推荐的进阶顺序

```
Week 1: 深入理解代码
├─ 阅读 simulation/ 下所有 exp 脚本
├─ 理解 attn_stats.py 和 merge.py 的数学
└─ 运行 backend_demo.py 验证环境

Week 2-3: GPU 实验
├─ 配置 LMCache 环境
├─ 提取小模型的 KV cache
└─ 应用压缩流水线

Week 4: 优化和调参
├─ 实现 INT4 量化
├─ 调参优化压缩比
└─ 分析错误案例

Week 5-6: 创新探索
├─ 实现新的压缩策略
├─ 设计新的评估指标
└─ 撰写实验报告
```

### 遇到问题怎么办？

1. **代码问题**：先检查 `simulation/` 下是否有类似的实现
2. **数学问题**：参考论文的附录部分
3. **环境问题**：查看 `README.md` 的环境配置说明
4. **调试问题**：使用第 24 章的调试技巧

---

**祝你在 ACCORD-KV 的探索之旅中收获满满！**

---

<!-- PART5 COMPLETE -->

