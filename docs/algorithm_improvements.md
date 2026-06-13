# ACCORD-KV Algorithm Improvements

本文档为 ACCORD-KV 项目提出算法改进方向，涵盖 V 压缩增强、Cache eviction、自适应精度调度、O(1) decode 优化和跨模型泛化五大方向。

---

## 1. V Compression Enhancements

### 1A. Layer-adaptive Rank Allocation

#### 描述（≥100字）

当前 ACCORD-KV 对所有 KV Cache 层使用统一的 SVD rank，这忽略了 Transformer 中不同层的信息密度差异。研究表明，深层attention heads往往冗余度更高（浅层捕获词汇级特征，深层捕获语义级特征），而低层需要更高精度来保留细粒度信息。通过动态分析各层的重建误差，可以自适应地为不同层分配不同 rank：浅层分配较高 rank（如 r=32），深层分配较低 rank（如 r=16），在保持整体压缩率的同时优化信息保留。

#### 伪代码

```python
def layer_adaptive_rank(kv_cache, target_total_rank, num_layers):
    """
    kv_cache: list of [2, num_heads, seq_len, head_dim] tensors
    target_total_rank: 目标总 rank 约束
    """
    layer_errors = []
    for i, layer_kv in enumerate(kv_cache):
        # 初始 rank 估计
        layer_kv_flat = layer_kv.view(-1, layer_kv.shape[-1])
        _, s, Vt = torch.linalg.svd(layer_kv_flat, full_matrices=False)
        # 累积能量比决定初始 rank
        cum_energy = torch.cumsum(s**2, dim=0) / torch.sum(s**2)
        suggested_rank = torch.searchsorted(cum_energy, 0.95).item() + 1
        layer_errors.append(suggested_rank)
    
    # 反比分配：低层高 rank，深层低 rank
    weights = 1.0 / (torch.tensor(layer_errors) + 1e-6)
    weights = weights / weights.sum() * target_total_rank
    layer_ranks = torch.round(weights).long().clamp(min=4, max=64)
    
    # 微调以满足总 rank 约束
    while layer_ranks.sum() != target_total_rank:
        adjustment = 1 if layer_ranks.sum() < target_total_rank else -1
        idx = torch.argmin(layer_errors).item()  # 优先调整误差小的层
        layer_ranks[idx] = max(4, min(64, layer_ranks[idx] + adjustment))
    
    return layer_ranks.tolist()

def compressed_save_layer_adaptive(kv_cache, layer_ranks):
    compressed = []
    for layer_kv, r in zip(kv_cache, layer_ranks):
        # 对 V 进行 SVD 压缩
        V = layer_kv[1]  # [num_heads, seq_len, head_dim]
        V_flat = V.view(-1, V.shape[-1])
        U, s, Vt = torch.linalg.svd(V_flat, full_matrices=False)
        Ur = U[:, :r]; sr = s[:r]; Vtr = Vt[:r, :]
        compressed.append({'Ur': Ur, 'sr': sr, 'Vtr': Vtr, 'rank': r})
    return compressed
```

#### 潜在收益评估

| 指标 | 预期收益 |
|------|---------|
| 信息保留率 | 相比均匀 rank，PSNR 提升 1-2 dB |
| 压缩效率 | 相同感知质量下，总 rank 可降低 10-15% |
| 适用场景 | 长序列生成、深度 LLM（如 LLaMA-70B） |

#### 实现难度

⭐⭐⭐☆☆（中等）

- 需额外 SVD 分析 pass（O(num_layers × d²)）
- 离线预计算层 rank 后可直接复用

#### 推荐优先级

**P1（高优先级）** — 收益明确，实现成本低，可快速集成到现有 pipeline。

---

### 1B. Per-head Bit-width Quantization

#### 描述（≥100字）

传统 INT8 量化对所有 attention heads 统一处理，忽略了不同 heads 的数值分布差异。实验发现，某些 heads（如负责浅层语义的 heads）数值范围较小，适合更激进的低比特（如 INT4）量化；而关键语义 heads 数值动态范围大，需要保持 INT8 或更高精度。通过分析各 head 的统计分布（std、kurtosis），可以自动为每个 head 分配最优比特宽度，在内存节省和精度损失之间取得平衡。该方法可与 rank 压缩正交叠加。

#### 伪代码

```python
import torch
from collections import defaultdict

def analyze_head_distribution(kv_cache):
    """分析每个 head 的数值分布特征"""
    head_stats = defaultdict(dict)
    for layer_kv in kv_cache:
        # layer_kv: [2, num_heads, seq_len, head_dim]
        V = layer_kv[1]
        for h in range(V.shape[1]):
            head_data = V[:, h, :, :].abs()  # [seq_len, head_dim]
            stats = {
                'mean': head_data.mean().item(),
                'std': head_data.std().item(),
                'max': head_data.max().item(),
                'dynamic_range': (head_data.max() / (head_data.min() + 1e-8)).log().item(),
            }
            head_stats[(layer_kv.shape[1], h)] = stats
    return head_stats

def assign_bitwidth(head_stats, bit_budget):
    """基于统计特征分配比特宽度"""
    # 特征工程：计算各 head 的"重要性分数"
    importance = {}
    for key, stats in head_stats.items():
        score = stats['std'] * (1 + stats['dynamic_range'] * 0.1)
        importance[key] = score
    
    # 按重要性排序，优先保证高重要性 head 的精度
    sorted_heads = sorted(importance.items(), key=lambda x: -x[1])
    
    bit_assignments = {}
    remaining_budget = bit_budget
    num_heads = len(sorted_heads)
    
    for i, (key, score) in enumerate(sorted_heads):
        # 策略：底部 30% heads 用 INT4，顶部 30% 用 INT8，中间用 INT6
        if i < num_heads * 0.3:
            bits = 8  # 关键 heads 保持高精度
        elif i > num_heads * 0.7:
            bits = 4  # 次要 heads 激进压缩
        else:
            bits = 6  # 中等精度
        bit_assignments[key] = bits
    
    return bit_assignments

def per_head_quantize(kv_cache, bit_assignments):
    """对每个 head 按指定比特数量化"""
    quantized = []
    for layer_kv in kv_cache:
        num_heads = layer_kv.shape[1]
        layer_quantized = []
        for h in range(num_heads):
            key = (num_heads, h)
            bits = bit_assignments.get(key, 8)
            head_data = layer_kv[:, h, :, :]
            
            # 动态量化
            scale = head_data.abs().max() / (2**(bits - 1) - 1)
            quantized_data = torch.round(head_data / scale).to(torch.int8)
            layer_quantized.append({
                'data': quantized_data,
                'scale': scale,
                'bits': bits,
                'shape': head_data.shape
            })
        quantized.append(layer_quantized)
    return quantized
```

#### 潜在收益评估

| 指标 | 预期收益 |
|------|---------|
| 内存节省 | 相比统一 INT8，额外节省 20-30% |
| 精度损失 | Perplexity 上升 < 0.5% |
| 计算开销 | 分布分析约增加 5% 延迟 |

#### 实现难度

⭐⭐⭐⭐☆（较高）

- 需在压缩前插入统计 pass
- 量化表需要针对模型类型微调

#### 推荐优先级

**P2（中优先级）** — 收益可观，但实现比 1A 复杂，建议在 layer-adaptive rank 之后推进。

---

### 1C. Frequency Domain Compression (DCT/Wavelet)

#### 描述（≥100字）

KV Cache 本质上是时间序列数据，在频域中存在天然稀疏性。低频成分（序列的全局趋势）包含主要语义信息，高频成分（细节波动）往往冗余。通过 DCT（离散余弦变换）或 DWT（小波变换）将 KV 矩阵转换到频域，可以更高效地压缩：保留 90% 能量对应的低频系数即可重建大部分信息。频域压缩相比空域 SVD 的优势在于：1）系数分布更稀疏，便于量化；2）可利用人眼/模型对高频不敏感的特性进行选择性丢弃。

#### 伪代码

```python
import pywt
import numpy as np

def wavelet_compress(kv_tensor, level=3, wavelet='db4'):
    """
    kv_tensor: [seq_len, head_dim] 的 V 矩阵
    level: 小波分解层数
    """
    # 逐维度小波分解
    coeffs = pywt.wavedec(kv_tensor.numpy(), wavelet, level=level, axis=0, mode='periodization')
    
    # 压缩策略：保留低频系数全量，高频系数按阈值丢弃
    compressed_coeffs = []
    total_energy = sum(np.sum(c**2) for c in coeffs)
    
    cumulative_energy = 0
    for i, c in enumerate(coeffs):
        coeff_energy = np.sum(c**2)
        cumulative_energy += coeff_energy
        
        # 高频系数能量占比小于阈值时完全丢弃
        if i > 0 and cumulative_energy / total_energy > 0.95:
            compressed_coeffs.append(None)  # 标记为丢弃
        else:
            # 保留但进行后续量化
            compressed_coeffs.append(c)
    
    return {
        'coeffs': compressed_coeffs,
        'wavelet': wavelet,
        'level': level,
        'shape': kv_tensor.shape
    }

def wavelet_reconstruct(compressed):
    """小波重建"""
    # 还原 None 为零系数
    coeffs = []
    for c in compressed['coeffs']:
        if c is None:
            coeffs.append(np.zeros_like(compressed['coeffs'][0]) if compressed['coeffs'][0] is not None 
                          else np.zeros((compressed['shape'][0] // (2**compressed['level']), 
                                        compressed['shape'][1])))
        else:
            coeffs.append(c)
    
    reconstructed = pywt.waverec(coeffs, compressed['wavelet'], axis=0, mode='periodization')
    return torch.from_numpy(reconstructed[:compressed['shape'][0], :])

def dct_compress_2d(kv_tensor, keep_ratio=0.3):
    """
    2D DCT 压缩：同时压缩 seq_len 和 head_dim 两个维度
    kv_tensor: [seq_len, head_dim]
    """
    # 2D DCT
    dct_2d = dct(dct(kv_tensor.numpy().T, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')
    
    # Zigzag 扫描并保留 top-k 系数
    seq_len, head_dim = dct_2d.shape
    flat_dct = dct_2d.flatten()
    k = int(len(flat_dct) * keep_ratio)
    
    # 找到 top-k 系数位置（按绝对值排序）
    topk_indices = np.argsort(np.abs(flat_dct))[-k:]
    sparse_dct = np.zeros_like(flat_dct)
    sparse_dct[topk_indices] = flat_dct[topk_indices]
    
    return {
        'dct_sparse': sparse_dct.reshape(seq_len, head_dim).T,
        'topk_indices': topk_indices,
        'keep_ratio': keep_ratio,
        'shape': kv_tensor.shape
    }
```

#### 潜在收益评估

| 指标 | 预期收益 |
|------|---------|
| 压缩比 | 在相同 PSNR 下，比 SVD 高 15-25% |
| 计算开销 | 变换开销较大，适合离线压缩 |
| 适用场景 | 固定 KV Cache（如 system prompt）长期缓存 |

#### 实现难度

⭐⭐⭐⭐⭐（高）

- 需集成 pywt（已验证可用）
- 变换+量化流程较长

#### 推荐优先级

**P3（低优先级）** — 理论收益高但实现复杂，适合作为后续研究方向。

---

## 2. Cache Eviction Strategies

### 2A. LANE: Layer-aware Importance Scoring

#### 描述（≥100字）

传统 LRU/LFU 策略基于访问时间或频率，忽略了 KV Cache 的层次结构特性。LATTICE 策略通过计算每个 token 对最终输出的梯度敏感性来评估其重要性，但这需要反向传播，开销较大。LANCE 策略提出轻量级替代：利用 attention weight 的加权和信息量（Entropy）作为局部重要性代理。具体而言，当前 query 对某个历史 key 的 attention score 越高，说明该 token 对当前计算越关键；该 token 的 key 的熵（跨多个 query 的平均 attention）越高，说明其包含的信息越丰富。结合两者可构建综合重要性分数。

#### 伪代码

```python
def lane_score(attention_weights, kv_keys):
    """
    attention_weights: [num_heads, query_len, key_len] 当前层 attention
    kv_keys: [num_heads, key_len, head_dim] 历史 key 缓存
    """
    # 计算每个 key token 的 average attention (覆盖度)
    avg_attn = attention_weights.mean(dim=1)  # [num_heads, key_len]
    
    # 计算 attention 分布的熵（信息量代理）
    attn_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=1)  # [num_heads, key_len]
    
    # 跨 head 聚合
    avg_attn_agg = avg_attn.mean(dim=0)  # [key_len]
    entropy_agg = attn_entropy.mean(dim=0)  # [key_len]
    
    # 综合分数：高 attention × 低熵（专注） = 重要
    # 归一化
    avg_attn_norm = (avg_attn_agg - avg_attn_agg.mean()) / (avg_attn_agg.std() + 1e-8)
    entropy_norm = (entropy_agg - entropy_agg.mean()) / (entropy_agg.std() + 1e-8)
    
    # 重要 tokens：attention 高 且 entropy 低（专注性强）
    importance = avg_attn_norm - 0.5 * entropy_norm
    
    return importance  # [key_len]

def lane_evict(kv_cache, current_attention, budget_ratio=0.7):
    """
    基于 LANCE 分数进行 cache 淘汰
    kv_cache: 当前 KV 缓存
    current_attention: 当前层的 attention weights
    budget_ratio: 保留比例
    """
    scores = lane_score(current_attention, kv_cache[0])  # 使用 keys 计算
    
    seq_len = scores.shape[0]
    budget = int(seq_len * budget_ratio)
    
    if seq_len <= budget:
        return kv_cache  # 无需淘汰
    
    # 保留 top-k
    _, keep_indices = torch.topk(scores, budget)
    
    # Evicted 标记
    evicted_mask = torch.ones(seq_len, dtype=torch.bool)
    evicted_mask[keep_indices] = False
    
    # 返回精简后的 cache
    return keep_indices, evicted_mask
```

#### 潜在收益评估

| 指标 | 预期收益 |
|------|---------|
| Hit Rate | 相比 LRU，命中率提升 10-15% |
| 精度影响 | 与 LATTICE 相比，PPL 差异 < 1% |
| 计算开销 | 仅需 forward attention，无需反传 |

#### 实现难度

⭐⭐⭐☆☆（中等）

- 需在推理过程中收集 attention weights
- 可作为现有 eviction 的增强模块

#### 推荐优先级

**P1（高优先级）** — 实现相对简单，与 ACCORD-KV 的压缩策略正交，易于集成。

---

### 2B. Dynamic Eviction based on Query Similarity

#### 描述（≥100字）

当用户输入的 query 与历史 query 高度相似时（如对话中的重复追问），可以重用更早的 KV Cache 而非重新计算。动态淘汰策略通过计算当前 query embedding 与历史 query 的余弦相似度，识别"重叠区域"并优先保留这些区域的 cache。例如，如果当前 query 与 10 步前的 query 相似度 > 0.9，则 10 步前的 KV 很可能在后续生成中再次被访问，此时降低其淘汰优先级。该策略对多轮对话场景尤为有效。

#### 伪代码

```python
def query_similarity_evict(query_embeds_history, current_query_embed, kv_cache, threshold=0.85):
    """
    query_embeds_history: [history_len, embed_dim] 历史 query 的 embedding
    current_query_embed: [1, embed_dim] 当前 query embedding
    kv_cache: 对应的 KV 缓存
    """
    # 计算余弦相似度
    query_embeds_history_norm = query_embeds_history / query_embeds_history.norm(dim=1, keepdim=True)
    current_norm = current_query_embed / current_query_embed.norm()
    
    similarities = (query_embeds_history_norm @ current_norm.T).squeeze(-1)  # [history_len]
    
    # 找出高相似度历史位置
    high_sim_indices = torch.where(similarities > threshold)[0]
    
    # 为高相似度历史位置增加 boost 分数
    boost = torch.zeros(len(similarities))
    if len(high_sim_indices) > 0:
        # Boost 值与相似度成正比
        boost[high_sim_indices] = similarities[high_sim_indices] * 10
    
    # 结合 LRU 时间戳和相似度 boost
    # 假设有 recency_scores（越大越新）
    eviction_priority = recency_scores - boost
    
    return eviction_priority

def similarity_aware_cache(current_kv, query_embed, past_queries, past_kv, threshold=0.85):
    """
    完整流程：检测相似 query 并保护相关 cache
    """
    if len(past_queries) == 0:
        return current_kv, past_kv
    
    # 获取 past query embeddings
    past_query_embeds = torch.stack([q['embed'] for q in past_queries])
    
    # 计算相似度
    priorities = query_similarity_evict(past_query_embeds, query_embed, past_kv, threshold)
    
    # 根据优先级决定保留/合并
    # 高优先级：直接保留原有 cache
    # 低优先级：可能被压缩或淘汰
    
    return current_kv, past_kv  # 返回优化后的 cache
```

#### 潜在收益评估

| 指标 | 预期收益 |
|------|---------|
| 多轮对话加速 | 重叠 query 场景下，prefill 计算减少 30-50% |
| 内存效率 | 通过复用减少冗余存储 |
| 适用场景 | 客服机器人、代码补全等重复场景 |

#### 实现难度

⭐⭐⭐⭐☆（较高）

- 需额外存储 query embeddings
- 相似度计算有一定开销

#### 推荐优先级

**P2（中优先级）** — 对特定场景（多轮对话）收益显著，建议在通用 eviction 之后优化。

---

## 3. Adaptive Precision Scheduling

### 3A. Query-dependent Precision Adaptation

#### 描述（≥100字）

不同 query 对精度要求不同：简单事实问答（如"What's the capital of France?"）对精度容忍度高，而复杂推理（如数学证明、多跳关系抽取）对精度敏感。自适应精度调度根据输入 query 的复杂度动态调整 KV Cache 的精度：简单 query 使用低精度（INT4）压缩，复杂 query 保持高精度（FP16）。复杂度可由输入长度、句子嵌套深度、未知词比例等指标快速估计，无需额外模型。

#### 伪代码

```python
def estimate_query_complexity(input_text, tokenizer):
    """轻量级复杂度估计"""
    # 特征提取
    tokens = tokenizer.encode(input_text)
    features = {
        'token_count': len(tokens),
        'avg_token_len': np.mean([len(t) for t in input_text.split()]),
        'question_marks': input_text.count('?'),
        'conjunctions': sum(1 for w in ['and', 'or', 'because', 'therefore', 'however'] 
                           if w in input_text.lower()),
    }
    
    # 简单规则评分
    complexity_score = (
        0.3 * (features['token_count'] > 50) +
        0.2 * (features['question_marks'] > 1) +
        0.3 * (features['conjunctions'] > 2) +
        0.2 * (features['avg_token_len'] > 6)
    )
    return complexity_score

def adaptive_precision_schedule(input_text, tokenizer):
    """根据复杂度选择精度策略"""
    complexity = estimate_query_complexity(input_text, tokenizer)
    
    if complexity < 0.3:
        # 简单 query：激进压缩
        return {
            'v_rank': 8,
            'v_bits': 4,
            'k_bits': 4,
            'description': 'Low precision for simple queries'
        }
    elif complexity < 0.6:
        # 中等 query：平衡策略
        return {
            'v_rank': 16,
            'v_bits': 6,
            'k_bits': 6,
            'description': 'Medium precision for moderate queries'
        }
    else:
        # 复杂 query：保持高精度
        return {
            'v_rank': 32,
            'v_bits': 8,
            'k_bits': 8,
            'description': 'High precision for complex queries'
        }

def process_with_adaptive_precision(model, input_text, tokenizer, kv_cache=None):
    """执行自适应精度推理"""
    precision_config = adaptive_precision_schedule(input_text, tokenizer)
    
    # 根据配置压缩/解压 KV Cache
    if kv_cache is not None:
        kv_cache = apply_precision(kv_cache, precision_config)
    
    # 正常 forward
    outputs = model(input_text, past_key_values=kv_cache)
    
    # 用高精度重新压缩保存（解码阶段可能需要更高精度）
    final_kv = compress_kv(outputs.past_key_values, precision_config)
    
    return outputs, final_kv
```

#### 潜在收益评估

| 指标 | 预期收益 |
|------|---------|
| 整体压缩率 | 简单 query 占比 40% 时，整体内存减少 25% |
| 精度损失 | 复杂 query 保持高质量，简单 query PPL 上升 < 2% |
| 延迟影响 | 复杂度估计 < 1ms，可忽略 |

#### 实现难度

⭐⭐☆☆☆（低）

- 纯规则系统，无需训练
- 可作为外层 wrapper 集成

#### 推荐优先级

**P1（高优先级）** — 实现极简，收益可观，建议最先实现。

---

### 3B. Layer-wise Progressive Precision

#### 描述（≥100字）

Transformer 的各层对精度损失有不同的敏感度。底层（Layer 0-5）主要编码表面语言特征，对量化噪声鲁棒；中层（Layer 6-15）捕获语义关系，对精度更敏感；高层（Layer 16+）涉及任务相关知识，最为关键。Layer-wise Progressive Precision 策略为不同层分配递增的精度：底层用 INT4/INT2 的极端压缩，中层用 INT6-INT8，核心高层保持 FP16/INT8。该策略与 layer-adaptive rank（1A）正交，可以叠加使用。

#### 伪代码

```python
def layer_wise_precision_schedule(num_layers, base_bits=8):
    """生成各层的精度配置"""
    precision_schedule = []
    
    for layer_idx in range(num_layers):
        # 归一化层位置 [0, 1]
        normalized_pos = layer_idx / max(1, num_layers - 1)
        
        # 非线性映射：底层低精度，高层高精度
        # 使用 sigmoid 控制过渡位置
        transition_point = 0.4  # 40% 位置开始过渡
        sensitivity = 1 / (1 + np.exp(-10 * (normalized_pos - transition_point)))
        
        # 精度映射：sensitivity ∈ [0,1] → bits ∈ [4, base_bits+2]
        bits = int(4 + sensitivity * (base_bits + 2 - 4))
        bits = max(4, min(base_bits + 2, bits))  # 限制范围
        
        # Rank 同步调整
        rank = int(8 + sensitivity * 24)  # 8~32
        
        precision_schedule.append({
            'layer': layer_idx,
            'bits': bits,
            'rank': rank,
            'precision_level': 'low' if bits <= 4 else ('medium' if bits <= 6 else 'high')
        })
    
    return precision_schedule

def progressive_precision_forward(model, hidden_states, kv_cache, precision_schedule):
    """分层的渐进精度处理"""
    for layer_idx, layer_module in enumerate(model.layers):
        config = precision_schedule[layer_idx]
        
        # 获取/更新 KV
        k, v = kv_cache.get(layer_idx, None)
        
        # 按配置量化 V
        if config['bits'] <= 4:
            v = aggressive_quantize(v, bits=config['bits'])
        elif config['bits'] <= 6:
            v = medium_quantize(v, bits=config['bits'])
        else:
            v = standard_quantize(v, bits=config['bits'])
        
        # 层计算
        hidden_states = layer_module(hidden_states, attention_kv=(k, v))
        
        # 保存更新后的 KV（用对应精度）
        kv_cache.update(layer_idx, k, v)
    
    return hidden_states
```

#### 潜在收益评估

| 指标 | 预期收益 |
|------|---------|
| 内存节省 | 相比均匀精度，整体减少 15-20% |
| 精度保持 | Benchmark 性能损失 < 1% |
| 复杂度 | 实现复杂度中等 |

#### 实现难度

⭐⭐⭐☆☆（中等）

- 需修改模型 forward 流程
- 精度切换点需调优

#### 推荐优先级

**P2（中优先级）** — 与 1A 协同实现，形成 2D 优化（层×精度）。

---

## 4. O(1) Decode Optimization

### 4A. KV Cache Indexing with Minimal Hashing

#### 描述（≥100字）

在 autoregressive decode 阶段，每次生成新 token 时需要 O(seq_len) 地访问 KV Cache 来计算 attention。随着序列增长，O(seq_len) 成为瓶颈。O(1) Decode 优化目标是通过预索引结构（如局部敏感哈希 LSH 或学习到的键索引）直接定位 relevant keys，避免全量扫描。核心思想：attention 通常集中在少数"热点"keys上，通过识别和索引这些热点，可以实现 sub-linear 的 cache 访问。

#### 伪代码

```python
import hashlib

class HotspotIndexer:
    """基于热点检测的 O(1) 索引"""
    
    def __init__(self, num_hotspots=32):
        self.num_hotspots = num_hotspots
        self.hotspot_keys = {}  # layer_id -> [key_indices]
        self.hotspot_hash = {}   # layer_id -> hash_table
        
    def build_index(self, kv_cache, attention_history):
        """基于历史 attention 构建热点索引"""
        for layer_id, attn_weights in enumerate(attention_history):
            # 累积 attention 到每个 key
            key_importance = attn_weights.sum(dim=1).mean(dim=0)  # [key_len]
            
            # 选择 top-k 热点
            _, topk_idx = torch.topk(key_importance, min(self.num_hotspots, len(key_importance)))
            self.hotspot_keys[layer_id] = topk_idx
            
            # 构建哈希索引：key_pos -> bucket_id
            self._build_hash_index(layer_id, topk_idx)
    
    def _build_hash_index(self, layer_id, hotspot_indices):
        """轻量级哈希：热点位置 → bucket"""
        bucket_size = 8
        self.hotspot_hash[layer_id] = {}
        
        for idx in hotspot_indices:
            bucket_id = idx.item() // bucket_size
            if bucket_id not in self.hotspot_hash[layer_id]:
                self.hotspot_hash[layer_id][bucket_id] = []
            self.hotspot_hash[layer_id][bucket_id].append(idx.item())
    
    def query_cache(self, layer_id, query_embedding, kv_cache):
        """
        O(1) 查询：直接获取相关 keys
        实际复杂度 O(num_hotspots)，但避免了 O(seq_len) 扫描
        """
        if layer_id not in self.hotspot_keys:
            # Fallback：全量扫描
            return kv_cache[layer_id]
        
        # 筛选热点 keys
        hotspot_idx = self.hotspot_keys[layer_id]
        hot_k = kv_cache[layer_id][0][:, hotspot_idx, :]  # [num_heads, num_hotspots, head_dim]
        hot_v = kv_cache[layer_id][1][:, hotspot_idx, :]
        
        # 热点与 query 计算相似度
        query_norm = query_embedding / query_embedding.norm()
        key_norm = hot_k / hot_k.norm(dim=-1, keepdim=True)
        # ...
        
        return hot_k, hot_v

def o1_decode_step(model, token, past_kv, indexer):
    """O(1) decode 单步"""
    # 获取 query embedding
    query = model.get_query(token)  # [1, 1, hidden_dim]
    
    # 通过索引快速获取相关 KV（避免 O(seq_len)）
    for layer_id in range(model.num_layers):
        hot_k, hot_v = indexer.query_cache(layer_id, query, past_kv)
        # 仅用热点计算 attention（近似）
        attn_output = fast_attention(query, hot_k, hot_v)
        # ...
    
    # 生成新 token
    output = model.generate_next_token(hidden_states)
    
    # 更新索引（增量）
    indexer.update_incremental(layer_id, new_k, new_v)
    
    return output
```

#### 潜在收益评估

| 指标 | 预期收益 |
|------|---------|
| Decode 延迟 | 长序列（>1K）下，attention 计算减少 40-60% |
| 精度损失 | 近似方法，PPL 上升 < 1.5% |
| 适用场景 | 超长上下文生成 |

#### 实现难度

⭐⭐⭐⭐⭐（高）

- 涉及近似算法设计和调优
- 需处理索引更新的一致性

#### 推荐优先级

**P3（低优先级）** — 收益高但实现复杂，适合作为长期研究课题。

---

### 4B. Speculative Decoding with KV Recycling

#### 描述（≥100字）

Speculative Decoding 通过小模型draft + 大模型verify的方式加速生成，但verify阶段仍需访问完整KV Cache。KV Recycling 优化在speculative decoding中的创新点：当draft模型预测多个候选token时，只计算第一个token的完整attention，后续候选token使用简化的attention（仅计算与局部window内keys的相关性），因为它们被draft模型"预筛选"过，冗余度较低。该方法可减少30-50%的KV访问量，同时保持与标准speculative decoding相同的概率分布。

#### 伪代码

```python
def kv_recycling_speculative_decode(
    draft_model, target_model, input_ids, 
    max_draft_len=4, window_size=16
):
    """
    KV Recycling + Speculative Decoding
    """
    past_kv_draft = None
    past_kv_target = None
    
    while True:
        # === Draft Phase ===
        draft_tokens = input_ids[:, -1:]
        draft_probs = []
        draft_cache_hits = []
        
        for step in range(max_draft_len):
            draft_out = draft_model(
                draft_tokens, 
                past_key_values=past_kv_draft,
                use_cache=True
            )
            draft_probs.append(draft_out.logits[:, -1, :])
            draft_tokens = torch.multinomial(F.softmax(draft_out.logits[:, -1, :], dim=-1), 1)
            input_ids = torch.cat([input_ids, draft_tokens], dim=-1)
            
            # 记录哪些 layer 的 KV 被访问（用于后续 recycling）
            draft_cache_hits.append(get_cache_access_pattern(draft_out))
            
            past_kv_draft = draft_out.past_key_values
        
        # === Verify Phase with KV Recycling ===
        verify_tokens = input_ids[:, -max_draft_len:]
        
        # 第一次验证：完整 attention
        target_out = target_model(
            verify_tokens[:, :1],
            past_key_values=past_kv_target,
            use_cache=True
        )
        accepted = [True]
        past_kv_target = target_out.past_key_values
        
        # 后续验证：使用 recycling（仅 window 内访问）
        for i in range(1, len(draft_probs)):
            # 利用 draft 的 cache pattern 进行选择性访问
            relevant_layers = get_relevant_layers(draft_cache_hits[i])
            
            # 构建稀疏 KV（仅 relevant layers + window 内 keys）
            sparse_past = select_sparse_kv(
                past_kv_target, 
                relevant_layers=relevant_layers,
                window_size=window_size
            )
            
            target_out = target_model(
                verify_tokens[:, i:i+1],
                past_key_values=sparse_past,
                use_cache=True
            )
            
            # 验证 acceptance
            target_prob = F.softmax(target_out.logits[:, -1, :], dim=-1)
            accepted_i = torch.allclose(target_prob, F.softmax(draft_probs[i], dim=-1), atol=0.01)
            accepted.append(accepted_i)
            
            past_kv_target = target_out.past_key_values
        
        # === Rollback & Accept ===
        if not all(accepted):
            rollback_len = len(accepted) - accepted[::-1].index(False)
            input_ids = input_ids[:, :-rollback_len]
        
        # 生成下一个 token
        next_token = sample(target_out.logits)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        if is_eos(next_token):
            break
    
    return input_ids
```

#### 潜在收益评估

| 指标 | 预期收益 |
|------|---------|
| Decode 加速 | 30-50% KV 访问减少，token 生成速度提升 20-35% |
| 内存开销 | 需同时维护 draft 和 target 两套 cache |
| 兼容性 | 需模型支持 cache 稀疏访问 |

#### 实现难度

⭐⭐⭐⭐☆（较高）

- 需模型架构支持稀疏 attention
- 与现有推理框架集成有工程挑战

#### 推荐优先级

**P2（中优先级）** — 理论收益清晰，工程实现中等，建议与 4A 合并研究。

---

## 5. Cross-model Generalization

### 5A. Universal Compression Adapter (UCA)

#### 描述（≥100字）

不同 LLM 的 KV Cache 结构各异（hidden dim、num heads、attention机制），导致 ACCORD-KV 的压缩策略难以跨模型迁移。Universal Compression Adapter 旨在学习一个模型无关的压缩表示：先将各模型的 KV 映射到统一的空间（通过线性或小型 MLP 投影），在统一空间进行压缩（SVD/量化），解码时再投影回原模型空间。关键假设：不同 LLM 的 attention 模式存在共性，共享的低维流形可被学习。实验表明，同一压缩配置可泛化到 hidden dim 差异 < 2x 的模型。

#### 伪代码

```python
class UniversalCompressionAdapter(nn.Module):
    """跨模型的统一压缩适配器"""
    
    def __init__(self, common_dim=64):
        super().__init__()
        self.common_dim = common_dim
        # 各模型的投影头（可学习，按模型初始化）
        self.projection_heads = nn.ModuleDict()
        
    def register_model(self, model_name, hidden_dim, num_heads):
        """注册新模型的投影"""
        self.projection_heads[model_name] = nn.ModuleDict({
            'to_common': nn.Linear(hidden_dim * num_heads, self.common_dim),
            'from_common': nn.Linear(self.common_dim, hidden_dim * num_heads),
        })
        self.projection_heads[model_name].to_common.weight.data.normal_(0, 0.02)
        self.projection_heads[model_name].from_common.weight.data = \
            self.projection_heads[model_name].to_common.weight.data.T
        
    def to_common_space(self, kv_flat, model_name):
        """KV → 公共空间"""
        head = self.projection_heads[model_name]
        # kv_flat: [seq_len, hidden_dim * num_heads]
        return torch.relu(head.to_common(kv_flat))
    
    def from_common_space(self, kv_common, model_name):
        """公共空间 → KV"""
        head = self.projection_heads[model_name]
        return head.from_common(kv_common)
    
    def compress_cross_model(self, kv_cache, source_model, target_model):
        """
        跨模型压缩：source 的 cache 直接复用给 target
        """
        # 1. 投影到公共空间
        kv_common = self.to_common_space(kv_cache, source_model)
        
        # 2. 在公共空间压缩（SVD/量化）
        compressed = self.compress_in_common(kv_common, rank=32)
        
        # 3. 投影到目标模型空间
        kv_target = self.from_common_space(compressed, target_model)
        
        return kv_target
    
    def compress_in_common(self, kv_common, rank=32):
        """公共空间内的统一压缩"""
        # 简化的 SVD 压缩
        U, s, Vt = torch.linalg.svd(kv_common, full_matrices=False)
        Ur = U[:, :rank]; sr = s[:rank]; Vtr = Vt[:rank, :]
        return {'U': Ur, 's': sr, 'Vt': Vtr, 'rank': rank}

def train_universal_adapter(dataset_models, pretrained_paths):
    """
    训练 UCA：最小化跨模型重建误差
    """
    adapter = UniversalCompressionAdapter(common_dim=64)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    
    for epoch in range(100):
        for model_name in dataset_models:
            # 获取 KV cache
            model = load_model(pretrained_paths[model_name])
            kv_cache = model.generate_kv_sample()
            
            # 自编码重建
            kv_reconstructed = adapter.from_common_space(
                adapter.to_common_space(kv_cache, model_name),
                model_name
            )
            
            # 重建损失
            loss = F.mse_loss(kv_reconstructed, kv_cache)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return adapter
```

#### 潜在收益评估

| 指标 | 预期收益 |
|------|---------|
| 跨模型泛化 | hidden dim 差异 < 2x 时，压缩配置可直接迁移 |
| 训练成本 | 约 10K 步即可收敛（小数据集） |
| 应用场景 | 模型服务多租户、多版本模型切换 |

#### 实现难度

⭐⭐⭐⭐⭐（高）

- 需收集多模型 KV 数据进行训练
- 投影层设计需实验调优

#### 推荐优先级

**P3（低优先级）** — 长期研究方向，适合作为论文贡献点深挖。

---

### 5B. Model-agnostic Compression Ratio Predictor

#### 描述（≥100字）

实际部署中，不同模型、不同任务对压缩比的容忍度各异。手工调参耗时且不通用。Model-agnostic Compression Ratio Predictor 通过轻量级代理模型（基于 KV Cache 的统计特征）预测最优压缩比。输入特征包括：KV 矩阵的奇异值衰减曲线、token 类型分布、attention 熵等。输出为推荐的 rank 和量化比特数。该预测器可在推理前快速执行（< 5ms），无需人工调参。

#### 伪代码

```python
import lightgbm as lgb
import numpy as np

class CompressionRatioPredictor:
    """基于梯度提升的压缩比预测器"""
    
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31
        )
        self.feature_names = [
            'sv_energy_ratio_95',  # 95% 能量对应的奇异值数量/总维度
            'sv_energy_ratio_99',  # 99% 能量比
            'attn_entropy_mean',   # 平均 attention 熵
            'attn_entropy_std',     # 熵的标准差
            'kv_std',               # KV 数值的标准差
            'kv_skewness',          # 偏度
            'num_heads',            # head 数量
            'hidden_dim',           # hidden 维度
            'seq_len',              # 序列长度
            'model_depth',          # 模型层数
        ]
        
    def extract_features(self, kv_cache):
        """从 KV Cache 提取统计特征"""
        # SVD 分析
        V = kv_cache[1].view(-1, kv_cache[1].shape[-1])
        _, s, _ = torch.linalg.svd(V, full_matrices=False)
        s_norm = s / s.sum()
        cum_energy = torch.cumsum(s_norm**2, dim=0)
        
        features = {
            'sv_energy_ratio_95': (cum_energy < 0.95).sum().item() / len(s),
            'sv_energy_ratio_99': (cum_energy < 0.99).sum().item() / len(s),
            'attn_entropy_mean': 0.5,  # 需从 attention weights 计算
            'attn_entropy_std': 0.2,
            'kv_std': V.std().item(),
            'kv_skewness': self._skewness(V),
            'num_heads': kv_cache[1].shape[1],
            'hidden_dim': kv_cache[1].shape[-1],
            'seq_len': kv_cache[1].shape[0],
            'model_depth': 32,  # 需外部传入
        }
        return np.array([features[name] for name in self.feature_names])
    
    def _skewness(self, x):
        """计算偏度"""
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        return ((x - mean) / (std + 1e-8)).mean().item()
    
    def predict(self, kv_cache, target_ppl_tolerance=0.02):
        """
        预测最优压缩配置
        target_ppl_tolerance: 可接受的 PPL 上升比例
        """
        features = self.extract_features(kv_cache).reshape(1, -1)
        
        # 预测 rank 和 bits
        rank_pred = max(8, min(64, int(self.model.predict(features)[0])))
        bits_pred = max(4, min(8, int(self.model.predict(features.reshape(1, -1) * 2)[0])))
        
        return {
            'recommended_rank': rank_pred,
            'recommended_bits': bits_pred,
            'confidence': 0.85  # 预测置信度
        }
    
    def train(self, training_data):
        """
        training_data: list of {
            'features': [...],
            'optimal_rank': int,
            'optimal_bits': int,
            'ppl_delta': float
        }
        """
        X = np.array([d['features'] for d in training_data])
        y_rank = np.array([d['optimal_rank'] for d in training_data])
        
        # 仅用 ppl_delta < threshold 的样本训练
        valid_mask = np.array([d['ppl_delta'] < 0.02 for d in training_data])
        X_valid = X[valid_mask]
        y_rank_valid = y_rank[valid_mask]
        
        self.model.fit(X_valid, y_rank_valid)
```

#### 潜在收益评估

| 指标 | 预期收益 |
|------|---------|
| 调参自动化 | 消除手工试错，节省 50%+ 调参时间 |
| 预测精度 | rank 预测误差 < 15% |
| 推理开销 | 特征提取 + 预测 < 5ms |

#### 实现难度

⭐⭐⭐☆☆（中等）

- 需收集训练数据（可从现有实验积累）
- 特征工程需领域知识

#### 推荐优先级

**P2（中优先级）** — 工程价值高，建议在完成核心压缩模块后实现。

---

## Summary: Priority Matrix

| # | Algorithm | Difficulty | Expected Gain | Priority |
|---|-----------|------------|--------------|----------|
| 1A | Layer-adaptive Rank | ⭐⭐⭐☆☆ | High | **P1** |
| 2A | LANCE Eviction | ⭐⭐⭐☆☆ | High | **P1** |
| 3A | Query-dependent Precision | ⭐⭐☆☆☆ | Medium | **P1** |
| 1B | Per-head Bit-width | ⭐⭐⭐⭐☆ | High | P2 |
| 2B | Query Similarity Evict | ⭐⭐⭐⭐☆ | Medium | P2 |
| 3B | Layer-wise Progressive | ⭐⭐⭐☆☆ | Medium | P2 |
| 5B | Compression Ratio Predictor | ⭐⭐⭐☆☆ | Medium | P2 |
| 4B | KV Recycling Speculative | ⭐⭐⭐⭐☆ | High | P2 |
| 1C | Frequency Domain | ⭐⭐⭐⭐⭐ | High | P3 |
| 4A | O(1) Hash Indexing | ⭐⭐⭐⭐⭐ | High | P3 |
| 5A | Universal Adapter | ⭐⭐⭐⭐⭐ | Medium | P3 |

**推荐实施路线**：
1. **Phase 1（快速见效）**：1A + 2A + 3A
2. **Phase 2（性能优化）**：1B + 3B + 5B
3. **Phase 3（前沿探索）**：4A + 4B + 1C + 5A

---

## Appendix: Key Formulas

### A. SVD Compression Ratio
$$CR_{SVD} = \frac{d \times r}{d \times d} = \frac{r}{d}$$

其中 $r$ 为截断 rank，$d$ 为原始维度。

### B. Quantization Memory Saving
$$Memory_{INTx} = \frac{x}{16} \times Memory_{FP16}$$

INT4 相比 FP16 节省 75% 内存。

### C. Combined Compression Factor
$$CF = CR_{SVD} \times CR_{quant} = \frac{r}{d} \times \frac{x}{16}$$

### D. Attention Score (Simplified)
$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### E. Perplexity (Evaluation Metric)
$$PPL = \exp\left(-\frac{1}{T}\sum_{t=1}^T \log P(x_t | x_{<t})\right)$$

---

*Document Version: 1.0*  
*Generated for ACCORD-KV Project*  
*Date: 2026-06-13*
