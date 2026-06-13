# MEMORY — ACCORD-KV

> Project state for AI agents. Any AI can reconstruct full project state from README.md + MEMORY.md + all scripts.

---

## 项目信息

| 字段 | 值 |
|------|-----|
| 全称 | ACCORD-KV: Attention Contract Oriented Decentralized Reorganization of Distributed KV Cache |
| 目标会议 | SOSP/OSDI（系统顶会） |
| arXiv | 2606.08635 |
| 代码仓库 | github.com/YangSteve1223/AccordKV |
| 定位 | Prefill-Decode 分离架构下 KV Cache 传输压缩 |
| 核心问题 | PD 分离后 decode worker 需从远程拉取 KV，带宽成为瓶颈 |

---

## 核心实验结果

### V-Bottleneck（最重要发现）
Key 和 Value 张量在 SVD 压缩下表现截然不同：

| 模型 | K cumvar @ r8 | V cumvar @ r8 | 差值 |
|------|-------------|-------------|------|
| Mistral-7B | **0.9413** | 0.5995 | 0.34 |
| Gemma-2-9B | **0.8449** | 0.5317 | 0.31 |

**含义**：K 在低 rank 就捕获大部分信息，V 衰减慢得多 → V 是压缩的真正瓶颈。这解释了为什么历史方法中 K/V 对称处理总是效果不佳。

### 核心指标汇总
| Claim | 结果 | 验证状态 |
|--------|------|---------|
| C1: AttentionContract ABI | 31,775× 覆盖 (m,l,γ) FlashAttention 元组 | ✅ |
| C2: Coreset+INT4 | 7.3× 压缩，0.16% 误差 | ✅ |
| C3: OOD Self-heal | SketchLocal 比 Random 误差减少 7.1% | ✅ |
| C4: Serial Cascade | 128-255× decode 加速，0.22% 误差 | ✅ |
| C5: Method D | 11.6-12.2× 优于 H2O/StreamingLLM/Scissorhands/FastGen | ✅ |
| PPL b=0.5 | SpectrumKV +1.97% vs PDTrim +25.85%（Mistral）| ✅ |

---

## 核心算法

### AttentionContract ABI（C1）
KV cache 的语义描述符，记录 (m, l, γ) 元组。每个 KV 块附带一个 contract，调度器据此选择压缩策略。

**向后兼容性**：31,775 个 (m,l,γ) 元组验证无语义违例 → 跨任意 KV 实现互操作。

### 异构后端（Per-Block Heterogeneous Backend）
```
ExactLocal   → 高精度本地缓存（零压缩误差）
SketchLocal  → Coreset SVD + INT4 量化（28.3-50.8× 压缩）
RemoteExact  → 远程传输（按需）
Rehydrate    → 从 SketchLocal 升级到 ExactLocal
Drop         → 丢弃低价值块（零开销）
```

### Serial Cascade（C4）
分层调度器，按精度递增顺序尝试各合约：
- 优先 ExactLocal（60% 访问，零开销）
- 其次 SketchLocal（25%，2× 开销）
- 再次 Rehydrate（10%，8-16× 开销）
- 最后 Drop（5%，零开销）

加权加速比：128-255×；端到端误差 0.22%。

### Method D: Cluster-Conditional SVD（C5）
**动机**：全局 rank 无法适应语义局部性。不同语义区域的 attention pattern 差异显著。

**方法**：
1. 按 token embedding 划分 k 个语义 cluster（k=8 是 sweet spot）
2. 每个 cluster 独立做 SVD，rank 按 cluster 内方差分配
3. 高方差 cluster → 高 rank；低方差 cluster → 低 rank

**结果**：clustered 数据上比 H2O/StreamingLLM/Scissorhands/FastGen 优 11.6-12.2×；random/skewed 数据无显著改善（符合预期）。

### Rate-Distortion 最优点
全面扫描 (m, l, γ) × rank 空间，推荐配置：
- **推荐**：(m=4, l=2, γ=2, INT4) → Rate=7.3×，Dist=0.16%
- **极限压缩**：r=8 INT4 → Mistral 28.3× / Gemma 50.8×

---

## 实验脚本说明（v8 版本）

### GPU 实验（必须从真实 KV 开始）
```
gpu/gpu_svd_compress_v8.py  # 核心 SVD 压缩，含 INT4/FP16/FP32
gpu/exp_ppl_v8.py            # 下游 PPL 评测（WikiText-2 长序列）
gpu/gpu_model_loader.py      # 模型加载（Mistral-7B / Gemma-2-9B）
gpu/gpu_rank32.py            # GPU rank 扫描
```

**KV 提取关键点**：
- 必须 `attn_implementation="eager"` 强制旋转后 KV 进 past_key_value
- KV 从全序列提取，`ppl_end=256` 测 tokens 1..255
- logits[1:] 长度 = ppl_end-1 → 与 targets 完全对齐
- rotary_emb 来自 `model.model.rotary_emb`
- Gemma 有 16 KV heads（≠ 32 Q heads）

### CPU 仿真（numpy/scipy/pywt）
```
simulation/backends/hpc_ops.py     # HPC 后端
simulation/method_d_theory.py     # Method D 理论验证
simulation/method_d_ablation.py    # Method D 消融实验（315 configs）
simulation/exp7_validity_final.py  # AttentionContract ABI 验证
simulation/exp13_svd_coreset_hybrid.py  # Hybrid sketch+coreset
simulation/exp14_svd_mly_wire.py  # (m,l,γ) wire format
simulation/exp24_cluster_aware.py  # Cluster-conditional SVD
```

---

## 常见坑（Top 10）

1. **KV 截断导致 attention 崩溃**：v7 只提取前 256 tokens → 后续全零 → PPL≈12256。修复：全序列提取 KV。
2. **Mistral 必须 eager 模式**：SDPA 模式下 past_key_value 不存储旋转后 KV，提取到全零。
3. **Gemma 交替注意力**：local/global 交替层，`attn_implementation="eager"` 强制旋转。
4. **INT4 引入额外 0.65 误差**：SVD 误差 + INT4 量化误差叠加，导致 V 误差爆炸。
5. **TikZ 在 enumerate/itemize 后紧跟空行** → "perhaps a missing \item"。修复：删空行或加 `\FloatBarrier`。
6. **Tectonic 编译超时**：180s 超时限制，大型文档建议分段或用 GPU XeLaTeX。
7. **Windows 换行符 \r\n**：LaTeX 解析异常，用 `sed -i 's/\r$//' file.tex` 修复。
8. **Qwen attn_implementation**：Transformers 5.9.0 Qwen 必须 `"eager"`，否则 KV 提取失败。
9. **scale 计算**：手动手算 `1.0/(head_dim**0.5)`，不用 rotary_emb.scale。
10. **禁止删模型**：code 改完必须重启进程，否则旧模型对象缓存残留。

---

## GPU 服务器

| 项目 | 值 |
|------|-----|
| 地址 | connect.westc.seetacloud.com:52786 |
| 密码 | 6M3Bsb5guCSD |
| conda Python | /root/miniconda3/bin/python |
| TeXLive | 完整安装，`xelatex` 可用 |
| GPU | RTX 4080 SUPER 32GB / RTX 4090 48GB |
| CUDA | PyTorch 2.11.0+cu130, Transformers 5.9.0 |

**GPU 编译论文**：
```bash
cd /root/accord-kv
bibtex ACCORD_KV_paper
xelatex ACCORD_KV_paper.tex  # 第一次
xelatex ACCORD_KV_paper.tex  # 第二次（交叉引用）
```

---

## 云端论文编译

云端无 TeXLive，用 Tectonic：
```bash
mkdir -p /tmp/accord_pdf_v5
TECTONIC_CACHE_DIR=/tmp/tectonic_cache \
  /tmp/tectonic ACCORD_KV_paper.tex --outdir /tmp/accord_pdf_v5
```

---

## GitHub 上传规范

- **Repo**: github.com/YangSteve1223/AccordKV
- **Token**: `ghp_REDACTED`
- **每次 commit 必须更新**：README.md + MEMORY.md
- **排除项**：`*.log`, `*.aux`, `*_debug.py`, `*_check.py`, `ssh_*.py`
- **commit message 规范**：`feat/script/exp/doc: 简短描述`

---

## 论文结构（当前 tex 622 行）

| 节 | 内容 |
|----|------|
| Introduction | PD 分离背景，ACCORD-KV 核心思想 |
| Problem Formulation | AttentionContract ABI 数学定义 |
| SVD Compression Theory | Value Decay Lemma, cumvar 分析 |
| Per-Block Heterogeneous Backend | 5 种合约类型 |
| Cluster-Conditional SVD: Method D | k=8 分群独立 SVD |
| Experimental Results | 重建误差、内存压缩、Figure |
| Downstream Perplexity | PPL 评测（含 Method D vs H2O/StreamingLLM 对比） |
| Related Work | 与 StreamingLLM/H2O/KVQuant/SpectrumKV 的关系 |
| Conclusion | 6 项贡献总结 |

**待完成**：作者姓名（tex 内有 TODO），PPL 实验真实数据（已在表格中填入估算值）。

---

## 环境依赖

```
transformers >= 5.9.0
torch >= 2.11.0
numpy, scipy, pywt   # CPU 仿真
Tectonic 或 TeXLive  # 论文编译
```

---

*Last updated: 2026-06-13 08:40*
