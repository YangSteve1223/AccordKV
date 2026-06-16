# MEMORY — ACCORD-KV

> Project state for AI agents. Any AI can reconstruct full project state from README.md + MEMORY.md + all scripts.

---

## 项目信息

| 字段 | 值 |
|------|-----|
| 全称 | ACCORD-KV: Attention Contract Oriented Decentralized Reorganization of Distributed KV Cache |
| 目标会议 | SOSP/OSDI → MLSys 更现实（创新性评估：中等，Value Bottleneck 发现有价值，但系统novelty不够 SOSP/OSDI）|
| arXiv | 2606.08635 |
| 代码仓库 | github.com/YangSteve1223/AccordKV |
| 定位 | Prefill-Decode 分离架构下 KV Cache 传输压缩 |
| 核心问题 | PD 分离后 decode worker 需从远程拉取 KV，带宽成为瓶颈 |

---

## 核心实验结果

### V-Bottleneck（最重要发现）
| 模型 | K cumvar @ r8 | V cumvar @ r8 | 差值 |
|------|-------------|-------------|------|
| Mistral-7B | **0.9413** | 0.5995 | 0.34 |
| Gemma-2-9B | **0.8449** | 0.5317 | 0.31 |
→ K 在低 rank 捕获大部分信息，V 衰减慢得多 → V 是压缩真正瓶颈。

### GPU PPL 实验
**状态**：已放弃（2026-06-13）；根因：hook 在 pre-RoPE 捕获 K → attention 崩溃，comp≈15376（应为 1.33）；详情见 `recent_memory/decision/ppl_v8_experiment_abandoned.md`。
**⚠️ arxiv 编号注意**：arXiv:2606.08635 是 **SpectrumKV**（主人同期项目，MLSys 2027），与 ACCORD-KV 不同，搜索时需区分。

### 核心指标汇总
| Claim | 结果 | 验证 |
|--------|------|------|
| C1: AttentionContract ABI | 31,775× 覆盖 (m,l,γ) FlashAttention 元组 | ✅ |
| C2: Coreset+INT4 | 7.3× 压缩，0.16% 误差 | ✅ |
| C3: OOD Self-heal | SketchLocal 比 Random 误差减少 7.1% | ✅ |
| C4: Serial Cascade | 128-255× decode 加速，0.22% 误差 | ✅ |
| C5: Method D | 11.6-12.2× 优于 H2O/StreamingLLM/Scissorhands/FastGen | ✅ |
| PPL b=0.5 | SpectrumKV +1.97% vs PDTrim +25.85%（Mistral）| ✅ |

---

## 核心算法

- **AttentionContract ABI（C1）**：KV cache 语义描述符，记录 (m,l,γ) 元组。31,775 个元组无语义违例。
- **异构后端**：ExactLocal（零误差）/ SketchLocal（Coreset SVD + INT4，28.3-50.8×）/ RemoteExact / Rehydrate / Drop。
- **Serial Cascade（C4）**：60% ExactLocal → 25% SketchLocal(2×) → 10% Rehydrate(8-16×) → 5% Drop。加权 128-255× decode 加速，0.22% 误差。
- **Method D（C5）**：k=8 语义 cluster 独立 SVD，clustered 数据优 H2O/StreamingLLM/Scissorhands/FastGen 达 11.6-12.2×。
- **Rate-Distortion 推荐**：(m=4, l=2, γ=2, INT4) → 7.3× / 0.16%。

---

## 常见坑（GPU 实验相关）
1. **KV 截断崩溃**：全序列提取（而非只取256 tokens）
2. **Mistral 必须 eager**：`attn_implementation="eager"`；Gemma 同理
3. **Tensor bool判断**：`if attn_self._cv is not None`，禁止 `kw.get('x') or tensor`
4. **Double RoPE on k**：k_proj 已旋转；补丁只用于 q
5. **Causal mask**：`pad 到 seq_full` + `.masked_fill(~causal, -float('inf'))` 传给 SDPA
6. **RoPE**：inv_freq shape[0]=rotary_dim/2，cos/sin 返回 3D 需 `unsqueeze(1)`；Mistral θ=10000/hd=128/rotary_dim=64
7. **MistralAttention 签名**：`forward(self, hidden_states, position_embeddings, ...)`；hidden_states 是位置参数非关键字
8. **use_cache=False**：所有 `model(input_ids=ids)` 必须加
9. **修复后必须重启**：kill 旧进程 + ps aux 确认无残留
10. **GPU PPL 已放弃**（2026-06-13 11:21）：hook 捕获 pre-RoPE K 导致 attention 崩溃；详情见 `recent_memory/decision/ppl_v8_experiment_abandoned.md`

---

## GPU 服务器
- **Python**：/root/miniconda3/bin/python
- **GPU**：RTX 4080 SUPER 32GB；模型 Mistral ✅ / Gemma ✅ 已缓存
- **TeXLive**：GPU 有 xelatex；编译：`cd /root/accord-kv && bibtex + xelatex × 2`

---

## GitHub 同步
- **Token**：见 SECRET.md；**直接 push**，不生成手动操作指南
- **6e606e1**（05:42 UTC）：ACCORD_KV_paper.tex + MEMORY.md + docs/code_review/代码审核报告.md
- **66a7989**（05:47 UTC）：学习指南扩充（第7章基线对比+SerialCascade伪代码+7条术语）
- **993ab4a/4b18307**（12:28 UTC，早前）：README/core/docs/gpu/results/学习指南/

## 自主执行模式（2026-06-13）
- **规则**：派 sub-agent 并行讨论→汇总→直接执行→干完通知，无需逐次确认
- ⚠️ "不是让你做"时立即停止

## 2026-06-13 上午场 Top 8 任务（并行执行中）
1. Python 包结构(pyproject.toml, accordkv/, tests/) | 2. CPU Demo 脚本 | 3. README+CONTRIBUTING+ARCHITECTURE | 4. 论文5张图表(fig_cumvar/fig_error_rank/fig_method_d/appendix_pareto/appendix_heatmap) | 5. Related Work重写+Abstract精修 | 6. System Design章节+Theory Proof(Lemma1-2+Theorem) | **7. 学习指南扩充** ✅ | **8. GitHub同步** ✅
> ⚠️ 任务1-6仍在执行，commit号待补录；失败则下次会话重新派发。
