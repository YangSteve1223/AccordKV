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

1. **KV 截断崩溃**：只提前 256 tokens → 后续全零 → PPL≈12256。修复：全序列提取。
2. **Mistral 必须 eager**：SDPA 不存旋转后 KV；Gemma 交替注意力同理 `attn_implementation="eager"`。
3. **Tensor bool判断**：`if attn_self._cv is not None`；禁止 `kw.get('x') or tensor`。
4. **Double RoPE on k**：k_proj 已旋转；补丁只用于 q。
5. **Causal mask**：pad 到 seq_full；`.masked_fill(~causal, -float('inf'))` 传给 SDPA。
6. **RoPE 路径**：inv_freq shape[0]=rotary_dim/2，Mistral θ=10000/hd=128/rotary_dim=64；`sys.path.insert(0, "/root/accord-kv/gpu")`；cos/sin 返回 3D，需 `unsqueeze(1)`。
7. **MistralAttention 签名**：forward(self, hidden_states, position_embeddings, ...)；hidden_states 是**位置参数**非关键字。
8. **use_cache=False**：所有 `model(input_ids=ids)` 必须加；否则 HuggingFace 缓存干扰 KV 注入。
9. **修复后必须重启**：kill 旧进程 + 上传 + 重启（debug 两小时发现旧进程在跑）；ps aux | grep 确认无残留。
10. **禁止删模型**：改 code 必须重启进程，旧缓存残留。
11. **GPU PPL 实验已放弃**（2026-06-13）：hook 在 pre-RoPE 捕获 K 导致 attention 崩溃；FP16 base comp≈15376（应为 1.33）；详情见 `recent_memory/decision/ppl_v8_experiment_abandoned.md`。

---

## GPU 服务器
- **SSH**：connect.westc.seetacloud.com:**26289**；**密码**：6M3Bsb5guCSD；**Python**：/root/miniconda3/bin/python
- **GPU**：RTX 4080 SUPER 32GB；模型 Mistral ✅ / Gemma ✅ 已缓存
- **TeXLive**：GPU 有 xelatex；编译：`cd /root/accord-kv && bibtex + xelatex × 2`

---

## GitHub 同步（2026-06-13）
- **Token**：见 SECRET.md；**直接 push**，不生成手动操作指南
- **993ab4a**（12:28 UTC）：README 完整版、.gitignore、paper/、core/（5个）、docs/（审核报告/可行性报告/文件索引）、gpu/、results/、学习指南/
- **4b18307**（12:29 UTC）：MEMORY.md 同步
- 项目文件空间已整理：`/paper/` `/docs/` `/GPU_exp/` `/simulation/` `/results/` `/gpu_results/` `/学习指南/` `/用户上传/`

## 自主执行模式（2026-06-13）
- **规则**：派 sub-agent 并行讨论→汇总→直接执行→干完通知，无需逐次确认
- 本次：7 个 agent（论文改进/SpectrumKV协作/代码质量/理论proof/可视化/写作/学习指南），结果待汇总后执行 top 8
- ⚠️ "不是让你做"时立即停止

## 2026-06-13 GitHub 同步与更新
- 完成项目空间 vs GitHub 全面对比
- 同步了 Python 包结构（pyproject.toml, accordkv/, tests/）
- 更新了 README + CONTRIBUTING + ARCHITECTURE
- 生成了 5 张论文图表（fig_cumvar, fig_error_rank, fig_method_d, pareto, heatmap）
- 重写了 Related Work + Abstract
- 补充了 System Design 章节 + Theory Proof
- 扩充了学习指南（基线对比章节 + Serial Cascade 伪代码）
- 同步了代码审核报告 → docs/code_review/代码审核报告.md
