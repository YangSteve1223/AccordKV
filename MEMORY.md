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
**状态**：已放弃（2026-06-13）；根因：hook 捕获 pre-RoPE K 导致 attention 错乱；脚本已保留至 `/accord_kv_failed_exp/`。
**⚠️ arxiv 编号**：主人同时做 ACCORD-KV 和 SpectrumKV（MLSys 2027）；搜索 arXiv:2606.08635 返回 **SpectrumKV**（PD 分离混合精度 KV 传输），方向不同，搜索时需区分。
**📄 学习资源 PDF**：`KVCache_Lab学习资源推荐.pdf`（项目文件），含12知识点完整资源，已整合入深度版学习指南（附录A 35条术语+附录D，1258行）。

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

- **AttentionContract ABI（C1）**：KV cache 语义描述符，记录 (m,l,γ) 元组。31,775 个元组验证无语义违例。
- **异构后端**：ExactLocal（零误差）/ SketchLocal（Coreset SVD + INT4，28.3-50.8×）/ RemoteExact / Rehydrate / Drop。
- **Serial Cascade（C4）**：ExactLocal（60%）→ SketchLocal（25%，2×）→ Rehydrate（10%，8-16×）→ Drop（5%）。加权加速 128-255×。
- **Method D（C5）**：按 token embedding 划分 k=8 语义 cluster，独立 SVD，高方差→高 rank。clustered 数据优 H2O/StreamingLLM/Scissorhands/FastGen 达 11.6-12.2×。
- **Rate-Distortion 推荐**：(m=4, l=2, γ=2, INT4) → 7.3× / 0.16%。

---

---

## 常见坑

1. **KV 截断崩溃**：v7 只提前 256 tokens → 后续全零 → PPL≈12256。修复：全序列提取。
2. **Mistral 必须 eager**：SDPA 不存旋转后 KV，past_key_value 全零。
3. **Gemma 交替注意力**：attn_implementation="eager" 强制旋转。
4. **INT4误差**：SVD+量化叠加，V误差爆炸。
5. **LaTeX**：TikZ空行→删；Tectonic超时→GPU xelatex。
6. **Qwen/scale**：Transformers 5.9.0 Qwen 必须 eager；scale 用 `1/(head_dim**0.5)`。
7. **禁止删模型**：改code必须重启进程，旧缓存残留
8. **Tensor bool判断**：if attn_self._cv → `if attn_self._cv is not None`；`kw.get('x')` 后用 `_h if _h is not None else fallback`；禁止用 `kw.get('x') or tensor`
9. **Double RoPE on k**：k_proj已旋转；补丁只用于q。
10. **Causal mask**：pad 到 seq_full；`.bool()` 后 `.masked_fill(~causal, -float('inf'))` 传给 SDPA。
11. **RoPE + 路径**：inv_freq shape[0]=rotary_dim/2，Mistral θ=10000/hd=128/rotary_dim=64；需 `sys.path.insert(0, "/root/accord-kv/gpu")`；cos/sin 广播返回 3D，需 `unsqueeze(1)`。
12. **MistralAttention 签名**：forward(self, hidden_states, position_embeddings, ...)；hidden_states 是**位置参数**非关键字；make_patched 需同时捕获位置参数和关键字参数。
13. **use_cache=False**：所有 model(input_ids=ids) 必须加 use_cache=False；否则 HuggingFace 缓存干扰 KV 注入。
14. **修复后必须重启**：每修复一个 bug 都要 kill 旧进程 + 上传 + 重启（debug 两小时发现旧进程在跑）。
15. **多进程残留**：ps aux | grep exp_ppl 确认没有旧进程残留再启动新进程。

---

## GPU 服务器
- **SSH**：connect.westc.seetacloud.com:**26289**；**密码**：6M3Bsb5guCSD；**Python**：/root/miniconda3/bin/python
- **GPU**：RTX 4080 SUPER 32GB；模型 Mistral ✅ / Gemma ✅ 已缓存
- **TeXLive**：GPU 有 xelatex；云端用 Tectonic；编译：`cd /root/accord-kv && bibtex + xelatex × 2`

---

## GitHub
- Repo: github.com/YangSteve1223/AccordKV；push 同步更新 README+MEMORY；排除：`*.log,*.aux,*_debug.py`

---

## 论文结构
**作者**：Yang Pengju（不写单位）

