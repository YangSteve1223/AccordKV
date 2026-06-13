# ACCORD-KV 项目文件空间 — 完整目录

> 本文档由 Agent 生成，记录项目文件空间中所有文件及其来源/用途。
> 更新于 2026-06-13

---

## 目录结构概览

```
/paper/          — 论文 LaTeX 源码与编译产物
/学习指南/       — 学习指南（markdown + docx）
/docs/           — 论文相关文档、分析报告、代码审核
/GPU_exp/        — GPU 实验脚本（exp_ppl_v8.py 等）
/simulation/     — CPU 模拟实验脚本（exp1-30）
/results/        — 模拟实验结果数据（JSON + Markdown 报告）
/gpu_results/    — GPU 真实实验结果
/reports/        — 可行性报告、代码审核报告
/用户上传/       — 用户上传的原始文件（PDF、CSV 等）
```

---

## /paper/ — 论文

| 文件 | 说明 |
|------|------|
| ACCORD_KV_paper.tex | 论文 LaTeX 主文件 |
| ACCORD_KV_paper.bib | BibTeX 参考文献 |
| ACCORD_KV_paper.bbl | 编译生成的参考文献 |
| ACCORD_KV_paper.pdf | 最新编译 PDF（v2） |
| ACCORD_KV_paper_wk.pdf | wkhtmltopdf 生成版（早期） |
| compile_paper.sh | 一键编译脚本 |
| COMPILE_README.md | 编译说明 |
| paper_outline_v0.1.md | 论文大纲 v0.1 |

**本地路径（用于 GitHub）：** `_staging/accord-kv/` 对应完整代码目录

---

## /学习指南/ — 学习指南

| 文件 | 说明 |
|------|------|
| ACCORD_KV_学习指南_深度版.md | 完整学习指南（含 35 条术语深度解释 + 学习资源推荐附录） |
| ACCORD_KV_学习指南.docx.parsed.md | docx 解析中间产物 |

**注意：** `ACCORD_KV_学习指南.docx` 在 `/docs/` 目录下

---

## /docs/ — 论文相关文档

### 核心文档
| 文件 | 说明 |
|------|------|
| README_new.md | 项目 README（来自 _staging/accord-kv/README_new.md） |
| LICENSE | MIT License |
| requirements.txt | Python 依赖 |
| repo_memory.md | Repo 记忆文档 |
| project_summary.md | 项目摘要 |

### 论文草稿与分析
| 文件 | 说明 |
|------|------|
| algorithm_improvements.md | 算法改进建议 |
| paper_outline_v0.1.md | 论文大纲 |
| ACCORD_KV_feasibility_report.md | ACCORD-KV 可行性报告 |
| AVL_feasibility_report.md | AVL 可行性报告 |
| spectrumkv_relationship.md | SpectrumKV 与 ACCORD-KV 关系说明 |

### Agent 任务文件（参考用）
| 文件 | 说明 |
|------|------|
| ACCORD_KV_agent_spec_1781182844622_0_zk3e.md | Agent 任务规格 v2 |
| ACCORD_KV_collision_matrix_1781182844622_1_xml3.csv | 碰撞矩阵 |
| ACCORD_KV_experiment_plan_1781182844622_2_mlni.csv | 实验计划 |
| ACCORD_KV_research_report_1781182844622_3_mifi.md | 研究报告 |
| ACCORD_KV_technical_route_1781182844622_4_mwjh.md | 技术路线 |

### 代码审核
| 文件 | 说明 |
|------|------|
| 代码审核报告.md | 完整代码审核报告 |
| 审核发现汇总.md | 审核发现摘要 |
| code_review/code_review_report.md | 代码审核报告 |
| code_review/code_review_findings.json | 审核发现 JSON |
| code_review/code_review_summary.csv | 审核发现 CSV |

### 子目录
| 目录 | 说明 |
|------|------|
| /anytime_theory/ | Anytime Theory 相关 |
| /backend_abstraction/ | Backend 抽象相关 |
| /baselines/ | Baseline 方法相关 |
| /fixes/ | Bug 修复记录 |
| /gpu_audit/ | GPU 脚本审计 |
| /method_d_ablation/ | Method D 消融实验 |
| /method_d_insight/ | Method D 洞察 |
| /method_d_proof/ | Method D 理论证明 |
| /paper_drafts/ | 论文各节草稿 |
| /scripts_review/ | 脚本审核 |
| /core/ | 核心算法实现（acr.py, merge.py, mock_attention_server.py 等） |

---

## /GPU_exp/ — GPU 实验脚本

| 文件 | 说明 |
|------|------|
| exp_ppl_v8.py | PPL 下游实验脚本 v8（当前最新版） |
| gpu_svd_compress_v8.py | GPU SVD 压缩核心脚本 v8 |
| gpu_model_loader.py | GPU 模型加载工具 |
| gpu_model_loader_fixed.py | 模型加载修复版 |
| exp_ppl_v7_final.py | PPL 实验 v7 最终版 |
| exp_ppl_v8.py | PPL 实验 v8 |
| gpu_all_exp_v5.py | 42 配置综合实验脚本 |
| gpu_rank32.py | Rank-32 特定实验 |
| gpu_svd_compress_v5.py / v6.py / v7.py | 各版本 SVD 压缩脚本 |
| gpu_run_exp.py | GPU 实验运行脚本 |
| mini_test.py | 迷你测试脚本 |
| ppl_results_v7.json | v7 PPL 结果 |

**注意：** `/accord_kv_failed_exp/` 下有 v8 的失败实验记录（ppl_results_v8.json, exp_v8.log）

---

## /simulation/ — CPU 模拟实验脚本

包含 exp1–exp30 的各种变体版本，关键文件：
- `accord_backend.py` — 核心 ACCORD 后端实现
- `anytime_theory.py` — Anytime 理论框架
- `policy_v2.py` — 策略实现
- `validity_v2.py` — 有效性验证
- `exp14_svd_mly_wire.py` — (m,l,y) Wire Format 实验
- `exp24_cluster_aware.py` — Cluster-Conditional V SVD
- `exp25_attention_sensitivity.py` — Attention 敏感性分析
- `exp28_lsh_pruning.py` — LSH 剪枝实验
- `lmcache_connector.py` — LMCache 连接器

---

## /results/ — 模拟实验结果

包含 exp3–exp30 的完整实验数据：
- `*_pareto.json` — Pareto 前沿数据
- `*_sweep.json` — 参数扫描结果
- `*_report.md` — 实验报告（Markdown）
- `*_sanity.json` — 完整性检查
- `ACCORD_E7_Review_Report.md` — E7 评审报告
- `bug_analysis.json` — Bug 分析
- `baselines_*` — Baseline 对比数据

---

## /gpu_results/ — GPU 真实实验结果

| 文件 | 说明 |
|------|------|
| gpu_real_experiment_report.md | 真实 GPU 实验报告 |
| gpu_verification_checklist.md | GPU 验证清单 |
| gpu_results_summary.json | GPU 结果汇总 |
| ACCORD_KV_exp_report.json | 完整实验报告 JSON |
| all_summary.json / all_exp_v5_summary.json | 实验汇总 |
| m_prefill*/m_decode* — Mistral 各配置结果 |
| g_prefill*/g_decode* — Gemma 各配置结果 |
| head_analysis*.json — Attention Head 分析 |

---

## /reports/ — 可行性与审核报告

| 文件 | 说明 |
|------|------|
| ACCORD_KV_feasibility_report.md | ACCORD-KV 可行性报告 |
| AVL_feasibility_report.md | AVL 可行性报告 |
| script_review.md | 脚本审核报告 |

---

## /用户上传/ — 用户上传原始文件（仅供参考）

| 文件 | 说明 |
|------|------|
| SpectrumKV学习指南_*.pdf | SpectrumKV 学习指南 PDF |
| main_*.pdf | 原始论文 PDF |
| KVCache_Lab学习资源推荐_*.pdf | KVCache Lab 学习资源 PDF |
| image_*.png | 截图 |
| ACCORD_KV_agent_spec_*.md | Agent 规格文件（多版本） |
| ACCORD_KV_collision_matrix_*.csv | 碰撞矩阵（多版本） |
| ACCORD_KV_experiment_plan_*.csv | 实验计划（多版本） |
| ACCORD_KV_research_report_*.md | 研究报告（多版本） |
| ACCORD_KV_technical_route_*.md | 技术路线（多版本） |

---

## 上传到 GitHub 的推荐结构

建议将 GitHub 仓库整理为以下结构：

```
ACCORD-KV/
├── README.md              ← README_new.md（重命名）
├── LICENSE                ← MIT License
├── requirements.txt
├── paper/
│   ├── ACCORD_KV_paper.tex
│   ├── ACCORD_KV_paper.bib
│   ├── ACCORD_KV_paper.bbl
│   ├── ACCORD_KV_paper.pdf   ← 编译后放入
│   ├── compile_paper.sh
│   └── COMPILE_README.md
├── 学习指南/
│   └── ACCORD_KV_学习指南_深度版.md
├── src/
│   ├── core/               ← acr.py, merge.py, exact_local.py 等
│   ├── simulation/         ← accord_backend.py, policy_v2.py 等
│   └── gpu/               ← gpu_svd_compress_v8.py, gpu_model_loader.py 等
├── scripts/
│   └── simulation/         ← exp1–exp30 各种变体
├── results/
│   ├── figures/            ← fig1_plot.py 等绘图脚本
│   └── data/              ← 各类 JSON 结果数据
└── docs/                   ← 代码审核报告、可行性报告等
```

**应排除：**
- `用户上传/` 下的所有文件（原始上传，仅供参考）
- `BUS-BRA/` 相关（已废弃的乳腺超声项目）
- `Flow_Matching*.md`（旧项目）
- `debug_*.py`、`check_*.py`（调试脚本）
- `_staging/accord-kv/` 中的 `node_modules/`、`__pycache__/`
