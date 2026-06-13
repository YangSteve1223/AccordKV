# MEMORY — ACCORD-KV Project

> 这是给 AI 智能体的项目状态文档。任何 AI 拉取仓库后，通过 README + 本文档 + 所有脚本即可重建完整项目状态。

## 项目概述
- **名称**: ACCORD-KV — Attention Contract-Based KV Cache Reorganization
- **arXiv**: 2606.08635
- **目标**: SOSP/OSDI 系统会议
- **核心问题**: Prefill-Decode 分离架构下 KV Cache 内存爆炸

## 核心算法（当前 v8）

### GPU 实验
| 文件 | 用途 |
|------|------|
| gpu/gpu_svd_compress_v8.py | SVD 压缩核心（含 INT4/FP16/FP32） |
| gpu/exp_ppl_v8.py | 下游 PPL 实验主脚本 |
| gpu/gpu_model_loader.py | 模型加载（支持 Mistral-7B/Gemma-2-9B） |
| gpu/gpu_rank32.py | GPU rank 测试 |

### CPU 仿真
| 文件 | 用途 |
|------|------|
| simulation/backends/hpc_ops.py | HPC 后端（numpy/scipy/pywt） |
| simulation/method_d_theory.py | Method D 理论 |
| simulation/exp7_validity_final.py | 有效性实验 |
| simulation/exp12_final_report.py | 汇总报告 |
| simulation/exp24_cluster_aware.py | Cluster-conditional SVD |
| simulation/exp26_rate_distortion.py | Rate-Distortion 曲线 |

### 核心脚本（Bugs 已修复）
- simulation/exp13_svd_coreset_hybrid.py — Bug: K_orig 未定义 → 已修复
- simulation/exp14_svd_mly_wire.py — Bug: avg_individual 未定义 → 已修复

## 关键实验结论
- **V-bottleneck**: K cumvar=0.94 @ rank8, V cumvar=0.60 @ rank8
- **Method D**: 11.6–12.2× 优于 H2O/StreamingLLM/Scissorhands/FastGen
- **INT4**: 7.3× 压缩，误差 0.16%
- **Serial Cascade**: 128–255× decode 加速，误差 0.22%
- **Rate-Distortion 最优**: (m=4, l=2, g=2, INT4)

## 论文状态
- **tex**: ACCORD_KV_paper.tex（含 C1-C4 Claims + Value Decay Lemma + Serial Cascade Corollary）
- **编译**: 云端 wkhtmltopdf 生成 PDF（paper/ACCORD_KV_paper_wk.pdf）
- **正式编译**: GPU 服务器 xelatex 执行 compile_paper.sh

## 环境依赖
- pip: numpy, scipy, scikit-learn, pywt
- GPU: transformers, torch（仅 GPU 服务器）
- 云端 sandbox: **禁止 import torch**

## GPU 服务器
- Host: connect.westc.seetacloud.com:52786
- 密码: 6M3Bsb5guCSD
- 模型: /root/autodl-tmp/Mistral-7B-Instruct-v0.3/
- 项目: /root/accord-kv/
- TeXLive: 已安装（xelatex 可用）

## 禁止上传
- 错误数据和 debug 文件
- 临时中间文件
- GPU 上 exp_v5.log 等调试日志

## 重要约束
- 云端 sandbox 禁止 import torch
- GPU 实验必须从真实 KV cache 提取开始
- 每次 commit 必须同步更新 README + MEMORY
