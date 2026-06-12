# GPU 脚本审核报告

**审核时间**: 2026-03-23
**审核文件**: gpu_svd_compress.py, gpu_wire_format.py, gpu_model_loader.py, gpu_run_exp.py

---

## P0 问题（阻断）

### 问题 1: KV 提取是仿真而非真实 KV Cache
- **文件**: gpu_run_exp.py
- **问题描述**: 
  - `_extract_kv_from_vllm()` 和 `_extract_kv_from_transformers()` 两个函数都用**随机投影**估算 KV Cache：
    ```python
    torch.manual_seed(layer_id)
    k_proj = torch.randn(hidden_dim, num_heads * head_dim, device=device) * 0.02
    K = (h @ k_proj).reshape(...)  # 这是模拟数据！
    ```
  - 注释说是 "simplified method"，但实际拿到的是与真实 KV Cache 无关的随机数据
  - 整个压缩/解压实验在仿真数据上运行，**无法验证 ACCORD 压缩在真实 KV 上的效果**
- **严重程度**: P0
- **建议修复**:
  1. **方案 A（推荐）**: 使用 KVCacheExtractor 从模型真实提取 KV（需要调试 hook 适配目标模型）
  2. **方案 B**: 如果 vLLM 无法提取，直接用 `torch.randn` 明确标注为仿真，不要假装从模型提取
  3. **方案 C**: 使用 `model.trace()` 或修改 vLLM 源码获取内部 KV（需参考 vLLM 源码）

### 问题 2: V 压缩未使用独立 SVD
- **文件**: gpu_svd_compress.py
- **问题描述**:
  - 理论上 K 和 V 都应做 SVD 压缩以捕获各自的低秩结构
  - 代码实际用 `V_proj = V_h.T @ U_h` 投影到 K 的 U 基上，而非对 V 独立 SVD
  - 这会导致 V 的压缩质量低于 K（V 的奇异值结构未被利用）
- **严重程度**: P0
- **建议修复**: 对 V 也做独立 SVD：`V = U_V @ S_V @ Vt_V`，或使用联合 SVD（K/V concat 后分解）

### 问题 3: KVCacheExtractor 可能无法工作
- **文件**: gpu_model_loader.py
- **问题描述**:
  - Hook 机制尝试三种路径获取 transformer 层：`model.model.layers`、`model.transformer.h`、`model.decoder.layers`
  - 对于 Mistral/Llama 架构这些路径可能有效
  - **Gemma-2-9B-it 的架构不同**，`attn_implementation="eager"` 但模型结构可能不匹配
  - Hook 注册在 layer 上，但 `_make_hook` 尝试提取 output tuple 中的 KV，很多模型的 output 格式不包含原始 KV
- **严重程度**: P0
- **建议修复**:
  1. 针对每个模型架构单独实现 KV 提取方法
  2. 对于 RoPE 模型（Mistral/Llama），使用 `past_key_value.key_cache/value_cache` 格式
  3. 添加 GQA/MQA 支持（当前只处理 num_kv_heads = num_q_heads 的情况）

---

## P1 问题（重要）

### 问题 4: V 投影压缩算法不合理
- **文件**: gpu_svd_compress.py (line ~54)
- **问题描述**:
  ```python
  V_proj = V_h.T @ U_h  # V 在 K 的 U 基上的投影
  V_comp_h = (U_h * S_h.unsqueeze(0)) @ V_proj.T
  ```
  - V_comp 重建依赖 K 的奇异向量，这不是真正的低秩近似
  - 如果 V 和 K 的主方向不一致，压缩质量会严重下降
- **严重程度**: P1
- **建议修复**: 
  ```python
  # 方案 1: 独立 SVD
  U_V, S_V, Vt_V = torch.linalg.svd(V_h, full_matrices=False)
  V_comp_h = (U_V[:, :rank] * S_V[:rank].unsqueeze(0)) @ Vt_V[:rank, :]
  
  # 方案 2: 联合分解 (更复杂但可能更好)
  # concat K 和 V，然后对 concat 做 SVD
  ```

### 问题 5: 压缩比计算不一致
- **文件**: gpu_run_exp.py (line ~178-180)
- **问题描述**:
  ```python
  compressed_bytes += cK["data"].numel() * 0.5 * 2  # 假设 K+V 都是 int4
  if cK["U"] is not None:
      compressed_bytes += cK["U"].numel() * 4 + cK["S"].numel() * 4
  ```
  - 这里计算的是解压后大小估算，但 `compress_kv_full` 返回的 `stats["compressed_bytes"]` 计算方式不同
  - `stats["compressed_bytes"]` 用 `cK["data"].numel() * 0.5 * 2`，忽略了 U/S 的存储开销
- **严重程度**: P1
- **建议修复**: 统一压缩比计算，或在 `gpu_svd_compress.py` 中添加准确的压缩后字节数计算

### 问题 6: Wire Format 压缩比计算错误
- **文件**: gpu_wire_format.py (line ~105)
- **问题描述**:
  ```python
  original_bytes = num_heads * num_tokens * head_dim * 4 * 2
  wire_bytes = len(wire)
  compression_ratio = original_bytes / max(wire_bytes, 1)
  ```
  - `wire_info` 中 `original_bytes` 只算了 float32，但 K/V 实际是 float16
  - 应为 `num_heads * num_tokens * head_dim * 2 * 2`（head_dim × 2 types × 2 bytes）
- **严重程度**: P1
- **建议修复**:
  ```python
  original_bytes = num_heads * num_tokens * head_dim * 2 * 2  # float16: 2 bytes
  ```

### 问题 7: SVD fallback 使用随机矩阵
- **文件**: gpu_svd_compress.py (line ~44-46)
- **问题描述**:
  ```python
  except Exception:
      U = torch.randn(head_dim, rank, device=K.device)
      S = torch.ones(rank, device=K.device)
      Vt = torch.randn(rank, num_tokens, device=K.device)
  ```
  - SVD 失败时用随机矩阵替代，结果毫无意义
  - 静默失败可能导致后续实验结果完全错误
- **严重程度**: P1
- **建议修复**:
  ```python
  except Exception as e:
      warnings.warn(f"SVD failed for head {h}: {e}. Using identity approximation.")
      U = torch.eye(head_dim, rank, device=K.device)  # 单位阵更安全
      S = torch.ones(rank, device=K.device)
      Vt = torch.eye(rank, num_tokens, device=K.device)
  ```

---

## P2 问题（次要）

### 问题 8: 没有 CUDA 环境检查
- **文件**: gpu_run_exp.py
- **问题描述**: 脚本假设 CUDA 始终可用，但如果 `torch.cuda.is_available() == False`，会导致崩溃
- **严重程度**: P2
- **建议修复**: 在 `main()` 开头添加检查
  ```python
  if not torch.cuda.is_available():
      raise RuntimeError("CUDA is required for this experiment")
  ```

### 问题 9: 内存泄漏风险
- **文件**: gpu_model_loader.py
- **问题描述**: `KVCacheExtractor.__del__` 调用 `self.clear()`，但 Python 不保证析构函数调用时机
- **严重程度**: P2
- **建议修复**: 使用 context manager 或显式调用 `clear()`

### 问题 10: 模型配置覆盖问题
- **文件**: gpu_model_loader.py (line ~37)
- **问题描述**: 
  ```python
  "google/gemma-2-9b-it": {
      "num_heads": 16,   # 这是 kv_heads
  }
  ```
  - Gemma-2-9B 是 GQA，num_heads 应为 32（q_heads），kv_heads 为 16
  - 注释写的是 kv_heads，但其他代码假设是 q_heads
- **严重程度**: P2
- **建议修复**:
  ```python
  "google/gemma-2-9b-it": {
      "num_q_heads": 32,
      "num_kv_heads": 16,
      ...
  }
  ```

---

## 总结

### 总体评价：⚠️ **需修复后上机**

### 核心阻断问题

| # | 问题 | 文件 | 影响 |
|---|------|------|------|
| P0-1 | KV 提取是仿真 | gpu_run_exp.py | **无法验证压缩效果** |
| P0-2 | V 压缩方法 | gpu_svd_compress.py | **V 压缩质量差** |
| P0-3 | KVCacheExtractor | gpu_model_loader.py | **无法提取真实 KV** |

### 上机前必须修复

1. **修复 P0-1**: 要么实现真实的 KV 提取（推荐），要么明确标注为仿真
2. **修复 P0-2**: 对 V 也做独立 SVD
3. **修复 P0-3**: 调试或移除 KVCacheExtractor，确保提取流程可用

### 建议修复优先级

```
P0: 阻断问题（必须修复）
    ├── P0-1: KV 提取是仿真
    ├── P0-2: V 压缩方法
    └── P0-3: KVCacheExtractor

P1: 重要问题（建议修复）
    ├── P1-4: V 投影算法
    ├── P1-5: 压缩比计算
    ├── P1-6: Wire Format 计算
    └── P1-7: SVD fallback

P2: 次要问题（可延后）
    ├── P2-8: CUDA 检查
    ├── P2-9: 内存泄漏
    └── P2-10: GQA 配置
```

### 建议下一步

1. **短期**: 先修复 P0-1 和 P0-2，在仿真模式下跑通 pipeline
2. **中期**: 参考 vLLM 源码（如 `vllm/worker/model_runner.py`）实现真实 KV 提取
3. **长期**: 针对每个模型架构单独优化 KV 提取方法

---

**审核人**: Code Review Agent
**状态**: 需要主 agent 决策是否继续修复
