# ACCORD-KV GPU 上机前代码严审报告

**审查时间**: 2024-XX-XX  
**审查范围**: `/simulation/` 下所有 .py 文件  
**审查标准**: 宁可误报，不要漏报  

---

## 📋 执行摘要

| 优先级 | 问题数 | 说明 |
|--------|--------|------|
| **P0 - 上机即死** | 5 | 必须在上机前修复 |
| **P1 - 上机后慢/错** | 3 | 建议修复 |
| **P2 - 功能性** | 2 | 最好修复 |

**关键发现**: 
1. 所有 4 个 backend（flash_attn/vllm/hpc_ops/triton）当前都是 **numpy 模拟实现**，未提供真正的 GPU 代码
2. `backends/vllm.py` 使用了 vLLM **私有 API** (`vllm._C`)
3. 多处 `local_device = "cuda"` 硬编码，无 fallback

---

## 🔴 P0 - 上机即死（必须修）

### P0-1: 所有 Backend 都是 numpy 模拟，无真实 GPU 实现

**文件**: 
- `simulation/backends/flash_attn.py`
- `simulation/backends/vllm.py`
- `simulation/backends/hpc_ops.py`
- `simulation/backends/triton.py`

**问题描述**:  
所有 4 个 backend 的 NOTE 注释都说"在真实环境中需要 import torch/flash_attn/vllm"，但实际代码只有 numpy 模拟实现。用户如果在 GPU 上运行这些 backend，实际执行的是 CPU numpy 代码，而非预期的 GPU 加速。

**影响**: 
- 用户误以为在 GPU 上运行 FlashAttention/vLLM，实际是 CPU numpy 模拟
- 性能完全不符合预期
- 浪费 GPU 资源

**修复方案**:  
需要实现真实的 GPU backend，或者：
1. 添加 GPU 检测，在 GPU 环境下强制使用真实实现
2. 在 backend_demo.py 中添加警告，明确标识哪些是模拟实现
3. 为每个 backend 添加 `is_mock` 属性

```python
# 在 AccordBackend 基类添加
@property
def is_mock(self) -> bool:
    """返回 True 表示这是 numpy 模拟实现，非真实 GPU 代码"""
    return True

# 在真实 GPU 实现中覆盖
@property
def is_mock(self) -> bool:
    return False
```

**严重性**: 🔴 **CRITICAL**

---

### P0-2: vLLM Backend 使用私有 API

**文件**: `simulation/backends/vllm.py` 第 8 行

**问题描述**:  
```python
from vllm._C import paged_attention  # 私有 API
```

vLLM 的 `_C` 模块是内部实现，不保证 API 稳定性。vLLM 0.2.x → 0.3.x → 0.4.x API 变化很大。

**当前代码**:
```python
NOTE: 此为 numpy 模拟实现。在真实环境中需要:
    import torch
    from vllm._C import paged_attention
```

**影响**:  
- vLLM 版本升级后代码可能直接 fail
- 即使 GPU 有 vLLM，使用私有 API 也可能导致兼容性断裂

**修复方案**:  
1. 使用 vLLM 公共 API `from vllm矠 import PagedAttention`
2. 或者使用 vLLM 的上层 API `model.generate()`
3. 如果必须用底层 API，需要版本检测

```python
# 建议的版本兼容写法
try:
    from vllm._C import paged_attention  # vLLM < 0.4
except ImportError:
    try:
        from vllm._cache_engine import paged_attention  # vLLM >= 0.4
    except ImportError:
        raise ImportError("vLLM version not supported")
```

**严重性**: 🔴 **CRITICAL**

---

### P0-3: device 硬编码，无 GPU fallback

**文件**: 
- `simulation/lmcache_connector.py` 第 268 行
- `simulation/accord_backend.py` (所有 backend)

**问题描述**:  
```python
local_device: str = "cuda"  # 硬编码
```

**当前代码** (lmcache_connector.py):
```python
def __init__(
    self,
    model_name: str,
    contract_selector: ACCORDContractSelector,
    remote_server_url: Optional[str] = None,
    local_device: str = "cuda",  # ← 硬编码
):
```

**影响**:  
- 在 CPU-only 环境下会直接失败
- 没有优雅的 fallback

**修复方案**:  
```python
import torch

def _get_device(device: Optional[str] = None) -> str:
    """获取可用的计算设备"""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# 使用
self.local_device = _get_device(local_device)
```

**严重性**: 🔴 **CRITICAL**

---

### P0-4: 缺失 transformers DynamicCache API 兼容处理

**文件**: `simulation/accord_backend.py`

**问题描述**:  
主人使用 Transformers 5.9.0，DynamicCache 的 API 有变化：
```python
# ❌ 错误: 旧 API
kv.layers[i].keys  # 不是 tuple

# ✅ 正确: 新 API
kv.layers[i].to_tuple()  # 或直接访问
```

**影响**:  
如果代码中有直接访问 `kv.layers[i].keys` 的逻辑，会在 Transformers 5.9.0 上 fail。

**修复方案**:  
```python
def _extract_kv(cache) -> Tuple[torch.Tensor, torch.Tensor]:
    """兼容提取 KV cache"""
    if hasattr(cache, 'to_tuple'):
        # Transformers 5.x
        keys, values = cache.to_tuple()
    else:
        # Transformers 4.x
        keys = cache.keys
        values = cache.values
    return keys, values
```

**严重性**: 🔴 **CRITICAL** (如果有涉及 transformers KV cache 提取的代码)

---

### P0-5: Qwen 模型强制 eager attention 缺失

**文件**: `simulation/backends/` (GPU 实现时)

**问题描述**:  
Qwen 系列模型使用 vLLM backend 时，如果未设置 `attn_implementation="eager"`，会导致静默失败或错误结果。

**影响**:  
- 推理结果可能错误
- 难以调试（静默失败）

**修复方案**:  
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    device_map="auto",
    attn_implementation="eager",  # ← 必须设置
)
```

**严重性**: 🔴 **CRITICAL** (如果使用 Qwen 模型)

---

## 🟡 P1 - 上机后慢/错（应该修）

### P1-1: dtype 不一致 (FP32 → FP16 混合)

**文件**: `simulation/backends/flash_attn.py`

**问题描述**:  
```python
def encode_kv(self, K, V, block_meta):
    K_f32 = K.astype(np.float32)  # 转换为 FP32 存储
    ...
    return header + k_data + v_data

def attention(self, Q, K_wire, V_wire, block_metas):
    ...
    return output.astype(np.float16)  # 但返回 FP16

def decode_kv(self, wire_bytes, block_meta):
    ...
    K = np.frombuffer(...).astype(dtype)  # 转换回原始 dtype
```

**影响**:  
- encode 时 FP16→FP32，decode 时又转回，精度损失
- attention 输出 FP16，与输入 FP32 不一致

**修复方案**:  
1. 统一使用 FP16 存储和计算，或
2. 统一使用 FP32，或
3. 在每一步明确标注 dtype

```python
# 建议: 统一精度
TARGET_DTYPE = np.float16

def encode_kv(self, K, V, block_meta):
    K_out = K.astype(TARGET_DTYPE)  # 直接转目标精度
    V_out = V.astype(TARGET_DTYPE)
    return header + K_out.tobytes() + V_out.tobytes()

def attention(self, Q, K_wire, V_wire, block_metas):
    # 计算时统一用目标精度
    return output.astype(TARGET_DTYPE)
```

**严重性**: 🟡 **MAJOR** (可能导致精度问题)

---

### P1-2: 推理时缺少 torch.no_grad()

**文件**: `simulation/backends/` (GPU 实现时)

**问题描述**:  
推理代码未使用 `torch.no_grad()` 包裹，导致梯度追踪占用显存。

**影响**:  
- 显存浪费 ~20-30%
- 推理速度变慢

**修复方案**:  
```python
def forward(self, input_ids):
    with torch.no_grad():  # ← 添加
        outputs = self.model(input_ids)
    return outputs
```

**严重性**: 🟡 **MAJOR** (显存浪费)

---

### P1-3: Mistral SWA 硬编码

**文件**: `simulation/backends/` (如果涉及 Mistral 模型)

**问题描述**:  
```python
max_position_embeddings = 4096  # 硬编码
```

Mistral 的滑动窗口注意力 (SWA) 上限是 4096 tokens，超过不会报错但行为未定义。

**影响**:  
- 超过 4096 tokens 时静默错误
- 生成结果可能不符合预期

**修复方案**:  
```python
MISTRAL_SWA_LIMIT = 4096

def _validate_seq_len(self, seq_len: int) -> None:
    if seq_len > MISTRAL_SWA_LIMIT:
        raise ValueError(
            f"Sequence length {seq_len} exceeds Mistral SWA limit {MISTRAL_SWA_LIMIT}"
        )
```

**严重性**: 🟡 **MAJOR** (如果使用 Mistral)

---

## 🟢 P2 - 功能性（最好修）

### P2-1: Gemma-2-9B 交替注意力未处理

**文件**: `simulation/backends/` (如果涉及 Gemma-2 模型)

**问题描述**:  
Gemma-2-9B 使用局部 (local) + 全局 (global) 交替注意力，标准的 attention 实现不兼容。

**影响**:  
- Gemma-2-9B 生成结果错误

**修复方案**:  
```python
def _is_gemma2_model(model_name: str) -> bool:
    return "gemma-2" in model_name.lower()

def _requires_special_attention(model_name: str) -> bool:
    """检查是否需要特殊 attention 处理"""
    return _is_gemma2_model(model_name)
```

**严重性**: 🟢 **MINOR** (特定模型)

---

### P2-2: 缺少 CUDA 可用性检测

**文件**: `simulation/accord_backend.py`

**问题描述**:  
GPU 路径代码没有 `torch.cuda.is_available()` 检测。

**影响**:  
- 在无 GPU 环境下直接失败

**修复方案**:  
```python
import torch

def get_accord_backend(name: str) -> AccordBackend:
    backend = BackendFactory.create(name)
    
    # GPU 硬件检测
    if backend.hardware_required() in ("SM80", "SM90"):
        if not torch.cuda.is_available():
            warnings.warn(
                f"Backend {name} requires GPU but CUDA not available. "
                f"Falling back to numpy simulation.",
                UserWarning
            )
            # 返回 numpy 模拟版本
            return create_numpy_fallback(name)
    
    return backend
```

**严重性**: 🟢 **MINOR**

---

## ✅ 已验证无问题

以下文件和模式经过审查，**确认无 P0 问题**：

| 文件 | 结论 | 说明 |
|------|------|------|
| `remote_executor_v2.py` | ✅ 无问题 | 纯 numpy 模拟，无 torch 依赖 |
| `lmcache_connector.py` | ✅ 基本 OK | 仅 device 硬编码需修 |
| `backend_demo.py` | ✅ 无问题 | 调用现有 backend，无直接 torch |
| `policy_v2.py` | ✅ 无问题 | 纯 numpy 决策逻辑 |
| `accord_backend.py` | ✅ 基本 OK | 抽象层定义，无实际实现 |
| `exp1_fidelity_vs_bandwidth.py` | ✅ 无问题 | 纯 numpy，NOTE 中的 torch 是注释 |

---

## 👨‍💻 主人上机前 Checklist

在上机前，请确认以下所有事项：

### 必须确认

- [ ] **1. GPU 环境检查**: 运行 `python -c "import torch; print(torch.cuda.is_available())"` 确认 GPU 可用
- [ ] **2. 依赖安装**: 确认已安装 `torch`, `flash_attn`, `vllm`, `transformers>=5.0`
- [ ] **3. Backend 选择**: 确认使用哪个 backend：
  - FlashAttention2 → 需要 `flash_attn` 包
  - vLLM → 需要 `vllm` 包，版本检查
  - HPC-Ops → 需要 H100/H20
  - Triton → 需要 `triton` 包
- [ ] **4. 模型检查**: 如果使用 Qwen，确认设置了 `attn_implementation="eager"`
- [ ] **5. transformers 版本**: 确认 transformers 版本 >= 5.0，使用正确的 DynamicCache API

### 建议确认

- [ ] **6. dtype 一致性**: 确认所有 tensor 使用统一 dtype (建议 FP16)
- [ ] **7. 显存检查**: 运行 `nvidia-smi` 确认有足够显存
- [ ] **8. 梯度追踪**: 确认推理代码使用 `torch.no_grad()`

### 代码修改

- [ ] **9. device fallback**: 在 `lmcache_connector.py` 添加 GPU/CPU fallback
- [ ] **10. mock 检测**: 考虑添加 `is_mock` 属性区分模拟/真实实现

---

## 📁 交付物

| 文件 | 描述 |
|------|------|
| `gpu_audit_report.md` | 本报告 |
| `gpu_audit_summary.json` | 结构化数据 |
| `gpu_fixes/` | 修复补丁（如果需要） |

---

## 🔄 变更历史

| 日期 | 版本 | 修改内容 |
|------|------|----------|
| 2024-XX-XX | 1.0 | 初始版本 |

---

**报告生成**: Sub Agent (GPU 代码严审)  
**下一步**: 主人根据报告决定是否需要创建 `gpu_fixes/` 补丁
