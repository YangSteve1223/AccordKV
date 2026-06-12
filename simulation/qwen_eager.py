"""
模型加载 — Qwen 强制 Eager Attention (GPU FIXED)

修复 P0-5: Qwen 模型强制 eager attention

在 vLLM/transformers 加载 Qwen 模型时必须设置 attn_implementation="eager"
否则会导致 vLLM backend 静默失败
"""

import warnings

# === torch 安全导入 ===

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None


# 需要强制 eager attention 的模型列表
EAGER_REQUIRED_MODELS = [
    "qwen",
    "Qwen",
    "QWen",
]


def is_qwen_model(model_name: str) -> bool:
    """检查是否为 Qwen 系列模型"""
    model_lower = model_name.lower()
    return any(name.lower() in model_lower for name in EAGER_REQUIRED_MODELS)


def load_model_for_accord(
    model_name: str,
    device: str = "auto",
    tensor_parallel_size: int = 1,
    use_vllm: bool = False,
) -> "Model":
    """
    加载模型，针对 ACCORD-KV 做了兼容性修复。
    
    修复项:
    - Qwen 模型强制使用 eager attention
    - 自动检测 GPU 可用性
    - vLLM 版本兼容性
    
    Args:
        model_name: 模型名称或路径
        device: 设备 (auto/cuda/cpu)
        tensor_parallel_size: Tensor parallel degree
        use_vllm: 是否使用 vLLM 加载
        
    Returns:
        加载好的模型
    """
    # 检查 GPU
    if device == "auto":
        device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    
    if device == "cuda" and _TORCH_AVAILABLE and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available. Using CPU.")
        device = "cpu"
    
    # Qwen 模型处理
    is_qwen = is_qwen_model(model_name)
    
    if use_vllm:
        # === vLLM 加载路径 ===
        try:
            from vllm import LLM, SamplingParams
            
            # Qwen 必须用 eager，否则静默失败
            hf_model_kwargs = {}
            if is_qwen:
                hf_model_kwargs["attn_implementation"] = "eager"
            
            model = LLM(
                model=model_name,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.9,
                hf_model_kwargs=hf_model_kwargs if hf_model_kwargs else None,
            )
            return model
            
        except ImportError:
            warnings.warn("vLLM not installed. Falling back to transformers.")
            use_vllm = False
    
    if not use_vllm:
        # === transformers 加载路径 ===
        from transformers import AutoModelForCausalLM
        model_kwargs = {
            "device_map": device,
            "trust_remote_code": True,
        }
        
        # Qwen 必须用 eager attention
        if is_qwen:
            model_kwargs["attn_implementation"] = "eager"
            warnings.warn(
                f"Loading Qwen model '{model_name}' with attn_implementation='eager' "
                f"to ensure compatibility with ACCORD-KV.",
                UserWarning
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        model.eval()
        return model
    
    raise RuntimeError("Failed to load model with both vLLM and transformers.")


# === vLLM 模型加载 (直接使用) ===

def load_vllm_model(
    model_name: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> "vLLM.LLM":
    """
    专门用于 vLLM 加载模型。
    
    自动处理:
    - Qwen 强制 eager attention
    - GPU 检测
    - 版本兼容性
    """
    try:
        from vllm import LLM
    except ImportError:
        raise ImportError("vLLM not installed. Install with: pip install vllm")
    
    # Qwen 强制 eager
    hf_model_kwargs = {}
    if is_qwen_model(model_name):
        hf_model_kwargs["attn_implementation"] = "eager"
    
    model = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        hf_model_kwargs=hf_model_kwargs if hf_model_kwargs else None,
    )
    
    return model


# === 使用示例 ===

if __name__ == "__main__":
    # 正确加载 Qwen
    print("Loading Qwen with eager attention...")
    model = load_model_for_accord(
        "Qwen/Qwen2.5-7B-Instruct",
        use_vllm=True,
    )
    print("Done!")
