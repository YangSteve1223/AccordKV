"""
vLLM 模型加载 + KV Cache 提取 — 真实 GPU 实现
==============================================

支持 Mistral-7B-Instruct-v0.3 和 Gemma-2-9B-it
从模型层提取原始 KV Cache，用于 ACCORD 压缩

Author: ACCORD-KV Team
"""

import torch
import warnings
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


# === 设备检测 ===

def get_device() -> str:
    """获取可用计算设备"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU is required for this experiment.")
    return "cuda"


# === 模型配置 ===

MODEL_CONFIG = {
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "local_path": "/root/autodl-tmp/Mistral-7B-Instruct-v0.3",
        "num_heads": 32,
        "head_dim": 128,
        "num_layers": 32,
        "context_len": 32768,
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",  # 强制使用 eager，禁用 flash attention
        "swi": 4096,
    },
    "google/gemma-2-9b-it": {
        "local_path": "/root/autodl-tmp/gemma-2-9b-it",
        "num_heads": 16,
        "head_dim": 256,
        "num_layers": 42,
        "context_len": 8192,
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
        "swi": None,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "num_heads": 32,
        "head_dim": 128,
        "num_layers": 28,
        "context_len": 32768,
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
    },
}


def get_model_config(model_name: str) -> dict:
    """获取模型配置"""
    if model_name in MODEL_CONFIG:
        return MODEL_CONFIG[model_name]
    
    for key in MODEL_CONFIG:
        if key.split("/")[-1].lower() in model_name.lower():
            return MODEL_CONFIG[key]
    
    warnings.warn(f"Unknown model: {model_name}. Using defaults.")
    return {
        "num_heads": 32, "head_dim": 128,
        "num_layers": 32, "context_len": 8192,
        "dtype": torch.bfloat16,
        "attn_implementation": "eager",
    }


# === vLLM 加载 ===

def load_with_vllm(model_name: str, gpu_memory_utilization: float = 0.9):
    """使用 vLLM 加载模型"""
    try:
        from vllm import LLM
    except ImportError:
        raise ImportError("vLLM not installed. Run: pip install vllm")

    config = get_model_config(model_name)
    local_path = config.get("local_path", model_name)

    hf_model_kwargs = {}
    if "attn_implementation" in config:
        hf_model_kwargs["attn_implementation"] = config["attn_implementation"]

    llm = LLM(
        model=local_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=config.get("context_len", 8192),
        hf_overrides=hf_model_kwargs if hf_model_kwargs else None,
    )
    return llm, None


# === Transformers 加载 ===

def load_with_transformers(model_name: str, device: str = "cuda"):
    """使用 HuggingFace Transformers 加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = get_model_config(model_name)
    local_path = config.get("local_path", model_name)
    
    # 强制使用 eager attention，禁用 flash attention
    hf_kwargs = {
        "device_map": device,
        "torch_dtype": config.get("dtype", torch.bfloat16),
        "trust_remote_code": True,
        "attn_implementation": "eager",  # 强制禁用 flash attention
    }

    print(f"Loading {model_name} from {local_path} with attn_implementation=eager")
    
    model = AutoModelForCausalLM.from_pretrained(local_path, **hf_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# === KV Cache 提取 ===

@dataclass
class LayerKVCache:
    """单层 KV Cache 数据结构"""
    K: torch.Tensor  # [batch, num_heads, seq_len, head_dim]
    V: torch.Tensor


class KVCacheExtractor:
    """KV Cache 提取器"""
    
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.num_heads = config.get("num_heads", 32)
        self.head_dim = config.get("head_dim", 128)
        self.num_layers = config.get("num_layers", 32)
        self.device = next(model.parameters()).device
        
        self.cached_k = {}
        self.cached_v = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册 forward hook 捕获 KV"""
        def get_kv_hook(name):
            def hook(module, input, output):
                # output[0] 是 hidden states, output[1] 是 attention cache
                if hasattr(output, 'past_key_values') and output.past_key_values is not None:
                    k, v = output.past_key_values
                    if name not in self.cached_k:
                        self.cached_k[name] = []
                        self.cached_v[name] = []
                    self.cached_k[name].append(k)
                    self.cached_v[name].append(v)
            return hook
        
        # 遍历 decoder 层
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for i, layer in enumerate(self.model.model.layers):
                layer.register_forward_hook(get_kv_hook(f"layer_{i}"))
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            for i, layer in enumerate(self.model.transformer.h):
                layer.register_forward_hook(get_kv_hook(f"layer_{i}"))
    
    def clear(self):
        """清空缓存"""
        self.cached_k.clear()
        self.cached_v.clear()
    
    def get_kv(self) -> List[LayerKVCache]:
        """获取所有层的 KV Cache"""
        result = []
        for name in sorted(self.cached_k.keys()):
            k = torch.cat(self.cached_k[name], dim=2)  # [batch, heads, total_seq, dim]
            v = torch.cat(self.cached_v[name], dim=2)
            result.append(LayerKVCache(K=k, V=v))
        return result


def extract_kv_cache(model, tokenizer, prompt: str, config: dict) -> List[LayerKVCache]:
    """提取 KV Cache"""
    extractor = KVCacheExtractor(model, config)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        model(**inputs)
    
    return extractor.get_kv()


# === 压缩与重建 ===

def compress_and_reconstruct(kv_list: List[LayerKVCache], rank: int = 8):
    """对 KV Cache 压缩并重建"""
    from gpu_svd_compress import compress_kv_full, decompress_kv
    
    compressed_k = []
    compressed_v = []
    stats_list = []
    
    for layer_kv in kv_list:
        K, V = layer_kv.K, layer_kv.V  # [batch, heads, seq, dim]
        
        # SVD 压缩
        cK, cV, stats = compress_kv_full(K, V, rank=rank, quantize=False)
        
        # 重建
        K_recon, V_recon = decompress_kv(cK, cV, target_dtype=K.dtype, device=K.device)
        
        compressed_k.append(K_recon)
        compressed_v.append(V_recon)
        stats_list.append(stats)
    
    return compressed_k, compressed_v, stats_list


# === 评估 ===

def evaluate_model(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    """评估模型"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# === 设备管理 ===

def print_memory_stats():
    """打印 GPU 显存统计"""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  [GPU Memory] Allocated: {alloc:.2f} GB, Reserved: {reserved:.2f} GB, Total: {total:.2f} GB")


# === 主函数 ===

if __name__ == "__main__":
    print("Testing model loader...")
    
    device = get_device()
    print(f"Device: {device}")
    
    # 测试 Mistral
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    config = get_model_config(model_name)
    print(f"Model: {model_name}")
    print(f"Config: {config}")
    
    try:
        model, tokenizer = load_with_transformers(model_name, device=device)
        print("Model loaded successfully!")
        print_memory_stats()
        
        # 测试生成
        prompt = "Hello, how are you?"
        output = evaluate_model(model, tokenizer, prompt, max_new_tokens=50)
        print(f"Generated: {output}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
