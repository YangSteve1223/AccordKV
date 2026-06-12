"""
accord_backend.py — Base Backend (GPU FIXED)

修复 P0-1 + P0-4:
1. 添加 is_mock 属性到基类
2. 添加 DynamicCache 兼容层
"""

# === 修复片段 ===

# === 1. 在 AccordBackend 基类添加 is_mock ===

class AccordBackend(ABC):
    """
    ACCORD Backend 抽象基类
    
    定义 KV Cache 编码/解码/attention 的统一接口。
    不同 backend 实现可以在不同硬件上运行。
    
    属性:
        name: Backend 名称
        hardware_required: 最低硬件要求
        is_mock: 是否为模拟实现 (True= numpy, False=真实 GPU)
    """
    
    @property
    def is_mock(self) -> bool:
        """
        返回 True 表示这是 numpy 模拟实现，非真实 GPU 代码。
        
        子类应覆盖此属性:
        - numpy 模拟 → return True
        - 真实 GPU 实现 → return False
        """
        return True
    
    @abstractmethod
    def encode_kv(self, K: np.ndarray, V: np.ndarray, block_meta: BlockMeta) -> bytes:
        """..."""
        pass
    
    # ... 其他方法 ...


# === 2. DynamicCache 兼容层 ===

import torch
from typing import Tuple, Optional, Union

def extract_kv_from_cache(
    cache: Union["DynamicCache", Tuple[torch.Tensor, torch.Tensor], dict],
    layer_idx: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    兼容提取 KV cache。
    
    支持多种 cache 格式:
    - Transformers 5.x: DynamicCache 对象，使用 .to_tuple()
    - Transformers 4.x: DynamicCache 对象，使用 .keys/.values
    - Tuple: (keys, values) 直接是 tuple
    - Dict: {"keys": ..., "values": ...}
    
    Args:
        cache: KV cache 对象
        layer_idx: 如果 cache 是 dict，取特定层的 KV
        
    Returns:
        Tuple[keys, values]
    """
    # Transformers 5.x DynamicCache
    if hasattr(cache, 'to_tuple'):
        keys, values = cache.to_tuple()
        return keys, values
    
    # Transformers 4.x DynamicCache (旧 API)
    if hasattr(cache, 'keys') and hasattr(cache, 'values'):
        keys = cache.keys
        values = cache.values
        # 如果是 property 而非直接属性，尝试调用
        if callable(keys):
            keys = keys()
        if callable(values):
            values = values()
        return keys, values
    
    # 直接是 tuple
    if isinstance(cache, tuple) and len(cache) == 2:
        return cache[0], cache[1]
    
    # Dict 格式
    if isinstance(cache, dict):
        if layer_idx is not None:
            return cache[layer_idx]["keys"], cache[layer_idx]["values"]
        return cache.get("keys"), cache.get("values")
    
    raise ValueError(f"Unknown cache format: {type(cache)}")


def get_cache_len(cache: Union["DynamicCache", Tuple, None]) -> int:
    """获取 KV cache 的序列长度"""
    if cache is None:
        return 0
    
    # DynamicCache 格式
    if hasattr(cache, 'seen_tokens'):
        return cache.seen_tokens
    if hasattr(cache, '_seen_tokens'):
        return cache._seen_tokens
    
    # Tuple 格式
    if isinstance(cache, tuple):
        keys = cache[0]
        if isinstance(keys, torch.Tensor):
            return keys.shape[-2]
    
    # Dict 格式
    if isinstance(cache, dict):
        first_layer = next(iter(cache.values()))
        return first_layer["keys"].shape[-2]
    
    return 0


# === 3. 在 BackendFactory.create_all 添加 GPU 检测 ===

@classmethod
def create_all(cls, **kwargs) -> Dict[str, AccordBackend]:
    """创建所有 backend 实例"""
    import warnings
    
    backends = {}
    for name in cls._backends:
        try:
            backend = cls.create(name, **kwargs)
            
            # GPU backend 但无 GPU 时警告
            if not backend.is_mock and backend.hardware_required() in ("SM80", "SM90"):
                try:
                    if not torch.cuda.is_available():
                        warnings.warn(
                            f"Backend '{name}' requires GPU but CUDA not available. "
                            f"GPU execution will fail.",
                            UserWarning
                        )
                except NameError:
                    pass  # torch 未导入
            
            backends[name] = backend
        except Exception as e:
            warnings.warn(f"Failed to create backend '{name}': {e}")
    
    return backends


print("Fix P0-1 + P0-4 applied: is_mock + DynamicCache compatibility")
