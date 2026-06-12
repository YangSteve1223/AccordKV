"""
ACCORD Backend 抽象层
=====================

为 ACCORD-KV 设计的跨硬件 KV Cache backend 抽象层。
支持 4 种实现：FlashAttention 2, vLLM PagedAttention, HPC-Ops, Triton

Author: ACCORD-KV Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np


@dataclass
class BlockMeta:
    """KV Block 元数据"""
    block_id: int
    num_tokens: int
    seq_len: int
    head_dim: int
    num_heads: int
    dtype: str = "float16"
    
    def __post_init__(self):
        assert self.block_id >= 0, "block_id must be non-negative"
        assert self.num_tokens > 0, "num_tokens must be positive"
        assert self.head_dim > 0, "head_dim must be positive"
        assert self.num_heads > 0, "num_heads must be positive"


class AccordBackend(ABC):
    """
    ACCORD Backend 抽象基类
    
    定义 KV Cache 编码/解码/attention 的统一接口。
    不同 backend 实现可以在不同硬件上运行。
    
    接口契约:
    1. encode_kv/decode_kv 必须互逆
    2. attention 输出 shape 必须与 Q 一致
    3. 所有 backend 必须支持相同的 BlockMeta 格式
    
    Attributes:
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
        """
        将 KV block 编码为 wire format
        
        Args:
            K: Key tensor, shape [num_heads, num_tokens, head_dim]
            V: Value tensor, shape [num_heads, num_tokens, head_dim]
            block_meta: Block 元数据
            
        Returns:
            bytes: 编码后的 wire format
            
        Raises:
            ValueError: 输入 shape 不匹配
            TypeError: 输入类型错误
        """
        pass
    
    @abstractmethod
    def decode_kv(self, wire_bytes: bytes, block_meta: BlockMeta) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 wire format 解码回 K, V
        
        Args:
            wire_bytes: 编码后的 bytes
            block_meta: Block 元数据
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (K, V) tensors
        """
        pass
    
    @abstractmethod
    def attention(
        self, 
        Q: np.ndarray, 
        K_wire: bytes, 
        V_wire: bytes, 
        block_metas: List[BlockMeta]
    ) -> np.ndarray:
        """
        使用 wire format K, V 运行 attention
        
        Args:
            Q: Query tensor, shape [batch, num_heads, seq_len, head_dim] 或 [batch, num_heads, head_dim]
            K_wire: 编码后的 K bytes
            V_wire: 编码后的 V bytes
            block_metas: 相关的 block 元数据列表
            
        Returns:
            np.ndarray: Attention 输出, shape 与 Q 一致 (除去 seq_len 维度)
        """
        pass
    
    @abstractmethod
    def name(self) -> str:
        """
        返回 backend 名称
        
        Returns:
            str: 'flash_attn2' | 'vllm' | 'hpc_ops' | 'triton'
        """
        pass
    
    @abstractmethod
    def hardware_required(self) -> str:
        """
        返回最低硬件要求
        
        Returns:
            str: 'SM80' (A100) | 'SM90' (H100) | 'CPU' | 'ANY'
        """
        pass
    
    @abstractmethod
    def supported_dtypes(self) -> List[str]:
        """
        返回支持的 dtype 列表
        
        Returns:
            List[str]: e.g., ['float16', 'bfloat16', 'fp8']
        """
        pass
    
    def verify_interface(self) -> Dict[str, bool]:
        """
        验证 backend 接口完整性
        
        Returns:
            Dict[str, bool]: 各方法的存在性检查结果
        """
        methods = ['encode_kv', 'decode_kv', 'attention', 'name', 'hardware_required', 'supported_dtypes']
        return {m: hasattr(self, m) and callable(getattr(self, m)) for m in methods}
    
    def encode_decode_test(self, K: np.ndarray, V: np.ndarray, block_meta: BlockMeta) -> float:
        """
        测试 encode/decode 互逆性
        
        Args:
            K, V: 输入 tensors
            block_meta: Block 元数据
            
        Returns:
            float: 最大绝对误差
        """
        wire = self.encode_kv(K, V, block_meta)
        K_dec, V_dec = self.decode_kv(wire, block_meta)
        
        k_err = np.max(np.abs(K - K_dec))
        v_err = np.max(np.abs(V - V_dec))
        return max(k_err, v_err)


# === DynamicCache 兼容层 (transformers 5.x support) ===

def extract_kv_from_cache(
    cache: Union["DynamicCache", Tuple[np.ndarray, np.ndarray], dict],
    layer_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    兼容提取 KV cache。
    
    支持多种 cache 格式:
    - Transformers 5.x: DynamicCache 对象，使用 .to_tuple()
    - Transformers 4.x: DynamicCache 对象，使用 .keys/.values (返回 tensor)
    - Tuple: (keys, values) 直接是 tuple
    - Dict: {"keys": ndarray, "values": ndarray} 或 {layer_idx: {"keys": ndarray, "values": ndarray}}
    
    Args:
        cache: KV cache 对象
        layer_idx: 如果 cache 是 dict，取特定层的 KV
        
    Returns:
        Tuple[keys, values]
    """
    # 优先检查是否是 dict（plain Python dict 优先于 hasattr 检查）
    # 因为 dict 有 .keys()/.values() 方法但不应走 Transformers 分支
    if isinstance(cache, dict):
        if layer_idx is not None:
            layer_data = cache[layer_idx]
            return layer_data["keys"], layer_data["values"]
        # 尝试直接取 "keys"/"values"
        if "keys" in cache and "values" in cache:
            keys_data = cache["keys"]
            values_data = cache["values"]
            # 如果是 ndarray 直接返回
            if isinstance(keys_data, np.ndarray):
                return keys_data, values_data
            # 否则可能是 {layer_idx: {"keys": ..., "values": ...}} 格式，取第一层
            first_layer = next(iter(cache.values()))
            return first_layer.get("keys"), first_layer.get("values")
        return None, None
    
    # Transformers 5.x DynamicCache (.to_tuple() 方法)
    if hasattr(cache, 'to_tuple'):
        keys, values = cache.to_tuple()
        return keys, values
    
    # Transformers 4.x DynamicCache (有 .keys/.values 属性，返回 tensor)
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
        # 检查是否是 ndarray（而非 dict_keys 等）
        if isinstance(keys, np.ndarray):
            return keys.shape[-2]
    
    # Dict 格式: {"keys": ndarray, "values": ndarray}
    if isinstance(cache, dict):
        if "keys" in cache and isinstance(cache["keys"], np.ndarray):
            return cache["keys"].shape[-2]
        # 否则尝试 {layer_idx: {"keys": ndarray}} 格式
        try:
            first_layer = next(iter(cache.values()))
            if isinstance(first_layer, dict) and "keys" in first_layer:
                return first_layer["keys"].shape[-2]
        except (StopIteration, AttributeError):
            pass
    
    return 0


class BackendFactory:
    """
    Backend 工厂类
    
    提供统一的 backend 实例化接口
    """
    
    _backends: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, backend_class: type):
        """注册 backend 实现"""
        assert issubclass(backend_class, AccordBackend)
        cls._backends[name] = backend_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> AccordBackend:
        """
        创建 backend 实例
        
        Args:
            name: backend 名称
            **kwargs: 传递给 backend 构造函数的参数
            
        Returns:
            AccordBackend: backend 实例
            
        Raises:
            ValueError: 未知的 backend 名称
        """
        if name not in cls._backends:
            available = list(cls._backends.keys())
            raise ValueError(f"Unknown backend: {name}. Available: {available}")
        return cls._backends[name](**kwargs)
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """列出所有注册的 backend"""
        return list(cls._backends.keys())
    
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
                        import torch
                        if not torch.cuda.is_available():
                            warnings.warn(
                                f"Backend '{name}' requires GPU but CUDA not available. "
                                f"GPU execution will fail.",
                                UserWarning
                            )
                    except (ImportError, NameError):
                        pass  # torch 未导入或不可用
                
                backends[name] = backend
            except Exception as e:
                warnings.warn(f"Failed to create backend '{name}': {e}")
        
        return backends


def benchmark_backends(
    backends: Dict[str, 'AccordBackend'],
    K: np.ndarray, V: np.ndarray, Q: np.ndarray,
    block_meta: 'BlockMeta',
    num_runs: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    基准测试多个 backend
    
    Args:
        backends: backend 名称到实例的映射
        K, V, Q: 测试 tensors
        block_meta: Block 元数据
        num_runs: 运行次数
        
    Returns:
        Dict[str, Dict[str, float]]: 各 backend 的性能指标
    """
    import time
    
    results = {}
    
    for name, backend in backends.items():
        result = {'encode_time': [], 'decode_time': [], 'attn_time': [], 'err': []}
        
        for _ in range(num_runs):
            # Encode
            start = time.perf_counter()
            wire = backend.encode_kv(K, V, block_meta)
            encode_t = time.perf_counter() - start
            result['encode_time'].append(encode_t)
            
            # Decode
            start = time.perf_counter()
            K_dec, V_dec = backend.decode_kv(wire, block_meta)
            decode_t = time.perf_counter() - start
            result['decode_time'].append(decode_t)
            
            # Attention
            start = time.perf_counter()
            attn_out = backend.attention(Q, wire, wire, [block_meta])
            attn_t = time.perf_counter() - start
            result['attn_time'].append(attn_t)
            
            # Error
            err = np.max(np.abs(K - K_dec))
            result['err'].append(err)
        
        results[name] = {
            'encode_mean': np.mean(result['encode_time']),
            'decode_mean': np.mean(result['decode_time']),
            'attn_mean': np.mean(result['attn_time']),
            'err_max': np.max(result['err']),
            'err_mean': np.mean(result['err']),
        }
    
    return results


# 懒加载所有 backend (避免循环导入)
def _get_all_backends():
    """懒加载所有 backend 实现"""
    from .backends.flash_attn import FlashAttention2Backend
    from .backends.vllm import VllmPagedAttentionBackend
    from .backends.hpc_ops import HPCOpsBackend
    from .backends.triton import TritonBackend
    
    BackendFactory.register('flash_attn2', FlashAttention2Backend)
    BackendFactory.register('vllm', VllmPagedAttentionBackend)
    BackendFactory.register('hpc_ops', HPCOpsBackend)
    BackendFactory.register('triton', TritonBackend)


# 注册所有 backend
try:
    _get_all_backends()
except ImportError:
    pass  # 稍后通过 demo 脚本加载
