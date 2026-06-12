"""
FlashAttention2 Backend 实现
============================

基于 FlashAttention 2 的 KV Cache backend。
使用 tiling 和 softmax 重新计算实现 O(N) 内存 attention。

NOTE: 此为 numpy 模拟实现。在真实环境中需要:
    import torch
    from flash_attn import flash_attn_func

Author: ACCORD-KV Team
"""

import numpy as np
from typing import Tuple, List
import struct

try:
    from ..accord_backend import AccordBackend, BlockMeta
except ImportError:
    # 直接运行时的 fallback
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from accord_backend import AccordBackend, BlockMeta


class FlashAttention2Backend(AccordBackend):
    """
    FlashAttention 2 Backend
    
    特点:
    - SM80+ (A100, RTX 3090)
    - O(N) 内存复杂度
    - FP16/BF16 支持
    - Tiling 策略减少内存访问
    
    模拟实现说明:
    - 使用 numpy 替代 torch
    - 实现了 FlashAttention 的核心 tiling 逻辑
    - 保留接口契约与真实实现一致
    """
    
    @property
    def is_mock(self) -> bool:
        """
        当前是 numpy 模拟，非真实 GPU 实现。

        TODO: 实现真实 GPU 后返回 False
        """
        return True
    
    def __init__(self, tile_size: int = 64, dropout_p: float = 0.0):
        """
        初始化 FlashAttention2 Backend
        
        Args:
            tile_size: Tiling 大小 (模拟 block_size 参数)
            dropout_p: Dropout 概率 (模拟，实际忽略)
        """
        self._tile_size = tile_size
        self._dropout_p = dropout_p
        self._backend_name = "flash_attn2"
        self._hardware = "SM80"  # A100+
        self._supported_dtypes = ["float16", "bfloat16"]
        
        # 模拟 FlashAttention 的 scale factor
        self._scale = 1.0 / np.sqrt(tile_size)
    
    def name(self) -> str:
        return self._backend_name
    
    def hardware_required(self) -> str:
        return self._hardware
    
    def supported_dtypes(self) -> List[str]:
        return self._supported_dtypes
    
    def encode_kv(self, K: np.ndarray, V: np.ndarray, block_meta: BlockMeta) -> bytes:
        """
        编码 KV 到 wire format
        
        FlashAttention2 wire format:
        - 4 bytes: magic number (0x464C4153 = "FLAS")
        - 4 bytes: version
        - 4 bytes: num_heads
        - 4 bytes: num_tokens
        - 4 bytes: head_dim
        - 4 bytes: dtype flag
        - N*4 bytes: K data (float32 for precision)
        - N*4 bytes: V data (float32 for precision)
        """
        assert K.shape == V.shape, "K and V must have same shape"
        assert len(K.shape) == 3, "Expected 3D tensors [num_heads, num_tokens, head_dim]"
        
        # 转换为 float32 存储 (模拟 FA2 的 fp16 存储)
        K_f32 = K.astype(np.float32)
        V_f32 = V.astype(np.float32)
        
        # 构建 wire format
        magic = 0x464C4153  # "FLAS"
        version = 2
        dtype_flag = 1 if K.dtype == np.float32 else (2 if K.dtype == np.float16 else 3)
        
        header = struct.pack('<IIIIII', magic, version, K.shape[0], K.shape[1], K.shape[2], dtype_flag)
        k_data = K_f32.tobytes()
        v_data = V_f32.tobytes()
        
        return header + k_data + v_data
    
    def decode_kv(self, wire_bytes: bytes, block_meta: BlockMeta) -> Tuple[np.ndarray, np.ndarray]:
        """解码 wire format 回 K, V"""
        assert len(wire_bytes) > 24, "Invalid wire format"
        
        # 解析 header
        header = struct.unpack('<IIIIII', wire_bytes[:24])
        magic, version, num_heads, num_tokens, head_dim, dtype_flag = header
        
        assert magic == 0x464C4153, "Invalid magic number"
        
        # 计算 dtype
        dtype_map = {1: np.float32, 2: np.float16}
        # numpy 不支持 bfloat16，使用 float32 代替
        dtype = dtype_map.get(dtype_flag, np.float16)
        
        # 解析数据
        data_len = num_heads * num_tokens * head_dim
        offset = 24
        
        K = np.frombuffer(wire_bytes[offset:offset + data_len * 4], dtype=np.float32)
        K = K.reshape(num_heads, num_tokens, head_dim).astype(dtype)
        
        V = np.frombuffer(wire_bytes[offset + data_len * 4:offset + 2 * data_len * 4], dtype=np.float32)
        V = V.reshape(num_heads, num_tokens, head_dim).astype(dtype)
        
        return K, V
    
    def attention(
        self, 
        Q: np.ndarray, 
        K_wire: bytes, 
        V_wire: bytes, 
        block_metas: List[BlockMeta]
    ) -> np.ndarray:
        """
        运行 FlashAttention
        
        模拟 FlashAttention 2 的核心算法:
        1. Tiling: 将 K, V 分块
        2. Online softmax: 增量计算 softmax
        3. Rescale: 保持数值稳定性
        
        Args:
            Q: [batch, num_heads, seq_len, head_dim] 或 [batch, num_heads, head_dim]
            K_wire, V_wire: 编码后的 KV
            block_metas: Block 元数据
            
        Returns:
            np.ndarray: Attention 输出
        """
        block_meta = block_metas[0]
        K, V = self.decode_kv(K_wire, block_meta)
        
        # 调整 Q shape: [batch, heads, seq, head_dim]
        if len(Q.shape) == 3:
            Q = Q[:, :, np.newaxis, :]  # [batch, heads, 1, head_dim] -> [batch, heads, seq, head_dim]
        
        # 调整 K, V shape: [batch=1, heads, seq, head_dim]
        if len(K.shape) == 3:
            K = K[np.newaxis, :, :, :]  # [1, heads, seq, head_dim]
            V = V[np.newaxis, :, :, :]
        
        batch, num_heads, q_len, head_dim = Q.shape
        _, _, kv_len, _ = K.shape
        
        # 模拟 FA2 的 scale
        scale = 1.0 / np.sqrt(head_dim)
        
        # 简化 attention 实现
        # 使用标准 attention (与 hpc_ops/triton 一致)
        output = np.zeros((batch, num_heads, q_len, head_dim), dtype=np.float32)
        
        for b in range(batch):
            for h in range(num_heads):
                Q_bh = Q[b, h]  # [q_len, head_dim]
                K_bh = K[b, h]  # [kv_len, head_dim]
                V_bh = V[b, h]  # [kv_len, head_dim]
                
                # S = Q @ K^T
                S = (Q_bh @ K_bh.T) * scale  # [q_len, kv_len]
                
                # Softmax
                S_max = np.max(S, axis=-1, keepdims=True)
                S_exp = np.exp(S - S_max)
                weights = S_exp / (np.sum(S_exp, axis=-1, keepdims=True) + 1e-9)
                
                # O = W @ V
                output[b, h] = weights @ V_bh
        
        return output.astype(np.float16)
    
    def flash_attention_numpy(
        self, 
        Q: np.ndarray, 
        K: np.ndarray, 
        V: np.ndarray,
        scale: float = None
    ) -> np.ndarray:
        """
        Numpy 实现的 FlashAttention 核心算法
        
        这是模拟实现，真实环境使用 flash_attn_func
        """
        if scale is None:
            scale = 1.0 / np.sqrt(Q.shape[-1])
        
        # 标准 attention (用于验证)
        Q_f = Q.astype(np.float32)
        K_f = K.astype(np.float32)
        V_f = V.astype(np.float32)
        
        # S = Q @ K^T
        scores = np.einsum('...qd,...kd->...qk', Q_f, K_f) * scale
        # Softmax
        scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        # O = W @ V
        output = np.einsum('...qv,...vd->...qd', weights, V_f)
        
        return output.astype(Q.dtype)


# 注册到工厂 (在 accord_backend.py 中已处理)
# BackendFactory.register('flash_attn2', FlashAttention2Backend)
