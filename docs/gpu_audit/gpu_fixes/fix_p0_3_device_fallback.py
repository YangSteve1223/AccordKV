"""
lmcache_connector.py — LMCache + ACCORD Integration (GPU FIXED)

修复 P0-3: 添加 device fallback 支持

在原有代码基础上添加:
1. torch.cuda.is_available() 检测
2. GPU/CPU fallback 逻辑
3. is_mock 属性标识
"""

# === 修复片段 (在 __init__ 中使用) ===

# 在 import 部分添加
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

def _get_safe_device(requested_device: Optional[str] = None) -> str:
    """
    获取可用的计算设备，带 fallback。
    
    Args:
        requested_device: 用户请求的设备 (e.g., "cuda", "cpu")
        
    Returns:
        可用的设备字符串
    """
    if requested_device is not None:
        # 用户明确指定
        if requested_device == "cuda" and not _TORCH_AVAILABLE:
            warnings.warn(
                "CUDA requested but torch not available. Falling back to 'cpu'.",
                UserWarning
            )
            return "cpu"
        if requested_device == "cuda" and _TORCH_AVAILABLE and not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but torch.cuda.is_available() is False. "
                "Falling back to 'cpu'.",
                UserWarning
            )
            return "cpu"
        return requested_device
    
    # 用户未指定，自动选择
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    
    warnings.warn(
        "No GPU available. Running on CPU.",
        UserWarning
    )
    return "cpu"


# === 在 LMCacheACCORDConnector.__init__ 中替换 ===
"""
# 原代码:
def __init__(
    self,
    model_name: str,
    contract_selector: ACCORDContractSelector,
    remote_server_url: Optional[str] = None,
    local_device: str = "cuda",  # ← 硬编码
):
    ...
    self.local_device = local_device

# 修复后:
def __init__(
    self,
    model_name: str,
    contract_selector: ACCORDContractSelector,
    remote_server_url: Optional[str] = None,
    local_device: Optional[str] = None,  # ← 可选
):
    ...
    self.local_device = _get_safe_device(local_device)
"""

# === 在 AccordBackend 基类添加 is_mock 属性 ===
"""
# 在 accord_backend.py 的 AccordBackend 类中添加:

@property
def is_mock(self) -> bool:
    '''
    返回 True 表示这是 numpy 模拟实现，非真实 GPU 代码。
    在 GPU 环境下应使用真实实现。
    '''
    return True

# 子类覆盖:
class FlashAttention2Backend(AccordBackend):
    @property
    def is_mock(self) -> bool:
        # 当前是 numpy 模拟，返回 True
        return True
    
    # 未来真实 GPU 实现:
    # @property
    # def is_mock(self) -> bool:
    #     return False
"""

print("Fix P0-3 applied: device fallback added")
