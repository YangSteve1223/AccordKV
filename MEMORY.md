# ACCORD-KV 项目记忆

## 项目基本信息
- **目标会议**: SOSP/OSDI
- **arXiv**: 2606.08635
- **核心贡献**: 基于注意力契约的KV缓存重组框架

## 核心算法
- compress_kv_full(): 完整KV压缩入口
- decompress_kv(): KV恢复函数
- compress_layerwise(): 分层压缩
- svd_preflight(): SVD预检

## 关键发现
- V-bottleneck: K cumvar=0.94, V cumvar=0.60
- Method D: H2O + Coreset + INT4
- Serial Cascade: 128-255x加速

## GPU配置
- 地址: connect.westc.seetacloud.com:52786
- 显存: >=24GB
- Mistral: attn_implementation=eager

## 常见坑
1. MistralSdpaAttention 无 past_kv 属性
2. K_proj hook 输出 2D
3. V 压缩 rank 需要更大 (>=16)
4. pywt mode=periodization
5. 模型加载必须设置 attn_implementation
6. SVD 数值稳定性
7. GPU 显存清理
8. KV cache 可能为空