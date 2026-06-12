"""
Backend Demo 测试脚本
====================

测试 4 个 backend 的接口一致性和端到端 attention 正确性。

运行方式:
    python backend_demo.py

Author: ACCORD-KV Team
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple

# 添加 simulation 目录到 path
import sys
import os
sim_dir = os.path.dirname(__file__)
sys.path.insert(0, sim_dir)

# Import backends via package-relative path for proper module loading
from .backends import (
    FlashAttention2Backend,
    VllmPagedAttentionBackend,
    HPCOpsBackend,
    TritonBackend
)
from .accord_backend import (
    AccordBackend, BlockMeta, BackendFactory, benchmark_backends, _get_all_backends
)

# 注册 backends
_get_all_backends()


def create_toy_tensors(
    num_heads: int = 2,
    num_tokens: int = 64,
    head_dim: int = 128,
    batch: int = 1,
    q_len: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, BlockMeta]:
    """
    创建测试用的 toy tensors
    
    Args:
        num_heads: Number of attention heads
        num_tokens: Number of KV tokens
        head_dim: Head dimension
        batch: Batch size
        q_len: Query sequence length
        
    Returns:
        (K, V, Q, block_meta)
    """
    np.random.seed(42)  # 可重复性
    
    # K, V: [num_heads, num_tokens, head_dim]
    K = np.random.randn(num_heads, num_tokens, head_dim).astype(np.float16)
    V = np.random.randn(num_heads, num_tokens, head_dim).astype(np.float16)
    
    # Q: [batch, num_heads, q_len, head_dim]
    Q = np.random.randn(batch, num_heads, q_len, head_dim).astype(np.float16)
    
    block_meta = BlockMeta(
        block_id=0,
        num_tokens=num_tokens,
        seq_len=num_tokens,
        head_dim=head_dim,
        num_heads=num_heads,
        dtype="float16"
    )
    
    return K, V, Q, block_meta


def test_interface_consistency(backends: Dict[str, AccordBackend]) -> Dict:
    """
    测试接口一致性
    
    检查:
    1. 所有 backend 都能 import
    2. 接口方法存在
    3. 相同的输入产生相同 shape 的输出
    """
    print("=" * 60)
    print("测试 1: 接口一致性检查")
    print("=" * 60)
    
    K, V, Q, block_meta = create_toy_tensors()
    results = {}
    
    for name, backend in backends.items():
        print(f"\n  Testing {name}...")
        
        # 接口验证
        interface_ok = backend.verify_interface()
        all_ok = all(interface_ok.values())
        
        # Encode/Decode 测试
        try:
            wire = backend.encode_kv(K, V, block_meta)
            K_dec, V_dec = backend.decode_kv(wire, block_meta)
            encode_decode_ok = K_dec.shape == K.shape and V_dec.shape == V.shape
        except Exception as e:
            print(f"    [FAIL] Encode/Decode: {e}")
            encode_decode_ok = False
            wire = None
        
        # Attention 测试
        try:
            if wire is not None:
                attn_out = backend.attention(Q, wire, wire, [block_meta])
                attn_ok = attn_out.shape[0] == Q.shape[0] and attn_out.shape[1] == Q.shape[1]
            else:
                attn_ok = False
        except Exception as e:
            print(f"    [FAIL] Attention: {e}")
            attn_ok = False
        
        results[name] = {
            "interface_ok": all(interface_ok.values()),
            "encode_decode_ok": encode_decode_ok,
            "attention_ok": attn_ok,
            "hardware_required": backend.hardware_required(),
            "supported_dtypes": backend.supported_dtypes(),
        }
        
        print(f"    Interface: {'PASS' if all(interface_ok.values()) else 'FAIL'}")
        print(f"    Hardware: {backend.hardware_required()}")
        print(f"    Encode/Decode: {'PASS' if encode_decode_ok else 'FAIL'}")
        print(f"    Attention: {'PASS' if attn_ok else 'FAIL'}")
    
    return results


def test_end_to_end_attention(backends: Dict[str, AccordBackend]) -> Dict:
    """
    测试端到端 attention 正确性
    
    比较不同 backend 的 attention 输出误差
    """
    print("\n" + "=" * 60)
    print("测试 2: 端到端 Attention 正确性")
    print("=" * 60)
    
    K, V, Q, block_meta = create_toy_tensors()
    results = {}
    
    # 使用标准 numpy attention 作为参考
    def numpy_attention(Q, K, V):
        scale = 1.0 / np.sqrt(Q.shape[-1])
        Q_f = Q.astype(np.float32)
        K_f = K.astype(np.float32)
        V_f = V.astype(np.float32)
        
        # Q: [B, H, Q, D], K: [H, N, D], V: [H, N, D]
        outputs = []
        for b in range(Q_f.shape[0]):
            for h in range(Q_f.shape[1]):
                q = Q_f[b, h]  # [Q, D]
                k = K_f[h]     # [N, D]
                v = V_f[h]     # [N, D]
                
                s = (q @ k.T) * scale
                s_exp = np.exp(s - np.max(s, axis=-1, keepdims=True))
                w = s_exp / (np.sum(s_exp, axis=-1, keepdims=True) + 1e-9)
                o = w @ v
                outputs.append(o)
        
        return np.array(outputs).reshape(Q.shape).astype(np.float16)
    
    # 参考输出
    ref_out = numpy_attention(Q, K, V)
    
    print("\n  参考实现: 标准 NumPy Attention")
    
    for name, backend in backends.items():
        print(f"\n  Testing {name}...")
        
        try:
            wire = backend.encode_kv(K, V, block_meta)
            attn_out = backend.attention(Q, wire, wire, [block_meta])
            
            # 计算误差
            err_abs = np.max(np.abs(attn_out.astype(np.float32) - ref_out.astype(np.float32)))
            err_rel = err_abs / (np.max(np.abs(ref_out)) + 1e-9)
            
            results[name] = {
                "max_abs_error": float(err_abs),
                "max_rel_error": float(err_rel),
                "output_shape": list(attn_out.shape),
                "output_mean": float(np.mean(np.abs(attn_out))),
            }
            
            print(f"    Shape: {attn_out.shape}")
            print(f"    Max Abs Error: {err_abs:.2e}")
            print(f"    Max Rel Error: {err_rel:.2e}")
            
            # 判断一致性
            if err_abs < 1e-4:
                print(f"    Status: PASS (< 1e-4)")
            elif err_abs < 1e-2:
                print(f"    Status: WARN (< 1e-2)")
            else:
                print(f"    Status: FAIL (> 1e-2)")
                
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[name] = {"error": str(e)}
    
    return results


def test_encode_decode_roundtrip(backends: Dict[str, AccordBackend]) -> Dict:
    """
    测试 encode/decode 往返精度
    """
    print("\n" + "=" * 60)
    print("测试 3: Encode/Decode 往返精度")
    print("=" * 60)
    
    K, V, Q, block_meta = create_toy_tensors()
    results = {}
    
    for name, backend in backends.items():
        print(f"\n  Testing {name}...")
        
        try:
            err = backend.encode_decode_test(K, V, block_meta)
            results[name] = {"max_error": float(err)}
            
            print(f"    Max Error: {err:.2e}")
            
            if err < 1e-3:
                print(f"    Status: PASS")
            else:
                print(f"    Status: WARN")
                
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[name] = {"error": str(e)}
    
    return results


def test_wire_format_interop(backends: Dict[str, AccordBackend]) -> Dict:
    """
    测试不同 backend 的 wire format 互操作性
    
    验证: 同一个 K, V 编码后的 wire format 结构正确
    """
    print("\n" + "=" * 60)
    print("测试 4: Wire Format 互操作性")
    print("=" * 60)
    
    K, V, Q, block_meta = create_toy_tensors()
    results = {}
    
    wire_formats = {}
    
    for name, backend in backends.items():
        print(f"\n  Encoding with {name}...")
        
        try:
            wire = backend.encode_kv(K, V, block_meta)
            wire_size = len(wire)
            wire_formats[name] = wire
            
            # 解码验证
            K_dec, V_dec = backend.decode_kv(wire, block_meta)
            
            # 验证 magic number 和版本
            magic = wire[:4].hex()
            
            results[name] = {
                "wire_size_bytes": wire_size,
                "magic_number": magic,
                "compression_ratio": wire_size / (K.nbytes + V.nbytes),
            }
            
            print(f"    Wire Size: {wire_size} bytes")
            print(f"    Magic: 0x{magic}")
            print(f"    Compression: {results[name]['compression_ratio']:.2f}x")
            
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[name] = {"error": str(e)}
    
    return results


def run_benchmark(backends: Dict[str, AccordBackend], num_runs: int = 10) -> Dict:
    """
    性能基准测试
    """
    print("\n" + "=" * 60)
    print(f"测试 5: 性能基准测试 (x{num_runs} runs)")
    print("=" * 60)
    
    K, V, Q, block_meta = create_toy_tensors(
        num_heads=4,
        num_tokens=256,
        head_dim=256
    )
    
    results = benchmark_backends(backends, K, V, Q, block_meta, num_runs)
    
    print("\n  性能对比表:")
    print("-" * 80)
    print(f"  {'Backend':<15} {'Encode (ms)':<15} {'Decode (ms)':<15} {'Attention (ms)':<15}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"  {name:<15} {metrics['encode_mean']*1000:<15.4f} {metrics['decode_mean']*1000:<15.4f} {metrics['attn_mean']*1000:<15.4f}")
    
    print("-" * 80)
    
    return results


def run_toy_test():
    """
    运行 toy test (任务要求的单配置测试)
    """
    print("=" * 60)
    print("ACCORD Backend Toy Test")
    print("=" * 60)
    
    # 创建 toy tensors: K=[1, 64, 128], V=[1, 64, 128], Q=[1, 1, 128]
    K, V, Q, block_meta = create_toy_tensors(
        num_heads=1,
        num_tokens=64,
        head_dim=128,
        batch=1,
        q_len=1
    )
    
    print(f"\nToy Config:")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")
    print(f"  Q shape: {Q.shape}")
    
    # 创建所有 backends
    backends = {
        'flash_attn2': FlashAttention2Backend(),
        'vllm': VllmPagedAttentionBackend(),
        'hpc_ops': HPCOpsBackend(),
        'triton': TritonBackend(),
    }
    
    # 测试每个 backend
    results = {}
    attn_outputs = {}
    
    for name, backend in backends.items():
        print(f"\n{'='*40}")
        print(f"Backend: {name}")
        print(f"{'='*40}")
        
        try:
            # 1. Encode
            wire = backend.encode_kv(K, V, block_meta)
            print(f"  [PASS] encode_kv: wire_size={len(wire)} bytes")
            
            # 2. Decode
            K_dec, V_dec = backend.decode_kv(wire, block_meta)
            print(f"  [PASS] decode_kv: K={K_dec.shape}, V={V_dec.shape}")
            
            # 3. Attention
            attn_out = backend.attention(Q, wire, wire, [block_meta])
            print(f"  [PASS] attention: output={attn_out.shape}")
            
            # 4. 计算误差
            err = np.max(np.abs(K - K_dec))
            results[name] = {
                "status": "PASS",
                "encode_decode_err": float(err),
                "attn_shape": list(attn_out.shape),
            }
            attn_outputs[name] = attn_out
            print(f"  [INFO] encode_decode_err={err:.2e}")
            
        except Exception as e:
            print(f"  [FAIL] {e}")
            results[name] = {"status": "FAIL", "error": str(e)}
    
    # 验证所有 backend 的 attn 输出 shape 一致
    print(f"\n{'='*40}")
    print("Interface Consistency Check")
    print(f"{'='*40}")
    
    shapes = [attn_outputs[name].shape for name in attn_outputs]
    all_same = all(s == shapes[0] for s in shapes)
    
    print(f"  All attention outputs same shape: {all_same}")
    for name, out in attn_outputs.items():
        print(f"    {name}: {out.shape}")
    
    # 计算 attention 输出之间的一致性
    if len(attn_outputs) >= 2:
        names = list(attn_outputs.keys())
        print(f"\n  Attention output consistency:")
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                err = np.max(np.abs(
                    attn_outputs[names[i]].astype(np.float32) - 
                    attn_outputs[names[j]].astype(np.float32)
                ))
                print(f"    {names[i]} vs {names[j]}: {err:.2e}")
    
    return results, attn_outputs


def main():
    """主函数"""
    print("\n" + "#" * 60)
    print("# ACCORD Backend 抽象层测试套件")
    print("#" * 60)
    
    # 创建所有 backend 实例
    backends = {
        'flash_attn2': FlashAttention2Backend(),
        'vllm': VllmPagedAttentionBackend(),
        'hpc_ops': HPCOpsBackend(),
        'triton': TritonBackend(),
    }
    
    # 1. Toy Test (必须通过)
    print("\n\n" + "=" * 60)
    print("必选项测试: Toy Test (K=[1,64,128], V=[1,64,128], Q=[1,1,128])")
    print("=" * 60)
    toy_results, toy_attn = run_toy_test()
    
    # 汇总结果
    all_pass = all(r.get("status") == "PASS" for r in toy_results.values())
    
    print("\n" + "=" * 60)
    print("Toy Test Summary")
    print("=" * 60)
    for name, r in toy_results.items():
        status = r.get("status", "UNKNOWN")
        hardware = backends[name].hardware_required()
        print(f"  {name:<15} {status:<10} hardware={hardware}")
    
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    
    # 2. 完整测试套件
    print("\n\n" + "=" * 60)
    print("完整测试套件")
    print("=" * 60)
    
    interface_results = test_interface_consistency(backends)
    e2e_results = test_end_to_end_attention(backends)
    roundtrip_results = test_encode_decode_roundtrip(backends)
    wire_results = test_wire_format_interop(backends)
    benchmark_results = run_benchmark(backends, num_runs=5)
    
    # 保存结果
    all_results = {
        "toy_test": toy_results,
        "interface_consistency": interface_results,
        "end_to_end_attention": e2e_results,
        "encode_decode_roundtrip": roundtrip_results,
        "wire_format_interop": wire_results,
        "benchmark": benchmark_results,
    }
    
    # 保存 JSON
    output_dir = "/app/data/所有对话/主对话/_staging/accord-kv/results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "backend_abstraction_data.json")
    
    # 自定义 JSON encoder 处理 numpy 类型
    import functools
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n\n结果已保存到: {output_file}")
    
    return all_results


if __name__ == "__main__":
    results = main()
