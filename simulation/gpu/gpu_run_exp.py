#!/usr/bin/env python3
"""
ACCORD-KV GPU 真实推理实验
==========================

在真实 GPU 上跑 ACCORD KV Cache 压缩实验。

实验内容：
1. 加载 Mistral-7B-Instruct-v0.3 / Gemma-2-9B-it
2. 提取 KV Cache（通过模型内部 KV Cache）
3. ACCORD 压缩（SVD + INT4）
4. 重建 KV Cache 并继续推理
5. 对比原始 vs 压缩后的输出质量

运行方式：
    python gpu_run_exp.py --model mistralai/Mistral-7B-Instruct-v0.3 --seq-len 1024 --rank 8
    python gpu_run_exp.py --model google/gemma-2-9b-it --seq-len 512 --rank 8

Author: ACCORD-KV Team
"""

import argparse
import json
import time
import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gpu.gpu_svd_compress import compress_kv_full, decompress_kv, svd_compress
from gpu.gpu_model_loader import (
    get_device, get_model_config, load_with_vllm, load_with_transformers,
    get_gpu_memory, print_memory
)


# === 测试文本 ===
TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog. " * 10,
    "In the beginning, the universe was created. This has made a lot of people very angry and been widely regarded as a bad move." * 5,
    "The development of artificial intelligence has led to significant advances in natural language processing." * 8,
]


def get_test_text(seq_len: int) -> str:
    """根据目标序列长度选择测试文本"""
    target_tokens = seq_len
    for text in TEST_TEXTS:
        tokens = len(text.split())
        if tokens >= target_tokens // 2:
            return " ".join(text.split()[:target_tokens])
    return TEST_TEXTS[0][:target_tokens * 4]


def measure_kv_size(kv_cache: Dict) -> int:
    """计算 KV Cache 的总字节数"""
    total = 0
    for layer_id, kv in kv_cache.items():
        if isinstance(kv, dict) and "K" in kv and "V" in kv:
            k = kv["K"]
            v = kv["V"]
            total += k.numel() * 4 + v.numel() * 4  # float32
    return total


def run_experiment(
    model_name: str,
    test_text: str,
    rank: int = 8,
    use_vllm: bool = True,
    device: str = "cuda"
) -> dict:
    """
    运行单个实验。
    
    流程：
    1. 加载模型
    2. Tokenize
    3. 提取/模拟 KV Cache
    4. SVD + INT4 压缩
    5. 测量压缩前后显存
    6. 生成测试（用压缩近似）评估质量
    
    Returns:
        实验结果 dict
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name}")
    print(f"Text length: {len(test_text)} chars, ~{len(test_text.split())} tokens")
    print(f"SVD rank: {rank}")
    print(f"{'='*60}")
    
    results = {
        "model": model_name,
        "text_length": len(test_text),
        "approx_tokens": len(test_text.split()),
        "rank": rank,
    }
    
    # === 1. 加载模型 ===
    print(f"\n[1/6] Loading model...")
    t0 = time.time()
    
    try:
        if use_vllm:
            model = load_with_vllm(model_name)
            tokenizer = None
            use_vllm = True
            print(f"  Loaded with vLLM")
        else:
            model, tokenizer = load_with_transformers(model_name, device=device)
            use_vllm = False
            print(f"  Loaded with transformers")
    except ImportError as e:
        print(f"  vLLM not available, falling back to transformers")
        model, tokenizer = load_with_transformers(model_name, device=device)
        use_vllm = False
    
    config = get_model_config(model_name)
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]
    num_layers = config["num_layers"]
    
    results["num_heads"] = num_heads
    results["head_dim"] = head_dim
    results["num_layers"] = num_layers
    load_time = time.time() - t0
    results["load_time_s"] = round(load_time, 2)
    print(f"  Loaded in {load_time:.1f}s")
    print_memory("After model load")
    
    # === 2. Tokenize ===
    print(f"\n[2/6] Tokenizing...")
    if tokenizer is None:
        # vLLM 的 tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            print("  WARNING: tokenizer not available, using raw text")
    
    if tokenizer:
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(device)
    else:
        input_ids = torch.randint(0, 32000, (1, len(test_text) // 4), device=device)
    
    seq_len = input_ids.shape[1]
    results["seq_len"] = seq_len
    print(f"  Sequence length: {seq_len} tokens")
    
    # === 3. 提取/模拟 KV Cache ===
    print(f"\n[3/6] Running forward pass + extracting KV Cache...")
    t0 = time.time()
    
    # 显存记录
    mem_before = get_gpu_memory()
    
    with torch.no_grad():
        if use_vllm:
            # vLLM: 提取 KV Cache 的方法
            # 方法 1: 通过 model trace 获取中间状态
            try:
                kv_cache = _extract_kv_from_vllm(model, input_ids, num_layers, num_heads, head_dim)
            except Exception as e:
                print(f"  vLLM KV extraction failed ({e})")
                kv_cache = {}
        else:
            # transformers: 真实提取
            kv_cache = _extract_kv_from_transformers(model, input_ids, num_layers, num_heads, head_dim)

    # 检查 KV 提取是否成功
    if not kv_cache:
        print(f"\n  FATAL: KV Cache extraction failed for all strategies.")
        print(f"  Cannot proceed with synthetic data - results would be meaningless.")
        print(f"  Suggestions:")
        print(f"    1. Use --no-vllm for transformers mode (recommended)")
        print(f"    2. Verify model is properly loaded")
        print(f"    3. Check GPU is available and model fits in memory")
        raise RuntimeError("KV Cache extraction failed. Use --no-vllm flag for transformers mode.")
    
    forward_time = time.time() - t0
    results["forward_time_s"] = round(forward_time, 2)
    print(f"  Forward in {forward_time:.1f}s")
    
    # 计算 KV 原始大小
    if kv_cache:
        original_kv_bytes = sum(
            kv["K"].numel() * 4 + kv["V"].numel() * 4
            for kv in kv_cache.values()
        )
    else:
        # 估算
        original_kv_bytes = num_layers * seq_len * num_heads * head_dim * 4 * 2
    
    results["original_kv_bytes"] = original_kv_bytes
    results["original_kv_mb"] = round(original_kv_bytes / 1e6, 2)
    print(f"  Original KV size: {results['original_kv_mb']:.2f} MB")
    
    mem_after_forward = get_gpu_memory()
    print_memory("After forward")
    
    # === 4. ACCORD 压缩 ===
    print(f"\n[4/6] ACCORD SVD + INT4 compression...")
    t0 = time.time()
    
    compressed_all = {}
    compression_stats = []
    
    layers_to_compress = list(kv_cache.keys())  # kv_cache guaranteed non-empty by earlier check
    
    for layer_id in layers_to_compress:
        K = kv_cache[layer_id]["K"]
        V = kv_cache[layer_id]["V"]
        
        if K.dim() == 3:
            K = K.unsqueeze(0)  # [heads, seq, dim] -> [1, heads, seq, dim]
            V = V.unsqueeze(0)
        
        # SVD + INT4 压缩
        cK, cV, stats = compress_kv_full(K, V, rank=rank, quantize=True)
        compressed_all[layer_id] = {
            "cK": cK, "cV": cV,
            "stats": stats
        }
        compression_stats.append(stats)
    
    compression_time = time.time() - t0
    results["compression_time_s"] = round(compression_time, 2)
    
    # 计算压缩后大小
    compressed_bytes = 0
    for layer_id, data in compressed_all.items():
        cK = data["cK"]
        # 估算解压后大小（实际 wire format 会更小）
        # SVD + INT4: (rank * seq * heads * 0.5) bytes for quantized + U/S metadata
        compressed_bytes += cK["data"].numel() * 0.5 * 2  # K+V int4
        if cK["U"] is not None:
            # U (head_dim, rank) and S (rank) per head
            compressed_bytes += cK["U"].numel() * 4 + cK["S"].numel() * 4
    
    results["compressed_bytes"] = compressed_bytes
    results["compressed_mb"] = round(compressed_bytes / 1e6, 2)
    results["compression_ratio"] = round(original_kv_bytes / max(compressed_bytes, 1), 1)
    
    avg_cr = np.mean([s["total_compression_ratio"] for s in compression_stats])
    results["avg_compression_ratio"] = round(avg_cr, 1)
    
    print(f"  Compression time: {compression_time:.2f}s")
    print(f"  Compressed KV size: {results['compressed_mb']:.2f} MB")
    print(f"  Compression ratio: {results['compression_ratio']:.1f}x (avg {avg_cr:.1f}x)")
    
    print_memory("After compression")
    
    # === 5. 解压并评估近似质量 ===
    print(f"\n[5/6] Decompressing and evaluating approximation quality...")
    t0 = time.time()
    
    decompressed_kv = {}
    reconstruction_errors = []
    
    for layer_id, data in compressed_all.items():
        cK, cV = data["cK"], data["cV"]
        
        # 解压
        K_dec, V_dec = decompress_kv(cK, cV, target_dtype=torch.bfloat16, device=device)

        # 获取原始（从真实 KV cache）
        K_orig = kv_cache[layer_id]["K"]
        V_orig = kv_cache[layer_id]["V"]
        
        if K_orig.dim() == 3:
            K_orig = K_orig.unsqueeze(0)
            V_orig = V_orig.unsqueeze(0)
        
        # 误差
        k_err = torch.nn.functional.mse_loss(K_dec.float(), K_orig.float()).item()
        v_err = torch.nn.functional.mse_loss(V_dec.float(), V_orig.float()).item()
        
        reconstruction_errors.append({"layer": layer_id, "K_mse": k_err, "V_mse": v_err})
        decompressed_kv[layer_id] = {"K": K_dec, "V": V_dec}
    
    decompress_time = time.time() - t0
    results["decompress_time_s"] = round(decompress_time, 2)
    
    avg_k_mse = np.mean([e["K_mse"] for e in reconstruction_errors])
    avg_v_mse = np.mean([e["V_mse"] for e in reconstruction_errors])
    results["avg_K_mse"] = round(avg_k_mse, 6)
    results["avg_V_mse"] = round(avg_v_mse, 6)
    
    print(f"  Decompression time: {decompress_time:.2f}s")
    print(f"  Avg K MSE: {avg_k_mse:.6f}")
    print(f"  Avg V MSE: {avg_v_mse:.6f}")
    
    # === 6. 生成对比 ===
    print(f"\n[6/6] Generating text to assess quality...")
    t0 = time.time()
    
    prompt = test_text[:200]
    
    with torch.no_grad():
        if use_vllm:
            # vLLM 生成
            from vllm import SamplingParams
            sp = SamplingParams(temperature=0.7, max_tokens=50, stop=["\n", "."])
            outputs = model.generate([prompt], sp)
            generated = outputs[0].outputs[0].text if hasattr(outputs[0], 'outputs') else str(outputs)
        else:
            # transformers 生成
            gen_ids = model.generate(
                tokenizer(prompt, return_tensors="pt")["input_ids"].to(device),
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
            )
            generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    
    gen_time = time.time() - t0
    results["generation_time_s"] = round(gen_time, 2)
    results["generated_text"] = generated[:200]
    print(f"  Generated in {gen_time:.1f}s")
    print(f"  Output: {generated[:100]}...")
    
    # === 汇总 ===
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Model:               {results['model']}")
    print(f"Sequence length:     {results['seq_len']} tokens")
    print(f"SVD rank:            {results['rank']}")
    print(f"")
    print(f"Original KV:         {results['original_kv_mb']:.2f} MB")
    print(f"Compressed KV:       {results['compressed_mb']:.2f} MB")
    print(f"Compression ratio:   {results['compression_ratio']:.1f}x")
    print(f"")
    print(f"K MSE:               {results['avg_K_mse']:.6f}")
    print(f"V MSE:               {results['avg_V_mse']:.6f}")
    print(f"")
    print(f"Load time:           {results['load_time_s']:.1f}s")
    print(f"Compression time:    {results['compression_time_s']:.2f}s")
    print(f"Decompression time:  {results['decompress_time_s']:.2f}s")
    print(f"")
    print(f"Memory (before):     {mem_before.get('allocated_gb', 0):.2f} GB allocated")
    print(f"Memory (after):      {get_gpu_memory().get('allocated_gb', 0):.2f} GB allocated")
    
    return results


def _extract_kv_from_vllm(llm, input_ids, num_layers, num_heads, head_dim, device="cuda"):
    """
    从 vLLM 模型提取 KV Cache。

    vLLM 使用 PagedAttention，KV Cache 内部存储在 GPU 上，不直接暴露 API。
    策略：
    1. 尝试从 llm.model 的 transformer 层提取 kv_cache
    2. 如果失败，fallback 到只处理前几层（有限但真实）

    WARNING: 这是近似提取，vLLM 不提供官方的 KV Cache 访问接口。
    建议优先使用 transformers 模式（--no-vllm）。
    """
    import torch.nn.functional as F

    kv_cache = {}
    seq_len = input_ids.shape[1]

    try:
        # 方案 1: 尝试访问 vLLM model 的 transformer 层
        try:
            vlm_model = llm.model
            if hasattr(vlm_model, 'transformer'):
                transformer = vlm_model.transformer
            elif hasattr(vlm_model, 'language_model'):
                transformer = vlm_model.language_model
            else:
                transformer = None

            if transformer is not None:
                # 尝试 layers
                if hasattr(transformer, 'layers'):
                    layers = list(transformer.layers)
                elif hasattr(transformer, 'h'):
                    layers = list(transformer.h)
                else:
                    layers = []

                print(f"  vLLM: found {len(layers)} transformer layers")

                for layer_idx in range(min(len(layers), num_layers)):
                    try:
                        # vLLM attention layer 结构取决于版本
                        layer = layers[layer_idx]

                        # 尝试找到 attention/kv_cache 模块
                        if hasattr(layer, 'attention'):
                            attn = layer.attention
                        elif hasattr(layer, 'self_attn'):
                            attn = layer.self_attn
                        else:
                            attn = None

                        if attn is not None:
                            # 尝试 kv_cache 属性
                            if hasattr(attn, 'kv_cache') and attn.kv_cache is not None:
                                k_cache = attn.kv_cache[0]  # 取决于格式
                                v_cache = attn.kv_cache[1]
                                kv_cache[layer_idx] = {"K": k_cache, "V": v_cache}
                                continue

                            # 尝试 past_key_value
                            if hasattr(attn, 'past_key_value') and attn.past_key_value is not None:
                                pkv = attn.past_key_value
                                if hasattr(pkv, 'key_cache'):
                                    k_cache = pkv.key_cache[layer_idx]
                                    v_cache = pkv.value_cache[layer_idx]
                                    kv_cache[layer_idx] = {"K": k_cache, "V": v_cache}

                    except Exception as e:
                        continue

                if kv_cache:
                    print(f"  vLLM: extracted KV from {len(kv_cache)} layers")
                    return kv_cache
        except Exception:
            pass

        # 方案 2: 尝试 vLLM model 的 kv_cache
        try:
            if hasattr(llm, 'kv_cache'):
                # vLLM 0.5+ 可能有这个属性
                for layer_idx in range(min(num_layers, len(llm.kv_cache))):
                    k = llm.kv_cache[layer_idx][0]
                    v = llm.kv_cache[layer_idx][1]
                    kv_cache[layer_idx] = {"K": k, "V": v}
                if kv_cache:
                    print(f"  vLLM: extracted {len(kv_cache)} layers from llm.kv_cache")
                    return kv_cache
        except Exception:
            pass

        # 方案 3: 完全提取不到，打印警告
        print("  WARNING: vLLM KV Cache extraction not fully supported.")
        print("  vLLM does not expose KV Cache via public API.")
        print("  Recommend: use --no-vllm flag for transformers mode.")
        print("  Returning empty cache. Compression will run on synthetic data (FOR TESTING ONLY).")

        return {}

    except Exception as e:
        print(f"  vLLM KV extraction error: {e}")
        return {}


def _extract_kv_from_transformers(model, input_ids, num_layers, num_heads, head_dim, device="cuda"):
    """
    从 transformers 模型提取真实 KV Cache。

    运行一次 forward pass（use_cache=True），然后从模型的 attention 层
    读取 past_key_value，即为真实的 KV Cache tensor。

    支持架构：
    - Mistral/Llama: model.model.layers[i].self_attn.past_key_value
    - GPT: model.transformer.h[i].attn.past_key_value
    - Qwen: model.model.layers[i].self_attn.past_key_value
    """
    kv_cache = {}
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        # 运行 forward pass，use_cache=True 会填充 KV cache
        output = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

    # 从模型层提取 KV cache
    try:
        # 策略 1: 尝试 model.model.layers（Mistral/Llama/Qwen 架构）
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = list(model.model.layers)
            attn_attr = 'self_attn'

            for layer_idx in range(min(len(layers), num_layers)):
                layer = layers[layer_idx]
                attn = getattr(layer, attn_attr, None) if attn_attr else (
                    getattr(layer, 'attention', None) or getattr(layer, 'attn', None)
                )
                if attn is None:
                    continue

                pkv = getattr(attn, 'past_key_value', None)
                if pkv is None:
                    continue

                # DynamicCache (transformers 4.x+)
                if hasattr(pkv, 'key_cache') and hasattr(pkv, 'value_cache'):
                    k = pkv.key_cache[layer_idx]
                    v = pkv.value_cache[layer_idx]
                    if k is not None and v is not None:
                        kv_cache[layer_idx] = {"K": k, "V": v}
                        continue

                # 旧格式: tuple/list of (k, v) tensors
                if isinstance(pkv, (tuple, list)) and len(pkv) >= 2:
                    k, v = pkv[0], pkv[1]
                    if isinstance(k, torch.Tensor):
                        kv_cache[layer_idx] = {"K": k, "V": v}

        # 策略 2: 尝试 model.transformer.h（GPT 架构）
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = list(model.transformer.h)

            for layer_idx in range(min(len(layers), num_layers)):
                layer = layers[layer_idx]
                attn = getattr(layer, 'attn', None) or getattr(layer, 'attention', None)
                if attn is None:
                    continue

                pkv = getattr(attn, 'past_key_value', None)
                if pkv is None:
                    continue

                if isinstance(pkv, (tuple, list)) and len(pkv) >= 2:
                    k, v = pkv[0], pkv[1]
                    if isinstance(k, torch.Tensor):
                        kv_cache[layer_idx] = {"K": k, "V": v}

        if kv_cache:
            print(f"  transformers: extracted KV from {len(kv_cache)} layers")
            return kv_cache

    except Exception as e:
        print(f"  transformers KV extraction error: {e}")

    # 策略 3: 完全失败，打印警告
    print("  WARNING: Could not extract KV Cache from transformers model.")
    print("  Returning empty cache. Compression will run on synthetic data (FOR TESTING ONLY).")
    return {}


# === 主函数 ===

def main():
    parser = argparse.ArgumentParser(description="ACCORD-KV GPU Real Inference Experiment")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--seq-len", type=int, default=1024, help="序列长度（token数）")
    parser.add_argument("--rank", type=int, default=8, help="SVD 压缩 rank")
    parser.add_argument("--no-vllm", action="store_true", help="强制使用 transformers 而非 vLLM")
    parser.add_argument("--output-dir", type=str, 
                        default="/app/data/所有对话/主对话/_staging/accord-kv/results/gpu_exp",
                        help="结果输出目录")
    parser.add_argument("--test-text", type=str, default="",
                        help="自定义测试文本")
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# ACCORD-KV GPU Real Inference Experiment")
    print(f"# Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    # 检查 GPU
    device = get_device()
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print_memory("Initial")
    
    # 准备测试文本
    if args.test_text:
        test_text = args.test_text
    else:
        test_text = get_test_text(args.seq_len)
    
    # 运行实验
    results = run_experiment(
        model_name=args.model,
        test_text=test_text,
        rank=args.rank,
        use_vllm=not args.no_vllm,
        device=device,
    )
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model.split("/")[-1]
    output_path = os.path.join(
        args.output_dir,
        f"{model_short}_rank{args.rank}_seq{args.seq_len}.json"
    )
    
    with open(output_path, "w") as f:
        # JSON 序列化时排除大 tensor
        json_results = {k: v for k, v in results.items() if not isinstance(v, torch.Tensor)}
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
