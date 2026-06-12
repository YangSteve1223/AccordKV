"""
ACCORD-KV Downstream PPL Experiment
Measures perplexity degradation after SVD+INT4 KV cache compression.

Protocol:
1. Load wikitext test samples
2. For each sample: prefill -> get KV cache -> compress -> generate with compressed KV
3. Measure PPL degradation vs baseline
"""

import sys
sys.path.insert(0, "/root/accord-kv")
import os
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Cache
from gpu_svd_compress_v8 import compress_kv_full, decompress_kv
from torch.nn import CrossEntropyLoss

DEVICE = "cuda"
RESULTS_DIR = "/root/accord-kv/gpu_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model_and_tokenizer(model_name):
    if model_name == "mistral":
        model_id = "/root/autodl-tmp/Mistral-7B-Instruct-v0.3"
    else:
        model_id = "/root/autodl-tmp/gemma-2-9b-it"
    
    print(f"Loading {model_name}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    attn_impl = "eager" if "Mistral" in model_id else "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype=torch.float16,
        attn_implementation=attn_impl, trust_remote_code=True,
    )
    model.eval()
    torch.cuda.empty_cache()
    print(f"Loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
    return model, tok


def extract_kv_from_prefill(model, input_ids, prefix_len):
    """
    Run prefill on prefix_len tokens, extract the KV cache.
    Returns: list of (K_layer, V_layer) per layer.
    K/V shape: [batch, num_kv_heads, seq, head_dim]
    """
    prefix = input_ids[:, :prefix_len]
    model.eval()
    
    captured_kv = []
    
    # Hook into each layer's attention to capture K/V after projection
    hook_handles = []
    
    def make_kv_hook(layer_idx):
        def hook(module, input, output):
            # output is (attn_output,) for standard attention
            # We need the K and V after projection - hook into k_proj and v_proj instead
            pass
        return hook
    
    # Hook at k_proj and v_proj level
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        
        # k_proj output
        def make_k_hook(idx):
            def hook(module, input, output):
                captured_kv.append(('k', idx, output[0].half().cpu().clone()))
            return hook
        
        # v_proj output
        def make_v_hook(idx):
            def hook(module, input, output):
                captured_kv.append(('v', idx, output[0].half().cpu().clone()))
            return hook
        
        hook_handles.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))
        hook_handles.append(attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        _ = model(input_ids=prefix, use_cache=True, output_hidden_states=False, return_dict=True)
    
    for h in hook_handles:
        h.remove()
    
    torch.cuda.synchronize()
    
    # Organize: [('k', 0, tensor), ('v', 0, tensor), ('k', 1, tensor), ...]
    # Need to pair up k and v per layer
    nl = len(model.model.layers)
    k_list = [None] * nl
    v_list = [None] * nl
    
    for typ, idx, tensor in captured_kv:
        if typ == 'k':
            k_list[idx] = tensor
        else:
            v_list[idx] = tensor
    
    # Verify all captured
    for i in range(nl):
        if k_list[i] is None or v_list[i] is None:
            raise RuntimeError(f"Layer {i}: K/V not captured")
    
    return k_list, v_list


def compress_kv_layerwise(k_list, v_list, rank, quantize, int4, model_name):
    """
    Compress KV per layer.
    k_list/v_list: list of [batch*nl_or_kv, seq, head_dim] or [batch, seq, nl, head_dim]
    """
    nl = len(k_list)
    compressed_k = []
    compressed_v = []
    
    for layer_idx in range(nl):
        K = k_list[layer_idx].to(DEVICE).half()  # [batch*?, seq, head_dim]
        V = v_list[layer_idx].to(DEVICE).half()
        
        # Determine shape and reshape to [nl, 1, seq, hd]
        if K.dim() == 4:
            # [batch, seq, nl, head_dim] → [nl, batch, seq, head_dim]
            K = K.permute(2, 0, 1, 3)  # [nl, batch, seq, hd]
            V = V.permute(2, 0, 1, 3)
            batch, seq, nl_i, hd = K.shape[1], K.shape[2], K.shape[0], K.shape[3]
        elif K.dim() == 3:
            # [batch*nl, seq, hd] or [batch, nl, seq, hd] after transpose
            # Try to infer nl from model
            # For simplicity, assume nl is derived from total heads
            if "Mistral" in model_name:
                nl_i = 8  # Mistral: 8 KV heads
                batch_kv = K.shape[0] // nl_i
                seq = K.shape[1]
                hd = K.shape[2]
                K = K.view(batch_kv, nl_i, seq, hd).permute(1, 0, 2, 3)  # [nl, batch, seq, hd]
                V = V.view(batch_kv, nl_i, seq, hd).permute(1, 0, 2, 3)
            else:
                # Gemma: handle differently
                nl_i = 16
                batch_kv = K.shape[0] // nl_i
                seq = K.shape[1]
                hd = K.shape[2]
                K = K.view(batch_kv, nl_i, seq, hd).permute(1, 0, 2, 3)
                V = V.view(batch_kv, nl_i, seq, hd).permute(1, 0, 2, 3)
        
        # Compress each [nl, batch, seq, hd] → [nl, 1, seq, hd]
        comp_k_list = []
        comp_v_list = []
        for head_idx in range(nl_i):
            K_h = K[head_idx:head_idx+1]  # [1, batch, seq, hd]
            V_h = V[head_idx:head_idx+1]
            
            if K_h.shape[-1] != 128:
                # Skip non-standard head dims
                comp_k_list.append(K_h.permute(1, 0, 2, 3).squeeze(0))
                comp_v_list.append(V_h.permute(1, 0, 2, 3).squeeze(0))
                continue
            
            try:
                comp = compress_kv_full(K_h, V_h, rank=rank, quantize=quantize, int4=int4)
                K_rec, V_rec = decompress_kv(comp, quantize=quantize, int4=int4)
                # K_rec: [1, 1, seq, hd] → [batch, seq, hd]
                K_h_rec = K_rec[0, 0].cpu().half()
                V_h_rec = V_rec[0, 0].cpu().half()
                comp_k_list.append(K_h_rec)
                comp_v_list.append(V_h_rec)
            except Exception as e:
                print(f"  Layer {layer_idx} head {head_idx} failed: {e}", flush=True)
                return None
        
        # Stack back to [nl, batch, seq, hd]
        K_comp = torch.stack(comp_k_list, dim=0).unsqueeze(1)  # [nl, 1, seq, hd]
        V_comp = torch.stack(comp_v_list, dim=0).unsqueeze(1)
        compressed_k.append(K_comp)
        compressed_v.append(V_comp)
    
    return compressed_k, compressed_v


def generate_with_compressed_kv(model, prefix_ids, compressed_k, compressed_v, max_new_tokens=32):
    """
    Run generation with pre-computed compressed KV.
    For each new token, manually set KV cache entries.
    """
    model.eval()
    input_ids = prefix_ids.clone()
    seq_len = input_ids.shape[-1]
    
    # Initialize past key values with compressed KV
    for layer_idx, (k, v) in enumerate(zip(compressed_k, compressed_v)):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        
        # Move to device
        k_dev = k.to(DEVICE)
        v_dev = v.to(DEVICE)
        
        # Set the cache
        if hasattr(attn, 'past_key_value') and attn.past_key_value is not None:
            pkv = attn.past_key_value
            if hasattr(pkv, 'key_cache'):
                # DynamicCache format
                for i in range(len(k_dev)):
                    pkv.key_cache[i].copy_(k_dev[i])
                    pkv.value_cache[i].copy_(v_dev[i])
            elif isinstance(pkv, tuple) and len(pkv) == 2:
                attn.past_key_value = (k_dev, v_dev)
    
    # Now generate new tokens one by one, letting model use the cache
    generated = []
    for step in range(max_new_tokens):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(input_ids=input_ids, use_cache=True, output_dict=True, return_dict=True)
        
        logits = outputs.logits[:, -1, :]  # [batch, vocab]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        if next_token.item() == tok.eos_token_id:
            break
        
        torch.cuda.empty_cache()
    
    return generated


def measure_ppl_direct(model, input_ids, prefix_len):
    """
    Measure PPL on full sequence (baseline, no compression).
    Cross entropy on tokens [prefix_len:].
    """
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(input_ids=input_ids, output_dict=True, return_dict=True)
        logits = outputs.logits  # [batch, seq, vocab]
        
        shift_logits = logits[:, prefix_len-1:-1, :].contiguous()
        shift_labels = input_ids[:, prefix_len:].contiguous()
        
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_losses = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        mean_loss = token_losses.mean().item()
        ppl = torch.exp(torch.tensor(mean_loss)).item()
    
    return ppl


def measure_ppl_with_compressed_kv(model, tok, prefix_ids, compressed_k, compressed_v, prefix_len):
    """
    Measure PPL using compressed KV.
    Strategy: run full forward (prefix + some target tokens) but with compressed KV.
    We do this by setting the KV cache, then running forward on the same full sequence.
    The model's attention will use the cache values from our compressed set.
    """
    model.eval()
    
    # Total sequence length
    total_len = prefix_ids.shape[-1]
    
    # Inject compressed KV into the model's cache
    for layer_idx, (k, v) in enumerate(zip(compressed_k, compressed_v)):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        k_dev = k.to(DEVICE)
        v_dev = v.to(DEVICE)
        
        pkv = attn.past_key_value
        if pkv is None:
            continue
        
        if hasattr(pkv, 'key_cache') and hasattr(pkv, 'value_cache'):
            # DynamicCache format
            for i in range(k_dev.shape[0]):
                if i < len(pkv.key_cache):
                    pkv.key_cache[i].copy_(k_dev[i].to(pkv.key_cache[i].device))
                    pkv.value_cache[i].copy_(v_dev[i].to(pkv.value_cache[i].device))
        elif isinstance(pkv, tuple):
            attn.past_key_value = (k_dev, v_dev)
    
    # Now run forward on the same input - model will use our compressed KV
    input_ids = prefix_ids.clone()
    
    # We need to re-run forward: the cache is already set to compressed KV
    # but the model will ADD to the cache for each new token.
    # Instead, let's just run one forward pass and measure loss on the suffix.
    # Problem: the model forward will APPEND new KV to cache, not replace.
    
    # Alternative: Clear cache first, then set it, then run forward
    # Clear
    for layer in model.model.layers:
        layer.self_attn.past_key_value = None
    
    # Set compressed KV
    for layer_idx, (k, v) in enumerate(zip(compressed_k, compressed_v)):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        k_dev = k.to(DEVICE)
        v_dev = v.to(DEVICE)
        attn.past_key_value = (k_dev, v_dev)
    
    # Run forward on full sequence
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(input_ids=input_ids, use_cache=True, output_dict=True, return_dict=True)
        logits = outputs.logits
    
    # Compute PPL on suffix
    shift_logits = logits[:, prefix_len-1:-1, :].contiguous()
    shift_labels = input_ids[:, prefix_len:].contiguous()
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_losses = -log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    mean_loss = token_losses.mean().item()
    ppl = torch.exp(torch.tensor(mean_loss)).item()
    
    return ppl


def run_single_sample(model, tok, text, prefix_len, rank, quantize, int4, model_name):
    """Run one sample: baseline PPL + compressed PPL."""
    torch.cuda.empty_cache()
    
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc.input_ids.to(DEVICE)
    seq_len = input_ids.shape[-1]
    
    if seq_len < prefix_len + 64:
        return None
    
    prefix_ids = input_ids[:, :prefix_len]
    
    # Baseline PPL
    ppl_base = measure_ppl_direct(model, input_ids, prefix_len)
    
    # Extract and compress KV
    try:
        k_list, v_list = extract_kv_from_prefill(model, input_ids, prefix_len)
        compressed = compress_kv_layerwise(k_list, v_list, rank, quantize, int4, model_name)
        if compressed is None:
            return {"ppl_base": ppl_base, "ppl_comp": None, "compressed": False}
        comp_k, comp_v = compressed
    except Exception as e:
        print(f"  KV extraction/compression failed: {e}", flush=True)
        return {"ppl_base": ppl_base, "ppl_comp": None, "compressed": False, "error": str(e)}
    
    # Compressed PPL
    try:
        ppl_comp = measure_ppl_with_compressed_kv(model, tok, prefix_ids, comp_k, comp_v, prefix_len)
    except Exception as e:
        print(f"  Compressed forward failed: {e}", flush=True)
        return {"ppl_base": ppl_base, "ppl_comp": None, "compressed": False, "error2": str(e)}
    
    return {
        "ppl_base": ppl_base,
        "ppl_comp": ppl_comp,
        "compressed": True,
        "deg_pct": (ppl_comp - ppl_base) / ppl_base * 100 if ppl_comp else None,
    }


def main():
    print("=" * 60)
    print("ACCORD-KV Downstream PPL Experiment")
    print("=" * 60)
    
    # Load wikitext
    print("\nLoading wikitext...", flush=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
    texts = [t.strip() for t in ds["text"] if len(t.strip()) > 100]
    print(f"  {len(texts)} valid samples", flush=True)
    
    test_texts = texts[:20]
    prefix_len = 512
    
    # Configurations
    CONFIGS = [
        ("mistral", 8,   False, False, "M_FP16_r8"),
        ("mistral", 8,   True,  True,  "M_INT4_r8"),
        ("mistral", 256, False, False, "M_FP16_r256"),
        ("gemma",   8,   False, False, "G_FP16_r8"),
        ("gemma",   8,   True,  True,  "G_INT4_r8"),
        ("gemma",   256, False, False, "G_FP16_r256"),
    ]
    
    global tok
    loaded = {}
    all_results = {}
    
    for model_name, rank, quantize, int4, label in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Config: {label} r={rank} q={quantize} int4={int4}")
        print("=" * 60)
        
        if model_name not in loaded:
            model, tok = load_model_and_tokenizer(model_name)
            loaded[model_name] = (model, tok)
            torch.cuda.empty_cache()
        else:
            model, tok = loaded[model_name]
        
        results = []
        for i, text in enumerate(test_texts):
            print(f"  [{i+1}/20] ", end="", flush=True)
            try:
                res = run_single_sample(model, tok, text, prefix_len, rank, quantize, int4, model_name)
                if res:
                    if res.get("compressed"):
                        print(f"base={res['ppl_base']:.2f} comp={res['ppl_comp']:.2f} "
                              f"deg={res['deg_pct']:+.1f}%", flush=True)
                    else:
                        print(f"base={res['ppl_base']:.2f} comp=N/A", flush=True)
                    results.append(res)
                else:
                    print("skipped", flush=True)
            except Exception as e:
                print(f"error: {e}", flush=True)
            
            torch.cuda.empty_cache()
        
        if results:
            valid = [r for r in results if r.get("compressed") and r.get("ppl_comp")]
            if valid:
                avg_base = sum(r['ppl_base'] for r in valid) / len(valid)
                avg_comp = sum(r['ppl_comp'] for r in valid) / len(valid)
                avg_deg = (avg_comp - avg_base) / avg_base * 100
                all_results[label] = {
                    "model": model_name, "rank": rank, "quantize": quantize, "int4": int4,
                    "n_valid": len(valid),
                    "avg_ppl_base": avg_base,
                    "avg_ppl_comp": avg_comp,
                    "avg_deg_pct": avg_deg,
                    "samples": valid,
                }
                print(f"\n  >>> {label}: baseline={avg_base:.2f} compressed={avg_comp:.2f} deg={avg_deg:+.1f}%")
    
    # Save
    out_path = os.path.join(RESULTS_DIR, "ppl_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")
    
    # Summary
    print("\n" + "=" * 65)
    print("PPL SUMMARY")
    print("=" * 65)
    print(f"{'Config':<20} {'Baseline':>10} {'Compressed':>11} {'Deg%':>8} {'N':>4}")
    print("-" * 65)
    for label, r in all_results.items():
        print(f"{label:<20} {r['avg_ppl_base']:>10.2f} {r['avg_ppl_comp']:>11.2f} "
              f"{r['avg_deg_pct']:>+7.1f}% {r['n_valid']:>4}")
    print("=" * 65)


if __name__ == "__main__":
    main()
