#!/usr/bin/env python3
"""GPU 实验: rank=32 prefill seq=512（rank=8 已跑过，跑 rank=32 对比）"""
import sys, os, json, torch, time
sys.path.insert(0, "/root/accord-kv/gpu")

from gpu_model_loader import load_model_and_get_hook
from gpu_svd_compress_fixed import compress_kv_full, decompress_kv, measure_reconstruction_error

REMOTE_RES = "/root/accord-kv/gpu_results"
REMOTE_LOG = "/root/accord-kv/exp.log"
os.makedirs(REMOTE_RES, exist_ok=True)

SEQ_LEN = 512
RANK    = 32

_k_cache, _v_cache, _handles = {}, [], []

def _mk_hook(ptype, name, d):
    def fn(module, input, output):
        if isinstance(output, torch.Tensor):
            d[name] = output.detach().float()
    return fn

def register_hooks(model):
    for h in _handles: h.remove()
    _handles.clear(); _k_cache.clear(); _v_cache.clear()
    for name, module in model.named_modules():
        for pt, d in [('k_proj', _k_cache), ('v_proj', _v_cache)]:
            if pt in name:
                _handles.append(module.register_forward_hook(_mk_hook(pt, name, d)))

def parse_kv(nl, num_kv=8, hd=128):
    k_tensors, v_tensors = [], []
    for li in range(nl):
        kn = next((k for k in _k_cache if f'layers.{li}.' in k and 'k_proj' in k), None)
        if kn:
            kt = _k_cache[kn]
            vt = _v_cache.get(kn.replace('k_proj', 'v_proj'))
        else:
            continue
        if kt is None: continue
        if kt.dim() == 3 and kt.shape[2] == num_kv * hd:
            B, T, HD = kt.shape
            ks = kt.view(B, T, num_kv, hd).permute(0, 2, 1, 3)[0]
            if vt is not None:
                vs = vt.view(B, T, num_kv, hd).permute(0, 2, 1, 3)[0]
        elif kt.dim() == 3 and kt.shape[1] == num_kv * hd:
            B, HD, T = kt.shape
            ks = kt.view(B, num_kv, hd, T).permute(0, 1, 3, 2)[0]
            if vt is not None:
                vs = vt.view(B, num_kv, hd, T).permute(0, 1, 3, 2)[0]
        else:
            continue
        if vt is not None:
            k_tensors.append(ks); v_tensors.append(vs)
    return k_tensors, v_tensors

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(REMOTE_LOG, "a") as f: f.write(line + "\n")

log("=" * 60)
log(f"GPU 实验: rank={RANK} prefill seq={SEQ_LEN}")
log("加载模型...")
model, device, tok = load_model_and_get_hook("mistral")
nl = len(model.model.layers)
log(f"  done. layers={nl}, device={device}")

log(f"注册 hooks...")
register_hooks(model)
log(f"  {len(_handles)} hooks")

log(f"推理 seq_len={SEQ_LEN}...")
torch.manual_seed(0)
input_ids = torch.randint(0, tok.vocab_size, (1, SEQ_LEN), device=device)
with torch.no_grad():
    output = model(input_ids, use_cache=True)
log(f"  logits: {output.logits.shape}")

for h in _handles: h.remove()

k_tensors, v_tensors = parse_kv(nl)
if not k_tensors:
    log("ERROR: KV empty"); sys.exit(1)

k_all = torch.cat(k_tensors, dim=0)
v_all = torch.cat(v_tensors, dim=0)
log(f"  KV: K={k_all.shape} V={v_all.shape}")

log(f"压缩 rank={RANK}...")
comp = compress_kv_full(k_all, v_all, rank=RANK, quantize=True, int4=True)
log(f"  压缩后 keys: {list(comp.keys())}")

log("解压...")
k_r, v_r = decompress_kv(comp, quantize=True, int4=True)

log("测量误差...")
err = measure_reconstruction_error(k_all, v_all, k_r, v_r)
for k, v in err.items():
    log(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

result = {"model": "Mistral-7B-Instruct-v0.3", "seq_len": SEQ_LEN, "rank": RANK, **err}
out_path = f"{REMOTE_RES}/mistral_r{RANK}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
log(f"\n✅ 结果 → {out_path}")
