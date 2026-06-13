#!/usr/bin/env python3
"""Minimal test of patched_attn vs native attention."""
import sys
sys.path.insert(0, "/root/accord-kv/gpu")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"

print("Loading model...")
m = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", torch_dtype=torch.float16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
m.eval()
print("Loaded!")

texts = ["The attention mechanism."]
enc = tok(texts, return_tensors="pt", truncation=True, max_length=256)
ids = enc.input_ids.to(DEVICE)
seq_len = ids.shape[1]
bs = ids.shape[0]
print(f"seq_len={seq_len}, bs={bs}")

# Native PPL
with torch.no_grad():
    out = m(input_ids=ids, use_cache=False)
    logits = out.logits
    loss = F.cross_entropy(logits[:, :100, :].reshape(-1, logits.size(-1)), ids[:, 1:101].reshape(-1))
    print(f"Native PPL: {torch.exp(loss).item():.4f}")

# Extract KV
k_list, v_list = [], []
def mk_k(i):
    def h(_m, _inp, out):
        k_list.append(out.cpu().half().clone())
    return h
def mk_v(i):
    def h(_m, _inp, out):
        v_list.append(out.cpu().half().clone())
    return h

handles = []
for i, layer in enumerate(m.model.layers):
    handles.append(layer.self_attn.k_proj.register_forward_hook(mk_k(i)))
    handles.append(layer.self_attn.v_proj.register_forward_hook(mk_v(i)))

with torch.no_grad():
    m(input_ids=ids, use_cache=False)
for h in handles: h.remove()
print(f"Extracted {len(k_list)} K layers")

# Build per-layer K, V
head_dim = 128
nk = 8
Ks, Vs = [], []
for li in range(len(k_list)):
    K3 = k_list[li]
    K_per_head = K3.view(bs, seq_len, nk, head_dim).permute(0, 2, 1, 3)
    V_per_head = v_list[li].view(bs, seq_len, nk, head_dim).permute(0, 2, 1, 3)
    Ks.append(K_per_head)
    Vs.append(V_per_head)

# RoPE params
THETA = 10000.0
hd = 128
rotary_dim = hd // 2  # 64
dim_half = rotary_dim // 2  # 32

def patched_attn(attn_self, hidden_states, position_ids=None):
    bs2, seq_q = hidden_states.shape[0], hidden_states.shape[1]
    print(f"  patched_attn: bs={bs2}, seq_q={seq_q}", flush=True)
    
    cK = attn_self._ck.to(hidden_states.device)
    cV = attn_self._cv.to(hidden_states.device)
    
    # Q
    q = attn_self.q_proj(hidden_states)
    num_q_heads = q.shape[-1] // hd
    q = q.view(bs2, seq_q, num_q_heads, hd).transpose(1, 2)
    
    # K expand
    num_groups = num_q_heads // nk
    k_exp = cK.transpose(1, 2).unsqueeze(2).expand(-1, -1, num_groups, -1, -1).transpose(1, 2).reshape(bs2, num_q_heads, seq_q, hd)
    v_exp = cV.transpose(1, 2).unsqueeze(2).expand(-1, -1, num_groups, -1, -1).transpose(1, 2).reshape(bs2, num_q_heads, seq_q, hd)
    
    # RoPE
    if position_ids is None:
        pos = torch.arange(seq_q, device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (THETA ** (2.0 * torch.arange(dim_half, device=q.device, dtype=torch.float32) / hd))
    freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos_emb = freqs.cos().to(q.dtype)
    sin_emb = freqs.sin().to(q.dtype)
    
    q_rot = q.clone()
    q0 = q_rot[..., :dim_half]
    q1 = q_rot[..., dim_half:rotary_dim]
    q_rot[..., :dim_half]           = q0 * cos_emb.unsqueeze(1) - q1 * sin_emb.unsqueeze(1)
    q_rot[..., dim_half:rotary_dim] = q0 * sin_emb.unsqueeze(1) + q1 * cos_emb.unsqueeze(1)
    
    # SDPA
    sc = 1.0 / (hd ** 0.5)
    causal = torch.triu(torch.ones(bs2, 1, seq_q, cK.shape[2], device=hidden_states.device, dtype=torch.bool), diagonal=1)
    out = F.scaled_dot_product_attention(q_rot, k_exp, v_exp, attn_mask=causal, dropout_p=0.0, is_causal=False, scale=sc)
    out = out.transpose(1, 2).contiguous()
    return (attn_self.o_proj(out.view(bs2, seq_q, -1)), None)

# Install patch
print("\nInstalling patch...")
orig_fwds = []
for li, layer in enumerate(m.model.layers):
    attn = layer.self_attn
    orig_fwds.append(attn.forward)
    attn._ck = Ks[li].half().cuda()
    attn._cv = Vs[li].half().cuda()
    
    def make_p(attn_self):
        def p(*args, **kw):
            hs = kw.get('hidden_states')
            if hs is None and len(args) > 0:
                hs = args[0]
            pos = kw.get('position_ids')
            print(f"  layer {li}: hs_shape={hs.shape if hs is not None else None}, pos={pos.shape if pos is not None else None}", flush=True)
            return patched_attn(attn_self, hs, pos)
        return p
    attn.forward = make_p(attn)

print("\nRunning patched forward...")
try:
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=False)
        logits_p = out.logits
        loss_p = F.cross_entropy(logits_p[:, :100, :].reshape(-1, logits_p.size(-1)), ids[:, 1:101].reshape(-1))
        print(f"Patched PPL: {torch.exp(loss_p).item():.4f}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback; traceback.print_exc()

# Restore
for li, layer in enumerate(m.model.layers):
    layer.self_attn.forward = orig_fwds[li]
    for attr in ['_ck', '_cv']:
        if hasattr(layer.self_attn, attr):
            delattr(layer.self_attn, attr)
print("Done")
