#!/usr/bin/env python3
"""
ACCORD-KV PPL Experiment v8 - FIXED
Core fix: extract KV from FULL sequence, NOT just plen.
Then measure PPL only within [0, plen-1] where KV is guaranteed complete.
Mistral: attn_implementation="eager" forces MistralAttention with past_key_value.
GPU: /root/accord-kv/
"""
import sys; sys.path.insert(0, "/root/accord-kv")
import os, json, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda"
RESULTS_DIR = "/root/accord-kv/gpu_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# RoPE auto-detect
# ─────────────────────────────────────────────────────────────
def detect_rope_mode(model, model_name):
    from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
    try:
        hidden = torch.randn(1, 8, model.config.hidden_size, device=DEVICE, dtype=torch.float16)
        pos_ids = torch.arange(8, device=DEVICE).unsqueeze(0)
        cos, sin = model.model.rotary_emb(hidden, pos_ids)
        print(f"  [RoPE] cos={cos.shape} sin={sin.shape}", flush=True)
        try:
            q = torch.randn(1, 8, 1, 128, device=DEVICE, dtype=torch.float16)
            k = torch.randn(1, 8, 1, 128, device=DEVICE, dtype=torch.float16)
            apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
            return "direct"
        except Exception as e1:
            print(f"  [RoPE direct failed: {e1}], trying unsqueeze2...", flush=True)
            try:
                q = torch.randn(1, 8, 1, 128, device=DEVICE, dtype=torch.float16)
                k = torch.randn(1, 8, 1, 128, device=DEVICE, dtype=torch.float16)
                cu = cos.unsqueeze(0) if cos.dim() == 2 else cos
                su = sin.unsqueeze(0) if sin.dim() == 2 else sin
                apply_rotary_pos_emb(q, k, cu, su, pos_ids)
                return "unsqueeze2"
            except Exception as e2:
                print(f"  [RoPE unsqueeze2 failed: {e2}], using manual...", flush=True)
                return "manual"
    except Exception as e:
        print(f"  [RoPE detect failed: {e}], defaulting manual", flush=True)
        return "manual"

# ─────────────────────────────────────────────────────────────
# Patched attention: uses pre-extracted compressed KV from FULL seq.
# Key insight (v8 fix): KV covers all seq positions, so attention at
# every position 0..seq-1 sees real (non-zero) K/V.
# PPL loss is computed on tokens [1 .. ppl_end-1] ⊂ [0 .. seq-1].
# ─────────────────────────────────────────────────────────────
def make_patched_attn():
    from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

    def patched_attn(attn_self, **kw):
        # ── positional args (HF ≥4.27 call site):
        #   hidden_states, past_key_value, attention_mask, position_ids, ...
        # attn_self._ck, _cv: (bs, nk, seq_full, hd) — FULL seq KV
        # attn_self._ps: total seq (seq_full)
        # attn_self._re, _rm: rotary_emb module + rope_mode string
        # ─────────────────────────────────────────────────────────────
        hs  = kw.get('hidden_states') or kw.get('hs')
        pos = kw.get('position_ids')

        rotary = attn_self._re
        rmode  = attn_self._rm
        seq_full = attn_self._ps        # actual sequence length of KV
        bs, seq_q = hs.shape[0], hs.shape[1]
        hd = attn_self.head_dim
        sc = 1.0 / (hd ** 0.5)

        # ── Move pre-extracted KV to GPU ──────────────────────────────
        cK = attn_self._ck.to(hs.device)   # (bs, nk, seq_full, hd)
        cV = attn_self._cv.to(hs.device)
        nk = cK.shape[1]
        ks = cK.shape[2]

        # ── If KV covers fewer positions than query, pad with zeros.
        #    Positions beyond ks get zero k/v → masked out by causal.
        #    (Only happens if the input was truncated before KV extract.)
        if ks < seq_q:
            pad = torch.zeros(bs, nk, seq_q - ks, hd, dtype=cK.dtype, device=hs.device)
            cK = torch.cat([cK, pad], dim=2)
            cV = torch.cat([cV, pad], dim=2)
        # cK/cV now: (bs, nk, seq_q, hd)

        # ── Q projection: (bs, seq_q, num_q_heads * hd) → (bs, num_q_heads, seq_q, hd)
        q = attn_self.q_proj(hs)
        num_q_total = q.shape[-1]
        num_q_heads = num_q_total // hd
        q = q.view(bs, seq_q, num_q_heads, hd).transpose(1, 2)

        # ── K/V expand: (bs, nk, seq_q, hd) → (bs, num_q_heads, seq_q, hd)
        #    Expansion: transpose → unsqueeze(group) → expand → reshape
        #    transpose(1,2):  (bs, nk, seq_q, hd) → (bs, seq_q, nk, hd)
        #    unsqueeze(2):    → (bs, seq_q, 1, nk, hd)
        #    expand(num_groups, dim=2): → (bs, seq_q, num_groups, nk, hd)
        #    transpose(1,2): → (bs, num_groups, seq_q, nk, hd)
        #    reshape:         → (bs, num_q_heads, seq_q, hd)
        num_groups = num_q_heads // nk
        k_exp = cK.transpose(1, 2)                     # (bs, seq_q, nk, hd)
        k_exp = k_exp.unsqueeze(2).expand(-1, -1, num_groups, -1, -1) \
                                   .transpose(1, 2)    # (bs, num_groups, seq_q, nk, hd)
        k_exp = k_exp.reshape(bs, num_q_heads, seq_q, hd)
        v_exp = cV.transpose(1, 2)
        v_exp = v_exp.unsqueeze(2).expand(-1, -1, num_groups, -1, -1) \
                                   .transpose(1, 2)
        v_exp = v_exp.reshape(bs, num_q_heads, seq_q, hd)

        # ── RoPE ─────────────────────────────────────────────────────
        if pos is None:
            pos = torch.arange(seq_q, device=hs.device).unsqueeze(0).expand(bs, -1)
        cos, sin = rotary(hs, pos)

        if rmode == "direct":
            qr, kr = apply_rotary_pos_emb(q.transpose(1, 2), k_exp, cos, sin, pos)
            q_rot = qr.transpose(1, 2)
            k_rot = kr
        elif rmode == "unsqueeze2":
            cu = cos.unsqueeze(0) if cos.dim() == 2 else cos
            su = sin.unsqueeze(0) if sin.dim() == 2 else sin
            qr, kr = apply_rotary_pos_emb(q.transpose(1, 2), k_exp, cu, su, pos)
            q_rot = qr.transpose(1, 2)
            k_rot = kr
        else:
            def manual_rope(q_, cos_, sin_):
                if cos_.dim() == 2: cos_ = cos_.unsqueeze(0).unsqueeze(0)
                elif cos_.dim() == 3: cos_ = cos_.unsqueeze(1)
                if sin_.dim() == 2: sin_ = sin_.unsqueeze(0).unsqueeze(0)
                elif sin_.dim() == 3: sin_ = sin_.unsqueeze(1)
                h = hd // 2
                qr = q_[..., :h] * cos_[..., :h] - q_[..., h:] * sin_[..., :h]
                ki = q_[..., :h] * sin_[..., :h] + q_[..., h:] * cos_[..., :h]
                return torch.cat([qr, ki], dim=-1)
            q_rot = manual_rope(q, cos, sin)
            k_rot = manual_rope(k_exp, cos, sin)

        # ── SDPA with explicit causal mask ────────────────────────────
        # causal[i,j]=True → position j is masked (cannot attend to future)
        # SDPA(attn_mask=causal, is_causal=False) reads the tensor
        causal = torch.triu(
            torch.ones(bs, 1, seq_q, seq_q, device=hs.device, dtype=torch.bool),
            diagonal=1)
        out = F.scaled_dot_product_attention(
            q_rot, k_rot, v_exp,
            attn_mask=causal,
            dropout_p=0.0,
            is_causal=False,
            scale=sc)

        out = out.transpose(1, 2).contiguous()
        # MistralAttention.forward returns (attn_output, None) — MUST match
        return (attn_self.o_proj(out.view(bs, seq_q, -1)), None)

    return patched_attn

# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────
PARAGRAPHS = [
    "The attention mechanism in transformer models has revolutionized natural language processing.",
    "Deep learning models require substantial computational resources for training and inference.",
    "Large language models have demonstrated remarkable capabilities across a wide range of tasks.",
    "Key-value caches are critical for efficient inference in autoregressive language models.",
    "Quantization techniques reduce the memory footprint and computational cost of neural networks.",
    "Singular value decomposition provides a mathematically principled approach to low-rank approximation.",
    "Memory bandwidth is often the bottleneck in large model inference.",
    "Attention sinks are tokens that receive disproportionately high attention weight.",
    "Prefill-decode disaggregation separates the prompt processing phase from the token generation phase.",
    "Error correction codes are essential for reliable large-scale systems.",
]

def make_long(tok, min_tokens=600):
    text = ""
    for p in PARAGRAPHS:
        while True:
            cand = text + " " + p if text else p
            ids = tok(cand, add_special_tokens=False)["input_ids"]
            if len(ids) >= min_tokens:
                return cand
            text = cand

def load_model(model_name):
    model_id = ("/root/autodl-tmp/Mistral-7B-Instruct-v0.3"
                if model_name == "mistral"
                else "/root/autodl-tmp/gemma-2-9b-it")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.cuda.empty_cache()
    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        max_memory={0: "12GiB", "cpu": "200GiB"},
        torch_dtype=torch.float16,
        attn_implementation="eager",   # forces eager path → past_key_value works
        trust_remote_code=True)
    m.eval()
    torch.cuda.empty_cache()
    print(f"  Loaded {model_name}. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB",
          flush=True)
    return m, tok

# ─────────────────────────────────────────────────────────────
# KV extraction — now extracts FULL sequence (the key v8 fix)
# Returns list of [nlayers] tensors: each (bs, nk, seq, hd)
# ─────────────────────────────────────────────────────────────
def extract_kv(model, ids, model_name):
    """
    Extract raw KV from model forward pass for the FULL input sequence.
    ids: full input_ids (bs, seq_total)
    Returns (Ks, Vs) where each entry is (bs, nk, seq_total, hd).
    """
    head_dim = model.config.head_dim if hasattr(model.config, 'head_dim') else 128
    nl = len(model.model.layers)
    k_list, v_list = [], []
    fired = [False]

    def mk_hook(i):
        def h(_m, _inp, out):
            # out shape: (batch, seq, n_kv_heads * head_dim)
            _bs, _seq = out.shape[0], out.shape[1]
            _nk = out.shape[2] // head_dim
            k_list.append((i, out.cpu().half().clone(), _nk, _seq))
            fired[0] = True
        return h

    def mv_hook(i):
        def h(_m, _inp, out):
            _bs, _seq = out.shape[0], out.shape[1]
            _nk = out.shape[2] // head_dim
            v_list.append((i, out.cpu().half().clone(), _nk, _seq))
            fired[0] = True
        return h

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(mk_hook(i)))
        handles.append(layer.self_attn.v_proj.register_forward_hook(mv_hook(i)))

    with torch.no_grad():
        model(input_ids=ids, use_cache=False)

    for h in handles:
        h.remove()

    if not fired[0]:
        raise RuntimeError("Hooks never fired — KV extraction failed")
    if len(k_list) != nl:
        raise RuntimeError(f"KV layer mismatch: got {len(k_list)}, expected {nl}")

    seq_actual = k_list[0][3]
    nk_actual  = k_list[0][2]
    print(f"  [extract_kv] layers={nl}, n_kv={nk_actual}, seq={seq_actual}", flush=True)

    Ks, Vs = [], []
    for (i, K3, nk, sq), (_, V3, vk, _) in zip(k_list, v_list):
        Ks.append(K3.view(-1, sq, nk, head_dim).permute(0, 2, 1, 3))
        Vs.append(V3.view(-1, sq, vk, head_dim).permute(0, 2, 1, 3))
    torch.cuda.empty_cache()
    return Ks, Vs

# ─────────────────────────────────────────────────────────────
# SVD compression (per head, per layer)
# ─────────────────────────────────────────────────────────────
def compress_layerwise(Ks, Vs, rank, quantize, int4):
    from gpu_svd_compress_v8 import compress_kv_full, decompress_kv
    comp_K, comp_V = [], []
    for li in range(len(Ks)):
        K, V = Ks[li].to(DEVICE), Vs[li].to(DEVICE)
        kh, vh = [], []
        for hi in range(K.shape[1]):          # iterate over nk heads
            try:
                c = compress_kv_full(
                    K[:, hi:hi+1], V[:, hi:hi+1],
                    rank=rank, quantize=quantize, int4=int4)
                Kr, Vr = decompress_kv(c, quantize=quantize, int4=int4)
                kh.append(Kr[0, 0].cpu().half())
                vh.append(Vr[0, 0].cpu().half())
            except Exception as e:
                print(f"  L{li}H{hi}: {e}", flush=True)
                return None
        comp_K.append(torch.stack(kh, dim=0).unsqueeze(1).transpose(0, 1))
        comp_V.append(torch.stack(vh, dim=0).unsqueeze(1).transpose(0, 1))
    return comp_K, comp_V

# ─────────────────────────────────────────────────────────────
# PPL measurement helpers
# PPL is measured on tokens [1, ppl_end-1]: cross-entropy of predicting token i
# from prefix [0..i-1], for i = 1, 2, ..., ppl_end-1.
# ─────────────────────────────────────────────────────────────
def measure_ppl(model, ids, ppl_end):
    """
    Compute cross-entropy loss on tokens [1, ppl_end-1]:
    predict token i from prefix [0..i-1] for i = 1, 2, ..., ppl_end-1.
    shift_logits = logits[:, 0:ppl_end-1]   ← positions 0..ppl_end-2 (ppl_end-1 terms)
    shift_labels = ids[:, 1:ppl_end]        ← positions 1..ppl_end-1 (ppl_end-1 tokens)
    """
    with torch.no_grad():
        out = model(input_ids=ids)
        logits = out.logits if hasattr(out, 'logits') else out[0]
        # logits: (bs, seq, vocab); shift: predict token i from prefix [0..i-1]
        shift_logits = logits[:, 0:ppl_end - 1].contiguous()
        shift_labels = ids[:, 1:ppl_end].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean')
    return torch.exp(loss).item()

def measure_ppl_comp(model, ids, comp_K, comp_V,
                     ppl_end, rope_mode):
    """
    Compressed forward + PPL on tokens [1, ppl_end).
    The patched attention uses pre-extracted comp_K/comp_V for the FULL seq,
    so every query position sees real K/V → no attention collapse.
    """
    rotary_emb = model.model.rotary_emb
    patch_fn = make_patched_attn()

    orig_fwds = []
    try:
        for li, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            orig_fwds.append(attn.forward)
            attn._ck = comp_K[li]
            attn._cv = comp_V[li]
            attn._re = rotary_emb
            attn._rm = rope_mode
            attn._ps = ids.shape[1]          # pass full seq length

            def make_patched(attn_self):
                def p(**kw):
                    return patch_fn(attn_self, **kw)
                return p
            attn.forward = make_patched(attn)
        with torch.no_grad():
            out = model(input_ids=ids)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            shift_logits = logits[:, 0:ppl_end - 1].contiguous()
            shift_labels = ids[:, 1:ppl_end].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean')
        ppl = torch.exp(loss).item()
    finally:
        for li, layer in enumerate(model.model.layers):
            layer.self_attn.forward = orig_fwds[li]
            for attr in ['_ck', '_cv', '_re', '_rm', '_ps']:
                if hasattr(layer.self_attn, attr):
                    delattr(layer.self_attn, attr)
    torch.cuda.empty_cache()
    return ppl

# ─────────────────────────────────────────────────────────────
# SVD pre-flight validation: check cumvar for given rank
# ─────────────────────────────────────────────────────────────
def svd_preflight(Ks, Vs, rank):
    """Quick SVD cumvar check on layer 0, head 0. Returns (cumvar_K, cumvar_V)."""
    from gpu_svd_compress_v8 import compress_kv_full, decompress_kv
    try:
        K0 = Ks[0][:, 0:1].to(DEVICE)   # (1, 1, seq, hd)
        V0 = Vs[0][:, 0:1].to(DEVICE)
        c = compress_kv_full(K0, V0, rank=rank, quantize=False, int4=False)
        Kr, Vr = decompress_kv(c, quantize=False, int4=False)
        K0f, V0f = K0.float(), V0.float()
        def cumvar(o, r):
            err = (o - r).pow(2).sum() / (o.pow(2).sum() + 1e-10)
            return 1.0 - err.item()
        cv_k = cumvar(K0f, Kr)
        cv_v = cumvar(K0f, Vr)
        return cv_k, cv_v
    except Exception as e:
        return None, None

# ─────────────────────────────────────────────────────────────
# Single sample runner
# ppl_end defines the PPL measurement window.
# MUST satisfy: ppl_end <= seq_total  (KV covers the full seq)
# ─────────────────────────────────────────────────────────────
def run_one(model, tok, text,
            ppl_end,
            rank, quantize, int4, model_name, rope_mode):
    torch.cuda.empty_cache()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    ids = enc.input_ids.to(DEVICE)
    seq_total = ids.shape[1]

    if seq_total < ppl_end + 64:
        return None

    # ── Base PPL (native forward, no compression) ───────────────
    ppl_b = measure_ppl(model, ids, ppl_end)

    # ── Extract KV from FULL sequence ───────────────────────────
    try:
        Ks, Vs = extract_kv(model, ids, model_name)
    except Exception as e:
        return {"base": ppl_b, "comp": None, "err": f"extract: {e}"}

    # ── Compress KV ───────────────────────────────────────────
    try:
        comp = compress_layerwise(Ks, Vs, rank, quantize, int4)
        if comp is None:
            return {"base": ppl_b, "comp": None, "err": "compress returned None"}
        cK, cV = comp
    except Exception as e:
        return {"base": ppl_b, "comp": None, "err": f"compress: {e}"}

    # ── Compressed PPL ─────────────────────────────────────────
    try:
        ppl_c = measure_ppl_comp(
            model, ids, cK, cV,
            ppl_end=ppl_end,
            rope_mode=rope_mode)
    except Exception as e:
        import traceback
        return {
            "base": ppl_b, "comp": None,
            "err2": str(e), "tb": traceback.format_exc()}

    deg = (ppl_c - ppl_b) / ppl_b * 100 if ppl_b > 0 else None
    return {"base": ppl_b, "comp": ppl_c, "deg": deg}

# ─────────────────────────────────────────────────────────────
# Main
# ppl_end=256: measure PPL on tokens 1..256
# (prefix is position 0, first loss token is position 1)
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ACCORD-KV PPL Experiment v8 — FIXED (full-seq KV)")
    print("Key fix: extract KV from ALL seq tokens, measure PPL")
    print("         on tokens 1..PPL_END")
    print("=" * 60)

    # ── Configuration ──────────────────────────────────────────
    # ppl_end: measure PPL on tokens 1..ppl_end (cross-entropy loss)
    # MUST be ≤ seq_total (KV covers the full seq)
    PPL_END   = 256
    N_SAMPLES = 12   # samples per configuration

    CONFIGS = [
        # (model, rank, quantize, int4, label)
        ("mistral", None, False, False, "M_FP16_base"),   # FP16 baseline (no compression)
        ("mistral", 8,   False, False, "M_FP16_r8"),      # FP16 + SVD rank-8
        ("mistral", 8,   True,  True,  "M_INT4_r8"),      # INT4 + SVD rank-8
        ("mistral", 16,  False, False, "M_FP16_r16"),     # FP16 + SVD rank-16
        ("mistral", 32,  False, False, "M_FP16_r32"),     # FP16 + SVD rank-32
    ]

    loaded = {}
    rope_modes = {}
    results = {}

    for mn, rank, qtz, i4, label in CONFIGS:
        print(f"\n{'='*50}\n  {label}\n{'='*50}", flush=True)

        if mn not in loaded:
            m, tok = load_model(mn)
            rope_modes[mn] = detect_rope_mode(m, mn)
            print(f"  RoPE mode: {rope_modes[mn]}", flush=True)
            loaded[mn] = (m, tok)
        else:
            m, tok = loaded[mn]

        texts = [make_long(tok) for _ in range(N_SAMPLES)]

        # ── SVD pre-flight for first text ───────────────────────
        if rank is not None:
            enc_pre = tok(texts[0], return_tensors="pt", truncation=True, max_length=1024)
            ids_pre = enc_pre.input_ids.to(DEVICE)
            try:
                Ks_pre, Vs_pre = extract_kv(m, ids_pre, mn)
                cv_k, cv_v = svd_preflight(Ks_pre, Vs_pre, rank)
                if cv_k is not None:
                    print(f"  [SVD preflight r={rank}] cumvar_K={cv_k:.4f} cumvar_V={cv_v:.4f}",
                          flush=True)
            except Exception as e:
                print(f"  [SVD preflight failed: {e}]", flush=True)

        res_list = []
        for i, txt in enumerate(texts):
            print(f"  [{i+1}/{N_SAMPLES}] ", end="", flush=True)
            r = run_one(
                m, tok, txt,
                ppl_end=PPL_END,
                rank=rank, quantize=qtz, int4=i4,
                model_name=mn, rope_mode=rope_modes[mn])
            if r:
                if r.get("comp") is not None:
                    print(f"base={r['base']:.4f} comp={r['comp']:.4f} "
                          f"deg={r['deg']:+.2f}%", flush=True)
                else:
                    print(f"base={r['base']:.4f} comp=N/A {r.get('err2', r.get('err',''))}",
                          flush=True)
                res_list.append(r)
            else:
                print("skip (too short)", flush=True)
            torch.cuda.empty_cache()

        valid = [x for x in res_list if x.get("comp") is not None]
        if valid:
            avgb = sum(x['base'] for x in valid) / len(valid)
            avgc = sum(x['comp'] for x in valid) / len(valid)
            avgd = (avgc - avgb) / avgb * 100
            results[label] = {
                "model": mn, "rank": rank,
                "quantize": qtz, "int4": i4,
                "ppl_end": PPL_END,
                "rope_mode": rope_modes[mn],
                "n": len(valid),
                "avg_base": avgb, "avg_comp": avgc, "avg_deg": avgd,
                "samples": valid}
            print(f"\n  >>> {label}: base={avgb:.4f} comp={avgc:.4f} deg={avgd:+.2f}%  n={len(valid)}")
        else:
            results[label] = {
                "model": mn, "rank": rank,
                "quantize": qtz, "int4": i4,
                "rope_mode": rope_modes[mn],
                "n": 0, "raw": res_list}
            print(f"\n  >>> {label}: no valid results")

    # ── Save results ───────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "ppl_results_v8.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # ── Summary table ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Config':<20} {'Rank':>5} {'Quant':>6} {'base PPL':>9} "
          f"{'comp PPL':>9} {'degradation':>11} {'n':>3}")
    print("-" * 70)
    for lbl, r in results.items():
        if r.get('avg_base') is not None:
            print(f"{lbl:<20} {str(r.get('rank','—')):>5} "
                  f"{str(r.get('int4','—')):>6} "
                  f"{r['avg_base']:>9.4f} {r['avg_comp']:>9.4f} "
                  f"{r['avg_deg']:>+10.2f}% {r['n']:>3}")
        else:
            print(f"{lbl:<20} {'—':>5} {'—':>6} {'N/A':>9} {'N/A':>9} {'—':>11} {r['n']:>3}")
    print("=" * 70)

if __name__ == "__main__":
    main()
