#!/usr/bin/env python3
"""
ACCORD-KV 综合实验 v5 — 覆盖论文所需的全部实机实验

实验分类：
  A  压缩质量：A1=FP16 vs INT4, A2=Rank sweep, A3=Groupsize消融, A4=Per-head adaptive, A5=Seq sweep
  B  内存效率：B1=压缩比 vs rank
  D  分析：D1=Per-head奇异值衰减
"""
import sys, os, json, torch, torch.nn.functional as F, time, gc, math
sys.path.insert(0, "/root/accord-kv/gpu")

from gpu_model_loader import load_model_and_get_hook
from gpu_svd_compress_v7 import compress_kv_full, decompress_kv, measure_reconstruction_error

REMOTE_RES = "/root/accord-kv/gpu_results"
REMOTE_LOG = "/root/accord-kv/exp_v5.log"
os.makedirs(REMOTE_RES, exist_ok=True)

# ── 实验配置 ─────────────────────────────────────────────────────────
CONFIGS = []

for model, nl, nk, hd in [("mistral", 32, 8, 128), ("gemma", 42, 8, 256)]:
    prefix = model[0].upper()   # M or G

    # A1: FP16 vs INT4 (seq=512,2048; rank=8,32)
    for seq in [512, 2048]:
        for rank in [8, 32]:
            CONFIGS.append({
                "model": model, "nl": nl, "nk": nk, "hd": hd,
                "seq": seq, "rank": rank, "quant": "both",
                "name": f"{prefix}_A1_r{rank}_s{seq}"
            })

    # A2: Rank sweep (seq=512, INT4, gs=128)
    for rank in [4, 16, 64, 128, 256]:
        CONFIGS.append({
            "model": model, "nl": nl, "nk": nk, "hd": hd,
            "seq": 512, "rank": rank, "quant": "int4",
            "name": f"{prefix}_A2_rank{rank}"
        })

    # A3: Groupsize sweep (seq=512, rank=8, INT4)
    for gs in [32, 64, 256]:
        CONFIGS.append({
            "model": model, "nl": nl, "nk": nk, "hd": hd,
            "seq": 512, "rank": 8, "quant": "int4",
            "groupsize": gs,
            "name": f"{prefix}_A3_gs{gs}"
        })

    # A4: Adaptive rank (seq=512, var_thresh=0.95)
    CONFIGS.append({
        "model": model, "nl": nl, "nk": nk, "hd": hd,
        "seq": 512, "rank": 32, "quant": "adaptive",
        "var_thresh": 0.95,
        "name": f"{prefix}_A4_adaptive"
    })

    # A5: Sequence length sweep (rank=8, INT4)
    for seq in [64, 128, 256, 1024, 4096]:
        CONFIGS.append({
            "model": model, "nl": nl, "nk": nk, "hd": hd,
            "seq": seq, "rank": 8, "quant": "int4",
            "name": f"{prefix}_A5_s{seq}"
        })

    # B1: Memory compression (rank=8,32,128; seq=512)
    for rank in [8, 32, 128]:
        CONFIGS.append({
            "model": model, "nl": nl, "nk": nk, "hd": hd,
            "seq": 512, "rank": rank, "quant": "int4",
            "measure_mem": True,
            "name": f"{prefix}_B1_mem_r{rank}"
        })

    # D1: Per-head SV analysis (FP16, rank=256)
    CONFIGS.append({
        "model": model, "nl": nl, "nk": nk, "hd": hd,
        "seq": 512, "rank": 256, "quant": "fp16",
        "sv_analysis": True,
        "name": f"{prefix}_D1_sv_analysis"
    })

# 去重
seen = set(); unique = []
for c in CONFIGS:
    key = (c["model"], c["seq"], c["rank"], c.get("quant"), c.get("groupsize"), c.get("sv_analysis"))
    if key not in seen:
        seen.add(key); unique.append(c)
CONFIGS = unique
print(f"[v5] Total configs: {len(CONFIGS)}")
for c in CONFIGS:
    print(f"  {c['name']}")

# ── 工具函数 ─────────────────────────────────────────────────────────
_hook_k_buf, _hook_v_buf, _hook_handles = {}, {}, []

def _mk_hook(key_name, storage):
    def fn(m, i, o):
        if isinstance(o, torch.Tensor): storage[key_name] = o.detach().float()
    return fn

def register_hooks(model):
    global _hook_k_buf, _hook_v_buf, _hook_handles
    for h in _hook_handles: h.remove()
    _hook_handles.clear(); _hook_k_buf.clear(); _hook_v_buf.clear()
    for n, m in model.named_modules():
        for pt, store in [("k_proj", _hook_k_buf), ("v_proj", _hook_v_buf)]:
            if pt in n:
                _hook_handles.append(m.register_forward_hook(_mk_hook(n, store)))

def unregister_hooks():
    for h in _hook_handles: h.remove(); _hook_handles.clear()

def detect_format(nl, nk, hd):
    sample = next(iter(_hook_k_buf.values()), None)
    if sample is None: return nk, hd, None
    s = sample.shape
    if len(s) != 3: return nk, hd, None
    for pnk, phd in [(nk,hd),(8,256),(16,256),(32,128),(16,128)]:
        if s[2] == pnk*phd: return pnk, phd, "BTHD"
        if s[1] == pnk*phd: return pnk, phd, "BHDT"
    return nk, hd, "BTHD"

def parse_kv(nl, nk_a, hd_a, fmt):
    kt_list, vt_list = [], []
    for li in range(nl):
        kn = next((k for k in _hook_k_buf
                   if ".layers.{}.".format(li) in k and "k_proj" in k), None)
        if not kn: continue
        kt = _hook_k_buf.get(kn)
        vt = _hook_v_buf.get(kn.replace("k_proj", "v_proj"))
        if kt is None or vt is None: continue
        if fmt == "BTHD":
            B, T, HD = kt.shape
            ks = kt.view(B, T, nk_a, hd_a).permute(0, 2, 1, 3)[0]
            vs = vt.view(B, T, nk_a, hd_a).permute(0, 2, 1, 3)[0]
        else:
            B, HD, T = kt.shape
            ks = kt.view(B, nk_a, hd_a, T).permute(0, 1, 3, 2)[0]
            vs = vt.view(B, nk_a, hd_a, T).permute(0, 1, 3, 2)[0]
        kt_list.append(ks); vt_list.append(vs)
    return kt_list, vt_list

def get_all_svs(kv4d):
    """返回每个 (layer, head) 的奇异值向量"""
    nl, nk, T, hd = kv4d.shape
    svs = []
    for l in range(nl):
        for k in range(nk):
            chunk = kv4d[l, k].float()
            try:
                _, St, _ = torch.linalg.svd(chunk, full_matrices=False)
            except:
                _, St, _ = torch.svd(chunk)
            svs.append(St.cpu())
    return svs  # list of [min(T,hd)]

def adaptive_rank_compress(kv4d, base_rank, var_thresh=0.95):
    """
    Per-head adaptive rank SVD:
    - 先算每个 head 的累积方差，确定 adaptive rank
    - 用 max_rank 补齐后统一量化
    返回 (compressed_dict, per_head_ranks)
    """
    nl, nk, T, hd = kv4d.shape
    max_rank = base_rank

    def build_svd_dict(kv4d_t):
        all_Ut, all_St, all_Vht = [], [], []
        for l in range(nl):
            for k in range(nk):
                chunk = kv4d_t[l, k].float()
                try:
                    Ut, St, Vht = torch.linalg.svd(chunk, full_matrices=False)
                except:
                    Ut, St, Vht = torch.svd(chunk)
                total_var = (St ** 2).sum()
                cumvar = 0.0
                ar = max_rank
                for i in range(len(St)):
                    cumvar += St[i] ** 2
                    if cumvar / (total_var + 1e-8) >= var_thresh:
                        ar = i + 1
                        break
                ar = min(ar, max_rank, len(St))
                Ut_p = torch.zeros(T, max_rank)
                St_p = torch.zeros(max_rank)
                Vht_p = torch.zeros(max_rank, hd)
                Ut_p[:, :ar] = Ut[:, :ar]
                St_p[:ar] = St[:ar]
                Vht_p[:ar, :] = Vht[:ar, :]
                all_Ut.append(Ut_p); all_St.append(St_p); all_Vht.append(Vht_p)
        U = torch.stack(all_Ut).reshape(nl, nk, T, max_rank)
        S = torch.stack(all_St).reshape(nl, nk, max_rank)
        Vh = torch.stack(all_Vht).reshape(nl, nk, max_rank, hd)
        return {"U": U, "S": S, "Vh": Vh}

    k_svd = build_svd_dict(kv4d)
    v_svd = build_svd_dict(kv4d)  # 对 V 独立做

    # 量化
    def quant_4d(t):
        groupsize = 128
        orig_shape = tuple(t.shape)
        # 保存原始形状用于解压
        save_shape = orig_shape
        # 支持 3D [H,R,D] 或 4D [nl,nk,rank,hd]
        if len(orig_shape) == 4:
            # 4D → 3D: [nl,nk,rank,hd] → [nl*nk, rank, hd]
            nl_, nk_, r_, hd_ = orig_shape
            t = t.reshape(nl_ * nk_, r_, hd_)
            orig_shape = tuple(t.shape)
        H, R, D = orig_shape[0], orig_shape[1], orig_shape[-1]
        D_pad = (groupsize - D % groupsize) % groupsize
        if D_pad: t = F.pad(t, (0, D_pad))
        ng = t.shape[-1] // groupsize
        t_r = t.reshape(H, R, ng, groupsize)
        scales = t_r.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        q = torch.round(t_r / scales).to(torch.int8)
        return q, scales, D, save_shape

    def dequant_4d(q, scales, D, orig_shape):
        mat = q.float() * scales
        H, R, ng, G = mat.shape
        mat = mat.view(H, R, ng * G)[:, :, :D].reshape(orig_shape)
        return mat

    K_Vh_q, K_sc, K_D, K_sh = quant_4d(k_svd["Vh"].half())
    V_Vh_q, V_sc, V_D, V_sh = quant_4d(v_svd["Vh"].half())

    return {
        "K_U": k_svd["U"].half(), "K_S": k_svd["S"].half(),
        "V_U": v_svd["U"].half(), "V_S": v_svd["S"].half(),
        "K_Vh_q": K_Vh_q, "K_Vh_scales": K_sc, "K_Vh_D": K_D, "K_Vh_orig": K_sh,
        "V_Vh_q": V_Vh_q, "V_Vh_scales": V_sc, "V_Vh_D": V_D, "V_Vh_orig": V_sh,
        "base_rank": base_rank,
    }

def decompress_adaptive(comp):
    def dq(q, sc, D, sh):
        mat = q.float() * sc
        H, R, ng, G = mat.shape
        mat = mat.view(H, R, ng * G)[:, :, :D]
        # 如果原始形状是4D，先reshape成3D再转4D
        if len(sh) == 4:
            nl_, nk_, r_, hd_ = sh
            mat = mat.reshape(nl_ * nk_, r_, hd_)
            mat = mat.reshape(sh)
        else:
            mat = mat.reshape(sh)
        return mat
    K_Vh = dq(comp["K_Vh_q"], comp["K_Vh_scales"], comp["K_Vh_D"], comp["K_Vh_orig"]).float()
    V_Vh = dq(comp["V_Vh_q"], comp["V_Vh_scales"], comp["V_Vh_D"], comp["V_Vh_orig"]).float()
    K = torch.matmul(comp["K_U"].float() * comp["K_S"].float().unsqueeze(2), K_Vh)
    V = torch.matmul(comp["V_U"].float() * comp["V_S"].float().unsqueeze(2), V_Vh)
    return K, V

def memory_bytes(comp):
    total = 0
    for k, v in comp.items():
        if isinstance(v, torch.Tensor):
            total += v.element_size() * v.nelement()
    return total

def cosine_sim(a, b):
    """计算两个 flatten 向量的 cosine similarity"""
    # 确保在同一设备上计算（移到CPU）
    a_f = a.reshape(-1).float().to("cpu")
    b_f = b.reshape(-1).float().to("cpu")
    denom = a_f.norm() * b_f.norm()
    if denom < 1e-8: return 0.0
    return torch.dot(a_f, b_f).item() / denom

# ── 日志 ─────────────────────────────────────────────────────────────
def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(REMOTE_LOG, "a") as f: f.write(line + "\n")

# ── 单次实验 ─────────────────────────────────────────────────────────
def run_single(model, device, tok, cfg):
    nl = cfg["nl"]; nk = cfg["nk"]; hd = cfg["hd"]
    seq = cfg["seq"]; rank = cfg["rank"]
    quant = cfg.get("quant", "int4")

    register_hooks(model)
    torch.manual_seed(42)
    ids = torch.randint(0, tok.vocab_size, (1, seq), device=device)
    with torch.no_grad():
        _ = model(ids, use_cache=True)
    unregister_hooks()

    nk_a, hd_a, fmt = detect_format(nl, nk, hd)
    ktl, vtl = parse_kv(nl, nk_a, hd_a, fmt)
    if not ktl:
        return None, "KV empty"

    K4 = torch.stack(ktl, dim=0).float()
    V4 = torch.stack(vtl, dim=0).float()

    result = {
        "name": cfg["name"],
        "model": cfg["model"],
        "seq": seq,
        "rank": rank,
        "quant": quant,
        "groupsize": cfg.get("groupsize", 128),
    }

    # ── FP16 (A1 baseline)
    if quant in ("both", "fp16"):
        c_fp = compress_kv_full(K4, V4, rank=rank, quantize=False)
        kr, vr = decompress_kv(c_fp, quantize=False)
        e = measure_reconstruction_error(K4, V4, kr, vr)
        result.update({
            "fp16_k_rel": e["k_rel_err"],
            "fp16_v_rel": e["v_rel_err"],
            "fp16_k_cos": cosine_sim(K4, kr),
            "fp16_v_cos": cosine_sim(V4, vr),
        })
        del kr, vr

    # ── INT4
    if quant in ("both", "int4"):
        gs = cfg.get("groupsize", 128)
        c_i4 = compress_kv_full(K4, V4, rank=rank, quantize=True, int4=True)
        kr, vr = decompress_kv(c_i4, quantize=True, int4=True)
        e = measure_reconstruction_error(K4, V4, kr, vr)
        result.update({
            "int4_k_rel": e["k_rel_err"],
            "int4_v_rel": e["v_rel_err"],
            "int4_k_cos": cosine_sim(K4, kr),
            "int4_v_cos": cosine_sim(V4, vr),
        })
        if cfg.get("measure_mem"):
            mem_orig = K4.element_size() * K4.nelement() + V4.element_size() * V4.nelement()
            mem_comp = memory_bytes(c_i4)
            result.update({
                "mem_orig_kb": mem_orig // 1024,
                "mem_comp_kb": mem_comp // 1024,
                "compression_ratio": mem_orig / (mem_comp + 1e-8),
                "mem_saving_pct": 100 * (1 - mem_comp / mem_orig),
            })
        del kr, vr; torch.cuda.empty_cache()

    # ── Adaptive rank
    if quant == "adaptive":
        thresh = cfg.get("var_thresh", 0.95)
        c_ad = adaptive_rank_compress(K4, base_rank=rank, var_thresh=thresh)
        kr, vr = decompress_adaptive(c_ad)
        e = measure_reconstruction_error(K4, V4, kr, vr)
        # 估算平均 rank
        all_ar = []
        for l in range(nl):
            for k in range(nk):
                chunk = K4[l, k].float()
                try:
                    _, St, _ = torch.linalg.svd(chunk, full_matrices=False)
                except:
                    _, St, _ = torch.svd(chunk)
                total_var = (St ** 2).sum()
                cumvar = 0.0
                ar = rank
                for i in range(len(St)):
                    cumvar += St[i] ** 2
                    if cumvar / (total_var + 1e-8) >= thresh:
                        ar = i + 1; break
                ar = min(ar, rank, len(St))
                all_ar.append(ar)
        result.update({
            "adaptive_k_rel": e["k_rel_err"],
            "adaptive_v_rel": e["v_rel_err"],
            "adaptive_avg_rank": sum(all_ar) / len(all_ar),
            "adaptive_max_rank": max(all_ar),
            "adaptive_min_rank": min(all_ar),
        })
        del kr, vr; torch.cuda.empty_cache()

    # ── SV analysis
    if cfg.get("sv_analysis"):
        svs_k = get_all_svs(K4)
        svs_v = get_all_svs(V4)
        # 统计累积方差
        stats_k = []
        for l in range(nl):
            for k in range(nk):
                sv = svs_k[l * nk + k]
                tot = (sv ** 2).sum().item()
                cv8 = min(8, len(sv))
                cv32 = min(32, len(sv))
                cv64 = min(64, len(sv))
                stats_k.append({
                    "layer": l, "head": k,
                    "cumvar_8": (sv[:cv8] ** 2).sum().item() / (tot + 1e-8),
                    "cumvar_32": (sv[:cv32] ** 2).sum().item() / (tot + 1e-8),
                    "cumvar_64": (sv[:cv64] ** 2).sum().item() / (tot + 1e-8),
                    "total_sv": tot,
                    "num_sv": len(sv),
                })
        stats_v = []
        for l in range(nl):
            for k in range(nk):
                sv = svs_v[l * nk + k]
                tot = (sv ** 2).sum().item()
                cv8 = min(8, len(sv))
                cv32 = min(32, len(sv))
                cv64 = min(64, len(sv))
                stats_v.append({
                    "layer": l, "head": k,
                    "cumvar_8": (sv[:cv8] ** 2).sum().item() / (tot + 1e-8),
                    "cumvar_32": (sv[:cv32] ** 2).sum().item() / (tot + 1e-8),
                    "cumvar_64": (sv[:cv64] ** 2).sum().item() / (tot + 1e-8),
                    "total_sv": tot,
                    "num_sv": len(sv),
                })
        n_total = len(stats_k)
        result.update({
            "sv_K": stats_k,
            "sv_V": stats_v,
            "avg_cumvar_k_8": sum(s["cumvar_8"] for s in stats_k) / n_total,
            "avg_cumvar_k_32": sum(s["cumvar_32"] for s in stats_k) / n_total,
            "avg_cumvar_k_64": sum(s["cumvar_64"] for s in stats_k) / n_total,
            "avg_cumvar_v_8": sum(s["cumvar_8"] for s in stats_v) / n_total,
            "avg_cumvar_v_32": sum(s["cumvar_32"] for s in stats_v) / n_total,
            "avg_cumvar_v_64": sum(s["cumvar_64"] for s in stats_v) / n_total,
        })

    return result, None

# ── 主循环 ────────────────────────────────────────────────────────────
log("=" * 60)
log("ACCORD-KV 综合实验 v5 开始")
log(f"总配置数: {len(CONFIGS)}")
log("=" * 60)

all_results = []
by_model = {}
for cfg in CONFIGS:
    by_model.setdefault(cfg["model"], []).append(cfg)

for model_key in ["mistral", "gemma"]:
    cfgs = by_model.get(model_key, [])
    if not cfgs: continue
    log(f"\n{'#'*60}")
    log(f"### Loading {model_key}")
    model, device, tok = load_model_and_get_hook(model_key)
    log(f"  nl={cfgs[0]['nl']} nk={cfgs[0]['nk']} hd={cfgs[0]['hd']}")

    for i, cfg in enumerate(cfgs):
        log(f"\n[{(i+1)}/{len(cfgs)}] {cfg['name']}")
        t0 = time.time()
        res, err = run_single(model, device, tok, cfg)
        elapsed = time.time() - t0
        if err:
            log(f"  ERROR: {err}")
            continue
        res["elapsed_s"] = round(elapsed, 2)
        all_results.append(res)
        # 打印关键行
        if "fp16_k_rel" in res:
            log(f"  FP16: K_rel={res['fp16_k_rel']:.4f}  V_rel={res['fp16_v_rel']:.4f}")
        if "int4_k_rel" in res:
            log(f"  INT4: K_rel={res['int4_k_rel']:.4f}  V_rel={res['int4_v_rel']:.4f}  "
                f"K_cos={res['int4_k_cos']:.4f}  V_cos={res['int4_v_cos']:.4f}")
        if "adaptive_avg_rank" in res:
            log(f"  Adaptive: avg_rank={res['adaptive_avg_rank']:.1f}  "
                f"K_rel={res['adaptive_k_rel']:.4f}  V_rel={res['adaptive_v_rel']:.4f}")
        if "compression_ratio" in res:
            log(f"  Memory: {res['compression_ratio']:.1f}x saving {res['mem_saving_pct']:.1f}%")
        if "avg_cumvar_k_8" in res:
            log(f"  SV: K_cumvar[8]={res['avg_cumvar_k_8']:.4f} "
                f"K_cumvar[32]={res['avg_cumvar_k_32']:.4f} "
                f"K_cumvar[64]={res['avg_cumvar_k_64']:.4f}")
            log(f"      V_cumvar[8]={res['avg_cumvar_v_8']:.4f} "
                f"V_cumvar[32]={res['avg_cumvar_v_32']:.4f} "
                f"V_cumvar[64]={res['avg_cumvar_v_64']:.4f}")

    del model; gc.collect(); torch.cuda.empty_cache()

# ── 保存结果 ─────────────────────────────────────────────────────────
def tensor_to_serializable(obj):
    """递归转换Tensor为Python标量或普通类型"""
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(tensor_to_serializable(v) for v in obj)
    return obj

summary_path = os.path.join(REMOTE_RES, "all_exp_v5_summary.json")
results_clean = tensor_to_serializable(all_results)
with open(summary_path, "w") as f:
    json.dump({"total": len(CONFIGS), "ok": len(results_clean), "results": results_clean}, f, indent=2)
log(f"\n{'='*60}")
log(f"DONE: {len(results_clean)}/{len(CONFIGS)} 成功")
log(f"Summary: {summary_path}")

# ── 打印最终表格 ─────────────────────────────────────────────────────
log("\n" + "="*60)
log("A2: Rank Sweep (INT4, seq=512)")
log("="*60)
for r in all_results:
    n = r.get("name","")
    if "_A2_" in n and "int4_k_rel" in r:
        log(f"  {n}: K_rel={r['int4_k_rel']:.4f}  V_rel={r['int4_v_rel']:.4f}  "
            f"K_cos={r['int4_k_cos']:.4f}")

log("\n" + "="*60)
log("A3: Groupsize Ablation (INT4, rank=8)")
log("="*60)
for r in all_results:
    n = r.get("name","")
    if "_A3_" in n and "int4_k_rel" in r:
        log(f"  {n}: K_rel={r['int4_k_rel']:.4f}  V_rel={r['int4_v_rel']:.4f}")

log("\n" + "="*60)
log("A5: Sequence Length Sweep (INT4, rank=8)")
log("="*60)
for r in all_results:
    n = r.get("name","")
    if "_A5_" in n and "int4_k_rel" in r:
        log(f"  {n}: K_rel={r['int4_k_rel']:.4f}  V_rel={r['int4_v_rel']:.4f}")

log("\n" + "="*60)
log("B1: Memory Compression")
log("="*60)
for r in all_results:
    n = r.get("name","")
    if "_B1_" in n and "compression_ratio" in r:
        log(f"  {n}: ratio={r['compression_ratio']:.1f}x  "
            f"saving={r['mem_saving_pct']:.1f}%  "
            f"K_rel={r['int4_k_rel']:.4f}")

log("\n" + "="*60)
log("D1: Singular Value Analysis")
log("="*60)
for r in all_results:
    n = r.get("name","")
    if "_D1_" in n and "avg_cumvar_k_8" in r:
        log(f"  {n}:")
        log(f"    K: cumvar_8={r['avg_cumvar_k_8']:.4f}  "
            f"cumvar_32={r['avg_cumvar_k_32']:.4f}  "
            f"cumvar_64={r['avg_cumvar_k_64']:.4f}")
        log(f"    V: cumvar_8={r['avg_cumvar_v_8']:.4f}  "
            f"cumvar_32={r['avg_cumvar_v_32']:.4f}  "
            f"cumvar_64={r['avg_cumvar_v_64']:.4f}")
