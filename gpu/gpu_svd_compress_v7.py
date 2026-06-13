#!/usr/bin/env python3
"""
gpu_svd_compress_fixed.py — 修复版 v7

核心: 回到 v4 原始逻辑（对 [T, hd] 做 SVD），同时支持 4D 输入。
      4D [nl,nk,T,hd]: 对每个 (layer, head) 的 [T, hd] 做 SVD
      3D [H,T,D]:      对每个 layer 的 [T, D] 做 SVD（原有逻辑）
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any

# ──────────────────────────────────────────────
# SVD: 对 [T, D] 做截断 SVD，提取 top-r 个奇异值
# 返回 {U:[T,r], S:[r], Vh:[r,D]}
# ──────────────────────────────────────────────
def _svd_td(tensor_td: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    """
    对 [T, D] 做截断 SVD，返回 {U:[T,r], S:[r], Vh:[r,D]}。
    当 rank > min(T,D) 时，自动 cap 并 pad 到精确 rank（保证 tensor shape 一致）。
    """
    T, D = tensor_td.shape
    try:
        Ut, St, Vht = torch.linalg.svd(tensor_td.float(), full_matrices=False)
    except (RuntimeError, TypeError):
        Ut, St, Vht = torch.svd(tensor_td.float())
    actual = min(rank, T, D)
    # Pad 到精确 rank（供批量 stack）
    U = torch.zeros(T, rank, dtype=Ut.dtype)
    S = torch.zeros(rank, dtype=St.dtype)
    Vh = torch.zeros(rank, D, dtype=Vht.dtype)
    U[:, :actual] = Ut[:, :actual]
    S[:actual] = St[:actual]
    Vh[:actual, :] = Vht[:actual, :]
    return {"U": U, "S": S, "Vh": Vh}


def _svd_single_3d(kv_3d: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    """
    3D [H, T, D]: 对每个 H 做 SVD。
    返回 U:[H,T,r], S:[H,r], Vh:[H,r,D]。
    """
    H, T, D = kv_3d.shape
    U_l, S_l, Vh_l = [], [], []
    for h in range(H):
        chunk = kv_3d[h, :, :]    # [T, D]
        res = _svd_td(chunk, rank)
        U_l.append(res["U"])
        S_l.append(res["S"])
        Vh_l.append(res["Vh"])
    return {
        "U":  torch.stack(U_l),   # [H, T, r]
        "S":  torch.stack(S_l),   # [H, r]
        "Vh": torch.stack(Vh_l),  # [H, r, D]
    }


def _svd_single_4d(kv_4d: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    """
    4D [nl, nk, T, hd]: 对每个 (layer, head) 的 [T, hd] 做 SVD。
    返回 U:[nl,nk,T,r], S:[nl,nk,r], Vh:[nl,nk,r,hd]。
    """
    nl, nk, T, hd = kv_4d.shape
    U_l, S_l, Vh_l = [], [], []
    for l in range(nl):
        for k in range(nk):
            chunk = kv_4d[l, k, :, :]   # [T, hd]
            res = _svd_td(chunk, rank)
            U_l.append(res["U"])
            S_l.append(res["S"])
            Vh_l.append(res["Vh"])
    U_all = torch.stack(U_l).reshape(nl, nk, T, rank)
    S_all = torch.stack(S_l).reshape(nl, nk, rank)
    Vh_all = torch.stack(Vh_l).reshape(nl, nk, rank, hd)
    return {"U": U_all, "S": S_all, "Vh": Vh_all}


def _svd_single_2d(kv_2d: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    """
    2D [H, D]: 对每行做 SVD，返回 {U:[H,r], S:[H,r], Vh:[H,r,D]}。
    用于 prefill: [nl*nk, T] — 每行是一个 head 的 temporal sequence。
    """
    H, D = kv_2d.shape
    U_l, S_l, Vh_l = [], [], []
    for h in range(H):
        row = kv_2d[h].float().unsqueeze(0)   # [1, D]
        try:
            Uh, Sh, Vhh = torch.linalg.svd(row, full_matrices=False)
        except (RuntimeError, TypeError):
            Uh, Sh, Vhh = torch.svd(row)
        Uh_r = Uh.squeeze(0)[:rank].clone()
        Sh_r = Sh[:rank].clone()
        Vh_r = Vhh[:rank, :].clone()
        U_l.append(Uh_r)
        S_l.append(Sh_r)
        Vh_l.append(Vh_r)
    return {
        "U":  torch.stack(U_l),   # [H, r]
        "S":  torch.stack(S_l),   # [H, r]
        "Vh": torch.stack(Vh_l),  # [H, r, D]
    }


# ──────────────────────────────────────────────
# INT4 量化/反量化
# ──────────────────────────────────────────────
def _quantize_int4(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """mat: [H, r, D] 或 [nl, nk, r, hd]"""
    groupsize = 128
    orig_shape = tuple(mat.shape)
    H = orig_shape[0]
    r = orig_shape[1]
    D = orig_shape[-1]
    D_orig = D
    pad = (groupsize - D_orig % groupsize) % groupsize
    if pad:
        mat = F.pad(mat, (0, pad))
    num_g = mat.shape[-1] // groupsize
    if mat.dim() == 3:
        g = mat.view(H, r, num_g, groupsize)
    elif mat.dim() == 4:
        # [nl, nk, r, hd] → view 需要先 reshape
        mat_r = mat.reshape(-1, r, num_g, groupsize)  # [nl*nk, r, ng, G]
        g = mat_r
        H_new = mat_r.shape[0]
        r_new = mat_r.shape[1]
        g = g.view(H_new, r_new, num_g, groupsize)
    else:
        raise ValueError(f"Unsupported dim: {mat.dim()}")
    scales = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    q = torch.round(g / scales).to(torch.int8)
    return q, scales, D_orig


def _dequantize_int4(q: torch.Tensor, scales: torch.Tensor, D_orig: int,
                     orig_shape: Tuple) -> torch.Tensor:
    """
    q: [H, r, ng, G], scales: [H, r, ng, 1] 或 [H*r, ng, 1]
    orig_shape: 原始 Vh shape
    """
    mat = q.float() * scales
    H, r, ng, G = mat.shape
    D_pad = ng * G
    mat = mat.view(H, r, D_pad)[:, :, :D_orig]
    return mat.reshape(orig_shape)


# ──────────────────────────────────────────────
# 完整压缩
# ──────────────────────────────────────────────
def compress_kv_full(k_input, v_input, rank=8, quantize=True, int4=True) -> Dict[str, Any]:
    """
    原生支持 4D [nl,nk,T,hd] 和 3D [H,T,D]。
    内部根据输入维度选择对应的 SVD 策略。
    """
    nd = k_input.dim()

    if nd == 4:
        k_comp = _svd_single_4d(k_input, rank)
        v_comp = _svd_single_4d(v_input, rank)
        result = {
            "_ndim": 4,
            "_k_shape": tuple(k_input.shape),
            "_v_shape": tuple(v_input.shape),
        }
    elif nd == 3:
        k_comp = _svd_single_3d(k_input, rank)
        v_comp = _svd_single_3d(v_input, rank)
        result = {
            "_ndim": 3,
            "_k_shape": tuple(k_input.shape),
            "_v_shape": tuple(v_input.shape),
        }
    elif nd == 2:
        k_comp = _svd_single_2d(k_input, rank)
        v_comp = _svd_single_2d(v_input, rank)
        result = {
            "_ndim": 2,
            "_k_shape": tuple(k_input.shape),
            "_v_shape": tuple(v_input.shape),
        }
    else:
        raise ValueError(f"Unsupported tensor dim: {nd}")

    result["K_U"] = k_comp["U"].half()
    result["K_S"] = k_comp["S"].half()
    result["V_U"] = v_comp["U"].half()
    result["V_S"] = v_comp["S"].half()

    if not quantize:
        result["K_Vh"] = k_comp["Vh"].half()
        result["V_Vh"] = v_comp["Vh"].half()
        return result

    if int4:
        q_k, sc_k, Dk = _quantize_int4(k_comp["Vh"])
        result["K_Vh_q"]     = q_k
        result["K_Vh_scales"] = sc_k
        result["K_Vh_D"]    = Dk
        result["K_Vh_orig_shape"] = tuple(k_comp["Vh"].shape)

        q_v, sc_v, Dv = _quantize_int4(v_comp["Vh"])
        result["V_Vh_q"]     = q_v
        result["V_Vh_scales"] = sc_v
        result["V_Vh_D"]    = Dv
        result["V_Vh_orig_shape"] = tuple(v_comp["Vh"].shape)
    else:
        result["K_Vh"] = k_comp["Vh"].half()
        result["V_Vh"] = v_comp["Vh"].half()

    return result


# ──────────────────────────────────────────────
# 解压
# ──────────────────────────────────────────────
def decompress_kv(comp: Dict[str, Any], quantize=True, int4=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    重建 KV tensor。
    3D:  K = matmul(K_U * S_K[:,None,:], K_Vh) → [H,T,D]
    4D:  K = matmul(K_U * S_K[:,:,None,:], K_Vh) → [nl,nk,T,hd]
    """
    ndim = comp.get("_ndim", 3)

    if quantize and int4:
        K_Vh_q   = comp["K_Vh_q"]
        K_scales = comp["K_Vh_scales"]
        K_D      = comp["K_Vh_D"]
        K_orig   = comp["K_Vh_orig_shape"]
        K_Vt     = _dequantize_int4(K_Vh_q, K_scales, K_D, K_orig)

        V_Vh_q   = comp["V_Vh_q"]
        V_scales = comp["V_Vh_scales"]
        V_D      = comp["V_Vh_D"]
        V_orig   = comp["V_Vh_orig_shape"]
        V_Vt     = _dequantize_int4(V_Vh_q, V_scales, V_D, V_orig)
    else:
        K_Vt = comp["K_Vh"].float()
        V_Vt = comp["V_Vh"].float()

    K_U = comp["K_U"].float()
    S_K = comp["K_S"].float()
    V_U = comp["V_U"].float()
    S_V = comp["V_S"].float()

    if ndim == 3:
        # K_U: [H, T, r], S_K: [H, r]
        # mid = (K_U * S_K[:, None, :]) → [H, T, r]
        # K = matmul(mid, K_Vt) → [H, T, D]
        K = torch.matmul(K_U * S_K.unsqueeze(1), K_Vt)
        V = torch.matmul(V_U * S_V.unsqueeze(1), V_Vt)
    elif ndim == 4:
        # K_U: [nl, nk, T, r], S_K: [nl, nk, r]
        # mid = (K_U * S_K[:, :, None, :]) → [nl, nk, T, r]
        # K = matmul(mid, K_Vt) → [nl, nk, T, hd]
        K = torch.matmul(K_U * S_K.unsqueeze(2), K_Vt)
        V = torch.matmul(V_U * S_V.unsqueeze(2), V_Vt)
    elif ndim == 2:
        # K_U: [H, r], S_K: [H, r], K_Vt: [H, r, D]
        # K = (K_U * S_K).unsqueeze(1) @ K_Vt → [H, 1, D] → squeeze → [H, D]
        K = torch.bmm((K_U * S_K).unsqueeze(1), K_Vt.float()).squeeze(1)
        V = torch.bmm((V_U * S_V).unsqueeze(1), V_Vt.float()).squeeze(1)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")

    return K, V


# ──────────────────────────────────────────────
# 误差测量
# ──────────────────────────────────────────────
def measure_reconstruction_error(k_orig, v_orig, k_rec, v_rec):
    def _e(o, r):
        # 确保在同一设备上计算
        dev = o.device
        o, r = o.to("cpu"), r.to("cpu")
        mx = (o - r).abs().max().item()
        mn = (o - r).abs().mean().item() / (o.abs().mean().item() + 1e-8)
        return mx, mn
    k_max, k_rel = _e(k_orig, k_rec)
    v_max, v_rel = _e(v_orig, v_rec)
    return {"k_max_abs": k_max, "k_rel_err": k_rel,
            "v_max_abs": v_max, "v_rel_err": v_rel}


# ──────────────────────────────────────────────
# 自测
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SVD v7 自测 (3D/4D per-[T,D] SVD)")
    print("=" * 60)

    # 3D [H,T,D]
    t3 = torch.randn(8, 512, 128)
    for rank in [8, 32]:
        c = compress_kv_full(t3, t3, rank=rank, quantize=True, int4=True)
        kr, vr = decompress_kv(c)
        e = measure_reconstruction_error(t3, t3, kr, vr)
        print(f"3D r={rank:2d}: K_rel={e['k_rel_err']:.4f} V_rel={e['v_rel_err']:.4f}")

    # 4D decode
    t4d = torch.randn(32, 8, 128, 128)
    for rank in [8, 32]:
        c = compress_kv_full(t4d, t4d, rank=rank, quantize=True, int4=True)
        kr, vr = decompress_kv(c)
        e = measure_reconstruction_error(t4d, t4d, kr, vr)
        print(f"4D-decode r={rank:2d}: K_rel={e['k_rel_err']:.4f} V_rel={e['v_rel_err']:.4f}")

    # 4D prefill
    t4p = torch.randn(32, 8, 512, 128)
    for rank in [8, 32]:
        c = compress_kv_full(t4p, t4p, rank=rank, quantize=True, int4=True)
        kr, vr = decompress_kv(c)
        e = measure_reconstruction_error(t4p, t4p, kr, vr)
        print(f"4D-prefill r={rank:2d}: K_rel={e['k_rel_err']:.4f} V_rel={e['v_rel_err']:.4f}")

    print("ALL OK")
