#!/usr/bin/env python3
"""
ACCORD-KV SVD+INT4 压缩核心库 v8
修复: _quantize_int4 对 4D tensor [nl,nk,rank,hd] 的 reshape 逻辑
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any

# ──────────────────────────────────────────────
# SVD: 对 [T, D] 做截断 SVD，提取 top-r 个奇异值
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
    U = torch.zeros(T, rank, dtype=Ut.dtype)
    S = torch.zeros(rank, dtype=St.dtype)
    Vh = torch.zeros(rank, D, dtype=Vht.dtype)
    U[:, :actual] = Ut[:, :actual]
    S[:actual] = St[:actual]
    Vh[:actual, :] = Vht[:actual, :]
    return {"U": U, "S": S, "Vh": Vh}


def _svd_single_3d(kv_3d: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    """3D [H, T, D]: 对每个 H 做 SVD。返回 U:[H,T,r], S:[H,r], Vh:[H,r,D]"""
    H, T, D = kv_3d.shape
    U_l, S_l, Vh_l = [], [], []
    for h in range(H):
        chunk = kv_3d[h, :, :]
        res = _svd_td(chunk, rank)
        U_l.append(res["U"]); S_l.append(res["S"]); Vh_l.append(res["Vh"])
    return {"U": torch.stack(U_l), "S": torch.stack(S_l), "Vh": torch.stack(Vh_l)}


def _svd_single_4d(kv_4d: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    """
    4D [nl, nk, T, hd]: 对每个 (layer, head) 的 [T, hd] 做 SVD。
    返回 U:[nl,nk,T,r], S:[nl,nk,r], Vh:[nl,nk,r,hd]。
    """
    nl, nk, T, hd = kv_4d.shape
    U_l, S_l, Vh_l = [], [], []
    for l in range(nl):
        for k in range(nk):
            chunk = kv_4d[l, k, :, :]
            res = _svd_td(chunk, rank)
            U_l.append(res["U"]); S_l.append(res["S"]); Vh_l.append(res["Vh"])
    U_all = torch.stack(U_l).reshape(nl, nk, T, rank)
    S_all = torch.stack(S_l).reshape(nl, nk, rank)
    Vh_all = torch.stack(Vh_l).reshape(nl, nk, rank, hd)
    return {"U": U_all, "S": S_all, "Vh": Vh_all}


def _svd_single_2d(kv_2d: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    """2D [H, D]: 对每行做 SVD"""
    H, D = kv_2d.shape
    U_l, S_l, Vh_l = [], [], []
    for h in range(H):
        row = kv_2d[h].float().unsqueeze(0)
        try:
            Uh, Sh, Vhh = torch.linalg.svd(row, full_matrices=False)
        except (RuntimeError, TypeError):
            Uh, Sh, Vhh = torch.svd(row)
        U_l.append(Uh.squeeze(0)[:rank].clone())
        S_l.append(Sh[:rank].clone())
        Vh_l.append(Vhh[:rank, :].clone())
    return {"U": torch.stack(U_l), "S": torch.stack(S_l), "Vh": torch.stack(Vh_l)}


# ──────────────────────────────────────────────
# INT4 量化 / 反量化（修复版，支持 4D）
# ──────────────────────────────────────────────
def _quantize_int4(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    对 SVD 的 Vh 矩阵做 INT4 分组量化。
    输入: [H, r, D] (3D) 或 [nl, nk, rank, hd] (4D)
    输出: q [5D/4D], scales [5D/4D], D_orig
    """
    groupsize = 128
    orig_shape = tuple(mat.shape)
    D = orig_shape[-1]
    D_orig = D
    pad = (groupsize - D_orig % groupsize) % groupsize
    if pad:
        mat = F.pad(mat, (0, pad))
    ng = mat.shape[-1] // groupsize  # hd/128 通常 = 1

    if mat.dim() == 3:
        # [H, r, D_pad] → [H, r, ng, G]
        H, r = orig_shape[0], orig_shape[1]
        g = mat.view(H, r, ng, groupsize)
    elif mat.dim() == 4:
        # [nl, nk, rank, D_pad] → [nl, nk, rank, ng, G]
        g = mat.unsqueeze(-2)                                  # [nl,nk,rank,1,D_pad]
        g = g.reshape(*orig_shape[:-1], ng, groupsize)        # [nl,nk,rank,ng,G]
    else:
        raise ValueError(f"Unsupported dim: {mat.dim()}")
    scales = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    q = torch.round(g / scales).to(torch.int8)
    return q, scales, D_orig


def _dequantize_int4(q: torch.Tensor, scales: torch.Tensor, D_orig: int,
                     orig_shape: Tuple) -> torch.Tensor:
    """
    反量化: q [5D/4D], scales [5D/4D] → 重建原始 shape 的 tensor
    """
    mat = q.float() * scales
    if mat.dim() == 5:
        # 4D path: [nl, nk, rank, ng, G] → reshape via [nl,nk,rank,hd]
        mat = mat.reshape(*mat.shape[:-2], -1)[:, :, :, :D_orig]
        return mat.reshape(orig_shape)
    else:
        # 3D path: [H, r, ng, G] → [H, r, D_orig]
        H, r, ng, G = mat.shape
        D_pad = ng * G
        mat = mat.view(H, r, D_pad)[:, :, :D_orig]
        return mat.reshape(orig_shape)


# ──────────────────────────────────────────────
# 完整压缩（4D/3D/2D 自适应）
# ──────────────────────────────────────────────
def compress_kv_full(k_input, v_input, rank=8, quantize=True, int4=True) -> Dict[str, Any]:
    """
    4D [nl,nk,T,hd]: 用 _svd_single_4d（per head SVD）
    3D [H,T,D]:      用 _svd_single_3d
    2D [H,D]:        用 _svd_single_2d
    """
    nd = k_input.dim()
    if nd == 4:
        k_comp = _svd_single_4d(k_input, rank)
        v_comp = _svd_single_4d(v_input, rank)
        ndim = 4
    elif nd == 3:
        k_comp = _svd_single_3d(k_input, rank)
        v_comp = _svd_single_3d(v_input, rank)
        ndim = 3
    elif nd == 2:
        k_comp = _svd_single_2d(k_input, rank)
        v_comp = _svd_single_2d(v_input, rank)
        ndim = 2
    else:
        raise ValueError(f"Unsupported tensor dim: {nd}")

    result = {"_ndim": ndim, "_k_shape": tuple(k_input.shape), "_v_shape": tuple(v_input.shape)}
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
        q_v, sc_v, Dv = _quantize_int4(v_comp["Vh"])
        result.update({
            "K_Vh_q": q_k, "K_Vh_scales": sc_k, "K_Vh_D": Dk,
            "K_Vh_orig_shape": tuple(k_comp["Vh"].shape),
            "V_Vh_q": q_v, "V_Vh_scales": sc_v, "V_Vh_D": Dv,
            "V_Vh_orig_shape": tuple(v_comp["Vh"].shape),
        })
    else:
        result["K_Vh"] = k_comp["Vh"].half()
        result["V_Vh"] = v_comp["Vh"].half()
    return result


def decompress_kv(comp: Dict[str, Any], quantize=True, int4=True):
    """
    重建 KV tensor。
    4D: K = matmul(K_U * S_K.unsqueeze(1).unsqueeze(1), K_Vh) → [nl,nk,T,hd]
    3D: K = matmul(K_U * S_K[:,None,:], K_Vh) → [H,T,D]
    """
    ndim = comp.get("_ndim", 3)
    if quantize and int4:
        K_Vt = _dequantize_int4(comp["K_Vh_q"], comp["K_Vh_scales"],
                                comp["K_Vh_D"], comp["K_Vh_orig_shape"])
        V_Vt = _dequantize_int4(comp["V_Vh_q"], comp["V_Vh_scales"],
                                comp["V_Vh_D"], comp["V_Vh_orig_shape"])
    else:
        K_Vt = comp["K_Vh"].float()
        V_Vt = comp["V_Vh"].float()

    K_U = comp["K_U"].float(); S_K = comp["K_S"].float()
    V_U = comp["V_U"].float(); S_V = comp["V_S"].float()

    if ndim == 4:
        # S_K: [nl, nk, rank] → [nl, nk, 1, rank] to broadcast with K_U: [nl, nk, T, rank]
        K = torch.matmul(K_U * S_K.unsqueeze(2), K_Vt)
        V = torch.matmul(V_U * S_V.unsqueeze(2), V_Vt)
    elif ndim == 3:
        K = torch.matmul(K_U * S_K[:, None, :], K_Vt)
        V = torch.matmul(V_U * S_V[:, None, :], V_Vt)
    else:
        K = torch.matmul(K_U * S_K, K_Vt)
        V = torch.matmul(V_U * S_V, V_Vt)
    return K, V


# ──────────────────────────────────────────────
# 误差测量
# ──────────────────────────────────────────────
def measure_reconstruction_error(k_orig, v_orig, k_rec, v_rec):
    def _e(o, r):
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
    print("SVD v8 自测 (4D INT4 修复版)")
    print("=" * 60)

    # 4D 测试
    nl, nk, T, hd = 32, 8, 512, 128
    K4 = torch.randn(nl, nk, T, hd)
    V4 = torch.randn(nl, nk, T, hd)

    for rank in [8, 32, 256]:
        c = compress_kv_full(K4, V4, rank=rank, quantize=True, int4=True)
        Kr, Vr = decompress_kv(c, quantize=True, int4=True)
        e = measure_reconstruction_error(K4, V4, Kr, Vr)
        print(f"[4D] rank={rank:3d}  K_rel={e['k_rel_err']:.4f}  V_rel={e['v_rel_err']:.4f}")

    # 3D 测试
    print()
    K3 = torch.randn(256, 512, 128)
    V3 = torch.randn(256, 512, 128)
    for rank in [8, 32]:
        c = compress_kv_full(K3, V3, rank=rank, quantize=True, int4=True)
        Kr, Vr = decompress_kv(c, quantize=True, int4=True)
        e = measure_reconstruction_error(K3, V3, Kr, Vr)
        print(f"[3D] rank={rank:3d}  K_rel={e['k_rel_err']:.4f}  V_rel={e['v_rel_err']:.4f}")

    # FP16 测试（无量化）
    print()
    c = compress_kv_full(K4, V4, rank=8, quantize=False)
    Kr, Vr = decompress_kv(c, quantize=False)
    e = measure_reconstruction_error(K4, V4, Kr, Vr)
    print(f"[4D] FP16 r=8  K_rel={e['k_rel_err']:.4f}  V_rel={e['v_rel_err']:.4f}")
    print("v8 自测完成")
