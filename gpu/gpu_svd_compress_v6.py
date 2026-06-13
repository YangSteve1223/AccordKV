#!/usr/bin/env python3
"""
gpu_svd_compress_fixed.py — 修复版 v6

修复记录:
  v5→v6: 彻底修正 reshape 逻辑
         - 4D [nl,nk,T,hd]: reshape → [nl*nk, T*hd] per-head SVD (正确！)
         - 3D [H,T,D]: reshape → [H*T, D] per-layer-time SVD (原始 v4 正确策略)
         - compress 内部自动选择 reshape 策略
  v3→v4: K/V 完全分离
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any

# ──────────────────────────────────────────────
# 工具: 判断并 reshape 输入为 2D
# ──────────────────────────────────────────────
def _flatten_kv(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple, int]:
    """
    将任意维度 tensor reshape 为 2D，并返回 (2d_tensor, orig_shape, strategy)
    strategy: 0=2D直接, 1=4D→[nl*nk, T*hd], 2=3D→[H*T, D]
    """
    orig_shape = tuple(tensor.shape)
    nd = tensor.dim()

    if nd == 2:
        return tensor, orig_shape, 0
    elif nd == 4:
        # [nl, nk, T, hd] → [nl*nk, T*hd]
        nl, nk, T, hd = tensor.shape
        t2d = tensor.permute(0, 1, 2, 3).reshape(nl * nk, T * hd)
        return t2d, orig_shape, 1
    elif nd == 3:
        # [H, T, D] → [H*T, D]
        H, T, D = tensor.shape
        t2d = tensor.reshape(H * T, D)
        return t2d, orig_shape, 2
    else:
        raise ValueError(f"Unsupported dim: {nd}")


def _unflatten_kv(tensor_2d: torch.Tensor, orig_shape: Tuple, strategy: int) -> torch.Tensor:
    """还原为原始维度"""
    if strategy == 0:
        return tensor_2d.reshape(orig_shape)
    elif strategy == 1:
        nl, nk, T, hd = orig_shape
        return tensor_2d.reshape(nl, nk, T, hd)
    elif strategy == 2:
        H, T, D = orig_shape
        return tensor_2d.reshape(H, T, D)
    else:
        return tensor_2d


# ──────────────────────────────────────────────
# SVD: 对 2D [B, D] 的每行做截断 SVD
# ──────────────────────────────────────────────
def _svd_per_row(kv_2d: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    """
    对 2D [B, D] 的每行做截断 SVD。
    返回 U:[B,r], S:[B,r], Vh:[B,r,D] — Vh 是右奇异向量行向量拼接。
    重建: row_b = Σ_r (U[b,r] * S[b,r]) * Vh[b,r,:]
    """
    B, D = kv_2d.shape
    U_l, S_l, Vh_l = [], [], []
    for b in range(B):
        row = kv_2d[b].float()
        # 确保 2D
        if row.dim() == 1:
            row = row.unsqueeze(0)
        try:
            Uh, Sh, Vhh = torch.linalg.svd(row, full_matrices=False)
        except (RuntimeError, TypeError):
            Uh, Sh, Vhh = torch.svd(row)
        # Uh: [1, D] or [D, D]; Sh: [min(1,D)] or [D]
        # Vhh = V^T (right singular vectors as rows)
        Uh_r = Uh.squeeze(0)[:rank].clone()       # [r]
        Sh_r = Sh[:rank].clone()                   # [r]
        Vh_r = Vhh[:rank, :].clone()              # [r, D]
        U_l.append(Uh_r)
        S_l.append(Sh_r)
        Vh_l.append(Vh_r)
    return {
        "U":  torch.stack(U_l),   # [B, r]
        "S":  torch.stack(S_l),   # [B, r]
        "Vh": torch.stack(Vh_l),  # [B, r, D]
    }


# ──────────────────────────────────────────────
# INT4 量化/反量化（对 Vh）
# ──────────────────────────────────────────────
def _quantize_int4(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    mat: [B, r, D]
    groupsize=128
    """
    groupsize = 128
    B, r, D = mat.shape
    D_orig = D
    pad = (groupsize - D_orig % groupsize) % groupsize
    if pad:
        mat = F.pad(mat, (0, pad))
    num_g = mat.shape[-1] // groupsize
    g = mat.view(B, r, num_g, groupsize)
    scales = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    q = torch.round(g / scales).to(torch.int8)
    return q, scales, D_orig


def _dequantize_int4(q: torch.Tensor, scales: torch.Tensor, D_orig: int) -> torch.Tensor:
    """q: [B, r, ng, G], scales: [B, r, ng, 1]"""
    B, r, ng, G = q.shape
    D_pad = ng * G
    mat = (q.float() * scales).view(B, r, D_pad)[:, :, :D_orig]
    return mat


# ──────────────────────────────────────────────
# 完整压缩
# ──────────────────────────────────────────────
def compress_kv_full(k_input, v_input, rank=8, quantize=True, int4=True) -> Dict[str, Any]:
    """
    原生支持 4D/3D/2D。
    内部统一 flatten → SVD → quantize → 存元信息供还原。
    """
    k2d, k_shape, k_strat = _flatten_kv(k_input)
    v2d, v_shape, v_strat = _flatten_kv(v_input)

    k_comp = _svd_per_row(k2d, rank)
    v_comp = _svd_per_row(v2d, rank)

    result = {
        "_k_shape": k_shape, "_k_strat": k_strat,
        "_v_shape": v_shape, "_v_strat": v_strat,
    }

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
        result["K_Vh_D"]     = Dk
        q_v, sc_v, Dv = _quantize_int4(v_comp["Vh"])
        result["V_Vh_q"]     = q_v
        result["V_Vh_scales"] = sc_v
        result["V_Vh_D"]     = Dv
    else:
        result["K_Vh"] = k_comp["Vh"].half()
        result["V_Vh"] = v_comp["Vh"].half()

    return result


# ──────────────────────────────────────────────
# 解压
# ──────────────────────────────────────────────
def decompress_kv(comp: Dict[str, Any], quantize=True, int4=True) -> Tuple[torch.Tensor, torch.Tensor]:
    k_shape = comp["_k_shape"]
    v_shape = comp["_v_shape"]
    k_strat = comp.get("_k_strat", 0)
    v_strat = comp.get("_v_strat", 0)

    if quantize and int4:
        K_Vt = _dequantize_int4(comp["K_Vh_q"], comp["K_Vh_scales"], comp["K_Vh_D"])
        V_Vt = _dequantize_int4(comp["V_Vh_q"], comp["V_Vh_scales"], comp["V_Vh_D"])
    else:
        K_Vt = comp["K_Vh"]
        V_Vt = comp["V_Vh"]

    K_U = comp["K_U"].float()  # [B, r]
    S_K = comp["K_S"].float()
    V_U = comp["V_U"].float()
    S_V = comp["V_S"].float()

    # 重建: Σ_r (U[b,r]*S[b,r]) * Vh[b,r,:]
    # (U*S) [B,r] unsqueeze → [B,1,r] × Vh [B,r,D] → bmm → [B,1,D] → squeeze
    K_2d = torch.bmm((K_U * S_K).unsqueeze(1), K_Vt.float()).squeeze(1)
    V_2d = torch.bmm((V_U * S_V).unsqueeze(1), V_Vt.float()).squeeze(1)

    K = _unflatten_kv(K_2d, k_shape, k_strat)
    V = _unflatten_kv(V_2d, v_shape, v_strat)
    return K, V


# ──────────────────────────────────────────────
# 误差测量
# ──────────────────────────────────────────────
def measure_reconstruction_error(k_orig, v_orig, k_rec, v_rec):
    def _e(o, r):
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
    print("SVD v6 自测 (3D/4D 正确 reshape)")
    print("=" * 60)

    cases = [
        ("3D [H,T,D]",       (8,  512, 128)),
        ("4D [nl,nk,T,hd]", (32,   8, 512, 128)),
        ("4D decode",        (32,   8, 128, 256)),
        ("2D [B,D]",         (8,  4096)),
    ]
    for label, shape in cases:
        print(f"\n  Case: {label} {shape}")
        if len(shape) == 3:
            tensor = torch.randn(shape)
            c = compress_kv_full(tensor, tensor, rank=8, quantize=True, int4=True)
            kr, vr = decompress_kv(c)
            err = measure_reconstruction_error(tensor, tensor, kr, vr)
            print(f"    I/O: {tensor.shape} → {kr.shape}  | rel_err K={err['k_rel_err']:.4f} V={err['v_rel_err']:.4f}")
        elif len(shape) == 4:
            K = torch.randn(shape)
            V = torch.randn(shape)
            c = compress_kv_full(K, V, rank=8, quantize=True, int4=True)
            kr, vr = decompress_kv(c)
            err = measure_reconstruction_error(K, V, kr, vr)
            print(f"    I/O: {K.shape} → {kr.shape}  | rel_err K={err['k_rel_err']:.4f} V={err['v_rel_err']:.4f}")
        else:
            tensor = torch.randn(shape)
            c = compress_kv_full(tensor, tensor, rank=8, quantize=True, int4=True)
            kr, vr = decompress_kv(c)
            err = measure_reconstruction_error(tensor, tensor, kr, vr)
            print(f"    I/O: {tensor.shape} → {kr.shape}  | rel_err K={err['k_rel_err']:.4f} V={err['v_rel_err']:.4f}")
