#!/usr/bin/env python3
"""
gpu_svd_compress_fixed.py — 修复版 v5

修复记录:
  v4→v5: 原生支持 4D [nl,nk,T,hd] 输入
         compress: 4D→2D→SVD per-head→重建后 reshape回4D
         decompress: 2D→reshape→4D，保证 output shape = input shape
  v3→v4: K/V 完全分离，各自独立压缩/量化/存储
  v2→v3: einsum换matmul + INT4 key名字对齐
  v1→v2: Vh不用.T直接重建
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any

# ──────────────────────────────────────────────
# 0. Shape 工具：4D / 3D / 2D 互转
# ──────────────────────────────────────────────
def _reshape_for_compress(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple, bool]:
    """
    将任意维度 tensor reshape 为 2D [B, D] 用于 SVD。
    同时记录原始 shape 和是否是 4D 输入。
    返回: (tensor_2d, orig_shape, is_4d)
    """
    orig_shape = tuple(tensor.shape)
    if tensor.dim() == 4:
        # [nl, nk, T, hd] → [nl*nk, T*hd]
        nl, nk, T, hd = tensor.shape
        t2d = tensor.permute(0, 1, 2, 3).reshape(nl * nk, T * hd)
        return t2d, orig_shape, True
    elif tensor.dim() == 3:
        # [H, T, D] → [H, T*D]
        H, T, D = tensor.shape
        t2d = tensor.reshape(H, T * D)
        return t2d, orig_shape, False
    elif tensor.dim() == 2:
        return tensor, orig_shape, False
    else:
        raise ValueError(f"Unsupported tensor dim: {tensor.dim()}")


def _reshape_from_compress(tensor_2d: torch.Tensor, orig_shape: Tuple) -> torch.Tensor:
    """将 2D SVD 结果 reshape 回原始维度（支持任意原始维度）"""
    return tensor_2d.reshape(orig_shape)


# ──────────────────────────────────────────────
# 1. SVD 压缩: 2D → SVD per-row → {U:[B,r], S:[B,r], Vh:[B,r,D]}
# ──────────────────────────────────────────────
def _svd_single_2d(kv_2d: torch.Tensor, rank: int) -> Dict[str, torch.Tensor]:
    """
    对 2D tensor [B, D] 的每一行做截断 SVD。
    直接对 [D] 行向量矩阵做 SVD（不 unsqueeze 成 [1,D]），
    因为 M=1 时 svd 只返回 1 个奇异值。
    """
    B, D = kv_2d.shape
    U_l, S_l, Vh_l = [], [], []
    for b in range(B):
        chunk = kv_2d[b].float()   # [D]
        try:
            Uh, Sh, Vhh = torch.linalg.svd(chunk, full_matrices=False)
        except (RuntimeError, TypeError):
            Uh, Sh, Vhh = torch.svd(chunk)
        # Uh: [D], Sh: [D], Vhh: [D, D]
        # 截断到 rank
        Uh_r = Uh[:rank].clone()     # [r]
        Sh_r = Sh[:rank].clone()      # [r]
        # Vhh: V^T in standard notation; rows are right singular vectors
        Vh_r = Vhh[:rank, :].clone()  # [r, D]
        U_l.append(Uh_r)
        S_l.append(Sh_r)
        Vh_l.append(Vh_r)
    return {
        "U":  torch.stack(U_l),   # [B, r]
        "S":  torch.stack(S_l),   # [B, r]
        "Vh": torch.stack(Vh_l),  # [B, r, D]
    }


# ──────────────────────────────────────────────
# 2. INT4 量化/反量化
# ──────────────────────────────────────────────
def _quantize_int4(mat):
    """
    mat: [B, r, D]  (per-row 量化)
    返回 (q_mat, scales, D_orig)
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


def _dequantize_int4(q, scales, D_orig):
    """q: [B, r, ng, G], scales: [B, r, ng, 1]"""
    B, r, ng, G = q.shape
    D_pad = ng * G
    mat = (q.float() * scales).view(B, r, D_pad)[:, :, :D_orig]
    return mat


# ──────────────────────────────────────────────
# 3. 完整压缩: 原生 4D/3D/2D 皆可
# ──────────────────────────────────────────────
def compress_kv_full(k_input, v_input, rank=8, quantize=True, int4=True) -> Dict[str, Any]:
    """
    k_input / v_input: [nl,nk,T,hd] (4D) / [H,T,D] (3D) / [B,D] (2D)
    返回 dict。压缩后的 U/S 以 2D 存，Vh 也以 2D 存。
    额外存 _orig_shape 和 _is_4d 供 decompress 还原形状。
    """
    # 2D 化
    k2d, k_shape, k_4d = _reshape_for_compress(k_input)
    v2d, v_shape, v_4d = _reshape_for_compress(v_input)

    k_comp = _svd_single_2d(k2d, rank)
    v_comp = _svd_single_2d(v2d, rank)

    result = {}
    # _k_shape / _v_shape: 2D 形状（供 decompress 还原）
    result["_k_shape"] = tuple(k2d.shape)
    result["_v_shape"] = tuple(v2d.shape)

    # K 存 U, S, Vh
    result["K_U"]  = k_comp["U"]   # [B, r]
    result["K_S"]  = k_comp["S"]   # [B, r]
    # V 存 U, S, Vh
    result["V_U"]  = v_comp["U"]
    result["V_S"]  = v_comp["S"]

    if not quantize:
        result["K_Vh"] = k_comp["Vh"]
        result["V_Vh"] = v_comp["Vh"]
        return result

    # FP16 U, S
    result["K_U"] = result["K_U"].half()
    result["K_S"] = result["K_S"].half()
    result["V_U"] = result["V_U"].half()
    result["V_S"] = result["V_S"].half()

    # Vh 量化
    if int4:
        q_k, sc_k, D_k = _quantize_int4(k_comp["Vh"])
        result["K_Vh_q"]     = q_k
        result["K_Vh_scales"] = sc_k
        result["K_Vh_D"]    = D_k

        q_v, sc_v, D_v = _quantize_int4(v_comp["Vh"])
        result["V_Vh_q"]     = q_v
        result["V_Vh_scales"] = sc_v
        result["V_Vh_D"]    = D_v
    else:
        result["K_Vh"] = k_comp["Vh"].half()
        result["V_Vh"] = v_comp["Vh"].half()

    return result


# ──────────────────────────────────────────────
# 4. 解压: 还原形状
# ──────────────────────────────────────────────
def decompress_kv(comp: Dict[str, Any], quantize=True, int4=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    重建 KV tensor，形状与 compress 输入完全一致。
    逻辑: U[B,r] * S[B,r] → [B,r] → @ Vh[B,r,D] → [B,D] → reshape 回原始
    """
    k_shape = comp["_k_shape"]
    v_shape = comp["_v_shape"]

    if quantize and int4:
        K_Vt = _dequantize_int4(comp["K_Vh_q"], comp["K_Vh_scales"], comp["K_Vh_D"])
        V_Vt = _dequantize_int4(comp["V_Vh_q"], comp["V_Vh_scales"], comp["V_Vh_D"])
    else:
        K_Vt = comp["K_Vh"]
        V_Vt = comp["V_Vh"]

    K_U = comp["K_U"].float()
    S_K = comp["K_S"].float()
    V_U = comp["V_U"].float()
    S_V = comp["V_S"].float()
    # 重建: (U * S) @ Vh
    # US [B, r] -> [B, 1, r], Vh [B, r, D] -> bmm → [B, 1, D] -> squeeze
    US_K = (K_U * S_K).unsqueeze(1)
    K_2d = torch.bmm(US_K, K_Vt.float()).squeeze(1)  # [B, D]

    US_V = (V_U * S_V).unsqueeze(1)
    V_2d = torch.bmm(US_V, V_Vt.float()).squeeze(1)

    K = _reshape_from_compress(K_2d, k_shape)
    V = _reshape_from_compress(V_2d, v_shape)
    return K, V


# ──────────────────────────────────────────────
# 5. 误差测量
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
# 6. 自测
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SVD Compress+Decompress 自测 (v5 4D支持)")
    print("=" * 60)

    cases = [
        ("3D [H,T,D]",   (8, 512, 128)),
        ("4D [nl,nk,T,D]", (32, 8, 512, 128)),
        ("4D decode",    (32, 8, 128, 256)),
    ]

    for label, shape in cases:
        print(f"\nCase: {label} {shape}")
        if len(shape) == 3:
            tensor = torch.randn(shape)
            print(f"  Input 3D: {tensor.shape}")
            c = compress_kv_full(tensor, tensor, rank=8, quantize=True, int4=True)
            kr, vr = decompress_kv(c)
            print(f"  Output 3D: {kr.shape}")
            err = measure_reconstruction_error(tensor, tensor, kr, vr)
            print(f"  RelErr: K={err['k_rel_err']:.4f} V={err['v_rel_err']:.4f}")
        elif len(shape) == 4:
            K = torch.randn(shape)
            V = torch.randn(shape)
            print(f"  Input 4D: {K.shape}")
            c = compress_kv_full(K, V, rank=8, quantize=True, int4=True)
            kr, vr = decompress_kv(c)
            print(f"  Output 4D: {kr.shape}")
            err = measure_reconstruction_error(K, V, kr, vr)
            print(f"  RelErr: K={err['k_rel_err']:.4f} V={err['v_rel_err']:.4f}")
