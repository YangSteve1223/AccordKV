#!/usr/bin/env python3
"""Fix v7: _quantize_int4 and _dequantize_int4 for 4D tensors."""

with open('/app/data/所有对话/主对话/_staging/accord-kv/gpu_svd_compress_v7.py', 'r') as f:
    code = f.read()

# ── Fix 1: _quantize_int4 ──────────────────────────────────────────
old_q = '''def _quantize_int4(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
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
        H_new = mat_r.shape[0]
        r_new = mat_r.shape[1]
        g = g.view(H_new, r_new, num_g, groupsize)
    else:
        raise ValueError(f"Unsupported dim: {mat.dim()}")
    scales = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    q = torch.round(g / scales).to(torch.int8)
    return q, scales, D_orig'''

new_q = '''def _quantize_int4(mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """mat: [H, r, D] 或 [nl, nk, rank, hd]"""
    groupsize = 128
    orig_shape = tuple(mat.shape)
    D = orig_shape[-1]
    D_orig = D
    pad = (groupsize - D_orig % groupsize) % groupsize
    if pad:
        mat = F.pad(mat, (0, pad))
    ng = mat.shape[-1] // groupsize

    if mat.dim() == 3:
        # [H, r, D_pad] → [H, r, ng, G]
        H, r = orig_shape[0], orig_shape[1]
        g = mat.view(H, r, ng, groupsize)
    elif mat.dim() == 4:
        # [nl, nk, rank, D_pad] → [nl, nk, rank, ng, G]
        g = mat.unsqueeze(-2)                                   # [nl,nk,rank,1,D_pad]
        g = g.reshape(*orig_shape[:-1], ng, groupsize)         # [nl,nk,rank,ng,G]
    else:
        raise ValueError(f"Unsupported dim: {mat.dim()}")
    scales = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    q = torch.round(g / scales).to(torch.int8)
    return q, scales, D_orig'''

code = code.replace(old_q, new_q)

# ── Fix 2: _dequantize_int4 ───────────────────────────────────────
old_dq = '''def _dequantize_int4(q: torch.Tensor, scales: torch.Tensor, D_orig: int,
                     orig_shape: Tuple) -> torch.Tensor:
    """
    q: [H, r, ng, G], scales: [H, r, ng, 1] 或 [H*r, ng, 1]
    orig_shape: 原始 Vh shape
    """
    mat = q.float() * scales
    H, r, ng, G = mat.shape
    D_pad = ng * G
    mat = mat.view(H, r, D_pad)[:, :, :D_orig]
    return mat.reshape(orig_shape)'''

new_dq = '''def _dequantize_int4(q: torch.Tensor, scales: torch.Tensor, D_orig: int,
                     orig_shape: Tuple) -> torch.Tensor:
    """
    q: [H, r, ng, G] (3D) 或 [nl, nk, rank, ng, G] (4D)
    scales: [H, r, ng, 1] (3D) 或 [nl, nk, rank, ng, 1] (4D)
    orig_shape: 原始 Vh shape
    """
    mat = q.float() * scales
    if mat.dim() == 5:
        # 4D path: [nl, nk, rank, ng, G] → [nl, nk, rank, hd]
        mat = mat.reshape(*mat.shape[:-2], -1)[:, :, :, :D_orig]
        return mat.reshape(orig_shape)
    else:
        # 3D path: [H, r, ng, G] → [H, r, D_orig]
        H, r, ng, G = mat.shape
        D_pad = ng * G
        mat = mat.view(H, r, D_pad)[:, :, :D_orig]
        return mat.reshape(orig_shape)'''

code = code.replace(old_dq, new_dq)

with open('/app/data/所有对话/主对话/_staging/accord-kv/gpu_svd_compress_v8.py', 'w') as f:
    f.write(code)
print("v8 written OK")
