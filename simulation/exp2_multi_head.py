"""
Exp2: Multi-head 真模型 + q_len=1 decoding + wavesize sweep。

跟 Exp1 差异：
- Exp1 是 single-head (H=1) 的 single-pass attention（mock）
- Exp2 是 multi-head (H=4) 的 block-wise FlashAttention（真模型），
  每 block 算 partial m/l/y 走 online softmax 公式

新增维度：
- H=4 heads (per-head d=64, total d=256)
- block_size=64（每个 KV block 是 64 个 token）
- num_shards sweep: {1, 4, 16}（看 wavesize 对 saving 的影响）
- q_len sweep: {1, 4, 16, 64, 256}（重点看 q=1 decoding）

三组路径同 Exp1:
    A. KV tensor path (整段传输)
    B. (m, l, y) stats path (per-shard compute + merge)
    C. ExactLocal (本地 baseline)

Success criteria:
- ratio 跟 Exp1 同量级（multi-head 不应改变 bytes_B 的相对量级，因为 head 是 stats 维度的扩展）
- err_B < 1e-3（真 block-wise 算法比 mock 应该精度略低，但仍在合理范围）
- q_len=1 时 ratio 应该到 256x ~ 32768x（理论 2*kv_len*4/d_64 = 8*kv_len = 8K ~ 128K，对应 8192x ~ 131072x 范围）

时间预算: 4*5*3 = 60 组 sweep, sandbox 大约 30-60s
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# 复用 Exp1 的 numpy 版 (m,l,y) + merge
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp1_fidelity_vs_bandwidth import (
    NumpyAttnStats,
    numpy_merge_stats,
    numpy_merge_stats_list,
    ground_truth,
    bytes_kv_path,
    bytes_stats_path,
)


# ============== 多头真模型 ==============


def flash_attn_blockwise(
    Q: np.ndarray,
    K_blocks: list[np.ndarray],
    V_blocks: list[np.ndarray],
    H: int,
) -> NumpyAttnStats:
    """多 head block-wise FlashAttention（online softmax 公式）。

    Parameters
    ----------
    Q : [q_len, d]            — query (单 head 的 dim)
    K_blocks, V_blocks : list of [block_size, d]
    H : int                    — heads 数（per-head 独立跑）

    Returns
    -------
    NumpyAttnStats (H 个 head 的 stats)
        m: [H, q_len, 1]
        l: [H, q_len, 1]
        y: [H, q_len, d]
    """
    Ql, d = Q.shape
    # 每个 head 用不同随机投影（简化版：H 个 head 共享 Q/K/V，但用不同 mask 模拟独立 attention）
    # 真实场景：Q/K/V 每个 head 独立投影。这里为了 sandbox 仿真可跑，用不同 head 看不同 block 子集
    # 但更现实的做法是：每个 head 的 (Q, K, V) 都不同，sandbox 仿真直接复制 Q 到 H 个 head
    # 这样 stats 的 head 维度是真实可合并的
    Q_h = np.broadcast_to(Q, (H, Ql, d))  # [H, q_len, d]

    out_m = np.full((H, Ql, 1), -np.inf, dtype=np.float32)
    out_l = np.zeros((H, Ql, 1), dtype=np.float32)
    out_y = np.zeros((H, Ql, d), dtype=np.float32)

    for K_blk, V_blk in zip(K_blocks, V_blocks):
        # scores[h, q, k_pos] = Q_h[h, q, :] @ K_blk[k_pos, :]
        # = Q @ K_blk.T  (broadcast over H)
        scores = np.einsum("hqd,kd->hqk", Q_h, K_blk)  # [H, q_len, block_size]

        # block-local max
        m_blk = scores.max(axis=-1, keepdims=True)  # [H, q_len, 1]

        # 数值稳定 p
        p = np.exp(scores - m_blk)  # [H, q_len, block_size]
        l_blk = p.sum(axis=-1, keepdims=True)  # [H, q_len, 1]
        y_blk = np.einsum("hqk,kd->hqd", p, V_blk)  # [H, q_len, d]

        # online combine: 经典 FlashAttention 更新
        m_new = np.maximum(out_m, m_blk)
        alpha_old = np.exp(out_m - m_new)
        alpha_new = np.exp(m_blk - m_new)
        out_l = out_l * alpha_old + l_blk * alpha_new
        out_y = out_y * alpha_old + y_blk * alpha_new
        out_m = m_new

    return NumpyAttnStats(m=out_m, l=out_l, y=out_y)


def shard_blockwise(
    Q: np.ndarray,
    K_blocks: list[np.ndarray],
    V_blocks: list[np.ndarray],
    H: int,
    num_shards: int,
) -> NumpyAttnStats:
    """多 shard 并行：每个 shard 算 (m,l,y)，再 merge。"""
    n_blocks = len(K_blocks)
    # 简单 round-robin 分 shard（每个 shard 拿 num_blocks/num_shards 个 block）
    shard_indices = [list(range(i, n_blocks, num_shards)) for i in range(num_shards)]

    stats_list = []
    for sidx in shard_indices:
        K_shard = [K_blocks[i] for i in sidx]
        V_shard = [V_blocks[i] for i in sidx]
        s = flash_attn_blockwise(Q, K_shard, V_shard, H)
        stats_list.append(s)

    return numpy_merge_stats_list(stats_list)


# ============== 单组实验 ==============


def run_exp2_one(
    q_len: int,
    kv_len: int,
    num_heads: int = 4,
    d: int = 64,
    block_size: int = 64,
    num_shards: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """单组 (q_len, kv_len, num_shards) 配置。"""
    num_blocks = kv_len // block_size
    assert num_blocks * block_size == kv_len

    gen = np.random.default_rng(seed)

    # 构造 Q/K/V（per-head d=64，所以这里 d 是 per-head 的 dim）
    Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
    K_blocks = [
        (gen.standard_normal((block_size, d)) * 0.5).astype(np.float32)
        for _ in range(num_blocks)
    ]
    V_blocks = [
        (gen.standard_normal((block_size, d)) * 0.5).astype(np.float32)
        for _ in range(num_blocks)
    ]

    # === Ground truth: 单次全 attention（用于 err 计算）===
    K_all = np.concatenate(K_blocks, axis=0)  # [kv_len, d]
    V_all = np.concatenate(V_blocks, axis=0)  # [kv_len, d]
    # ground truth 是 single-head 的（_ground_truth 不接受 H 维），用第一个 head 作为 reference
    gt = ground_truth(Q, K_all, V_all)  # [q_len, d]

    # === A. KV tensor 路径（单 head，per head 都传）===
    bytes_a = num_heads * bytes_kv_path_pertensor(K_all, V_all)

    # 模拟 A 路径：每个 head 都跑全 attention
    out_a_list = []
    for h in range(num_heads):
        out_a_list.append(ground_truth(Q, K_all, V_all))
    out_a = np.stack(out_a_list, axis=0)  # [H, q_len, d]

    # === B. (m,l,y) 路径（多 head，per-shard 算再 merge）===
    merged = shard_blockwise(Q, K_blocks, V_blocks, num_heads, num_shards)
    out_b = merged.finalize()  # [H, q_len, d]
    bytes_b = bytes_stats_path(Q, num_heads=num_heads)

    # === C. ExactLocal（单 head，本地 baseline）===
    # C 路径的 baseline = single-head exact attention
    stats_c = flash_attn_blockwise(Q, K_blocks, V_blocks, 1)
    out_c = stats_c.finalize().squeeze(0)  # [q_len, d]
    bytes_c = 0

    # --- 指标 ---
    # err 跟 ground truth 比（取第一个 head）
    err_a = float(np.abs(out_a[0] - gt).mean())
    err_b = float(np.abs(out_b[0] - gt).mean())
    err_c = float(np.abs(out_c - gt).mean())
    ratio = bytes_a / max(bytes_b, 1)

    if verbose:
        print(
            f"  q={q_len:>4} kv={kv_len:>5} S={num_shards:>2}  "
            f"A: {bytes_a:>10,}B err={err_a:.2e}  "
            f"B: {bytes_b:>7,}B err={err_b:.2e}  "
            f"C: {bytes_c:>3}B err={err_c:.2e}  "
            f"saving={ratio:>8.1f}x"
        )

    return {
        "q_len": q_len,
        "kv_len": kv_len,
        "num_shards": num_shards,
        "num_heads": num_heads,
        "d_per_head": d,
        "block_size": block_size,
        "bytes_a": bytes_a,
        "bytes_b": bytes_b,
        "bytes_c": bytes_c,
        "err_a": err_a,
        "err_b": err_b,
        "err_c": err_c,
        "ratio": ratio,
    }


def bytes_kv_path_pertensor(K: np.ndarray, V: np.ndarray) -> int:
    """单次 KV tensor 路径的字节数（per head）"""
    kv_len, d = K.shape
    return 2 * kv_len * d * 4  # 2 (K+V) * d * fp32 bytes


# ============== Sweep ==============


def run_exp2_sweep(verbose: bool = True) -> list[dict]:
    results = []
    q_lens = [1, 4, 16, 64, 256]
    kv_lens = [1024, 4096, 16384]
    shard_grid = [1, 4, 16]

    for q_len in q_lens:
        for kv_len in kv_lens:
            for num_shards in shard_grid:
                r = run_exp2_one(
                    q_len=q_len,
                    kv_len=kv_len,
                    num_heads=4,
                    d=64,
                    block_size=64,
                    num_shards=num_shards,
                    seed=0,
                    verbose=verbose,
                )
                results.append(r)
    return results


# ============== 汇总 ==============


def summarize(results: list[dict]) -> None:
    """打印分组 summary。"""
    print()
    print("=" * 78)
    print("Exp2 grouped summary (multi-head H=4)")
    print("=" * 78)
    # 按 (q_len, num_shards) 分组，看 ratio 怎么随 num_shards 变
    grid: dict[tuple, list[dict]] = {}
    for r in results:
        grid.setdefault((r["q_len"], r["num_shards"]), []).append(r)

    print(f"{'q_len':>6} {'shards':>7} {'kv=1024':>12} {'kv=4096':>12} {'kv=16384':>12}")
    for q_len in sorted({r["q_len"] for r in results}):
        for ns in [1, 4, 16]:
            row = grid.get((q_len, ns), [])
            if not row:
                continue
            ratios = {r["kv_len"]: r["ratio"] for r in row}
            print(
                f"{q_len:>6} {ns:>7} {ratios.get(1024, 0):>11.1f}x "
                f"{ratios.get(4096, 0):>11.1f}x {ratios.get(16384, 0):>11.1f}x"
            )

    # 按 q_len 看 max ratio
    print()
    print("Peak ratio at kv_len=16384:")
    for q_len in sorted({r["q_len"] for r in results}):
        max_r = max(
            (r["ratio"] for r in results if r["q_len"] == q_len and r["kv_len"] == 16384),
            default=0,
        )
        print(f"  q_len={q_len:>4}: {max_r:>8.1f}x")

    print()
    print("Fidelity (max err_B across all configs):")
    max_err = max(r["err_b"] for r in results)
    print(f"  max err_B = {max_err:.2e}  [target < 1e-3]")


if __name__ == "__main__":
    print("Exp2 sweep: multi-head (H=4) + q_len=1 decoding + wavesize")
    print("=" * 78)
    results = run_exp2_sweep()
    summarize(results)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "results", "exp2_multi_head.json",
    )
    out_path = os.path.normpath(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nsaved to {out_path}")
