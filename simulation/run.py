"""
仿真入口 — 跑 Phase 1 所有实验。

当前实验：
- Exp1: Fidelity vs Bandwidth (单 head, 1D sweep)
- Exp2: Multi-head + q_len=1 decoding + wavesize sweep (3D sweep)

用法（在 accord-kv/ 下）:
    python -m simulation.run
    python -m simulation.run --exp exp1
    python -m simulation.run --exp exp2
    python -m simulation.run --exp all
    python -m simulation.run --json results/exp1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def run_exp1(json_path: str | None = None) -> list[dict]:
    from simulation.exp1_fidelity_vs_bandwidth import run_exp1_sweep
    print("=" * 78)
    print("Exp1: Fidelity vs Bandwidth (q_len x kv_len sweep)")
    print("=" * 78)
    results = run_exp1_sweep()

    # 总结表
    print()
    print("=" * 78)
    print("Summary table")
    print("=" * 78)
    print(
        f"{'q_len':>6} {'kv_len':>6} {'bytes_A':>11} {'bytes_B':>9} "
        f"{'saving':>9} {'err_A':>10} {'err_B':>10} {'err_C':>10}"
    )
    for r in results:
        print(
            f"{r['q_len']:>6} {r['kv_len']:>6} {r['bytes_a']:>11,} "
            f"{r['bytes_b']:>9,} {r['ratio']:>8.1f}x "
            f"{r['err_a']:>10.2e} {r['err_b']:>10.2e} {r['err_c']:>10.2e}"
        )

    # 判定（v3 — 按 q_len 分级，因为 ratio 上限 ∝ kv_len/q_len）
    print()
    print("=" * 78)
    print("Success criteria check (v3 — q_len-graded)")
    print("=" * 78)
    min_ratio = min(r["ratio"] for r in results)
    max_err_b = max(r["err_b"] for r in results)
    max_err_a = max(r["err_a"] for r in results)
    max_err_c = max(r["err_c"] for r in results)

    # 按 q_len 分组
    r_by_q: dict[int, list[dict]] = {}
    for r in results:
        r_by_q.setdefault(r["q_len"], []).append(r)
    min_r_16 = min(r["ratio"] for r in r_by_q.get(16, []))
    min_r_64 = min(r["ratio"] for r in r_by_q.get(64, []))
    min_r_256 = min(r["ratio"] for r in r_by_q.get(256, []))

    print(f"min ratio (all 9 configs)         = {min_ratio:>7.1f}x   [target ≥ 7.0x — physical floor 2·kv_len/q_len]")
    print(f"min ratio at q_len=16            = {min_r_16:>7.1f}x   [target ≥ 120x]")
    print(f"min ratio at q_len=64            = {min_r_64:>7.1f}x   [target ≥ 30x]")
    print(f"min ratio at q_len=256           = {min_r_256:>7.1f}x   [target ≥ 7.0x]")
    print(f"max err_B                        = {max_err_b:.2e}   [target < 1e-3]")
    print(f"max err_A (sanity)               = {max_err_a:.2e}   [target < 1e-5]")
    print(f"max err_C (sanity)               = {max_err_c:.2e}   [target < 1e-5]")

    pass_min = min_ratio >= 7.0
    pass_16 = min_r_16 >= 120.0
    pass_64 = min_r_64 >= 30.0
    pass_256 = min_r_256 >= 7.0
    pass_err_b = max_err_b < 1e-3
    pass_sanity = max_err_a < 1e-5 and max_err_c < 1e-5
    overall = pass_min and pass_16 and pass_64 and pass_256 and pass_err_b and pass_sanity
    print()
    print(f"  worst-case ≥ 7x        : {'✓' if pass_min else '✗'}")
    print(f"  q_len=16  ≥ 120x       : {'✓' if pass_16 else '✗'}")
    print(f"  q_len=64  ≥ 30x        : {'✓' if pass_64 else '✗'}")
    print(f"  q_len=256 ≥ 7x         : {'✓' if pass_256 else '✗'}")
    print(f"  err_B < 1e-3           : {'✓' if pass_err_b else '✗'}")
    print(f"  sanity (A,C ≈ 0)       : {'✓' if pass_sanity else '✗'}")
    print()
    print(f"OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}")
    print()
    print("Note: ratio 上限 ∝ kv_len/q_len (d 大时 ratio ≈ 2·kv_len/q_len).")
    print("      AVL sweet spot = small q_len (decoding / sparse query) + large kv_len.")

    if json_path:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"saved to {json_path}")

    return results


def run_exp2(json_path: str | None = None) -> list[dict]:
    from simulation.exp2_multi_head import run_exp2_sweep, summarize

    print("=" * 78)
    print("Exp2: Multi-head (H=4) + q_len=1 decoding + wavesize sweep")
    print("=" * 78)
    results = run_exp2_sweep()
    summarize(results)

    # 判定
    print()
    print("=" * 78)
    print("Success criteria check (Exp2)")
    print("=" * 78)
    max_err_b = max(r["err_b"] for r in results)
    r_q1_16384 = next(
        (r for r in results if r["q_len"] == 1 and r["kv_len"] == 16384), None
    )
    r_q1_1024 = next(
        (r for r in results if r["q_len"] == 1 and r["kv_len"] == 1024), None
    )
    r_q256_1024 = next(
        (r for r in results if r["q_len"] == 256 and r["kv_len"] == 1024), None
    )

    print(f"max err_B (all 45 configs)      = {max_err_b:.2e}   [target < 1e-3]")
    if r_q1_1024:
        print(f"q=1, kv=1024 ratio             = {r_q1_1024['ratio']:>7.1f}x   [decoding baseline]")
    if r_q1_16384:
        print(f"q=1, kv=16384 ratio            = {r_q1_16384['ratio']:>7.1f}x   [long-context decoding]")
    if r_q256_1024:
        print(f"q=256, kv=1024 ratio           = {r_q256_1024['ratio']:>7.1f}x   [prefill worst case]")

    pass_err_b = max_err_b < 1e-3
    pass_q1 = r_q1_16384["ratio"] > 1000.0 if r_q1_16384 else False
    pass_floor = r_q256_1024["ratio"] >= 7.0 if r_q256_1024 else False
    overall = pass_err_b and pass_q1 and pass_floor

    print()
    print(f"  err_B < 1e-3         : {'✓' if pass_err_b else '✗'}")
    print(f"  q=1/kv=16384 > 1000x : {'✓' if pass_q1 else '✗'}")
    print(f"  q=256/kv=1024 ≥ 7x   : {'✓' if pass_floor else '✗'}")
    print()
    print(f"OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}")

    if json_path:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"saved to {json_path}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exp",
        default="all",
        choices=["exp1", "exp2", "all"],
        help="which experiment (default: all)",
    )
    p.add_argument("--json", default=None, help="optional path to dump JSON results")
    args = p.parse_args()

    if args.exp == "exp1":
        run_exp1(args.json)
    elif args.exp == "exp2":
        run_exp2(args.json)
    elif args.exp == "all":
        run_exp1()
        print()
        run_exp2()
    else:
        raise SystemExit(f"unknown exp: {args.exp}")


if __name__ == "__main__":
    main()
