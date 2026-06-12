#!/usr/bin/env python3
"""
EXP21: Full Pipeline Sweep (FLP -> SVD -> INT4 vs Coreset -> SVD -> INT4)
"""

import json
import time
import os
import sys

_REPO_ROOT = '/app/data/所有对话/主对话/_staging/accord-kv'
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from simulation.exp21_facility_location import (
    make_random_kv, make_clustered_kv, make_skewed_kv,
    build_full_pipeline, eval_full_pipeline,
    build_coreset_sketch, eval_coreset_sketch, ground_truth,
    svd_compress, svd_reconstruct, quantize_nbit, dequantize_nbit
)
import numpy as np

def main():
    print("=" * 60)
    print("EXP21: Full Pipeline Sweep")
    print("=" * 60)

    d = 128
    results = []
    start_time = time.time()

    kv_types = ["clustered", "random", "skewed"]
    kv_lens = [1024, 4096]
    q_lens = [16, 64]
    flp_ratios_1024 = [0.0625, 0.125, 0.25]
    flp_ratios_4096 = [0.0625, 0.125]
    svd_r_values = [4, 8]
    int4_bits_values = [4, 8]

    total_configs = 0
    for kv_len in kv_lens:
        flp_ratios = flp_ratios_4096 if kv_len == 4096 else flp_ratios_1024
        total_configs += len(kv_types) * len(q_lens) * len(flp_ratios) * len(svd_r_values) * len(int4_bits_values)
    print(f"Total configs: {total_configs}")

    config_idx = 0
    seed = 42

    for kv_type in kv_types:
        for kv_len in kv_lens:
            flp_ratios = flp_ratios_4096 if kv_len == 4096 else flp_ratios_1024
            
            if kv_type == "clustered":
                K, V = make_clustered_kv(kv_len, d, seed=seed)
            elif kv_type == "random":
                K, V = make_random_kv(kv_len, d, seed=seed)
            else:
                K, V = make_skewed_kv(kv_len, d, seed=seed)
            
            for q_len in q_lens:
                Q = np.random.default_rng(seed + 100).standard_normal((q_len, d)).astype(np.float32) * 0.5
                gt = ground_truth(Q, K, V)
                bytes_full = 2 * kv_len * d * 4
                
                for flp_ratio in flp_ratios:
                    r = max(4, int(kv_len * flp_ratio))
                    
                    for svd_r in svd_r_values:
                        for bits in int4_bits_values:
                            config_idx += 1
                            
                            t0 = time.time()
                            try:
                                pipeline, info = build_full_pipeline(K, V, r, svd_r, bits, seed=seed, verbose=False)
                                out_flp = eval_full_pipeline(pipeline, Q, d)
                                err_flp = float(np.abs(out_flp - gt).mean())
                                bytes_flp = pipeline.bytes_size()
                                
                                coreset_cent, coreset_val, coreset_w, _ = build_coreset_sketch(K, V, r, seed=seed)
                                U_c, S_c, Vt_c = svd_compress(coreset_val, svd_r)
                                V_c_svd = svd_reconstruct(U_c, S_c, Vt_c)
                                V_c_quant, V_c_scales = quantize_nbit(V_c_svd, bits)
                                V_c_deq = dequantize_nbit(V_c_quant, V_c_scales)
                                stats_baseline = eval_coreset_sketch(coreset_cent, V_c_deq, coreset_w, Q, d)
                                out_coreset = stats_baseline.finalize().squeeze(0)
                                err_coreset = float(np.abs(out_coreset - gt).mean())
                                bytes_coreset = (coreset_cent.size * 4 + coreset_val.size * 4 + coreset_w.size * 4 + 
                                                 U_c.size * 4 + S_c.size * 4 + V_c_quant.size * (bits/8) + V_c_scales.size * 4)
                                
                                pipeline_time = time.time() - t0
                                ratio_flp = bytes_flp / bytes_full
                                ratio_coreset = bytes_coreset / bytes_full
                                winner = "FLP" if err_flp < err_coreset else "Coreset"
                                
                                results.append({
                                    "kv_type": kv_type, "kv_len": kv_len, "q_len": q_len,
                                    "flp_r": r, "svd_r": svd_r, "int4_bits": bits,
                                    "err_flp": err_flp, "err_coreset": err_coreset,
                                    "bytes_flp": int(bytes_flp), "bytes_coreset": float(bytes_coreset),
                                    "ratio_flp": ratio_flp, "ratio_coreset": ratio_coreset,
                                    "time_pipeline": pipeline_time, "winner": winner,
                                    "svd_error": info["svd_error"], "quant_error": info["quant_error"],
                                })
                            except Exception as e:
                                print(f"Error at config {config_idx}: {e}")
                                continue
                            
                            if config_idx % 10 == 0:
                                elapsed = time.time() - start_time
                                print(f"Progress: {config_idx}/{total_configs} ({elapsed:.1f}s)")

    elapsed_total = time.time() - start_time
    print(f"\nComplete: {len(results)} configs in {elapsed_total:.1f}s")

    by_kv_type = {}
    for kv_type in kv_types:
        subset = [r for r in results if r["kv_type"] == kv_type]
        if subset:
            flp_wins = sum(1 for r in subset if r["winner"] == "FLP")
            by_kv_type[kv_type] = {
                "count": len(subset), "flp_wins": flp_wins,
                "coreset_wins": len(subset) - flp_wins,
                "win_rate_flp": flp_wins / len(subset),
                "avg_err_flp": float(np.mean([r["err_flp"] for r in subset])),
                "avg_err_coreset": float(np.mean([r["err_coreset"] for r in subset])),
            }
            print(f"{kv_type}: FLP {flp_wins}/{len(subset)} wins")

    violations = [r for r in results if r["ratio_flp"] > 2.0 or r["ratio_coreset"] > 2.0]
    print(f"\nPhysical honesty violations: {len(violations)}")

    output = {
        "results": results, "by_kv_type": by_kv_type,
        "total_time": elapsed_total, "config_count": len(results),
    }
    output_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp21_pipeline_sweep.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
