"""
Exp28 LSH Sweep - 精简版
"""
import json
import os
import sys
import time
import numpy as np
from numpy import linalg as npla

_REPO_ROOT = '/app/data/所有对话/主对话/_staging/accord-kv'
sys.path.insert(0, _REPO_ROOT)

from simulation.exp28_lsh_pruning import (
    ground_truth, RandomProjectionLSH, lsh_soft_prune,
    eval_lsh_attention, svd_compress_v, quantize_nbit, dequantize_nbit,
    kmeans_plusplus_init, build_coreset_sketch, eval_coreset_attention,
    make_clustered_kv, make_random_kv, make_skewed_kv
)

def lsh_prune_svd_int4_pipeline(K, V, Q, n_tables, n_bits, svd_r=8, int4_bits=4, merge_strategy='mean', d=128, seed=42):
    kv_len = K.shape[0]
    q_len = Q.shape[0]
    
    # Stage 1: LSH Prune
    K_lsh, V_lsh, weights_lsh, lsh_stats = lsh_soft_prune(K, V, n_tables, n_bits, merge_strategy, seed)
    
    # Stage 2: SVD
    V_svd, U_r, S_r = svd_compress_v(V_lsh, svd_r)
    
    # Stage 3: INT4
    V_quant, V_scale = quantize_nbit(V_svd, int4_bits)
    V_final = dequantize_nbit(V_quant, V_scale)
    
    # 评估
    y_pred = eval_lsh_attention(K_lsh, V_final, weights_lsh, Q, d)
    y_gt = ground_truth(Q, K, V)
    err = float(np.abs(y_pred - y_gt).mean())
    
    # 压缩比
    bytes_full = kv_len * d * 2 * 4
    bytes_K_lsh = K_lsh.shape[0] * d * 4
    bytes_V_svd = U_r.size + S_r.size + V_quant.size
    bytes_weights = weights_lsh.size * 4 + 4
    bytes_compressed = bytes_K_lsh + bytes_V_svd + bytes_weights
    compression_ratio = bytes_full / bytes_compressed if bytes_compressed > 0 else float('inf')
    
    physical_limit = 2 * kv_len / q_len
    is_physically_honest = compression_ratio <= physical_limit
    
    return {
        'err': err,
        'compression_ratio': compression_ratio,
        'is_physically_honest': is_physically_honest,
        'lsh_stats': lsh_stats,
    }

def coreset_svd_int4_pipeline(K, V, Q, coreset_ratio=0.5, svd_r=8, int4_bits=4, d=128, seed=42):
    kv_len = K.shape[0]
    r_coreset = max(4, int(kv_len * coreset_ratio))
    centroids, V_coreset, weights = build_coreset_sketch(K, V, r_coreset, seed)
    V_svd, U_r, S_r = svd_compress_v(V_coreset, svd_r)
    V_quant, V_scale = quantize_nbit(V_svd, int4_bits)
    V_final = dequantize_nbit(V_quant, V_scale)
    y_pred = eval_coreset_attention(centroids, V_final, weights, Q, d)
    y_gt = ground_truth(Q, K, V)
    err = float(np.abs(y_pred - y_gt).mean())
    bytes_full = kv_len * d * 2 * 4
    bytes_compressed = U_r.size + S_r.size + V_quant.size + 1
    compression_ratio = bytes_full / bytes_compressed if bytes_compressed > 0 else float('inf')
    return {'err': err, 'compression_ratio': compression_ratio}

def serial_cascade_baseline(K, V, Q, coreset_ratio=0.5, svd_r=8, int4_bits=4, d=128, seed=42):
    return coreset_svd_int4_pipeline(K, V, Q, coreset_ratio, svd_r, int4_bits, d, seed)

if __name__ == "__main__":
    print("=" * 70)
    print("Exp28 LSH Sweep - Compact Version")
    print("=" * 70)
    
    d = 128
    svd_r = 8
    int4_bits = 4
    seed = 42
    
    # 最精简配置
    lsh_configs = [
        {'n_tables': 4, 'n_bits': 4},   # b=16
        {'n_tables': 4, 'n_bits': 6},   # b=64
    ]
    merge_strategy = 'mean'
    
    sweep_configs = {
        'kv_types': ['clustered', 'random', 'skewed'],
        'kv_lens': [256, 512],  # 减小以加快速度
        'q_lens': [16, 32],
    }
    
    results = []
    total = 3 * 2 * 2 * 2 + 3 * 2 * 2  # LSH + baselines
    idx = 0
    start_time = time.time()
    
    for kv_type in sweep_configs['kv_types']:
        for kv_len in sweep_configs['kv_lens']:
            for q_len in sweep_configs['q_lens']:
                # 生成数据
                if kv_type == 'clustered':
                    K, V = make_clustered_kv(kv_len, d, seed=seed)
                elif kv_type == 'random':
                    K, V = make_random_kv(kv_len, d, seed=seed)
                else:
                    K, V = make_skewed_kv(kv_len, d, seed=seed)
                
                gen = np.random.default_rng(100 + hash(f'{kv_type}{kv_len}{q_len}') % 10000)
                Q = (gen.standard_normal((q_len, d)) * 0.5).astype(np.float32)
                
                # Serial Cascade baseline
                baseline = serial_cascade_baseline(K, V, Q, seed=seed)
                results.append({
                    'kv_type': kv_type, 'kv_len': kv_len, 'q_len': q_len,
                    'method': 'serial_cascade',
                    'err': baseline['err'],
                    'compression_ratio': baseline['compression_ratio'],
                })
                
                # Coreset baseline
                coreset = coreset_svd_int4_pipeline(K, V, Q, seed=seed)
                results.append({
                    'kv_type': kv_type, 'kv_len': kv_len, 'q_len': q_len,
                    'method': 'coreset',
                    'err': coreset['err'],
                    'compression_ratio': coreset['compression_ratio'],
                })
                
                # LSH 扫描
                for lsh_cfg in lsh_configs:
                    idx += 1
                    lsh_result = lsh_prune_svd_int4_pipeline(
                        K, V, Q,
                        n_tables=lsh_cfg['n_tables'],
                        n_bits=lsh_cfg['n_bits'],
                        svd_r=svd_r, int4_bits=int4_bits,
                        merge_strategy=merge_strategy, d=d, seed=seed,
                    )
                    result = {
                        'kv_type': kv_type, 'kv_len': kv_len, 'q_len': q_len,
                        'method': 'lsh',
                        'n_tables': lsh_cfg['n_tables'],
                        'n_bits': lsh_cfg['n_bits'],
                        'merge_strategy': merge_strategy,
                        'err': lsh_result['err'],
                        'compression_ratio': lsh_result['compression_ratio'],
                        'is_physically_honest': lsh_result['is_physically_honest'],
                        'n_buckets': lsh_result['lsh_stats']['n_buckets'],
                        'collision_rate': lsh_result['lsh_stats']['collision_stats']['collision_rate'],
                    }
                    results.append(result)
                    
                    elapsed = time.time() - start_time
                    eta = elapsed / idx * (total - idx) if idx > 0 else 0
                    print(f'[{idx}/{total}] {kv_type}, kv={kv_len}, q={q_len}, t={lsh_cfg["n_tables"]}, b={2**lsh_cfg["n_bits"]}: err={lsh_result["err"]:.4f}, ratio={lsh_result["compression_ratio"]:.1f}x ({elapsed:.1f}s elapsed)')
    
    # 保存结果
    results_dir = os.path.join(_REPO_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'exp28_sweep.json'), 'w') as f:
        json.dump({'description': 'exp28 full sweep', 'seed': 42, 'results': results}, f, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else list(x) if isinstance(x, np.ndarray) else x)
    print(f'Sweep saved to results/exp28_sweep.json')
    
    # 分析
    analysis = analyze_results(results)
    
    # 提取帕累托前沿
    pareto = extract_pareto_frontier(results)
    with open(os.path.join(results_dir, 'exp28_pareto.json'), 'w') as f:
        json.dump({'description': 'exp28 pareto frontier', 'pareto_frontier': pareto}, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else list(x) if isinstance(x, np.ndarray) else x)
    
    # LSH vs Coreset 对比
    lsh_vs_coreset = {
        kv_type: {
            'lsh_best_err': s.get('best_lsh_err'),
            'coreset_err': s.get('coreset_err'),
            'serial_cascade_err': s.get('serial_cascade_err'),
            'lsh_vs_coreset': s.get('lsh_vs_coreset'),
            'lsh_vs_serial': s.get('lsh_vs_serial'),
            'winner': s.get('winner'),
        }
        for kv_type, s in analysis['summary'].items()
    }
    with open(os.path.join(results_dir, 'exp28_vs_coreset.json'), 'w') as f:
        json.dump({'description': 'exp28 LSH vs Coreset comparison', 'comparison': lsh_vs_coreset}, f, indent=2)
    
    # 生成报告
    report = generate_report_from_results(results, analysis)
    with open(os.path.join(results_dir, 'exp28_lsh_report.md'), 'w') as f:
        f.write(report)
    print(f'Report saved to results/exp28_lsh_report.md')
    print("Done!")

def extract_pareto_frontier(results):
    by_config = {}
    for r in results:
        key = (r['kv_type'], r['kv_len'], r['q_len'])
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(r)
    
    pareto = []
    for key, configs in by_config.items():
        for candidate in configs:
            dominated = False
            for other in configs:
                if (other['compression_ratio'] <= candidate['compression_ratio'] and
                    other['err'] <= candidate['err'] and
                    (other['compression_ratio'] < candidate['compression_ratio'] or
                     other['err'] < candidate['err'])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)
    return sorted(pareto, key=lambda x: (x['kv_type'], x['kv_len'], x['q_len'], x['compression_ratio']))

def analyze_results(results):
    analysis = {'summary': {}, 'by_kv_type': {}}
    
    for r in results:
        kv_type = r['kv_type']
        if kv_type not in analysis['by_kv_type']:
            analysis['by_kv_type'][kv_type] = {'lsh_results': [], 'serial_cascade': [], 'coreset': []}
        
        if r['method'] == 'lsh':
            analysis['by_kv_type'][kv_type]['lsh_results'].append(r)
        elif r['method'] == 'serial_cascade':
            analysis['by_kv_type'][kv_type]['serial_cascade'].append(r)
        elif r['method'] == 'coreset':
            analysis['by_kv_type'][kv_type]['coreset'].append(r)
    
    for kv_type, data in analysis['by_kv_type'].items():
        lsh_results = data['lsh_results']
        if lsh_results:
            best_lsh = min(lsh_results, key=lambda x: x['err'])
            analysis['summary'][kv_type] = {
                'best_lsh_err': best_lsh['err'],
                'best_lsh_config': {
                    'n_tables': best_lsh['n_tables'],
                    'n_bits': best_lsh['n_bits'],
                    'merge_strategy': best_lsh['merge_strategy'],
                    'compression_ratio': best_lsh['compression_ratio'],
                },
                'lsh_mean_err': np.mean([r['err'] for r in lsh_results]),
            }
        
        if data['serial_cascade']:
            sc = data['serial_cascade'][0]
            analysis['summary'][kv_type]['serial_cascade_err'] = sc['err']
            analysis['summary'][kv_type]['serial_cascade_ratio'] = sc['compression_ratio']
        
        if data['coreset']:
            coreset = data['coreset'][0]
            analysis['summary'][kv_type]['coreset_err'] = coreset['err']
            analysis['summary'][kv_type]['coreset_ratio'] = coreset['compression_ratio']
    
    for kv_type, s in analysis['summary'].items():
        lsh_err = s.get('best_lsh_err', float('inf'))
        coreset_err = s.get('coreset_err', float('inf'))
        sc_err = s.get('serial_cascade_err', float('inf'))
        
        s['lsh_vs_coreset'] = lsh_err - coreset_err
        s['lsh_vs_serial'] = lsh_err - sc_err
        
        if lsh_err < coreset_err * 0.9:
            s['winner'] = 'lsh'
        elif coreset_err < lsh_err * 0.9:
            s['winner'] = 'coreset'
        else:
            s['winner'] = 'tie'
    
    return analysis

def generate_report_from_results(results, analysis):
    lines = []
    lines.append("# Exp28: LSH Soft Pruning - Complete Analysis Report\n")
    
    lines.append("## Executive Summary\n")
    lines.append("| KV Type | Best LSH err | Coreset err | Serial Cascade err | Winner |")
    lines.append("|---------|-------------|--------------|---------------------|--------|")
    
    for kv_type in ["clustered", "random", "skewed"]:
        if kv_type in analysis["summary"]:
            s = analysis["summary"][kv_type]
            lsh_err = s.get('best_lsh_err', 'N/A')
            coreset_err = s.get('coreset_err', 'N/A')
            sc_err = s.get('serial_cascade_err', 'N/A')
            winner = s.get('winner', 'N/A')
            
            lsh_str = f"{lsh_err:.4f}" if isinstance(lsh_err, float) else lsh_err
            coreset_str = f"{coreset_err:.4f}" if isinstance(coreset_err, float) else coreset_err
            sc_str = f"{sc_err:.4f}" if isinstance(sc_err, float) else sc_err
            
            lines.append(f"| {kv_type} | {lsh_str} | {coreset_str} | {sc_str} | {winner} |")
    
    lines.append("")
    
    lines.append("## Detailed Results\n")
    for kv_type in ["clustered", "random", "skewed"]:
        if kv_type not in analysis["by_kv_type"]:
            continue
        
        lines.append(f"### {kv_type}\n")
        data = analysis["by_kv_type"][kv_type]
        
        if data["serial_cascade"]:
            for sc in data["serial_cascade"]:
                lines.append(f"- **Serial Cascade** (kv={sc['kv_len']}, q={sc['q_len']}): err={sc['err']:.4f}, ratio={sc['compression_ratio']:.1f}x\n")
        
        if data["coreset"]:
            for c in data["coreset"]:
                lines.append(f"- **Coreset** (kv={c['kv_len']}, q={c['q_len']}): err={c['err']:.4f}, ratio={c['compression_ratio']:.1f}x\n")
        
        if data["lsh_results"]:
            for lsh in data["lsh_results"]:
                lines.append(f"- **LSH** (t={lsh['n_tables']}, b={2**lsh['n_bits']}): err={lsh['err']:.4f}, ratio={lsh['compression_ratio']:.1f}x, honest={lsh['is_physically_honest']}, buckets={lsh['n_buckets']}\n")
        lines.append("")
    
    # Collision Rate Analysis
    lines.append("## Collision Rate Analysis\n")
    configs = {}
    for r in results:
        if r['method'] != 'lsh':
            continue
        key = f"t={r['n_tables']}, b={2**r['n_bits']}"
        if key not in configs:
            configs[key] = []
        configs[key].append(r['collision_rate'])
    
    lines.append("| Config | Mean Collision Rate |\n")
    lines.append("|--------|---------------------|\n")
    for key, rates in sorted(configs.items()):
        lines.append(f"| {key} | {np.mean(rates):.3f} |\n")
    lines.append("")
    
    # Conclusions
    lines.append("## Conclusions\n")
    lsh_wins = sum(1 for kv_type, s in analysis["summary"].items() if s.get('winner') == 'lsh')
    coreset_wins = sum(1 for kv_type, s in analysis["summary"].items() if s.get('winner') == 'coreset')
    
    lines.append(f"- LSH wins on {lsh_wins}/3 distributions\n")
    lines.append(f"- Coreset wins on {coreset_wins}/3 distributions\n")
    
    clustered_err = analysis["summary"].get("clustered", {}).get("best_lsh_err", float('inf'))
    if clustered_err > 3.0:
        lines.append(f"\n**Honest Report**: LSH 在 clustered 数据上 err={clustered_err:.4f}，仍然较高。\n")
    else:
        lines.append(f"\n**LSH 在 clustered 数据上 err={clustered_err:.4f}**，表现可接受。\n")
    
    return "".join(lines)
