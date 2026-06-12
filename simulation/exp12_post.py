#!/usr/bin/env python3
"""Generate physical consistency and report from saved results."""
import json, os, sys
import numpy as np

REPO_ROOT = '/app/data/所有对话/主对话/_staging/accord-kv'
sys.path.insert(0, REPO_ROOT)
from simulation.exp12_wavelet_attention import (
    make_clustered_kv, make_random_kv, make_skewed_kv, make_smooth_kv,
    wavelet_compressed_kv_attention, wavelet_full_idwt_baseline,
    ground_truth,
)

def run_phys_and_report():
    out_dir = os.path.join(REPO_ROOT, 'results')
    
    # Load existing results
    with open(os.path.join(out_dir, 'exp12_wavelet_sweep.json')) as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} results")
    
    # Physical consistency
    phys = {}
    make_fns = [
        ('smooth', make_smooth_kv),
        ('clustered', make_clustered_kv),
        ('random', make_random_kv),
        ('skewed', make_skewed_kv),
    ]
    for sig_type, make_fn in make_fns:
        K, V = make_fn(4096, 128, seed=0)
        Q = (np.random.default_rng(1000).standard_normal((16, 128)) * 0.5).astype(np.float32)
        gt = ground_truth(Q, K, V)
        phys[sig_type] = {}
        for level in [1, 2, 3, 4]:
            y, _, _ = wavelet_compressed_kv_attention(Q, K, V, 'db4', level, mode='zero')
            err = float(np.abs(y - gt).mean())
            idwt = wavelet_full_idwt_baseline(Q, K, V, 'db4', level)
            err_idwt = float(np.abs(idwt - gt).mean())
            phys[sig_type][f'level_{level}'] = {
                'err': err, 'err_idwt': err_idwt, 'compression': 2**level, 'idwt_ok': err_idwt < 1e-5,
            }
            print(f"  phys {sig_type} L={level}: err={err:.4f} idwt={err_idwt:.2e}")
    
    with open(os.path.join(out_dir, 'exp12_physical_consistency.json'), 'w') as f:
        json.dump(phys, f, indent=2)
    print("Saved physical_consistency.json")
    
    # Pareto
    pareto = []
    for r in results:
        dominated = False
        for p in pareto:
            if (p['bytes_wavelet'] <= r['bytes_wavelet'] and 
                p['err_wavelet'] <= r['err_wavelet'] and
                (p['bytes_wavelet'] < r['bytes_wavelet'] or p['err_wavelet'] < r['err_wavelet'])):
                dominated = True; break
        if not dominated:
            pareto = [p for p in pareto if not (
                r['bytes_wavelet'] <= p['bytes_wavelet'] and
                r['err_wavelet'] <= p['err_wavelet'] and
                (r['bytes_wavelet'] < p['bytes_wavelet'] or r['err_wavelet'] < p['err_wavelet']))]
            pareto.append(r)
    pareto = sorted(pareto, key=lambda x: x['bytes_wavelet'])
    
    with open(os.path.join(out_dir, 'exp12_pareto.json'), 'w') as f:
        json.dump(pareto, f, indent=2)
    print(f"Saved pareto.json ({len(pareto)} points)")
    
    # vs_core summary
    vs_core_list = []
    for r in results:
        vs_core_list.append({
            'kv_type': r['kv_type'], 'kv_len': r['kv_len'], 'level': r['level'],
            'wavelet_win_rate': float(r['wavelet_wins_vs_core']),
            'mean_wavelet_err': r['err_wavelet'],
            'mean_coreset_err': r['err_coreset'],
            'wavelet_better': r['err_wavelet'] < r['err_coreset'],
        })
    vs_core_list = sorted(vs_core_list, key=lambda x: (x['kv_type'], x['kv_len'], x['level']))
    
    with open(os.path.join(out_dir, 'exp12_vs_coreset.json'), 'w') as f:
        json.dump(vs_core_list, f, indent=2)
    print(f"Saved vs_coreset.json")
    
    # Stats
    errs = [r['err_wavelet'] for r in results]
    print(f"\nMean err: {np.mean(errs):.6f}, Median: {np.median(errs):.6f}")
    for kt in ['clustered', 'random', 'skewed', 'smooth']:
        sub = [r for r in results if r['kv_type'] == kt]
        if sub:
            wins = sum(r['wavelet_wins_vs_core'] for r in sub)
            print(f"  {kt}: mean_err={np.mean([r['err_wavelet'] for r in sub]):.6f}, win_rate={wins/len(sub)*100:.0f}%")
    
    # Generate report
    print("\nGenerating report...")
    from simulation.exp12_wavelet_attention import generate_report
    report = generate_report(results, pareto, vs_core_list, phys)
    report_path = os.path.join(out_dir, 'exp12_wavelet_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved: {report_path}")
    
    print("Done!")

if __name__ == '__main__':
    run_phys_and_report()
