#!/usr/bin/env python3
"""Ultra-minimal exp12 test - 8 configs for quick validation."""
import json, os, sys, time
import numpy as np

REPO_ROOT = '/app/data/所有对话/主对话/_staging/accord-kv'
sys.path.insert(0, REPO_ROOT)
import pywt
from simulation.exp12_wavelet_attention import (
    make_clustered_kv, make_random_kv, make_skewed_kv, make_smooth_kv,
    wavelet_compressed_kv_attention, wavelet_full_idwt_baseline,
    kmeans_coreset, pca_attention_sketch, ground_truth,
    make_kv_factory,
)

def evaluate_single(Q, K, V, wavelet, level, kv_type, seed):
    gt = ground_truth(Q, K, V)
    kv_len = K.shape[0]
    d = K.shape[1]
    
    idwt = wavelet_full_idwt_baseline(Q, K, V, wavelet, level)
    err_idwt = float(np.abs(idwt - gt).mean())
    
    y_wav, n_orig, n_comp = wavelet_compressed_kv_attention(Q, K, V, wavelet, level, mode='zero')
    err_wav = float(np.abs(y_wav - gt).mean())
    bytes_wav = n_comp * 4
    
    r_core = max(4, min(16, kv_len // 2**level))
    y_core, _, _ = kmeans_coreset(Q, K, V, r_core, seed)
    err_core = float(np.abs(y_core - gt).mean())
    bytes_core = (r_core * d * 2 + r_core) * 4
    
    r_pca = max(4, min(16, kv_len // 2**level))
    y_pca, _, _ = pca_attention_sketch(Q, K, V, r_pca, seed)
    err_pca = float(np.abs(y_pca - gt).mean())
    bytes_pca = (d * r_pca + kv_len * r_pca * 2) * 4
    
    return {
        'wavelet': wavelet, 'level': level, 'kv_type': kv_type,
        'kv_len': kv_len, 'q_len': Q.shape[0], 'd': d,
        'compression_factor': 2**level,
        'err_idwt_baseline': err_idwt, 'idwt_is_zero': err_idwt < 1e-5,
        'err_wavelet': err_wav, 'bytes_wavelet': bytes_wav,
        'r_coreset': r_core, 'err_coreset': err_core, 'bytes_coreset': bytes_core,
        'err_pca': err_pca, 'bytes_pca': bytes_pca,
        'wavelet_wins_vs_core': err_wav < err_core,
        'wavelet_wins_vs_pca': err_wav < err_pca,
    }

def run_tiny():
    t0 = time.time()
    results = []
    
    # 2 kv_types × 2 kv_lens × 2 levels × 1 wavelet × 1 seed = 8 configs
    for kt in ['clustered', 'random']:
        for kl in [1024, 4096]:
            for level in [2, 3]:
                for wavelet in ['db4']:
                    for seed in [0, 1]:
                        make_kv = make_kv_factory(kt)
                        K, V = make_kv(kl, 128, seed=seed)
                        Q = (np.random.default_rng(seed+1000).standard_normal((16, 128)) * 0.5).astype(np.float32)
                        
                        r = evaluate_single(Q, K, V, wavelet, level, kt, seed)
                        results.append(r)
                        print(f"  {kt} kl={kl} L={level} s={seed}: err={r['err_wavelet']:.4f} core={r['err_coreset']:.4f} pca={r['err_pca']:.4f} idwt={r['err_idwt_baseline']:.2e}", flush=True)
    
    print(f"\n{len(results)} results in {time.time()-t0:.1f}s")
    
    # Physical consistency
    phys = {}
    for sig_type, make_fn in [
        ('smooth', lambda n,d,s: make_smooth_kv(n,d,seed=s)),
        ('clustered', lambda n,d,s: make_clustered_kv(n,d,seed=s)),
        ('random', lambda n,d,s: make_random_kv(n,d,seed=s)),
    ]:
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
            print(f"  phys {sig_type} L={level}: err={err:.4f} idwt={err_idwt:.2e}", flush=True)
    
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
    
    # vs_core summary
    vs_core = []
    for r in results:
        vs_core.append({
            'kv_type': r['kv_type'], 'kv_len': r['kv_len'], 'level': r['level'],
            'wavelet_win_rate': float(r['wavelet_wins_vs_core']),
            'mean_wavelet_err': r['err_wavelet'],
            'mean_coreset_err': r['err_coreset'],
            'wavelet_better': r['err_wavelet'] < r['err_coreset'],
        })
    
    # Stats
    errs = [r['err_wavelet'] for r in results]
    print(f"\nMean err: {np.mean(errs):.6f}, Median: {np.median(errs):.6f}")
    for kt in ['clustered', 'random']:
        sub = [r for r in results if r['kv_type'] == kt]
        if sub:
            wins = sum(r['wavelet_wins_vs_core'] for r in sub)
            print(f"  {kt}: mean_err={np.mean([r['err_wavelet'] for r in sub]):.6f}, win_rate={wins/len(sub)*100:.0f}%")
    
    # Save
    out_dir = os.path.join(REPO_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'exp12_wavelet_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, 'exp12_pareto.json'), 'w') as f:
        json.dump(pareto, f, indent=2)
    with open(os.path.join(out_dir, 'exp12_vs_coreset.json'), 'w') as f:
        json.dump(vs_core, f, indent=2)
    with open(os.path.join(out_dir, 'exp12_physical_consistency.json'), 'w') as f:
        json.dump(phys, f, indent=2)
    
    print(f"\nTotal: {time.time()-t0:.1f}s. Saved!")
    return results, pareto, vs_core, phys

if __name__ == '__main__':
    run_tiny()
