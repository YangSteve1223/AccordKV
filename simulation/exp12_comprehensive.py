#!/usr/bin/env python3
"""Final comprehensive run using evaluate_wavelet from the module."""
import json, os, sys, time
import numpy as np

REPO_ROOT = '/app/data/所有对话/主对话/_staging/accord-kv'
sys.path.insert(0, REPO_ROOT)
import pywt
from simulation.exp12_wavelet_attention import (
    make_clustered_kv, make_random_kv, make_skewed_kv, make_smooth_kv,
    wavelet_compressed_kv_attention, wavelet_full_idwt_baseline,
    ground_truth, make_kv_factory,
)

def fast_kmeans(Q, K, V, r, seed):
    """Fast k-means for comparison."""
    gen = np.random.default_rng(seed)
    kv_len = K.shape[0]
    d = K.shape[1]
    r = min(r, kv_len)
    centroids = np.zeros((r, d), dtype=np.float32)
    centroids[0] = K[gen.integers(0, kv_len)]
    for j in range(1, r):
        dists = np.sum((K - centroids[:j, None]) ** 2, axis=2)
        min_dists = dists.min(axis=0)
        probs = min_dists / min_dists.sum()
        centroids[j] = K[gen.choice(kv_len, p=probs)]
    for _ in range(3):
        dists = np.sum((K[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assign = dists.argmin(axis=1)
        for j in range(r):
            mask = assign == j
            if mask.any():
                centroids[j] = K[mask].mean(axis=0)
    v_agg = np.zeros((r, d), dtype=np.float32)
    for j in range(r):
        mask = assign == j
        if mask.any():
            v_agg[j] = V[mask].mean(axis=0)
    weights = np.bincount(assign, minlength=r).astype(np.float32)
    weights = weights / weights.sum()
    scores = (Q @ centroids.T) / np.sqrt(d) + np.log(weights + 1e-30)
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y = p @ v_agg
    y = y / np.clip(l, 1e-30, None)
    return y.astype(np.float32)

def fast_pca_sketch(Q, K, V, r, seed):
    """Fast PCA sketch for comparison."""
    kv_len = K.shape[0]
    d = K.shape[1]
    _, _, Vt = np.linalg.svd(K, full_matrices=False)
    V_pca_basis = Vt[:r, :].T
    K_pca = K @ V_pca_basis
    V_pca = V @ V_pca_basis
    Q_pca = Q @ V_pca_basis
    r_sqrt = np.sqrt(r)
    scores = (Q_pca @ K_pca.T) / r_sqrt
    m = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - m)
    l = p.sum(axis=-1, keepdims=True)
    y_r = p @ V_pca
    y = y_r / np.clip(l, 1e-30, None)
    y = y @ V_pca_basis.T
    return y.astype(np.float32)

def run_full():
    t0 = time.time()
    results = []
    
    # Grid: 4 types × 2 lens × 4 levels × 3 wavelets × 2 seeds = 192 configs
    configs = []
    for kt in ['clustered', 'random', 'skewed', 'smooth']:
        for kl in [1024, 4096]:
            for level in [1, 2, 3, 4]:
                for wavelet in ['db4', 'haar', 'sym4']:
                    for seed in [0, 1]:
                        configs.append((kt, kl, level, wavelet, seed))
    
    print(f"Running {len(configs)} configs...")
    
    for i, (kt, kl, level, wavelet, seed) in enumerate(configs):
        make_kv = make_kv_factory(kt)
        K, V = make_kv(kl, 128, seed=seed)
        Q = (np.random.default_rng(seed+1000).standard_normal((16, 128)) * 0.5).astype(np.float32)
        
        gt = ground_truth(Q, K, V)
        
        # IDWT baseline
        idwt = wavelet_full_idwt_baseline(Q, K, V, wavelet, level)
        err_idwt = float(np.abs(idwt - gt).mean())
        
        # Wavelet
        y_wav, n_orig, n_comp = wavelet_compressed_kv_attention(Q, K, V, wavelet, level, mode='zero')
        err_wav = float(np.abs(y_wav - gt).mean())
        bytes_wav = n_comp * 4
        
        # Coreset (r = kv_len/2^level, capped at 8)
        r_core = max(4, min(8, kl // 2**level))
        y_core = fast_kmeans(Q, K, V, r_core, seed)
        err_core = float(np.abs(y_core - gt).mean())
        bytes_core = (r_core * 128 * 2 + r_core) * 4
        
        # PCA sketch
        r_pca = max(4, min(8, kl // 2**level))
        y_pca = fast_pca_sketch(Q, K, V, r_pca, seed)
        err_pca = float(np.abs(y_pca - gt).mean())
        bytes_pca = (128 * r_pca + kl * r_pca * 2) * 4
        
        results.append({
            'wavelet': wavelet, 'level': level, 'kv_type': kt,
            'kv_len': kl, 'q_len': 16, 'd': 128,
            'compression_factor': 2**level,
            'err_idwt_baseline': err_idwt,
            'idwt_is_zero': err_idwt < 1e-5,
            'err_wavelet': err_wav,
            'bytes_wavelet': bytes_wav,
            'r_coreset': r_core,
            'err_coreset': err_core,
            'bytes_coreset': bytes_core,
            'err_svd_coreset': err_pca,
            'bytes_svd_coreset': bytes_pca,
            'wavelet_wins_vs_core': err_wav < err_core,
            'wavelet_wins_vs_svdc': err_wav < err_pca,
        })
        
        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            remaining = (len(configs) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(configs)} ({100*(i+1)/len(configs):.0f}%) ETA={remaining:.0f}s", flush=True)
    
    print(f"\n{len(results)} results in {time.time()-t0:.1f}s")
    
    # Physical consistency
    phys = {}
    for sig_type, make_fn in [
        ('smooth', make_smooth_kv), ('clustered', make_clustered_kv),
        ('random', make_random_kv), ('skewed', make_skewed_kv),
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
    
    # Save results
    out_dir = os.path.join(REPO_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'exp12_wavelet_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
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
    
    # vs_core
    vs_core = []
    for r in results:
        vs_core.append({
            'kv_type': r['kv_type'], 'kv_len': r['kv_len'], 'level': r['level'],
            'wavelet_win_rate': float(r['wavelet_wins_vs_core']),
            'mean_wavelet_err': r['err_wavelet'],
            'mean_coreset_err': r['err_coreset'],
            'wavelet_better': r['err_wavelet'] < r['err_coreset'],
        })
    vs_core = sorted(vs_core, key=lambda x: (x['kv_type'], x['kv_len'], x['level']))
    
    with open(os.path.join(out_dir, 'exp12_vs_coreset.json'), 'w') as f:
        json.dump(vs_core, f, indent=2)
    
    with open(os.path.join(out_dir, 'exp12_physical_consistency.json'), 'w') as f:
        json.dump(phys, f, indent=2)
    
    # Stats
    errs = [r['err_wavelet'] for r in results]
    print(f"\nMean: {np.mean(errs):.6f}, Median: {np.median(errs):.6f}")
    for kt in ['clustered', 'random', 'skewed', 'smooth']:
        sub = [r for r in results if r['kv_type'] == kt]
        if sub:
            wins = sum(r['wavelet_wins_vs_core'] for r in sub)
            print(f"  {kt}: mean={np.mean([r['err_wavelet'] for r in sub]):.4f}, win={wins/len(sub)*100:.0f}%")
    for lvl in [1, 2, 3, 4]:
        sub = [r for r in results if r['level'] == lvl]
        if sub:
            print(f"  L={lvl}: mean={np.mean([r['err_wavelet'] for r in sub]):.4f}")
    
    print(f"\nTotal: {time.time()-t0:.1f}s. Saved!")
    return results, pareto, vs_core, phys

if __name__ == '__main__':
    run_full()
