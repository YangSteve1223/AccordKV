#!/usr/bin/env python3
"""
Standalone runner for exp12 wavelet attention experiments.
Generates all results and saves to results/ directory.
"""
import json, os, sys, time
import numpy as np
import pywt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
from simulation.exp12_wavelet_attention import (
    make_clustered_kv, make_random_kv, make_skewed_kv, make_smooth_kv,
    wavelet_compressed_kv_attention, wavelet_full_idwt_baseline,
    kmeans_coreset, pca_attention_sketch, ground_truth,
    evaluate_wavelet, compute_pareto, vs_coreset_summary, physical_consistency_checks,
    generate_report
)

KV_FACTORIES = {
    'clustered': lambda n, d, s: make_clustered_kv(n, d, seed=s),
    'random': lambda n, d, s: make_random_kv(n, d, seed=s),
    'skewed': lambda n, d, s: make_skewed_kv(n, d, seed=s),
    'smooth': lambda n, d, s: make_smooth_kv(n, d, seed=s),
}

def run_grid(
    kv_types=['clustered','random','skewed','smooth'],
    kv_lens=[1024, 4096],
    ds=[128, 256],
    levels=[1, 2, 3],
    wavelets=['db4', 'haar'],
    q_len=16,
    seeds=[0, 1],
    verbose=True,
):
    """Run grid with all configs."""
    results = []
    total = 0
    
    for kt in kv_types:
        for kl in kv_lens:
            for d in ds:
                max_l = pywt.dwt_max_level(kl, wavelets[0])
                lvls = [L for L in levels if L <= max_l]
                total += len(lvls) * len(wavelets) * len(seeds)
    
    done = 0
    t0 = time.time()
    
    for kt in kv_types:
        for kl in kv_lens:
            for d in ds:
                max_l = pywt.dwt_max_level(kl, wavelets[0])
                lvls = [L for L in levels if L <= max_l]
                
                for seed in seeds:
                    make_kv = KV_FACTORIES[kt]
                    K, V = make_kv(kl, d, seed=seed)
                    Q = (np.random.default_rng(seed+1000).standard_normal((q_len, d)) * 0.5).astype(np.float32)
                    
                    for level in lvls:
                        for wavelet in wavelets:
                            try:
                                r = evaluate_wavelet(Q, K, V, wavelet, level, kt, seed)
                                results.append(r)
                            except Exception as e:
                                if verbose:
                                    print(f"  ERROR: {wavelet} L{level} {kt}: {e}")
                            done += 1
                            if verbose and done % 20 == 0:
                                elapsed = time.time() - t0
                                rate = done / elapsed
                                remaining = (total - done) / rate if rate > 0 else 0
                                print(f"  {done}/{total} ({100*done/total:.0f}%) ETA={remaining:.0f}s")
    
    return results

def main():
    print("=" * 80)
    print("Exp12: Wavelet-Domain Attention Compression")
    print("=" * 80)
    
    t0 = time.time()
    
    # Run main sweep
    print("\n[Main sweep]")
    results = run_grid(
        kv_types=['clustered', 'random', 'skewed', 'smooth'],
        kv_lens=[1024, 4096],
        ds=[128, 256],
        levels=[1, 2, 3, 4],
        wavelets=['db4', 'haar', 'sym4'],
        q_len=16,
        seeds=[0, 1, 2],
        verbose=True,
    )
    
    print(f"\nTotal experiments: {len(results)}")
    print(f"Sweep time: {time.time()-t0:.1f}s")
    
    # Physical consistency
    print("\n[Physical Consistency]")
    phys = physical_consistency_checks(kv_len=4096, d=128, wavelet='db4', q_len=16, seed=0)
    for sig_type, levels in phys.items():
        print(f"  {sig_type}: ", end="")
        for lvl, data in levels.items():
            idwt_ok = '✓' if data['idwt_ok'] else '✗'
            print(f"L{lvl[-1]}={data['err']:.4f}{idwt_ok} ", end="")
        print()
    
    # Summaries
    pareto = compute_pareto(results)
    vs_core = vs_coreset_summary(results)
    
    errs = [r['err_wavelet'] for r in results]
    print(f"\n[Wavelet Stats]")
    print(f"  Mean err: {np.mean(errs):.6f}")
    print(f"  Median err: {np.median(errs):.6f}")
    
    for kt in ['clustered', 'random', 'skewed', 'smooth']:
        sub = [r for r in results if r['kv_type'] == kt]
        if sub:
            wins = sum(r['wavelet_wins_vs_core'] for r in sub)
            print(f"  {kt}: mean_err={np.mean([r['err_wavelet'] for r in sub]):.6f}, win_rate={wins/len(sub)*100:.0f}%")
    
    for lvl in [1, 2, 3, 4]:
        sub = [r for r in results if r['level'] == lvl]
        if sub:
            print(f"  level={lvl}: mean_err={np.mean([r['err_wavelet'] for r in sub]):.6f}")
    
    # Save results
    out_dir = os.path.join(REPO_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    sweep_path = os.path.join(out_dir, 'exp12_wavelet_sweep.json')
    with open(sweep_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {sweep_path}")
    
    pareto_path = os.path.join(out_dir, 'exp12_pareto.json')
    with open(pareto_path, 'w') as f:
        json.dump(pareto, f, indent=2)
    print(f"Saved: {pareto_path}")
    
    vs_path = os.path.join(out_dir, 'exp12_vs_coreset.json')
    with open(vs_path, 'w') as f:
        json.dump(vs_core, f, indent=2)
    print(f"Saved: {vs_path}")
    
    phys_path = os.path.join(out_dir, 'exp12_physical_consistency.json')
    with open(phys_path, 'w') as f:
        json.dump(phys, f, indent=2)
    print(f"Saved: {phys_path}")
    
    # Report
    print("\n[Report]")
    report = generate_report(results, pareto, vs_core, phys)
    report_path = os.path.join(out_dir, 'exp12_wavelet_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved: {report_path}")
    
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print("Done!")

if __name__ == '__main__':
    main()
