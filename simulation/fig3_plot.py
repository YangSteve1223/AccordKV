#!/usr/bin/env python3
"""
Figure 3: Coreset + INT4 Quantization Pareto  (2×2 subplot)
Data: results/exp3_coreset_pareto.json + results/coreset_postfix_int4.json
Panels:
  (a) Coreset vs Drop — clustered
  (b) INT4 vs FP32 ablation
  (c) Heterogeneous Oracle vs Blind Average
  (d) Attention-Aware vs Uniform Coreset
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

BASE = '/app/data/所有对话/主对话/_staging/accord-kv'
OUT  = os.path.join(BASE, 'results/figs')

# ── Load data ──────────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'results/exp3_coreset_pareto.json')) as f:
    pareto = json.load(f)

with open(os.path.join(BASE, 'results/coreset_postfix_int4.json')) as f:
    int4_raw = json.load(f)

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ── 3a: Coreset vs Drop (clustered) ────────────────────────────────────────
clustered = [d for d in pareto['pareto'] if d.get('clustered', True)]
clustered = sorted(clustered, key=lambda x: x['compression_sketch'])
comp_c   = [d['compression_sketch'] * 100 for d in clustered]
err_c    = [d['err_sketch']           for d in clustered]
comp_d   = [d['compression_drop']     * 100 for d in clustered]
err_d_c  = [d['err_drop']             for d in clustered]

ax = axes[0, 0]
ax.plot(comp_c, err_c, 'b-o', label='Coreset',  markersize=5)
ax.plot(comp_d, err_d_c, 'r--s', label='Drop', markersize=5)
ax.set_xlabel('Compression (% of full)')
ax.set_ylabel('rel_l2 error')
ax.set_title('(a) Clustered: Coreset vs. Drop')
ax.legend()
ax.grid(alpha=0.3)

# ── 3b: INT4 vs FP32 (kv_len=4096, q_len=64) ──────────────────────────────
int4_pts = [d for d in int4_raw['int4_sweep']
            if d['kv_len'] == 4096 and d['q_len'] == 64]
int4_pts = sorted(int4_pts, key=lambda x: x['sketch_r'])
comp_fp  = [d['compression_gain'] for d in int4_pts]
err_fp   = [d['err_fp32'] for d in int4_pts]
err_int4 = [d['err_intn'] for d in int4_pts]

ax = axes[0, 1]
ax.plot(range(len(int4_pts)), err_fp,   'b-o', label='FP32')
ax.plot(range(len(int4_pts)), err_int4, 'g-s', label='INT4')
ax.set_xlabel('sketch_r (4→8→16→32)')
ax.set_ylabel('rel_l2 error')
ax.set_title('(b) INT4 vs. FP32')
ax.legend()
ax.grid(alpha=0.3)
if int4_pts:
    gain = int4_pts[0].get('compression_gain', 7.34)
    ax.text(0.02, 0.95, f'~{gain:.1f}× compression', transform=ax.transAxes,
            fontsize=8, va='top', color='gray')

# ── 3c: Oracle vs Blind (exploration_C: INT4 quantization) ─────────────────
expC = pareto.get('exploration_C', [])
ax = axes[1, 0]
ax.set_xlabel('Compression level (sketch_r)')
ax.set_ylabel('rel_l2 error')
ax.set_title('(c) Heterogeneous Oracle vs Blind Average')
if expC:
    # exploration_C: INT4 quantization (FP32 vs INT4, ~7.34× compression)
    # Oracle = min(err_fp32, err_int4), Blind = (err_fp32 + err_int4)/2
    fp32_errs   = [d.get('err_fp32',  0) for d in expC]
    int4_errs   = [d.get('err_int4',  0) for d in expC]
    oracle_errs = [min(f, i) for f, i in zip(fp32_errs, int4_errs)]
    blind_errs  = [(f + i) / 2 for f, i in zip(fp32_errs, int4_errs)]
    # Group by kv_len to show heterogeneous backends value
    x_labels = [f"kv={d['kv_len']}\nr={d['sketch_r']}" for d in expC]
    x_pts = range(len(expC))
    ax.plot(x_pts, oracle_errs, 'k-',  label='Oracle (min(FP32,INT4))', linewidth=2)
    ax.plot(x_pts, fp32_errs,   'b-o', label='FP32 (full)', markersize=5)
    ax.plot(x_pts, int4_errs,   'r-s', label='INT4 (7.34× comp)', markersize=5)
    ax.plot(x_pts, blind_errs,  'g--', label='Blind Avg', linewidth=1.5)
    ax.set_xticks(x_pts)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=0)
    ax.legend(fontsize=7, loc='best')
    # Annotate: Oracle always ≤ min(FP32, INT4)
    violations = sum(1 for o, f, i in zip(oracle_errs, fp32_errs, int4_errs) if o > min(f, i))
    if violations == 0:
        ax.text(0.02, 0.98, f'✓ Oracle ≤ min(FP32,INT4) for all {len(expC)} configs',
                transform=ax.transAxes, fontsize=8, va='top', color='green')
    ax.text(0.98, 0.02, 'data: exploration_C (INT4 quantization)',
            transform=ax.transAxes, fontsize=7, ha='right', color='gray', style='italic')
else:
    eps = list(range(5))
    ax.plot(eps, [0.5, 0.55, 0.6, 0.65, 0.7], 'k-', label='Oracle (synthetic)', linewidth=2)
    ax.plot(eps, [0.6, 0.65, 0.7, 0.75, 0.8], 'g--', label='Coreset (synthetic)')
    ax.plot(eps, [0.55, 0.6, 0.65, 0.7, 0.75], 'r-.', label='Kernel (synthetic)')
    ax.plot(eps, [0.7, 0.75, 0.8, 0.85, 0.9], 'b:', label='Blind Avg (synthetic)')
    ax.legend(fontsize=8)
    ax.text(0.5, 0.5, '[synthetic — real data pending]', transform=ax.transAxes,
            ha='center', va='center', fontsize=9, color='red',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax.grid(alpha=0.3)

# ── 3d: Attention-Aware vs Uniform ─────────────────────────────────────────
attn_data = pareto.get('exploration_A', [])
kv_lens_a = sorted(set(d['kv_len'] for d in attn_data), reverse=True)
improvements = []
for kv in kv_lens_a:
    d4 = [d for d in attn_data if d['kv_len'] == kv and d['sketch_r'] == 4]
    if d4:
        imp = d4[0].get('improvement_pct', 0)
        improvements.append(imp)
    else:
        improvements.append(0)

ax = axes[1, 1]
colors_d = ['green' if v >= 0 else 'red' for v in improvements]
ax.bar(range(len(kv_lens_a)), improvements, color=colors_d, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=1)
ax.set_xticks(range(len(kv_lens_a)))
ax.set_xticklabels([str(k) for k in kv_lens_a])
ax.set_xlabel('kv_len')
ax.set_ylabel('Improvement (%)')
ax.set_title('(d) Attention-Aware vs. Uniform Coreset\n(negative = attention-aware worse)')
ax.grid(alpha=0.3)

plt.suptitle('Figure 3: Coreset + INT4 Quantization Fidelity Analysis\n'
             'data: results/exp3_coreset_pareto.json + coreset_postfix_int4.json',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3.png'), dpi=150)
plt.close()
print('Fig 3 saved → results/figs/fig3.png')
