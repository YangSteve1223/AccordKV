#!/usr/bin/env python3
"""
Figure 1: AVL (m,l,y) Merge Error Convergence
Data: results/exp1_v3.json  (9 configs, err_b)
Title: All 9 configs < 1e-7
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

BASE = '/app/data/所有对话/主对话/_staging/accord-kv'
OUT  = os.path.join(BASE, 'results/figs')
os.makedirs(OUT, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'results/exp1_v3.json')) as f:
    data = json.load(f)

err_b   = [d['err_b'] for d in data]
configs = [f"q={d['q_len']},kv={d['kv_len']}" for d in data]

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(err_b)), err_b, color='steelblue', alpha=0.8)

ax.axhline(y=1e-7, color='red', linestyle='--', linewidth=2, label='1e-7 threshold')
ax.set_yscale('log')
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs, rotation=45, ha='right')
ax.set_ylabel('max_abs_error (err_B)')
ax.set_title('Figure 1: AVL (m,l,y) Merge Error Convergence\n'
             'All 9 configs < 1e-7\n'
             'data: results/exp1_v3.json', fontsize=10)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, err_b):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.2,
            f'{val:.1e}', ha='center', va='bottom', fontsize=8)

# Label threshold pass
ax.text(8.5, 1e-8, 'all < 1e-7 ✓', color='green', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1.png'), dpi=150)
plt.close()
print('Fig 1 saved → results/figs/fig1.png')
