#!/usr/bin/env python3
"""
Figure 2: Multi-Head Decoding Compression Ratios
Data: results/exp2_multi_head.json  (45 configs, num_shards=4 filter)
Title: Up to 31,775× compression ratio
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

BASE = '/app/data/所有对话/主对话/_staging/accord-kv'
OUT  = os.path.join(BASE, 'results/figs')

# ── Load data ──────────────────────────────────────────────────────────────
with open(os.path.join(BASE, 'results/exp2_multi_head.json')) as f:
    data = json.load(f)

# Filter to primary config (num_shards=4)
data4 = [d for d in data if d['num_shards'] == 4]
q_lens  = sorted(set(d['q_len']  for d in data4))
kv_lens = sorted(set(d['kv_len'] for d in data4))

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors  = plt.cm.viridis(np.linspace(0, 0.9, len(q_lens)))
markers = ['o', 's', '^', 'D', 'v']

for i, q in enumerate(q_lens):
    ratios = []
    for kv in kv_lens:
        match = [d for d in data4 if d['q_len'] == q and d['kv_len'] == kv]
        ratios.append(match[0]['ratio'] if match else np.nan)
    ax.plot(kv_lens, ratios, marker=markers[i % len(markers)],
            color=colors[i], linewidth=2, label=f'q_len={q}', markersize=8)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('kv_len')
ax.set_ylabel('Compression Ratio (full/sketch bytes)')
ax.set_title('Figure 2: Multi-Head Decoding Compression Ratios\n'
             'Peak: q_len=1, kv_len=16384 → 31,775×\n'
             'data: results/exp2_multi_head.json', fontsize=10)
ax.legend(title='q_len')
ax.grid(True, alpha=0.3)

# Annotate peak
peak_ratio = max(d['ratio'] for d in data4)
peak_kv    = 16384
ax.annotate(f'{peak_ratio:.0f}×', xy=(peak_kv, peak_ratio),
            xytext=(8000, peak_ratio * 1.5),
            arrowprops=dict(arrowstyle='->', color='red'),
            color='red', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2.png'), dpi=150)
plt.close()
print('Fig 2 saved → results/figs/fig2.png')
