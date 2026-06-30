"""
Regenerate all 5 paper figures from saved CSVs.
Fixes: tau notation, titles, figure sizing, y-axis scaling.
Outputs directly to paper/.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'legend.fontsize':  10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

N_VALUES   = [5, 10, 15, 20, 25]
TAU        = 0.25
OUT        = 'vix paper'
os.makedirs(OUT, exist_ok=True)

# CSV model keys → display names for figure titles
DISPLAY = {'Random Forest': 'Random Forest', 'XGBoost': 'Hist. Grad. Boosting (HGB)'}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Main results: baseline vs selective accuracy by horizon
# ─────────────────────────────────────────────────────────────────────────────
res = pd.read_csv('results/unified_model_results.csv')

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle('Accuracy vs Prediction Horizon: Baseline and Selective Predicting '
             '(fixed-spec calibrated pipeline)', fontsize=13)

for ax, model_name in zip(axes, ['Random Forest', 'XGBoost']):
    sub = res[res['Model'] == model_name].sort_values('N')
    ax.plot(sub['N'], sub['Base_Acc'], '--', color='#aaaaaa', linewidth=1.8,
            label='Baseline (uncalibrated)')
    ax.plot(sub['N'], sub['Sel_Acc'], '-o', color='#2e7d32', linewidth=2.2,
            markersize=7, label=f'Selective (τ={TAU})')
    ax.plot(sub['N'], sub['Bal_Acc'], '-.s', color='#e67e22', linewidth=1.8,
            markersize=6, label='Balanced acc (covered days)')
    ax.set_title(DISPLAY.get(model_name, model_name))
    ax.set_xlabel('N (trading days ahead)')
    ax.set_xticks(N_VALUES)
    ax.set_ylim(50, 100)
    ax.grid(True, alpha=0.2)
    ax.legend(framealpha=0.9)

axes[0].set_ylabel('Accuracy (%)')
plt.tight_layout()
plt.savefig(f'{OUT}/tuned_model_comparison.png', bbox_inches='tight')
plt.close()
print('Fig 1 saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Threshold sensitivity: selective accuracy vs τ
# ─────────────────────────────────────────────────────────────────────────────
th = pd.read_csv('results/threshold_optimization.csv')

colors_n = {5: '#e74c3c', 10: '#e67e22', 15: '#27ae60', 20: '#2980b9', 25: '#8e44ad'}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Selective Accuracy vs Confidence Threshold (τ) by Horizon', fontsize=13)

for ax, model_name in zip(axes, ['Random Forest', 'XGBoost']):
    for n in N_VALUES:
        sub = th[(th['Model'] == model_name) & (th['N'] == n)].sort_values('Threshold')
        if sub.empty:
            continue
        ax.plot(sub['Threshold'], sub['Sel_Acc'], marker='o',
                color=colors_n[n], label=f'N={n}', linewidth=2, markersize=6)
    ax.set_title(DISPLAY.get(model_name, model_name))
    ax.set_xlabel('Confidence Threshold (τ)')
    ax.set_ylabel('Selective Accuracy (%)')
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    ax.set_ylim(78, 100)
    ax.axvline(TAU, color='black', linestyle=':', linewidth=1.2, alpha=0.5, label=f'τ={TAU} (primary)')
    ax.legend(framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/threshold_optimization.png', bbox_inches='tight')
plt.close()
print('Fig 2 saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Robustness across VIX thresholds
# ─────────────────────────────────────────────────────────────────────────────
rob = pd.read_csv('results/robustness_check.csv')

colors_v = {18: '#e74c3c', 20: '#2196F3', 22: '#4CAF50'}
markers_v = {18: 's', 20: 'o', 22: '^'}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Robustness: Selective Predicting Accuracy Gain (τ={TAU}) by VIX Threshold', fontsize=13)

for ax, model_name in zip(axes, ['Random Forest', 'XGBoost']):
    for vt in [18, 20, 22]:
        sub = rob[(rob['Model'] == model_name) & (rob['VIX_Threshold'] == vt)].sort_values('N')
        if sub.empty:
            continue
        ax.plot(sub['N'], sub['Delta_Acc'], marker=markers_v[vt],
                color=colors_v[vt], label=f'VIX ≥ {vt}', linewidth=2, markersize=7)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.9, alpha=0.5)
    ax.set_title(DISPLAY.get(model_name, model_name))
    ax.set_xlabel('N (trading days ahead)')
    ax.set_ylabel('Accuracy Gain vs Baseline (pp)')
    ax.set_xticks(N_VALUES)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/robustness_check.png', bbox_inches='tight')
plt.close()
print('Fig 3 saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Walk-forward validation with error bars
# ─────────────────────────────────────────────────────────────────────────────
wf = pd.read_csv('results/walk_forward.csv')

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle(f'Walk-Forward Validation (5 folds): Baseline vs Selective (τ={TAU})', fontsize=13)

for ax, model_name in zip(axes, ['Random Forest', 'XGBoost']):
    sub = wf[wf['Model'] == model_name].sort_values('N')
    x   = sub['N'].values

    ax.errorbar(x - 0.4, sub['Mean_Base'], yerr=sub['Std_Base'],
                fmt='o--', label='Baseline', color='#888888',
                capsize=5, linewidth=2, markersize=7)
    ax.errorbar(x + 0.4, sub['Mean_Sel'], yerr=sub['Std_Sel'],
                fmt='s-', label=f'Selective (τ={TAU})', color='#2e7d32',
                capsize=5, linewidth=2, markersize=7)

    # shade N=20 region to highlight failure
    ax.axvspan(17, 23, color='#ffcccc', alpha=0.25, zorder=0)
    ax.text(20, ax.get_ylim()[0] + 2 if ax.get_ylim()[0] > 30 else 33,
            'N=20\n(gains collapse)', ha='center', fontsize=8.5, color='#cc0000', style='italic')

    ax.set_title(DISPLAY.get(model_name, model_name))
    ax.set_xlabel('N (trading days ahead)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(f'{OUT}/walk_forward.png', bbox_inches='tight')
plt.close()
print('Fig 4 saved.')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Permutation importance (unbiased, averaged across horizons)
# ─────────────────────────────────────────────────────────────────────────────
fi = pd.read_csv('results/permutation_importance.csv', index_col=0)
fi.columns = [c for c in fi.columns]  # preserve column names
fi['Avg_Importance'] = fi['Avg']
top20 = fi.sort_values('Avg_Importance', ascending=False).head(20)

def feat_color(name):
    if 'VIX' in name:   return '#e74c3c'
    if 'SP500' in name: return '#3498db'
    if 'SKEW' in name or 'GOLD' in name: return '#2ecc71'
    return '#e67e22'

colors_fi = [feat_color(f) for f in top20.index[::-1]]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(top20)), top20['Avg_Importance'].values[::-1],
        color=colors_fi, alpha=0.88, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index[::-1], fontsize=10)
ax.set_title('Permutation Feature Importance Across Horizons (N=5, 10, 20)\n'
             '(Mean accuracy drop when feature values are shuffled)', fontsize=12)
ax.set_xlabel('Mean Accuracy Decrease (permutation importance, N=5, 10, 20)')
ax.grid(True, alpha=0.2, axis='x')

legend_handles = [
    mpatches.Patch(color='#e74c3c', label='VIX features'),
    mpatches.Patch(color='#3498db', label='S&P 500 features'),
    mpatches.Patch(color='#2ecc71', label='SKEW / Gold features'),
    mpatches.Patch(color='#e67e22', label='Macro / Other'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT}/feature_importance_avg.png', bbox_inches='tight')
plt.close()
print('Fig 5 saved.')

print('\nAll figures saved to paper/.')
