"""
Event case studies: model behavior before major volatility spikes in the test set.

Fits the RF on training data, then shows calibrated probability and abstention
status for the 40 days leading up to each identified spike event.

Key test-period events (train ends ~April/May 2022):
  - 2022 bear market peak volatility (May-Oct 2022)
  - SVB banking crisis (March 2023)
  - Japan carry-trade unwind / global selloff (Aug 2024, VIX hit ~65)

Outputs:
  results/event_dates.csv
  vix paper/event_case_studies.png
"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
RAW_COLS   = ['VIX', 'VIX3M', 'SP500', 'GOLD', 'TNY', 'DXY',
              'FEDFUNDS', 'YIELD_CURVE', 'VIX9D', 'SKEW']
LABEL_COLS = (['LABEL'] +
              [f'LABEL_{n}' for n in [5, 10, 15, 20, 25]] +
              [f'LABEL_SUSTAINED_{n}' for n in [5, 10, 15, 20, 25]])
feature_cols = [c for c in df.columns if c not in RAW_COLS + LABEL_COLS]
tscv = TimeSeriesSplit(n_splits=5)

print(f"{'='*65}")
print("  EVENT CASE STUDIES")
print(f"{'='*65}")

# Fit N=5 model (most responsive to near-term spikes)
n = 5
label_col = f'LABEL_{n}'
sub = df[feature_cols + [label_col, 'VIX']].dropna(subset=[label_col])
split_idx = int(len(sub) * 0.8)

split_date = sub.index[split_idx]
print(f"\n  Train/test split: {sub.index[0].date()} to {split_date.date()} (train)"
      f" | {split_date.date()} to {sub.index[-1].date()} (test)")

sub_tr = sub.iloc[:split_idx]
sub_te = sub.iloc[split_idx:]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(sub_tr[feature_cols].values)
X_te_s = scaler.transform(sub_te[feature_cols].values)
y_tr   = sub_tr[label_col].astype(int).values

rf  = RandomForestClassifier(n_estimators=200, max_depth=10,
                               min_samples_leaf=5, random_state=42)
cal = CalibratedClassifierCV(rf, cv=tscv, method='isotonic')
cal.fit(X_tr_s, y_tr)

# Attach predictions back to test index
proba_te = cal.predict_proba(X_te_s)[:, 1]
pred_df  = pd.DataFrame({
    'VIX':     sub_te['VIX'].values,
    'Proba':   proba_te,
    'Abstain': (np.abs(proba_te - 0.5) <= 0.25),
    'Label':   sub_te[label_col].values,
}, index=sub_te.index)

# ── Find VIX spikes: days when VIX crosses from <20 to ≥25 within 5 days ──
# Identify "spike days" = first day VIX≥25 after a calm period
vix_te = pred_df['VIX'].values
calm_before = np.zeros(len(vix_te), dtype=bool)
for i in range(5, len(vix_te)):
    if vix_te[i] >= 25 and vix_te[i-5] < 20:
        calm_before[i] = True

spike_indices = np.where(calm_before)[0]
spike_dates   = pred_df.index[spike_indices]
print(f"\n  Detected {len(spike_dates)} spike events (VIX crossed 20→25 within 5 days):")
for d, idx in zip(spike_dates, spike_indices):
    print(f"    {d.date()}  VIX={vix_te[idx]:.1f}  "
          f"5d-prior VIX={vix_te[max(0,idx-5)]:.1f}")

# Save events
event_df = pd.DataFrame({
    'Date': spike_dates,
    'VIX_spike': vix_te[spike_indices],
    'VIX_5d_prior': [vix_te[max(0,i-5)] for i in spike_indices],
})
event_df.to_csv('results/event_dates.csv', index=False)
print("\nSaved → results/event_dates.csv")

# ── Manual spotlight events (known crises in test period) ──────────────────
# We supplement auto-detected spikes with manually identified peaks
MANUAL_EVENTS = {
    'June 2022\n(Bear market peak)':   '2022-06-13',
    'Oct 2022\n(UK gilts crisis)':     '2022-10-13',
    'March 2023\n(SVB banking crisis)':'2023-03-13',
    'Aug 2024\n(Carry-trade unwind)':  '2024-08-05',
}

# Filter to events actually in test set
events_in_test = {}
for label, date_str in MANUAL_EVENTS.items():
    dt = pd.Timestamp(date_str)
    if dt in pred_df.index:
        events_in_test[label] = dt
    else:
        # Find nearest trading day
        nearest = pred_df.index[pred_df.index.searchsorted(dt)]
        if nearest <= pred_df.index[-1]:
            events_in_test[label] = nearest
            print(f"  Note: {date_str} → nearest trading day {nearest.date()}")

print(f"\n  Events in test set: {list(events_in_test.keys())}")

# ── Figure: 2×2 event case studies ────────────────────────────────────────
TAU = 0.25
WINDOW = 40  # trading days before event

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Event Case Studies: Model Behavior Before Major Volatility Spikes\n'
             '(N=5, τ=0.25 confidence threshold; test set only)',
             fontsize=13, fontweight='bold')

for ax, (event_label, event_date) in zip(axes.flatten(), events_in_test.items()):
    loc = pred_df.index.get_loc(event_date)
    start_loc = max(0, loc - WINDOW)
    end_loc   = min(len(pred_df) - 1, loc + 5)
    window_df = pred_df.iloc[start_loc:end_loc + 1]

    dates  = window_df.index
    vix    = window_df['VIX'].values
    proba  = window_df['Proba'].values
    abstain = window_df['Abstain'].values

    # Shade abstention zone
    ax.axhspan(0.5 - TAU, 0.5 + TAU, alpha=0.12, color='gray',
               label='Abstention zone (|p̂−0.5|≤0.25)')

    # Plot VIX / 100 as background context
    ax2 = ax.twinx()
    ax2.fill_between(dates, vix, alpha=0.08, color='black')
    ax2.plot(dates, vix, color='black', alpha=0.3, linewidth=1, linestyle=':')
    ax2.axhline(20, color='black', alpha=0.3, linewidth=0.8, linestyle='--')
    ax2.set_ylabel('VIX level', color='gray', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)
    ax2.set_ylim(0, max(vix) * 1.3)

    # Plot calibrated probability
    ax.plot(dates, proba, color='#2c3e50', linewidth=2, label='Calibrated prob p̂(high)', zorder=5)

    # Mark abstaining days with circles
    abs_dates = dates[abstain]
    abs_prob  = proba[abstain]
    ax.scatter(abs_dates, abs_prob, color='#e74c3c', s=40, zorder=6,
               label='Abstaining (uncertain)')

    # Non-abstaining calm predictions
    pred_calm  = ~abstain & (proba < 0.5)
    pred_high  = ~abstain & (proba >= 0.5)
    ax.scatter(dates[pred_calm], proba[pred_calm], color='#27ae60', s=30, marker='^',
               zorder=6, label='Predicts calm (confident)')
    ax.scatter(dates[pred_high], proba[pred_high], color='#e67e22', s=30, marker='v',
               zorder=6, label='Predicts high (confident)')

    # Mark event date
    ax.axvline(event_date, color='#c0392b', linewidth=2, linestyle='-', alpha=0.8, zorder=7)
    ax.text(event_date, 0.97, '▼ Spike', color='#c0392b', fontsize=8,
            ha='center', va='top', transform=ax.get_xaxis_transform())

    ax.axhline(0.5, color='navy', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel('Calibrated probability')
    ax.set_title(event_label)
    ax.legend(loc='upper left', fontsize=7, ncol=1)

    # Count abstaining days in window before event
    pre_event  = window_df.loc[:event_date]
    n_abstain  = pre_event['Abstain'].sum()
    n_total    = len(pre_event)
    ax.text(0.98, 0.05, f'Abstention rate\nbefore spike:\n{n_abstain}/{n_total} = {n_abstain/n_total*100:.0f}%',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#ffe0e0', alpha=0.8))

    # Rotate x tick labels
    ax.tick_params(axis='x', rotation=30, labelsize=8)

plt.tight_layout()
plt.savefig('vix paper/event_case_studies.png', bbox_inches='tight')
plt.close()
print("Saved → vix paper/event_case_studies.png")
