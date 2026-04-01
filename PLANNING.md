# NYCU AI HW1 — Cross-Modal BPM Imputation Project
## Handoff Context for Claude Code

### PROJECT OVERVIEW
- **Course:** NYCU AI Spring 2026, Professor Tsaipei Wang (王才沛)
- **Due:** April 3, 2026
- **Task:** Cross-modal heart rate imputation — use step count data from Huawei Band 7 to predict missing BPM readings
- **Deliverables:** 10-page PDF report, code appendix, dataset on GitHub

### DATA FILES (in project root)
- `bpm_data.csv` — 11,823 rows: date, bpm, time
- `step_data.csv` — 4,604 rows: date, steps, time_start, time_end
- `timeline_aligned.csv` — 36,930 rows: minute-level aligned timeline (generated in Phase 1)

### WHAT'S DONE (Phase 1: Data Exploration & Alignment)
- Loaded and parsed both CSVs
- Built unified minute-level timeline (36,930 minutes, 32% BPM coverage, 68% missing)
- Data quality analysis: day imbalance (17x ratio sparse→dense), no fully missing days
- Correlation: Pearson r=0.239, Spearman r=0.226, Chi-square χ²=139.1 (p<1e-25), Cramér's V=0.132
- BPM distribution: mean=92, std=18, slight right skew (0.232), roughly balanced across bins
- Generated 3 plots: exploration.png, day_timeline.png, bpm_vs_steps.png

### WHAT'S NEXT
- **Phase 2:** Feature engineering — build training features from aligned timeline:
  - Step windows (current, prev 5min, next 5min sum/mean)
  - Time of day (sin/cos encoded)
  - Neighboring BPM (last known before gap, next known after gap)
  - Step acceleration (change in steps)
- **Phase 3:** Model 1 — Linear Regression baseline
- **Phase 4:** Model 2 — Random Forest regressor
- **Phase 5:** Experiments — vary training data amount, feature ablation, cross-validation, with/without PCA
- **Phase 6:** Plots & visualizations for report
- **Phase 7:** Report writing (10 pages max)

### METHODS PLAN
- Linear Regression (baseline) + Random Forest (main model)
- Optional: SAITS (deep learning, via PyPOTS library) as 3rd method
- Evaluation: cross-validation, MSE/RMSE, feature importance
- Experiments required by HW spec:
  - Effect of training data amount
  - Effect of data composition/balance
  - Data augmentation comparison
  - With/without dimensionality reduction (PCA)

### KEY FINDINGS FOR REPORT
- Motivation: clinical EWS, HRV analysis (unstable at 10% missing), hypoglycemia prediction, mental health monitoring
- BPM↔Steps correlation is real (p<1e-25) but weak (r=0.239) — motivates needing neighboring BPM + temporal features, not just steps alone
- Day imbalance: early March sparse (50-120 readings/day), late March dense (700-860/day) — natural experiment for data quantity effect
- SAITS paper (arxiv 2202.08516): DMSA mechanism, MIT+ORT joint training, outperforms BRITS by 12-38% MAE

### HW REQUIREMENTS CHECKLIST
- [ ] Dataset not already available (✓ own Huawei Band data)
- [ ] At least 2 supervised learning methods, max 1 deep learning
- [ ] Cross-validation evaluation
- [ ] Confusion matrix / MSE metrics
- [ ] Experiment: effect of training data amount
- [ ] Experiment: data composition/balance
- [ ] Experiment: data augmentation
- [ ] Experiment: with/without dimensionality reduction
- [ ] Discussion section
- [ ] Code appendix
- [ ] Dataset on GitHub with documentation
- [ ] Max 10 pages, size-12+ fonts

---

## ALL CODE CELLS (copy into notebook)

### Cell 1: Load Data
```python
import pandas as pd

bpm = pd.read_csv("bpm_data.csv")
steps = pd.read_csv("step_data.csv")

print("=== BPM Data ===")
print(f"Shape: {bpm.shape}")
print(bpm.head(10))
print(f"\nDtypes:\n{bpm.dtypes}")

print("\n=== Steps Data ===")
print(f"Shape: {steps.shape}")
print(steps.head(10))
print(f"\nDtypes:\n{steps.dtypes}")
```

### Cell 2: Parse Datetimes & Basic Stats
```python
import pandas as pd

bpm['datetime'] = pd.to_datetime(bpm['date'] + ' ' + bpm['time'], format='%m/%d/%Y %H:%M')
bpm = bpm.sort_values('datetime').reset_index(drop=True)

steps['datetime_start'] = pd.to_datetime(steps['date'] + ' ' + steps['time_start'], format='%m/%d/%Y %H:%M')
steps['datetime_end'] = pd.to_datetime(steps['date'] + ' ' + steps['time_end'], format='%m/%d/%Y %H:%M')
steps = steps.sort_values('datetime_start').reset_index(drop=True)

print("=== BPM after parsing ===")
print(f"Range: {bpm['datetime'].min()} → {bpm['datetime'].max()}")
print(f"Total readings: {len(bpm)}")
print(f"Unique dates: {bpm['datetime'].dt.date.nunique()}")
print(f"BPM stats:\n{bpm['bpm'].describe()}")

print("\n=== Steps after parsing ===")
print(f"Range: {steps['datetime_start'].min()} → {steps['datetime_start'].max()}")
print(f"Total readings: {len(steps)}")
print(f"Steps per interval stats:\n{steps['steps'].describe()}")

# Readings per day
print("\n=== BPM readings per day ===")
bpm_daily = bpm.groupby(bpm['datetime'].dt.date).size()
print(bpm_daily.to_string())

print("\n=== Step readings per day ===")
step_daily = steps.groupby(steps['datetime_start'].dt.date).size()
print(step_daily.to_string())
```

### Cell 3: Exploration Plots
```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Huawei Band 7 — Data Exploration', fontsize=14, fontweight='bold')

# 1. BPM distribution
axes[0,0].hist(bpm['bpm'], bins=50, color='#e53935', alpha=0.8, edgecolor='#333')
axes[0,0].set_xlabel('BPM')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title(f'BPM Distribution (n={len(bpm)}, μ={bpm["bpm"].mean():.1f}, σ={bpm["bpm"].std():.1f})')
axes[0,0].axvline(bpm['bpm'].mean(), color='black', linestyle='--', alpha=0.5)

# 2. Steps distribution
axes[0,1].hist(steps['steps'], bins=50, color='#00e676', alpha=0.8, edgecolor='#333')
axes[0,1].set_xlabel('Steps per minute')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title(f'Steps Distribution (n={len(steps)}, μ={steps["steps"].mean():.1f})')

# 3. Readings per day
daily_bpm = bpm.groupby(bpm['datetime'].dt.date).size()
daily_steps = steps.groupby(steps['datetime_start'].dt.date).size()
x = range(len(daily_bpm))
axes[1,0].bar(x, daily_bpm.values, color='#e53935', alpha=0.7, label='BPM')
axes[1,0].bar(x, daily_steps.values, color='#00e676', alpha=0.7, label='Steps')
axes[1,0].set_xlabel('Day (March 2026)')
axes[1,0].set_ylabel('Number of readings')
axes[1,0].set_title('Readings per Day')
axes[1,0].set_xticks(x[::5])
axes[1,0].set_xticklabels([str(d.day) for d in daily_bpm.index[::5]])
axes[1,0].legend()

# 4. BPM gaps
gaps = bpm['datetime'].diff().dt.total_seconds() / 60
gaps = gaps.dropna()
gaps_clipped = gaps[gaps < 120]
axes[1,1].hist(gaps_clipped, bins=60, color='#ff9800', alpha=0.8, edgecolor='#333')
axes[1,1].set_xlabel('Gap between readings (minutes)')
axes[1,1].set_ylabel('Count')
axes[1,1].set_title(f'BPM Sampling Gaps (median={gaps.median():.1f}min, mean={gaps.mean():.1f}min)')
axes[1,1].axvline(gaps.median(), color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('exploration.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Cell 4: Day Timeline (March 20)
```python
day = '2026-03-20'
bpm_day = bpm[bpm['datetime'].dt.date == pd.Timestamp(day).date()]
steps_day = steps[steps['datetime_start'].dt.date == pd.Timestamp(day).date()]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
fig.suptitle(f'March 20, 2026 — BPM & Steps Timeline', fontsize=13, fontweight='bold')

ax1.plot(bpm_day['datetime'], bpm_day['bpm'], color='#e53935', linewidth=0.8, alpha=0.8)
ax1.scatter(bpm_day['datetime'], bpm_day['bpm'], color='#e53935', s=3, alpha=0.5)
ax1.set_ylabel('BPM')
ax1.set_ylim(40, 180)
ax1.grid(alpha=0.2)

ax2.bar(steps_day['datetime_start'], steps_day['steps'], width=0.0007, color='#00e676', alpha=0.8)
ax2.set_ylabel('Steps/min')
ax2.set_xlabel('Time')
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('day_timeline.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Cell 5: BPM vs Steps Correlation
```python
bpm['minute'] = bpm['datetime'].dt.floor('min')
steps['minute'] = steps['datetime_start'].dt.floor('min')
merged = bpm.merge(steps[['minute', 'steps']], on='minute', how='inner')

print(f"Aligned BPM-Steps pairs (same minute): {len(merged)}")
corr = merged['bpm'].corr(merged['steps'])
print(f"Pearson correlation (BPM vs Steps): {corr:.3f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(merged['steps'], merged['bpm'], alpha=0.15, s=8, color='#00b0ff')
ax.set_xlabel('Steps per minute')
ax.set_ylabel('BPM')
ax.set_title(f'BPM vs Steps (n={len(merged)}, r={corr:.3f})')
z = np.polyfit(merged['steps'], merged['bpm'], 1)
p = np.poly1d(z)
x_line = np.linspace(0, merged['steps'].max(), 100)
ax.plot(x_line, p(x_line), color='#e53935', linewidth=2, linestyle='--')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('bpm_vs_steps.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Cell 6: Build Unified Minute-Level Timeline
```python
import numpy as np

date_min = min(bpm['datetime'].min(), steps['datetime_start'].min()).floor('min')
date_max = max(bpm['datetime'].max(), steps['datetime_start'].max()).ceil('min')
timeline = pd.DataFrame({'minute': pd.date_range(date_min, date_max, freq='min')})

# Merge BPM
bpm['minute'] = bpm['datetime'].dt.floor('min')
bpm_per_min = bpm.groupby('minute')['bpm'].mean().round().astype(int).reset_index()
timeline = timeline.merge(bpm_per_min, on='minute', how='left')

# Merge Steps
steps['minute'] = steps['datetime_start'].dt.floor('min')
steps_per_min = steps.groupby('minute')['steps'].sum().reset_index()
timeline = timeline.merge(steps_per_min, on='minute', how='left')
timeline['steps'] = timeline['steps'].fillna(0).astype(int)

# Time features
timeline['hour'] = timeline['minute'].dt.hour
timeline['hour_sin'] = np.sin(2 * np.pi * timeline['hour'] / 24)
timeline['hour_cos'] = np.cos(2 * np.pi * timeline['hour'] / 24)
timeline['day'] = timeline['minute'].dt.date

print(f"Shape: {timeline.shape}")
print(f"BPM coverage: {timeline['bpm'].notna().sum()} / {len(timeline)} ({timeline['bpm'].notna().mean()*100:.1f}%)")
print(f"BPM missing: {timeline['bpm'].isna().sum()} ({timeline['bpm'].isna().mean()*100:.1f}%)")

timeline.to_csv('timeline_aligned.csv', index=False)
```

### Cell 7: Data Quality Analysis
```python
from scipy import stats

# Day imbalance
daily = bpm.groupby(bpm['datetime'].dt.date).agg(
    count=('bpm', 'size'),
    hours_covered=('datetime', lambda x: (x.max() - x.min()).total_seconds() / 3600),
    mean_bpm=('bpm', 'mean')
)
daily['category'] = pd.cut(daily['count'], bins=[0, 100, 300, 500, 900],
    labels=['Sparse (<100)', 'Low (100-300)', 'Medium (300-500)', 'Dense (500+)'])
print("=== Day Categories ===")
print(daily['category'].value_counts().sort_index())
print(f"Ratio max/min: {daily['count'].max() / daily['count'].min():.1f}x")

# Correlation analysis
merged = bpm.merge(steps[['minute', 'steps']], on='minute', how='inner')
pearson_r, pearson_p = stats.pearsonr(merged['bpm'], merged['steps'])
spearman_r, spearman_p = stats.spearmanr(merged['bpm'], merged['steps'])
print(f"\nPearson:  r={pearson_r:.3f}, p={pearson_p:.2e}")
print(f"Spearman: r={spearman_r:.3f}, p={spearman_p:.2e}")

# Chi-square
bpm_bins = pd.cut(merged['bpm'], bins=[0, 70, 90, 110, 180],
    labels=['Low(<70)', 'Rest(70-90)', 'Moderate(90-110)', 'High(110+)'])
step_bins = pd.cut(merged['steps'], bins=[-1, 10, 50, 80, 140],
    labels=['Idle(0-10)', 'Light(11-50)', 'Walking(51-80)', 'Active(81+)'])
contingency = pd.crosstab(bpm_bins, step_bins)
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
cramers_v = np.sqrt(chi2 / (len(merged) * (min(contingency.shape) - 1)))
print(f"\nChi-square: χ²={chi2:.1f}, p={p_chi:.2e}, Cramér's V={cramers_v:.3f}")
print(f"\nContingency table:\n{contingency}")

# BPM distribution
print(f"\nSkewness: {bpm['bpm'].skew():.3f}")
print(f"Kurtosis: {bpm['bpm'].kurtosis():.3f}")
```
