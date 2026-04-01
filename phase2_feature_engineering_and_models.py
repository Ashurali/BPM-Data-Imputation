"""
Phase 2: Feature Engineering & Model Training
- Improves Phase 1 timeline (finer time encoding, day-of-week, gap distance)
- Builds training features: step windows, neighboring BPM, step acceleration
- Trains Linear Regression (baseline) and Random Forest models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD & IMPROVE TIMELINE
# ============================================================
print("=" * 60)
print("PHASE 2: Feature Engineering & Model Training")
print("=" * 60)

timeline = pd.read_csv("Data/timeline_aligned.csv", parse_dates=['minute'])
timeline['day'] = pd.to_datetime(timeline['day'])
print(f"\nLoaded timeline: {timeline.shape}")
print(f"BPM coverage: {timeline['bpm'].notna().sum()} / {len(timeline)} "
      f"({timeline['bpm'].notna().mean()*100:.1f}%)")

# --- Phase 1 Improvements ---
print("\n--- Improving Phase 1 features ---")

# 1a. Minute-of-day (0-1439) — much finer than hour alone
timeline['minute_of_day'] = timeline['minute'].dt.hour * 60 + timeline['minute'].dt.minute
timeline['minute_sin'] = np.sin(2 * np.pi * timeline['minute_of_day'] / 1440)
timeline['minute_cos'] = np.cos(2 * np.pi * timeline['minute_of_day'] / 1440)

# 1b. Day of week (0=Mon, 6=Sun), with sin/cos encoding
timeline['day_of_week'] = timeline['minute'].dt.dayofweek
timeline['dow_sin'] = np.sin(2 * np.pi * timeline['day_of_week'] / 7)
timeline['dow_cos'] = np.cos(2 * np.pi * timeline['day_of_week'] / 7)

# 1c. Is-weekend flag
timeline['is_weekend'] = (timeline['day_of_week'] >= 5).astype(int)

# 1d. Day index (0-based, for tracking data density progression)
timeline['day_index'] = (timeline['day'] - timeline['day'].min()).dt.days

print(f"  Added: minute_of_day (sin/cos), day_of_week (sin/cos), is_weekend, day_index")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n--- Building training features ---")

# 2a. Step rolling windows
for w in [3, 5, 10, 15]:
    timeline[f'steps_roll_{w}'] = timeline['steps'].rolling(w, center=True, min_periods=1).mean()
    timeline[f'steps_sum_{w}'] = timeline['steps'].rolling(w, center=True, min_periods=1).sum()

# 2b. Step acceleration (change in steps)
timeline['step_diff'] = timeline['steps'].diff().fillna(0)
timeline['step_diff_abs'] = timeline['step_diff'].abs()

# 2c. Step statistics in local windows
timeline['steps_std_5'] = timeline['steps'].rolling(5, center=True, min_periods=1).std().fillna(0)
timeline['steps_max_5'] = timeline['steps'].rolling(5, center=True, min_periods=1).max()

# 2d. Activity state (based on rolling step average)
timeline['activity_state'] = pd.cut(
    timeline['steps_roll_5'],
    bins=[-1, 2, 30, 70, 999],
    labels=[0, 1, 2, 3]  # idle, light, walking, active
).astype(int)

print(f"  Added step windows (3/5/10/15), step acceleration, step stats, activity state")

# 2e. Neighboring BPM — last known before and next known after each row
bpm_known = timeline['bpm'].copy()

# Forward fill: last known BPM before this point
timeline['bpm_last_known'] = bpm_known.ffill()
# Backward fill: next known BPM after this point
timeline['bpm_next_known'] = bpm_known.bfill()

# Distance (in minutes) to nearest known BPM
bpm_mask = timeline['bpm'].notna()
# Forward distance: minutes since last known BPM
fwd_group = (~bpm_mask).cumsum()
timeline['dist_to_last_bpm'] = timeline.groupby(fwd_group).cumcount()
# Where BPM is known, distance = 0
timeline.loc[bpm_mask, 'dist_to_last_bpm'] = 0

# Backward distance: minutes until next known BPM
bpm_mask_rev = bpm_mask[::-1]
bwd_group = (~bpm_mask_rev).cumsum()
timeline['dist_to_next_bpm'] = timeline.iloc[::-1].groupby(bwd_group).cumcount().values
timeline.loc[bpm_mask, 'dist_to_next_bpm'] = 0

# Interpolated BPM from neighbors (weighted by distance)
total_dist = timeline['dist_to_last_bpm'] + timeline['dist_to_next_bpm']
total_dist = total_dist.replace(0, 1)  # avoid division by zero
timeline['bpm_neighbor_interp'] = (
    timeline['bpm_last_known'] * (1 - timeline['dist_to_last_bpm'] / total_dist) +
    timeline['bpm_next_known'] * (1 - timeline['dist_to_next_bpm'] / total_dist)
)

print(f"  Added neighboring BPM (last/next known, distances, interpolation)")

# 2f. Gap length — total length of the current gap
gap_id = (bpm_mask != bpm_mask.shift()).cumsum()
gap_lengths = timeline.groupby(gap_id)['bpm'].transform('count')
timeline['gap_length'] = gap_lengths
timeline.loc[bpm_mask, 'gap_length'] = 0

print(f"  Added gap_length feature")

# ============================================================
# 3. PREPARE TRAINING DATA
# ============================================================
print("\n--- Preparing training data ---")

feature_cols = [
    # Time features
    'minute_sin', 'minute_cos', 'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos', 'is_weekend',
    # Step features
    'steps', 'steps_roll_3', 'steps_roll_5', 'steps_roll_10', 'steps_roll_15',
    'steps_sum_5', 'steps_sum_10',
    'step_diff', 'step_diff_abs', 'steps_std_5', 'steps_max_5',
    'activity_state',
    # Neighbor BPM features
    'bpm_last_known', 'bpm_next_known', 'bpm_neighbor_interp',
    'dist_to_last_bpm', 'dist_to_next_bpm', 'gap_length',
    # Day progression
    'day_index',
]

# Training set: rows where BPM is known
train_mask = timeline['bpm'].notna()

# For training rows, neighboring BPM features come from OTHER known points,
# not the point itself. We need to simulate "as if this point were missing".
# Simple approach: use the feature values as-is since for known points
# dist_to_last_bpm=0 and bpm_last_known=bpm itself. This is realistic because
# during inference, nearby known points WILL provide context.

# However, we should be careful: for points with dist=0, the neighbor IS the point.
# To avoid data leakage, we re-compute neighbors excluding the current point.
print("  Computing leave-one-out neighbor features for training data...")

train_df = timeline[train_mask].copy()
known_indices = train_df.index.values

# For each training row, find the previous and next known BPM (excluding self)
bpm_series = timeline['bpm'].copy()

# Shift-based approach: for known points, look at the previous/next known
train_df['bpm_last_known_loo'] = bpm_series.shift(1).ffill().loc[train_mask]
train_df['bpm_next_known_loo'] = bpm_series.shift(-1).bfill().loc[train_mask]

# Recompute distances for LOO
prev_known_idx = bpm_mask.shift(1, fill_value=False)
# For training points, dist_to_last = distance to previous known point
train_df['dist_to_last_bpm_loo'] = 0
train_df['dist_to_next_bpm_loo'] = 0

# Use vectorized approach: for each known point, find gap to prev/next known
for i, idx in enumerate(known_indices):
    if i > 0:
        train_df.loc[idx, 'dist_to_last_bpm_loo'] = idx - known_indices[i-1]
    else:
        train_df.loc[idx, 'dist_to_last_bpm_loo'] = 0
    if i < len(known_indices) - 1:
        train_df.loc[idx, 'dist_to_next_bpm_loo'] = known_indices[i+1] - idx
    else:
        train_df.loc[idx, 'dist_to_next_bpm_loo'] = 0

# Recompute interpolation
total_dist_loo = train_df['dist_to_last_bpm_loo'] + train_df['dist_to_next_bpm_loo']
total_dist_loo = total_dist_loo.replace(0, 1)
train_df['bpm_neighbor_interp_loo'] = (
    train_df['bpm_last_known_loo'] * (1 - train_df['dist_to_last_bpm_loo'] / total_dist_loo) +
    train_df['bpm_next_known_loo'] * (1 - train_df['dist_to_next_bpm_loo'] / total_dist_loo)
)

# Replace neighbor features in training with LOO versions
loo_map = {
    'bpm_last_known': 'bpm_last_known_loo',
    'bpm_next_known': 'bpm_next_known_loo',
    'dist_to_last_bpm': 'dist_to_last_bpm_loo',
    'dist_to_next_bpm': 'dist_to_next_bpm_loo',
    'bpm_neighbor_interp': 'bpm_neighbor_interp_loo',
}

feature_cols_train = feature_cols.copy()
for orig, loo in loo_map.items():
    train_df[orig] = train_df[loo]

X_train = train_df[feature_cols].values
y_train = train_df['bpm'].values

# Drop any rows with NaN features
valid = ~np.any(np.isnan(X_train), axis=1)
X_train = X_train[valid]
y_train = y_train[valid]

print(f"  Training samples: {len(X_train)}")
print(f"  Features: {len(feature_cols)}")
print(f"  Feature names: {feature_cols}")

# ============================================================
# 4. MODEL TRAINING WITH CROSS-VALIDATION
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- 4a. Linear Regression ---
print("\n" + "=" * 60)
print("MODEL 1: Linear Regression")
print("=" * 60)

lr = LinearRegression()

lr_rmse = -cross_val_score(lr, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
lr_mae = -cross_val_score(lr, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
lr_r2 = cross_val_score(lr, X_train, y_train, cv=kf, scoring='r2')

print(f"  5-Fold CV Results:")
print(f"    RMSE: {lr_rmse.mean():.2f} +/- {lr_rmse.std():.2f}")
print(f"    MAE:  {lr_mae.mean():.2f} +/- {lr_mae.std():.2f}")
print(f"    R2:   {lr_r2.mean():.4f} +/- {lr_r2.std():.4f}")

# Fit on full data
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_train)
print(f"\n  Full-data fit:")
print(f"    RMSE: {np.sqrt(mean_squared_error(y_train, lr_pred)):.2f}")
print(f"    MAE:  {mean_absolute_error(y_train, lr_pred):.2f}")
print(f"    R2:   {r2_score(y_train, lr_pred):.4f}")

# Feature importance (coefficients)
print(f"\n  Top 10 features by |coefficient| (standardized):")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
lr_scaled = LinearRegression().fit(X_scaled, y_train)
coef_importance = pd.Series(np.abs(lr_scaled.coef_), index=feature_cols).sort_values(ascending=False)
for feat, coef in coef_importance.head(10).items():
    print(f"    {feat:30s} {coef:.3f}")

# --- 4b. Random Forest ---
print("\n" + "=" * 60)
print("MODEL 2: Random Forest")
print("=" * 60)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_rmse = -cross_val_score(rf, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
rf_mae = -cross_val_score(rf, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
rf_r2 = cross_val_score(rf, X_train, y_train, cv=kf, scoring='r2')

print(f"  5-Fold CV Results:")
print(f"    RMSE: {rf_rmse.mean():.2f} +/- {rf_rmse.std():.2f}")
print(f"    MAE:  {rf_mae.mean():.2f} +/- {rf_mae.std():.2f}")
print(f"    R2:   {rf_r2.mean():.4f} +/- {rf_r2.std():.4f}")

# Fit on full data
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_train)
print(f"\n  Full-data fit:")
print(f"    RMSE: {np.sqrt(mean_squared_error(y_train, rf_pred)):.2f}")
print(f"    MAE:  {mean_absolute_error(y_train, rf_pred):.2f}")
print(f"    R2:   {r2_score(y_train, rf_pred):.4f}")

# Feature importance
print(f"\n  Top 10 features by importance:")
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, imp in feat_imp.head(10).items():
    print(f"    {feat:30s} {imp:.4f}")

# ============================================================
# 5. MODEL COMPARISON PLOTS
# ============================================================
print("\n--- Generating comparison plots ---")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Phase 2: Model Comparison — Linear Regression vs Random Forest', fontsize=14, fontweight='bold')

# 5a. CV RMSE comparison
models = ['Linear Reg', 'Random Forest']
rmse_means = [lr_rmse.mean(), rf_rmse.mean()]
rmse_stds = [lr_rmse.std(), rf_rmse.std()]
axes[0,0].bar(models, rmse_means, yerr=rmse_stds, color=['#42a5f5', '#66bb6a'], capsize=10, edgecolor='#333')
axes[0,0].set_ylabel('RMSE (BPM)')
axes[0,0].set_title('5-Fold CV RMSE')
for i, (m, s) in enumerate(zip(rmse_means, rmse_stds)):
    axes[0,0].text(i, m + s + 0.3, f'{m:.2f}', ha='center', fontweight='bold')

# 5b. CV R2 comparison
r2_means = [lr_r2.mean(), rf_r2.mean()]
r2_stds = [lr_r2.std(), rf_r2.std()]
axes[0,1].bar(models, r2_means, yerr=r2_stds, color=['#42a5f5', '#66bb6a'], capsize=10, edgecolor='#333')
axes[0,1].set_ylabel('R² Score')
axes[0,1].set_title('5-Fold CV R²')
for i, (m, s) in enumerate(zip(r2_means, r2_stds)):
    axes[0,1].text(i, m + s + 0.01, f'{m:.4f}', ha='center', fontweight='bold')

# 5c. Predicted vs Actual (LR)
sample_idx = np.random.RandomState(42).choice(len(y_train), min(2000, len(y_train)), replace=False)
axes[0,2].scatter(y_train[sample_idx], lr_pred[sample_idx], alpha=0.2, s=5, color='#42a5f5')
axes[0,2].plot([40, 180], [40, 180], 'r--', linewidth=1.5)
axes[0,2].set_xlabel('Actual BPM')
axes[0,2].set_ylabel('Predicted BPM')
axes[0,2].set_title(f'Linear Reg: Predicted vs Actual')
axes[0,2].set_xlim(40, 180)
axes[0,2].set_ylim(40, 180)

# 5d. Predicted vs Actual (RF)
axes[1,0].scatter(y_train[sample_idx], rf_pred[sample_idx], alpha=0.2, s=5, color='#66bb6a')
axes[1,0].plot([40, 180], [40, 180], 'r--', linewidth=1.5)
axes[1,0].set_xlabel('Actual BPM')
axes[1,0].set_ylabel('Predicted BPM')
axes[1,0].set_title(f'Random Forest: Predicted vs Actual')
axes[1,0].set_xlim(40, 180)
axes[1,0].set_ylim(40, 180)

# 5e. Feature importance (RF)
top_feats = feat_imp.head(10)
axes[1,1].barh(range(len(top_feats)), top_feats.values, color='#66bb6a', edgecolor='#333')
axes[1,1].set_yticks(range(len(top_feats)))
axes[1,1].set_yticklabels(top_feats.index, fontsize=8)
axes[1,1].set_xlabel('Importance')
axes[1,1].set_title('RF Feature Importance (Top 10)')
axes[1,1].invert_yaxis()

# 5f. Residual distribution comparison
lr_resid = y_train - lr_pred
rf_resid = y_train - rf_pred
axes[1,2].hist(lr_resid, bins=60, alpha=0.6, color='#42a5f5', label=f'LR (std={lr_resid.std():.1f})', density=True)
axes[1,2].hist(rf_resid, bins=60, alpha=0.6, color='#66bb6a', label=f'RF (std={rf_resid.std():.1f})', density=True)
axes[1,2].set_xlabel('Residual (Actual - Predicted)')
axes[1,2].set_ylabel('Density')
axes[1,2].set_title('Residual Distribution')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('Charts/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Charts/model_comparison.png")

# ============================================================
# 6. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

summary = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'CV RMSE': [f"{lr_rmse.mean():.2f} +/- {lr_rmse.std():.2f}",
                f"{rf_rmse.mean():.2f} +/- {rf_rmse.std():.2f}"],
    'CV MAE': [f"{lr_mae.mean():.2f} +/- {lr_mae.std():.2f}",
               f"{rf_mae.mean():.2f} +/- {rf_mae.std():.2f}"],
    'CV R2': [f"{lr_r2.mean():.4f} +/- {lr_r2.std():.4f}",
              f"{rf_r2.mean():.4f} +/- {rf_r2.std():.4f}"],
})
print(summary.to_string(index=False))

# ============================================================
# 7. SAVE ENHANCED TIMELINE & MODELS
# ============================================================
timeline.to_csv('Data/timeline_features.csv', index=False)
print(f"\nSaved enhanced timeline: Data/timeline_features.csv ({timeline.shape})")

# Save imputed BPM for missing rows
missing_mask = timeline['bpm'].isna()
X_missing = timeline.loc[missing_mask, feature_cols].values
valid_missing = ~np.any(np.isnan(X_missing), axis=1)

if valid_missing.sum() > 0:
    X_missing_clean = X_missing[valid_missing]
    rf_imputed = rf.predict(X_missing_clean)
    lr_imputed = lr.predict(X_missing_clean)

    imputed_df = timeline.loc[missing_mask].copy()
    imputed_df = imputed_df[valid_missing.tolist()]  # align
    imputed_df['bpm_rf'] = rf_imputed
    imputed_df['bpm_lr'] = lr_imputed
    imputed_df[['minute', 'bpm_rf', 'bpm_lr']].to_csv('Data/imputed_bpm.csv', index=False)
    print(f"Saved imputed BPM: Data/imputed_bpm.csv ({valid_missing.sum()} rows)")

print(f"\nPhase 2 complete!")
