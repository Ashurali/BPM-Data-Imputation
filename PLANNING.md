# NYCU AI HW1 — Cross-Modal BPM Imputation Project

## PROJECT OVERVIEW
- **Course:** NYCU AI Spring 2026, Professor Tsaipei Wang (王才沛)
- **Due:** April 3, 2026
- **Task:** Cross-modal heart rate imputation — use step count data from Huawei Band 7 to predict missing BPM readings
- **Deliverables:** 10-page PDF report, code appendix, dataset on GitHub

## DATA FILES (`Data/`)
- `bpm_data.csv` — 11,823 rows: date, bpm, time
- `step_data.csv` — 4,604 rows: date, steps, time_start, time_end
- `timeline_aligned.csv` — 36,930 rows: minute-level aligned timeline (Phase 1)
- `timeline_features.csv` — timeline + all engineered features (Phase 2)
- `imputed_bpm.csv` — original + imputed BPM from LR and RF (Phase 2)

---

## Phase 1: Data Exploration & Preprocessing ✅
**Notebook:** `phase1_exploration.ipynb`

- Load & parse BPM + step CSVs, datetime alignment
- Central tendency & distribution (histogram + normal overlay, Q-Q plot, Shapiro-Wilk)
- Data quality check (outliers, duplicates — deduplication handled in alignment)
- Unified minute-level timeline (36,930 min, 32% BPM, 68% missing)
- Time series visualization (full month + single dense day)
- Covariance matrix, Pearson r=0.239, Spearman ρ=0.226, cross-correlation at lags
- Chi-square χ²=139.1 (df=9, p<1e-25), Cramér's V=0.132 — significant but weak
- MCAR/MAR/MNAR analysis → **MAR confirmed** (missingness depends on time & activity)
- BPM autocorrelation (high at lag-1 → neighboring BPM is strongest predictor)
- Circadian rhythm (BPM varies by hour-of-day)
- Day-by-day coverage (17x density ratio, natural experiment for data amount)
- Sleep detection from data (irregular hours, not fixed nighttime)
- BPM regime separation (Sleep / Resting / Active — it's a mixture, not one distribution)
- Watch-off vs sensor-gap classification
- Sampling frequency bias (watch oversamples active states → hypothesis to verify)

**Charts:** `central_tendency.png`, `full_month_timeseries.png`, `day_timeline_dense.png`, `correlation_analysis.png`, `chi_square_analysis.png`, `missingness_analysis.png`, `autocorrelation_circadian.png`, `daily_coverage.png`, `sleep_detection.png`, `bpm_regimes.png`, `gap_classification.png`, `sampling_bias.png`

---

## Phase 2: Feature Engineering & Model Training ✅
**Notebook:** `phase2_models.ipynb`

- **Temporal features:** minute-of-day sin/cos, hour sin/cos, day-of-week sin/cos, is_weekend, day_index
- **Step features:** rolling mean/sum (3/5/10/15 min), acceleration, std, max, activity state
- **Neighboring BPM:** last/next known BPM, distances, weighted interpolation, gap length
- **Data leakage prevention:** leave-one-out neighbor features for training
- **Model 1: Linear Regression** — 5-fold CV (baseline)
- **Model 2: Random Forest** — 200 trees, max_depth=20, 5-fold CV, feature importance
- Per-regime error analysis (Sleep/Rest/Active) — tests sampling bias hypothesis
- Impute all missing BPM, visualize before/after
- Saves: `timeline_features.csv`, `imputed_bpm.csv`

**Charts:** `model_comparison.png`, `imputation_result.png`

---

## Phase 3: Experiments (HW Requirements) ✅
**Notebook:** `phase3_experiments.ipynb`

- [x] **Exp 1: Training data amount** — 3/5/7/10/14/21/all days, both models
- [x] **Exp 2: Data composition** — sparse-day vs dense-day vs all training
- [x] **Exp 3: Data augmentation** — Gaussian noise 5% (2x), 5+10% (3x)
- [x] **Exp 4: Dimensionality reduction** — PCA at 3/5/8/10/15/20/all components
- [x] **Exp 5: Feature ablation** — steps-only → +windows → +time → +neighbors
- [x] **Exp 6: Step-to-BPM time lag** — test if lagged steps (1-10 min) predict BPM better than current steps
- [x] **Exp 7: RF hyperparameter comparison** — depth 10/15/20, trees 50/100/200, min_samples_split=5

**Charts:** `exp1_data_amount.png`, `exp2_composition.png`, `exp3_augmentation.png`, `exp4_pca.png`, `exp5_ablation.png`, `exp6_time_lag.png`, `exp7_rf_hyperparams.png`

---

## Phase 4: SAITS Deep Learning Model (Optional) 🔲

- 3rd model: self-attention for time series imputation (DMSA mechanism)
- Uses the 1 allowed deep learning slot per HW rules
- Library: PyPOTS (`pip install pypots`)
- Paper: arxiv 2202.08516, outperforms BRITS by 12-38% MAE
- Compare against LR and RF on same CV splits

---

## Phase 5: Report Visualizations 🔲

- Finalize all charts (consistent styling, report-ready)
- Additional comparison plots from Phase 3 experiments
- Before/after imputation full-month overview

---

## Phase 6: Report Writing 🔲

- Max 10 pages, 12pt+ font, PDF format
- Sections: Introduction, Dataset, Methods, Experiments, Results, Discussion
- Discussion topics: bias-variance tradeoff, MAR implications, feature importance, sampling bias finding, comparison of methods
- Code appendix (separate, not counted in page limit)
- Dataset + documentation on GitHub

---

## HW REQUIREMENTS CHECKLIST
- [x] Own dataset (not pre-existing) — Huawei Band 7 BPM + steps
- [x] At least 2 supervised learning methods — Linear Regression + Random Forest
- [ ] Max 1 deep learning — SAITS (optional, Phase 4)
- [x] Cross-validation — 5-fold CV
- [x] MSE/RMSE metrics
- [x] Experiment: effect of training data amount
- [x] Experiment: data composition/balance
- [x] Experiment: data augmentation
- [x] Experiment: with/without dimensionality reduction
- [ ] Discussion section
- [ ] Code appendix
- [ ] Dataset on GitHub with documentation
- [ ] Max 10 pages, 12pt+ font

## KEY FINDINGS FOR REPORT
- Motivation: clinical EWS, HRV analysis (unstable at 10% missing), hypoglycemia prediction
- BPM↔Steps correlation real (p<1e-25) but weak (r=0.239) — motivates multi-feature approach
- Day imbalance: early March sparse, late March dense — natural experiment for data quantity
- MAR missingness — model must use temporal/activity features
- BPM is a mixture of 3 regimes (Sleep/Rest/Active), not one distribution
- Sampling bias: observed BPM mean may be inflated by active-state oversampling
