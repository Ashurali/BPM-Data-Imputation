# Cross-Modal Heart Rate Imputation
Author : Michael Kurniawan Soegeng (吳忠賢)

Using wearable step data to predict missing BPM readings from a Huawei Band 7.

> **Course:** NYCU Artificial Intelligence (Spring 2026)
> **Instructor:** Professor Tsaipei Wang

## Overview

Consumer wearable devices sample heart rate intermittently, creating gaps that limit downstream clinical analysis (HRV, Early Warning Scores). This project frames BPM imputation as a supervised regression problem, using step count data, temporal features, and neighboring BPM values to fill missing readings.

We compare three methods spanning the complexity spectrum:

| Model | Type | RMSE | R² | Features |
|-------|------|------|----|----------|
| Linear Regression | Supervised | 9.560 | 0.730 | 27 (incl. neighbor BPM) |
| **Random Forest** | Supervised | **9.322** | **0.743** | 27 (incl. neighbor BPM) |
| SAITS | Self-supervised | 9.675 | 0.700 | 11 (no neighbor BPM) |

**Key finding:** RF achieves the best pointwise accuracy, but SAITS produces the most temporally coherent (natural-looking) BPM curves via self-attention — a tradeoff not captured by RMSE/R².

## Dataset

- **Source:** Huawei Band 7, March 1–26, 2026 (26 days, single user)
- **BPM:** 11,823 records (32% coverage of minute-level timeline)
- **Steps:** 4,604 interval records
- **Missingness:** MAR (Missing At Random) — depends on time-of-day and activity level
- **Github Link:** Dataset can be found in another github link here : https://github.com/Ashurali/BPM-Data

| File | Description |
|------|-------------|
| `Data/bpm_data.csv` | Raw heart rate records (date, bpm, time) |
| `Data/step_data.csv` | Raw step count records (date, steps, time_start, time_end) |
| `Data/timeline_aligned.csv` | Minute-level aligned timeline (36,930 rows) |
| `Data/timeline_features.csv` | Timeline with 42 engineered features |
| `Data/imputed_bpm_all.csv` | Final imputed BPM from all three models |

## Project Structure

```
.
├── phase1_exploration.ipynb     # Data exploration & statistical analysis
├── phase2_models.ipynb          # Feature engineering, LR + RF training
├── phase3_experiments.ipynb     # 7 systematic experiments
├── phase4_saits.ipynb           # SAITS deep learning model
├── phase5_report_visuals.ipynb  # Report-ready figures
├── report.docx                  # Final report
├── Data/                        # Raw and processed datasets
├── Charts/                      # All generated visualizations
└── PLANNING.md                  # Project planning document
```

## Experiments

| # | Experiment | Key Finding |
|---|-----------|-------------|
| 1 | Training data amount | Diminishing returns after ~14 days |
| 2 | Data composition | Dense days > sparse days, but all data combined is best |
| 3 | Data augmentation | Gaussian noise showed apparent improvement (caveat: pre-CV augmentation) |
| 4 | PCA dimensionality reduction | Hurts RF (destroys feature interactions); full features optimal |
| 5 | Feature ablation | Neighbor BPM features account for 60.5% of RF importance |
| 6 | Step-to-BPM time lag | Current steps (lag=0) outperform lagged features |
| 7 | RF hyperparameters | Depth=10 marginally better than depth=20 (bias-variance tradeoff) |

## Requirements

```
python >= 3.11
pandas
numpy
scikit-learn
matplotlib
pypots          # for SAITS
torch           # PyTorch (CPU is sufficient)
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib pypots torch
```

## Reproducing Results

Run the notebooks in order:

```bash
jupyter notebook phase1_exploration.ipynb   # ~2 min
jupyter notebook phase2_models.ipynb        # ~3 min
jupyter notebook phase3_experiments.ipynb   # ~5 min
jupyter notebook phase4_saits.ipynb         # ~15 min (CPU)
jupyter notebook phase5_report_visuals.ipynb # ~5 min
```

## AI Disclosure

This project was developed with the assistance of **Claude** (Anthropic). Claude was used as a coding assistant and teaching tool throughout the project, including:
- Data exploration strategy and statistical test selection
- Feature engineering design and data leakage prevention
- Model implementation and experiment design
- Report drafting and visualization

All concepts were discussed and understood by the student before implementation. The dataset, experimental decisions, and analysis interpretations are the student's own work.

## License

This project was created for academic coursework. Dataset is personal health data collected by the author.
