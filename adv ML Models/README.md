# Advanced ML Models

This folder contains the reproducible code used to produce the completed
full-data Phase 6 benchmark and the selected LightGBM test predictions. It does
not contain interim logs, failed trials, raw Bosch data, or generated model
binaries.

## Inputs

The runners expect the local Bosch project data and prior feature-engineering
outputs:

- `data/raw/train_numeric.csv`, `train_categorical.csv`, and their test files
- Phase 4 engineered date features in `data/processed`
- Phase 5 product-family features in `data/processed`

Missing values are retained: their absence is treated as a process signal and
numeric missingness indicators are included by the Phase 6 feature builder.

## Reproduce the benchmark

Run the main benchmark first. It evaluates the 75 combinations of three
stratified splits, five numeric present-rate gates, and five model families.

```powershell
.\.venv\Scripts\python.exe "adv ML Models\run_phase6_split_rate_benchmark.py"
```

On constrained machines, complete the Logistic Regression and Random Forest
rows using the memory-safe runner:

```powershell
.\.venv\Scripts\python.exe "adv ML Models\retry_memory_safe_lr_rf.py"
```

The completed table is
`phase6_full_clean_split_rate_metrics_completed.csv`.

## Score the test data

The selected configuration is LightGBM with an 80/20 feature-selection split
and numeric present rate of 0.0025. The scorer retrains it on all labelled rows
after freezing selection rules learned from the training partition, then scores
numeric, categorical, date/timing, and product-path features for the full test
set.

```powershell
.\.venv\Scripts\python.exe "adv ML Models\train_and_score_selected_lightgbm.py"
```

It writes the complete probabilities, `Id,Response` sample submission,
prediction summary, risk preview, and model card to `data/processed`, `reports`,
and `models`. Test labels are unavailable, so those outputs are predictions,
not test accuracy.
