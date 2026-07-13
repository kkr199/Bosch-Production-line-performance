# Phase 6: Predictive Failure Modeling

## Source Data Used

- Raw `train_numeric.csv` and `test_numeric.csv` for numeric features and the train target.
- Raw `train_categorical.csv` and `test_categorical.csv` for selected one-hot categorical features.
- Raw-date-derived Phase 4 timing/path features from `train_date.csv` and `test_date.csv`.
- Phase 5 product-family labels derived from raw station presence matrices.

## Dataset Construction

- Sampled modeling rows: 226,879
- Train rows: 170,159
- Validation rows: 56,720
- Kaggle test rows scored: 1,183,748
- Positive failures in sampled rows: 6,879
- Final model feature count: 323

Missing values were handled in two ways: important missingness was kept as explicit `__is_missing` features, while model input values were median-imputed inside each modeling pipeline.

## Correlation Before Modeling

The pipeline writes raw numeric, raw categorical-presence, and final model-feature correlation reports before training models.

| feature                  |   corr_with_response |   abs_corr_with_response |   missing_rate |
|:-------------------------|---------------------:|-------------------------:|---------------:|
| L1_S24_F867              |            -0.362969 |                 0.362969 |       0.989839 |
| L1_S24_F1723             |            -0.354718 |                 0.354718 |       0.943353 |
| L1_S24_F1695             |            -0.260824 |                 0.260824 |       0.943353 |
| L1_S24_F839              |            -0.252313 |                 0.252313 |       0.989839 |
| L1_S24_F1632             |            -0.240996 |                 0.240996 |       0.9439   |
| L1_S24_F1604             |            -0.218173 |                 0.218173 |       0.9439   |
| L1_S24_F1758             |            -0.185008 |                 0.185008 |       0.943353 |
| L1_S24_F1667             |            -0.167131 |                 0.167131 |       0.9439   |
| L3_S32_F3850__is_missing |            -0.158183 |                 0.158183 |       0        |
| L1_S24_F902              |            -0.146971 |                 0.146971 |       0.989839 |
| L1_S24_F1846             |            -0.136803 |                 0.136803 |       0.887458 |
| L1_S24_F1002             |             0.127398 |                 0.127398 |       0.99018  |
| L2_S28_F3259             |            -0.123704 |                 0.123704 |       0.992043 |
| L1_S24_F1298             |            -0.121488 |                 0.121488 |       0.988334 |
| L1_S24_F988              |             0.115801 |                 0.115801 |       0.990139 |

## Model Comparison

| model               |   threshold |      mcc |   precision |   recall |       f1 |   pr_auc |   rank |
|:--------------------|------------:|---------:|------------:|---------:|---------:|---------:|-------:|
| LightGBM            |    0.785083 | 0.338642 |    0.574335 | 0.213372 | 0.311149 | 0.252867 |      1 |
| XGBoost             |    0.747938 | 0.335935 |    0.567442 | 0.212791 | 0.309514 | 0.246895 |      2 |
| CatBoost            |    0.796958 | 0.327926 |    0.557121 | 0.206977 | 0.301823 | 0.226068 |      3 |
| Random Forest       |    0.674328 | 0.325003 |    0.552426 | 0.205233 | 0.299279 | 0.239001 |      4 |
| Logistic Regression |    0.897303 | 0.304546 |    0.519562 | 0.193023 | 0.281475 | 0.165042 |      5 |

## Selected Model

The selected model is **LightGBM**, ranked by MCC first and PR-AUC second. It reached MCC 0.3386, precision 0.5743, recall 0.2134, F1 0.3111, and PR-AUC 0.2529 on the validation split.

## Output Files

- `data/processed/phase6_sampled_training_ids.csv`
- `data/processed/phase6_train_dataset.csv`
- `data/processed/phase6_validation_dataset.csv`
- `data/processed/phase6_test_dataset_preview.csv`
- `data/processed/phase6_test_predictions.csv`
- `reports/phase6_numeric_correlation_report.csv`
- `reports/phase6_categorical_presence_report.csv`
- `reports/phase6_selected_numeric_features.csv`
- `reports/phase6_selected_categorical_features.csv`
- `reports/phase6_final_feature_correlation_report.csv`
- `reports/phase6_model_comparison_metrics.csv`
- `models/phase6_best_model.joblib`