# Phase 6 Model Improvement Report

## Technical Summary

No production-safe improvement candidate beat the accepted Phase 6 LightGBM baseline. The official model should remain **Phase 6 LightGBM** with validation MCC **0.3386**.

The validation-optimized blend is included as a research upper bound, not as the official production model, because the blending weights were selected on the same validation set used for reporting.

## Key Findings

| model                                      | production_safe   |   threshold |    mcc |   precision |   recall |     f1 |   pr_auc | notes                                                                       | experiment                   | config_name   |   feature_count |
|:-------------------------------------------|:------------------|------------:|-------:|------------:|---------:|-------:|---------:|:----------------------------------------------------------------------------|:-----------------------------|:--------------|----------------:|
| Phase 6 LightGBM baseline                  | True              |      0.7851 | 0.3386 |      0.5743 |   0.2134 | 0.3111 |   0.2529 | Accepted Phase 6 production-safe benchmark.                                 |                              |               |                 |
| Validation-optimized blend                 | False             |      0.9926 | 0.3238 |      0.5920 |   0.1890 | 0.2865 |   0.1875 | Research upper bound; weights selected on validation.                       |                              |               |                 |
| Tuned LightGBM - original features         | True              |      0.5125 | 0.3144 |      0.5343 |   0.1994 | 0.2904 |   0.1534 | Best config: balanced_wide.                                                 | original_phase6_features     | balanced_wide |        323.0000 |
| Tuned LightGBM - graph trajectory features | True              |      0.5125 | 0.3115 |      0.5323 |   0.1965 | 0.2870 |   0.1512 | Best config: balanced_wide.                                                 | phase6_plus_graph_trajectory | balanced_wide |        334.0000 |
| Family-aware LightGBM                      | True              |      0.5123 | 0.2459 |      0.3934 |   0.1727 | 0.2400 |   0.1397 | Family-specific models where sample size allows; global fallback otherwise. |                              |               |                 |

## Scope And Metric Definitions

- Cohort: existing Phase 6 train/validation split built from raw Bosch numeric, categorical, and date inputs.
- Target: `Response`, where `1` means product failure.
- Primary metric: Matthews correlation coefficient (MCC), selected because the failure rate is highly imbalanced.
- Supporting metrics: precision, recall, F1, and PR-AUC.
- Thresholding: each model is evaluated at the validation threshold that maximizes MCC.

## Methodology

The improvement phase tested three production-safe modeling paths:

1. Hyperparameter tuning of LightGBM on the original Phase 6 feature matrix.
2. Hyperparameter tuning of LightGBM after adding Phase 10 graph and trajectory features.
3. Family-aware LightGBM models where product-family sample size and failure count were sufficient; smaller families fall back to the global model.

The research blend combines normalized validation scores from the strongest candidates and Phase 10 trajectory risk. It is useful for estimating remaining headroom, but it should be validated with cross-validation or a fresh holdout before becoming an official model.

## LightGBM Tuning Runs

| experiment                   | config_name        |   threshold |    mcc |   precision |   recall |     f1 |   pr_auc |   feature_count |
|:-----------------------------|:-------------------|------------:|-------:|------------:|---------:|-------:|---------:|----------------:|
| original_phase6_features     | balanced_wide      |      0.5125 | 0.3144 |      0.5343 |   0.1994 | 0.2904 |   0.1534 |             323 |
| original_phase6_features     | high_recall        |      0.5150 | 0.3144 |      0.5343 |   0.1994 | 0.2904 |   0.1545 |             323 |
| phase6_plus_graph_trajectory | balanced_wide      |      0.5125 | 0.3115 |      0.5323 |   0.1965 | 0.2870 |   0.1512 |             334 |
| original_phase6_features     | balanced_compact   |      0.5166 | 0.2945 |      0.4702 |   0.2017 | 0.2823 |   0.1428 |             323 |
| original_phase6_features     | deeper_regularized |      0.5121 | 0.2941 |      0.4689 |   0.2017 | 0.2821 |   0.1559 |             323 |
| phase6_plus_graph_trajectory | deeper_regularized |      0.5125 | 0.2602 |      0.4924 |   0.1500 | 0.2299 |   0.1295 |             334 |
| phase6_plus_graph_trajectory | high_recall        |      0.5150 | 0.2602 |      0.4924 |   0.1500 | 0.2299 |   0.1263 |             334 |
| phase6_plus_graph_trajectory | balanced_compact   |      0.5175 | 0.2599 |      0.4933 |   0.1494 | 0.2294 |   0.1261 |             334 |

## Family-Aware Model Diagnostics

|   family | status       |   train_rows |   train_failures |   valid_rows |   threshold |    mcc |   precision |   recall |     f1 |   pr_auc |
|---------:|:-------------|-------------:|-----------------:|-------------:|------------:|-------:|------------:|---------:|-------:|---------:|
|        0 | family_model |        36199 |             1036 |        11976 |      0.5125 | 0.1721 |      0.3197 |   0.1083 | 0.1618 |   0.0767 |
|        2 | family_model |        34165 |              930 |        11415 |      0.5125 | 0.2327 |      0.4151 |   0.1438 | 0.2136 |   0.0938 |
|        7 | family_model |        21164 |              607 |         7166 |      0.5125 | 0.2668 |      0.4638 |   0.1667 | 0.2452 |   0.1193 |
|        1 | family_model |        19202 |              744 |         6395 |      0.5125 | 0.3963 |      0.5170 |   0.3304 | 0.4032 |   0.2543 |
|        4 | family_model |        18392 |              708 |         5947 |      0.5123 | 0.2689 |      0.2553 |   0.3602 | 0.2988 |   0.1680 |
|        3 | family_model |        18210 |              515 |         6107 |      0.5124 | 0.2196 |      0.4000 |   0.1348 | 0.2017 |   0.1046 |
|        5 | family_model |        14175 |              392 |         4863 |      0.5125 | 0.3092 |      0.5185 |   0.1986 | 0.2872 |   0.1614 |
|        6 | family_model |         8652 |              227 |         2851 |      0.5121 | 0.0717 |      0.0402 |   0.6494 | 0.0758 |   0.0428 |

## Validation-Optimized Blend Weights

|   best_original_lgbm |   best_enhanced_lgbm |   family_aware_lgbm |   trajectory_failure_risk |
|---------------------:|---------------------:|--------------------:|--------------------------:|
|               0.7000 |               0.0000 |              0.1500 |                    0.1500 |

## Limitations And Robustness

- This is still a single train/validation split, so small improvements should be treated carefully.
- The validation-optimized blend is intentionally labeled as research because it tunes weights on validation.
- Family-specific models can become unstable for low-volume or low-failure families; fallback rules are used to control that risk.
- The production-safe candidates do not use the leaderboard-style nearby-label/order-leak features.

## Recommended Next Steps

Use the best production-safe candidate only if it improves MCC materially over the Phase 6 baseline. If the lift is small or negative, keep the original Phase 6 LightGBM as the official model and treat the improvement artifacts as tuning evidence. For a stronger final push, run repeated cross-validation or a time-style holdout and then choose one stable model.
