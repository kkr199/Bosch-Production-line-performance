# Phase 6 Leaderboard Boost

## What Changed

This extension adds public-solution-inspired order and leak-style features:

- Previous/next known failure indicators after sorting by `Id`, `start_time`, and `end_time`.
- Counts of known failures near each part's start time.
- Distance to previous, next, and nearest known failure in production time.
- Previous/next deltas for timing, path, line-duration, and product-family features.

These features mirror the public Kaggle insight that failures cluster among nearby parts in production order. They are useful for competition scoring because train and test were randomly sampled from the same production period.

## Validation Results

| model                |   threshold |      mcc |   precision |   recall |       f1 |   pr_auc |   rank |
|:---------------------|------------:|---------:|------------:|---------:|---------:|---------:|-------:|
| Leaderboard XGBoost  |    0.822409 | 0.834966 |    0.876982 | 0.80407  | 0.838944 | 0.905169 |      1 |
| Leaderboard LightGBM |    0.712054 | 0.834341 |    0.876347 | 0.803488 | 0.838338 | 0.910125 |      2 |

## Selected Boost Model

The selected boost model is **Leaderboard XGBoost**, with MCC 0.8350, precision 0.8770, recall 0.8041, F1 0.8389, and PR-AUC 0.9052.

## Important Caveat

This is a competition-style validation result. Because order/leak features use nearby known training failures, the validation score is optimistic compared with a true future-time production deployment test.

## Output Files

- `data/processed/phase6_leaderboard_boost_train_features.csv`
- `data/processed/phase6_leaderboard_boost_test_features.csv`
- `data/processed/phase6_leaderboard_boost_test_predictions.csv`
- `reports/phase6_leaderboard_boost_metrics.csv`
- `models/phase6_leaderboard_boost_best_model.joblib`