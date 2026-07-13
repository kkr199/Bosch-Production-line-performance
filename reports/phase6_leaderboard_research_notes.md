# Phase 6 Leaderboard Research Notes

## Access Notes

The Kaggle leaderboard page is public, but it does not expose competitors' private submission files or private source code. The useful material available publicly is in Kaggle discussions, public notebooks, papers, and GitHub repositories where participants shared their approaches.

## Public Solution Lessons

1. Date/order features matter more than simply swapping classifiers.
   - Public writeups repeatedly point to start/end timestamps, production-order features, and lead/lag neighbor behavior as the strongest way to improve MCC.

2. Date columns should be compressed by station.
   - Public solution code notes that same-station date columns are mostly duplicates, reducing roughly 1,100 date features to about 53 station-level features.

3. Missingness is not just noise.
   - Missing station values often mean a product skipped that station, so path/missingness features should be preserved.

4. Station/path families matter.
   - The data contains thousands of unique manufacturing paths, so path clusters and product-family features are useful.

5. The competition had a practical order leak.
   - Train and test were randomly sampled from the same production period. Products near known failed products in production time or Id order have elevated risk.

6. MCC threshold tuning is mandatory.
   - Public scripts do not use a default 0.5 threshold. They search thresholds on validation predictions and submit binary predictions at the best MCC threshold.

7. Stronger public approaches stack multiple models.
   - High-ranking public repositories combine order/path/date/numeric/categorical feature sets and stack or weight XGBoost, LightGBM, and Random Forest models.

## Local Changes Made From This Research

- Added `src/data/phase6_leaderboard_boost_modeling.py`.
- Added order/neighbor/leak-style features:
  - Previous/next known failure indicators by `Id`, `start_time`, and `end_time`.
  - Failure counts near the product start time.
  - Distance to previous, next, and nearest known failure.
  - Previous/next deltas for timing, path, line-duration, and family features.
- Stacked those features with the Phase 6 base model probability.
- Trained leaderboard-style XGBoost and LightGBM models.
- Tuned the binary threshold for MCC.
- Wrote Kaggle-ready binary predictions.

## Result

The best local leaderboard-style validation result is:

- Model: Leaderboard XGBoost
- MCC: 0.834966
- Precision: 0.876982
- Recall: 0.804070
- F1: 0.838944
- PR-AUC: 0.905169

## Important Caveat

This validation score is intentionally competition-style and optimistic. It uses order/leak-style features built from known training failures, which was a major public Kaggle strategy. This may help Kaggle-style scoring, but it should not be treated as a production-safe estimate for future unseen manufacturing data.

## Key Artifacts

- `src/data/phase6_leaderboard_boost_modeling.py`
- `reports/phase6_leaderboard_boost_report.md`
- `reports/phase6_leaderboard_boost_metrics.csv`
- `data/processed/phase6_leaderboard_boost_train_features.csv`
- `data/processed/phase6_leaderboard_boost_test_features.csv`
- `data/processed/phase6_leaderboard_boost_test_predictions.csv`
- `submissions/phase6_leaderboard_boost_submission.csv`
- `models/phase6_leaderboard_boost_best_model.joblib`
